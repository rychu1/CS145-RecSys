import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from abc import ABC, abstractmethod
from typing import Optional

# Make sure you have PySpark installed and configured
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as sf

# --- 0. Base Recommender Class ---
class BaseRecommender(ABC):
    """Abstract base class for recommender models, compatible with sim4rec."""
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

    @abstractmethod
    def fit(self, log: DataFrame, user_features: Optional[DataFrame] = None, item_features: Optional[DataFrame] = None):
        """Trains the recommender model."""
        pass

    @abstractmethod
    def predict(self, log: DataFrame, k: int, users: DataFrame, items: DataFrame,
                user_features: Optional[DataFrame] = None, item_features: Optional[DataFrame] = None,
                filter_seen_items: bool = True) -> DataFrame:
        """Makes top-K recommendations for a set of users."""
        pass

# --- 2. GCN Model (Internal PyTorch Module) ---
class GCNModel(nn.Module):
    """A shallow LightGCN‑flavoured encoder/decoder."""

    def __init__(self, num_nodes: int, embedding_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_size, max_norm=1.0)
        self.convs = nn.ModuleList([GCNConv(embedding_size, embedding_size) for _ in range(num_layers)])
        self.dropout = nn.Dropout(p=dropout)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Returns node embeddings of shape (num_nodes, embedding_size)."""
        x = self.embedding.weight
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu_(x)
            x = self.dropout(x)
        return x

    @staticmethod
    def decode(z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        """Dot‑product decoder for a set of edges."""
        src, dst = z[edge_label_index[0]], z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)  # (num_edges,)

# -----------------------------------------------------------------------------
# 2. Recommender implementation
# -----------------------------------------------------------------------------
class GraphCNRecommender(BaseRecommender):
    """Graph‑based collaborative filtering using shallow GCN / BPR‑loss."""

    def __init__(self, seed: Optional[int] = None,
                 embedding_size: int = 64,
                 num_layers: int = 2,
                 epochs: int = 50,
                 learning_rate: float = 5e-3,
                 weight_decay: float = 1e-4,
                 dropout: float = 0.0,
                 price_weighting: bool = False):
        super().__init__(seed)
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.price_weighting = price_weighting

        # runtime members
        self.model: Optional[GCNModel] = None
        self.data: Optional[Data] = None
        self.user_mapping: dict[int, int] = {}
        self.item_mapping: dict[int, int] = {}
        self.item_features_pd: Optional[pd.DataFrame] = None
        self.spark: Optional[SparkSession] = None

    # ---------------------------------------------------------------------
    # 2.1 Fit
    # ---------------------------------------------------------------------
    def fit(self, log: DataFrame, user_features: Optional[DataFrame] = None,
            item_features: Optional[DataFrame] = None):
        if item_features is None:
            raise ValueError("`item_features` (incl. at least 'item_idx' column) must be provided.")

        # Spark → Pandas
        self.spark = log.sparkSession
        log_pd = log.select("user_idx", "item_idx").toPandas()
        self.item_features_pd = item_features.toPandas()

        # **Full vocab**
        all_users = pd.Series(log_pd["user_idx"].unique())
        # If user_features provided, merge its user list too
        if user_features is not None:
            all_users = pd.concat([all_users, user_features.select("user_idx").toPandas()["user_idx"]])
        all_items = pd.Series(self.item_features_pd["item_idx"].unique())

        self._build_graph(log_pd, all_users.unique(), all_items.unique())

        # Model
        self.model = GCNModel(self.data.num_nodes, self.embedding_size, self.num_layers, self.dropout)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Training loop (BPR) ------------------------------------------------
        self.model.train()
        train_edges = self.data.train_edge_index
        for epoch in range(self.epochs):
            z = self.model(self.data.edge_index)

            # user–item negative sampling only
            try:
                neg_edge_index = negative_sampling(
                    edge_index=self.data.edge_index,
                    num_nodes=self.data.num_nodes,
                    num_neg_samples=train_edges.size(1),
                    bipartite=(self.data.num_users, self.data.num_items)  # torch‑geometric ≥2.4
                )
            except TypeError:
                # Fallback for older torch‑geometric (may include invalid pairs but acceptable on small graphs)
                neg_edge_index = negative_sampling(
                    edge_index=self.data.edge_index,
                    num_nodes=self.data.num_nodes,
                    num_neg_samples=train_edges.size(1),
                )

            pos_scores = GCNModel.decode(z, train_edges)
            neg_scores = GCNModel.decode(z, neg_edge_index)
            loss = -(pos_scores - neg_scores).sigmoid().log().mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

        return self

    # ---------------------------------------------------------------------
    # 2.2 Build graph
    # ---------------------------------------------------------------------
    def _build_graph(self, log_pd: pd.DataFrame, all_users: np.ndarray, all_items: np.ndarray):
        """Creates a bipartite graph and related index mappings."""
        # Id‑to‑contiguous index maps
        self.user_mapping = {int(u): i for i, u in enumerate(sorted(all_users))}
        self.item_mapping = {int(it): i for i, it in enumerate(sorted(all_items))}

        # Remap training interactions
        log_pd["user_map_idx"] = log_pd["user_idx"].map(self.user_mapping)
        log_pd["item_map_idx"] = log_pd["item_idx"].map(self.item_mapping)

        num_users = len(self.user_mapping)
        user_idx_tensor = torch.tensor(log_pd["user_map_idx"].values, dtype=torch.long)
        item_idx_tensor = torch.tensor(log_pd["item_map_idx"].values, dtype=torch.long) + num_users

        edge_ui = torch.stack([user_idx_tensor, item_idx_tensor])
        edge_iu = torch.stack([item_idx_tensor, user_idx_tensor])
        edge_index = torch.cat([edge_ui, edge_iu], dim=1)

        self.data = Data(edge_index=edge_index)
        self.data.num_users = num_users
        self.data.num_items = len(self.item_mapping)
        self.data.num_nodes = self.data.num_users + self.data.num_items
        self.data.train_edge_index = edge_ui  # directed user→item for BPR

    # ---------------------------------------------------------------------
    # 2.3 Predict top‑K
    # ---------------------------------------------------------------------
    def predict(self, log: DataFrame, k: int, users: DataFrame, items: DataFrame,
                user_features: Optional[DataFrame] = None, item_features: Optional[DataFrame] = None,
                filter_seen_items: bool = True) -> DataFrame:
        if self.model is None or self.spark is None:
            raise RuntimeError("Call fit() before predict().")

        self.model.eval()
        spark = self.spark

        # Outer join user × item (broadcast via Pandas → beware of size!)
        users_pd = users.select("user_idx").toPandas()
        items_pd = items.select("item_idx").toPandas()
        all_pairs = users_pd.assign(key=1).merge(items_pd.assign(key=1), on="key").drop("key", axis=1)

        # Optionally filter already‑seen edges
        if filter_seen_items:
            seen_pd = log.select("user_idx", "item_idx").toPandas()
            all_pairs = all_pairs.merge(seen_pd, on=["user_idx", "item_idx"], how="left", indicator=True)
            all_pairs = all_pairs[all_pairs["_merge"] == "left_only"].drop("_merge", axis=1)

        # Remap IDs – drop pairs containing unknown nodes (should be none)
        all_pairs["user_map_idx"] = all_pairs["user_idx"].map(self.user_mapping)
        all_pairs["item_map_idx"] = all_pairs["item_idx"].map(self.item_mapping)
        all_pairs.dropna(subset=["user_map_idx", "item_map_idx"], inplace=True)
        all_pairs = all_pairs.astype({"user_map_idx": int, "item_map_idx": int})

        # Build prediction edge tensor
        num_users = self.data.num_users
        src = torch.tensor(all_pairs["user_map_idx"].values, dtype=torch.long)
        dst = torch.tensor(all_pairs["item_map_idx"].values, dtype=torch.long) + num_users
        pred_edge_index = torch.stack([src, dst])

        with torch.no_grad():
            z = self.model(self.data.edge_index)
            scores = GCNModel.decode(z, pred_edge_index)

        all_pairs["score"] = scores.cpu().numpy()

        # Price weighting (optional)
        if self.price_weighting:
            price_lookup = self.item_features_pd.set_index("item_idx")["price"]
            all_pairs["price"] = all_pairs["item_idx"].map(price_lookup)
            all_pairs["relevance"] = all_pairs["score"] * all_pairs["price"].fillna(1.0)
        else:
            all_pairs["relevance"] = all_pairs["score"]

        # top‑K per user
        all_pairs.sort_values(["user_idx", "relevance"], ascending=[True, False], inplace=True)
        top_k = all_pairs.groupby("user_idx").head(k)
        
        final_recs_pd = top_k[['user_idx', 'item_idx', 'relevance']]
        
        recs_spark = self.spark.createDataFrame(final_recs_pd)
        
        recs_spark = recs_spark.withColumn("user_idx", sf.col("user_idx").cast("int")) \
                               .withColumn("item_idx", sf.col("item_idx").cast("int")) \
                               .withColumn("relevance", sf.col("relevance").cast("double"))

        return recs_spark
