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
# This assumes you have a local file `sample_recommenders.py` with BaseRecommender
from sample_recommenders import BaseRecommender
current_params = {'embedding_size': 64, 'num_layers': 2, 'epochs': 50, 'learning_rate': 0.001, 'weight_decay': 9e-05, 'dropout': 0.1}

# --- 2. GCN Model (Internal PyTorch Module) ---
class GCNModel(nn.Module):
    """A LightGCN-flavoured encoder with a dot-product decoder."""
    def __init__(self, num_nodes: int, embedding_size: int, num_layers: int, dropout: float):
        super().__init__()
        # Embedding layer with max_norm regularization
        self.embedding = nn.Embedding(num_nodes, embedding_size, max_norm=1.0)
        self.convs = nn.ModuleList([GCNConv(embedding_size, embedding_size) for _ in range(num_layers)])
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

    def init_weights(self):
        """Initialize embeddings with Xavier uniform distribution."""
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Computes node embeddings through GCN layers."""
        x = self.embedding.weight
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x) # Apply ReLU activation
            x = self.dropout(x)
        return x

    @staticmethod
    def decode(z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        """Computes dot-product scores for a given set of edges."""
        src, dst = z[edge_label_index[0]], z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)

# --- 3. GraphCN Recommender Implementation ---
class GraphCNRecommender(BaseRecommender):
    """Graph-based collaborative filtering using a GCN and BPR loss."""
    def __init__(self, seed: Optional[int] = None,
                 embedding_size: int = current_params['embedding_size'],
                 num_layers: int = current_params['num_layers'],
                 epochs: int = current_params['epochs'],
                 learning_rate: float = current_params['learning_rate'],
                 weight_decay: float = current_params['weight_decay'],
                 dropout: float = current_params['dropout'],
                 price_weighting: bool = True):
        super().__init__(seed)
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.price_weighting = price_weighting

        # Runtime members, initialized in fit()
        self.model: Optional[GCNModel] = None
        self.data: Optional[Data] = None
        self.user_mapping: dict[int, int] = {}
        self.item_mapping: dict[int, int] = {}
        self.item_features_pd: Optional[pd.DataFrame] = None
        self.spark: Optional[SparkSession] = None

    def fit(self, log: DataFrame, user_features: Optional[DataFrame] = None, item_features: Optional[DataFrame] = None):
        if item_features is None:
            raise ValueError("`item_features` (with 'item_idx') must be provided.")

        print("\n--- [GraphCN] Starting fit() method ---")
        self.spark = log.sparkSession
        log_pd = log.select("user_idx", "item_idx").toPandas()
        self.item_features_pd = item_features.toPandas()

        # Build vocabulary from ALL available users and items to avoid cold-start blindness
        if user_features is not None:
            all_users = np.union1d(log_pd["user_idx"].unique(),
                                   user_features.select("user_idx").toPandas()["user_idx"].unique())
        else:
            all_users = log_pd["user_idx"].unique()
        all_items = self.item_features_pd["item_idx"].unique()


        self._build_graph(log_pd, all_users, all_items)

        self.model = GCNModel(self.data.num_nodes, self.embedding_size, self.num_layers, self.dropout)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Training loop with BPR Loss
        print(f"--- [GraphCN] Starting training for {self.epochs} epochs ---")
        self.model.train()
        train_edges = self.data.train_edge_index
        num_pos_samples = train_edges.size(1)
        
        for epoch in range(self.epochs):
            z = self.model(self.data.edge_index)

            # Bipartite-aware negative sampling with robust fallback
            try:
                # PyG >= 2.4 API
                neg_edge_index = negative_sampling(
                    edge_index=self.data.edge_index,
                    num_nodes=self.data.num_nodes,
                    num_neg_samples=num_pos_samples,
                    method='bipartite',
                )
            except (TypeError, AssertionError):
                # Fallback for older PyG versions
                neg_edges = []
                # Ensure neg_edges is a list of tensors before checking its length
                collected_neg_edges = 0
                while collected_neg_edges < num_pos_samples:
                    raw_neg_batch = negative_sampling(
                        edge_index=self.data.edge_index,
                        num_nodes=self.data.num_nodes,
                        num_neg_samples=num_pos_samples,
                    )
                    mask = (raw_neg_batch[0] < self.data.num_users) & (raw_neg_batch[1] >= self.data.num_users)
                    valid_neg_batch = raw_neg_batch[:, mask]
                    neg_edges.append(valid_neg_batch)
                    collected_neg_edges += valid_neg_batch.size(1)

                neg_edge_index = torch.cat(neg_edges, dim=1)[:, :num_pos_samples]

            pos_scores = GCNModel.decode(z, train_edges)
            neg_scores = GCNModel.decode(z, neg_edge_index)
            
            # BPR Loss calculation
            loss = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"--- [GraphCN] Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f} ---")

        print("--- [GraphCN] Training complete ---")
        return self

    def _build_graph(self, log_pd: pd.DataFrame, all_users: np.ndarray, all_items: np.ndarray):
        """Creates a bipartite graph using the full user/item vocabulary."""
        print("--- [GraphCN] Building graph... ---")
        self.user_mapping = {int(u): i for i, u in enumerate(sorted(all_users))}
        self.item_mapping = {int(it): i for i, it in enumerate(sorted(all_items))}

        log_pd["user_map_idx"] = log_pd["user_idx"].map(self.user_mapping)
        log_pd["item_map_idx"] = log_pd["item_idx"].map(self.item_mapping)
        log_pd.dropna(subset=['user_map_idx', 'item_map_idx'], inplace=True)

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
        self.data.train_edge_index = edge_ui
        print("--- [GraphCN] Graph construction complete ---")


    def predict(self, log: DataFrame, k: int, users: DataFrame, items: DataFrame,
                user_features: Optional[DataFrame] = None, item_features: Optional[DataFrame] = None,
                filter_seen_items: bool = True) -> DataFrame:
        if self.model is None or self.spark is None:
            raise RuntimeError("Call fit() before predict().")
        
        print("\n--- [GraphCN] Starting predict() method ---")
        self.model.eval()

        users_pd = users.select("user_idx").toPandas()
        items_pd = items.select("item_idx").toPandas()
        all_pairs = users_pd.assign(key=1).merge(items_pd.assign(key=1), on="key").drop("key", axis=1)

        if filter_seen_items:
            seen_pd = log.select("user_idx", "item_idx").toPandas()
            all_pairs = all_pairs.merge(seen_pd, on=["user_idx", "item_idx"], how="left", indicator=True)
            all_pairs = all_pairs[all_pairs["_merge"] == "left_only"].drop("_merge", axis=1)

        all_pairs["user_map_idx"] = all_pairs["user_idx"].map(self.user_mapping)
        all_pairs["item_map_idx"] = all_pairs["item_idx"].map(self.item_mapping)
        all_pairs.dropna(subset=["user_map_idx", "item_map_idx"], inplace=True)
        all_pairs = all_pairs.astype({"user_map_idx": int, "item_map_idx": int})

        num_users = self.data.num_users
        src = torch.tensor(all_pairs["user_map_idx"].values, dtype=torch.long)
        dst = torch.tensor(all_pairs["item_map_idx"].values, dtype=torch.long) + num_users
        pred_edge_index = torch.stack([src, dst])

        with torch.no_grad():
            z = self.model(self.data.edge_index)
            scores = GCNModel.decode(z, pred_edge_index)

        all_pairs["score"] = scores.cpu().numpy()

        if self.price_weighting:
            price_lookup = self.item_features_pd.set_index("item_idx")["price"]
            all_pairs["price"] = all_pairs["item_idx"].map(price_lookup)
            all_pairs["relevance"] = all_pairs["score"] * all_pairs["price"].fillna(1.0)
        else:
            all_pairs["relevance"] = all_pairs["score"]

        all_pairs.sort_values(["user_idx", "relevance"], ascending=[True, False], inplace=True)
        top_k = all_pairs.groupby("user_idx").head(k)
        
        final_recs_pd = top_k[['user_idx', 'item_idx', 'relevance']]
        
        recs_spark = self.spark.createDataFrame(final_recs_pd)
        
        recs_spark = recs_spark.withColumn("user_idx", sf.col("user_idx").cast("int")) \
                               .withColumn("item_idx", sf.col("item_idx").cast("int")) \
                               .withColumn("relevance", sf.col("relevance").cast("double"))

        print("--- [GraphCN] Prediction complete, returning results ---")
        return recs_spark
