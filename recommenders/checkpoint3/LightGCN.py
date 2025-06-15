import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling, add_self_loops, degree
from typing import Optional

# Make sure you have PySpark installed and configured
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as sf

# --- 0. Base Recommender Class ---
# This assumes you have a local file `sample_recommenders.py` with BaseRecommender
from sample_recommenders import BaseRecommender

# This dictionary holds the best parameters found from your hyperparameter tuning script.
# It's used to set the default values for the recommender.
current_params = {'embedding_size': 64, 'num_layers': 2, 'epochs': 50, 'learning_rate': 0.001, 'weight_decay': 9e-05, 'dropout': 0.1}

# =========================================================================================
# === Model 1: Standard Graph Convolutional Network (GCN) =================================
# =========================================================================================

class GCNModel(nn.Module):
    """
    A standard Graph Convolutional Network (GCN) model for recommendation.
    It uses GCN layers that include feature transformations (weight matrices) and non-linear activations.
    """
    def __init__(self, num_nodes: int, embedding_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_size, max_norm=1.0)
        self.convs = nn.ModuleList([GCNConv(embedding_size, embedding_size) for _ in range(num_layers)])
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.embedding.weight
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)
        return x

    @staticmethod
    def decode(z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        src, dst = z[edge_label_index[0]], z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)

class GraphCNRecommender(BaseRecommender):
    """
    Recommender implementation using the standard GCNModel trained with BPR loss.
    """
    def __init__(self, seed: Optional[int] = None, **kwargs):
        super().__init__(seed)
        # Set hyperparameters from kwargs or use defaults from current_params
        self.embedding_size = kwargs.get('embedding_size', current_params['embedding_size'])
        self.num_layers = kwargs.get('num_layers', current_params['num_layers'])
        self.epochs = kwargs.get('epochs', current_params['epochs'])
        self.lr = kwargs.get('learning_rate', current_params['learning_rate'])
        self.weight_decay = kwargs.get('weight_decay', current_params['weight_decay'])
        self.dropout = kwargs.get('dropout', current_params['dropout'])
        self.price_weighting = kwargs.get('price_weighting', True)
        self.model, self.data, self.user_mapping, self.item_mapping, self.item_features_pd, self.spark = None, None, {}, {}, None, None

    def fit(self, log: DataFrame, user_features: Optional[DataFrame] = None, item_features: Optional[DataFrame] = None):
        if item_features is None: raise ValueError("`item_features` must be provided.")
        print("\n--- [GCN] Starting fit() method ---")
        self.spark = log.sparkSession
        log_pd = log.select("user_idx", "item_idx").toPandas()
        self.item_features_pd = item_features.toPandas()
        all_users = np.union1d(log_pd["user_idx"].unique(), user_features.select("user_idx").toPandas()["user_idx"].unique()) if user_features else log_pd["user_idx"].unique()
        all_items = self.item_features_pd["item_idx"].unique()
        self._build_graph(log_pd, all_users, all_items)
        self.model = GCNModel(self.data.num_nodes, self.embedding_size, self.num_layers, self.dropout)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        print(f"--- [GCN] Starting training for {self.epochs} epochs ---")
        self.model.train()
        train_edges = self.data.train_edge_index
        num_pos_samples = train_edges.size(1)
        for epoch in range(self.epochs):
            z = self.model(self.data.edge_index)
            neg_edge_index = self._sample_negatives(train_edges, num_pos_samples)
            pos_scores = GCNModel.decode(z, train_edges)
            neg_scores = GCNModel.decode(z, neg_edge_index)
            loss = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            if (epoch + 1) % 10 == 0: print(f"--- [GCN] Epoch {epoch+1}/{self.epochs}, BPR Loss: {loss.item():.4f} ---")
        print("--- [GCN] Training complete ---")
        return self

    def _build_graph(self, log_pd: pd.DataFrame, all_users: np.ndarray, all_items: np.ndarray):
        print("--- [GCN] Building graph... ---")
        self.user_mapping = {int(u): i for i, u in enumerate(sorted(all_users))}
        self.item_mapping = {int(it): i for i, it in enumerate(sorted(all_items))}
        log_pd["user_map_idx"] = log_pd["user_idx"].map(self.user_mapping)
        log_pd["item_map_idx"] = log_pd["item_idx"].map(self.item_mapping)
        log_pd.dropna(subset=['user_map_idx', 'item_map_idx'], inplace=True)
        num_users = len(self.user_mapping)
        u_tensor = torch.tensor(log_pd["user_map_idx"].values, dtype=torch.long)
        i_tensor = torch.tensor(log_pd["item_map_idx"].values, dtype=torch.long) + num_users
        edge_ui = torch.stack([u_tensor, i_tensor])
        edge_iu = torch.stack([i_tensor, u_tensor])
        self.data = Data(edge_index=torch.cat([edge_ui, edge_iu], dim=1))
        self.data.num_users, self.data.num_items = num_users, len(self.item_mapping)
        self.data.num_nodes = self.data.num_users + self.data.num_items
        self.data.train_edge_index = edge_ui
        print("--- [GCN] Graph construction complete ---")

    def _sample_negatives(self, train_edges, num_pos_samples):
        try:
            return negative_sampling(edge_index=self.data.edge_index, num_nodes=self.data.num_nodes, num_neg_samples=num_pos_samples, method='bipartite')
        except (TypeError, AssertionError):
            neg_edges = []
            collected_neg_edges = 0
            while collected_neg_edges < num_pos_samples:
                raw_neg = negative_sampling(edge_index=self.data.edge_index, num_nodes=self.data.num_nodes, num_neg_samples=num_pos_samples)
                mask = (raw_neg[0] < self.data.num_users) & (raw_neg[1] >= self.data.num_users)
                valid_neg = raw_neg[:, mask]
                neg_edges.append(valid_neg)
                collected_neg_edges += valid_neg.size(1)
            return torch.cat(neg_edges, dim=1)[:, :num_pos_samples]

    def predict(self, log: DataFrame, k: int, users: DataFrame, items: DataFrame, **kwargs) -> DataFrame:
        if not all([self.model, self.spark]): raise RuntimeError("Call fit() first.")
        print("\n--- [GCN] Starting predict() method ---")
        self.model.eval()
        users_pd, items_pd = users.select("user_idx").toPandas(), items.select("item_idx").toPandas()
        all_pairs = users_pd.assign(key=1).merge(items_pd.assign(key=1), on="key").drop("key", axis=1)
        if kwargs.get('filter_seen_items', True):
            seen_pd = log.select("user_idx", "item_idx").toPandas()
            all_pairs = all_pairs.merge(seen_pd, on=["user_idx", "item_idx"], how="left", indicator=True)
            all_pairs = all_pairs[all_pairs["_merge"] == "left_only"].drop("_merge", axis=1)
        all_pairs["user_map_idx"] = all_pairs["user_idx"].map(self.user_mapping)
        all_pairs["item_map_idx"] = all_pairs["item_idx"].map(self.item_mapping)
        all_pairs.dropna(subset=["user_map_idx", "item_map_idx"], inplace=True)
        all_pairs = all_pairs.astype({"user_map_idx": int, "item_map_idx": int})
        src = torch.tensor(all_pairs["user_map_idx"].values, dtype=torch.long)
        dst = torch.tensor(all_pairs["item_map_idx"].values, dtype=torch.long) + self.data.num_users
        with torch.no_grad():
            z = self.model(self.data.edge_index)
            scores = GCNModel.decode(z, torch.stack([src, dst]))
        all_pairs["score"] = scores.cpu().numpy()
        if self.price_weighting:
            price_lookup = self.item_features_pd.set_index("item_idx")["price"]
            all_pairs["price"] = all_pairs["item_idx"].map(price_lookup)
            all_pairs["relevance"] = all_pairs["score"] * all_pairs["price"].fillna(1.0)
        else:
            all_pairs["relevance"] = all_pairs["score"]
        all_pairs.sort_values(["user_idx", "relevance"], ascending=[True, False], inplace=True)
        top_k = all_pairs.groupby("user_idx").head(k)
        # Select final columns and ensure user/item IDs are 64-bit integers in pandas
        final_recs_pd = top_k[['user_idx', 'item_idx', 'relevance']].astype({
            'user_idx': np.int64,
            'item_idx': np.int64
        })

        # --- Formatting for Submission ---
        # Convert the pandas DataFrame back to a Spark DataFrame.
        recs_spark = self.spark.createDataFrame(final_recs_pd)
        # Ensure the output schema has the correct data types.
        # np.int64 in pandas correctly maps to LongType in Spark.
        # We explicitly cast relevance to DoubleType for safety.
        recs_spark = recs_spark.withColumn("relevance", sf.col("relevance").cast("double"))
        print("--- [GCN] Prediction complete ---")
        return recs_spark 

# =========================================================================================
# === Model 2: LightGCN ===================================================================
# =========================================================================================

class LightGCNConv(MessagePassing):
    """A single, simplified LightGCN layer for propagating embeddings."""
    def __init__(self):
        super().__init__(aggr='add') # 'add' aggregation.

    def forward(self, x, edge_index):
        # Symmetrically normalize the adjacency matrix.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # Messages are just the normalized embeddings of the neighbors.
        return norm.view(-1, 1) * x_j

class LightGCNModel(nn.Module):
    """
    A LightGCN model that simplifies GCN by removing feature transformations and non-linearities.
    It learns user and item embeddings by aggregating them from their neighbors across multiple layers.
    """
    def __init__(self, num_nodes: int, embedding_size: int, num_layers: int):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_size)
        self.convs = nn.ModuleList([LightGCNConv() for _ in range(num_layers)])
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.1)

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        """The forward pass which aggregates embeddings across all layers."""
        x = self.embedding.weight
        # Collect embeddings from all layers
        all_layer_embeddings = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            all_layer_embeddings.append(x)
        # The final embedding is the mean of embeddings from all layers.
        return torch.mean(torch.stack(all_layer_embeddings, dim=0), dim=0)

    @staticmethod
    def decode(z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        src, dst = z[edge_label_index[0]], z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)

class LightGCNRecommender(GraphCNRecommender):
    """
    Recommender implementation using the LightGCNModel. It inherits most of its
    functionality from GraphCNRecommender but uses the specialized LightGCN architecture.
    """
    def __init__(self, seed: Optional[int] = None, **kwargs):
        super().__init__(seed, **kwargs)
        # LightGCN is parameter-free in its layers, so dropout is not applied during convolution.
        self.dropout = 0 # Not used in LightGCN layers

    def fit(self, log: DataFrame, user_features: Optional[DataFrame] = None, item_features: Optional[DataFrame] = None):
        if item_features is None: raise ValueError("`item_features` must be provided.")
        print("\n--- [LightGCN] Starting fit() method ---")
        self.spark = log.sparkSession
        log_pd = log.select("user_idx", "item_idx").toPandas()
        self.item_features_pd = item_features.toPandas()
        all_users = np.union1d(log_pd["user_idx"].unique(), user_features.select("user_idx").toPandas()["user_idx"].unique()) if user_features else log_pd["user_idx"].unique()
        all_items = self.item_features_pd["item_idx"].unique()
        self._build_graph(log_pd, all_users, all_items)
        
        # Instantiate the LightGCNModel instead of the GCNModel
        self.model = LightGCNModel(self.data.num_nodes, self.embedding_size, self.num_layers)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        print(f"--- [LightGCN] Starting training for {self.epochs} epochs ---")
        self.model.train()
        train_edges = self.data.train_edge_index
        num_pos_samples = train_edges.size(1)
        for epoch in range(self.epochs):
            z = self.model(self.data.edge_index)
            neg_edge_index = self._sample_negatives(train_edges, num_pos_samples)
            pos_scores = LightGCNModel.decode(z, train_edges)
            neg_scores = LightGCNModel.decode(z, neg_edge_index)
            loss = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0: print(f"--- [LightGCN] Epoch {epoch+1}/{self.epochs}, BPR Loss: {loss.item():.4f} ---")
        print("--- [LightGCN] Training complete ---")
        return self
    
    def predict(self, log: DataFrame, k: int, users: DataFrame, items: DataFrame, **kwargs) -> DataFrame:
        # This method can be inherited directly, but we override it to change the debug print statements
        if not all([self.model, self.spark]): raise RuntimeError("Call fit() first.")
        print("\n--- [LightGCN] Starting predict() method ---")
        # The rest of the prediction logic is identical to the parent class
        return super().predict(log, k, users, items, **kwargs)

