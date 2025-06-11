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
    """The PyTorch Geometric GCN model for link prediction."""
    def __init__(self, num_nodes, embedding_size, num_layers):
        super(GCNModel, self).__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_size)
        self.convs = nn.ModuleList([GCNConv(embedding_size, embedding_size) for _ in range(num_layers)])
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, edge_index):
        x = self.embedding.weight
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        return x

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)

# --- 3. GraphCN Recommender Implementation ---
class GraphCNRecommender(BaseRecommender):
    def __init__(self, seed: Optional[int] = None,
                 embedding_size: int = 256,
                 num_layers: int = 10,           # Using a 2-layer GCN is a safer baseline
                 epochs: int = 200,             # Increased epochs for more training time
                 learning_rate: float = 0.001,  # Lower learning rate for stability
                 weight_decay: float = 1e-5):   # Added L2 regularization to prevent overfitting
        super().__init__(seed=seed)
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = learning_rate
        self.weight_decay = weight_decay 
        
        self.model = None
        self.data = None
        self.user_mapping = None
        self.item_mapping = None
        self.item_features = None
        self.spark = None  # To store the SparkSession

    def fit(self, log: DataFrame, user_features: Optional[DataFrame] = None, item_features: Optional[DataFrame] = None):
        if item_features is None:
            raise ValueError("`item_features` with a 'price' column are required for this recommender.")
        
        self.spark = log.sparkSession
        
        log_pd = log.toPandas()
        self.item_features = item_features.toPandas()

        self._build_graph(log_pd)
        
        self.model = GCNModel(
            num_nodes=self.data.num_nodes,
            embedding_size=self.embedding_size,
            num_layers=self.num_layers
        )
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            z = self.model.forward(self.data.edge_index)
            
            neg_edge_index = negative_sampling(
                edge_index=self.data.edge_index,
                num_nodes=self.data.num_nodes,
                num_neg_samples=self.data.train_edge_index.size(1),
            )
            
            edge_label_index = torch.cat([self.data.train_edge_index, neg_edge_index], dim=1)
            edge_label = torch.cat([
                torch.ones(self.data.train_edge_index.size(1)),
                torch.zeros(neg_edge_index.size(1))
            ], dim=0)
            
            out = self.model.decode(z, edge_label_index)
            loss = loss_fn(out, edge_label)
            loss.backward()
            optimizer.step()
        
        return self

    def _build_graph(self, log_pd: pd.DataFrame):
        """Constructs the PyTorch Geometric graph data object from a pandas DataFrame."""
        unique_users = sorted(log_pd['user_idx'].unique())
        unique_items = sorted(log_pd['item_idx'].unique())
        num_users = len(unique_users)
        num_items = len(unique_items)
        
        self.user_mapping = {idx: i for i, idx in enumerate(unique_users)}
        self.item_mapping = {idx: i for i, idx in enumerate(unique_items)}
        
        log_pd['user_map_idx'] = log_pd['user_idx'].map(self.user_mapping)
        log_pd['item_map_idx'] = log_pd['item_idx'].map(self.item_mapping)

        user_indices = torch.tensor(log_pd['user_map_idx'].values, dtype=torch.long)
        item_indices = torch.tensor(log_pd['item_map_idx'].values, dtype=torch.long) + num_users
        
        edge_index_user_to_item = torch.stack([user_indices, item_indices])
        edge_index_item_to_user = torch.stack([item_indices, user_indices])
        
        self.data = Data(edge_index=torch.cat([edge_index_user_to_item, edge_index_item_to_user], dim=1))
        self.data.num_users = num_users
        self.data.num_items = num_items
        self.data.num_nodes = num_users + num_items
        self.data.train_edge_index = edge_index_user_to_item

    def predict(self, log: DataFrame, k: int, users: DataFrame, items: DataFrame,
                user_features: Optional[DataFrame] = None, item_features: Optional[DataFrame] = None,
                filter_seen_items: bool = True) -> DataFrame:
        if self.model is None or self.spark is None:
            raise RuntimeError("Model must be fitted before making predictions. Ensure fit() was called.")

        self.model.eval()

        users_pd = users.toPandas()
        all_pairs = users_pd.assign(key=1).merge(self.item_features.assign(key=1), on='key').drop('key', axis=1)

        if filter_seen_items:
            seen_items = log.toPandas()
            all_pairs = all_pairs.merge(seen_items, on=['user_idx', 'item_idx'], how='left', indicator=True)
            all_pairs = all_pairs[all_pairs['_merge'] == 'left_only'].drop('_merge', axis=1)

        all_pairs['user_map_idx'] = all_pairs['user_idx'].map(self.user_mapping)
        all_pairs['item_map_idx'] = all_pairs['item_idx'].map(self.item_mapping)

        all_pairs.dropna(subset=['user_map_idx', 'item_map_idx'], inplace=True)
        all_pairs['user_map_idx'] = all_pairs['user_map_idx'].astype(int)
        all_pairs['item_map_idx'] = all_pairs['item_map_idx'].astype(int)
        
        user_node_idx = torch.tensor(all_pairs['user_map_idx'].values, dtype=torch.long)
        item_node_idx = torch.tensor(all_pairs['item_map_idx'].values, dtype=torch.long) + self.data.num_users
        pred_edge_index = torch.stack([user_node_idx, item_node_idx])

        with torch.no_grad():
            z = self.model.forward(self.data.edge_index)
            scores = self.model.decode(z, pred_edge_index).sigmoid()
        
        all_pairs['probability'] = scores.cpu().numpy()
        all_pairs['relevance'] = all_pairs['probability'] * all_pairs['price']
        
        all_pairs.sort_values(by=['user_idx', 'relevance'], ascending=[True, False], inplace=True)
        top_k_recs = all_pairs.groupby('user_idx').head(k)
        
        final_recs_pd = top_k_recs[['user_idx', 'item_idx', 'relevance']]
        
        recs_spark = self.spark.createDataFrame(final_recs_pd)
        
        recs_spark = recs_spark.withColumn("user_idx", sf.col("user_idx").cast("int")) \
                               .withColumn("item_idx", sf.col("item_idx").cast("int")) \
                               .withColumn("relevance", sf.col("relevance").cast("double"))

        return recs_spark
