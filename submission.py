import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
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

# --- 2. GCN Model (Internal PyTorch Module) ---
class GCNModel(nn.Module):
    """
    A Graph Convolutional Network (GCN) model designed for recommendation.
    This model learns embeddings for users and items by passing messages along the user-item interaction graph.
    It's inspired by LightGCN, focusing on simple convolutions without extra transformations.
    """
    def __init__(self, num_nodes: int, embedding_size: int, num_layers: int, dropout: float):
        super().__init__()
        # --- Layers ---
        # The embedding layer stores a unique vector for each user and item.
        # `max_norm=1.0` is a regularization technique that keeps the embedding vectors from growing too large.
        self.embedding = nn.Embedding(num_nodes, embedding_size, max_norm=1.0)

        # A list of GCN layers. Each layer performs one round of message passing.
        self.convs = nn.ModuleList([GCNConv(embedding_size, embedding_size) for _ in range(num_layers)])

        # Dropout is a regularization technique to prevent overfitting by randomly setting some activations to zero.
        self.dropout = nn.Dropout(p=dropout)

        # --- Initialization ---
        self.init_weights()

    def init_weights(self):
        """Initialize embedding weights with Xavier uniform distribution for better training stability."""
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the model, which computes the final node embeddings.
        This is where the "learning" from the graph structure happens.
        """
        # Start with the initial, randomly initialized embeddings.
        x = self.embedding.weight
        # Propagate information through the GCN layers.
        for conv in self.convs:
            # Pass messages along the edges defined in edge_index.
            x = conv(x, edge_index)
            # Apply a non-linear activation function (ReLU).
            x = torch.relu(x)
            # Apply dropout for regularization.
            x = self.dropout(x)
        return x

    @staticmethod
    def decode(z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        """
        Computes relevance scores for user-item pairs using a dot product.
        A higher dot product between a user and item embedding means they are more similar.
        """
        # Get the final embeddings for the source (users) and destination (items) nodes.
        src, dst = z[edge_label_index[0]], z[edge_label_index[1]]
        # Calculate the dot product to get a single score per pair.
        return (src * dst).sum(dim=-1)

# --- 3. GraphCN Recommender Implementation ---
class MyRecommender(BaseRecommender):
    """
    A graph-based recommender that uses a GCN model to learn from user-item interactions.
    It is trained with Bayesian Personalized Ranking (BPR) loss, which is specifically designed
    for optimizing the ranking of items.

    This recommender is designed to handle "cold-start" scenarios by building its vocabulary
    from all known users and items, not just those with interactions.
    """
    def __init__(self, seed: Optional[int] = None,
                 embedding_size: int = current_params['embedding_size'],
                 num_layers: int = current_params['num_layers'],
                 epochs: int = current_params['epochs'],
                 learning_rate: float = current_params['learning_rate'],
                 weight_decay: float = current_params['weight_decay'],
                 dropout: float = current_params['dropout'],
                 price_weighting: bool = True):
        """
        Initializes the recommender with a set of hyperparameters.

        Args:
            seed (Optional[int]): Random seed for reproducibility.
            embedding_size (int): The dimensionality of the user and item vectors.
            num_layers (int): The number of GCN layers (hops of message passing).
            epochs (int): The number of full passes over the training data.
            learning_rate (float): The step size for the optimizer.
            weight_decay (float): The strength of L2 regularization to prevent overfitting.
            dropout (float): The probability of an element to be zeroed out during training.
            price_weighting (bool): If True, relevance score is `model_score * price`.
        """
        super().__init__(seed)
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.price_weighting = price_weighting

        # These members are initialized during the `fit` and `predict` calls.
        self.model: Optional[GCNModel] = None
        self.data: Optional[Data] = None
        self.user_mapping: dict[int, int] = {}
        self.item_mapping: dict[int, int] = {}
        self.item_features_pd: Optional[pd.DataFrame] = None
        self.spark: Optional[SparkSession] = None

    def fit(self, log: DataFrame, user_features: Optional[DataFrame] = None, item_features: Optional[DataFrame] = None):
        """
        Trains the GCN model on the provided interaction log.
        """
        if item_features is None:
            raise ValueError("`item_features` (with 'item_idx') must be provided.")

        print("\n--- [GraphCN] Starting fit() method ---")
        # Store the spark session to be used later in predict()
        self.spark = log.sparkSession
        # Convert Spark DataFrames to pandas for local processing with PyTorch
        log_pd = log.select("user_idx", "item_idx").toPandas()
        self.item_features_pd = item_features.toPandas()

        # --- Vocabulary Construction ---
        # Build the graph vocabulary from ALL known users and items. This is crucial
        # to ensure the model can make predictions for items that haven't been seen yet.
        if user_features is not None:
            all_users = np.union1d(log_pd["user_idx"].unique(),
                                   user_features.select("user_idx").toPandas()["user_idx"].unique())
        else:
            all_users = log_pd["user_idx"].unique()
        all_items = self.item_features_pd["item_idx"].unique()

        # Build the graph data structure
        self._build_graph(log_pd, all_users, all_items)

        # --- Model and Optimizer Setup ---
        self.model = GCNModel(self.data.num_nodes, self.embedding_size, self.num_layers, self.dropout)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # --- Training Loop ---
        print(f"--- [GraphCN] Starting training for {self.epochs} epochs ---")
        self.model.train() # Set the model to training mode
        train_edges = self.data.train_edge_index
        num_pos_samples = train_edges.size(1)

        for epoch in range(self.epochs):
            # 1. Forward pass: Get the latest node embeddings
            z = self.model(self.data.edge_index)

            # 2. Negative Sampling: For each positive interaction, sample a negative one.
            # This is done in a bipartite-aware way to ensure we only sample user-item pairs.
            try:
                # Use the modern PyG API if available (>=2.4)
                neg_edge_index = negative_sampling(
                    edge_index=self.data.edge_index,
                    num_nodes=self.data.num_nodes,
                    num_neg_samples=num_pos_samples,
                    method='bipartite',
                )
            except (TypeError, AssertionError):
                # Fallback for older PyG versions. This loop ensures we get enough valid samples.
                neg_edges = []
                collected_neg_edges = 0
                while collected_neg_edges < num_pos_samples:
                    raw_neg_batch = negative_sampling(edge_index=self.data.edge_index, num_nodes=self.data.num_nodes, num_neg_samples=num_pos_samples)
                    mask = (raw_neg_batch[0] < self.data.num_users) & (raw_neg_batch[1] >= self.data.num_users)
                    valid_neg_batch = raw_neg_batch[:, mask]
                    neg_edges.append(valid_neg_batch)
                    collected_neg_edges += valid_neg_batch.size(1)
                neg_edge_index = torch.cat(neg_edges, dim=1)[:, :num_pos_samples]

            # 3. Score calculation
            pos_scores = GCNModel.decode(z, train_edges)
            neg_scores = GCNModel.decode(z, neg_edge_index)

            # 4. BPR Loss Calculation: The goal is to make positive scores higher than negative scores.
            loss = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()

            # 5. Backpropagation
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # Gradient clipping for stability
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"--- [GraphCN] Epoch {epoch+1}/{self.epochs}, BPR Loss: {loss.item():.4f} ---")

        print("--- [GraphCN] Training complete ---")
        return self

    def _build_graph(self, log_pd: pd.DataFrame, all_users: np.ndarray, all_items: np.ndarray):
        """Helper method to construct the PyTorch Geometric graph data structure."""
        print("--- [GraphCN] Building graph... ---")
        # Create mappings from original user/item IDs to new, continuous integer indices (0, 1, 2, ...).
        self.user_mapping = {int(u): i for i, u in enumerate(sorted(all_users))}
        self.item_mapping = {int(it): i for i, it in enumerate(sorted(all_items))}

        # Apply these mappings to the interaction log.
        log_pd["user_map_idx"] = log_pd["user_idx"].map(self.user_mapping)
        log_pd["item_map_idx"] = log_pd["item_idx"].map(self.item_mapping)
        log_pd.dropna(subset=['user_map_idx', 'item_map_idx'], inplace=True)

        # Create the edge index tensor for the graph. Item indices are offset by the number of users.
        num_users = len(self.user_mapping)
        user_idx_tensor = torch.tensor(log_pd["user_map_idx"].values, dtype=torch.long)
        item_idx_tensor = torch.tensor(log_pd["item_map_idx"].values, dtype=torch.long) + num_users

        # Create edges in both directions (user->item and item->user) for message passing.
        edge_ui = torch.stack([user_idx_tensor, item_idx_tensor])
        edge_iu = torch.stack([item_idx_tensor, user_idx_tensor])
        edge_index = torch.cat([edge_ui, edge_iu], dim=1)

        # Store everything in a PyG Data object.
        self.data = Data(edge_index=edge_index)
        self.data.num_users = num_users
        self.data.num_items = len(self.item_mapping)
        self.data.num_nodes = self.data.num_users + self.data.num_items
        self.data.train_edge_index = edge_ui # Store user->item edges for BPR training
        print("--- [GraphCN] Graph construction complete ---")


    def predict(self, log: DataFrame, k: int, users: DataFrame, items: DataFrame,
                user_features: Optional[DataFrame] = None, item_features: Optional[DataFrame] = None,
                filter_seen_items: bool = True) -> DataFrame:
        """
        Generates top-K recommendations for the given users.
        """
        if self.model is None or self.spark is None:
            raise RuntimeError("Call fit() before predict().")

        print("\n--- [GraphCN] Starting predict() method ---")
        self.model.eval() # Set the model to evaluation mode

        # --- Candidate Generation ---
        # Create all possible user-item pairs to score.
        users_pd = users.select("user_idx").toPandas()
        items_pd = items.select("item_idx").toPandas()
        all_pairs = users_pd.assign(key=1).merge(items_pd.assign(key=1), on="key").drop("key", axis=1)

        # --- Filtering ---
        # Remove items that users have already interacted with.
        if filter_seen_items:
            seen_pd = log.select("user_idx", "item_idx").toPandas()
            all_pairs = all_pairs.merge(seen_pd, on=["user_idx", "item_idx"], how="left", indicator=True)
            all_pairs = all_pairs[all_pairs["_merge"] == "left_only"].drop("_merge", axis=1)

        # Map original IDs to the model's internal integer indices.
        all_pairs["user_map_idx"] = all_pairs["user_idx"].map(self.user_mapping)
        all_pairs["item_map_idx"] = all_pairs["item_idx"].map(self.item_mapping)
        all_pairs.dropna(subset=["user_map_idx", "item_map_idx"], inplace=True)
        all_pairs = all_pairs.astype({"user_map_idx": int, "item_map_idx": int})

        # --- Scoring ---
        # Prepare the data for PyTorch.
        num_users = self.data.num_users
        src = torch.tensor(all_pairs["user_map_idx"].values, dtype=torch.long)
        dst = torch.tensor(all_pairs["item_map_idx"].values, dtype=torch.long) + num_users
        pred_edge_index = torch.stack([src, dst])

        # Get scores from the model.
        with torch.no_grad():
            z = self.model(self.data.edge_index)
            scores = GCNModel.decode(z, pred_edge_index)
        all_pairs["score"] = scores.cpu().numpy()

        # --- Ranking ---
        # Calculate final relevance, potentially weighting by price.
        if self.price_weighting:
            price_lookup = self.item_features_pd.set_index("item_idx")["price"]
            all_pairs["price"] = all_pairs["item_idx"].map(price_lookup)
            all_pairs["relevance"] = all_pairs["score"] * all_pairs["price"].fillna(1.0)
        else:
            all_pairs["relevance"] = all_pairs["score"]

        # Get the top K recommendations for each user.
        all_pairs.sort_values(["user_idx", "relevance"], ascending=[True, False], inplace=True)
        top_k = all_pairs.groupby("user_idx").head(k)
        final_recs_pd = top_k[['user_idx', 'item_idx', 'relevance']]

        # --- Formatting for Submission ---
        # Convert the pandas DataFrame back to a Spark DataFrame.
        recs_spark = self.spark.createDataFrame(final_recs_pd)
        # Ensure the output schema has the correct data types.
        recs_spark = recs_spark.withColumn("user_idx", sf.col("user_idx").cast("int")) \
                               .withColumn("item_idx", sf.col("item_idx").cast("int")) \
                               .withColumn("relevance", sf.col("relevance").cast("double"))

        print("--- [GraphCN] Prediction complete, returning results ---")
        return recs_spark
