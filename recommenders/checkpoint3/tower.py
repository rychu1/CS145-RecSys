import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from typing import Optional, Dict, List

# Make sure you have PySpark and scikit-learn installed
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as sf
from sklearn.preprocessing import StandardScaler

# This assumes you have a local file `sample_recommenders.py` with BaseRecommender
from sample_recommenders import BaseRecommender

# --- Modules and Dataset classes are unchanged ---

class TowerModel(nn.Module):
    """
    A generic tower for either users or items. It processes numerical and categorical
    features to produce a single output embedding.
    """
    def __init__(self, numerical_dim: int, categorical_vocabs: Dict[str, int], embedding_dim: int, output_dim: int):
        super().__init__()
        # Embedding layers for each categorical feature
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings, embedding_dim) for num_embeddings in categorical_vocabs.values()
        ])
        
        # MLP to process the concatenated features
        total_input_dim = numerical_dim + len(categorical_vocabs) * embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, total_input_dim * 2),
            nn.ReLU(),
            nn.Linear(total_input_dim * 2, output_dim)
        )

    def forward(self, numerical_feats: torch.Tensor, categorical_feats: torch.Tensor):
        cat_embeds = [
            embedding(categorical_feats[:, i]) for i, embedding in enumerate(self.cat_embeddings)
        ]
        cat_embeds_tensor = torch.cat(cat_embeds, dim=1) if cat_embeds else torch.empty(numerical_feats.shape[0], 0)
        
        all_feats = torch.cat([numerical_feats, cat_embeds_tensor], dim=1)
        return self.mlp(all_feats)

class TwoTowerModel(nn.Module):
    """The main recommender model, containing a tower for users and a tower for items."""
    def __init__(self, user_tower: TowerModel, item_tower: TowerModel):
        super().__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower

    def forward(self, user_feats, item_feats):
        user_embedding = self.user_tower(user_feats[0], user_feats[1])
        item_embedding = self.item_tower(item_feats[0], item_feats[1])
        return (user_embedding * item_embedding).sum(dim=1)

class RecSysDataset(Dataset):
    """PyTorch Dataset for loading interaction triplets (user, pos_item, neg_item)."""
    def __init__(self, log_pd, all_item_ids):
        self.users = log_pd['user_idx'].values
        self.pos_items = log_pd['item_idx'].values
        self.all_item_ids = all_item_ids
        self.user_item_set = set(zip(self.users, self.pos_items))

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user_id = self.users[idx]
        pos_item_id = self.pos_items[idx]
        neg_item_id = np.random.choice(self.all_item_ids)
        while (user_id, neg_item_id) in self.user_item_set:
            neg_item_id = np.random.choice(self.all_item_ids)
        return user_id, pos_item_id, neg_item_id

# --- Recommender Class with Price Weighting ---

class TwoTowerRecommender(BaseRecommender):
    """
    A Two-Tower Recommender that automatically detects feature types and learns from them.
    Includes an option for price-weighting in the final ranking.
    """
    def __init__(self, seed: Optional[int] = None, epochs: int = 500, learning_rate: float = 0.001,
                 embedding_dim: int = 64, output_dim: int = 64, 
                 cardinality_threshold: int = 50,
                 price_weighting: bool = True): # NEW parameter
        super().__init__(seed)
        self.epochs = epochs
        self.lr = learning_rate
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.cardinality_threshold = cardinality_threshold
        self.price_weighting = price_weighting # NEW attribute

        self.user_numerical_cols: List[str] = []
        self.user_categorical_cols: List[str] = []
        self.item_numerical_cols: List[str] = []
        self.item_categorical_cols: List[str] = []
        
        self.model: Optional[TwoTowerModel] = None
        self.spark: Optional[SparkSession] = None
        self.user_features_pd: Optional[pd.DataFrame] = None
        self.item_features_pd: Optional[pd.DataFrame] = None
        
        self.user_scaler = StandardScaler()
        self.item_scaler = StandardScaler()
        self.user_cat_vocabs: Dict[str, Dict[int, int]] = {}
        self.item_cat_vocabs: Dict[str, Dict[int, int]] = {}

    def _discover_feature_types(self, df: pd.DataFrame, id_col: str):
        numerical_cols = []
        categorical_cols = []
        for col in df.columns:
            if col == id_col:
                continue
            if pd.api.types.is_float_dtype(df[col]):
                numerical_cols.append(col)
            elif pd.api.types.is_object_dtype(df[col]):
                categorical_cols.append(col)
            elif pd.api.types.is_integer_dtype(df[col]):
                if df[col].nunique() < self.cardinality_threshold:
                    categorical_cols.append(col)
                else:
                    numerical_cols.append(col)
        return numerical_cols, categorical_cols

    def _preprocess_features(self, features_pd: pd.DataFrame, numerical_cols: List[str], categorical_cols: List[str], is_training: bool):
        num_feats = features_pd[numerical_cols].values.astype(np.float32) if numerical_cols else np.empty((len(features_pd), 0), dtype=np.float32)
        cat_feats = features_pd[categorical_cols].values if categorical_cols else np.empty((len(features_pd), 0), dtype=np.int64)
        is_user_features = 'user_idx' in features_pd.index.name
        
        if numerical_cols:
            scaler = self.user_scaler if is_user_features else self.item_scaler
            if is_training:
                num_feats = scaler.fit_transform(num_feats)
            else:
                num_feats = scaler.transform(num_feats)
        
        if categorical_cols:
            mapped_cat_feats = np.zeros_like(cat_feats, dtype=np.int64)
            vocabs = self.user_cat_vocabs if is_user_features else self.item_cat_vocabs
            for i, col in enumerate(categorical_cols):
                current_col_values = cat_feats[:, i]
                if is_training:
                    unique_vals = np.unique(current_col_values)
                    vocabs[col] = {val: j for j, val in enumerate(unique_vals)}
                mapper = np.vectorize(lambda x: vocabs[col].get(x, 0))
                mapped_cat_feats[:, i] = mapper(current_col_values)
            cat_feats = mapped_cat_feats
        return torch.tensor(num_feats), torch.tensor(cat_feats, dtype=torch.long)

    def fit(self, log: DataFrame, user_features: DataFrame, item_features: DataFrame):
        print("\n--- [TwoTower] Starting fit() method ---")
        self.spark = log.sparkSession
        
        log_pd = log.select("user_idx", "item_idx").toPandas()
        self.user_features_pd = user_features.toPandas().set_index('user_idx')
        self.item_features_pd = item_features.toPandas().set_index('item_idx')
        self.user_features_pd.index.name = 'user_idx'
        self.item_features_pd.index.name = 'item_idx'

        self.user_numerical_cols, self.user_categorical_cols = self._discover_feature_types(self.user_features_pd.reset_index(), 'user_idx')
        self.item_numerical_cols, self.item_categorical_cols = self._discover_feature_types(self.item_features_pd.reset_index(), 'item_idx')

        print("--- [TwoTower] Discovered User Features ---")
        print(f"Numerical: {self.user_numerical_cols}")
        print(f"Categorical: {self.user_categorical_cols}")
        print("--- [TwoTower] Discovered Item Features ---")
        print(f"Numerical: {self.item_numerical_cols}")
        print(f"Categorical: {self.item_categorical_cols}")

        print("--- [TwoTower] Preprocessing features... ---")
        self.user_num_T, self.user_cat_T = self._preprocess_features(self.user_features_pd, self.user_numerical_cols, self.user_categorical_cols, is_training=True)
        self.item_num_T, self.item_cat_T = self._preprocess_features(self.item_features_pd, self.item_numerical_cols, self.item_categorical_cols, is_training=True)

        user_cat_vocab_sizes = {col: len(v) for col, v in self.user_cat_vocabs.items()}
        item_cat_vocab_sizes = {col: len(v) for col, v in self.item_cat_vocabs.items()}

        user_tower = TowerModel(len(self.user_numerical_cols), user_cat_vocab_sizes, self.embedding_dim, self.output_dim)
        item_tower = TowerModel(len(self.item_numerical_cols), item_cat_vocab_sizes, self.embedding_dim, self.output_dim)
        self.model = TwoTowerModel(user_tower, item_tower)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        dataset = RecSysDataset(log_pd, self.item_features_pd.index.values)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
        
        user_id_map = {id: i for i, id in enumerate(self.user_features_pd.index)}
        item_id_map = {id: i for i, id in enumerate(self.item_features_pd.index)}

        print(f"--- [TwoTower] Starting training for {self.epochs} epochs ---")
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for user_ids, pos_item_ids, neg_item_ids in dataloader:
                optimizer.zero_grad()
                user_indices = torch.tensor([user_id_map[uid.item()] for uid in user_ids], dtype=torch.long)
                pos_item_indices = torch.tensor([item_id_map[iid.item()] for iid in pos_item_ids], dtype=torch.long)
                neg_item_indices = torch.tensor([item_id_map[iid.item()] for iid in neg_item_ids], dtype=torch.long)

                user_emb = self.model.user_tower(self.user_num_T[user_indices], self.user_cat_T[user_indices])
                pos_item_emb = self.model.item_tower(self.item_num_T[pos_item_indices], self.item_cat_T[pos_item_indices])
                neg_item_emb = self.model.item_tower(self.item_num_T[neg_item_indices], self.item_cat_T[neg_item_indices])
                
                pos_scores = (user_emb * pos_item_emb).sum(dim=1)
                neg_scores = (user_emb * neg_item_emb).sum(dim=1)
                loss = -F.logsigmoid(pos_scores - neg_scores).mean()
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"--- [TwoTower] Epoch {epoch+1}/{self.epochs}, BPR Loss: {total_loss/len(dataloader):.4f} ---")
        
        print("--- [TwoTower] Training complete ---")
        return self

    # MODIFIED predict method
    def predict(self, log: DataFrame, k: int, users: DataFrame, items: DataFrame,
                user_features: DataFrame, item_features: DataFrame, filter_seen_items: bool = True) -> DataFrame:
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")

        print("\n--- [TwoTower] Starting predict() method ---")
        self.model.eval()

        users_pd = users.select("user_idx").toPandas()
        items_pd = items.select("item_idx").toPandas()
        
        pred_user_features = self.user_features_pd.loc[users_pd['user_idx'].unique()]
        pred_item_features = self.item_features_pd.loc[items_pd['item_idx'].unique()]
        pred_user_features.index.name = 'user_idx'
        pred_item_features.index.name = 'item_idx'

        with torch.no_grad():
            user_num_T, user_cat_T = self._preprocess_features(pred_user_features, self.user_numerical_cols, self.user_categorical_cols, is_training=False)
            item_num_T, item_cat_T = self._preprocess_features(pred_item_features, self.item_numerical_cols, self.item_categorical_cols, is_training=False)
            
            user_embeddings = self.model.user_tower(user_num_T, user_cat_T)
            item_embeddings = self.model.item_tower(item_num_T, item_cat_T)

            all_scores = torch.matmul(user_embeddings, item_embeddings.T)

        # --- MODIFIED: Ranking logic with optional price weighting ---
        recs = []
        user_ids = pred_user_features.index.values
        item_ids = pred_item_features.index.values

        # Create a price lookup dictionary if weighting is enabled
        price_lookup = None
        if self.price_weighting:
            print("--- [TwoTower] Applying price weighting to scores ---")
            if 'price' in self.item_features_pd.columns:
                price_lookup = self.item_features_pd['price'].to_dict()
            else:
                print("Warning: 'price' column not found. Cannot apply price weighting.")

        if filter_seen_items:
            seen_items_by_user = log.select("user_idx", "item_idx").toPandas().groupby('user_idx')['item_idx'].apply(set).to_dict()

        for i, user_id in enumerate(user_ids):
            model_scores = all_scores[i].cpu().numpy()
            
            # Combine model scores with price to get final relevance
            if price_lookup:
                # Multiply score by price, using 1.0 as default if price is missing
                relevance_scores = [model_scores[j] * price_lookup.get(item_id, 1.0) for j, item_id in enumerate(item_ids)]
            else:
                relevance_scores = model_scores
                
            item_relevance = list(zip(item_ids, relevance_scores))
            
            if filter_seen_items:
                seen_items = seen_items_by_user.get(user_id, set())
                item_relevance = [pair for pair in item_relevance if pair[0] not in seen_items]
            
            item_relevance.sort(key=lambda x: x[1], reverse=True)
            for item_id, relevance in item_relevance[:k]:
                recs.append({'user_idx': user_id, 'item_idx': item_id, 'relevance': float(relevance)})
        
        final_recs_pd = pd.DataFrame(recs)
        
        if final_recs_pd.empty:
            return self.spark.createDataFrame(final_recs_pd, schema="user_idx long, item_idx long, relevance double")

        recs_spark = self.spark.createDataFrame(final_recs_pd)
        recs_spark = recs_spark.withColumn("relevance", sf.col("relevance").cast("double"))

        print("--- [TwoTower] Prediction complete ---")
        return recs_spark