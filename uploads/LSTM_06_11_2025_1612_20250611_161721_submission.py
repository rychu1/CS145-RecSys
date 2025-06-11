# ===== at the top of recommender_analysis_visualization.py, add these imports =====

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pyspark.sql.functions import col
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np

from sim4rec.utils import pandas_to_spark  # for converting pandas → Spark

# ===== Replace the existing MyRecommender with this implementation =====

# ===== at the top of recommender_analysis_visualization.py =====

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from pyspark.sql.functions import col
from sim4rec.utils import pandas_to_spark
import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
import pandas as pd

class BaseRecommender:
    def __init__(self, seed=None):
        self.seed = seed
        np.random.seed(seed)
    def fit(self, log, user_features=None, item_features=None):
        """
        No training needed for random recommender.
        
        Args:
            log: Interaction log
            user_features: User features (optional)
            item_features: Item features (optional)
        """
        # No training needed
        raise NotImplemented()
    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        raise NotImplemented()
    
import sklearn 
from sklearn.preprocessing import StandardScaler
from sim4rec.utils import pandas_to_spark
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
import torch
import torch.nn as nn

class LSTMRecModel(nn.Module):
    """
    An LSTM‐based next‐item predictor that incorporates:
      1) item embeddings (for item IDs),
      2) category embeddings (for each item’s category),
      3) price embeddings (projecting a scalar price → hidden),
      4) static user features (projected into initial hidden state of the LSTM).
    Input:
      - item_seq:    LongTensor of shape (B, L) with shifted item IDs (0=PAD).
      - cat_seq:     LongTensor of shape (B, L) with shifted category IDs (0=PAD).
      - price_seq:   FloatTensor of shape (B, L) with scaled price (float).
      - user_feat:   FloatTensor of shape (B, user_feat_dim) with static user features.
    Output:
      - logits: FloatTensor of shape (B, num_items) giving scores for “next‐item” (only final time‐step used).
    """

    def __init__(
        self,
        num_items: int,
        item_emb_dim: int,
        num_categories: int,
        category_emb_dim: int,
        price_hidden_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        padding_idx: int,
        user_feat_dim: int
    ):
        """
        Args:
          num_items:       vocabulary size of item IDs (including PAD=0).
          item_emb_dim:    embedding dimension for item IDs.
          num_categories:  vocabulary size of categories (including PAD=0).
          category_emb_dim: embedding dimension for category IDs.
          price_hidden_dim: dimension to project scalar price→vector at each time‐step.
          hidden_dim:       hidden dimension of the LSTM.
          num_layers:       number of stacked LSTM layers.
          dropout:          dropout between LSTM layers (if num_layers>1).
          padding_idx:      index reserved for PAD in both item & category embeddings.
          user_feat_dim:    dimension of static user features (after one‐hot & scaling).
        """
        super().__init__()
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 1) Item embedding: PAD=0
        self.item_embedding = nn.Embedding(num_items, item_emb_dim, padding_idx=padding_idx)

        # 2) Category embedding: PAD=0
        self.cat_embedding = nn.Embedding(num_categories, category_emb_dim, padding_idx=padding_idx)

        # 3) Price projection: at each time‐step we have a scalar price. We expand to (B,L,1) then:
        #    FloatTensor → Linear(1 → price_hidden_dim) → ReLU
        self.price_proj = nn.Sequential(
            nn.Linear(1, price_hidden_dim),
            nn.ReLU()
        )

        # 4) LSTM: input size = item_emb_dim + category_emb_dim + price_hidden_dim
        self.input_dim = item_emb_dim + category_emb_dim + price_hidden_dim
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # 5) Project user features → initial hidden state (num_layers * hidden_dim)
        self.user_feat_proj = nn.Linear(user_feat_dim, num_layers * hidden_dim)

        # 6) Final classifier: hidden_dim → num_items
        self.fc = nn.Linear(hidden_dim, num_items)

    def forward(
        self,
        item_seq: torch.LongTensor,
        cat_seq: torch.LongTensor,
        price_seq: torch.FloatTensor,
        user_feat: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Args:
          item_seq:   (B, L) LongTensor of shifted item IDs (0=PAD).
          cat_seq:    (B, L) LongTensor of shifted category IDs (0=PAD).
          price_seq:  (B, L) FloatTensor of scaled prices.
          user_feat:  (B, user_feat_dim) FloatTensor of static user features.
        Returns:
          logits: (B, num_items) – scores over next‐item vocabulary (only final time‐step used).
        """
        B, L = item_seq.size()

        # a) Embed items → (B, L, item_emb_dim)
        item_emb = self.item_embedding(item_seq)

        # b) Embed categories → (B, L, category_emb_dim)
        cat_emb = self.cat_embedding(cat_seq)

        # c) Project price: price_seq is (B, L). Unsqueeze → (B, L, 1), then:
        #    (B, L, 1) → price_proj → (B, L, price_hidden_dim)
        price_in = price_seq.unsqueeze(-1)          # (B, L, 1)
        price_emb = self.price_proj(price_in)        # (B, L, price_hidden_dim)

        # d) Concatenate along the last dimension → (B, L, input_dim)
        x = torch.cat([item_emb, cat_emb, price_emb], dim=-1)  # (B, L, input_dim)

        # e) Build initial hidden state h0 from user features:
        #    user_feat: (B, user_feat_dim) → Linear → (B, num_layers * hidden_dim)
        #    reshape → (num_layers, B, hidden_dim)
        h0 = self.user_feat_proj(user_feat)  # (B, num_layers * hidden_dim)
        h0 = h0.view(B, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        #    Initialize c0 = zeros
        c0 = torch.zeros(self.num_layers, B, self.hidden_dim, device=x.device)

        # f) Run LSTM:
        #    x: (B, L, input_dim), (h0, c0) → out: (B, L, hidden_dim), (hn, cn)
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # g) Take final time‐step’s hidden (top layer): hn[-1] → (B, hidden_dim)
        final_hidden = hn[-1]

        # h) Project to logits over items: (B, hidden_dim) → (B, num_items)
        logits = self.fc(final_hidden)
        return logits



# --- 3. GraphCN Recommender Implementation ---
class MyRecommender(BaseRecommender):
    """
    LSTM‐based sequential recommender with embeddings for:
      – item IDs,
      – each item’s category,
      – each item’s price (scalar → hidden),
      – static user features (e.g. user_attr_*, segment).
    Training uses “teacher‐forcing” on the final time step. Inference feeds a user’s full history
    of length L to score next‐item logits.
    """

    def __init__(
        self,
        seed: int = 42,
        item_emb_dim: int = 128,
        category_emb_dim: int = 16,
        price_hidden_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
        max_seq_len: int = 50,
        batch_size: int = 64,
        num_epochs: int = 30,
        lr: float = 1e-3,
        device: str = None
    ):
        """
        Args:
            seed: random seed
            item_emb_dim: dimension of item‐ID embedding
            category_emb_dim: dimension of category embedding
            price_hidden_dim: dimension to project scalar price→vector
            hidden_dim: LSTM hidden size
            num_layers: number of LSTM layers (1–3)
            dropout: dropout between LSTM layers (0.1–0.5 if num_layers>1)
            max_seq_len: each user sequence is padded/truncated to this length
            batch_size, num_epochs, lr: training hyperparameters
            device: 'cuda' or 'cpu' (auto‐detect if None)
        """
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.item_emb_dim = item_emb_dim
        self.category_emb_dim = category_emb_dim
        self.price_hidden_dim = price_hidden_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_seq_len = max_seq_len

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr

        # Device selection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Placeholders (set in fit)
        self.model = None
        self.optimizer = None
        self.criterion = None

        # Will set these in fit():
        self.num_items = None        # total distinct items + 1 for PAD
        self.num_categories = None   # distinct categories + 1 for PAD
        self.item_offset = 1         # shift original item_idx→[1..], 0 reserved for PAD
        self.cat_offset = 1          # shift original category→[1..], 0 reserved for PAD

        # Per‐user sequences (populated in fit):
        #   item → list of shifted item IDs,
        #   cat → list of shifted category IDs,
        #   price → list of scaled prices
        self.user_item_seq = {}
        self.user_cat_seq = {}
        self.user_price_seq = {}

        # Static user features:
        # We will one‐hot encode 'segment' and scale all 20 'user_attr_*' plus the one‐hot columns.
        self.user_features_dict = {}   # user_idx → np.ndarray(user_feat_dim,)

        # Scalers
        self.price_scaler = StandardScaler()
        self.user_feat_scaler = StandardScaler()

        self.category_to_idx = {}    # category string → shifted int
        self.user_feat_dim = None    # dimension after scaling & one‐hot
        self.item_feat_cols = None   # not strictly needed here

    def fit(self, log, user_features=None, item_features=None):
        """
        Train the LSTM on a next‐item task (teacher forcing on final time step).
        Steps:
         1. Convert `log` to Pandas, sort by timestamp.
         2. One‐hot + scale `user_features`.
         3. One‐hot + scale `item_features`.
         4. Build `self.user_features_dict[user_idx] → np.ndarray(user_feat_dim,)`.
         5. Merge `log` with raw item_features to attach `category` & `price`.
         6. Build per‐user sequences of (item_shifted, category_shifted, price_scaled).
         7. Pad/truncate each to length L, produce training arrays:
               item_inputs, cat_inputs, price_inputs, and targets (all shape (num_users, L)).
         8. Create a Dataset returning (item_seq, cat_seq, price_seq, user_feat, target_seq).
         9. Instantiate LSTMRecModelWithUserFeatures and train.
        """
        # 1. Pull log → pandas, sort by timestamp
        pandas_log = log.toPandas()
        if "timestamp" in pandas_log.columns:
            pandas_log = pandas_log.sort_values(by="timestamp")
        else:
            pandas_log = pandas_log.sort_index()

        # 2. Pull user_features → pandas, one‐hot + scale
        if user_features is None:
            raise ValueError("`user_features` Spark DataFrame must be provided.")
        users_pd = user_features.toPandas()

        # One‐hot encode 'segment'
        if "segment" in users_pd.columns:
            users_pd = pd.get_dummies(users_pd, columns=["segment"], prefix="segment")
        user_feat_cols = [c for c in users_pd.columns if c != "user_idx"]
        if len(user_feat_cols) == 0:
            raise ValueError("After one‐hot encoding, no user feature columns remain (besides user_idx).")

        user_matrix = users_pd[user_feat_cols].values.astype(float)  # shape (n_users, user_feat_dim)
        self.user_feat_scaler.fit(user_matrix)
        scaled_user_feats = self.user_feat_scaler.transform(user_matrix)  # (n_users, user_feat_dim)
        self.user_feat_dim = scaled_user_feats.shape[1]

        # Build user_features_dict
        self.user_features_dict = {
            int(uid): scaled_user_feats[i]
            for i, uid in enumerate(users_pd["user_idx"].astype(int))
        }

        # 3. Pull item_features → pandas, one‐hot encode 'category', keep 'price'
        if item_features is None:
            raise ValueError("`item_features` Spark DataFrame must be provided.")
        items_pd = item_features.toPandas()

        if "price" not in items_pd.columns:
            raise ValueError("Your `item_features` DataFrame must contain a column named 'price'.")
        if "category" not in items_pd.columns:
            raise ValueError("Your `item_features` DataFrame must contain a column named 'category' for one‐hot encoding.")

        items_onehot = pd.get_dummies(items_pd, columns=["category"], prefix="category")
        item_feat_cols = [c for c in items_onehot.columns if c != "item_idx"]
        if len(item_feat_cols) == 0:
            raise ValueError("After one‐hot encoding, no item feature columns remain (besides item_idx).")
        self.item_feat_cols = item_feat_cols

        # Fit price scaler on items_onehot["price"]
        all_prices = items_onehot[["price"]].values.astype(float).reshape(-1, 1)
        self.price_scaler.fit(all_prices)

        # 4. Build category→idx map (reserve 0 for PAD)
        unique_categories = items_pd["category"].astype(str).unique().tolist()
        self.category_to_idx = {cat: idx + self.cat_offset for idx, cat in enumerate(unique_categories)}
        self.num_categories = len(self.category_to_idx) + 1  # +1 for PAD=0

        # 5. Determine number of distinct items; reserve index 0 for PAD
        max_orig_item = pandas_log["item_idx"].max()
        self.num_items = int(max_orig_item) + 2  # +1 shift, +1 for PAD=0

        # 6. Merge `pandas_log` with raw item_features to attach `category` & `price`
        raw_item_df = item_features.toPandas()[["item_idx", "category", "price"]]
        merged = pandas_log.merge(raw_item_df, on="item_idx", how="left")

        # Map category → shifted int (0 = PAD)
        merged["category_shifted"] = (
            merged["category"].astype(str)
                  .map(self.category_to_idx)
                  .fillna(0)
                  .astype(int)
        )

        # Scale price → price_scaled
        if "price" not in merged.columns:
            raise KeyError("After merging, expected `merged` to have a column named 'price'.")
        merged["price_scaled"] = self.price_scaler.transform(merged[["price"]])

        # Shift item IDs by +1 (0=PAD)
        merged["item_shifted"] = merged["item_idx"].astype(int) + self.item_offset

        # 7. Build per‐user sequences
        self.user_item_seq.clear()
        self.user_cat_seq.clear()
        self.user_price_seq.clear()
        for uid, group in merged.groupby("user_idx"):
            self.user_item_seq[int(uid)] = group["item_shifted"].tolist()
            self.user_cat_seq[int(uid)]  = group["category_shifted"].tolist()
            self.user_price_seq[int(uid)] = group["price_scaled"].tolist()

        # 8. Build training arrays (num_users × L) of inputs vs. targets
        user_ids = list(self.user_item_seq.keys())
        num_users = len(user_ids)
        L = self.max_seq_len

        item_inputs  = np.zeros((num_users, L), dtype=np.int64)
        cat_inputs   = np.zeros((num_users, L), dtype=np.int64)
        price_inputs = np.zeros((num_users, L), dtype=np.float32)
        targets      = np.zeros((num_users, L), dtype=np.int64)

        for idx, uid in enumerate(user_ids):
            seq_items  = self.user_item_seq[uid]
            seq_cats   = self.user_cat_seq[uid]
            seq_prices = self.user_price_seq[uid]
            T = len(seq_items)

            # Build “input at t” by shifting right with a 0 in front
            input_items  = [0] + seq_items[:-1]
            input_cats   = [0] + seq_cats[:-1]
            input_prices = [0.0] + seq_prices[:-1]
            target_items = seq_items[:]

            if T >= L:
                item_inputs[idx, :]  = np.array(input_items[-L:], dtype=np.int64)
                cat_inputs[idx, :]   = np.array(input_cats[-L:], dtype=np.int64)
                price_inputs[idx, :] = np.array(input_prices[-L:], dtype=np.float32)
                targets[idx, :]      = np.array(target_items[-L:], dtype=np.int64)
            else:
                pad_len = L - T
                item_inputs[idx, :]  = np.array([0]*pad_len + input_items, dtype=np.int64)
                cat_inputs[idx, :]   = np.array([0]*pad_len + input_cats, dtype=np.int64)
                price_inputs[idx, :] = np.array([0.0]*pad_len + input_prices, dtype=np.float32)
                targets[idx, :]      = np.array([0]*pad_len + target_items, dtype=np.int64)

        # 9. Build PyTorch Dataset returning (item_seq, cat_seq, price_seq, user_feat, target_seq)
        class SeqDataset(Dataset):
            def __init__(self, item_seq, cat_seq, price_seq, user_feats, target_seq):
                self.items     = torch.from_numpy(item_seq).long()   # (num_users, L)
                self.cats      = torch.from_numpy(cat_seq).long()    # (num_users, L)
                self.prices    = torch.from_numpy(price_seq).float() # (num_users, L)
                self.user_feats= torch.from_numpy(user_feats).float()# (num_users, user_feat_dim)
                self.targets   = torch.from_numpy(target_seq).long() # (num_users, L)

            def __len__(self):
                return self.items.size(0)

            def __getitem__(self, idx):
                return (
                    self.items[idx],      # (L,)
                    self.cats[idx],       # (L,)
                    self.prices[idx],     # (L,)
                    self.user_feats[idx], # (user_feat_dim,)
                    self.targets[idx]     # (L,)
                )

        # Build user_feats_array in the same order as user_ids
        user_feats_array = np.vstack([
            self.user_features_dict.get(uid, np.zeros(self.user_feat_dim,))
            for uid in user_ids
        ])

        dataset = SeqDataset(item_inputs, cat_inputs, price_inputs, user_feats_array, targets)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False
        )

        # 10. Instantiate LSTM model
        self.model = LSTMRecModel(
            num_items=self.num_items,
            item_emb_dim=self.item_emb_dim,
            num_categories=self.num_categories,
            category_emb_dim=self.category_emb_dim,
            price_hidden_dim=self.price_hidden_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            padding_idx=0,
            user_feat_dim=self.user_feat_dim
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        # 11. Training loop (teacher forcing on final time‐step)
        self.model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for batch_items, batch_cats, batch_prices, batch_user_feats, batch_targets in dataloader:
                batch_items      = batch_items.to(self.device)        # (B, L)
                batch_cats       = batch_cats.to(self.device)         # (B, L)
                batch_prices     = batch_prices.to(self.device)       # (B, L)
                batch_user_feats = batch_user_feats.to(self.device)   # (B, user_feat_dim)
                batch_targets    = batch_targets.to(self.device)      # (B, L)

                self.optimizer.zero_grad()
                logits = self.model(batch_items, batch_cats, batch_prices, batch_user_feats)  # (B, num_items)

                # Only compute loss on the final time‐step
                final_logits  = logits                 # (B, num_items)
                final_targets = batch_targets[:, -1]    # (B,)
                loss = self.criterion(final_logits, final_targets)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * batch_items.size(0)

            avg_loss = epoch_loss / len(dataset)
            print(f"[Epoch {epoch+1}/{self.num_epochs}] training loss = {avg_loss:.4f}")

        self.model.eval()

    def predict(
        self,
        log,
        k: int,
        users,
        items,
        user_features=None,
        item_features=None,
        filter_seen_items: bool = True
    ):
        """
        For each user in `users`, build a padded sequence of length L = max_seq_len:
          – item_seq[t] = item at (t−1) (0 if t=0 or pad),
          – cat_seq[t] = category at (t−1),
          – price_seq[t] = price_scaled at (t−1).
        Also look up each user’s static feature vector. Then:
          • Feed (item_seq, cat_seq, price_seq, user_feat) through the LSTM to get
            final‐time logits (B, num_items).
          • Mask out seen items (if requested).
          • Take top‐k, and return a Spark DataFrame with (user_idx, item_idx, relevance).
        """
        # 1. Pull latest log → pandas; merge with raw item_features to get 'category' & 'price'
        pandas_log = log.toPandas()
        if "timestamp" in pandas_log.columns:
            pandas_log = pandas_log.sort_values(by="timestamp")
        else:
            pandas_log = pandas_log.sort_index()

        if item_features is None:
            raise ValueError("Must pass `item_features` to predict().")
        items_pd = item_features.toPandas()
        raw_item_df = items_pd[["item_idx", "category", "price"]]

        merged = pandas_log.merge(raw_item_df, on="item_idx", how="left")
        merged["category_shifted"] = (
            merged["category"].astype(str)
                  .map(self.category_to_idx)
                  .fillna(0)
                  .astype(int)
        )
        merged["price_scaled"] = self.price_scaler.transform(merged[["price"]])
        merged["item_shifted"] = merged["item_idx"].astype(int) + self.item_offset

        # Rebuild per‐user sequences
        current_user_item  = {}
        current_user_cat   = {}
        current_user_price = {}
        for uid, grp in merged.groupby("user_idx"):
            current_user_item[int(uid)]  = grp["item_shifted"].tolist()
            current_user_cat[int(uid)]   = grp["category_shifted"].tolist()
            current_user_price[int(uid)] = grp["price_scaled"].tolist()

        # 2. Re‐pull user_features to update one‐hot + scale
        if user_features is None:
            raise ValueError("Must pass `user_features` to predict().")
        users_pd = user_features.toPandas()
        if "segment" in users_pd.columns:
            users_pd = pd.get_dummies(users_pd, columns=["segment"], prefix="segment")
        user_feat_cols = [c for c in users_pd.columns if c != "user_idx"]
        if len(user_feat_cols) > 0:
            user_matrix = users_pd[user_feat_cols].values.astype(float)
            scaled_user_feats = self.user_feat_scaler.transform(user_matrix)
            self.user_features_dict = {
                int(uid): scaled_user_feats[i]
                for i, uid in enumerate(users_pd["user_idx"].astype(int))
            }

        # 3. Build padded input sequences for each user in `users`
        users_pd2 = users.toPandas()
        user_ids = users_pd2["user_idx"].astype(int).tolist()
        num_users = len(user_ids)
        L = self.max_seq_len

        batch_user_items  = []
        batch_user_cats   = []
        batch_user_prices = []
        batch_user_feats  = []

        for uid in user_ids:
            seq_items  = current_user_item.get(uid, [])
            seq_cats   = current_user_cat.get(uid, [])
            seq_prices = current_user_price.get(uid, [])
            T = len(seq_items)

            if T == 0:
                batch_user_items.append([0] * L)
                batch_user_cats.append([0] * L)
                batch_user_prices.append([0.0] * L)
            else:
                input_items  = [0] + seq_items[:-1]
                input_cats   = [0] + seq_cats[:-1]
                input_prices = [0.0] + seq_prices[:-1]

                if len(input_items) >= L:
                    batch_user_items.append(input_items[-L:])
                    batch_user_cats.append(input_cats[-L:])
                    batch_user_prices.append(input_prices[-L:])
                else:
                    pad_len = L - len(input_items)
                    batch_user_items.append([0]*pad_len + input_items)
                    batch_user_cats.append([0]*pad_len + input_cats)
                    batch_user_prices.append([0.0]*pad_len + input_prices)

            user_feat_vec = self.user_features_dict.get(uid, np.zeros(self.user_feat_dim,))
            batch_user_feats.append(user_feat_vec)

        if num_users == 0:
            empty_pd = pd.DataFrame(columns=["user_idx", "item_idx", "relevance"])
            return pandas_to_spark(empty_pd)

        # Convert to tensors
        items_tensor      = torch.LongTensor(batch_user_items).to(self.device)     # (U, L)
        cats_tensor       = torch.LongTensor(batch_user_cats).to(self.device)      # (U, L)
        prices_tensor     = torch.FloatTensor(batch_user_prices).to(self.device)   # (U, L)
        user_feats_tensor = torch.FloatTensor(np.stack(batch_user_feats, axis=0)).to(self.device)  # (U, user_feat_dim)

        # 4. Forward pass
        with torch.no_grad():
            logits = self.model(items_tensor, cats_tensor, prices_tensor, user_feats_tensor)  # (U, num_items)
            if filter_seen_items:
                mask = torch.zeros_like(logits, dtype=torch.bool)  # (U, num_items)
                for i, uid in enumerate(user_ids):
                    seen = set(current_user_item.get(uid, []))
                    for s in seen:
                        if 0 <= s < self.num_items:
                            mask[i, s] = True
                logits = logits.masked_fill(mask, float("-inf"))

            topk_vals, topk_idx = torch.topk(logits, k, dim=1)  # both: (U, k)

        # 5. Build Pandas DataFrame
        rec_rows = []
        for i, uid in enumerate(user_ids):
            for j in range(k):
                shifted_item = int(topk_idx[i, j].item())
                score = float(topk_vals[i, j].item())
                orig_item = shifted_item - self.item_offset
                if orig_item < 0:
                    continue
                rec_rows.append({
                    "user_idx": uid,
                    "item_idx": orig_item,
                    "relevance": score
                })

        if not rec_rows:
            empty_pd = pd.DataFrame(columns=["user_idx", "item_idx", "relevance"])
            return pandas_to_spark(empty_pd)

        recs_pd = pd.DataFrame(rec_rows)
        recs_spark = pandas_to_spark(recs_pd)
        recs_spark = (
            recs_spark
            .withColumn("user_idx", col("user_idx").cast("int"))
            .withColumn("item_idx", col("item_idx").cast("int"))
        )
        return recs_spark