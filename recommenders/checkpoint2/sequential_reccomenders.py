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
from recommenders.checkpoint2.RNN import RNNRecModel                     # our new model 
from recommenders.checkpoint2.LSTM import LSTMRecModel                   # if you want to use LSTM instead of GRU
from sim4rec.utils import pandas_to_spark
from recommenders.checkpoint2.transformer import TransformerRecModel        # the model we just defined
# ================================================================================

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

# ================================================================================


class RNNRecommender:
    """
    RNN‐based sequential recommender that incorporates:
      • item embeddings (for item IDs),
      • item side‐features (e.g. price, category one‐hot),
      • static user features (e.g. user_attr_*, segment one‐hot).
    At training time, we build many (prefix, next‐item) examples with sliding windows,
    each labeled by the next item. At inference, we feed the user’s entire history,
    condition on that user’s static features, and produce logits for candidate items.
    """

    def __init__(
        self,
        seed: int = 42,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
        max_seq_len: int = 50,
        batch_size: int = 64,
        num_epochs: int = 5,
        lr: float = 1e-3,
        device: str = None
    ):
        """
        Args:
            seed: random seed
            embedding_dim, hidden_dim: dims for RNN (+ item‐feature projection)
            num_layers, dropout: for GRU
            max_seq_len: truncate/pad each user sequence to this length
            batch_size, num_epochs, lr: training hyperparams
            device: 'cuda' or 'cpu' (if None, auto‐detect)
        """
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_seq_len = max_seq_len

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr

        # PyTorch device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # placeholders; will be set in fit()
        self.model = None                 # RNNRecModelWithFeatures
        self.optimizer = None
        self.criterion = None

        self.num_items = None             # size of item‐vocab + 1 (for padding)
        self.item_offset = 1              # we shift item_idx→[1..num_items-1], pad=0

        # scalers for numeric features
        self.user_feat_scaler = StandardScaler()
        self.item_feat_scaler = StandardScaler()

        self.user_feat_dim = None
        self.item_feat_dim = None

        # to hold per‐user history sequences of shifted item IDs
        self.user_to_seq = {}

        # to hold static user feature vectors: user_idx → np.ndarray(user_feat_dim,)
        self.user_features_dict = {}

        # to hold side‐item feature vectors: item_idx → np.ndarray(item_feat_dim,)
        self.item_features_dict = {}

    def fit(self, log, user_features=None, item_features=None):
        """
        Train the RNN with item‐IDs + item side‐features + user static features.

        Args:
            log: Spark DataFrame with columns ['user_idx','item_idx','relevance','timestamp'].
            user_features: Spark DataFrame with column 'user_idx', 'user_attr_*' (20 floats), and 'segment' (string).
            item_features: Spark DataFrame with column 'item_idx', 'item_attr_*' (20 floats), 'category' (string), 'price' (float).
        """
        if user_features is None or item_features is None:
            raise ValueError("Both user_features and item_features must be provided for this recommender.")

        # 1. Pull log into pandas + sort by timestamp
        pandas_log = log.toPandas()
        if "timestamp" in pandas_log.columns:
            pandas_log = pandas_log.sort_values(by="timestamp")
        else:
            pandas_log = pandas_log.sort_index()

        # 2. Pull user_features and item_features into pandas
        users_pd = user_features.toPandas()
        items_pd = item_features.toPandas()

        # 3. One‐hot encode 'segment' in users_pd
        #    All other columns in users_pd that start with "user_attr_" are already numeric.
        if "segment" in users_pd.columns:
            users_pd = pd.get_dummies(users_pd, columns=["segment"], prefix="segment")
        #    Now find all numeric user‐feature columns (user_attr_* plus segment_* dummies)
        user_feat_cols = [c for c in users_pd.columns if c != "user_idx"]

        if len(user_feat_cols) == 0:
            raise ValueError("No user feature columns found after dropping 'user_idx'—check your DataFrame.")

        # 4. Fit scaler on user features
        user_feat_matrix = users_pd[user_feat_cols].values.astype(float)  # shape (n_users, user_feat_dim)
        self.user_feat_scaler.fit(user_feat_matrix)
        scaled_user_feats = self.user_feat_scaler.transform(user_feat_matrix)
        self.user_feat_dim = scaled_user_feats.shape[1]

        # 5. Build dictionary: user_idx → scaled feature vector
        for i, uid in enumerate(users_pd["user_idx"].astype(int)):
            self.user_features_dict[int(uid)] = scaled_user_feats[i]

        # 6. One‐hot encode 'category' in items_pd; keep 'price' and 'item_attr_*'
        if "category" in items_pd.columns:
            items_pd = pd.get_dummies(items_pd, columns=["category"], prefix="category")
        #    All columns that start with "item_attr_" are numeric; "price" is numeric; plus any category_* dummies.
        item_feat_cols = [c for c in items_pd.columns if c not in ("item_idx",)]

        if len(item_feat_cols) == 0:
            raise ValueError("No item feature columns found after dropping 'item_idx'—check your DataFrame.")

        # 7. Fit scaler on item features
        item_feat_matrix = items_pd[item_feat_cols].values.astype(float)  # shape (n_items, item_feat_dim)
        self.item_feat_scaler.fit(item_feat_matrix)
        scaled_item_feats = self.item_feat_scaler.transform(item_feat_matrix)
        self.item_feat_dim = scaled_item_feats.shape[1]

        # 8. Build dictionary: item_idx → scaled feature vector
        for i, iid in enumerate(items_pd["item_idx"].astype(int)):
            self.item_features_dict[int(iid)] = scaled_item_feats[i]

        # 9. Re‐index item IDs so that 0 is reserved for PAD, i.e. shift every item by +1.
        max_orig_item = pandas_log["item_idx"].max()
        self.num_items = int(max_orig_item) + 2  # +1 shift, +1 for PAD=0
        pandas_log["item_shifted"] = pandas_log["item_idx"].astype(int) + self.item_offset

        # 10. Build per‐user sequence of shifted item IDs
        self.user_to_seq.clear()
        for user_id, group in pandas_log.groupby("user_idx"):
            seq = group["item_shifted"].tolist()
            self.user_to_seq[int(user_id)] = seq

        # 11. Prepare training examples with sliding windows
        #     We accumulate lists of:
        #       • inputs_item_ids (shape L),
        #       • inputs_item_feat_seq (shape L × item_feat_dim),
        #       • inputs_user_feat (shape user_feat_dim),
        #       • targets (scalar next_item_shifted).
        inputs_item = []
        inputs_item_feat = []
        inputs_user_feat = []
        targets = []

        L = self.max_seq_len
        for uid, seq in self.user_to_seq.items():
            # get this user’s static feature vector (or zeros if missing)
            user_feat_vec = self.user_features_dict.get(int(uid), np.zeros(self.user_feat_dim, dtype=float))

            T = len(seq)
            for idx in range(1, T):
                prefix = seq[:idx]
                next_item = seq[idx]

                # a) Build input_item_ids: pad/truncate prefix to length L
                if len(prefix) >= L:
                    input_ids = prefix[-L:]
                else:
                    pad_len = L - len(prefix)
                    input_ids = [0] * pad_len + prefix

                # b) Build input_item_feat_seq: lookup each ID’s side features
                feat_seq = []
                for sid in input_ids:
                    if sid == 0:
                        feat_seq.append(np.zeros(self.item_feat_dim, dtype=float))
                    else:
                        orig_id = sid - self.item_offset
                        feat_seq.append(self.item_features_dict.get(orig_id, np.zeros(self.item_feat_dim, dtype=float)))
                feat_seq = np.stack(feat_seq, axis=0)  # shape (L, item_feat_dim)

                inputs_item.append(np.array(input_ids, dtype=np.int64))
                inputs_item_feat.append(feat_seq.astype(np.float32))
                inputs_user_feat.append(user_feat_vec.astype(np.float32))
                targets.append(np.int64(next_item))

        if len(inputs_item) == 0:
            # No training examples
            return

        # 12. Stack into arrays
        X_items = np.stack(inputs_item, axis=0)            # (N, L)
        X_item_feats = np.stack(inputs_item_feat, axis=0)  # (N, L, item_feat_dim)
        X_user_feats = np.stack(inputs_user_feat, axis=0)  # (N, user_feat_dim)
        Y = np.stack(targets, axis=0)                      # (N,)

        # 13. Create PyTorch Dataset
        class FeatureSeqDataset(Dataset):
            def __init__(self, items, item_feats, user_feats, targets):
                self.items = torch.from_numpy(items).long()             # (N, L)
                self.item_feats = torch.from_numpy(item_feats).float()   # (N, L, item_feat_dim)
                self.user_feats = torch.from_numpy(user_feats).float()   # (N, user_feat_dim)
                self.targets = torch.from_numpy(targets).long()          # (N,)
            def __len__(self):
                return self.items.size(0)
            def __getitem__(self, idx):
                return (
                    self.items[idx],      # (L,)
                    self.item_feats[idx],  # (L, item_feat_dim)
                    self.user_feats[idx],  # (user_feat_dim,)
                    self.targets[idx]      # scalar
                )

        dataset = FeatureSeqDataset(X_items, X_item_feats, X_user_feats, Y)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False
        )

        # 14. Instantiate model / optimizer / criterion
        self.model = RNNRecModel(
            num_items=self.num_items,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            padding_idx=0,
            item_feat_dim=self.item_feat_dim,
            user_feat_dim=self.user_feat_dim
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # 15. Training loop
        self.model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for batch_items, batch_item_feats, batch_user_feats, batch_targets in dataloader:
                batch_items = batch_items.to(self.device)            # (B, L)
                batch_item_feats = batch_item_feats.to(self.device)  # (B, L, item_feat_dim)
                batch_user_feats = batch_user_feats.to(self.device)  # (B, user_feat_dim)
                batch_targets = batch_targets.to(self.device)        # (B,)

                self.optimizer.zero_grad()
                logits = self.model(batch_items, batch_item_feats, batch_user_feats)  # (B, num_items)
                loss = self.criterion(logits, batch_targets)
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
        For each user in `users`, build:
          • their full history of (item_ids, item_side_features),
          • pad/truncate to max_seq_len,
          • fetch that user’s static features → user_feat_vec,
          • run the model to get logits (B, num_items),
          • mask out seen items if requested,
          • take top‐k, and return as a Spark DataFrame.
        """
        if user_features is None or item_features is None:
            raise ValueError("Must pass both user_features and item_features to predict().")

        # 1. Convert log → pandas, and build per‐user sequences of shifted IDs
        pandas_log = log.toPandas()
        if "timestamp" in pandas_log.columns:
            pandas_log = pandas_log.sort_values(by="timestamp")
        else:
            pandas_log = pandas_log.sort_index()
        pandas_log["item_shifted"] = pandas_log["item_idx"].astype(int) + self.item_offset

        current_user_to_seq = {}
        for uid, group in pandas_log.groupby("user_idx"):
            current_user_to_seq[int(uid)] = group["item_shifted"].tolist()

        # 2. Rebuild (or update) user_features_dict in case new users appeared
        users_pd = user_features.toPandas()
        if "segment" in users_pd.columns:
            users_pd = pd.get_dummies(users_pd, columns=["segment"], prefix="segment")
        user_feat_cols = [c for c in users_pd.columns if c != "user_idx"]
        if len(user_feat_cols) > 0:
            user_matrix = users_pd[user_feat_cols].values.astype(float)
            scaled_user = self.user_feat_scaler.transform(user_matrix)
            self.user_features_dict = {int(uid): scaled_user[i]
                                       for i, uid in enumerate(users_pd["user_idx"].astype(int))}

        # 3. Rebuild (or update) item_features_dict in case new items appeared
        items_pd = item_features.toPandas()
        if "category" in items_pd.columns:
            items_pd = pd.get_dummies(items_pd, columns=["category"], prefix="category")
        item_feat_cols = [c for c in items_pd.columns if c != "item_idx"]
        if len(item_feat_cols) > 0:
            item_matrix = items_pd[item_feat_cols].values.astype(float)
            scaled_item = self.item_feat_scaler.transform(item_matrix)
            self.item_features_dict = {int(iid): scaled_item[i]
                                       for i, iid in enumerate(items_pd["item_idx"].astype(int))}

        # 4. Build input sequences for each user in `users`
        users_pd = users.toPandas()
        user_ids = users_pd["user_idx"].astype(int).tolist()
        L = self.max_seq_len

        batch_item_ids = []
        batch_item_feats = []
        batch_user_feats = []

        for uid in user_ids:
            seq = current_user_to_seq.get(uid, [])
            T = len(seq)

            # a) Build item ID sequence (prefix) = [0] + seq[:-1]
            if T == 0:
                in_ids = [0] * L
            else:
                prefix = [0] + seq[:-1]
                if len(prefix) >= L:
                    in_ids = prefix[-L:]
                else:
                    pad_len = L - len(prefix)
                    in_ids = [0] * pad_len + prefix

            # b) Build item_feat sequence aligned to in_ids
            feat_seq = []
            for sid in in_ids:
                if sid == 0:
                    feat_seq.append(np.zeros(self.item_feat_dim, dtype=float))
                else:
                    orig_id = sid - self.item_offset
                    feat_seq.append(self.item_features_dict.get(orig_id, np.zeros(self.item_feat_dim, dtype=float)))
            feat_seq = np.stack(feat_seq, axis=0)  # (L, item_feat_dim)

            # c) Static user feature
            user_feat_vec = self.user_features_dict.get(uid, np.zeros(self.user_feat_dim, dtype=float))

            batch_item_ids.append(np.array(in_ids, dtype=np.int64))
            batch_item_feats.append(feat_seq.astype(np.float32))
            batch_user_feats.append(user_feat_vec.astype(np.float32))

        if len(user_ids) == 0:
            empty_pd = pd.DataFrame(columns=["user_idx", "item_idx", "relevance"])
            return pandas_to_spark(empty_pd)

        items_tensor = torch.LongTensor(batch_item_ids).to(self.device)            # (U, L)
        item_feats_tensor = torch.FloatTensor(batch_item_feats).to(self.device)    # (U, L, item_feat_dim)
        user_feats_tensor = torch.FloatTensor(batch_user_feats).to(self.device)    # (U, user_feat_dim)

        # 5. Forward pass
        with torch.no_grad():
            logits = self.model(items_tensor, item_feats_tensor, user_feats_tensor)  # (U, num_items)
            if filter_seen_items:
                mask = torch.zeros_like(logits, dtype=torch.bool)  # (U, num_items)
                for i, uid in enumerate(user_ids):
                    seen = set(current_user_to_seq.get(uid, []))
                    for s in seen:
                        if 0 <= s < self.num_items:
                            mask[i, s] = True
                logits = logits.masked_fill(mask, float("-inf"))

            topk_vals, topk_idx = torch.topk(logits, k, dim=1)  # both: (U, k)

        # 6. Build pandas DataFrame of recs
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
class LSTMRecommender:
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

class TransformerRecommender:
    """
    Transformer‐based sequential recommender (SASRec‐style) that ingests:
      • item indices
      • category indices
      • price (continuous) per interaction
      • static user features (e.g. one‐hot segment or other numeric user attributes)
    Trains with teacher forcing (next‐item at every position). At inference, uses final time step’s logits.
    """

    def __init__(
        self,
        seed: int = 42,
        max_seq_len: int = 50,
        emb_dim: int = 128,
        n_heads: int = 4,
        ff_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        category_emb_dim: int = 16,
        price_hidden_dim: int = 16,
        batch_size: int = 64,
        num_epochs: int = 5,
        lr: float = 1e-3,
        device: str = None
    ):
        """
        Args:
          seed: random seed
          max_seq_len: maximum user‐history length
          emb_dim: embedding & Transformer hidden size
          n_heads: number of attention heads (2‐8)
          ff_dim: feed‐forward dimension in each block (≥ emb_dim)
          n_layers: number of Transformer encoder layers
          dropout: dropout probability inside Transformer
          category_emb_dim: dimension for category embedding before projecting to emb_dim
          price_hidden_dim: MLP hidden size for price → emb_dim
          batch_size, num_epochs, lr: training hyperparams
          device: "cuda" or "cpu" (auto‐detect if None)
        """
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.max_seq_len = max_seq_len
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.category_emb_dim = category_emb_dim
        self.price_hidden_dim = price_hidden_dim

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr

        # Device selection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Will be set in fit()
        self.model = None
        self.optimizer = None
        self.criterion = None

        self.num_items = None       # total distinct items + 1 for PAD
        self.num_categories = None  # total distinct categories + 1 for PAD
        self.item_offset = 1        # shift original item_idx → [1..] so 0 = PAD

        # Per‐user sequences
        self.user_item_seq = {}     # user_idx → [shifted_item_ids]
        self.user_cat_seq = {}      # user_idx → [shifted_cat_ids]
        self.user_price_seq = {}    # user_idx → [scaled_prices]

        # Static user features
        self.user_features_dict = {}   # user_idx → 1D numpy array (scaled)
        self.user_feat_dim = 0
        self.user_feat_scaler = StandardScaler()

        # Price scaler
        self.price_scaler = StandardScaler()

        # Category → index map
        self.category_to_idx = {}

    def fit(self, log, user_features=None, item_features=None):
        """
        Train the SASRecWithFeatures model.

        1. Convert `log` to pandas and sort by timestamp if present.
        2. Convert `user_features` to pandas, one‐hot any categorical user columns (e.g. 'segment'),
           then scale all numeric user features. Build `self.user_features_dict`.
        3. Convert `item_features` to pandas, build category→index map, scale price,
           then merge with `log` to attach category & price per interaction.
        4. Determine `self.num_items`, `self.num_categories`, shift item & category IDs by +1.
        5. For each user, collect their full time‐ordered sequence of (item_shifted, cat_shifted, price_scaled).
        6. Pad/truncate each user’s sequence to exactly `L = self.max_seq_len`. Build:
             - item_inputs:  (num_users, L)
             - cat_inputs:   (num_users, L)
             - price_inputs: (num_users, L)
             - targets:      (num_users, L)   (the next‐item index at each position, 0 for PAD)
        7. Create a PyTorch Dataset that yields (item_seq, cat_seq, price_seq, user_feat_vector, target_seq).
        8. Instantiate `SASRecWithFeatures(num_items, num_categories, …)` on `self.device`.
        9. Train for `num_epochs` with CrossEntropyLoss(ignore_index=0) using teacher forcing.
        """
        # --------------------------
        # 1) Pull `log` into pandas
        # --------------------------
        pandas_log = log.toPandas()
        if "timestamp" in pandas_log.columns:
            pandas_log = pandas_log.sort_values(by="timestamp")
        else:
            pandas_log = pandas_log.sort_index()

        # --------------------------
        # 2) Handle `user_features`
        # --------------------------
        if user_features is None:
            raise ValueError("`user_features` must be provided to fit().")
        users_pd = user_features.toPandas()

        # If there’s a “segment” column or any categorical, one‐hot encode it.
        if "segment" in users_pd.columns:
            users_pd = pd.get_dummies(users_pd, columns=["segment"], prefix="segment")

        # Collect all user‐feature columns except user_idx
        user_feat_cols = [c for c in users_pd.columns if c != "user_idx"]
        if not user_feat_cols:
            raise ValueError("After one‐hot encoding, no user feature columns remain.")

        # Build numeric matrix for scaler
        user_matrix = users_pd[user_feat_cols].values.astype(float)  # shape: (n_users, F)
        self.user_feat_scaler.fit(user_matrix)
        scaled_user_feats = self.user_feat_scaler.transform(user_matrix)  # (n_users, F)
        self.user_feat_dim = scaled_user_feats.shape[1]

        # Map user_idx → scaled feature vector
        self.user_features_dict = {
            int(uid): scaled_user_feats[i]
            for i, uid in enumerate(users_pd["user_idx"].astype(int))
        }

        # --------------------------
        # 3) Handle `item_features`
        # --------------------------
        if item_features is None:
            raise ValueError("`item_features` must be provided to fit().")
        items_pd = item_features.toPandas()

        if "price" not in items_pd.columns:
            raise ValueError("`item_features` must contain a 'price' column.")
        if "category" not in items_pd.columns:
            raise ValueError("`item_features` must contain a 'category' column.")

        # Keep raw for merging
        raw_item_df = items_pd[["item_idx", "category", "price"]].copy()

        # Build category→index mapping (reserve 0 for PAD)
        unique_cats = items_pd["category"].astype(str).unique().tolist()
        self.category_to_idx = {cat: idx + 1 for idx, cat in enumerate(unique_cats)}
        self.num_categories = len(self.category_to_idx) + 1  # +1 for PAD

        # Fit price scaler on all item prices
        items_pd["price"] = items_pd["price"].astype(float)
        all_prices = items_pd[["price"]].values.reshape(-1, 1)
        self.price_scaler.fit(all_prices)

        # --------------------------
        # 4) Merge `log` with raw_item_df
        # --------------------------
        # Determine num_items before shift
        max_orig_item = pandas_log["item_idx"].max()
        self.num_items = int(max_orig_item) + 2  # +1 shift, +1 PAD

        # Shift item_idx → [1..] so 0=PAD
        pandas_log["item_shifted"] = pandas_log["item_idx"].astype(int) + self.item_offset

        # Merge to attach category & price to each interaction
        merged = pandas_log.merge(raw_item_df, on="item_idx", how="left")

        # Shift category → [1..], PAD=0 if missing
        merged["category_shifted"] = (
            merged["category"].astype(str)
                  .map(self.category_to_idx)
                  .fillna(0)
                  .astype(int)
        )

        # Scale price to float
        merged["price"] = merged["price"].astype(float)
        merged["price_scaled"] = self.price_scaler.transform(merged[["price"]])  # (N,1) → (N,)

        # Ensure item_shifted is correct
        merged["item_shifted"] = merged["item_idx"].astype(int) + self.item_offset

        # --------------------------
        # 5) Build per‐user sequences
        # --------------------------
        self.user_item_seq.clear()
        self.user_cat_seq.clear()
        self.user_price_seq.clear()

        for uid, grp in merged.groupby("user_idx"):
            uid_i = int(uid)
            self.user_item_seq[uid_i]  = grp["item_shifted"].tolist()
            self.user_cat_seq[uid_i]   = grp["category_shifted"].tolist()
            self.user_price_seq[uid_i] = grp["price_scaled"].tolist()

        # --------------------------
        # 6) Pad/truncate to (num_users, L)
        # --------------------------
        user_ids = list(self.user_item_seq.keys())
        num_users = len(user_ids)
        L = self.max_seq_len

        # Initialize arrays
        item_inputs  = np.zeros((num_users, L), dtype=np.int64)
        cat_inputs   = np.zeros((num_users, L), dtype=np.int64)
        price_inputs = np.zeros((num_users, L), dtype=np.float32)
        targets      = np.zeros((num_users, L), dtype=np.int64)

        for idx, uid in enumerate(user_ids):
            seq_items  = self.user_item_seq[uid]
            seq_cats   = self.user_cat_seq[uid]
            seq_prices = self.user_price_seq[uid]
            T = len(seq_items)

            # Build the “teacher forcing” input & target
            # input = [0] + seq[:-1], target = seq
            input_items  = [0] + seq_items[:-1]
            input_cats   = [0] + seq_cats[:-1]
            input_prices = [0.0] + seq_prices[:-1]
            target_items = seq_items[:]

            if T >= L:
                item_inputs[idx]  = np.array(input_items[-L:], dtype=np.int64)
                cat_inputs[idx]   = np.array(input_cats[-L:], dtype=np.int64)
                price_inputs[idx] = np.array(input_prices[-L:], dtype=np.float32)
                targets[idx]      = np.array(target_items[-L:], dtype=np.int64)
            else:
                pad_len = L - T
                item_inputs[idx]  = np.array([0]*pad_len + input_items, dtype=np.int64)
                cat_inputs[idx]   = np.array([0]*pad_len + input_cats, dtype=np.int64)
                price_inputs[idx] = np.array([0.0]*pad_len + input_prices, dtype=np.float32)
                targets[idx]      = np.array([0]*pad_len + target_items, dtype=np.int64)

        # --------------------------
        # 7) Build PyTorch Dataset
        # --------------------------
        class SeqDataset(Dataset):
            def __init__(self, item_seq, cat_seq, price_seq, user_feats, target_seq):
                # item_seq, cat_seq: np.ndarray(num_users, L) of ints
                # price_seq: np.ndarray(num_users, L) of floats
                # user_feats: np.ndarray(num_users, F) of floats
                # target_seq: np.ndarray(num_users, L) of ints
                self.items = torch.from_numpy(item_seq).long()     # (num_users, L)
                self.cats  = torch.from_numpy(cat_seq).long()      # (num_users, L)
                self.prices = torch.from_numpy(price_seq).float()  # (num_users, L)
                self.user_feats = torch.from_numpy(user_feats).float()  # (num_users, F)
                self.targets = torch.from_numpy(target_seq).long() # (num_users, L)

            def __len__(self):
                return self.items.size(0)

            def __getitem__(self, idx):
                return (
                    self.items[idx],      # (L,)
                    self.cats[idx],       # (L,)
                    self.prices[idx],     # (L,)
                    self.user_feats[idx], # (F,)
                    self.targets[idx]     # (L,)
                )

        # Build the (num_users, F) user‐feature array in the same order
        user_feats_array = np.vstack([
            self.user_features_dict.get(uid, np.zeros(self.user_feat_dim,))
            for uid in user_ids
        ])  # shape: (num_users, F)

        dataset = SeqDataset(item_inputs, cat_inputs, price_inputs, user_feats_array, targets)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=False)

        # --------------------------
        # 8) Instantiate model, optimizer, loss
        # --------------------------
        self.model = TransformerRecModel(
            num_items=self.num_items,
            num_categories=self.num_categories,
            max_seq_len=self.max_seq_len,
            emb_dim=self.emb_dim,
            n_heads=self.n_heads,
            ff_dim=self.ff_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
            padding_idx=0,
            category_emb_dim=self.category_emb_dim,
            price_hidden_dim=self.price_hidden_dim,
            user_feat_dim=self.user_feat_dim
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        # --------------------------
        # 9) Training loop (teacher forcing)
        # --------------------------
        self.model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for batch_items, batch_cats, batch_prices, batch_user_feats, batch_targets in dataloader:
                # Move to device
                batch_items      = batch_items.to(self.device)      # (B, L)
                batch_cats       = batch_cats.to(self.device)       # (B, L)
                batch_prices     = batch_prices.to(self.device)     # (B, L)
                batch_user_feats = batch_user_feats.to(self.device) # (B, F)
                batch_targets    = batch_targets.to(self.device)    # (B, L)

                self.optimizer.zero_grad()
                # → logits (B, L, num_items)
                logits = self.model(batch_items, batch_cats, batch_prices, batch_user_feats)

                B, L, V = logits.size()
                logits_flat  = logits.view(-1, V)       # (B*L, V)
                targets_flat = batch_targets.view(-1)   # (B*L,)

                loss = self.criterion(logits_flat, targets_flat)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * B

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
        Generate top‐k next‐item recommendations. Steps:

          1. Convert `log` → pandas, rebuild per‐user sequences of
             (item_shifted, category_shifted, price_scaled), sorted by timestamp.
          2. Convert `user_features` → pandas, one‐hot + scale, refill `self.user_features_dict`.
          3. For each user in `users`:
               a. Take their sequence (item, cat, price). Build “input” = [0] + seq[:-1],
                  pad/truncate it to length L.
               b. Fetch that user’s static feature vector → (F,).
          4. Build four tensors of shape (U, L): item_seq, cat_seq, price_seq plus (U, F) user_feats.
          5. Forward through model → logits_all (U, L, num_items). Take final time‐step logits: (U, num_items).
          6. If filter_seen_items=True, mask out any already‐seen item indices by replacing with −∞.
          7. Take topk from each row of final_logits. Build a pandas DataFrame [user_idx, item_idx, relevance].
          8. Convert that DataFrame to Spark via `pandas_to_spark`, cast user_idx and item_idx to int.
        """
        # --------------------------
        # 1) Rebuild per‐user sequences from `log`
        # --------------------------
        pandas_log = log.toPandas()
        if "timestamp" in pandas_log.columns:
            pandas_log = pandas_log.sort_values(by="timestamp")
        else:
            pandas_log = pandas_log.sort_index()

        pandas_log["item_shifted"] = pandas_log["item_idx"].astype(int) + self.item_offset

        # Merge with raw item_features for category & price
        if item_features is None:
            raise ValueError("`item_features` must be provided at predict().")
        items_pd = item_features.toPandas()
        raw_item_df = items_pd[["item_idx", "category", "price"]].copy()

        merged = pandas_log.merge(raw_item_df, on="item_idx", how="left")
        merged["category_shifted"] = (
            merged["category"].astype(str)
                  .map(self.category_to_idx)
                  .fillna(0)
                  .astype(int)
        )
        merged["price"] = merged["price"].astype(float)
        merged["price_scaled"] = self.price_scaler.transform(merged[["price"]])
        merged["item_shifted"] = merged["item_idx"].astype(int) + self.item_offset

        current_user_item  = {}
        current_user_cat   = {}
        current_user_price = {}
        for uid, grp in merged.groupby("user_idx"):
            current_user_item[int(uid)]  = grp["item_shifted"].tolist()
            current_user_cat[int(uid)]   = grp["category_shifted"].tolist()
            current_user_price[int(uid)] = grp["price_scaled"].tolist()

        # --------------------------
        # 2) Re‐pull & rescale `user_features`
        # --------------------------
        if user_features is None:
            raise ValueError("`user_features` must be provided at predict().")
        users_pd_all = user_features.toPandas()
        if "segment" in users_pd_all.columns:
            users_pd_all = pd.get_dummies(users_pd_all, columns=["segment"], prefix="segment")
        user_feat_cols = [c for c in users_pd_all.columns if c != "user_idx"]
        if user_feat_cols:
            user_matrix_all = users_pd_all[user_feat_cols].values.astype(float)
            scaled_user_feats_all = self.user_feat_scaler.transform(user_matrix_all)
            self.user_features_dict = {
                int(uid): scaled_user_feats_all[i]
                for i, uid in enumerate(users_pd_all["user_idx"].astype(int))
            }

        # --------------------------
        # 3) Build prediction inputs for “users”
        # --------------------------
        users_pd = users.toPandas()
        user_ids = users_pd["user_idx"].astype(int).tolist()
        U = len(user_ids)
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
                # entirely PAD
                batch_user_items.append([0]*L)
                batch_user_cats.append([0]*L)
                batch_user_prices.append([0.0]*L)
            else:
                # teacher forcing input: [0] + seq[:-1]
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

            # static user feature
            feat_vec = self.user_features_dict.get(uid, np.zeros(self.user_feat_dim,))
            batch_user_feats.append(feat_vec)

        if U == 0:
            empty_pd = pd.DataFrame(columns=["user_idx", "item_idx", "relevance"])
            return pandas_to_spark(empty_pd)

        # Convert to tensors
        items_tensor      = torch.LongTensor(batch_user_items).to(self.device)   # (U, L)
        cats_tensor       = torch.LongTensor(batch_user_cats).to(self.device)    # (U, L)
        prices_tensor     = torch.FloatTensor(batch_user_prices).to(self.device) # (U, L)
        user_feats_tensor = torch.FloatTensor(np.stack(batch_user_feats, axis=0)).to(self.device)  # (U, F)

        # --------------------------
        # 4) Forward pass & top‐k
        # --------------------------
        with torch.no_grad():
            logits_all = self.model(items_tensor, cats_tensor, prices_tensor, user_feats_tensor)
            # logits_all: (U, L, num_items). We only care about the final position → next‐item logits
            final_logits = logits_all[:, -1, :]  # (U, num_items)

            if filter_seen_items:
                mask = torch.zeros_like(final_logits, dtype=torch.bool)  # (U, num_items)
                for i, uid in enumerate(user_ids):
                    seen = set(current_user_item.get(uid, []))
                    for s in seen:
                        if 0 <= s < self.num_items:
                            mask[i, s] = True
                final_logits = final_logits.masked_fill(mask, float("-inf"))

            topk_vals, topk_idx = torch.topk(final_logits, k, dim=1)  # (U, k)

        # --------------------------
        # 5) Build pandas DataFrame of recommendations
        # --------------------------
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
