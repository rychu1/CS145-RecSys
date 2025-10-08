# model.py
import torch

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pyspark.sql.functions import col
from sklearn.preprocessing import StandardScaler
from sim4rec.utils import pandas_to_spark  # f
import pandas as pd
import numpy as np
class LSTMRecModel(nn.Module):
    # ... (init method is unchanged) ...
    def __init__(self, num_items: int, item_emb_dim: int, num_categories: int, category_emb_dim: int, price_hidden_dim: int, hidden_dim: int, num_layers: int, dropout: float, padding_idx: int, user_feat_dim: int):
        super().__init__()
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.item_embedding = nn.Embedding(num_items, item_emb_dim, padding_idx=padding_idx)
        self.cat_embedding = nn.Embedding(num_categories, category_emb_dim, padding_idx=padding_idx)
        self.price_proj = nn.Sequential(nn.Linear(1, price_hidden_dim), nn.ReLU())
        self.input_dim = item_emb_dim + category_emb_dim + price_hidden_dim
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.user_feat_proj = nn.Linear(user_feat_dim, num_layers * hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_items)


    # In your model.py file, inside the LSTMRecModel class

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
        logits: (B, L, num_items) – scores over next‐item vocabulary for each position.
        """
        B, L = item_seq.size()

        # a) Embed items → (B, L, item_emb_dim)
        item_emb = self.item_embedding(item_seq)

        # b) Embed categories → (B, L, category_emb_dim)
        cat_emb = self.cat_embedding(cat_seq)

        # c) Project price: (B, L, 1) → (B, L, price_hidden_dim)
        price_in = price_seq.unsqueeze(-1)
        price_emb = self.price_proj(price_in)

        # d) Concatenate along the last dimension → (B, L, input_dim)
        x = torch.cat([item_emb, cat_emb, price_emb], dim=-1)

        # e) Build initial hidden state h0 from user features
        h0 = self.user_feat_proj(user_feat)
        h0 = h0.view(B, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        c0 = torch.zeros(self.num_layers, B, self.hidden_dim, device=x.device)

        # f) Run LSTM: x → out: (B, L, hidden_dim)
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # --- CORRECTED PART ---
        # Project all hidden states to get logits for each position in the sequence.
        # Use .reshape() which handles non-contiguous memory gracefully.
        logits_flat = self.fc(out.reshape(-1, self.hidden_dim))
        
        # Reshape back to (B, L, num_items)
        logits = logits_flat.reshape(B, L, self.num_items)
        
        return logits
        
# recommender.py

class MyLSTMRecommender:
    """
    IMPROVED LSTM-based sequential recommender.
    - Learns from every valid prefix in a user's history (not just the last one).
    - Uses teacher-forcing across the whole sequence for efficient training.
    - Includes optional price-weighting for final recommendations.
    """

    def __init__(
    self,
    seed: int = 42,
    item_emb_dim: int = 256,            # Increased
    category_emb_dim: int = 32,             # Increased
    price_hidden_dim: int = 32,             # Increased
    hidden_dim: int = 256,              # Increased
    num_layers: int = 2,                # Increased
    dropout: float = 0.2,               # Slightly Increased for regularization
    max_seq_len: int = 50,
    batch_size: int = 128,              # Increased
    num_epochs: int = 20,               # Increased
    lr: float = 5e-4,                   # Slightly Decreased for stability
    price_weighting: bool = True,
    price_log_transform: bool = True,
    device: str = None
):
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Model Hyperparameters
        self.item_emb_dim = item_emb_dim
        self.category_emb_dim = category_emb_dim
        self.price_hidden_dim = price_hidden_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_seq_len = max_seq_len

        # Training Hyperparameters
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        
        # NEW: Price Weighting attributes
        self.price_weighting = price_weighting
        self.price_log_transform = price_log_transform
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Placeholders set in fit()
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.num_items = None
        self.num_categories = None
        self.item_offset = 1
        self.cat_offset = 1
        self.user_item_seq, self.user_cat_seq, self.user_price_seq = {}, {}, {}
        self.user_features_dict = {}
        self.price_scaler = StandardScaler()
        self.user_feat_scaler = StandardScaler()
        self.category_to_idx = {}
        self.user_feat_dim = None

    def fit(self, log, user_features=None, item_features=None):
        """
        Train the LSTM with a sliding window approach for efficient learning.
        """
        # --- Steps 1-5: Data preparation (largely the same) ---
        # 1. Pull log → pandas, sort by timestamp
        pandas_log = log.toPandas()
        if "timestamp" in pandas_log.columns:
            pandas_log = pandas_log.sort_values(by="timestamp")
        else:
            pandas_log = pandas_log.sort_index()
        users_pd = user_features.toPandas()
        items_pd = item_features.toPandas()

        # Handle user features
        if "segment" in users_pd.columns:
            users_pd = pd.get_dummies(users_pd, columns=["segment"], prefix="segment")
        user_feat_cols = [c for c in users_pd.columns if c != "user_idx"]
        user_matrix = users_pd[user_feat_cols].values.astype(float)
        scaled_user_feats = self.user_feat_scaler.fit_transform(user_matrix)
        self.user_feat_dim = scaled_user_feats.shape[1]
        self.user_features_dict = {int(uid): scaled_user_feats[i] for i, uid in enumerate(users_pd["user_idx"])}

        # Handle item features
        raw_item_df = items_pd[["item_idx", "category", "price"]].copy()
        all_prices = raw_item_df[["price"]].values.astype(float).reshape(-1, 1)
        self.price_scaler.fit(all_prices)
        unique_categories = raw_item_df["category"].astype(str).unique()
        self.category_to_idx = {cat: idx + self.cat_offset for idx, cat in enumerate(unique_categories)}
        self.num_categories = len(self.category_to_idx) + 1
        self.num_items = int(pandas_log["item_idx"].max()) + 2

        # Merge features into log
        merged = pandas_log.merge(raw_item_df, on="item_idx", how="left")
        merged["category_shifted"] = merged["category"].astype(str).map(self.category_to_idx).fillna(0).astype(int)
        merged["price_scaled"] = self.price_scaler.transform(merged[["price"]])
        merged["item_shifted"] = merged["item_idx"].astype(int) + self.item_offset

        # Build per-user sequences
        for uid, group in merged.groupby("user_idx"):
            self.user_item_seq[int(uid)] = group["item_shifted"].tolist()
            self.user_cat_seq[int(uid)] = group["category_shifted"].tolist()
            self.user_price_seq[int(uid)] = group["price_scaled"].tolist()

        # --- Step 6: IMPROVED Training Data Generation (Sliding Window) ---
        print("Generating training examples with sliding window...")
        inputs_item, inputs_cat, inputs_price = [], [], []
        inputs_user_feat, targets = [], []
        L = self.max_seq_len

        for uid, item_seq in self.user_item_seq.items():
            cat_seq = self.user_cat_seq[uid]
            price_seq = self.user_price_seq[uid]
            user_feat_vec = self.user_features_dict.get(uid, np.zeros(self.user_feat_dim))

            # Create examples from every valid prefix of the sequence
            for i in range(1, len(item_seq)):
                prefix_items = item_seq[:i]
                prefix_cats = cat_seq[:i]
                prefix_prices = price_seq[:i]
                target_item = item_seq[i]

                # Pad/truncate prefix to length L
                padded_items = np.zeros(L, dtype=np.int64)
                padded_cats = np.zeros(L, dtype=np.int64)
                padded_prices = np.zeros(L, dtype=np.float32)

                if len(prefix_items) >= L:
                    padded_items[:] = prefix_items[-L:]
                    padded_cats[:] = prefix_cats[-L:]
                    padded_prices[:] = prefix_prices[-L:]
                else:
                    pad_len = L - len(prefix_items)
                    padded_items[pad_len:] = prefix_items
                    padded_cats[pad_len:] = prefix_cats
                    padded_prices[pad_len:] = prefix_prices

                inputs_item.append(padded_items)
                inputs_cat.append(padded_cats)
                inputs_price.append(padded_prices)
                inputs_user_feat.append(user_feat_vec)
                targets.append(target_item)
        
        # --- Step 7: Create Dataset and DataLoader ---
        class SeqDataset(Dataset):
            def __init__(self, items, cats, prices, user_feats, targets):
                self.items = torch.from_numpy(np.array(items)).long()
                self.cats = torch.from_numpy(np.array(cats)).long()
                self.prices = torch.from_numpy(np.array(prices)).float()
                self.user_feats = torch.from_numpy(np.array(user_feats)).float()
                self.targets = torch.from_numpy(np.array(targets)).long()
            def __len__(self): return len(self.items)
            def __getitem__(self, idx):
                return self.items[idx], self.cats[idx], self.prices[idx], self.user_feats[idx], self.targets[idx]

        dataset = SeqDataset(inputs_item, inputs_cat, inputs_price, inputs_user_feat, targets)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # --- Step 8: Instantiate Model and Optimizer ---
        self.model = LSTMRecModel(
            num_items=self.num_items, item_emb_dim=self.item_emb_dim, num_categories=self.num_categories,
            category_emb_dim=self.category_emb_dim, price_hidden_dim=self.price_hidden_dim,
            hidden_dim=self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout,
            padding_idx=0, user_feat_dim=self.user_feat_dim
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0) # Ignore PAD targets

        # --- Step 9: IMPROVED Training Loop (Teacher Forcing on final step of each prefix) ---
        print("Starting training...")
        self.model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for batch_items, batch_cats, batch_prices, batch_user_feats, batch_targets in dataloader:
                # Move to device
                batch_items, batch_cats, batch_prices, batch_user_feats, batch_targets = \
                    batch_items.to(self.device), batch_cats.to(self.device), batch_prices.to(self.device), \
                    batch_user_feats.to(self.device), batch_targets.to(self.device)

                self.optimizer.zero_grad()
                
                # Logits shape: (B, L, num_items)
                logits = self.model(batch_items, batch_cats, batch_prices, batch_user_feats)
                
                # We only care about the prediction at the final non-padded time step.
                # Since each example in the batch can have a different length, we use the targets.
                # However, our sliding window approach makes each sequence's target a single item.
                # We take the prediction at the last time-step.
                final_logits = logits[:, -1, :] # (B, num_items)

                loss = self.criterion(final_logits, batch_targets)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * batch_items.size(0)

            avg_loss = epoch_loss / len(dataset)
            print(f"[Epoch {epoch+1}/{self.num_epochs}] training loss = {avg_loss:.4f}")
        self.model.eval()
        return self # Return self after fitting

    def predict(self, log, k: int, users, items, user_features=None, item_features=None, filter_seen_items: bool = True):
        """
        Generate top-k next-item recommendations for a batch of users.
        """
        if self.model is None:
            raise RuntimeError("The model has not been fitted yet. Please call fit() first.")
            
        print("Starting prediction...")
        # 1. Prepare all necessary data and mappings
        pandas_log = log.toPandas()
        if "timestamp" in pandas_log.columns:
            pandas_log = pandas_log.sort_values(by="timestamp")
        else:
            pandas_log = pandas_log.sort_index()
        items_pd = item_features.toPandas()
        users_to_predict = users.toPandas()['user_idx'].tolist()

        # Rebuild current user sequences from the log
        merged = pandas_log.merge(items_pd[["item_idx", "category", "price"]], on="item_idx", how="left")
        merged["category_shifted"] = merged["category"].astype(str).map(self.category_to_idx).fillna(0).astype(int)
        merged["price_scaled"] = self.price_scaler.transform(merged[["price"]])
        merged["item_shifted"] = merged["item_idx"].astype(int) + self.item_offset
        
        current_user_sequences = {
            'item': {uid: g["item_shifted"].tolist() for uid, g in merged.groupby("user_idx")},
            'cat': {uid: g["category_shifted"].tolist() for uid, g in merged.groupby("user_idx")},
            'price': {uid: g["price_scaled"].tolist() for uid, g in merged.groupby("user_idx")}
        }

        # 2. Build input tensors for the users we need to predict for
        batch_items, batch_cats, batch_prices, batch_user_feats = [], [], [], []
        L = self.max_seq_len

        for uid in users_to_predict:
            item_seq = current_user_sequences['item'].get(uid, [])
            cat_seq = current_user_sequences['cat'].get(uid, [])
            price_seq = current_user_sequences['price'].get(uid, [])
            user_feat = self.user_features_dict.get(uid, np.zeros(self.user_feat_dim))

            padded_items = np.zeros(L, dtype=np.int64)
            padded_cats = np.zeros(L, dtype=np.int64)
            padded_prices = np.zeros(L, dtype=np.float32)

            if item_seq:
                if len(item_seq) >= L:
                    padded_items[:] = item_seq[-L:]
                    padded_cats[:] = cat_seq[-L:]
                    padded_prices[:] = price_seq[-L:]
                else:
                    padded_items[L-len(item_seq):] = item_seq
                    padded_cats[L-len(cat_seq):] = cat_seq
                    padded_prices[L-len(price_seq):] = price_seq
            
            batch_items.append(padded_items)
            batch_cats.append(padded_cats)
            batch_prices.append(padded_prices)
            batch_user_feats.append(user_feat)

        # 3. Run model inference
        with torch.no_grad():
            items_tensor = torch.from_numpy(np.array(batch_items)).long().to(self.device)
            cats_tensor = torch.from_numpy(np.array(batch_cats)).long().to(self.device)
            prices_tensor = torch.from_numpy(np.array(batch_prices)).float().to(self.device)
            user_feats_tensor = torch.from_numpy(np.array(batch_user_feats)).float().to(self.device)

            logits = self.model(items_tensor, cats_tensor, prices_tensor, user_feats_tensor)
            final_logits = logits[:, -1, :] # Get predictions from the last time step

            if filter_seen_items:
                mask = torch.zeros_like(final_logits)
                for i, uid in enumerate(users_to_predict):
                    seen_shifted = set(current_user_sequences['item'].get(uid, []))
                    mask[i, list(seen_shifted)] = 1
                final_logits[mask.bool()] = -float('inf')

            top_scores, top_indices = torch.topk(final_logits, k, dim=1)

        # 4. Apply price weighting and format results
        price_dict = {}
        if self.price_weighting:
            price_dict = pd.Series(items_pd.price.values, index=items_pd.item_idx).to_dict()

        rec_rows = []
        for i, uid in enumerate(users_to_predict):
            for j in range(k):
                shifted_item_id = top_indices[i, j].item()
                model_score = top_scores[i, j].item()
                
                original_item_id = shifted_item_id - self.item_offset
                if original_item_id < 0: continue
                    
                relevance = model_score
                if self.price_weighting and price_dict:
                    price = price_dict.get(original_item_id, 1.0) # Default price is 1.0
                    price_multiplier = np.log1p(price) if self.price_log_transform else price
                    relevance *= price_multiplier
                
                rec_rows.append({
                    "user_idx": uid,
                    "item_idx": original_item_id,
                    "relevance": relevance
                })

        # 5. Convert to Spark DataFrame and return
        if not rec_rows:
            return pandas_to_spark(pd.DataFrame(columns=["user_idx", "item_idx", "relevance"]))

        recs_pd = pd.DataFrame(rec_rows)
        return pandas_to_spark(recs_pd)