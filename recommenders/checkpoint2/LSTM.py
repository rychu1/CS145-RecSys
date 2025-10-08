# model.py

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
