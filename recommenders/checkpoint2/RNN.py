# model.py

import torch
import torch.nn as nn

class RNNRecModel(nn.Module):
    """
    RNN (GRU) based next‐item predictor that incorporates:
      • item embeddings (for item IDs),
      • item‐side features (projected into the same embedding space),
      • static user features (projected into initial hidden state).
    Input:
      - item_seq: LongTensor of shape (B, L) containing item_shifted IDs (0=PAD).
      - item_feat_seq: FloatTensor of shape (B, L, item_feat_dim) containing each timestep’s side features.
      - user_feat: FloatTensor of shape (B, user_feat_dim) containing static user features.
    Output:
      - logits: FloatTensor of shape (B, num_items) – next‐item scores (only final timestep used).
    """

    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        padding_idx: int,
        item_feat_dim: int,
        user_feat_dim: int
    ):
        """
        Args:
            num_items: vocabulary size for item IDs (+1 for PAD=0).
            embedding_dim: dimension for item embedding & item_feat projection.
            hidden_dim: hidden dimension of GRU.
            num_layers: number of stacked GRU layers.
            dropout: dropout probability between GRU layers (only if num_layers > 1).
            padding_idx: index reserved for PAD in item embedding (=0).
            item_feat_dim: number of side‐features per item per timestep.
            user_feat_dim: dimensionality of static user features.
        """
        super().__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 1) Embed item IDs into embedding_dim
        self.item_embedding = nn.Embedding(
            num_items,
            embedding_dim,
            padding_idx=padding_idx
        )

        # 2) Project item side‐features from item_feat_dim → embedding_dim
        self.item_feat_proj = nn.Linear(item_feat_dim, embedding_dim)

        # 3) Project user features into initial hidden state: 
        #    We need one hidden vector per layer, so total size = num_layers * hidden_dim
        self.user_feat_proj = nn.Linear(user_feat_dim, num_layers * hidden_dim)

        # 4) GRU: input_size = embedding_dim (item_emb + item_feat_emb), hidden_size = hidden_dim
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # 5) Final projection: hidden_dim → num_items
        self.fc = nn.Linear(hidden_dim, num_items)

    def forward(
        self,
        item_seq: torch.LongTensor,
        item_feat_seq: torch.FloatTensor,
        user_feat: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Args:
            item_seq: (B, L) LongTensor of shifted item IDs (1..num_items-1, 0=PAD).
            item_feat_seq: (B, L, item_feat_dim) FloatTensor of side‐features.
            user_feat: (B, user_feat_dim) FloatTensor of static user features.
        Returns:
            logits: FloatTensor of shape (B, num_items) – next‐item scores (for final timestep).
        """
        B, L = item_seq.size()

        # a) Item embedding: (B, L) → (B, L, embedding_dim)
        emb = self.item_embedding(item_seq)

        # b) Item‐side features projection: (B, L, item_feat_dim) → (B, L, embedding_dim)
        feat_emb = self.item_feat_proj(item_feat_seq)

        # c) Combine them by simple elementwise sum:
        x = emb + feat_emb  # shape (B, L, embedding_dim)

        # d) Build initial hidden state from user features:
        #    user_feat: (B, user_feat_dim) → (B, num_layers*hidden_dim)
        h0 = self.user_feat_proj(user_feat)  # shape (B, num_layers*hidden_dim)
        #    reshape into (num_layers, B, hidden_dim)
        h0 = h0.view(B, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()

        # e) Run GRU:
        #    x: (B, L, embedding_dim), h0: (num_layers, B, hidden_dim)
        out, h_n = self.gru(x, h0)  # out: (B, L, hidden_dim), h_n: (num_layers, B, hidden_dim)

        # f) We only need the final timestep’s hidden vector from top layer:
        final_hidden = h_n[-1]  # shape (B, hidden_dim)

        # g) Project to logits over items:
        logits = self.fc(final_hidden)  # shape (B, num_items)
        return logits

