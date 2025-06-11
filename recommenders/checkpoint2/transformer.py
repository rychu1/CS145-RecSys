# ===== model.py =====

import torch
import torch.nn as nn
import math

class TransformerRecModel(nn.Module):
    """
    SASRec‐style Transformer encoder extended to also ingest:
      • category embeddings,
      • price embeddings (via an MLP),
      • static user‐feature embeddings (added to each position).
    Input: 
      - item_seq:      LongTensor(B, L)   (indices in [0..num_items-1], 0=PAD)
      - cat_seq:       LongTensor(B, L)   (indices in [0..num_cats-1], 0=PAD)
      - price_seq:     FloatTensor(B, L)  (real‐valued, already scaled)
      - user_feats:    FloatTensor(B, F)  (static user‐features per user)
    Output:
      - logits: FloatTensor(B, L, num_items),
        unnormalized next‐item scores at each position.
    """
    def __init__(
        self,
        num_items: int,
        num_categories: int,
        max_seq_len: int,
        emb_dim: int = 128,
        n_heads: int = 4,
        ff_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        padding_idx: int = 0,
        category_emb_dim: int = 16,
        price_hidden_dim: int = 16,
        user_feat_dim: int = 0
    ):
        super().__init__()

        self.num_items = num_items
        self.num_categories = num_categories
        self.max_seq_len = max_seq_len
        self.emb_dim = emb_dim
        self.padding_idx = padding_idx

        # 1) Item embedding + Category embedding + Learned positional embedding
        self.item_embedding = nn.Embedding(num_items, emb_dim, padding_idx=padding_idx)
        self.category_embedding = nn.Embedding(num_categories, category_emb_dim, padding_idx=0)
        
        # Project category_emb_dim → emb_dim
        self.category_proj = nn.Linear(category_emb_dim, emb_dim, bias=False)

        # 2) Price: MLP from a single float → emb_dim
        self.price_mlp = nn.Sequential(
            nn.Linear(1, price_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(price_hidden_dim, emb_dim)
        )

        # 3) Static user‐feature projection (if user_feat_dim > 0)
        if user_feat_dim > 0:
            self.user_feat_proj = nn.Linear(user_feat_dim, emb_dim, bias=False)
        else:
            self.user_feat_proj = None

        # 4) Positional embeddings (learned)
        self.pos_embedding = nn.Embedding(max_seq_len, emb_dim)

        # 5) Dropout on the sum of embeddings
        self.emb_dropout = nn.Dropout(dropout)

        # 6) Stacked TransformerEncoder layers (SASRec style)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True  # so (B, L, emb_dim)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        # 7) Final linear to project hidden → num_items (for next‐item logits)
        self.fc = nn.Linear(emb_dim, num_items)

        # 8) LayerNorm on output of each transformer block (optional but common in SASRec)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(
        self,
        item_seq: torch.LongTensor,       # (B, L)
        cat_seq: torch.LongTensor,        # (B, L)
        price_seq: torch.FloatTensor,     # (B, L)
        user_feats: torch.FloatTensor      # (B, F)   or None if F=0
    ) -> torch.FloatTensor:
        B, L = item_seq.size()
        device = item_seq.device

        # 1) Item embeddings
        item_emb = self.item_embedding(item_seq)                 # (B, L, emb_dim)

        # 2) Category embeddings → project to emb_dim
        cat_emb_raw = self.category_embedding(cat_seq)           # (B, L, category_emb_dim)
        cat_emb = self.category_proj(cat_emb_raw)                # (B, L, emb_dim)

        # 3) Price embedding
        #    price_seq is (B, L). We unsqueeze to (B, L, 1), feed through MLP → (B, L, emb_dim)
        price_input = price_seq.unsqueeze(-1)                    # (B, L, 1)
        price_emb = self.price_mlp(price_input)                  # (B, L, emb_dim)

        # 4) Static user‐features → project to emb_dim and broadcast to all positions
        if self.user_feat_proj is not None:
            # user_feats: (B, F) → (B, emb_dim) → unsqueeze →
            # (B, 1, emb_dim) → expand to (B, L, emb_dim)
            user_emb = self.user_feat_proj(user_feats)           # (B, emb_dim)
            user_emb = user_emb.unsqueeze(1).expand(-1, L, -1)   # (B, L, emb_dim)
        else:
            user_emb = torch.zeros((B, L, self.emb_dim), device=device)

        # 5) Positional embeddings
        positions = torch.arange(L, device=device).unsqueeze(0)  # (1, L)
        pos_emb = self.pos_embedding(positions)                  # (1, L, emb_dim)
        pos_emb = pos_emb.expand(B, -1, -1)                      # (B, L, emb_dim)

        # 6) Sum all embeddings
        x = item_emb + cat_emb + price_emb + user_emb + pos_emb   # (B, L, emb_dim)
        x = self.emb_dropout(x)

        # 7) Build causal (subsequent) mask so that each position only attends to earlier positions.
        #    mask shape for Torch ≥1.10: (L, L), where mask[i,j] = True means “block j from i‐th query.”
        causal_mask = torch.triu(torch.ones((L, L), device=device), diagonal=1).bool()

        # 8) Build padding mask: True where the item_seq == padding_idx (0), so Transformer ignores them.
        key_padding_mask = (item_seq == self.padding_idx)        # (B, L)

        # 9) Feed into TransformerEncoder
        #    Since batch_first=True, input shape = (B, L, emb_dim).
        h = self.transformer_encoder(
            x,
            mask=causal_mask,               # (L, L)
            src_key_padding_mask=key_padding_mask  # (B, L)
        )  # → (B, L, emb_dim)

        h = self.norm(h)  # optional LayerNorm after last encoder block

        # 10) Project to next‐item logits
        logits = self.fc(h)  # (B, L, num_items)
        return logits
