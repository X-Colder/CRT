"""
Path 2: Causal Constrained Attention.

Modifies the standard multi-head attention to incorporate causal structure
as an inductive bias. The causal graph acts as a prior on which tokens
should attend to which, breaking spurious correlations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class CausalConstrainedAttention(nn.Module):
    """
    Multi-head attention where the attention mask is modulated by a
    causal adjacency matrix. Tokens corresponding to causally unrelated
    variables receive dampened attention scores.

    causal_mask[i][j] = 1.0 if j is a causal ancestor/descendant of i,
                        alpha (e.g. 0.1) otherwise.
    """

    def __init__(self, d_model: int, num_heads: int, causal_weight: float = 0.8):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.causal_weight = causal_weight

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        causal_adj: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            causal_adj: (seq_len, seq_len) soft adjacency matrix, values in [0,1].
                        If None, falls back to standard attention.
            mask: (batch, seq_len) boolean mask for padding.
        """
        B, L, _ = x.shape

        Q = self.W_q(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if causal_adj is not None:
            # Build causal bias: strongly attend along causal edges,
            # weakly attend elsewhere.
            causal_bias = self._build_causal_bias(causal_adj, L, x.device)
            scores = scores + causal_bias.unsqueeze(0).unsqueeze(0)

        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.W_o(out)

    def _build_causal_bias(self, adj: torch.Tensor, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Convert adjacency matrix to attention bias.
        Causal connections get bonus, non-causal get penalty.
        """
        # Expand adjacency to include indirect causation (transitive closure).
        # A_transitive = (I + A)^n > 0 gives reachability.
        n = adj.shape[0]
        if n != seq_len:
            # Pad or truncate adjacency to match sequence length.
            bias = torch.zeros(seq_len, seq_len, device=device)
            m = min(n, seq_len)
            bias[:m, :m] = adj[:m, :m]
            adj = bias

        reachable = torch.eye(seq_len, device=device) + adj
        reachable = torch.matrix_power((reachable > 0).float(), min(seq_len, 8))
        reachable = (reachable > 0).float()

        # Bias: log(weight) for causal, log(1-weight) for non-causal.
        w = self.causal_weight
        bias = reachable * math.log(w + 1e-8) + (1 - reachable) * math.log(1 - w + 1e-8)
        return bias


class CausalTransformerBlock(nn.Module):
    """A single transformer block with causal-constrained attention."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, causal_weight: float = 0.8):
        super().__init__()
        self.attn = CausalConstrainedAttention(d_model, num_heads, causal_weight)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), causal_adj))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x
