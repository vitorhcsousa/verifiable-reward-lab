"""Attention mechanisms built from raw tensors.

Implements scaled dot-product attention and multi-head attention
with shape annotations at every step. No torch.nn.MultiheadAttention.

Reference: "Attention Is All You Need to Implement"
    https://www.vitorsousa.com/foundations/attention-from-scratch
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from rlvr_from_scratch.model.positional import RotaryPositionalEmbedding


# =========================================================================
# Scaled Dot-Product Attention
# =========================================================================


def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Scaled dot-product attention.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        Q: Query tensor  (B, H, T_q, d_k)
        K: Key tensor    (B, H, T_k, d_k)
        V: Value tensor  (B, H, T_k, d_v)
        mask: Additive mask (B|1, 1|H, T_q, T_k)
              0.0 = allowed, -inf = blocked

    Returns:
        output:  (B, H, T_q, d_v)
        weights: (B, H, T_q, T_k)
    """
    d_k = Q.size(-1)

    # =========================================
    # 1. Score: how much does each query match each key?
    # =========================================
    # (B, H, T_q, d_k) @ (B, H, d_k, T_k) -> (B, H, T_q, T_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # =========================================
    # 2. Mask: block positions that shouldn't be attended to
    # =========================================
    if mask is not None:
        scores = scores + mask

    # =========================================
    # 3. Normalize: convert scores to probabilities
    # =========================================
    # (B, H, T_q, T_k) — each row sums to 1
    weights = F.softmax(scores, dim=-1)

    # =========================================
    # 4. Aggregate: weighted sum of values
    # =========================================
    # (B, H, T_q, T_k) @ (B, H, T_k, d_v) -> (B, H, T_q, d_v)
    output = torch.matmul(weights, V)

    return output, weights


# =========================================================================
# Causal Mask
# =========================================================================


def causal_mask(T: int, device: torch.device | None = None) -> Tensor:
    """Create additive causal mask.

    Convention: 0.0 = allowed, -inf = blocked.
    After softmax, e^(-inf) = 0 — blocked positions contribute nothing.

    Returns:
        mask: (1, 1, T, T) — broadcastable over batch and heads.
    """
    # Upper triangle (above diagonal) = True = future positions = blocked
    mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
    return mask.float().masked_fill(mask, float("-inf")).unsqueeze(0).unsqueeze(0)


# =========================================================================
# Multi-Head Attention
# =========================================================================


class MultiHeadAttention(nn.Module):
    """Multi-head attention with explicit projections.

    Splits d_model into H parallel heads of dimension d_k = d_model / H.
    Each head computes independent attention, outputs are concatenated
    and projected through W_O.

    Supports:
        - Self-attention: pass same tensor for query, key, value
        - Cross-attention: pass different tensors
        - KV-cache: for efficient incremental decoding

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads. Must divide d_model evenly.
        bias: Whether to use bias in projection layers.
        rope: Optional rotary position embedding, applied to Q and K per
              head before attention. Default None (no positional rotation).
              Assumes self-attention — Q and K share absolute positions.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        bias: bool = False,
        rope: RotaryPositionalEmbedding | None = None,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            msg = f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            raise ValueError(msg)

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.rope = rope

        # =========================================
        # Four learned projections
        # =========================================
        self.W_Q = nn.Linear(d_model, d_model, bias=bias)
        self.W_K = nn.Linear(d_model, d_model, bias=bias)
        self.W_V = nn.Linear(d_model, d_model, bias=bias)
        self.W_O = nn.Linear(d_model, d_model, bias=bias)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
        kv_cache: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor] | None]:
        """Forward pass.

        Args:
            query: (B, T_q, d_model)
            key:   (B, T_k, d_model)
            value: (B, T_k, d_model)
            mask:  Additive mask (B|1, 1|H, T_q, T_k). 0.0 = allowed, -inf = blocked.
            kv_cache: Optional cached (K, V) from previous steps,
                      each (B, H, T_prev, d_k).

        Returns:
            output:       (B, T_q, d_model)
            weights:      (B, H, T_q, T_k)
            new_kv_cache: Updated (K, V) cache, or None if no cache input.
        """
        B, T_q, _ = query.shape

        # =========================================
        # 1. Project
        # =========================================
        Q = self.W_Q(query)  # (B, T_q, d_model)
        K = self.W_K(key)  # (B, T_k, d_model)
        V = self.W_V(value)  # (B, T_k, d_model)

        # =========================================
        # 2. Split heads
        # =========================================
        Q = self._split_heads(Q)  # (B, H, T_q, d_k)
        K = self._split_heads(K)  # (B, H, T_k, d_k)
        V = self._split_heads(V)  # (B, H, T_k, d_k)

        # =========================================
        # 3. Rotary position embedding (RoPE), applied per-head
        # =========================================
        # RoPE rotates Q and the *current* K by their absolute positions.
        # With a KV-cache the new tokens begin at position = cached length,
        # so that becomes the offset; cached K were already rotated at their
        # own step. V is never rotated.
        if self.rope is not None:
            offset = kv_cache[0].size(2) if kv_cache is not None else 0
            Q, K = self.rope(Q, K, offset=offset)  # both (B, H, T, d_k)

        # =========================================
        # 4. KV-cache (for incremental decoding)
        # =========================================
        new_kv_cache: tuple[Tensor, Tensor] | None = None
        if kv_cache is not None:
            K_prev, V_prev = kv_cache
            # (B, H, T_prev, d_k) cat (B, H, T_k, d_k) -> (B, H, T_prev+T_k, d_k)
            K = torch.cat([K_prev, K], dim=2)
            V = torch.cat([V_prev, V], dim=2)
            new_kv_cache = (K, V)

        # =========================================
        # 5. Attention
        # =========================================
        attn_output, weights = scaled_dot_product_attention(Q, K, V, mask)

        # =========================================
        # 6. Merge heads + output projection
        # =========================================
        attn_output = self._merge_heads(attn_output)  # (B, T_q, d_model)
        output = self.W_O(attn_output)  # (B, T_q, d_model)

        return output, weights, new_kv_cache

    def _split_heads(self, x: Tensor) -> Tensor:
        """(B, T, d_model) -> (B, H, T, d_k)"""
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """(B, H, T, d_k) -> (B, T, d_model)"""
        B, _, T, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.d_model)
