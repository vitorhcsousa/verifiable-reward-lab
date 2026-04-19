"""Positional encoding built from scratch.

Implements sinusoidal, learned, rotary (RoPE), and ALiBi position
encodings with shape annotations at every step. No torch.nn helpers
beyond Embedding.

Reference: "Positional Encoding: Teaching Transformers to Count"
    https://www.vitorsousa.com/foundations/positional-encoding
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


# =========================================================================
# Sinusoidal Positional Encoding
# =========================================================================


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding from "Attention Is All You Need".

    PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
    PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))

    No learnable parameters — the encoding is deterministic and added
    to token embeddings before the first attention layer.

    Args:
        d_model: Model dimension. Must be even (sin/cos are paired).
        max_len: Maximum sequence length to precompute.
    """

    pe: Tensor  # registered buffer — annotated for type checker

    def __init__(self, d_model: int, max_len: int = 8192) -> None:
        super().__init__()
        if d_model % 2 != 0:
            msg = f"d_model ({d_model}) must be even for sin/cos pairs"
            raise ValueError(msg)

        self.d_model = d_model
        self.max_len = max_len

        # =========================================
        # 1. Position index and frequency vector
        # =========================================
        position = torch.arange(max_len).unsqueeze(1).float()  # (max_len, 1)
        # div_term = 1 / 10000^(2i / d_model), computed in log-space
        # for numerical stability at large i.
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model / 2,)

        # =========================================
        # 2. Build encoding matrix: (max_len, d_model)
        # =========================================
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims

        # =========================================
        # 3. Register as buffer (not a parameter, but saved with model)
        # =========================================
        # Shape: (1, max_len, d_model) for broadcasting over batch.
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to input embeddings.

        Args:
            x: (B, T, d_model)

        Returns:
            output: (B, T, d_model) — x + positional encoding.
        """
        T = x.size(1)
        if T > self.max_len:
            msg = f"sequence length {T} exceeds max_len ({self.max_len})"
            raise ValueError(msg)

        # self.pe: (1, max_len, d_model) -> slice to (1, T, d_model)
        return x + self.pe[:, :T, :]


# =========================================================================
# Learned Positional Embedding
# =========================================================================


class LearnedPositionalEmbedding(nn.Module):
    """Learned positional embedding — a lookup table.

    Each position in 0..max_len-1 gets its own learned vector. Simple
    and effective within the trained range, but cannot extrapolate:
    position max_len has no embedding.

    Used by GPT-2 and BERT.

    Args:
        d_model: Model dimension.
        max_len: Maximum sequence length (rows in the embedding table).
    """

    def __init__(self, d_model: int, max_len: int = 8192) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Add learned positional embeddings to input.

        Args:
            x: (B, T, d_model)

        Returns:
            output: (B, T, d_model) — x + position embeddings.
        """
        T = x.size(1)
        if T > self.max_len:
            msg = f"sequence length {T} exceeds max_len ({self.max_len})"
            raise ValueError(msg)

        # =========================================
        # Look up position vectors and broadcast over batch
        # =========================================
        positions = torch.arange(T, device=x.device)  # (T,)
        pos_emb = self.embedding(positions)  # (T, d_model)
        return x + pos_emb  # broadcast: (B, T, d_model)


# =========================================================================
# Rotary Position Embedding (RoPE)
# =========================================================================


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Applies a position-dependent rotation to Q and K (not V). Relative
    position emerges from the dot-product algebra:

        RoPE(q, m)^T @ RoPE(k, n) = q^T @ R(n - m) @ k

    The attention score between positions m and n depends only on
    n - m, not on the absolute positions. Used by LLaMA, Qwen, Mistral,
    and most modern open-weight LLMs.

    Note: operates on the per-head dimension d_k, so it is applied
    after splitting heads in MultiHeadAttention.

    Args:
        d_k: Per-head dimension. Must be even (rotations are 2D pairs).
        max_len: Maximum sequence length to precompute.
        base: Base for frequency computation. 10000 in the original paper.
    """

    inv_freq: Tensor  # registered buffers — annotated for type checker
    cos_cached: Tensor
    sin_cached: Tensor

    def __init__(
        self,
        d_k: int,
        max_len: int = 8192,
        *,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        if d_k % 2 != 0:
            msg = f"d_k ({d_k}) must be even for RoPE pairs"
            raise ValueError(msg)

        self.d_k = d_k
        self.max_len = max_len
        self.base = base

        # =========================================
        # 1. Inverse frequencies: θ_i = base^(-2i / d_k) for i in 0..d_k/2-1
        # =========================================
        inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer("inv_freq", inv_freq)  # (d_k / 2,)

        # =========================================
        # 2. Precompute cos/sin cache for all positions
        # =========================================
        self._build_cache(max_len)

    def _build_cache(self, max_len: int) -> None:
        """Precompute cos/sin values for positions 0..max_len-1."""
        positions = torch.arange(max_len).float()  # (max_len,)

        # Outer product: (max_len,) x (d_k / 2,) -> (max_len, d_k / 2)
        freqs = torch.outer(positions, self.inv_freq)

        # Duplicate to (max_len, d_k) — each frequency is used for both
        # halves of the vector so that _rotate_half works correctly.
        freqs = torch.cat([freqs, freqs], dim=-1)

        # Shape: (1, 1, max_len, d_k) — broadcastable over (B, H).
        self.register_buffer("cos_cached", freqs.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_cached", freqs.sin().unsqueeze(0).unsqueeze(0))

    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        offset: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """Apply rotary embedding to Q and K.

        Args:
            Q: (B, H, T, d_k)
            K: (B, H, T, d_k)
            offset: Starting absolute position (for KV-cache incremental
                    decoding). During the first pass use 0; at step t of
                    incremental decoding use the current cache length.

        Returns:
            Q_rotated: (B, H, T, d_k)
            K_rotated: (B, H, T, d_k)
        """
        T = Q.size(2)
        if offset + T > self.max_len:
            msg = f"offset + T ({offset + T}) exceeds cached max_len ({self.max_len})"
            raise ValueError(msg)

        # =========================================
        # 1. Slice precomputed cos/sin for the current positions
        # =========================================
        # (1, 1, T, d_k) — broadcastable over (B, H).
        cos = self.cos_cached[:, :, offset : offset + T, :]
        sin = self.sin_cached[:, :, offset : offset + T, :]

        # =========================================
        # 2. Apply rotation: q' = q * cos + rotate_half(q) * sin
        # =========================================
        Q_rotated = (Q * cos) + (self._rotate_half(Q) * sin)
        K_rotated = (K * cos) + (self._rotate_half(K) * sin)

        return Q_rotated, K_rotated

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        """Swap-and-negate halves: [x1, x2] -> [-x2, x1].

        Implements the rotation identity: for each 2D pair (x_{2i}, x_{2i+1}),
        a rotation by angle θ maps (a, b) -> (a*cos - b*sin, a*sin + b*cos).
        Here we pair (first half, second half) instead of (even, odd) —
        mathematically equivalent and efficient as a single concat.
        """
        d_half = x.shape[-1] // 2
        x1 = x[..., :d_half]
        x2 = x[..., d_half:]
        return torch.cat([-x2, x1], dim=-1)


# =========================================================================
# ALiBi: Attention with Linear Biases
# =========================================================================


class ALiBi(nn.Module):
    """Attention with Linear Biases (ALiBi).

    Does not encode position in embeddings. Instead, adds a linear
    penalty to attention scores based on token distance:

        scores[i, j] += -m_h * |i - j|

    Each head gets a head-specific slope m_h, geometrically spaced:
        m_h = 1 / 2^(8h / H)   for h = 1, ..., H

    Steep slopes create local-focused heads; gentle slopes create
    long-range heads. Extrapolates to sequences longer than training.

    Used by BLOOM and MPT.

    Args:
        n_heads: Number of attention heads.
        max_len: Maximum sequence length to precompute.
    """

    bias: Tensor  # registered buffer — annotated for type checker

    def __init__(self, n_heads: int, max_len: int = 8192) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.max_len = max_len

        # =========================================
        # 1. Head-specific slopes: m_h = 1 / 2^(8h / H)
        # =========================================
        slopes = torch.tensor(
            [1.0 / (2 ** (8 * h / n_heads)) for h in range(1, n_heads + 1)]
        )  # (H,)

        # =========================================
        # 2. Pairwise distance matrix: |i - j|
        # =========================================
        positions = torch.arange(max_len)
        distance = (
            (positions.unsqueeze(0) - positions.unsqueeze(1)).abs().float()
        )  # (max_len, max_len)

        # =========================================
        # 3. Bias: negative penalty per head
        # =========================================
        # (H, 1, 1) * (1, max_len, max_len) -> (H, max_len, max_len)
        bias = -slopes.view(-1, 1, 1) * distance.unsqueeze(0)

        # Shape: (1, H, max_len, max_len) — broadcastable over batch.
        self.register_buffer("bias", bias.unsqueeze(0))

    def forward(self, T: int) -> Tensor:
        """Return the ALiBi bias for sequence length T.

        Add the returned tensor to attention scores *before* softmax,
        alongside any causal mask.

        Args:
            T: Current sequence length.

        Returns:
            bias: (1, H, T, T) — additive, non-positive.
        """
        if T > self.max_len:
            msg = f"sequence length {T} exceeds max_len ({self.max_len})"
            raise ValueError(msg)
        return self.bias[:, :, :T, :T]
