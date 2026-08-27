"""Feed-forward networks built from raw tensors.

Implements SwiGLU (Llama-style, gated) and a vanilla GELU FFN with
shape annotations at every step. No torch.nn helpers beyond Linear.

Reference: "GELU / SwiGLU from Scratch"
    https://www.vitorsousa.com/bits/gelu-swiglu
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =========================================================================
# SwiGLU — Shazeer (2020), arXiv:2002.05202
# =========================================================================


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network.

    Replaces the standard FFN's single hidden projection with a gated
    variant: an "up" projection produces values, a "gate" projection
    produces a SiLU-activated gate, the two are multiplied element-wise,
    and a "down" projection brings the result back to d_model.

        SwiGLU(x) = (SiLU(x W_gate) ⊙ (x W_up)) W_down

    where SiLU(x) = x * sigmoid(x), also called Swish-1. The gate gives
    the network a content-dependent way to scale each hidden unit, which
    Shazeer's experiments show outperforms vanilla FFNs at matched
    parameter count.

    Note: gate and up each project to d_ff, so SwiGLU carries three
    weight matrices versus the vanilla FFN's two. To match parameter
    count, d_ff is usually scaled by ~2/3 (Llama uses ~8/3 * d_model).

    Used by Llama, Qwen, Mistral, and most modern open-weight LLMs.

    Args:
        d_model: Model dimension (input and output).
        d_ff:    Hidden dimension.
        bias:    Whether linear layers use bias. Default False.
    """

    def __init__(self, d_model: int, d_ff: int, *, bias: bool = False) -> None:
        super().__init__()
        if d_model <= 0:
            msg = f"d_model must be positive, got {d_model}"
            raise ValueError(msg)
        if d_ff <= 0:
            msg = f"d_ff must be positive, got {d_ff}"
            raise ValueError(msg)

        self.d_model = d_model
        self.d_ff = d_ff

        # =========================================
        # Three learned projections: gate, up, down
        # =========================================
        self.W_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.W_up = nn.Linear(d_model, d_ff, bias=bias)
        self.W_down = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: (B, T, d_model)

        Returns:
            (B, T, d_model)
        """
        gate = F.silu(self.W_gate(x))  # (B, T, d_ff)
        up = self.W_up(x)  # (B, T, d_ff)
        hidden = gate * up  # (B, T, d_ff)
        return self.W_down(hidden)  # (B, T, d_model)


# =========================================================================
# GELU FFN — vanilla "Attention Is All You Need" feed-forward
# =========================================================================


class GeluFFN(nn.Module):
    """Vanilla feed-forward network with GELU activation.

    The classical transformer FFN: project up to d_ff, apply a
    pointwise non-linearity, project back down to d_model.

        FFN(x) = GELU(x W_1) W_2

    Two weight matrices, no gating. Kept as a teaching baseline and
    classical contrast to SwiGLU. The original paper used ReLU; GELU
    became the default in GPT/BERT-era models.

    Args:
        d_model: Model dimension (input and output).
        d_ff:    Hidden dimension.
        bias:    Whether linear layers use bias. Default False.
    """

    def __init__(self, d_model: int, d_ff: int, *, bias: bool = False) -> None:
        super().__init__()
        if d_model <= 0:
            msg = f"d_model must be positive, got {d_model}"
            raise ValueError(msg)
        if d_ff <= 0:
            msg = f"d_ff must be positive, got {d_ff}"
            raise ValueError(msg)

        self.d_model = d_model
        self.d_ff = d_ff

        # =========================================
        # Two learned projections: up, down
        # =========================================
        self.W_1 = nn.Linear(d_model, d_ff, bias=bias)
        self.W_2 = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: (B, T, d_model)

        Returns:
            (B, T, d_model)
        """
        hidden = F.gelu(self.W_1(x))  # (B, T, d_ff)
        return self.W_2(hidden)  # (B, T, d_model)
