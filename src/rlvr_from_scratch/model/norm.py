"""Normalization layers built from scratch.

Implements RMSNorm and LayerNorm with shape annotations at every step.
No torch.nn.LayerNorm, no F.rms_norm.

Reference: "RMSNorm vs LayerNorm — When and Why"
    https://www.vitorsousa.com/bits/rmsnorm-vs-layernorm
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


# =========================================================================
# RMSNorm — Zhang & Sennrich (2019), arXiv:1910.07467
# =========================================================================


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Normalizes inputs by their root-mean-square, then applies a learned
    per-feature scale. Drops the mean-subtraction and learned shift of
    LayerNorm — the hypothesis is that re-scaling, not re-centering,
    is what matters for training stability.

        RMS(x) = sqrt(mean(x^2))
        RMSNorm(x) = gamma * x / (RMS(x) + eps)

    Epsilon convention: eps lives INSIDE the sqrt, matching Llama /
    HuggingFace / F.rms_norm. Changing this silently breaks weight
    loading from external checkpoints.

    Args:
        d_model: Feature dimension to normalize over (the last dim).
        eps:     Numerical stability constant. Default 1e-6.
    """

    def __init__(self, d_model: int, *, eps: float = 1e-6) -> None:
        super().__init__()
        if d_model <= 0:
            msg = f"d_model must be positive, got {d_model}"
            raise ValueError(msg)

        self.d_model = d_model
        self.eps = eps

        # =========================================
        # Learned scale (no shift)
        # =========================================
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: (..., d_model)

        Returns:
            (..., d_model)
        """
        # =========================================
        # 1. Mean of squares along the feature dim
        # =========================================
        # (..., d_model) -> (..., 1)
        mean_sq = x.pow(2).mean(dim=-1, keepdim=True)

        # =========================================
        # 2. Inverse RMS — eps INSIDE the sqrt
        # =========================================
        # (..., 1)
        rms_inv = torch.rsqrt(mean_sq + self.eps)

        # =========================================
        # 3. Normalize, then scale
        # =========================================
        # (..., d_model) * (..., 1) -> (..., d_model)
        x_hat = x * rms_inv

        # (..., d_model) * (d_model,) -> (..., d_model)
        return self.gamma * x_hat


# =========================================================================
# LayerNorm — Ba, Kiros, Hinton (2016), arXiv:1607.06450
# =========================================================================


class LayerNorm(nn.Module):
    """Layer Normalization.

    Normalizes inputs to zero mean and unit variance along the feature
    dimension, then applies a learned per-feature scale and shift.

        LN(x) = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta

    Kept here as a teaching baseline for RMSNorm. The rlvr model uses
    RMSNorm throughout — this class is not imported by the transformer.

    Args:
        d_model: Feature dimension to normalize over (the last dim).
        eps:     Numerical stability constant. Default 1e-5.
    """

    def __init__(self, d_model: int, *, eps: float = 1e-5) -> None:
        super().__init__()
        if d_model <= 0:
            msg = f"d_model must be positive, got {d_model}"
            raise ValueError(msg)

        self.d_model = d_model
        self.eps = eps

        # =========================================
        # Learned scale and shift
        # =========================================
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: (..., d_model)

        Returns:
            (..., d_model)
        """
        # =========================================
        # 1. Mean and variance along the feature dim
        # =========================================
        # (..., d_model) -> (..., 1)
        mean = x.mean(dim=-1, keepdim=True)
        # Biased variance (divide by d, not d-1) — matches torch.nn.LayerNorm
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # =========================================
        # 2. Normalize to zero mean, unit variance
        # =========================================
        # (..., d_model)
        x_hat = (x - mean) * torch.rsqrt(var + self.eps)

        # =========================================
        # 3. Scale and shift
        # =========================================
        # (..., d_model) * (d_model,) + (d_model,) -> (..., d_model)
        return self.gamma * x_hat + self.beta
