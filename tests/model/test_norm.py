"""Tests for normalization layers.

Covers:
- Shape preservation across (d,), (B, d), (B, T, d), (B, H, T, d)
- Numerical correctness (RMS ≈ 1; mean ≈ 0, std ≈ 1)
- Invariants (scale-invariance for both; shift-invariance for LayerNorm only)
- Equivalence with torch.nn.LayerNorm and F.rms_norm
- Gradient flow
- Edge cases (zero input, large input, invalid args)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from rlvr_from_scratch.model.norm import RMSNorm, LayerNorm

# =========================================================================
# Constants and fixtures
# =========================================================================

D_MODEL = 64
SHAPES = [
    (D_MODEL,),
    (8, D_MODEL),
    (4, 16, D_MODEL),
    (2, 8, 16, D_MODEL),
]


@pytest.fixture(autouse=True)
def set_seed() -> None:
    torch.manual_seed(0)


# =========================================================================
# Shape preservation
# =========================================================================


@pytest.mark.parametrize("shape", SHAPES)
def test_rmsnorm_preserves_shape(shape: tuple[int, ...]) -> None:
    norm = RMSNorm(D_MODEL)
    x = torch.randn(shape)
    assert norm(x).shape == x.shape


@pytest.mark.parametrize("shape", SHAPES)
def test_layernorm_preserves_shape(shape: tuple[int, ...]) -> None:
    norm = LayerNorm(D_MODEL)
    x = torch.randn(shape)
    assert norm(x).shape == x.shape


def test_rmsnorm_preserves_dtype() -> None:
    norm = RMSNorm(D_MODEL)
    x = torch.randn(4, 16, D_MODEL, dtype=torch.float32)
    assert norm(x).dtype == torch.float32


def test_layernorm_preserves_dtype() -> None:
    norm = LayerNorm(D_MODEL)
    x = torch.randn(4, 16, D_MODEL, dtype=torch.float32)
    assert norm(x).dtype == torch.float32


# =========================================================================
# Numerical correctness
# =========================================================================


def test_rmsnorm_output_has_unit_rms() -> None:
    """With gamma=1, RMS of output along the last dim should be ≈ 1."""
    norm = RMSNorm(D_MODEL, eps=1e-12)
    x = torch.randn(4, 16, D_MODEL)
    out = norm(x)

    # RMS = sqrt(mean(x^2))
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


def test_layernorm_output_has_zero_mean_and_unit_std() -> None:
    """With gamma=1, beta=0, output has mean ≈ 0 and std ≈ 1 along the last dim."""
    norm = LayerNorm(D_MODEL, eps=1e-12)
    x = torch.randn(4, 16, D_MODEL)
    out = norm(x)

    mean = out.mean(dim=-1)
    # Biased std (divide by d), matching the biased variance inside LayerNorm
    std = out.var(dim=-1, unbiased=False).sqrt()

    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-5)


# =========================================================================
# Invariants
# =========================================================================


def test_rmsnorm_is_scale_invariant() -> None:
    """RMSNorm(alpha * x) ≈ RMSNorm(x) for positive alpha."""
    norm = RMSNorm(D_MODEL, eps=1e-12)
    x = torch.randn(4, 16, D_MODEL)
    alpha = 7.3

    assert torch.allclose(norm(x), norm(alpha * x), atol=1e-4)


def test_rmsnorm_is_not_shift_invariant() -> None:
    """RMSNorm does NOT subtract the mean — shifting x changes the output.

    Asserting the negative to guard against a copy-paste bug where someone
    adds mean-subtraction inside RMSNorm.
    """
    norm = RMSNorm(D_MODEL)
    x = torch.randn(4, 16, D_MODEL)
    c = 2.5

    assert not torch.allclose(norm(x), norm(x + c), atol=1e-3)


def test_layernorm_is_scale_invariant() -> None:
    """LayerNorm(alpha * x) ≈ LayerNorm(x) for positive alpha."""
    norm = LayerNorm(D_MODEL, eps=1e-12)
    x = torch.randn(4, 16, D_MODEL)
    alpha = 7.3

    assert torch.allclose(norm(x), norm(alpha * x), atol=1e-4)


def test_layernorm_is_shift_invariant() -> None:
    """LayerNorm(x + c) ≈ LayerNorm(x) for scalar c."""
    norm = LayerNorm(D_MODEL, eps=1e-12)
    x = torch.randn(4, 16, D_MODEL)
    c = 2.5

    assert torch.allclose(norm(x), norm(x + c), atol=1e-4)


# =========================================================================
# Equivalence with PyTorch reference implementations
# =========================================================================


def test_rmsnorm_matches_functional_rms_norm() -> None:
    """Our RMSNorm should match torch.nn.functional.rms_norm.

    This is the test that catches the epsilon-placement bug — if eps
    goes outside the sqrt instead of inside, this fails.
    """
    eps = 1e-6
    norm = RMSNorm(D_MODEL, eps=eps)
    # Randomize gamma so the scale actually matters
    with torch.no_grad():
        norm.gamma.copy_(torch.randn(D_MODEL))

    x = torch.randn(4, 16, D_MODEL)
    ours = norm(x)
    reference = F.rms_norm(x, (D_MODEL,), weight=norm.gamma, eps=eps)
    assert torch.allclose(ours, reference, atol=1e-6)


def test_layernorm_matches_torch_layernorm() -> None:
    """Our LayerNorm should match torch.nn.LayerNorm when parameters match."""
    eps = 1e-5
    ours = LayerNorm(D_MODEL, eps=eps)
    reference = torch.nn.LayerNorm(D_MODEL, eps=eps)

    # Sync parameters
    with torch.no_grad():
        ours.gamma.copy_(reference.weight)
        ours.beta.copy_(reference.bias)

    x = torch.randn(4, 16, D_MODEL)
    assert torch.allclose(ours(x), reference(x), atol=1e-6)


# =========================================================================
# Gradient flow
# =========================================================================


def test_rmsnorm_gradients_are_finite() -> None:
    norm = RMSNorm(D_MODEL)
    x = torch.randn(4, 16, D_MODEL, requires_grad=True)

    loss = norm(x).sum()
    loss.backward()

    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(norm.gamma.grad).all()
    assert (norm.gamma.grad != 0).any()


def test_layernorm_gradients_are_finite() -> None:
    norm = LayerNorm(D_MODEL)
    x = torch.randn(4, 16, D_MODEL, requires_grad=True)

    loss = norm(x).sum()
    loss.backward()

    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(norm.gamma.grad).all()
    assert torch.isfinite(norm.beta.grad).all()
    assert (norm.gamma.grad != 0).any()


# =========================================================================
# Edge cases
# =========================================================================


def test_rmsnorm_zero_input_is_finite() -> None:
    """All-zero input: RMS is 0, but eps saves us. Output should be zero."""
    norm = RMSNorm(D_MODEL)
    x = torch.zeros(4, 16, D_MODEL)
    out = norm(x)

    assert torch.isfinite(out).all()
    assert torch.allclose(out, torch.zeros_like(out))


def test_layernorm_zero_input_is_finite() -> None:
    norm = LayerNorm(D_MODEL)
    x = torch.zeros(4, 16, D_MODEL)
    out = norm(x)

    assert torch.isfinite(out).all()


def test_rmsnorm_large_input_stays_finite() -> None:
    """The whole point of normalization: input magnitude doesn't blow up the output."""
    norm = RMSNorm(D_MODEL)
    x = torch.randn(4, 16, D_MODEL) * 1e6
    out = norm(x)

    assert torch.isfinite(out).all()


def test_layernorm_large_input_stays_finite() -> None:
    norm = LayerNorm(D_MODEL)
    x = torch.randn(4, 16, D_MODEL) * 1e6
    out = norm(x)

    assert torch.isfinite(out).all()


def test_rmsnorm_rejects_invalid_d_model() -> None:
    with pytest.raises(ValueError, match="d_model must be positive"):
        RMSNorm(0)


def test_layernorm_rejects_invalid_d_model() -> None:
    with pytest.raises(ValueError, match="d_model must be positive"):
        LayerNorm(-1)
