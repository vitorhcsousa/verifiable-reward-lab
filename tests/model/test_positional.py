"""Tests for positional encoding modules."""

from __future__ import annotations

import pytest
import torch

from rlvr_from_scratch.model.positional import (
    ALiBi,
    LearnedPositionalEmbedding,
    RotaryPositionalEmbedding,
    SinusoidalPositionalEncoding,
)


# =========================================================================
# Sinusoidal Positional Encoding
# =========================================================================


def test_sinusoidal_output_shape() -> None:
    """Output shape matches input shape."""
    B, T, d_model = 2, 10, 64
    pe = SinusoidalPositionalEncoding(d_model)
    x = torch.randn(B, T, d_model)

    out = pe(x)

    assert out.shape == x.shape


def test_sinusoidal_distinct_per_position() -> None:
    """Different positions must produce different encoding vectors."""
    pe = SinusoidalPositionalEncoding(d_model=32, max_len=16)
    zeros = torch.zeros(1, 16, 32)

    encoded = pe(zeros)[0]  # (16, 32) — purely the positional signal

    # No two positions should be identical.
    for i in range(16):
        for j in range(i + 1, 16):
            assert not torch.allclose(encoded[i], encoded[j], atol=1e-4)


def test_sinusoidal_has_no_learnable_parameters() -> None:
    """The encoding is fixed; no tensor should be a trainable parameter."""
    pe = SinusoidalPositionalEncoding(d_model=32)
    assert len(list(pe.parameters())) == 0


def test_sinusoidal_is_deterministic_across_instances() -> None:
    """Two freshly-built instances produce byte-identical encodings."""
    pe1 = SinusoidalPositionalEncoding(d_model=32, max_len=16)
    pe2 = SinusoidalPositionalEncoding(d_model=32, max_len=16)

    x = torch.zeros(1, 16, 32)
    assert torch.equal(pe1(x), pe2(x))


def test_sinusoidal_rejects_odd_d_model() -> None:
    """Sin/cos pairs require d_model to be even."""
    with pytest.raises(ValueError, match="even"):
        SinusoidalPositionalEncoding(d_model=63)


def test_sinusoidal_rejects_sequence_beyond_max_len() -> None:
    pe = SinusoidalPositionalEncoding(d_model=32, max_len=5)
    with pytest.raises(ValueError, match="exceeds max_len"):
        pe(torch.randn(1, 10, 32))


# =========================================================================
# Learned Positional Embedding
# =========================================================================


def test_learned_output_shape() -> None:
    B, T, d_model = 2, 10, 64
    pe = LearnedPositionalEmbedding(d_model)
    x = torch.randn(B, T, d_model)

    assert pe(x).shape == x.shape


def test_learned_gradient_flows() -> None:
    """Backprop reaches the embedding table."""
    pe = LearnedPositionalEmbedding(d_model=32, max_len=16)
    x = torch.randn(1, 8, 32)

    pe(x).sum().backward()

    assert pe.embedding.weight.grad is not None
    assert pe.embedding.weight.grad.abs().sum() > 0


def test_learned_rejects_sequence_beyond_max_len() -> None:
    pe = LearnedPositionalEmbedding(d_model=32, max_len=5)
    with pytest.raises(ValueError, match="exceeds max_len"):
        pe(torch.randn(1, 10, 32))


# =========================================================================
# Rotary Position Embedding (RoPE)
# =========================================================================


def test_rope_output_shapes() -> None:
    B, H, T, d_k = 2, 4, 10, 16
    rope = RotaryPositionalEmbedding(d_k)
    Q = torch.randn(B, H, T, d_k)
    K = torch.randn(B, H, T, d_k)

    Q_rot, K_rot = rope(Q, K)

    assert Q_rot.shape == Q.shape
    assert K_rot.shape == K.shape


def test_rope_rejects_odd_d_k() -> None:
    with pytest.raises(ValueError, match="even"):
        RotaryPositionalEmbedding(d_k=7)


def test_rope_preserves_norm() -> None:
    """Rotation is an isometry: ||R q|| == ||q||."""
    B, H, T, d_k = 2, 4, 10, 16
    rope = RotaryPositionalEmbedding(d_k)
    Q = torch.randn(B, H, T, d_k)

    Q_rot, _ = rope(Q, Q)

    assert torch.allclose(Q.norm(dim=-1), Q_rot.norm(dim=-1), atol=1e-5)


def test_rope_relative_position_property() -> None:
    """q_m^T k_n depends only on m - n, not on absolute positions.

    The hallmark property of RoPE: rotating q at m and k at n and taking
    their dot product equals q^T R(n-m) k. So pairs with equal relative
    offset must give equal dot products.
    """
    torch.manual_seed(0)
    d_k = 16
    rope = RotaryPositionalEmbedding(d_k, max_len=32)

    # Same q, k tiled across all positions so only position differs.
    q_base = torch.randn(1, 1, 1, d_k)
    k_base = torch.randn(1, 1, 1, d_k)
    Q = q_base.expand(1, 1, 20, d_k).contiguous()
    K = k_base.expand(1, 1, 20, d_k).contiguous()

    Q_rot, K_rot = rope(Q, K)

    # Pairs with the same relative offset (n - m = 5) must match.
    dot_at = lambda m, n: (Q_rot[0, 0, m] * K_rot[0, 0, n]).sum()
    assert torch.allclose(dot_at(0, 5), dot_at(3, 8), atol=1e-5)
    assert torch.allclose(dot_at(1, 6), dot_at(10, 15), atol=1e-5)

    # Different relative offsets should generally not match.
    assert not torch.allclose(dot_at(0, 3), dot_at(0, 5), atol=1e-3)


def test_rope_offset_matches_full_pass() -> None:
    """Rotating with offset=k equals rotating a full pass and slicing."""
    torch.manual_seed(0)
    d_k = 16
    rope = RotaryPositionalEmbedding(d_k, max_len=32)

    full = torch.randn(1, 1, 10, d_k)
    Q_full, _ = rope(full, full)

    # Rotate the same slice with offset = 4 and compare to the full slice.
    slice_ = full[:, :, 4:7, :]
    Q_sliced, _ = rope(slice_, slice_, offset=4)

    assert torch.allclose(Q_full[:, :, 4:7, :], Q_sliced, atol=1e-6)


def test_rope_rejects_offset_beyond_cache() -> None:
    rope = RotaryPositionalEmbedding(d_k=16, max_len=10)
    q = torch.randn(1, 1, 5, 16)
    with pytest.raises(ValueError, match="exceeds cached max_len"):
        rope(q, q, offset=8)  # 8 + 5 > 10


def test_rope_has_no_learnable_parameters() -> None:
    rope = RotaryPositionalEmbedding(d_k=16)
    assert len(list(rope.parameters())) == 0


# =========================================================================
# ALiBi
# =========================================================================


def test_alibi_output_shape() -> None:
    H, T = 8, 12
    alibi = ALiBi(n_heads=H)
    assert alibi(T).shape == (1, H, T, T)


def test_alibi_bias_is_non_positive() -> None:
    """ALiBi penalizes distance, so every entry is <= 0."""
    alibi = ALiBi(n_heads=8)
    assert (alibi(12) <= 0).all()


def test_alibi_diagonal_is_zero() -> None:
    """Distance |i - i| = 0 -> zero penalty for self-attention."""
    H, T = 8, 12
    alibi = ALiBi(n_heads=H)
    bias = alibi(T)

    for h in range(H):
        assert torch.allclose(torch.diagonal(bias[0, h]), torch.zeros(T))


def test_alibi_bias_is_symmetric_in_distance() -> None:
    """|i - j| = |j - i|, so bias[i, j] == bias[j, i]."""
    alibi = ALiBi(n_heads=4)
    bias = alibi(10)[0]  # (H, T, T)
    assert torch.allclose(bias, bias.transpose(-1, -2))


def test_alibi_slopes_are_geometric() -> None:
    """Slopes follow m_h = 1 / 2^(8h/H); extract them from bias[:, 0, 1]."""
    H = 8
    alibi = ALiBi(n_heads=H)
    bias = alibi(2)[0, :, 0, 1]  # (H,) — bias at distance 1 is -m_h

    slopes = -bias
    expected = torch.tensor([1.0 / (2 ** (8 * h / H)) for h in range(1, H + 1)])
    assert torch.allclose(slopes, expected, atol=1e-6)


def test_alibi_rejects_sequence_beyond_max_len() -> None:
    alibi = ALiBi(n_heads=4, max_len=5)
    with pytest.raises(ValueError, match="exceeds max_len"):
        alibi(10)