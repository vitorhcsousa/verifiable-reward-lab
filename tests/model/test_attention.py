"""Tests for attention mechanisms."""

from __future__ import annotations


import pytest
import torch
from torch import Tensor

from rlvr_from_scratch.model.attention import (
    MultiHeadAttention,
    causal_mask,
    scaled_dot_product_attention,
)


# =========================================================================
# Scaled Dot-Product Attention
# =========================================================================


def test_sdpa_output_shape() -> None:
    """Output is (B, H, T_q, d_v), weights are (B, H, T_q, T_k)."""
    B, H, T_q, T_k, d_k, d_v = 2, 4, 5, 7, 16, 32
    Q = torch.randn(B, H, T_q, d_k)
    K = torch.randn(B, H, T_k, d_k)
    V = torch.randn(B, H, T_k, d_v)

    out, weights = scaled_dot_product_attention(Q, K, V)

    assert out.shape == (B, H, T_q, d_v)
    assert weights.shape == (B, H, T_q, T_k)


def test_sdpa_weights_are_probabilities() -> None:
    """Weights are non-negative and rows sum to 1 (softmax)."""
    torch.manual_seed(0)
    Q, K, V = (torch.randn(2, 4, 5, 8) for _ in range(3))

    _, weights = scaled_dot_product_attention(Q, K, V)

    assert (weights >= 0).all()
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4, 5), atol=1e-6)


def test_sdpa_matches_hand_computation() -> None:
    """Numerical parity with a hand-computed 1-dim example."""
    # One query, two keys/values, d_k = 1 (no scaling beyond sqrt(1) = 1).
    Q = torch.tensor([[[[1.0]]]])  # (1, 1, 1, 1)
    K = torch.tensor([[[[2.0], [3.0]]]])  # (1, 1, 2, 1)
    V = torch.tensor([[[[10.0], [20.0]]]])  # (1, 1, 2, 1)

    out, weights = scaled_dot_product_attention(Q, K, V)

    # scores = [1*2, 1*3] / sqrt(1) = [2, 3]
    expected_w = torch.softmax(torch.tensor([2.0, 3.0]), dim=-1)
    expected_out = (expected_w * torch.tensor([10.0, 20.0])).sum()

    assert torch.allclose(weights[0, 0, 0], expected_w)
    assert torch.allclose(out[0, 0, 0, 0], expected_out)


def test_sdpa_mask_blocks_specific_positions() -> None:
    """A -inf entry in the mask drives that weight to exactly 0."""
    T = 4
    Q = torch.randn(1, 1, T, 8)
    K = torch.randn(1, 1, T, 8)
    V = torch.randn(1, 1, T, 8)

    mask = torch.zeros(1, 1, T, T)
    mask[..., 0, 2] = float("-inf")  # block key 2 for query 0

    _, weights = scaled_dot_product_attention(Q, K, V, mask=mask)

    assert weights[0, 0, 0, 2].item() == 0.0
    # Other queries unaffected: their rows still sum to 1
    assert torch.allclose(weights[0, 0, 1:].sum(dim=-1), torch.ones(3))


def test_sdpa_causal_mask_blocks_future() -> None:
    """Under a causal mask, every weight above the diagonal is 0."""
    T = 6
    Q = torch.randn(1, 1, T, 8)
    K = torch.randn(1, 1, T, 8)
    V = torch.randn(1, 1, T, 8)

    _, weights = scaled_dot_product_attention(Q, K, V, mask=causal_mask(T))

    upper = torch.triu(torch.ones(T, T), diagonal=1).bool()
    assert (weights[0, 0][upper] == 0).all()


def test_sdpa_gradient_flows() -> None:
    """Gradients reach Q, K, and V."""
    Q = torch.randn(1, 1, 3, 4, requires_grad=True)
    K = torch.randn(1, 1, 3, 4, requires_grad=True)
    V = torch.randn(1, 1, 3, 4, requires_grad=True)

    out, _ = scaled_dot_product_attention(Q, K, V)
    out.sum().backward()

    assert Q.grad is not None and Q.grad.abs().sum() > 0
    assert K.grad is not None and K.grad.abs().sum() > 0
    assert V.grad is not None and V.grad.abs().sum() > 0


# =========================================================================
# Causal Mask
# =========================================================================


def test_causal_mask_shape() -> None:
    """Shape is (1, 1, T, T) — broadcastable over batch and heads."""
    assert causal_mask(7).shape == (1, 1, 7, 7)


def test_causal_mask_values() -> None:
    """Upper triangle (strictly above diagonal) is -inf; rest is 0."""
    T = 5
    mask = causal_mask(T)[0, 0]

    for i in range(T):
        for j in range(T):
            if j > i:
                assert mask[i, j].item() == float("-inf")
            else:
                assert mask[i, j].item() == 0.0


# =========================================================================
# Multi-Head Attention
# =========================================================================


def test_mha_rejects_misaligned_dimensions() -> None:
    """d_model must be divisible by n_heads."""
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(d_model=30, n_heads=4)


def test_mha_self_attention_shape() -> None:
    """Self-attention output matches input shape."""
    B, T, d_model, H = 2, 6, 32, 4
    mha = MultiHeadAttention(d_model, H)
    x = torch.randn(B, T, d_model)

    out, weights, cache = mha(x, x, x)

    assert out.shape == (B, T, d_model)
    assert weights.shape == (B, H, T, T)
    assert cache is None  # no cache input -> no cache output


def test_mha_cross_attention_shape() -> None:
    """Cross-attention handles different query/key lengths."""
    B, T_q, T_k, d_model, H = 2, 3, 7, 32, 4
    mha = MultiHeadAttention(d_model, H)
    q = torch.randn(B, T_q, d_model)
    kv = torch.randn(B, T_k, d_model)

    out, weights, _ = mha(q, kv, kv)

    assert out.shape == (B, T_q, d_model)
    assert weights.shape == (B, H, T_q, T_k)


def test_mha_gradient_flows() -> None:
    """Gradients propagate through all four projections."""
    mha = MultiHeadAttention(d_model=16, n_heads=4)
    x = torch.randn(1, 4, 16, requires_grad=True)

    out, _, _ = mha(x, x, x)
    out.sum().backward()

    assert x.grad is not None and x.grad.abs().sum() > 0
    for p in mha.parameters():
        assert p.grad is not None


def test_mha_kv_cache_matches_full_forward() -> None:
    """Incremental decoding with KV-cache equals a full causal forward."""
    torch.manual_seed(0)
    B, T, d_model, H = 1, 5, 16, 4
    d_k = d_model // H
    mha = MultiHeadAttention(d_model, H).eval()
    x = torch.randn(B, T, d_model)

    # Reference: full causal forward pass.
    with torch.no_grad():
        ref_out, _, _ = mha(x, x, x, mask=causal_mask(T))

    # Stepwise decoding: feed one token at a time with growing KV-cache.
    cache: tuple[Tensor, Tensor] = (
        torch.empty(B, H, 0, d_k),
        torch.empty(B, H, 0, d_k),
    )
    step_outputs: list[Tensor] = []
    with torch.no_grad():
        for t in range(T):
            xt = x[:, t : t + 1, :]
            # No mask needed: single query, all cached keys are past.
            out_t, _, cache = mha(xt, xt, xt, kv_cache=cache)
            step_outputs.append(out_t)

    stepped = torch.cat(step_outputs, dim=1)  # (B, T, d_model)

    assert torch.allclose(ref_out, stepped, atol=1e-6)


def test_mha_bias_flag_controls_projection_bias() -> None:
    """bias=True adds bias parameters; bias=False omits them."""
    mha_no_bias = MultiHeadAttention(16, 4, bias=False)
    mha_with_bias = MultiHeadAttention(16, 4, bias=True)

    assert mha_no_bias.W_Q.bias is None
    assert mha_with_bias.W_Q.bias is not None
