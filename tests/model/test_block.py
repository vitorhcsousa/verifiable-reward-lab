"""Tests for the transformer block."""

from __future__ import annotations

import pytest
import torch

from rlvr_from_scratch.model.attention import causal_mask
from rlvr_from_scratch.model.block import TransformerBlock
from rlvr_from_scratch.model.ffn import GeluFFN, SwiGLU
from rlvr_from_scratch.model.norm import LayerNorm, RMSNorm
from rlvr_from_scratch.model.positional import RotaryPositionalEmbedding

B, T, D_MODEL, N_HEADS, D_FF = 2, 6, 16, 4, 64


def _block(**kw) -> TransformerBlock:
    return TransformerBlock(D_MODEL, N_HEADS, D_FF, **kw)


# =========================================================================
# Shape & defaults
# =========================================================================


def test_output_shape_preserved() -> None:
    """The block is shape-preserving: (B, T, d_model) -> (B, T, d_model)."""
    block = _block()
    x = torch.randn(B, T, D_MODEL)
    out, cache = block(x)
    assert out.shape == (B, T, D_MODEL)
    assert cache is None  # no cache passed in


def test_default_is_modern() -> None:
    """Defaults should be pre-norm + RMSNorm + SwiGLU."""
    block = _block()
    assert block.pre_norm is True
    assert isinstance(block.norm1, RMSNorm)
    assert isinstance(block.ffn, SwiGLU)


def test_residual_changes_input() -> None:
    """Output is not identical to input (sublayers actually contribute)."""
    block = _block()
    x = torch.randn(B, T, D_MODEL)
    out, _ = block(x)
    assert not torch.allclose(out, x)


# =========================================================================
# Pre-norm vs post-norm
# =========================================================================


@pytest.mark.parametrize("pre_norm", [True, False])
def test_both_norm_orders_run(pre_norm: bool) -> None:
    block = _block(pre_norm=pre_norm)
    out, _ = block(torch.randn(B, T, D_MODEL))
    assert out.shape == (B, T, D_MODEL)
    assert torch.isfinite(out).all()


def test_pre_and_post_norm_differ() -> None:
    """Same weights, same input, different residual order -> different output."""
    torch.manual_seed(0)
    pre = _block(pre_norm=True)
    torch.manual_seed(0)
    post = _block(pre_norm=False)
    x = torch.randn(B, T, D_MODEL)
    assert not torch.allclose(pre(x)[0], post(x)[0])


# =========================================================================
# Parametrizable norm / FFN (the classical contrast)
# =========================================================================


def test_classical_configuration() -> None:
    """Post-norm + LayerNorm + GeluFFN recovers the classical block."""
    block = TransformerBlock(
        D_MODEL, N_HEADS, D_FF, pre_norm=False, norm_cls=LayerNorm, ffn_cls=GeluFFN
    )
    assert isinstance(block.norm1, LayerNorm)
    assert isinstance(block.ffn, GeluFFN)
    out, _ = block(torch.randn(B, T, D_MODEL))
    assert out.shape == (B, T, D_MODEL)


# =========================================================================
# Masking & KV-cache
# =========================================================================


def test_accepts_causal_mask() -> None:
    block = _block()
    x = torch.randn(B, T, D_MODEL)
    out, _ = block(x, mask=causal_mask(T))
    assert out.shape == (B, T, D_MODEL)


def test_kv_cache_passthrough_shape() -> None:
    """A single decode step with an empty cache returns a grown (K, V)."""
    block = _block()
    d_k = D_MODEL // N_HEADS
    empty = (
        torch.zeros(B, N_HEADS, 0, d_k),
        torch.zeros(B, N_HEADS, 0, d_k),
    )
    x_step = torch.randn(B, 1, D_MODEL)
    out, new_cache = block(x_step, kv_cache=empty)
    assert out.shape == (B, 1, D_MODEL)
    assert new_cache is not None
    k, v = new_cache
    assert k.shape == (B, N_HEADS, 1, d_k)
    assert v.shape == (B, N_HEADS, 1, d_k)


# =========================================================================
# Gradients
# =========================================================================


def test_gradients_flow() -> None:
    block = _block()
    x = torch.randn(B, T, D_MODEL, requires_grad=True)
    block(x)[0].sum().backward()
    assert x.grad is not None
    assert block.attn.W_Q.weight.grad is not None
    assert block.ffn.W_gate.weight.grad is not None


# =========================================================================
# RoPE injection
# =========================================================================


def test_rope_is_injected_into_attention() -> None:
    rope = RotaryPositionalEmbedding(D_MODEL // N_HEADS, max_len=64)
    block = _block(rope=rope)
    assert block.attn.rope is rope


def test_block_runs_with_rope() -> None:
    rope = RotaryPositionalEmbedding(D_MODEL // N_HEADS, max_len=64)
    block = _block(rope=rope)
    out, _ = block(torch.randn(B, T, D_MODEL), mask=causal_mask(T))
    assert out.shape == (B, T, D_MODEL)
    assert torch.isfinite(out).all()
