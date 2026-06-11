"""Tests for RoPE wired into multi-head attention.

These are additive to the existing attention tests — they cover only
the rotary-embedding hook added to MultiHeadAttention.
"""

from __future__ import annotations

import torch

from rlvr_from_scratch.model.attention import MultiHeadAttention, causal_mask
from rlvr_from_scratch.model.positional import RotaryPositionalEmbedding

D_MODEL, N_HEADS, T = 16, 4, 5
D_K = D_MODEL // N_HEADS


def _rope() -> RotaryPositionalEmbedding:
    return RotaryPositionalEmbedding(D_K, max_len=64)


# =========================================================================
# Wiring & shape
# =========================================================================


def test_rope_is_optional_and_stored() -> None:
    assert MultiHeadAttention(D_MODEL, N_HEADS).rope is None
    rope = _rope()
    assert MultiHeadAttention(D_MODEL, N_HEADS, rope=rope).rope is rope


def test_rope_preserves_output_shape() -> None:
    mha = MultiHeadAttention(D_MODEL, N_HEADS, rope=_rope())
    x = torch.randn(2, T, D_MODEL)
    out, _, _ = mha(x, x, x, mask=causal_mask(T))
    assert out.shape == (2, T, D_MODEL)


def test_rope_changes_output() -> None:
    """Same weights, same input — RoPE must change the result."""
    torch.manual_seed(0)
    plain = MultiHeadAttention(D_MODEL, N_HEADS)
    torch.manual_seed(0)
    rotary = MultiHeadAttention(D_MODEL, N_HEADS, rope=_rope())
    x = torch.randn(2, T, D_MODEL)
    out_plain, _, _ = plain(x, x, x, mask=causal_mask(T))
    out_rope, _, _ = rotary(x, x, x, mask=causal_mask(T))
    assert not torch.allclose(out_plain, out_rope)


# =========================================================================
# The offset correctness check
# =========================================================================


def test_incremental_decode_matches_full_forward() -> None:
    """Token-by-token decode with a KV-cache must equal a single full pass.

    This is the test that validates the RoPE offset: each cached key was
    rotated by its absolute position at its own step, and each new token
    is rotated by offset = current cache length. If the offset were wrong,
    the two paths would diverge.
    """
    torch.manual_seed(0)
    mha = MultiHeadAttention(D_MODEL, N_HEADS, rope=_rope())
    mha.eval()

    x = torch.randn(1, T, D_MODEL)

    # Full forward, causal mask.
    full_out, _, _ = mha(x, x, x, mask=causal_mask(T))

    # Incremental: one token at a time, growing cache, no mask needed
    # (every key in the cache is a valid past/current position).
    cache: tuple[torch.Tensor, torch.Tensor] = (
        torch.zeros(1, N_HEADS, 0, D_K),
        torch.zeros(1, N_HEADS, 0, D_K),
    )
    steps = []
    for t in range(T):
        step = x[:, t : t + 1, :]
        out, _, cache = mha(step, step, step, mask=None, kv_cache=cache)
        steps.append(out)
    incremental_out = torch.cat(steps, dim=1)

    assert torch.allclose(full_out, incremental_out, atol=1e-5)
