from __future__ import annotations

import pytest
import torch

from rlvr_from_scratch.model.sampling import sample

B, V = 4, 50


@pytest.fixture
def logits() -> torch.Tensor:
    """A deterministic (B, V) logits batch shared by the greedy tests."""
    torch.manual_seed(0)
    return torch.randn(B, V)


def test_greedy_equals_argmax(logits: torch.Tensor) -> None:
    """At temperature=0, sample() returns each row's argmax token and
    takes the argmax branch directly — never dividing by zero."""
    out = sample(logits, temperature=0.0)
    expected = logits.argmax(dim=-1, keepdim=True)
    assert torch.equal(out, expected)
    assert torch.isfinite(logits).all()          # τ=0 must hit argmax, never logits/0
    assert (out >= 0).all() and (out < V).all()  # valid token ids


def test_output_shape_and_dtype(logits: torch.Tensor) -> None:
    """sample() returns one token id per row: shape (B, 1), dtype long."""
    out = sample(logits, temperature=0.0)
    assert out.shape == (B, 1)
    assert out.dtype == torch.long


def test_greedy_is_deterministic(logits: torch.Tensor) -> None:
    """The greedy path uses no RNG, so identical inputs give identical
    ids on every call."""
    a = sample(logits, temperature=0.0)
    b = sample(logits, temperature=0.0)
    assert torch.equal(a, b)


def test_per_row_independence() -> None:
    """Argmax must reduce over the vocab axis for each row independently,
    not across the batch; distinct per-row maxima expose a wrong-axis reduce."""
    logits = torch.zeros(B, V)
    cols = [i * 7 for i in range(B)]  # [0, 7, 14, 21]
    for i, c in enumerate(cols):
        logits[i, c] = 10.0
    out = sample(logits, temperature=0.0)
    assert torch.equal(out.squeeze(-1), torch.tensor(cols))


def test_tie_breaking_matches_argmax() -> None:
    """When two logits tie for the max, greedy stays deterministic and
    picks the first (lowest-index) one, matching torch.argmax."""
    logits = torch.full((1, V), -1.0)
    logits[0, 0] = 5.0
    logits[0, 1] = 5.0  # cols 0 and 1 both tie for the max
    out = sample(logits, temperature=0.0)
    assert out.item() == 0