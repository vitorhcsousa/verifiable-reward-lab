from __future__ import annotations

import pytest
import torch

from rlvr_from_scratch.model.sampling import _apply_top_p, sample

B, V = 4, 50


@pytest.fixture
def logits() -> torch.Tensor:
    """A deterministic (B, V) logits batch shared by the greedy tests."""
    torch.manual_seed(0)
    return torch.randn(B, V)


@pytest.fixture
def nucleus_logits() -> torch.Tensor:
    """Logits whose softmax is exactly [0.50, 0.30, 0.15, 0.05].

    With top_p=0.75 the cumulative sums are [0.50, 0.80, 0.95, 1.00], so
    the correct nucleus is {0, 1}: token 1 is the one that crosses p and
    must be KEPT (the off-by-one every buggy implementation drops).
    """
    return torch.log(torch.tensor([[0.50, 0.30, 0.15, 0.05]]))


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


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


# =========================================================================
# Temperature + reproducibility (640.2 / 640.6)
# =========================================================================


def test_seeded_sampling_is_reproducible(logits: torch.Tensor) -> None:
    """Same generator seed -> identical ids across two calls."""
    a = sample(logits, temperature=0.8, top_k=10, top_p=0.9, generator=_gen(42))
    b = sample(logits, temperature=0.8, top_k=10, top_p=0.9, generator=_gen(42))
    assert torch.equal(a, b)


def test_temperature_is_a_logit_rescale(logits: torch.Tensor) -> None:
    """sample(logits, T) must equal sample(logits/T, 1) draw-for-draw."""
    a = sample(logits, temperature=2.0, generator=_gen(5))
    b = sample(logits / 2.0, generator=_gen(5))  # temperature=1.0 default
    assert torch.equal(a, b)


def test_rejects_negative_temperature(logits: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="temperature"):
        sample(logits, temperature=-0.1)


def test_rejects_non_positive_top_k(logits: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="top_k"):
        sample(logits, top_k=0)


@pytest.mark.parametrize("bad_p", [0.0, -0.5, 1.5])
def test_rejects_out_of_range_top_p(logits: torch.Tensor, bad_p: float) -> None:
    with pytest.raises(ValueError, match="top_p"):
        sample(logits, top_p=bad_p)


# =========================================================================
# top-k (640.3)
# =========================================================================


def test_top_k_one_equals_greedy(logits: torch.Tensor) -> None:
    """Acceptance bullet: top_k=1 must reduce to greedy argmax."""
    out = sample(logits, top_k=1, generator=_gen(0))
    assert torch.equal(out, logits.argmax(dim=-1, keepdim=True))


@pytest.mark.parametrize("k", [V, V + 7])
def test_top_k_at_least_vocab_is_noop(logits: torch.Tensor, k: int) -> None:
    """top_k >= vocab must not change the sampled distribution at all."""
    filtered = sample(logits, top_k=k, generator=_gen(9))
    plain = sample(logits, generator=_gen(9))
    assert torch.equal(filtered, plain)


def test_top_k_restricts_support_per_row(logits: torch.Tensor) -> None:
    """Every draw must land inside each row's own top-k set (right axis,
    right broadcast: a wrong-axis threshold fails this immediately)."""
    k = 5
    allowed = logits.topk(k, dim=-1).indices  # (B, k)
    g = _gen(0)
    draws = torch.cat(
        [sample(logits, top_k=k, generator=g) for _ in range(200)], dim=1
    )  # (B, 200)
    for row in range(B):
        assert set(draws[row].tolist()) <= set(allowed[row].tolist())


# =========================================================================
# top-p (640.4)
# =========================================================================


def test_top_p_one_equals_full_softmax_sampling(logits: torch.Tensor) -> None:
    """Acceptance bullet: top_p=1.0 == plain full-softmax sampling."""
    filtered = sample(logits, top_p=1.0, generator=_gen(7))
    plain = sample(logits, generator=_gen(7))
    assert torch.equal(filtered, plain)


def test_top_p_keeps_first_token_crossing_p(nucleus_logits: torch.Tensor) -> None:
    """The off-by-one: with p=0.75 over [.50,.30,.15,.05] the nucleus is
    {0, 1} — the token that crosses p is kept, everything after is not."""
    g = _gen(0)
    draws = torch.cat(
        [sample(nucleus_logits, top_p=0.75, generator=g) for _ in range(200)],
        dim=1,
    )  # (1, 200)
    seen = set(draws.unique().tolist())
    assert seen <= {0, 1}  # nothing outside the nucleus, ever
    assert 1 in seen  # the boundary token is genuinely reachable


def test_top_p_renormalizes_and_zeroes_out_of_set(
    nucleus_logits: torch.Tensor,
) -> None:
    """Acceptance bullet: nucleus renormalises; out-of-set prob is exactly 0.

    Masked softmax over {0, 1} must give [.5/.8, .3/.8, 0, 0].
    """
    masked = _apply_top_p(nucleus_logits, 0.75)
    probs = torch.softmax(masked, dim=-1)
    expected = torch.tensor([[0.625, 0.375, 0.0, 0.0]])
    assert torch.allclose(probs, expected, atol=1e-6)
    assert (probs[..., 2:] == 0).all()  # exactly zero, not merely small
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1))


def test_top_p_tiny_keeps_argmax(logits: torch.Tensor) -> None:
    """A tiny p still keeps >= 1 survivor: exactly the argmax."""
    out = sample(logits, top_p=1e-6, generator=_gen(0))
    assert torch.equal(out, logits.argmax(dim=-1, keepdim=True))


def test_top_k_and_top_p_compose(logits: torch.Tensor) -> None:
    """Filters stack: draws stay inside the top-k set when both are on."""
    k = 3
    allowed = logits.topk(k, dim=-1).indices
    g = _gen(1)
    draws = torch.cat(
        [sample(logits, top_k=k, top_p=0.99, generator=g) for _ in range(100)],
        dim=1,
    )
    for row in range(B):
        assert set(draws[row].tolist()) <= set(allowed[row].tolist())
