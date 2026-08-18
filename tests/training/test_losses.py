"""Tests for the training objective.

Covers:
- The ln(V) anchor: an untrained model is exactly as confused as a uniform
  guess. Catches a wrong reduction, a wrong axis, and a label shift at once.
- Reduction is a mean, not a sum (invariant to batch size)
- Perfect and worst-case predictions, checked against hand-computable values
- One AdamW step provably decreases the loss on a fixed batch
- Gradient clipping returns a finite total norm
- Shape and dtype validation

Written before the implementation. Red until losses.py is filled in.
"""

from __future__ import annotations

import math

import pytest
import torch

from rlvr_from_scratch.model import DecoderTransformer, TransformerConfig
from rlvr_from_scratch.training.losses import cross_entropy_loss

# =========================================================================
# Constants and fixtures
# =========================================================================

VOCAB = 65  # TinyShakespeare's character vocabulary
B, T = 4, 16


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


def _model(**overrides: object) -> DecoderTransformer:
    """A small model with the real init path — the ln(V) claim is about
    the initialisation, so this must not be a hand-built stub."""
    cfg = {
        "vocab_size": VOCAB,
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 4,
        "max_seq_len": 32,
        **overrides,
    }
    return DecoderTransformer(TransformerConfig(**cfg))  # type: ignore[arg-type]


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.randint(0, VOCAB, (B, T + 1))
    return ids[:, :-1], ids[:, 1:]


# =========================================================================
# The ln(V) anchor — the reason this box exists
# =========================================================================


def test_untrained_loss_near_ln_vocab() -> None:
    """A randomly initialised model must be exactly as confused as a
    uniform guess over the vocabulary.

    Uniform probability is 1/V for every token, so the cross-entropy is
    -ln(1/V) = ln(V). For V=65 that is 4.174.

    This one assertion catches three different bugs at once:
      * a sum reduction instead of a mean  -> ~B*T times too large
      * the wrong axis in the flatten      -> nonsense, usually much larger
      * a label shift                      -> still finite, but off

    If this number is right, the objective is wired correctly. If it is
    wrong, every curve you plot afterwards measures the wrong thing.
    """
    model = _model()
    x, y = _batch()
    logits, _ = model(x)
    loss = cross_entropy_loss(logits, y)
    assert float(loss) == pytest.approx(math.log(VOCAB), abs=0.15)


@pytest.mark.parametrize("vocab", [4, 65, 256])
def test_ln_vocab_holds_across_vocab_sizes(vocab: int) -> None:
    # ln(V) is a claim about V, not about 65. If it only holds for one
    # vocabulary the test is fitting a constant, not checking a property.
    model = _model(vocab_size=vocab)
    ids = torch.randint(0, vocab, (B, T + 1))
    logits, _ = model(ids[:, :-1])
    loss = cross_entropy_loss(logits, ids[:, 1:])
    assert float(loss) == pytest.approx(math.log(vocab), abs=0.2)


# =========================================================================
# Reduction and hand-checkable values
# =========================================================================


def test_loss_is_invariant_to_batch_size() -> None:
    """Same example repeated: B=1 and B=8 must give the same loss.
    Proves the reduction is a mean and not a sum."""
    logits = torch.randn(1, T, VOCAB)
    targets = torch.randint(0, VOCAB, (1, T))
    one = cross_entropy_loss(logits, targets)
    eight = cross_entropy_loss(logits.repeat(8, 1, 1), targets.repeat(8, 1))
    assert float(one) == pytest.approx(float(eight), abs=1e-6)


def test_confident_and_correct_gives_near_zero() -> None:
    # A model that puts all its mass on the right token pays almost nothing.
    targets = torch.randint(0, VOCAB, (B, T))
    logits = torch.zeros(B, T, VOCAB)
    logits.scatter_(2, targets.unsqueeze(-1), 20.0)
    assert float(cross_entropy_loss(logits, targets)) < 1e-6


def test_confident_and_wrong_is_large() -> None:
    # The other end: mass on a token that is never the answer.
    targets = torch.zeros(B, T, dtype=torch.long)
    logits = torch.zeros(B, T, VOCAB)
    logits[:, :, 1] = 20.0
    assert float(cross_entropy_loss(logits, targets)) > 15.0


def test_uniform_logits_give_exactly_ln_vocab() -> None:
    # No model involved: all-equal logits are a uniform distribution, so
    # the loss must be ln(V) to floating-point precision, not approximately.
    logits = torch.zeros(B, T, VOCAB)
    targets = torch.randint(0, VOCAB, (B, T))
    assert float(cross_entropy_loss(logits, targets)) == pytest.approx(
        math.log(VOCAB), abs=1e-5
    )


# =========================================================================
# Gradients and one optimizer step
# =========================================================================


def test_loss_is_a_scalar_with_grad() -> None:
    model = _model()
    x, y = _batch()
    logits, _ = model(x)
    loss = cross_entropy_loss(logits, y)
    assert loss.ndim == 0
    assert loss.requires_grad


def test_one_step_strictly_decreases_loss() -> None:
    """AdamW on a fixed batch with a fixed seed. The loss after one step
    must be strictly lower.

    If it is not, one of three things is broken: gradients are not
    reaching the parameters, the sign is inverted, or the optimizer was
    handed the wrong parameter list. No training curve tells you this as
    fast as one assertion does.
    """
    model = _model()
    x, y = _batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    logits, _ = model(x)
    before = cross_entropy_loss(logits, y)
    before.backward()
    optimizer.step()
    optimizer.zero_grad()

    logits, _ = model(x)
    after = cross_entropy_loss(logits, y)
    assert float(after) < float(before)


def test_gradients_reach_every_parameter() -> None:
    # A finite, non-zero grad on every leaf. Catches a detached branch or a
    # sub-module that silently never learns.
    model = _model()
    x, y = _batch()
    logits, _ = model(x)
    cross_entropy_loss(logits, y).backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite grad"


def test_grad_clipping_returns_a_finite_norm() -> None:
    # clip_grad_norm_ returns the total norm BEFORE clipping. NaN here
    # means the forward pass already produced NaN — fail loudly, early.
    model = _model()
    x, y = _batch()
    cross_entropy_loss(model(x)[0], y).backward()
    total = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    assert torch.isfinite(total)
    assert float(total) > 0.0


# =========================================================================
# Validation
# =========================================================================


def test_rejects_mismatched_shapes() -> None:
    logits = torch.randn(B, T, VOCAB)
    targets = torch.randint(0, VOCAB, (B, T + 1))
    with pytest.raises(ValueError):
        cross_entropy_loss(logits, targets)


def test_rejects_float_targets() -> None:
    # get_batch returns long. A float here means something upstream lost
    # the dtype, and the C++ error message would not say where.
    logits = torch.randn(B, T, VOCAB)
    targets = torch.randint(0, VOCAB, (B, T)).float()
    with pytest.raises((ValueError, TypeError)):
        cross_entropy_loss(logits, targets)
