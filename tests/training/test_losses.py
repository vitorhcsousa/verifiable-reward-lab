"""Tests for the training objective.

Covers:
- The ln(V) anchor: an untrained model is exactly as confused as a uniform
  guess, across several vocabulary sizes
- Reduction is a mean, not a sum (invariant to batch size)
- Perfect and worst-case predictions against hand-computable values
- The analytic gradient: dL/dlogits = (softmax - onehot) / N
- One AdamW step provably decreases the loss on a fixed batch
- Gradient flow to every parameter, and a finite clipping norm
- Shape and dtype validation
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
def set_seed() -> None:
    torch.manual_seed(0)


def _model(vocab_size: int = VOCAB) -> DecoderTransformer:
    """A small model on the real init path.

    The ln(V) claim is a claim about the initialization, so this must not
    be a hand-built stub with hand-set logits.
    """
    return DecoderTransformer(
        TransformerConfig(
            vocab_size=vocab_size,
            d_model=64,
            n_layers=2,
            n_heads=4,
            max_seq_len=32,
        )
    )


def _batch(vocab_size: int = VOCAB) -> tuple[torch.Tensor, torch.Tensor]:
    """Inputs and next-token targets, shifted by one."""
    ids = torch.randint(0, vocab_size, (B, T + 1))
    return ids[:, :-1], ids[:, 1:]


# =========================================================================
# The ln(V) anchor
# =========================================================================


def test_untrained_loss_near_ln_vocab() -> None:
    """A randomly initialized model must be as confused as a uniform guess.

    Uniform probability is 1/V for every token, so the cross-entropy is
    -ln(1/V) = ln(V). For V=65 that is 4.174.

    This single assertion catches three distinct bugs:
      * a sum reduction instead of a mean  -> ~B*T times too large
      * the wrong axis in the flatten      -> nonsense, usually larger
      * a label shift                      -> finite, but off
    """
    model = _model()
    x, y = _batch()
    logits, _ = model(x)
    loss = cross_entropy_loss(logits, y)
    assert float(loss) == pytest.approx(math.log(VOCAB), abs=0.15)


@pytest.mark.parametrize("vocab_size", [4, 65, 256])
def test_ln_vocab_holds_across_vocab_sizes(vocab_size: int) -> None:
    # ln(V) is a claim about V, not about 65. If it only holds for one
    # vocabulary, the test is fitting a constant rather than a property.
    model = _model(vocab_size)
    x, y = _batch(vocab_size)
    logits, _ = model(x)
    loss = cross_entropy_loss(logits, y)
    assert float(loss) == pytest.approx(math.log(vocab_size), abs=0.2)


# =========================================================================
# Reduction and hand-checkable values
# =========================================================================


def test_loss_is_invariant_to_batch_size() -> None:
    """The same example repeated must give the same loss.

    Proves the reduction is a mean and not a sum.
    """
    logits = torch.randn(1, T, VOCAB)
    targets = torch.randint(0, VOCAB, (1, T))
    one = cross_entropy_loss(logits, targets)
    eight = cross_entropy_loss(logits.repeat(8, 1, 1), targets.repeat(8, 1))
    assert float(one) == pytest.approx(float(eight), abs=1e-6)


def test_uniform_logits_give_exactly_ln_vocab() -> None:
    # No model involved: all-equal logits ARE the uniform distribution, so
    # the loss must be ln(V) to floating-point precision, not approximately.
    logits = torch.zeros(B, T, VOCAB)
    targets = torch.randint(0, VOCAB, (B, T))
    assert float(cross_entropy_loss(logits, targets)) == pytest.approx(
        math.log(VOCAB), abs=1e-5
    )


def test_confident_and_correct_gives_near_zero() -> None:
    # All mass on the right token: -ln(p) with p -> 1 costs almost nothing.
    targets = torch.randint(0, VOCAB, (B, T))
    logits = torch.zeros(B, T, VOCAB)
    logits.scatter_(2, targets.unsqueeze(-1), 20.0)
    assert float(cross_entropy_loss(logits, targets)) < 1e-6


def test_confident_and_wrong_is_large() -> None:
    # All mass on a token that is never the answer. The loss approaches the
    # logit gap itself: logsumexp(20, 0...) - 0 -> 20.
    targets = torch.zeros(B, T, dtype=torch.long)
    logits = torch.zeros(B, T, VOCAB)
    logits[:, :, 1] = 20.0
    assert float(cross_entropy_loss(logits, targets)) > 15.0


# =========================================================================
# Gradients
# =========================================================================


def test_loss_is_a_scalar_with_grad() -> None:
    model = _model()
    x, y = _batch()
    logits, _ = model(x)
    loss = cross_entropy_loss(logits, y)
    assert loss.ndim == 0
    assert loss.requires_grad


def test_grad_wrt_logits_is_softmax_minus_onehot() -> None:
    """The whole reason cross-entropy pairs with softmax.

    dL/dz = (softmax(z) - onehot(y)) / N, where N = B*T. No exponentials
    survive into the gradient — that is what keeps the backward pass
    numerically stable, and it is checkable by hand.
    """
    logits = torch.randn(B, T, VOCAB, requires_grad=True)
    targets = torch.randint(0, VOCAB, (B, T))

    cross_entropy_loss(logits, targets).backward()

    probs = torch.softmax(logits.detach(), dim=-1)
    onehot = torch.zeros_like(probs).scatter_(2, targets.unsqueeze(-1), 1.0)
    expected = (probs - onehot) / (B * T)

    assert logits.grad is not None
    torch.testing.assert_close(logits.grad, expected, atol=1e-6, rtol=1e-5)


def test_gradients_reach_every_parameter() -> None:
    # A finite gradient on every leaf. Catches a detached branch or a
    # sub-module that silently never learns.
    model = _model()
    x, y = _batch()
    logits, _ = model(x)
    cross_entropy_loss(logits, y).backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite grad"


def test_grad_clipping_returns_a_finite_norm() -> None:
    # clip_grad_norm_ returns the total norm BEFORE clipping. A NaN here
    # means the forward pass already produced NaN — fail loudly, early.
    model = _model()
    x, y = _batch()
    logits, _ = model(x)
    cross_entropy_loss(logits, y).backward()
    total = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    assert torch.isfinite(total)
    assert float(total) > 0.0


def test_one_step_strictly_decreases_loss() -> None:
    """AdamW on a fixed batch. The loss after one step must be lower.

    If it is not, one of three things is broken: gradients are not
    reaching the parameters, the sign is inverted, or the optimizer was
    handed the wrong parameter list. There is no dropout in the config,
    so both forward passes are deterministic and the comparison is exact.
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


# =========================================================================
# Validation
# =========================================================================


def test_rejects_mismatched_shapes() -> None:
    logits = torch.randn(B, T, VOCAB)
    targets = torch.randint(0, VOCAB, (B, T + 1))
    with pytest.raises(ValueError, match="leading dims"):
        cross_entropy_loss(logits, targets)


def test_error_message_names_both_shapes() -> None:
    # The point of the custom error is that it says WHICH tensor is wrong.
    logits = torch.randn(B, T, VOCAB)
    targets = torch.randint(0, VOCAB, (B, T + 1))
    with pytest.raises(ValueError) as excinfo:
        cross_entropy_loss(logits, targets)
    message = str(excinfo.value)
    assert str((B, T)) in message
    assert str((B, T + 1)) in message


def test_rejects_float_targets() -> None:
    # get_batch returns long. A float here means something upstream lost
    # the dtype, and the C++ error message would not say where.
    logits = torch.randn(B, T, VOCAB)
    targets = torch.randint(0, VOCAB, (B, T)).float()
    with pytest.raises(ValueError, match="torch.long"):
        cross_entropy_loss(logits, targets)


def test_rejects_int32_targets() -> None:
    # numpy -> torch produces int32 with no warning. Rejecting it at the
    # boundary is deliberate: F.cross_entropy would fail deeper down.
    logits = torch.randn(B, T, VOCAB)
    targets = torch.randint(0, VOCAB, (B, T), dtype=torch.int32)
    with pytest.raises(ValueError, match="torch.long"):
        cross_entropy_loss(logits, targets)
