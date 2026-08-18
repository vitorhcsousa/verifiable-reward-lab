"""
The training objective: logits in, one scalar out.

Kept out of the model on purpose. `DecoderTransformer.forward` already
returns a tuple `(logits, caches)`; threading a `targets=None` argument
through it would make the shape of the return value depend on the shape of
the call — sometimes two items, sometimes three. The model's job is to
produce logits. Turning logits into a number you can differentiate is a
separate job, and it lives here.

──────────────────────────────────────────────────────────────────────────
HOW TO WORK THIS FILE

    uv run pytest tests/training/ -q      # red until you fill this in

Order:
    1. cross_entropy_loss   -> the shape/validation tests
    2. everything else is already covered by the tests you have

The single most valuable line in the whole box is the ln(V) assertion. A
model that has learned nothing must be exactly as confused as a uniform
guess over the vocabulary. If that number is wrong, every loss curve you
plot for the rest of this project is measuring the wrong thing.
──────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn.functional as F  # noqa: N812

if TYPE_CHECKING:
    from torch import Tensor


def cross_entropy_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Mean cross-entropy over every position in the batch.

    Args:
        logits:  (B, T, V) float. Raw scores, NOT softmaxed — F.cross_entropy
            applies log_softmax internally, and doing it twice silently
            flattens the distribution instead of raising.
        targets: (B, T) long. The next-token ids from get_batch.

    Returns:
        A 0-dim tensor carrying grad. Scalar, because .backward() needs one.

    Raises:
        ValueError: if the leading dims disagree or targets is not integral.
    """
    # ── 1 ─ validate the shapes before touching them ──────────────────
    #   if logits.shape[:2] != targets.shape: raise ValueError(...)
    #   Name BOTH shapes in the message. The alternative is a torch error
    #   about a mismatch that does not say which of the two tensors is the
    #   one you got wrong, and at 5pm that costs ten minutes.
    #
    #   Also check targets.dtype is an integer type. A float targets tensor
    #   fails deep inside the C++ kernel with "expected scalar type Long",
    #   and nothing in that message points back at get_batch.
    #
    # ── 2 ─ flatten to (B*T, V) and (B*T,) ────────────────────────────
    #   vocab_size = logits.size(-1)
    #   flat_logits  = logits.reshape(-1, vocab_size)
    #   flat_targets = targets.reshape(-1)
    #
    #   reshape, not view. view() requires contiguous memory and raises if
    #   the logits came out of an op that left non-contiguous strides —
    #   which depends on the model internals, so it works today and breaks
    #   the week you change the lm head. reshape falls back to a copy.
    #
    #   Why flatten at all: F.cross_entropy wants (N, C) against (N,). The
    #   B and T axes carry no meaning for the loss — every one of the B*T
    #   positions is an independent prediction, which is exactly why a
    #   window of T tokens is T training examples and not one.
    #
    # ── 3 ─ the loss ──────────────────────────────────────────────────
    #   return F.cross_entropy(flat_logits, flat_targets, reduction="mean")
    #
    #   reduction="mean" is the default, but write it. The difference
    #   between mean and sum is a factor of B*T — the loss still falls,
    #   the curve still looks like learning, and your effective learning
    #   rate is 4096x what you think it is.
    raise NotImplementedError("step 1-3 — see the plan above")
