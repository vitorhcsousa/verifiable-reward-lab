"""Training objectives built from scratch.

Implements token-level cross-entropy for autoregressive language modelling,
with shape annotations at every step.

Kept out of the model on purpose. `DecoderTransformer.forward` already
returns `(logits, caches)`; threading a `targets=None` argument through it
would make the arity of the return value depend on the arity of the call.
The model produces logits. Turning logits into a differentiable scalar is
a separate job, and it lives here.

Reference: "Cross-Entropy for Language Models"
    https://www.vitorsousa.com/bits/cross-entropy-language-models
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

# =========================================================================
# Token-level cross-entropy
# =========================================================================


def cross_entropy_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Mean cross-entropy over every position in the batch.

    Every one of the B*T positions is an independent next-token
    prediction, which is exactly why a window of T tokens is T training
    examples and not one. The batch and time axes carry no meaning for
    the loss, so they are flattened away.

    Sanity check: an untrained model must score ln(V) — exactly as
    confused as a uniform guess over the vocabulary. If that number is
    wrong, every loss curve downstream is measuring the wrong thing.

    Args:
        logits:  (B, T, V) float. Raw scores, NOT softmaxed —
            `F.cross_entropy` applies log_softmax internally, and doing
            it twice silently flattens the distribution instead of
            raising.
        targets: (B, T) long. The next-token ids from `get_batch`.

    Returns:
        (), a 0-dim tensor carrying grad. Scalar, because `.backward()`
        needs one.

    Raises:
        ValueError: If the leading dims disagree, or targets is not long.
    """
    # =========================================
    # 1. Validate — name both shapes
    # =========================================
    if logits.shape[:2] != targets.shape:
        msg = (
            f"leading dims must match: logits {tuple(logits.shape)} "
            f"gives (B, T) = {tuple(logits.shape[:2])}, "
            f"targets is {tuple(targets.shape)}"
        )
        raise ValueError(msg)

    if targets.dtype != torch.long:
        msg = (
            f"targets must be torch.long, got {targets.dtype}. "
            f"Check the dtype coming out of get_batch."
        )
        raise ValueError(msg)

    # =========================================
    # 2. Flatten to the (N, C) / (N,) form F.cross_entropy wants
    # =========================================
    vocab_size = logits.size(-1)

    # reshape, not view. view() requires contiguous memory and raises if
    # the logits came out of an op that left non-contiguous strides —
    # which depends on the lm head. reshape falls back to a copy.
    # (B, T, V) -> (B*T, V)
    flat_logits = logits.reshape(-1, vocab_size)
    # (B, T) -> (B*T,)
    flat_targets = targets.reshape(-1)

    # =========================================
    # 3. Reduce — "mean" written explicitly
    # =========================================
    # The difference between mean and sum is a factor of B*T: the loss
    # still falls and the curve still looks like learning, but the
    # effective learning rate is B*T times what you think it is.
    # (B*T, V), (B*T,) -> ()
    return F.cross_entropy(flat_logits, flat_targets, reduction="mean")
