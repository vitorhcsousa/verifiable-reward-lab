"""Decoding strategies: turn next-token logits into one sampled token id.

One pure function, ``sample``, covering the standard decoding modes:

    greedy        temperature == 0     argmax, no randomness
    temperature   rescale logits/T     T < 1 sharpens, T > 1 flattens
    top-k         keep k best logits   hard cap on the candidate set
    top-p         nucleus sampling     smallest prefix with cum. prob >= p

Filters are applied on *logits* (masking to -inf), before the temperature
rescale and the softmax. Masked tokens therefore get probability exactly
zero, and the surviving distribution is renormalized for free by softmax.

Order of operations:

    logits -> top-k mask -> top-p mask -> /temperature -> softmax -> draw

The greedy guard short-circuits before any filtering: the argmax survives
every filter, and returning early avoids dividing logits by zero.

Reference: Holtzman et al., "The Curious Case of Neural Text Degeneration"
    https://arxiv.org/abs/1904.09751
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _apply_top_k(logits: Tensor, top_k: int) -> Tensor:
    """Keep the ``top_k`` largest logits per row; mask the rest to -inf.

    ``top_k >= vocab`` is a no-op. Ties at the threshold all survive,
    matching the "keep everything at least as good as the k-th" reading.
    """
    k = min(top_k, logits.size(-1))
    # k-th largest value per row, keepdim so it broadcasts: (B, 1).
    threshold = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _apply_top_p(logits: Tensor, top_p: float) -> Tensor:
    """Nucleus filter: keep the smallest prefix whose cum. prob crosses p.

    The classic off-by-one lives here: after ``cumprobs > top_p`` the mask
    is shifted right by one so the *first token crossing p is kept* — this
    also guarantees at least one survivor per row.
    """
    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
    cumprobs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    remove = cumprobs > top_p
    # Shift right: position i inherits the flag of position i-1, so the
    # boundary token (the one that crosses p) stays in the nucleus.
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    # Scatter the sorted-order mask back to vocabulary order. sorted_idx is
    # a permutation, so every position is written exactly once.
    mask = remove.scatter(-1, sorted_idx, remove)
    return logits.masked_fill(mask, float("-inf"))


def sample(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample one next-token id per row from ``logits``.

    Args:
        logits:      Raw next-token scores, shape (B, vocab).
        temperature: Softmax temperature. ``0`` means greedy argmax
                     (filters are irrelevant then — argmax survives all).
        top_k:       If set, keep only the k largest logits per row.
        top_p:       If set, keep the smallest high-probability prefix
                     whose cumulative probability crosses ``top_p``.
                     ``1.0`` keeps everything (full-softmax sampling).
        generator:   Optional RNG for reproducible draws.

    Returns:
        Token ids, shape (B, 1), dtype long.
    """
    if temperature < 0:
        msg = f"temperature must be >= 0, got {temperature}"
        raise ValueError(msg)
    if top_k is not None and top_k < 1:
        msg = f"top_k must be >= 1, got {top_k}"
        raise ValueError(msg)
    if top_p is not None and not 0.0 < top_p <= 1.0:
        msg = f"top_p must be in (0, 1], got {top_p}"
        raise ValueError(msg)

    # Greedy guard: never divide by zero, never touch the RNG.
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    if top_k is not None:
        logits = _apply_top_k(logits, top_k)
    if top_p is not None:
        logits = _apply_top_p(logits, top_p)

    probs = F.softmax(logits / temperature, dim=-1)  # (B, vocab), rows sum to 1
    return torch.multinomial(probs, num_samples=1, generator=generator)  # (B, 1)


if __name__ == "__main__":
    torch.manual_seed(0)
    demo_logits = torch.randn(2, 8)
    g = torch.Generator().manual_seed(0)

    greedy = sample(demo_logits, temperature=0.0)
    top_k_ids = sample(demo_logits, top_k=3, generator=g)
    top_p_ids = sample(demo_logits, top_p=0.9, generator=g)

    assert greedy.shape == top_k_ids.shape == top_p_ids.shape == (2, 1)
    assert torch.equal(sample(demo_logits, top_k=1, generator=g), greedy)

    print("greedy:", greedy.squeeze(-1).tolist())
    print("top-k :", top_k_ids.squeeze(-1).tolist())
    print("top-p :", top_p_ids.squeeze(-1).tolist())
    print("OK")
