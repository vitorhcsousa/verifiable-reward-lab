"""
Training data path: a corpus encoded once, split by position, sampled into
(x, y) batches where y is x shifted one token to the left.

No model in here, no training loop. This file answers exactly one question —
"what does the next batch look like?" — and answers it deterministically,
which is why every entry point takes an explicit torch.Generator instead of
reaching for the global RNG.

The pairing is the whole idea. A window of T tokens is not one training
example, it is T of them: at every position t the model sees x[:, :t+1] and
must predict y[:, t]. Get the shift wrong by one and the model trains on a
different objective than you think, the loss still falls, and nothing
complains until you read the samples days later.

──────────────────────────────────────────────────────────────────────────
HOW TO WORK THIS FILE

Every function below is a stub with a numbered plan in its body. Delete the
comments as you replace them with code — a step still commented is a step
still to do.

Run the tests first and let them drive the order:

    uv run pytest tests/data/ -q          # 18 red

Suggested order, each stage turning a group green:
    1. Corpus.split      -> the "unknown split" test
    2. load_corpus       -> the construction tests (6 of them)
    3. get_batch         -> everything else

The tests were checked by mutation: seven plausible wrong implementations
were written against them and all seven were caught. If a test fails, it is
much more likely your code than the test.
──────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch

from rlvr_from_scratch.tokenizer.char import CharTokenizer

if TYPE_CHECKING:
    from torch import Tensor

Split = Literal["train", "val"]


@dataclass(frozen=True)
class Corpus:
    """A corpus encoded once and split by position.

    Frozen because the split is a fact about a run: if it can be mutated
    after the fact, the vocabulary saved next to a checkpoint and the data
    the checkpoint was trained on can silently disagree.
    """

    train: Tensor
    """Training token ids, 1-D, dtype long."""

    val: Tensor
    """Validation token ids, 1-D, dtype long. Comes strictly after train."""

    tokenizer: CharTokenizer
    """The tokenizer these ids were produced with. Travels with the data so
    decoding a batch never needs a second source of truth."""

    @property
    def vocab_size(self) -> int:
        """Width of the embedding table and of the lm head."""
        return self.tokenizer.vocab_size

    def split(self, split: Split) -> Tensor:
        """Return the token tensor for `split`.

        Args:
            split: "train" or "val".

        Returns:
            The 1-D long tensor for that split.

        Raises:
            ValueError: on any other name.
        """
        if split == "train":
            return self.train
        if split == "val":
            return self.val
        else:
            msg = f"The split value {split} is not on of the allowed values ['split','val']."
            raise ValueError(msg)


def load_corpus(path: Path, *, train_frac: float = 0.9) -> Corpus:
    """Read a text file, build the vocab, encode once, split by position.

    Args:
        path:       Path to a UTF-8 .txt file.
        train_frac: Fraction of tokens that go to train. Must be in (0, 1).

    Returns:
        A Corpus whose train and val tensors are 1-D and dtype long.

    Raises:
        ValueError: if train_frac is outside (0, 1), or the corpus is too
            small for the split to leave both sides non-empty.
    """
    # ── 1 ─ validate train_frac BEFORE reading anything ───────────────
    #   if not 0.0 < train_frac < 1.0: raise ValueError(...)
    #   Both ends are exclusive: 0.0 means no training data, 1.0 means no
    #   validation data. Neither is a split. Put the received value in the
    #   message with !r — "must be in (0, 1)" alone makes the caller go
    #   looking for what they actually passed.
    if not 0.0 < train_frac < 1.0:
        msg = f"The value of train_frac must be in (0,1), got {train_frac}"
        raise ValueError(msg)
    text = path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)

    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(train_frac * len(ids))
    train, val = ids[:n], ids[n:]
    return Corpus(train=train, val=val, tokenizer=tokenizer)


def get_batch(
    corpus: Corpus,
    split: Split,
    *,
    batch_size: int,
    block_size: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Sample a batch of contiguous windows and their shifted targets.

    Args:
        corpus:     The encoded corpus.
        split:      "train" or "val".
        batch_size: Number of independent windows, B.
        block_size: Tokens per window, T.
        generator:  Explicit RNG. Required, and keyword-only.

    Returns:
        (x, y), both (B, T) dtype long, where y[b, t] is the token that
        follows x[b, t] in the corpus.

    Raises:
        ValueError: if block_size leaves no room for a full window in the
            chosen split.
    """
    # ── 1 ─ pick the tensor ───────────────────────────────────────────
    #   data = corpus.split(split)
    #   Go through Corpus.split, do not reach into corpus.train directly.
    #   One gate means one place where "val" can be got wrong.

    # ── 2 ─ the upper bound on a start index ──────────────────────────
    #   hi = len(data) - block_size - 1
    #
    #   Where the -1 comes from: a window consumes T+1 tokens, T for x and
    #   one more for the final target. With randint(0, hi) the largest
    #   start is hi-1, so the last index y touches is
    #       (hi - 1) + block_size = len(data) - 2
    #   which is in bounds. Drop the -1 and the largest start makes y run
    #   off the end. torch does NOT raise: slicing past the end returns a
    #   SHORT tensor, and you get a torch.stack shape error instead —
    #   on roughly 0.1% of batches. Green today, mystifying in November.
    #
    #   Do not trust the paragraph you just read. Change the -1 to +1 and
    #   watch test_never_reads_past_the_end_of_the_split go red; that test
    #   exists because this exact mutation slipped through an earlier
    #   version of the suite.
    #
    # ── 3 ─ guard before sampling ─────────────────────────────────────
    #   if hi < 1: raise ValueError(...)
    #   torch.randint with an empty range reports a problem that names
    #   neither block_size nor which split was too short. Say both.
    #
    # ── 4 ─ draw the start indices ────────────────────────────────────
    #   ix = torch.randint(0, hi, (batch_size,), generator=generator)
    #
    #   Pass the generator. Without it torch uses the global RNG, and then
    #   reproducibility is a property of the process rather than of this
    #   function: a stray torch.manual_seed() in an unrelated file starts
    #   changing your batches, and two runs with "the same seed" diverge.
    #   Sampling WITH replacement is fine — repeated windows in one batch
    #   are harmless at this scale.
    #
    # ── 5 ─ stack the windows ─────────────────────────────────────────
    #   x = torch.stack([data[i     : i + block_size    ] for i in ix])
    #   y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    #
    #   Build y from `data`, not from `x`. Deriving it from x (rolling,
    #   or concatenating a token onto a slice of x) makes the two
    #   self-consistent by construction, which passes a naive shift test
    #   while neither of them is a real window of the corpus.
    #
    # ── 6 ─ return (x, y) ─────────────────────────────────────────────
    #   Both (B, T), dtype long, ready for the loss you write on Tuesday.
    raise NotImplementedError("step 1-6 — see the plan above")
