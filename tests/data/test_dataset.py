"""Tests for the training data path.

Covers:
- Corpus construction: split by position, ratio, tokenizer travels with it
- Batch shapes and dtype
- The shift: y is x moved one token left (the test the box exists for)
- Determinism: same generator seed -> identical batches, different -> not
- Train and val windows never overlap
- Edge cases: block_size too large, unknown split name

Written before the implementation. These fail until dataset.py is filled
in — that is the point.
"""

from __future__ import annotations

import random
import string
from pathlib import Path

import pytest
import torch

from rlvr_from_scratch.data.dataset import Corpus, get_batch, load_corpus

# =========================================================================
# Constants and fixtures
# =========================================================================

# Deterministic but NON-PERIODIC. A repeating corpus like "abcdefghij" * 26
# makes every 8-char window appear dozens of times, so any test that looks a
# window up in the corpus finds a match that is not the one sampled — the
# test passes without testing anything.
_RNG = random.Random(0)
_ALPHABET = string.ascii_lowercase + " \n"
TEXT = "".join(_RNG.choice(_ALPHABET) for _ in range(2000))

B = 4
T = 8


@pytest.fixture
def corpus_path(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.txt"
    path.write_text(TEXT, encoding="utf-8")
    return path


@pytest.fixture
def corpus(corpus_path: Path) -> Corpus:
    return load_corpus(corpus_path)


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# =========================================================================
# Corpus construction
# =========================================================================


def test_split_is_by_position_not_shuffled(corpus: Corpus) -> None:
    # val must be the tail of the corpus, verbatim. A shuffle would put
    # validation text inside training context and the loss would not say so.
    ids = torch.cat([corpus.train, corpus.val])
    expected = torch.tensor(corpus.tokenizer.encode(TEXT), dtype=torch.long)
    assert torch.equal(ids, expected)


def test_split_ratio_is_respected(corpus: Corpus) -> None:
    total = len(corpus.train) + len(corpus.val)
    assert len(corpus.train) / total == pytest.approx(0.9, abs=0.01)


def test_tensors_are_1d_long(corpus: Corpus) -> None:
    for part in (corpus.train, corpus.val):
        assert part.ndim == 1
        assert part.dtype == torch.long


def test_vocab_size_matches_unique_chars(corpus: Corpus) -> None:
    assert corpus.vocab_size == len(set(TEXT))


def test_rejects_degenerate_train_frac(corpus_path: Path) -> None:
    for bad in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            load_corpus(corpus_path, train_frac=bad)


def test_rejects_corpus_that_splits_to_nothing(tmp_path: Path) -> None:
    """A legal train_frac can still produce an empty side.

    3 characters at train_frac=0.1 gives n = int(0.3) = 0: train is empty
    and the Corpus is unusable. Nothing about that is visible until a
    batch is requested, so the error belongs here, where train_frac is
    still in scope.
    """
    path = tmp_path / "too_small.txt"
    path.write_text("abc", encoding="utf-8")
    with pytest.raises(ValueError, match="too small"):
        load_corpus(path, train_frac=0.1)


def test_short_but_non_empty_split_fails_with_a_useful_message(
    tmp_path: Path,
) -> None:
    """The case load_corpus deliberately does NOT catch.

    12 characters at 0.99 leaves val with 1 token: a legal split, just too
    short for any real block_size. load_corpus cannot know that — it never
    sees block_size — so the error correctly arrives later, from get_batch.
    What matters is that it names the three facts you need: which
    block_size, which split, and how long that split actually was.
    """
    path = tmp_path / "short_val.txt"
    path.write_text("abcdefghijkl", encoding="utf-8")
    small = load_corpus(path, train_frac=0.99)  # must NOT raise

    with pytest.raises(ValueError) as excinfo:
        get_batch(small, "val", batch_size=1, block_size=T, generator=_gen(0))
    message = str(excinfo.value)
    assert str(T) in message  # the block_size asked for
    assert "val" in message  # which split
    assert str(len(small.val)) in message  # how long it was


def test_split_rejects_unknown_name(corpus: Corpus) -> None:
    # Must raise, not quietly fall back to train.
    with pytest.raises((ValueError, KeyError)):
        corpus.split("test")  # type: ignore[arg-type]


def test_unknown_split_message_names_the_legal_values(corpus: Corpus) -> None:
    # The message has to point somewhere useful: a reader who follows it
    # must end up with a working call, not a second copy of this error.
    with pytest.raises(ValueError) as excinfo:
        corpus.split("valid")  # type: ignore[arg-type]
    message = str(excinfo.value)
    assert "'valid'" in message  # what was passed
    assert "train" in message and "val" in message  # what is accepted


# =========================================================================
# Batch shape
# =========================================================================


@pytest.mark.parametrize("split", ["train", "val"])
def test_batch_shapes_and_dtype(corpus: Corpus, split: str) -> None:
    x, y = get_batch(corpus, split, batch_size=B, block_size=T, generator=_gen(0))
    assert x.shape == y.shape == (B, T)
    assert x.dtype == y.dtype == torch.long


def test_ids_are_within_vocab(corpus: Corpus) -> None:
    x, y = get_batch(corpus, "train", batch_size=B, block_size=T, generator=_gen(0))
    for part in (x, y):
        assert int(part.min()) >= 0
        assert int(part.max()) < corpus.vocab_size


# =========================================================================
# The shift — the reason this box exists
# =========================================================================


def test_targets_are_inputs_shifted_by_one(corpus: Corpus) -> None:
    """y[b, t] is the token that follows x[b, t].

    An off-by-one here trains a different objective than intended — often
    copying the input instead of predicting the next token. The loss falls
    convincingly either way. This is the one test in the box that cannot
    be cut.
    """
    x, y = get_batch(corpus, "train", batch_size=B, block_size=T, generator=_gen(0))
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_shift_holds_against_the_raw_corpus(corpus: Corpus) -> None:
    # Stronger than the relation above: the pair must match a real window
    # of the corpus, not merely be self-consistent.
    x, y = get_batch(corpus, "train", batch_size=1, block_size=T, generator=_gen(7))
    data = corpus.train
    row = x[0]
    # unfold over the WHOLE tensor: slicing it first leaves the last few
    # start positions uncovered and the test fails at random.
    matches = (data.unfold(0, T, 1) == row).all(dim=1).nonzero()
    assert len(matches) == 1, "corpus must be non-periodic for this test"
    start = int(matches[0])
    assert torch.equal(y[0], data[start + 1 : start + 1 + T])


# =========================================================================
# Determinism
# =========================================================================


def test_same_seed_gives_identical_batches(corpus: Corpus) -> None:
    a = get_batch(corpus, "train", batch_size=B, block_size=T, generator=_gen(42))
    b = get_batch(corpus, "train", batch_size=B, block_size=T, generator=_gen(42))
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[1], b[1])


def test_different_seed_gives_different_batches(corpus: Corpus) -> None:
    # Without this, the test above would also pass for a batcher that
    # always returns the same window.
    a = get_batch(corpus, "train", batch_size=B, block_size=T, generator=_gen(1))
    b = get_batch(corpus, "train", batch_size=B, block_size=T, generator=_gen(2))
    assert not torch.equal(a[0], b[0])


def test_does_not_touch_global_rng(corpus: Corpus) -> None:
    # Sampling must not advance the global RNG, or two unrelated parts of a
    # run start influencing each other's randomness.
    torch.manual_seed(0)
    before = torch.rand(1)
    torch.manual_seed(0)
    get_batch(corpus, "train", batch_size=B, block_size=T, generator=_gen(0))
    after = torch.rand(1)
    assert torch.equal(before, after)


# =========================================================================
# Split isolation
# =========================================================================


def test_train_and_val_windows_never_overlap(corpus: Corpus) -> None:
    # Every val window must come from the val tensor only. Sampling val
    # windows out of the full corpus is the classic leak.
    val_windows = corpus.val.unfold(0, T, 1)
    for seed in range(8):
        x, _ = get_batch(
            corpus, "val", batch_size=B, block_size=T, generator=_gen(seed)
        )
        for row in x:
            found = (val_windows == row).all(dim=1).any()
            assert bool(found), "a val batch contained a window not in val"


# =========================================================================
# Edge cases
# =========================================================================


def test_raises_when_block_size_exceeds_split(corpus: Corpus) -> None:
    with pytest.raises(ValueError):
        get_batch(
            corpus,
            "val",
            batch_size=B,
            block_size=len(corpus.val) + 1,
            generator=_gen(0),
        )


def test_never_reads_past_the_end_of_the_split(tmp_path: Path) -> None:
    """The `hi = len(data) - block_size - 1` bound, stressed at the edge.

    A window needs T+1 tokens: T for x and one more for the last target.
    Get the bound wrong by one and the largest start index makes y run off
    the end — torch silently returns a SHORT slice, torch.stack then blows
    up on mismatched shapes, and it only happens on the rare batch that
    samples the very last position. On a big corpus with a fixed seed that
    is a ~0.1% event, which means CI green today and a confusing crash in
    three weeks.

    So: a deliberately tiny split, sampled many times, so every legal start
    index is hit. Every batch must be exactly (B, T).
    """
    path = tmp_path / "tiny.txt"
    path.write_text("".join(_ALPHABET[i % len(_ALPHABET)] for i in range(40)))
    small = load_corpus(path, train_frac=0.5)
    for seed in range(60):
        x, y = get_batch(small, "val", batch_size=8, block_size=T, generator=_gen(seed))
        assert x.shape == y.shape == (8, T)
        assert int(y.max()) < small.vocab_size


def test_generator_is_keyword_only(corpus: Corpus) -> None:
    with pytest.raises(TypeError):
        get_batch(corpus, "train", B, T, _gen(0))  # type: ignore[misc]
