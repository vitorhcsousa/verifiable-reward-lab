"""Fixtures shared across the test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from rlvr_from_scratch.model.transformer import TransformerConfig
from rlvr_from_scratch.training.config import TrainConfig

# a pangram, so the vocab is exactly 26 letters plus space and newline.
# repeated because get_batch needs a corpus longer than one window. keeps
# the suite off the network, which train() would otherwise reach for.
TINY_TEXT = "the quick brown fox jumps over the lazy dog\n" * 200
TINY_VOCAB_SIZE = 28


def tiny_config(**kw: object) -> TrainConfig:
    """A config that runs in about a second and still exercises every path."""
    base: dict[str, object] = {
        "seed": 0,
        "batch_size": 4,
        "block_size": 16,
        "max_steps": 40,
        "warmup_steps": 4,
        "eval_interval": 10,
        "eval_iters": 2,
        "lr": 3e-3,
        "device": "cpu",
        "model": TransformerConfig(
            vocab_size=TINY_VOCAB_SIZE,
            d_model=32,
            n_layers=1,
            n_heads=2,
            max_seq_len=32,
        ),
    }
    return TrainConfig(**{**base, **kw})  # ty: ignore[invalid-argument-type]


@pytest.fixture
def tiny_corpus(tmp_path: Path) -> Path:
    """A corpus small enough to train on in about a second."""
    path = tmp_path / "corpus.txt"
    path.write_text(TINY_TEXT, encoding="utf-8")
    return path
