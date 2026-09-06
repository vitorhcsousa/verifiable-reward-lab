"""
Tests for the SFT baseline config.

The defaults here are not preferences, they are the frozen reference arm of
RQ1. A test that pins them is what makes an accidental edit show up as a red
test instead of as a baseline that quietly moved.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from verifiable_reward_lab.training.sft_config import VERSION, SFTConfig

COMMITTED = Path(__file__).resolve().parents[2] / "configs" / "sft_qwen05b.yaml"


def test_committed_config_loads() -> None:
    """The file the protocol names has to be readable by this code."""
    assert SFTConfig.load(COMMITTED).model_name == "Qwen/Qwen2.5-0.5B"


def test_committed_config_is_the_frozen_one() -> None:
    assert SFTConfig.load(COMMITTED) == SFTConfig()


def test_frozen_choices() -> None:
    c = SFTConfig()
    assert (c.epochs, c.lr, c.effective_batch) == (2, 1e-5, 16)
    assert c.mask_prompt_loss is True


def test_model_is_pinned_to_a_sha() -> None:
    """`main` moves, and a baseline trained from a moving checkpoint is not
    a baseline anyone can reproduce."""
    revision = SFTConfig().model_revision
    assert len(revision) == 40
    assert revision != "main"


def test_split_seed_matches_the_protocol() -> None:
    """If this drifts from the split seed, the model trains on examples the
    dev set will later grade it on."""
    assert SFTConfig().split_seed == 1337


def test_round_trip_through_yaml(tmp_path: Path) -> None:
    c = dataclasses.replace(SFTConfig(), epochs=1, device="cpu")
    path = tmp_path / "sft.yaml"
    c.save(path)
    assert SFTConfig.load(path) == c


@pytest.mark.parametrize("field", ["epochs", "batch_size", "grad_accum", "max_seq_len"])
def test_non_positive_sizes_are_rejected(field: str) -> None:
    with pytest.raises(ValueError, match=f"{field} must be > 0"):
        dataclasses.replace(SFTConfig(), **{field: 0})


def test_negative_weight_decay_is_rejected() -> None:
    with pytest.raises(ValueError, match="weight_decay must be >= 0"):
        dataclasses.replace(SFTConfig(), weight_decay=-0.1)


def test_warmup_ratio_out_of_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="warmup_ratio must be in"):
        dataclasses.replace(SFTConfig(), warmup_ratio=1.0)


def test_unknown_dtype_is_rejected() -> None:
    with pytest.raises(ValueError, match="dtype must be one of"):
        dataclasses.replace(SFTConfig(), dtype="float8")


def test_unknown_key_is_not_silently_defaulted() -> None:
    payload = SFTConfig().to_dict() | {"optimizer": "lion"}
    with pytest.raises(ValueError, match="unknown config keys"):
        SFTConfig.from_dict(payload)


def test_version_mismatch_is_rejected() -> None:
    payload = SFTConfig().to_dict() | {"version": VERSION + 1}
    with pytest.raises(ValueError, match="cannot be read by version"):
        SFTConfig.from_dict(payload)
