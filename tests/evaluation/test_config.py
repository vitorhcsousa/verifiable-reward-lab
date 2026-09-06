"""Tests for the eval config: the round trip, and the runs it refuses."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from verifiable_reward_lab.evaluation.config import VERSION, EvalConfig


def test_defaults_are_the_frozen_choices() -> None:
    """These four are the protocol's §3, not preferences: if one changes,
    the baseline number stops being comparable."""
    c = EvalConfig()
    assert (c.split, c.n_examples, c.n_shots, c.seed) == ("dev", 200, 4, 1337)


def test_round_trip_through_yaml(tmp_path: Path) -> None:
    c = dataclasses.replace(EvalConfig(), n_examples=8, device="cpu")
    path = tmp_path / "config.yaml"
    c.save(path)
    assert EvalConfig.load(path) == c


def test_held_out_needs_an_explicit_confirmation() -> None:
    """One edited word must not be enough to read the held-out set."""
    with pytest.raises(ValueError, match="frozen until the block closes"):
        dataclasses.replace(EvalConfig(), split="held_out")


def test_held_out_with_confirmation_is_allowed() -> None:
    c = dataclasses.replace(EvalConfig(), split="held_out", confirm_held_out=True)
    assert c.split == "held_out"


def test_unknown_split_is_rejected() -> None:
    with pytest.raises(ValueError, match="split must be one of"):
        dataclasses.replace(EvalConfig(), split="train")


def test_unknown_dtype_is_rejected() -> None:
    with pytest.raises(ValueError, match="dtype must be one of"):
        dataclasses.replace(EvalConfig(), dtype="float8")


def test_too_many_shots_is_rejected() -> None:
    with pytest.raises(ValueError, match="n_shots must be in"):
        dataclasses.replace(EvalConfig(), n_shots=9)


@pytest.mark.parametrize("field", ["n_examples", "max_new_tokens", "batch_size"])
def test_non_positive_sizes_are_rejected(field: str) -> None:
    with pytest.raises(ValueError, match=f"{field} must be > 0"):
        dataclasses.replace(EvalConfig(), **{field: 0})


def test_unknown_key_is_not_silently_defaulted() -> None:
    payload = EvalConfig().to_dict() | {"temperature": 0.7}
    with pytest.raises(ValueError, match="unknown config keys"):
        EvalConfig.from_dict(payload)


def test_version_mismatch_is_rejected() -> None:
    payload = EvalConfig().to_dict() | {"version": VERSION + 1}
    with pytest.raises(ValueError, match="cannot be read by version"):
        EvalConfig.from_dict(payload)


def test_missing_version_is_rejected() -> None:
    payload = EvalConfig().to_dict()
    del payload["version"]
    with pytest.raises(ValueError, match="no 'version'"):
        EvalConfig.from_dict(payload)


def test_load_rejects_a_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- not\n- a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a YAML mapping"):
        EvalConfig.load(path)
