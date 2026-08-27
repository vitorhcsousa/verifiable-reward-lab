"""
Tests for the training config.

The config is the committed record of a run, so these tests care about two
things: that it refuses to describe a run that cannot happen, and that it
survives a trip to disk and back unchanged.

Nothing here trains anything. A config test that builds a model is testing
the model.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml

from rlvr_from_scratch.model.transformer import TransformerConfig
from rlvr_from_scratch.training.config import VERSION, TrainConfig


def small_model(**kw: object) -> TransformerConfig:
    """A model config small enough that nothing here is slow."""
    base = {
        "vocab_size": 65,
        "d_model": 32,
        "n_layers": 1,
        "n_heads": 2,
        "max_seq_len": 64,
    }
    return TransformerConfig(**{**base, **kw})  # ty: ignore[missing-argument]


# ---- construction -------------------------------------------------------


def test_defaults_are_a_valid_config() -> None:
    """The shipped defaults must themselves be coherent."""
    # if the defaults do not survive their own validation, every example in
    # the repo starts with a workaround.
    TrainConfig()


def test_is_frozen() -> None:
    cfg = TrainConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.lr = 1.0  # ty: ignore[invalid-assignment]


def test_replace_produces_a_variant() -> None:
    """Ablations go through dataclasses.replace, so it has to work."""
    cfg = TrainConfig()
    other = dataclasses.replace(cfg, lr=1e-3)
    assert other.lr == 1e-3
    assert cfg.lr != 1e-3
    assert other.seed == cfg.seed


# ---- block_size vs the model's context ----------------------------------


def test_block_size_may_equal_max_seq_len() -> None:
    """The boundary is legal: a window exactly the size of the context."""
    TrainConfig(model=small_model(max_seq_len=64), block_size=64)


def test_block_size_above_max_seq_len_raises() -> None:
    with pytest.raises(ValueError):
        TrainConfig(model=small_model(max_seq_len=64), block_size=65)


def test_block_size_message_names_both_numbers() -> None:
    # "invalid block_size" sends you looking in the wrong file. the message
    # has to carry the model's limit too.
    with pytest.raises(ValueError) as exc:
        TrainConfig(model=small_model(max_seq_len=64), block_size=128)
    assert "128" in str(exc.value)
    assert "64" in str(exc.value)


# ---- schedule -----------------------------------------------------------


def test_warmup_may_equal_max_steps() -> None:
    TrainConfig(max_steps=100, warmup_steps=100)


def test_warmup_above_max_steps_raises() -> None:
    """Otherwise the lr never leaves the ramp and the curve is a lie."""
    with pytest.raises(ValueError) as exc:
        TrainConfig(max_steps=100, warmup_steps=101)
    assert "101" in str(exc.value)
    assert "100" in str(exc.value)


def test_zero_warmup_is_allowed() -> None:
    """No warmup is a choice, not an error."""
    TrainConfig(warmup_steps=0)


# ---- numeric ranges -----------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "field_name"),
    [
        ({"max_steps": 0, "warmup_steps": 0}, "max_steps"),
        ({"lr": 0.0}, "lr"),
        ({"lr": -1e-4}, "lr"),
        ({"grad_clip": 0.0}, "grad_clip"),
        ({"weight_decay": -0.1}, "weight_decay"),
        ({"batch_size": 0}, "batch_size"),
        ({"block_size": 0}, "block_size"),
        ({"eval_interval": 0}, "eval_interval"),
        ({"eval_iters": 0}, "eval_iters"),
        ({"warmup_steps": -1}, "warmup_steps"),
    ],
)
def test_rejects_out_of_range(kwargs: dict[str, object], field_name: str) -> None:
    """Each case violates exactly one rule, so the message must name it."""
    with pytest.raises(ValueError) as exc:
        TrainConfig(**kwargs)  # ty: ignore[invalid-argument-type]
    assert field_name in str(exc.value)


@pytest.mark.parametrize("value", [0.0, 0.5, 0.999])
def test_weight_decay_zero_and_positive_allowed(value: float) -> None:
    TrainConfig(weight_decay=value)


@pytest.mark.parametrize("betas", [(0.0, 0.95), (0.9, 1.0), (-0.1, 0.95), (0.9, 1.5)])
def test_betas_outside_the_open_unit_interval_raise(
    betas: tuple[float, float],
) -> None:
    with pytest.raises(ValueError):
        TrainConfig(betas=betas)


@pytest.mark.parametrize("ratio", [0.0, 0.5, 1.0])
def test_min_lr_ratio_bounds_are_inclusive(ratio: float) -> None:
    """0.0 decays to nothing, 1.0 is a constant lr. Both are real choices."""
    TrainConfig(min_lr_ratio=ratio)


@pytest.mark.parametrize("ratio", [-0.01, 1.01])
def test_min_lr_ratio_outside_bounds_raises(ratio: float) -> None:
    with pytest.raises(ValueError):
        TrainConfig(min_lr_ratio=ratio)


@pytest.mark.parametrize("frac", [0.0, 1.0, -0.1, 1.5])
def test_train_frac_must_be_strictly_inside_zero_one(frac: float) -> None:
    with pytest.raises(ValueError):
        TrainConfig(train_frac=frac)


# ---- to_dict ------------------------------------------------------------


def test_to_dict_carries_the_version() -> None:
    assert TrainConfig().to_dict()["version"] == VERSION


def test_to_dict_covers_every_field() -> None:
    """A new field that never reaches to_dict is a run you cannot reproduce.

    Derived from dataclasses.fields rather than a hand-written list, so
    adding a field to the config fails this test until it is serialised.
    """
    payload = TrainConfig().to_dict()
    for f in dataclasses.fields(TrainConfig):
        assert f.name in payload, f"{f.name} is missing from to_dict()"


def test_to_dict_nests_the_model_as_plain_data() -> None:
    payload = TrainConfig().to_dict()
    assert isinstance(payload["model"], dict)
    assert payload["model"]["d_model"] == TrainConfig().model.d_model


def test_to_dict_is_unchanged_by_a_yaml_round_trip() -> None:
    """to_dict must already be plain data, not merely dumpable.

    safe_dump accepts a tuple and writes it as a list, so a tuple left in
    the payload does not fail here — it comes back a different type, and
    to_dict() and load(save()) then quietly disagree. Comparing before and
    after is what pins that down.
    """
    payload = TrainConfig().to_dict()
    assert yaml.safe_load(yaml.safe_dump(payload)) == payload


# ---- from_dict ----------------------------------------------------------


def test_round_trip_through_dict() -> None:
    cfg = TrainConfig(seed=7, lr=1e-3, model=small_model(), block_size=32)
    assert TrainConfig.from_dict(cfg.to_dict()) == cfg


def test_round_trip_restores_betas_as_a_tuple() -> None:
    """yaml gives a list back; a naive rebuild leaves [0.9, 0.95] != (0.9, 0.95).

    Two values that look identical on screen and compare unequal, which is
    why this gets its own test instead of hiding inside the round-trip.
    """
    restored = TrainConfig.from_dict(TrainConfig().to_dict())
    assert isinstance(restored.betas, tuple)


def test_round_trip_restores_the_model_type() -> None:
    restored = TrainConfig.from_dict(TrainConfig().to_dict())
    assert isinstance(restored.model, TransformerConfig)


def test_from_dict_rejects_a_missing_version() -> None:
    payload = TrainConfig().to_dict()
    del payload["version"]
    with pytest.raises(ValueError, match="version"):
        TrainConfig.from_dict(payload)


def test_from_dict_rejects_a_different_version() -> None:
    payload = TrainConfig().to_dict()
    payload["version"] = VERSION + 1
    with pytest.raises(ValueError) as exc:
        TrainConfig.from_dict(payload)
    assert str(VERSION + 1) in str(exc.value)
    assert str(VERSION) in str(exc.value)


def test_from_dict_still_validates() -> None:
    """A hand-edited file must not get past the rules the constructor applies."""
    payload = TrainConfig().to_dict()
    payload["lr"] = -1.0
    with pytest.raises(ValueError):
        TrainConfig.from_dict(payload)


# ---- save / load --------------------------------------------------------


def test_save_load_round_trip(tmp_path: Path) -> None:
    cfg = TrainConfig(seed=99, max_steps=500, warmup_steps=50)
    path = tmp_path / "config.yaml"
    cfg.save(path)
    assert TrainConfig.load(path) == cfg


def test_saved_file_is_readable_yaml(tmp_path: Path) -> None:
    """The point of yaml over pickle is that a human can read the run."""
    path = tmp_path / "config.yaml"
    TrainConfig().save(path)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert loaded["version"] == VERSION
    assert loaded["seed"] == TrainConfig().seed


def test_load_rejects_a_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError):
        TrainConfig.load(path)
