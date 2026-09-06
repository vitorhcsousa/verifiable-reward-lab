"""
Tests for the GRPO base config and the experiment plan.

Two things are being protected here. The hyperparameters, which were frozen
before any GRPO number existed and must not quietly follow the results. And
the budget: a run that costs more than the block can pay is not an ambitious
experiment, it is an experiment that does not finish.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from verifiable_reward_lab.training.grpo_config import (
    ABLATION_SEEDS,
    SEEDS,
    VERSION,
    GRPOConfig,
)

COMMITTED = Path(__file__).resolve().parents[2] / "configs" / "grpo_base.yaml"


def test_committed_config_is_the_frozen_one() -> None:
    assert GRPOConfig.load(COMMITTED) == GRPOConfig()


def test_frozen_choices() -> None:
    c = GRPOConfig()
    assert (c.group_size, c.questions_per_step, c.steps) == (8, 8, 60)
    assert (c.lr, c.kl_coef, c.clip_eps) == (1e-6, 0.04, 0.2)


def test_rollouts_are_sampled_not_greedy() -> None:
    """Evaluation is greedy so the metric carries no sampling variance;
    rollouts must not be, or every completion in a group is identical and
    the within-group advantage is zero."""
    assert GRPOConfig().temperature == 1.0


def test_three_seeds_for_rq1_one_per_ablation() -> None:
    """The plan, in code: if these change, the compute budget changes with
    them and the change should be deliberate."""
    assert SEEDS == (1337, 1338, 1339)
    assert len(ABLATION_SEEDS) == 1
    assert ABLATION_SEEDS[0] in SEEDS


def test_run_cost_is_arithmetic_not_hope() -> None:
    c = GRPOConfig()
    assert c.completions == 60 * 8 * 8
    assert c.estimated_hours == pytest.approx(3.52, abs=0.05)


def test_the_committed_run_fits_the_block() -> None:
    """RQ1 is three seeds of this config. The block affords 6-8 runs total,
    so one run has to stay near four hours - this test is the tripwire for
    a `steps` bumped without doing the multiplication."""
    assert GRPOConfig().estimated_hours < 4.0
    assert GRPOConfig().estimated_hours * len(SEEDS) < 12.0


def test_group_of_one_is_rejected() -> None:
    """No spread inside the group means every advantage is zero and the run
    trains on nothing while looking healthy."""
    with pytest.raises(ValueError, match="group_size must be >= 2"):
        dataclasses.replace(GRPOConfig(), group_size=1)


def test_zero_kl_is_allowed() -> None:
    """Turning KL off is one of the planned ablations, not a broken config."""
    assert dataclasses.replace(GRPOConfig(), kl_coef=0.0).kl_coef == 0.0


def test_negative_kl_is_rejected() -> None:
    with pytest.raises(ValueError, match="kl_coef must be >= 0"):
        dataclasses.replace(GRPOConfig(), kl_coef=-0.01)


def test_zero_temperature_is_rejected() -> None:
    with pytest.raises(ValueError, match="temperature must be > 0"):
        dataclasses.replace(GRPOConfig(), temperature=0.0)


@pytest.mark.parametrize("top_p", [0.0, 1.5])
def test_top_p_out_of_range_is_rejected(top_p: float) -> None:
    with pytest.raises(ValueError, match="top_p must be in"):
        dataclasses.replace(GRPOConfig(), top_p=top_p)


def test_round_trip_through_yaml(tmp_path: Path) -> None:
    c = dataclasses.replace(GRPOConfig(), seed=1338, steps=10)
    path = tmp_path / "grpo.yaml"
    c.save(path)
    assert GRPOConfig.load(path) == c


def test_unknown_key_is_not_silently_defaulted() -> None:
    payload = GRPOConfig().to_dict() | {"beta": 0.1}
    with pytest.raises(ValueError, match="unknown config keys"):
        GRPOConfig.from_dict(payload)


def test_version_mismatch_is_rejected() -> None:
    payload = GRPOConfig().to_dict() | {"version": VERSION + 1}
    with pytest.raises(ValueError, match="cannot be read by version"):
        GRPOConfig.from_dict(payload)
