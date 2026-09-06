"""
GRPO base hyperparameters, and the experiment plan they are spent on.

Frozen in W36, before a single GRPO number exists, because a hyperparameter
chosen after seeing a result is a hyperparameter fitted to the result.

The plan is bounded by compute, not by taste. The W36 reality check measured
3.3 s per completion on this hardware, and a run costs
`steps * questions_per_step * group_size` completions. At the defaults below
that is one run per ~4 h, which is what the whole block can pay 6-8 of. Hence:

- **RQ1 gets the three seeds.** SFT vs GRPO is the question the study is for,
  and it is the only place where the seed-to-seed spread is measured rather
  than assumed;
- **each ablation gets one seed**, against the base GRPO arm already trained.
  The RQ1 spread is the ruler: an ablation whose delta is smaller than it is
  reported as below the noise, and that is a result, not a failure;
- **arms are compared per question**, on the same frozen dev prefix, so a
  delta is a paired quantity. Two overlapping confidence intervals are not a
  conclusion - at n=200 the interval is +/-7 points, wide enough to hide any
  effect this study is likely to produce.

Ablation candidates are group size, KL coefficient, and one of learning rate
or rollouts per question. Format reward was the third candidate until the
W36 baseline came back at 200/200 on format with zero no-answers: there is no
headroom for it to move, so spending an ablation on it would measure zero.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import yaml

from verifiable_reward_lab.evaluation.config import DTYPES

if TYPE_CHECKING:
    from pathlib import Path

VERSION = 1

SEEDS = (1337, 1338, 1339)
"""The three seeds of RQ1. Both arms run all three - a delta between a
three-seed mean and a one-seed point is not a delta."""

ABLATION_SEEDS = SEEDS[:1]
"""One seed per ablation, by compute. The RQ1 spread supplies the scale."""

SECONDS_PER_COMPLETION = 3.3
"""Measured in W36: Qwen 2.5-0.5B, MPS, float32, batch 8, 81 tokens per
completion. A planning figure, not a claim about anyone else's hardware."""


@dataclass(frozen=True)
class GRPOConfig:
    """Hyperparameters for one GRPO run. Vary it with dataclasses.replace."""

    # identity
    seed: int = 1337
    init_from: str = "runs/sft/seed-1337"
    """The SFT run this starts from. GRPO is measured against its own
    starting point, so the two travel together."""

    # the group
    group_size: int = 8
    """Completions per question. The advantage is computed within the group,
    so this is what the estimator's variance rides on."""

    questions_per_step: int = 8
    steps: int = 60

    # the update
    lr: float = 1e-6
    kl_coef: float = 0.04
    clip_eps: float = 0.2
    grad_clip: float = 1.0

    # the rollout: sampled, unlike evaluation, which stays greedy
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 256

    # runtime
    device: str = "mps"
    dtype: str = "float32"

    @property
    def completions(self) -> int:
        """Total generations one run pays for."""
        return self.steps * self.questions_per_step * self.group_size

    @property
    def estimated_hours(self) -> float:
        """What that costs at the W36 measured throughput."""
        return self.completions * SECONDS_PER_COMPLETION / 3600

    def __post_init__(self) -> None:
        """Reject a run the protocol or the arithmetic does not allow.

        Raises:
            ValueError: On an unknown dtype, a group too small to have a
                spread, or any value out of range.
        """
        if self.dtype not in DTYPES:
            msg = f"dtype must be one of {list(DTYPES)}, got {self.dtype!r}"
            raise ValueError(msg)

        # a group of one has no within-group spread, so every advantage is
        # zero and the run silently learns nothing
        if self.group_size < 2:
            msg = (
                "group_size must be >= 2 for a within-group advantage, "
                f"got {self.group_size}"
            )
            raise ValueError(msg)

        for name, value in (
            ("questions_per_step", self.questions_per_step),
            ("steps", self.steps),
            ("max_new_tokens", self.max_new_tokens),
        ):
            if value <= 0:
                msg = f"{name} must be > 0, got {value}"
                raise ValueError(msg)

        for name, value in (("lr", self.lr), ("grad_clip", self.grad_clip)):
            if value <= 0:
                msg = f"{name} must be > 0, got {value}"
                raise ValueError(msg)

        if self.kl_coef < 0:
            msg = f"kl_coef must be >= 0, got {self.kl_coef}"
            raise ValueError(msg)

        if self.clip_eps <= 0:
            msg = f"clip_eps must be > 0, got {self.clip_eps}"
            raise ValueError(msg)

        if self.temperature <= 0:
            msg = (
                f"temperature must be > 0 for sampled rollouts, got {self.temperature}"
            )
            raise ValueError(msg)

        if not 0.0 < self.top_p <= 1.0:
            msg = f"top_p must be in (0, 1], got {self.top_p}"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Plain-data view, ready to be written as YAML."""
        payload: dict[str, Any] = {"version": VERSION}
        payload.update(dataclasses.asdict(self))
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GRPOConfig:
        """Rebuild a config from to_dict output.

        Raises:
            ValueError: On a missing or mismatched version, an unknown key,
                or values the constructor rejects.
        """
        if "version" not in payload:
            msg = f"config payload has no 'version'; this code writes {VERSION}"
            raise ValueError(msg)

        found = payload["version"]
        if found != VERSION:
            msg = f"config version {found} cannot be read by version {VERSION}"
            raise ValueError(msg)

        names = {f.name for f in dataclasses.fields(cls)}
        data = {k: v for k, v in payload.items() if k != "version"}
        unknown = sorted(set(data) - names)
        if unknown:
            msg = f"unknown config keys {unknown}; known are {sorted(names)}"
            raise ValueError(msg)

        return cls(**data)

    def save(self, path: Path) -> None:
        """Write the config to `path` as YAML. Parent directory must exist."""
        path.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> GRPOConfig:
        """Read back a config written by save.

        Raises:
            ValueError: If the file is not a YAML mapping, or from_dict
                rejects its contents.
        """
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            msg = f"{path} is not a YAML mapping: parsed as {type(payload).__name__}"
            raise ValueError(msg)
        return cls.from_dict(payload)
