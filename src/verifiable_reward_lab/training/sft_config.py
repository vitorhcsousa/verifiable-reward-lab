"""
The SFT baseline: every choice that decides what GRPO is compared against.

This is the reference arm of RQ1, so it is frozen in W36, before any GRPO
number exists. The base model's few-shot accuracy is *not* the baseline -
it is a sanity check on the harness. GRPO starts from these weights, so the
only honest comparison is against them.

Two decisions worth stating rather than burying:

- loss is computed on the answer tokens only. Training on the question as
  well teaches the model to write GSM8K questions, which nothing in the
  study asks it to do, and it dilutes the gradient that matters;
- the SFT model is evaluated at `n_shots: 0`. It was trained on the format,
  so showing it four examples measures the prompt rather than the training.
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


@dataclass(frozen=True)
class SFTConfig:
    """Hyperparameters for one SFT run. Vary it with dataclasses.replace."""

    # identity
    seed: int = 1337
    """Training seed. The split seed is separate and must not drift with it."""

    split_seed: int = 1337
    """Must equal the protocol's split seed, or the model trains on examples
    a later dev set will grade it on."""

    model_name: str = "Qwen/Qwen2.5-0.5B"
    model_revision: str = "060db6499f32faf8b98477b0a26969ef7d8b9987"
    """Pinned to the sha the W36 baseline run resolved, not to `main`."""

    # data
    max_seq_len: int = 640
    """Question plus answer. GSM8K answers are short; anything longer is a
    corpus that is not this one."""

    mask_prompt_loss: bool = True

    # optimisation
    epochs: int = 2
    lr: float = 1e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    batch_size: int = 8
    grad_accum: int = 2
    """Effective batch is batch_size * grad_accum, and that product is what
    the learning rate was chosen against."""

    # runtime
    device: str = "mps"
    dtype: str = "float32"
    """float32 because the baseline has to be reproducible across the two
    machines this study runs on, and low precision is where they diverge."""

    @property
    def effective_batch(self) -> int:
        """What the optimiser actually sees per step."""
        return self.batch_size * self.grad_accum

    def __post_init__(self) -> None:
        """Reject a run that cannot be the baseline the protocol names.

        Raises:
            ValueError: On an unknown dtype or any value out of range.
        """
        if self.dtype not in DTYPES:
            msg = f"dtype must be one of {list(DTYPES)}, got {self.dtype!r}"
            raise ValueError(msg)

        for name, value in (
            ("epochs", self.epochs),
            ("batch_size", self.batch_size),
            ("grad_accum", self.grad_accum),
            ("max_seq_len", self.max_seq_len),
        ):
            if value <= 0:
                msg = f"{name} must be > 0, got {value}"
                raise ValueError(msg)

        for name, value in (("lr", self.lr), ("grad_clip", self.grad_clip)):
            if value <= 0:
                msg = f"{name} must be > 0, got {value}"
                raise ValueError(msg)

        if self.weight_decay < 0:
            msg = f"weight_decay must be >= 0, got {self.weight_decay}"
            raise ValueError(msg)

        if not 0.0 <= self.warmup_ratio < 1.0:
            msg = f"warmup_ratio must be in [0, 1), got {self.warmup_ratio}"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Plain-data view, ready to be written as YAML."""
        payload: dict[str, Any] = {"version": VERSION}
        payload.update(dataclasses.asdict(self))
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SFTConfig:
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
    def load(cls, path: Path) -> SFTConfig:
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
