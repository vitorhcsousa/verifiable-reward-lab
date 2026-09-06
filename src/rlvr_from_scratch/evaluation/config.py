"""One frozen object holding every choice that moves the accuracy number."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import yaml

from rlvr_from_scratch.evaluation.prompt import EXEMPLARS

if TYPE_CHECKING:
    from pathlib import Path

VERSION = 1

DTYPES = ("float32", "bfloat16", "float16")
SPLITS = ("dev", "held_out")


@dataclass(frozen=True)
class EvalConfig:
    """What one evaluation run is. Vary it with dataclasses.replace."""

    # what is being measured on
    seed: int = 1337
    """Must match the split seed the protocol froze, or 'dev' means something
    else than it did last week."""

    split: str = "dev"
    n_examples: int = 200
    """A prefix of the split's frozen order, so growing it is inclusive: the
    first 200 stay the first 200."""

    confirm_held_out: bool = False
    """Held-out is untouched until the block closes. Reading it takes a
    second, deliberate flag, so it cannot happen by editing one word."""

    # what is being measured
    model_name: str = "Qwen/Qwen2.5-0.5B"
    model_revision: str = "main"
    """A commit sha belongs here. The first run reports the sha it resolved
    to in summary.json; paste it back and the model stops being a moving
    target."""

    n_shots: int = 4
    max_new_tokens: int = 256

    # how it is generated: greedy, so the metric carries no sampling
    # variance on top of seed variance. sampling belongs to training.
    batch_size: int = 8
    device: str = "mps"
    dtype: str = "float32"
    """Part of the run's identity: bfloat16 and float32 do not agree."""

    def __post_init__(self) -> None:
        """Reject a config that describes a measurement that should not run.

        Raises:
            ValueError: On an unknown split or dtype, a value out of range,
                or held-out without the explicit confirmation.
        """
        if self.split not in SPLITS:
            msg = f"split must be one of {list(SPLITS)}, got {self.split!r}"
            raise ValueError(msg)

        if self.split == "held_out" and not self.confirm_held_out:
            msg = (
                "the held-out split is frozen until the block closes; set "
                "confirm_held_out: true in the config to read it on purpose"
            )
            raise ValueError(msg)

        if self.dtype not in DTYPES:
            msg = f"dtype must be one of {list(DTYPES)}, got {self.dtype!r}"
            raise ValueError(msg)

        if not 0 <= self.n_shots <= len(EXEMPLARS):
            msg = f"n_shots must be in [0, {len(EXEMPLARS)}], got {self.n_shots}"
            raise ValueError(msg)

        for name, value in (
            ("n_examples", self.n_examples),
            ("max_new_tokens", self.max_new_tokens),
            ("batch_size", self.batch_size),
        ):
            if value <= 0:
                msg = f"{name} must be > 0, got {value}"
                raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Plain-data view, ready to be written as YAML."""
        payload: dict[str, Any] = {"version": VERSION}
        payload.update(dataclasses.asdict(self))
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvalConfig:
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
        text = yaml.safe_dump(self.to_dict(), sort_keys=False)
        path.write_text(text, encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> EvalConfig:
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
