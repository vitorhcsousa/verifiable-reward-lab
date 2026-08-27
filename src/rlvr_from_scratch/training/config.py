"""One frozen object holding every hyperparameter that moves the loss curve."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import yaml

from rlvr_from_scratch.model.transformer import TransformerConfig

if TYPE_CHECKING:
    from pathlib import Path

# bumped only when the meaning of an existing field changes, not when one is
# added. same contract as _FORMAT_VERSION in tokenizer/char.py.
VERSION = 1


def _default_model() -> TransformerConfig:
    """Model shape for the shakespeare run: ~5M non-embedding params."""
    # vocab_size 65 is tinyshakespeare's character vocabulary, written out
    # rather than derived: a config has to describe the model on its own.
    return TransformerConfig(
        vocab_size=65,
        d_model=256,
        n_layers=6,
        n_heads=8,
        max_seq_len=256,
    )


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for one training run. Vary it with dataclasses.replace."""

    # identity of the run
    seed: int = 1337
    corpus: str = "shakespeare"
    """Key into fetch.SOURCES. Pins which data, not only which model."""

    # data
    batch_size: int = 32
    block_size: int = 128
    """Context length sampled per window. Must fit within model.max_seq_len."""
    train_frac: float = 0.9

    # model
    model: TransformerConfig = field(default_factory=_default_model)

    # optimisation
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # schedule
    max_steps: int = 2000
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1
    """Cosine floor as a fraction of lr. 0.0 decays to nothing, 1.0 is flat."""

    # evaluation
    eval_interval: int = 100
    eval_iters: int = 20

    # runtime
    device: str = "cpu"
    """Part of the run's identity: the same config on mps and on cpu does
    not produce the same numbers."""

    def __post_init__(self) -> None:
        """Reject a config that describes a run which cannot happen.

        Raises:
            ValueError: On any field out of range, or any pair of fields
                that contradict each other.
        """
        # a window longer than the context does not fail in get_batch, it
        # fails deep inside forward, in a message naming neither number.
        if self.block_size > self.model.max_seq_len:
            msg = (
                f"block_size {self.block_size} exceeds the model's "
                f"max_seq_len {self.model.max_seq_len}"
            )
            raise ValueError(msg)

        if self.max_steps <= 0:
            msg = f"max_steps must be > 0, got {self.max_steps}"
            raise ValueError(msg)

        if self.warmup_steps < 0:
            msg = f"warmup_steps must be >= 0, got {self.warmup_steps}"
            raise ValueError(msg)

        # equal is allowed: an all-warmup run is a choice. longer is not.
        if self.warmup_steps > self.max_steps:
            msg = f"warmup_steps {self.warmup_steps} exceeds max_steps {self.max_steps}"
            raise ValueError(msg)

        for name, value in (("lr", self.lr), ("grad_clip", self.grad_clip)):
            if value <= 0:
                msg = f"{name} must be > 0, got {value}"
                raise ValueError(msg)

        if self.weight_decay < 0:
            msg = f"weight_decay must be >= 0, got {self.weight_decay}"
            raise ValueError(msg)

        # betas exclusive at both ends, min_lr_ratio inclusive at both
        for i, beta in enumerate(self.betas):
            if not 0.0 < beta < 1.0:
                msg = f"betas[{i}] must be strictly inside (0, 1), got {beta}"
                raise ValueError(msg)

        if not 0.0 <= self.min_lr_ratio <= 1.0:
            msg = f"min_lr_ratio must be in [0, 1], got {self.min_lr_ratio}"
            raise ValueError(msg)

        if not 0.0 < self.train_frac < 1.0:
            msg = f"train_frac must be strictly inside (0, 1), got {self.train_frac}"
            raise ValueError(msg)

        for name, value in (
            ("batch_size", self.batch_size),
            ("block_size", self.block_size),
            ("eval_interval", self.eval_interval),
            ("eval_iters", self.eval_iters),
        ):
            if value <= 0:
                msg = f"{name} must be > 0, got {value}"
                raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Plain-data view of the config, ready to be written as YAML.

        Returns:
            Every field, model flattened to a dict, plus a "version" key.
            Survives a YAML round trip unchanged, not merely dumpable.
        """
        # asdict recurses into TransformerConfig, so a new field arrives
        # here without being listed by hand
        payload: dict[str, Any] = {"version": VERSION}
        payload.update(dataclasses.asdict(self))
        # safe_dump writes a tuple as a list, so it would come back a
        # different type and to_dict() and load(save()) would disagree
        payload["betas"] = list(self.betas)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainConfig:
        """Rebuild a config from to_dict output.

        Args:
            payload: A mapping as produced by to_dict.

        Returns:
            The reconstructed config, validated by the constructor.

        Raises:
            ValueError: On a missing or mismatched version, an unknown key,
                or values that fail validation.
        """
        # check rather than trust: the alternative is a TypeError thousands
        # of lines from the file that caused it
        if "version" not in payload:
            msg = f"config payload has no 'version'; this code writes {VERSION}"
            raise ValueError(msg)

        found = payload["version"]
        if found != VERSION:
            msg = f"config version {found} cannot be read by version {VERSION}"
            raise ValueError(msg)

        # a renamed or stale key must fail rather than be silently replaced
        # by a default
        names = {f.name for f in dataclasses.fields(cls)}
        data: dict[str, Any] = {k: v for k, v in payload.items() if k != "version"}
        unknown = sorted(set(data) - names)
        if unknown:
            msg = f"unknown config keys {unknown}; known are {sorted(names)}"
            raise ValueError(msg)

        model = data.get("model")
        if isinstance(model, dict):
            data["model"] = TransformerConfig(**model)

        # yaml hands back a list, and (0.9, 0.95) != [0.9, 0.95]
        betas = data.get("betas")
        if betas is not None:
            first, second = betas
            data["betas"] = (float(first), float(second))

        return cls(**data)

    def save(self, path: Path) -> None:
        """Write the config to `path` as YAML. Parent directory must exist."""
        # sort_keys=False keeps the field grouping above intact in the file
        text = yaml.safe_dump(self.to_dict(), sort_keys=False)
        path.write_text(text, encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> TrainConfig:
        """Read back a config written by save.

        Args:
            path: Path to a YAML file produced by save.

        Returns:
            The reconstructed config.

        Raises:
            ValueError: If the file is not a YAML mapping, or from_dict
                rejects its contents.
        """
        # safe_load, never load: the second constructs arbitrary objects
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))

        if not isinstance(payload, dict):
            msg = f"{path} is not a YAML mapping: parsed as {type(payload).__name__}"
            raise ValueError(msg)

        return cls.from_dict(payload)
