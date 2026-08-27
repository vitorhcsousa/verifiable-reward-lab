"""
Everything that decides what the loss curve looks like, in one frozen object.

The rule this file enforces: no hyperparameter that moves the curve lives
anywhere else. Not in a function default, not in a notebook cell, not in a
flag. trainer.py reads from here and nowhere else, so a run is described
entirely by one object that can be written to disk beside its metrics.

Defaults here are not "hidden defaults" — they sit on one line each, under
version control, and go out with to_dict(). What is banned is a default
buried in a signature, where nothing records it and nothing reproduces it.

Two facts have to travel together or the run stops being reproducible: the
model's vocab_size and the corpus it was trained on. This config pins both —
the corpus by the key fetch.py downloads it under, whose bytes are pinned by
sha256 over there. A config plus a clean clone is the whole run.
"""

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
    # vocab_size 65 is tinyshakespeare's character vocabulary. it is written
    # out here rather than derived because a config has to describe the model
    # on its own; __post_init__ and train() are what stop it drifting from
    # the corpus it claims to describe.
    return TransformerConfig(
        vocab_size=65,
        d_model=256,
        n_layers=6,
        n_heads=8,
        max_seq_len=256,
    )


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for one training run.

    Frozen for the same reason Corpus is: if this can be mutated after being
    logged, the config in the log and the run that actually happened can
    disagree, and nothing will say so. Vary it with dataclasses.replace.
    """

    # ---- identity of the run --------------------------------------------
    seed: int = 1337
    corpus: str = "shakespeare"
    """Key into fetch.SOURCES. Pins which data, not only which model."""

    # ---- data ------------------------------------------------------------
    batch_size: int = 32
    block_size: int = 128
    """Context length sampled per window. Must fit within model.max_seq_len."""
    train_frac: float = 0.9

    # ---- model -----------------------------------------------------------
    model: TransformerConfig = field(default_factory=_default_model)

    # ---- optimisation ----------------------------------------------------
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # ---- schedule --------------------------------------------------------
    max_steps: int = 2000
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1
    """Cosine floor as a fraction of lr. 0.0 decays to nothing, 1.0 is flat."""

    # ---- evaluation ------------------------------------------------------
    eval_interval: int = 100
    eval_iters: int = 20

    # ---- runtime ---------------------------------------------------------
    device: str = "cpu"
    """Part of the run's identity, not an execution detail: the same config
    on mps and on cpu does not produce the same numbers."""

    # =====================================================================
    # Validation
    # =====================================================================

    def __post_init__(self) -> None:
        """Reject a config that describes a run which cannot happen.

        Raises:
            ValueError: On any field out of range, or any pair of fields
                that contradict each other. Messages name the offending
                value, and both values when two of them disagree.
        """
        # =========================================
        # 1. The window has to fit the model
        # =========================================
        # This is the one that actually bites. get_batch happily samples a
        # window longer than the context; it blows up deep inside forward,
        # in a message naming neither number. Both go in ours.
        if self.block_size > self.model.max_seq_len:
            msg = (
                f"block_size {self.block_size} exceeds the model's "
                f"max_seq_len {self.model.max_seq_len}: a window cannot be "
                f"longer than the context it is fed into"
            )
            raise ValueError(msg)

        # =========================================
        # 2. The schedule has to be coherent
        # =========================================
        if self.max_steps <= 0:
            msg = f"max_steps must be > 0, got {self.max_steps}"
            raise ValueError(msg)

        if self.warmup_steps < 0:
            msg = f"warmup_steps must be >= 0, got {self.warmup_steps}"
            raise ValueError(msg)

        # equal is allowed on purpose: an all-warmup run is a choice. longer
        # than the run is not — the lr never leaves the ramp, and the curve
        # then measures the warmup rather than the schedule.
        if self.warmup_steps > self.max_steps:
            msg = (
                f"warmup_steps {self.warmup_steps} exceeds max_steps "
                f"{self.max_steps}: the lr would never leave the ramp"
            )
            raise ValueError(msg)

        # =========================================
        # 3. The optimiser values have to be in range
        # =========================================
        for name, value in (("lr", self.lr), ("grad_clip", self.grad_clip)):
            if value <= 0:
                msg = f"{name} must be > 0, got {value}"
                raise ValueError(msg)

        if self.weight_decay < 0:
            msg = f"weight_decay must be >= 0, got {self.weight_decay}"
            raise ValueError(msg)

        # two kinds of bound, one line apart: betas are exclusive at both
        # ends (beta=1.0 is an average that never forgets, beta=0.0 is no
        # average at all), min_lr_ratio is inclusive at both.
        for i, beta in enumerate(self.betas):
            if not 0.0 < beta < 1.0:
                msg = f"betas[{i}] must be strictly inside (0, 1), got {beta}"
                raise ValueError(msg)

        if not 0.0 <= self.min_lr_ratio <= 1.0:
            msg = f"min_lr_ratio must be in [0, 1], got {self.min_lr_ratio}"
            raise ValueError(msg)

        # =========================================
        # 4. The data and eval settings have to be usable
        # =========================================
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

    # =====================================================================
    # Serialisation
    # =====================================================================

    def to_dict(self) -> dict[str, Any]:
        """Plain-data view of the config, ready to be written as YAML.

        Returns:
            Every field, with the nested model config flattened to a dict,
            plus a "version" key. The result survives a YAML round trip
            unchanged — not merely dumpable.
        """
        # asdict recurses into TransformerConfig for free, so a field added
        # to either config arrives here without being listed by hand. a
        # hand-written dict is exactly how a field silently stops being
        # recorded.
        payload: dict[str, Any] = {"version": VERSION}
        payload.update(dataclasses.asdict(self))

        # safe_dump accepts a tuple and writes it as a list, so leaving one
        # here would not fail on the way out — it would come back a
        # different type, and to_dict() and load(save()) would then quietly
        # disagree about what a config is. Convert once, here.
        payload["betas"] = list(self.betas)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainConfig:
        """Rebuild a config from to_dict output.

        Args:
            payload: A mapping as produced by to_dict.

        Returns:
            The reconstructed config, validated exactly as the constructor
            validates it.

        Raises:
            ValueError: If the version is missing or does not match, if a
                key is not a field of this config, or if the values fail
                validation.
        """
        # =========================================
        # 1. Gate on the version before reading anything else
        # =========================================
        # Check rather than trust: the alternative failure is a TypeError
        # thousands of lines from the file that caused it.
        if "version" not in payload:
            msg = (
                f"config payload carries no 'version' key; this code writes "
                f"and reads version {VERSION}"
            )
            raise ValueError(msg)

        found = payload["version"]
        if found != VERSION:
            msg = (
                f"config version {found} cannot be read by this code, which "
                f"writes version {VERSION}"
            )
            raise ValueError(msg)

        # =========================================
        # 2. Reject keys this config does not have
        # =========================================
        # A renamed or stale key has to fail loudly instead of being dropped
        # on the floor and quietly replaced by a default.
        names = {f.name for f in dataclasses.fields(cls)}
        data: dict[str, Any] = {k: v for k, v in payload.items() if k != "version"}
        unknown = sorted(set(data) - names)
        if unknown:
            msg = (
                f"config payload has keys TrainConfig does not define: "
                f"{unknown}; known fields are {sorted(names)}"
            )
            raise ValueError(msg)

        # =========================================
        # 3. Rebuild the types YAML flattened
        # =========================================
        # model arrives as a dict and has to leave as a TransformerConfig,
        # or every attribute access downstream breaks.
        model = data.get("model")
        if isinstance(model, dict):
            data["model"] = TransformerConfig(**model)

        # YAML hands back a list. a naive rebuild gives [0.9, 0.95], and
        # (0.9, 0.95) != [0.9, 0.95] — two values that look identical on
        # screen and compare unequal.
        betas = data.get("betas")
        if betas is not None:
            first, second = betas
            data["betas"] = (float(first), float(second))

        # the constructor validates. from_dict does not get a private door.
        return cls(**data)

    def save(self, path: Path) -> None:
        """Write the config to `path` as YAML. Parent directory must exist."""
        # yaml rather than json because a run's config is read by people as
        # often as by code, and pyyaml is already a dependency.
        # sort_keys=False keeps the field grouping above intact in the file,
        # which is the whole reason the groups exist.
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
            ValueError: If the file is not a YAML mapping, or if from_dict
                rejects its contents.
        """
        # safe_load, never load: the second one constructs arbitrary Python
        # objects from a file on disk.
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))

        # Check the shape before trusting it. A list or a bare scalar in the
        # file is a ValueError naming what was found, not an AttributeError
        # somewhere later.
        if not isinstance(payload, dict):
            msg = (
                f"{path} does not hold a YAML mapping: it parsed as "
                f"{type(payload).__name__}"
            )
            raise ValueError(msg)

        # this method parses; it does not interpret. the version gate and
        # the betas fix live in exactly one place.
        return cls.from_dict(payload)
