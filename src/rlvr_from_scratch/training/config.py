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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rlvr_from_scratch.model.transformer import TransformerConfig

if TYPE_CHECKING:
    from pathlib import Path

# ===========================================================================
# SCAFFOLD — delete this block when the file is done.
#
# Every comment marked `TODO:` below is a step to write and then delete.
# Nothing else here is temporary: the docstrings and the fields are final.
#
#     grep -n "TODO:" src/rlvr_from_scratch/training/config.py
#     uv run pytest tests/training/test_config.py -q
#
# Order matters. __post_init__ first — while it raises, no TrainConfig can
# be built at all and the other 30 tests fail for a reason that is not
# theirs.
# ===========================================================================

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
        # TODO: 1. check the window fits the model, and raise naming both
        #    block_size must not exceed model.max_seq_len. this is the one
        #    that actually bites: get_batch samples a longer window without
        #    complaining, and it only blows up deep inside forward, in a
        #    message naming neither number. put both in yours.

        # TODO: 2. check the schedule is coherent
        #    max_steps > 0, warmup_steps >= 0, warmup_steps <= max_steps.
        #    a warmup longer than the run means the lr never leaves the ramp
        #    and the curve measures something else. let them be equal — an
        #    all-warmup run is a choice, not a mistake.

        # TODO: 3. check the optimiser values are in range
        #    lr > 0, grad_clip > 0, weight_decay >= 0, both betas strictly
        #    inside (0, 1), min_lr_ratio inside [0, 1].
        #    mind the two kinds of bound: betas exclusive at both ends,
        #    min_lr_ratio inclusive at both.

        # TODO: 4. check the data and eval settings are usable
        #    0 < train_frac < 1, and batch_size, block_size, eval_interval
        #    and eval_iters all strictly positive.

        raise NotImplementedError  # TODO: delete once the checks are in

    # =====================================================================
    # Serialisation
    # =====================================================================

    def to_dict(self) -> dict[str, Any]:
        """Plain-data view of the config, ready to be written as YAML.

        Returns:
            Every field, with the nested model config flattened to a dict,
            plus a "version" key. The result must survive a YAML round trip
            unchanged — not merely be dumpable.
        """
        # TODO: 1. flatten every field into one dict
        #    use dataclasses.asdict — it recurses into TransformerConfig for
        #    free, so a field added to either config arrives here without
        #    being listed by hand. a hand-written dict is exactly how a field
        #    silently stops being recorded.

        # TODO: 2. stamp the version on top
        #    so a file written today gets read deliberately, not hopefully.

        # TODO: 3. leave nothing behind that changes type in YAML
        #    the property to hit is
        #        yaml.safe_load(yaml.safe_dump(payload)) == payload
        #    safe_dump accepts a tuple and writes it as a list, so betas does
        #    not fail on the way out — it comes back a different type, and
        #    then to_dict() and load(save()) disagree about what a config is.

        raise NotImplementedError  # TODO: delete once the payload is built

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
        # TODO: 1. gate on the version before reading anything else
        #    missing or mismatched -> ValueError naming both versions.
        #    check rather than trust: the alternative failure is a TypeError
        #    thousands of lines from the file that caused it.

        # TODO: 2. rebuild the nested model config
        #    payload["model"] arrives as a dict and has to leave as a
        #    TransformerConfig, or every attribute access downstream breaks.

        # TODO: 3. convert betas back to a tuple
        #    YAML hands back a list. a naive rebuild gives [0.9, 0.95], and
        #    (0.9, 0.95) != [0.9, 0.95] — two values that look identical on
        #    screen and compare unequal.

        # TODO: 4. reject keys this config does not have
        #    a renamed or stale key must fail loudly instead of being dropped
        #    on the floor and quietly replaced by a default.

        raise NotImplementedError  # TODO: delete once the config is rebuilt

    def save(self, path: Path) -> None:
        """Write the config to `path` as YAML. Parent directory must exist."""
        # TODO: 1. dump to_dict() as YAML
        #    yaml rather than json because AGENTS.md pins
        #    experiments/<exp>/config.yaml and pyyaml is already a dependency.
        #    pass sort_keys=False — it keeps the field grouping above intact
        #    in the file, which is the whole reason the groups exist.

        raise NotImplementedError  # TODO: delete once the file is written

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
        # TODO: 1. parse with yaml.safe_load, never yaml.load
        #    the second one constructs arbitrary Python objects from a file
        #    on disk.

        # TODO: 2. check the shape before trusting it
        #    a list or a bare scalar in the file is a ValueError naming what
        #    was found, not an AttributeError somewhere later.

        # TODO: 3. hand off to from_dict
        #    the version gate and the betas fix live in exactly one place.
        #    this method parses; it does not interpret.

        raise NotImplementedError  # TODO: delete once the file is parsed
