"""Training the custom model: the run config, the objective, and the loop."""

from __future__ import annotations

from rlvr_from_scratch.training.config import VERSION, TrainConfig
from rlvr_from_scratch.training.losses import cross_entropy_loss
from rlvr_from_scratch.training.trainer import (
    TrainResult,
    configure_optimizer,
    estimate_loss,
    lr_at,
    resolve_device,
    set_seed,
    train,
)

__all__ = [
    "VERSION",
    "TrainConfig",
    "TrainResult",
    "configure_optimizer",
    "cross_entropy_loss",
    "estimate_loss",
    "lr_at",
    "resolve_device",
    "set_seed",
    "train",
]
