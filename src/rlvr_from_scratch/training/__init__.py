"""Training the custom model.

Phase 2 (training foundation): the frozen config that describes a run, the
objective, and the loop that turns one into the other. The custom
transformer exists to prove the mechanics are understood end to end — it is
not the model the RLVR study is run on.

`fetch` and the CLI are deliberately not re-exported here: both run as
entry points, and importing them from this __init__ would load the module
once as a package attribute and again as __main__.
"""

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
