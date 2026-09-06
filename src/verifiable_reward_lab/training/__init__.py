"""Training the custom model: the run config, the objective, and the loop."""

from __future__ import annotations

from verifiable_reward_lab.training.config import VERSION, TrainConfig
from verifiable_reward_lab.training.grpo_config import GRPOConfig
from verifiable_reward_lab.training.losses import cross_entropy_loss
from verifiable_reward_lab.training.sft_config import SFTConfig
from verifiable_reward_lab.training.trainer import (
    TrainResult,
    configure_optimizer,
    estimate_loss,
    lr_at,
    resolve_device,
    set_seed,
    train,
)

# VERSION here is TrainConfig's. SFTConfig and GRPOConfig carry their own,
# in their own modules, because they version independently.
__all__ = [
    "VERSION",
    "GRPOConfig",
    "SFTConfig",
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
