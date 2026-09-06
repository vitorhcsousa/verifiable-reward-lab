"""Verifier and format reward.

The verifier is the objective function of the study, so it lives in one
file, is made of pure functions, and is re-exported here for the eval
harness and the GRPO reward to share. Format reward is a GRPO
hyperparameter and is not written yet.
"""

from __future__ import annotations

from verifiable_reward_lab.rewards.verifier import (
    Outcome,
    Verdict,
    extract,
    gold_value,
    parse_number,
    verify,
)

__all__ = ["Outcome", "Verdict", "extract", "gold_value", "parse_number", "verify"]
