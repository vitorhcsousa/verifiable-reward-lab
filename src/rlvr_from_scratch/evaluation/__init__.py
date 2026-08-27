"""Evaluation harness.

Phase 3 (RLVR). Empty on purpose: the training loop's own loss estimate
lives in `training.trainer.estimate_loss`, because it is part of the loop.
What belongs here is the held-out task evaluation for GSM8K — a separate
job, run on a frozen split, with a metric fixed before any result is seen.
"""
