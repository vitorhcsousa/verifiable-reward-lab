"""Task evaluation for GSM8K.

The metric, the prompt and the truncation rule live here; the verifier that
decides right from wrong lives in `verifiable_reward_lab.rewards`, because the
GRPO reward and this harness have to agree on it by construction rather than
by discipline.

`hf` is not re-exported: importing it pulls in transformers and a checkpoint,
which the eval loop's own tests have no use for.
"""

from __future__ import annotations

from verifiable_reward_lab.evaluation.config import EvalConfig
from verifiable_reward_lab.evaluation.gsm8k_eval import (
    EvalResult,
    Generation,
    Generator,
    evaluate,
    run,
    select,
)

__all__ = [
    "EvalConfig",
    "EvalResult",
    "Generation",
    "Generator",
    "evaluate",
    "run",
    "select",
]
