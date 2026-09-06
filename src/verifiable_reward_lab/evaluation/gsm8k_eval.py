"""
One evaluation run: prompt, generate, verify, count.

The primary metric is exact-match accuracy over a fixed prefix of a frozen
split, and `no_answer` is reported beside it rather than folded into wrong -
a policy that stops answering while its reward rises is RQ5, and a single
accuracy number would hide it.

Generation is greedy on purpose. The metric already carries seed variance
from training; adding sampling variance on top would mean a difference
between two arms could be neither, and every comparison would need more runs
to say anything. Sampling belongs to the rollout, not to the measurement.

`evaluate` does no I/O and knows nothing about transformers: it takes a
Generator, which the tests satisfy with a fake. Only `run` touches disk and
weights.
"""

from __future__ import annotations

import json
import resource
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from verifiable_reward_lab.data.gsm8k import load
from verifiable_reward_lab.evaluation.prompt import build_prompt, truncate_completion
from verifiable_reward_lab.rewards.verifier import Outcome, verify

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from verifiable_reward_lab.data.gsm8k import Example
    from verifiable_reward_lab.evaluation.config import EvalConfig


@dataclass(frozen=True)
class Generation:
    """One completion, plus what it cost to produce."""

    text: str
    n_new_tokens: int


class Generator(Protocol):
    """Whatever turns prompts into completions.

    A protocol rather than a class so the eval loop can be tested in
    milliseconds against a fake, and so the GRPO rollout can hand its own
    policy to the same loop in W38 without inheriting anything.
    """

    def generate(self, prompts: Sequence[str]) -> list[Generation]:
        """Complete each prompt, in order."""
        ...


@dataclass(frozen=True)
class EvalResult:
    """What one evaluation is, reduced to numbers. Written to summary.json."""

    n: int
    n_correct: int
    n_incorrect: int
    n_no_answer: int
    n_new_tokens: int
    wall_clock_s: float

    @property
    def accuracy(self) -> float:
        """The primary metric: correct / n, with no partial credit."""
        return self.n_correct / self.n

    @property
    def no_answer_rate(self) -> float:
        """Reported beside accuracy, never inside it."""
        return self.n_no_answer / self.n

    @property
    def tokens_per_s(self) -> float:
        return self.n_new_tokens / self.wall_clock_s if self.wall_clock_s else 0.0

    def to_dict(self) -> dict[str, float | int]:
        """Plain-data view, ready for json.dump."""
        return {
            "n": self.n,
            "n_correct": self.n_correct,
            "n_incorrect": self.n_incorrect,
            "n_no_answer": self.n_no_answer,
            "accuracy": self.accuracy,
            "no_answer_rate": self.no_answer_rate,
            "n_new_tokens": self.n_new_tokens,
            "wall_clock_s": self.wall_clock_s,
            "tokens_per_s": self.tokens_per_s,
        }


def peak_memory_mb() -> float:
    """Peak resident memory of this process, in MB.

    Deliberately process-level rather than per-accelerator: what the compute
    budget in Protocol §7 needs to know is whether a run fits on the machine,
    and an mps-only number would not answer that.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # linux reports kilobytes, macOS bytes
    return peak / 1024 / 1024 if sys.platform == "darwin" else peak / 1024


def evaluate(
    examples: Sequence[Example],
    generator: Generator,
    *,
    n_shots: int,
    batch_size: int,
) -> tuple[EvalResult, list[dict[str, Any]]]:
    """Score a generator on examples.

    Args:
        examples: The questions, in the order the split froze them.
        generator: Produces one completion per prompt.
        n_shots: Exemplars in the prompt.
        batch_size: Prompts per generate call.

    Returns:
        The aggregate result, and one record per example for the failure
        analysis in W41. Records keep the truncated completion, which is
        what was actually verified.
    """
    counts = dict.fromkeys(Outcome, 0)
    records: list[dict[str, Any]] = []
    n_new_tokens = 0

    started = time.perf_counter()
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        prompts = [build_prompt(e.question, n_shots) for e in batch]

        for example, generation in zip(batch, generator.generate(prompts), strict=True):
            completion = truncate_completion(generation.text)
            verdict = verify(completion, example.answer)

            counts[verdict.outcome] += 1
            n_new_tokens += generation.n_new_tokens
            extracted = None if verdict.extracted is None else str(verdict.extracted)
            records.append(
                {
                    "question": example.question,
                    "gold": str(verdict.gold),
                    "extracted": extracted,
                    "outcome": verdict.outcome.value,
                    "completion": completion,
                }
            )

    result = EvalResult(
        n=len(examples),
        n_correct=counts[Outcome.CORRECT],
        n_incorrect=counts[Outcome.INCORRECT],
        n_no_answer=counts[Outcome.NO_ANSWER],
        n_new_tokens=n_new_tokens,
        wall_clock_s=time.perf_counter() - started,
    )
    return result, records


def select(config: EvalConfig, data_dir: Path | None = None) -> tuple[Example, ...]:
    """The examples this config measures on.

    Raises:
        ValueError: The split holds fewer examples than the config asks for,
            or the corpus is missing.
    """
    split = load(data_dir, seed=config.seed) if data_dir else load(seed=config.seed)
    pool = split.dev if config.split == "dev" else split.held_out

    if config.n_examples > len(pool):
        msg = f"{config.split} holds {len(pool)} examples, config asks for {config.n_examples}"
        raise ValueError(msg)

    return pool[: config.n_examples]


def run(
    config: EvalConfig,
    out_dir: Path,
    *,
    data_dir: Path | None = None,
    verbose: bool = True,
) -> EvalResult:
    """Evaluate one model and write the run to disk.

    Writes `config.yaml`, `predictions.jsonl` and `summary.json` into
    out_dir: the config beside the numbers is the run that happened.

    Raises:
        ValueError: A config the data cannot satisfy.
    """
    # imported here so the eval loop, its tests and the GRPO rollout do not
    # pay for transformers when they never touch a checkpoint
    from verifiable_reward_lab.evaluation.hf import HFGenerator

    examples = select(config, data_dir)
    generator = HFGenerator(config)
    if verbose:
        print(f"{config.model_name} @ {generator.resolved_revision} on {config.device}")

    result, records = evaluate(
        examples,
        generator,
        n_shots=config.n_shots,
        batch_size=config.batch_size,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    config.save(out_dir / "config.yaml")
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    summary: dict[str, Any] = {
        "model_name": config.model_name,
        "model_revision_resolved": generator.resolved_revision,
        "split": config.split,
        "seed": config.seed,
        "n_shots": config.n_shots,
        "device": config.device,
        "dtype": config.dtype,
        "peak_memory_mb": peak_memory_mb(),
        **result.to_dict(),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return result
