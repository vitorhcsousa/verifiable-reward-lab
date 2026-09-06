"""
Tests for the eval loop.

No model here: the loop takes a Generator, and a fake one that replays fixed
completions is enough to pin down everything the loop is responsible for -
prompting, truncating, verifying, counting and batching. What a real
checkpoint would say is not this file's question.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest

from rlvr_from_scratch.data.gsm8k import Example
from rlvr_from_scratch.evaluation.config import EvalConfig
from rlvr_from_scratch.evaluation.gsm8k_eval import (
    Generation,
    evaluate,
    peak_memory_mb,
    select,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

GOLDS = ("#### 56", "#### 115", "#### 18")
EXAMPLES = tuple(
    Example(question=f"question {i}", answer=g) for i, g in enumerate(GOLDS)
)


class FakeGenerator:
    """Replays fixed completions and records the prompts it was handed."""

    def __init__(self, completions: Sequence[str], tokens: int = 10) -> None:
        self.completions = list(completions)
        self.tokens = tokens
        self.seen: list[str] = []
        self.calls = 0

    def generate(self, prompts: Sequence[str]) -> list[Generation]:
        self.calls += 1
        self.seen.extend(prompts)
        out = self.completions[: len(prompts)]
        self.completions = self.completions[len(prompts) :]
        return [Generation(text=t, n_new_tokens=self.tokens) for t in out]


def run_eval(completions: Sequence[str], batch_size: int = 8) -> tuple:
    gen = FakeGenerator(completions)
    result, records = evaluate(EXAMPLES, gen, n_shots=4, batch_size=batch_size)
    return result, records, gen


# ---- counting -----------------------------------------------------------


def test_counts_the_three_outcomes_separately() -> None:
    result, _, _ = run_eval(["#### 56", "#### 999", "I do not know"])
    assert (result.n_correct, result.n_incorrect, result.n_no_answer) == (1, 1, 1)
    assert result.n == 3


def test_accuracy_excludes_no_answer_from_the_numerator_only() -> None:
    """No-answer is not correct, and it is not wrong either - but it stays
    in the denominator, or abstaining would raise accuracy."""
    result, _, _ = run_eval(["#### 56", "no idea", "no idea"])
    assert result.accuracy == pytest.approx(1 / 3)
    assert result.no_answer_rate == pytest.approx(2 / 3)


def test_all_correct() -> None:
    result, _, _ = run_eval(["#### 56", "#### 115", "#### 18"])
    assert result.accuracy == 1.0


def test_throughput_is_reported() -> None:
    result, _, _ = run_eval(["#### 56", "#### 115", "#### 18"])
    assert result.n_new_tokens == 30
    assert result.tokens_per_s > 0
    assert result.wall_clock_s > 0


def test_summary_payload_carries_both_rates() -> None:
    result, _, _ = run_eval(["#### 56", "#### 999", "no idea"])
    payload = result.to_dict()
    assert payload["accuracy"] == pytest.approx(1 / 3)
    assert payload["no_answer_rate"] == pytest.approx(1 / 3)
    assert payload["n"] == 3


# ---- what the loop does to a completion ---------------------------------


def test_completion_is_truncated_before_verifying() -> None:
    """Without the cut, the invented question's answer would be scored."""
    raw = "#### 56\n\nQuestion: something else\nAnswer: #### 999"
    result, records, _ = run_eval([raw, "#### 115", "#### 18"])
    assert result.n_correct == 3
    assert records[0]["completion"] == "#### 56"


def test_records_hold_what_the_failure_analysis_needs() -> None:
    _, records, _ = run_eval(["#### 999", "#### 115", "no idea"])
    assert records[0] == {
        "question": "question 0",
        "gold": "56",
        "extracted": "999",
        "outcome": "incorrect",
        "completion": "#### 999",
    }
    assert records[2]["extracted"] is None
    assert records[2]["outcome"] == "no_answer"


def test_prompts_carry_the_shots_and_the_question() -> None:
    _, _, gen = run_eval(["#### 56", "#### 115", "#### 18"])
    assert len(gen.seen) == 3
    assert gen.seen[0].endswith("Question: question 0\nAnswer: ")
    assert "muffins" in gen.seen[0]  # the first exemplar


# ---- batching -----------------------------------------------------------


@pytest.mark.parametrize(("batch_size", "calls"), [(1, 3), (2, 2), (3, 1), (99, 1)])
def test_batch_size_changes_the_calls_not_the_result(
    batch_size: int, calls: int
) -> None:
    result, records, gen = run_eval(["#### 56", "#### 115", "#### 18"], batch_size)
    assert (gen.calls, result.n_correct, len(records)) == (calls, 3, 3)


def test_a_generator_returning_the_wrong_count_fails_loudly() -> None:
    """Silently zipping to the shorter list would drop examples from the
    denominator and quietly change the metric."""
    gen = FakeGenerator(["#### 56"])
    with pytest.raises(ValueError, match="argument 2 is shorter"):
        evaluate(EXAMPLES, gen, n_shots=0, batch_size=8)


# ---- selection ----------------------------------------------------------


class FakeSplit:
    def __init__(self) -> None:
        self.dev = tuple(Example(f"dev {i}", "#### 1") for i in range(10))
        self.held_out = tuple(Example(f"held {i}", "#### 1") for i in range(5))


@pytest.fixture
def fake_split(monkeypatch: pytest.MonkeyPatch) -> FakeSplit:
    split = FakeSplit()
    monkeypatch.setattr(
        "rlvr_from_scratch.evaluation.gsm8k_eval.load",
        lambda *_, **__: split,
    )
    return split


def test_select_takes_a_prefix_of_dev(fake_split: FakeSplit) -> None:
    """A prefix, so growing n_examples never moves an example out of the
    set the previous number was measured on."""
    small = select(dataclasses.replace(EvalConfig(), n_examples=3))
    large = select(dataclasses.replace(EvalConfig(), n_examples=6))
    assert small == fake_split.dev[:3]
    assert large[:3] == small


def test_select_reads_held_out_only_when_asked(fake_split: FakeSplit) -> None:
    config = dataclasses.replace(
        EvalConfig(), split="held_out", confirm_held_out=True, n_examples=2
    )
    assert select(config) == fake_split.held_out[:2]


def test_select_rejects_more_than_the_split_holds(fake_split: FakeSplit) -> None:
    with pytest.raises(ValueError, match="config asks for"):
        select(dataclasses.replace(EvalConfig(), n_examples=11))


# ---- compute reality check ----------------------------------------------


def test_peak_memory_is_positive() -> None:
    """P2c writes this into Protocol §7; a zero would mean the units are
    wrong on this platform, not that the run was free."""
    assert peak_memory_mb() > 0
