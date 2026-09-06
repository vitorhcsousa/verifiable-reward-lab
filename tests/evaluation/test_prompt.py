"""Tests for the frozen prompt and the truncation rule."""

from __future__ import annotations

import pytest

from rlvr_from_scratch.evaluation.prompt import (
    ANSWER_PREFIX,
    EXEMPLARS,
    QUESTION_PREFIX,
    build_prompt,
    truncate_completion,
)


def test_zero_shot_is_just_the_question() -> None:
    prompt = build_prompt("How many?", n_shots=0)
    assert prompt == f"{QUESTION_PREFIX}How many?\n{ANSWER_PREFIX}"


def test_shots_are_prepended_in_order() -> None:
    prompt = build_prompt("How many?", n_shots=4)
    positions = [prompt.index(q) for q, _ in EXEMPLARS]
    assert positions == sorted(positions)


def test_prompt_ends_open_for_the_model_to_continue() -> None:
    """A trailing space after `Answer:` and nothing else - the model has to
    write the answer, not choose between finishing a line and starting one."""
    assert build_prompt("How many?", n_shots=4).endswith(f"\n{ANSWER_PREFIX}")


def test_every_exemplar_shows_the_answer_format() -> None:
    """The exemplars exist to teach `#### <n>`; one without it teaches the
    model that the marker is optional."""
    for _, answer in EXEMPLARS:
        assert "\n#### " in answer


def test_exemplars_are_solvable_as_written() -> None:
    """Cheap guard against a typo in an invented example: the number after
    the marker must be the one the reasoning arrives at."""
    expected = ("56", "115", "18", "9")
    for (_, answer), value in zip(EXEMPLARS, expected, strict=True):
        assert answer.rsplit("#### ", 1)[1].strip() == value


def test_too_many_shots_is_rejected() -> None:
    with pytest.raises(ValueError, match="n_shots must be in"):
        build_prompt("How many?", n_shots=len(EXEMPLARS) + 1)


def test_negative_shots_is_rejected() -> None:
    with pytest.raises(ValueError, match="n_shots must be in"):
        build_prompt("How many?", n_shots=-1)


def test_truncation_cuts_the_invented_question() -> None:
    """The failure this rule exists for: a base model continues the pattern,
    and the verifier's last-#### rule would score the invented one."""
    raw = "He has 56 left.\n#### 56\n\nQuestion: A train leaves...\nAnswer: #### 99"
    assert truncate_completion(raw) == "He has 56 left.\n#### 56"


def test_truncation_leaves_a_clean_completion_alone() -> None:
    assert truncate_completion("96 - 40 = 56\n#### 56") == "96 - 40 = 56\n#### 56"


def test_truncation_of_empty_output() -> None:
    assert truncate_completion("") == ""
