"""
Tests for the GSM8K verifier.

Half of these are not "does it work" but "what exactly does it do when the
completion is ambiguous". Those cases are named for the behaviour they pin
down, because the verifier is the reward: the day a GRPO run starts scoring
suspiciously well, this file is what says whether the model found a hole
that was already known about.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from rlvr_from_scratch.rewards.verifier import (
    Outcome,
    extract,
    gold_value,
    parse_number,
    verify,
)

GOLD = "Natalia sold 48/2 = 24 clips in May.\n#### 72"


# ---- extraction precedence ----------------------------------------------


def test_marker_wins() -> None:
    assert extract("some reasoning with 13 in it\n#### 72") == 72


def test_last_marker_wins() -> None:
    """A model that restates its answer is scored on the final claim."""
    assert extract("#### 13\nwait, no.\n#### 72") == 72


def test_marker_takes_the_first_number_after_it() -> None:
    assert extract("#### 5 apples and 3 pears") == 5


def test_marker_ignores_currency_and_separators() -> None:
    assert extract("#### $1,000") == 1000


def test_marker_ignores_a_sentence_period() -> None:
    assert extract("#### 72.") == 72


def test_last_number_without_a_marker() -> None:
    assert extract("he had 13, then bought 59 more, so 72") == 72


def test_marker_beats_a_later_number() -> None:
    """The slot is a claim, and the claim is honoured even when a better
    number appears after it."""
    assert extract("#### 72\n(that is 72 clips in total, or 6 dozen)") == 72


# ---- extraction: the ambiguous cases ------------------------------------


def test_two_numbers_on_the_last_line_takes_the_second() -> None:
    """Known weakness, not a bug: with no marker the rule is positional."""
    assert extract("so the answer is 72 out of 100") == 100


def test_reasoning_past_the_conclusion_is_scored_on_the_last_number() -> None:
    """Known weakness: without a marker the verifier cannot tell a
    conclusion from an afterthought."""
    assert extract("the answer is 72. for reference, May alone was 24") == 24


def test_truncated_generation_is_scored_on_what_survived() -> None:
    assert extract("48 + 24 = 72, and then he sold another 1") == 1


def test_answer_in_words_is_no_answer() -> None:
    """Known weakness: no word-to-number mapping, deliberately."""
    assert extract("the answer is seventy-two") is None


def test_empty_completion_is_no_answer() -> None:
    assert extract("") is None


def test_marker_without_a_number_does_not_fall_back() -> None:
    """The model pointed at the slot and put nothing usable in it. Falling
    back to an earlier number would credit it for a number it disclaimed."""
    assert extract("48 + 24 = 72\n#### unknown") is None


# ---- equivalence --------------------------------------------------------


@pytest.mark.parametrize(
    ("token", "value"),
    [
        ("5", 5),
        ("5.0", 5),
        ("5.00", 5),
        ("+5", 5),
        ("-3", -3),
        ("$5", 5),
        ("50%", 50),
        ("1,000", 1000),
        ("1,000,000", 1000000),
        ("  7 ", 7),
        ("-$4", -4),
        ("2.5", Decimal("2.5")),
    ],
)
def test_parse_number_canonicalises(token: str, value: int | Decimal) -> None:
    assert parse_number(token) == value


def test_decimal_comma_abstains() -> None:
    """`1,5` is a European decimal, not a thousands group. Reading it as 15
    would be a silent wrong answer; None is an honest one."""
    assert parse_number("1,5") is None


def test_malformed_grouping_abstains() -> None:
    assert parse_number("1,00") is None


def test_comparison_has_no_tolerance() -> None:
    """Gold is always an integer, so near-misses are misses."""
    assert verify("#### 4.999", "#### 5").outcome is Outcome.INCORRECT
    assert verify("#### 5.000", "#### 5").outcome is Outcome.CORRECT


# ---- verify -------------------------------------------------------------


def test_correct() -> None:
    v = verify("48 + 24 = 72\n#### 72", GOLD)
    assert v.outcome is Outcome.CORRECT
    assert (v.extracted, v.gold) == (72, 72)


def test_incorrect() -> None:
    v = verify("48 + 24 = 71\n#### 71", GOLD)
    assert v.outcome is Outcome.INCORRECT
    assert (v.extracted, v.gold) == (71, 72)


def test_no_answer_is_not_incorrect() -> None:
    """RQ5 lives here: a policy that stops answering must be visible as
    something other than a wrong answer."""
    v = verify("I am not sure.", GOLD)
    assert v.outcome is Outcome.NO_ANSWER
    assert v.extracted is None
    assert v.gold == 72


def test_formatting_does_not_change_the_verdict() -> None:
    for completion in ("#### 72", "####72", "#### $72", "#### 72.0", "  #### 72  \n"):
        assert verify(completion, GOLD).outcome is Outcome.CORRECT


def test_gold_with_separators() -> None:
    assert verify("#### 1000", "#### 1,000").outcome is Outcome.CORRECT


def test_negative_gold() -> None:
    assert verify("#### -5", "#### -5").outcome is Outcome.CORRECT


# ---- the gold side ------------------------------------------------------


def test_gold_value_reads_the_marker() -> None:
    assert gold_value(GOLD) == 72


def test_gold_without_a_marker_raises() -> None:
    """A corpus without the marker is a different corpus - fail loudly
    rather than score every completion against a guess."""
    with pytest.raises(ValueError, match="no ####"):
        gold_value("Natalia sold 72 clips.")


def test_gold_with_an_unparseable_marker_raises() -> None:
    with pytest.raises(ValueError, match="is not a number"):
        gold_value("reasoning\n#### seventy-two")
