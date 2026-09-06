"""
The GSM8K verifier: does this completion state the right final number?

This is not a utility. It is the objective function - every hole in it is
something GRPO is allowed to walk through, and a reward that can be earned
without solving the problem is the failure mode this study is looking for.
So the rules are written down here in full, and they are deliberately dumb:
predictable beats clever, because a clever verifier is one whose exploits
are hard to recognise in the failure analysis.

Extraction precedence, in order:

1. the text after the *last* `####` - the model claimed the answer slot, so
   that claim is honoured even if a better number appears elsewhere. The
   first number in that text is the answer;
2. otherwise, the *last* number anywhere in the completion;
3. otherwise NO_ANSWER, which is a third outcome and never counted as wrong.

Equivalence, applied to one extracted token:

- `$` and `%` are dropped anywhere in the token: `$1,000` and `50%` are the
  same claims as `1000` and `50`;
- a leading `+` is dropped, a leading `-` is kept;
- commas are dropped only when they group thousands (`1,000,000`); any other
  comma (`1,5`) leaves the token unparseable rather than silently meaning
  something. English decimal commas are not a GSM8K phenomenon, and guessing
  is worse than abstaining;
- comparison is exact, on Decimal. `5.0`, `5.00` and `5` are equal; `4.999`
  is not. No tolerance: on 2026-09-06 every gold answer in both splits was an
  integer (7473 train, 1319 test; 0 decimals, 93 with thousands separators,
  5 negative), so a tolerance would only buy the model credit for being
  nearly right.

Known weaknesses are listed at the bottom of this docstring rather than
fixed, because each fix makes the verifier less predictable:

- an answer written in words ("seventy-two") is NO_ANSWER;
- with no `####`, a completion that reasons past its own conclusion is
  scored on its last number, not its intended one;
- a completion whose last line holds two numbers is scored on the second;
- units are not checked at all: "5 apples" and "5 hours" both verify as 5.

Pure functions, no model, no I/O: usable by the eval harness and by the
GRPO reward without either of them owning it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum

ANSWER_MARKER = "####"

# one numeric token: optional sign, optional $, digits with optional
# thousands groups, optional decimal part, optional %. a trailing sentence
# period is not part of the number, which is why the decimal part requires
# a digit after the dot.
NUMBER = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?%?")

THOUSANDS = re.compile(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?")


class Outcome(str, Enum):
    """The three ways a completion can end up.

    NO_ANSWER is separate from INCORRECT on purpose: a policy that stops
    answering while its reward goes up is RQ5, and folding the two together
    would hide exactly that.
    """

    CORRECT = "correct"
    INCORRECT = "incorrect"
    NO_ANSWER = "no_answer"


@dataclass(frozen=True)
class Verdict:
    """The outcome plus what it was derived from.

    The extracted value travels with the verdict so the W41 failure analysis
    can ask *what did the model actually say* without re-running extraction
    against a verifier that may have changed by then.
    """

    outcome: Outcome
    extracted: Decimal | None
    gold: Decimal


def parse_number(token: str) -> Decimal | None:
    """Canonicalise one numeric token.

    Args:
        token: A single number as it appeared in the text, e.g. `$1,000`.

    Returns:
        Its value, or None if the token is not a number under the rules in
        the module docstring.
    """
    t = token.strip().replace("$", "").replace("%", "").removeprefix("+")

    if "," in t:
        if not THOUSANDS.fullmatch(t):
            return None  # abstain rather than guess what the comma meant
        t = t.replace(",", "")

    try:
        return Decimal(t)
    except InvalidOperation:
        return None


def extract(completion: str) -> Decimal | None:
    """Pull the answer out of a completion, by the documented precedence."""
    if ANSWER_MARKER in completion:
        claimed = completion.rsplit(ANSWER_MARKER, 1)[1]
        m = NUMBER.search(claimed)
        # the marker is a claim: an unparseable one abstains rather than
        # falling back to a number the model did not point at
        return parse_number(m.group()) if m else None

    matches = NUMBER.findall(completion)
    return parse_number(matches[-1]) if matches else None


def gold_value(answer: str) -> Decimal:
    """The reference answer from a GSM8K `answer` field.

    Args:
        answer: The dataset field, ending in `#### <number>`.

    Returns:
        The gold value.

    Raises:
        ValueError: The field does not end in a parseable `####` line, which
            means the corpus is not the one this study was built against.
    """
    if ANSWER_MARKER not in answer:
        msg = f"gold answer has no {ANSWER_MARKER} line: {answer[-60:]!r}"
        raise ValueError(msg)

    m = NUMBER.search(answer.rsplit(ANSWER_MARKER, 1)[1])
    value = parse_number(m.group()) if m else None
    if value is None:
        msg = f"gold answer after {ANSWER_MARKER} is not a number: {answer[-60:]!r}"
        raise ValueError(msg)
    return value


def verify(completion: str, answer: str) -> Verdict:
    """Score one completion against one GSM8K gold answer.

    Args:
        completion: What the model produced.
        answer: The dataset `answer` field for the same question.

    Returns:
        The verdict. CORRECT only on exact equality after canonicalisation.
    """
    gold = gold_value(answer)
    got = extract(completion)

    if got is None:
        return Verdict(Outcome.NO_ANSWER, None, gold)
    outcome = Outcome.CORRECT if got == gold else Outcome.INCORRECT
    return Verdict(outcome, got, gold)
