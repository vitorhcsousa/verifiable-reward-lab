"""
The frozen few-shot prompt, and where a completion is cut off.

Qwen 2.5-0.5B **base** is not instruction-tuned: zero-shot it mostly gets
graded on whether it guesses the answer format, which would contaminate RQ4
before the format reward exists. So the prompt shows the format four times
and then asks.

The exemplars are written here rather than sampled from the corpus. Sampling
from train would make the prompt depend on the split seed, and any example
drawn from the pool can land in dev - an exemplar that reappears as a
question is the quietest kind of leak. These are invented; they teach the
shape of an answer, not any of its content.

Truncation matters as much as the prompt. A base model continues the pattern
it was given, so after answering it happily invents the next `Question:`. Left
alone, the verifier's "last ####" rule would score a hallucinated question's
answer. The completion is therefore cut at the first `Question:` the model
emits, and that cut is part of the frozen procedure, not a detail.

`STOP` is also handed to generation as a stop string, so decoding ends where
this cut would have fallen instead of running to the token cap. Both live off
this one constant on purpose: if they ever disagreed, the run would be paying
for tokens that the score then ignores.
"""

from __future__ import annotations

QUESTION_PREFIX = "Question: "
ANSWER_PREFIX = "Answer: "
STOP = "\nQuestion:"

EXEMPLARS: tuple[tuple[str, str], ...] = (
    (
        "A baker has 12 trays with 8 muffins each. He sells 40 muffins. "
        "How many muffins does he have left?",
        "The trays hold 12 * 8 = 96 muffins.\n"
        "After selling he has 96 - 40 = 56 muffins.\n"
        "#### 56",
    ),
    (
        "Sara reads 15 pages a day for 6 days, then 25 pages on the seventh "
        "day. How many pages does she read in the week?",
        "In the first six days she reads 15 * 6 = 90 pages.\n"
        "In the week she reads 90 + 25 = 115 pages.\n"
        "#### 115",
    ),
    (
        "A shirt costs $24. It is on sale for a quarter off. How much does "
        "it cost now?",
        "A quarter of the price is 24 / 4 = 6 dollars.\n"
        "The sale price is 24 - 6 = 18 dollars.\n"
        "#### 18",
    ),
    (
        "Tom has 3 boxes of 15 pencils. He gives 9 pencils to each of his 4 "
        "friends. How many pencils does he have left?",
        "He starts with 3 * 15 = 45 pencils.\n"
        "He gives away 9 * 4 = 36 pencils.\n"
        "He has 45 - 36 = 9 pencils left.\n"
        "#### 9",
    ),
)


def format_example(question: str, answer: str) -> str:
    """One question/answer pair in the shape the model is asked to continue."""
    return f"{QUESTION_PREFIX}{question}\n{ANSWER_PREFIX}{answer}"


def build_prompt(question: str, n_shots: int) -> str:
    """The full prompt for one question.

    Args:
        question: The GSM8K question to answer.
        n_shots: How many exemplars to prepend, at most len(EXEMPLARS).

    Returns:
        The exemplars, then the question with an empty answer to continue.

    Raises:
        ValueError: n_shots is negative or exceeds the exemplars available.
    """
    if not 0 <= n_shots <= len(EXEMPLARS):
        msg = f"n_shots must be in [0, {len(EXEMPLARS)}], got {n_shots}"
        raise ValueError(msg)

    shots = [format_example(q, a) for q, a in EXEMPLARS[:n_shots]]
    shots.append(f"{QUESTION_PREFIX}{question}\n{ANSWER_PREFIX}")
    return "\n\n".join(shots)


def truncate_completion(completion: str) -> str:
    """Cut a completion at the first question the model invents for itself."""
    return completion.split(STOP, 1)[0].rstrip()
