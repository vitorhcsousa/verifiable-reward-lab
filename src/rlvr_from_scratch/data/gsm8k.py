"""
GSM8K: load the pinned jsonl files and cut a split that is a function of the
data rather than of an RNG.

Upstream ships train.jsonl and test.jsonl. test.jsonl is the held-out set:
it is loaded and counted here and then left alone until the block closes.
Dev is carved out of the official train split, and every intermediate
decision in the study is measured on dev, never on held-out.

The train/dev cut is content-addressed - an example lands in dev because
sha256(seed, question) sorts early, not because it was the n-th draw of a
seeded shuffle. Both are deterministic today; only one of them is still
deterministic if the RNG implementation changes, if the file is re-fetched
in a different order, or if the pool is filtered later. The split has to be
re-derivable in W42 for the numbers to mean anything.

    python -m rlvr_from_scratch.data.gsm8k
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from rlvr_from_scratch.data.fetch import DATA_DIR, SOURCES

DEV_SIZE = 500
SPLIT_SEED = 1337  # the seed convention of configs/tiny.yaml

TRAIN_FILE = SOURCES["gsm8k-train"].fname
HELD_OUT_FILE = SOURCES["gsm8k-test"].fname


@dataclass(frozen=True)
class Example:
    """One GSM8K item, exactly as the pinned file holds it.

    The answer keeps its reasoning and its `#### <value>` line. Pulling the
    final value out is the verifier's job; doing it here would put the same
    parsing rule in two places and let them drift.
    """

    question: str
    answer: str


@dataclass(frozen=True)
class Split:
    """A frozen train/dev/held-out cut plus the numbers that identify it."""

    train: tuple[Example, ...]
    dev: tuple[Example, ...]
    held_out: tuple[Example, ...]
    """Official test split. Not to be selected on before the block closes."""

    seed: int
    fingerprint: str
    """sha256 over the dev questions in dev order. Two runs claiming the same
    split must print the same string; that is cheaper to check than 500
    questions."""


def load_jsonl(path: Path) -> tuple[Example, ...]:
    """Read one Example per non-empty line, in file order.

    Args:
        path: A jsonl file with `question` and `answer` on every line.

    Returns:
        The examples, in the order the file holds them.

    Raises:
        ValueError: The file is missing, or a line lacks the expected keys.
    """
    if not path.exists():
        msg = f"{path} is missing - run `make data` to fetch it"
        raise ValueError(msg)

    out: list[Example] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if "question" not in rec or "answer" not in rec:
                msg = f"{path}:{i} has keys {sorted(rec)}, expected question and answer"
                raise ValueError(msg)
            out.append(Example(question=rec["question"], answer=rec["answer"]))
    return tuple(out)


def sort_key(example: Example, seed: int) -> str:
    """Where an example sits in the shuffled order, under this seed."""
    return hashlib.sha256(f"{seed}:{example.question}".encode()).hexdigest()


def fingerprint(dev: tuple[Example, ...]) -> str:
    """One hex string identifying exactly this dev set, order included."""
    h = hashlib.sha256()
    for e in dev:
        h.update(e.question.encode())
        h.update(b"\0")  # so two questions can't merge into one preimage
    return h.hexdigest()


def cut(
    pool: tuple[Example, ...], *, seed: int = SPLIT_SEED, dev_size: int = DEV_SIZE
) -> tuple[tuple[Example, ...], tuple[Example, ...]]:
    """Split a pool into (train, dev) deterministically.

    Args:
        pool: The official train split, in any order.
        seed: Changes which examples land in dev, nothing else.
        dev_size: How many examples dev takes.

    Returns:
        (train, dev), disjoint, together covering the pool.

    Raises:
        ValueError: dev_size does not leave a train set behind.
    """
    if not 0 < dev_size < len(pool):
        msg = f"dev_size {dev_size} does not fit a pool of {len(pool)}"
        raise ValueError(msg)

    # sorted is stable, so duplicate questions keep file order rather than
    # landing on either side depending on the sort implementation
    order = sorted(pool, key=lambda e: sort_key(e, seed))
    return tuple(order[dev_size:]), tuple(order[:dev_size])


def load(
    data_dir: Path = DATA_DIR, *, seed: int = SPLIT_SEED, dev_size: int = DEV_SIZE
) -> Split:
    """Load the pinned files and return the frozen split."""
    pool = load_jsonl(data_dir / TRAIN_FILE)
    held_out = load_jsonl(data_dir / HELD_OUT_FILE)
    train, dev = cut(pool, seed=seed, dev_size=dev_size)
    return Split(
        train=train,
        dev=dev,
        held_out=held_out,
        seed=seed,
        fingerprint=fingerprint(dev),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="report the frozen GSM8K split")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--seed", type=int, default=SPLIT_SEED)
    p.add_argument("--dev-size", type=int, default=DEV_SIZE)
    args = p.parse_args(argv)

    try:
        s = load(args.data_dir, seed=args.seed, dev_size=args.dev_size)
    except (ValueError, OSError) as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1

    pool = len(s.train) + len(s.dev)
    print(f"pool (official train)  {pool}")
    print(f"n_train                {len(s.train)}")
    print(f"n_dev                  {len(s.dev)}  (seed {s.seed})")
    print(f"n_eval (held-out)      {len(s.held_out)}  untouched")
    print(f"dev fingerprint        {s.fingerprint}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
