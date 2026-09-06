"""
Tests for the GSM8K split.

Nothing here reads the real corpus - data/ is gitignored, so a suite that
needed it would be red on a fresh clone until someone ran `make data`. The
fixtures below are synthetic and tiny; what is being tested is the split
rule, and the split rule does not know what a maths word problem is.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from verifiable_reward_lab.data.fetch import SOURCES
from verifiable_reward_lab.data.gsm8k import (
    HELD_OUT_FILE,
    TRAIN_FILE,
    Example,
    cut,
    fingerprint,
    load,
    load_jsonl,
)


def write_jsonl(path: Path, n: int, tag: str) -> None:
    """n synthetic items, distinguishable by tag."""
    lines = [
        json.dumps(
            {"question": f"{tag} question {i}", "answer": f"reasoning\n#### {i}"}
        )
        for i in range(n)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    write_jsonl(tmp_path / TRAIN_FILE, 100, "train")
    write_jsonl(tmp_path / HELD_OUT_FILE, 20, "test")
    return tmp_path


def pool(n: int, tag: str = "q") -> tuple[Example, ...]:
    return tuple(Example(question=f"{tag} {i}", answer=str(i)) for i in range(n))


# ---- load_jsonl ---------------------------------------------------------


def test_load_jsonl_reads_in_file_order(tmp_path: Path) -> None:
    p = tmp_path / "x.jsonl"
    write_jsonl(p, 5, "train")
    got = load_jsonl(p)
    assert len(got) == 5
    assert got[0].question == "train question 0"
    assert got[0].answer.endswith("#### 0")


def test_load_jsonl_ignores_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "x.jsonl"
    p.write_text('{"question": "q", "answer": "a"}\n\n\n', encoding="utf-8")
    assert len(load_jsonl(p)) == 1


def test_load_jsonl_missing_file_points_at_make_data(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="make data"):
        load_jsonl(tmp_path / "absent.jsonl")


def test_load_jsonl_rejects_unexpected_schema(tmp_path: Path) -> None:
    """Upstream changing key names must stop the run, not train on nothing."""
    p = tmp_path / "x.jsonl"
    p.write_text('{"prompt": "q", "completion": "a"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="expected question and answer"):
        load_jsonl(p)


# ---- the cut ------------------------------------------------------------


def test_cut_sizes_and_disjointness() -> None:
    train, dev = cut(pool(100), dev_size=10)
    assert (len(train), len(dev)) == (90, 10)
    assert not set(train) & set(dev)
    assert set(train) | set(dev) == set(pool(100))


def test_cut_is_deterministic() -> None:
    a = cut(pool(100), dev_size=10)
    b = cut(pool(100), dev_size=10)
    assert a == b


def test_cut_does_not_depend_on_input_order() -> None:
    """The whole point of hashing instead of shuffling: re-fetch the file in
    another order and the split is still the same split."""
    forward = pool(100)
    _, dev_a = cut(forward, dev_size=10)
    _, dev_b = cut(tuple(reversed(forward)), dev_size=10)
    assert dev_a == dev_b


def test_seed_changes_the_dev_set() -> None:
    _, dev_a = cut(pool(100), seed=1337, dev_size=10)
    _, dev_b = cut(pool(100), seed=7, dev_size=10)
    assert dev_a != dev_b


def test_dev_size_grows_by_inclusion() -> None:
    """A bigger dev set contains the smaller one, so widening dev never
    silently moves an example out of it."""
    _, small = cut(pool(100), dev_size=10)
    _, large = cut(pool(100), dev_size=20)
    assert set(small) <= set(large)


def test_cut_rejects_dev_size_that_leaves_no_train() -> None:
    with pytest.raises(ValueError, match="does not fit"):
        cut(pool(10), dev_size=10)


def test_cut_rejects_empty_dev() -> None:
    with pytest.raises(ValueError, match="does not fit"):
        cut(pool(10), dev_size=0)


# ---- fingerprint --------------------------------------------------------


def test_fingerprint_is_stable() -> None:
    _, dev = cut(pool(100), dev_size=10)
    assert fingerprint(dev) == fingerprint(dev)


def test_fingerprint_tracks_the_seed() -> None:
    _, a = cut(pool(100), seed=1337, dev_size=10)
    _, b = cut(pool(100), seed=7, dev_size=10)
    assert fingerprint(a) != fingerprint(b)


def test_fingerprint_separates_questions() -> None:
    """Concatenation without a separator would hash ("ab", "c") and
    ("a", "bc") to the same string."""
    left = (Example("ab", "-"), Example("c", "-"))
    right = (Example("a", "-"), Example("bc", "-"))
    assert fingerprint(left) != fingerprint(right)


# ---- load ---------------------------------------------------------------


def test_load_reports_the_three_sizes(data_dir: Path) -> None:
    s = load(data_dir, dev_size=10)
    assert (len(s.train), len(s.dev), len(s.held_out)) == (90, 10, 20)
    assert s.fingerprint == fingerprint(s.dev)


def test_held_out_is_the_official_test_file(data_dir: Path) -> None:
    """Held-out is upstream's split, untouched - never re-cut from train."""
    s = load(data_dir, dev_size=10)
    assert s.held_out == load_jsonl(data_dir / HELD_OUT_FILE)
    assert not set(s.held_out) & (set(s.train) | set(s.dev))


def test_load_without_data_points_at_make_data(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="make data"):
        load(tmp_path)


# ---- the filenames come from the pins -----------------------------------


def test_filenames_track_the_registry() -> None:
    assert TRAIN_FILE == SOURCES["gsm8k-train"].fname
    assert HELD_OUT_FILE == SOURCES["gsm8k-test"].fname
