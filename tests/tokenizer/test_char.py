"""Tests for the character-level tokenizer.

Covers:
- Construction: empty, multi-character and duplicate vocabularies all raise
- Order is the contract: __init__ preserves it, from_text is the only sort
- Determinism: from_text is stable across PYTHONHASHSEED values
- encode/decode: roundtrip, length, unknown characters, out-of-range ids
- save/load: order survives verbatim, and malformed files raise rather than
  silently producing a wrong vocabulary
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rlvr_from_scratch.tokenizer.char import CharTokenizer

# =========================================================================
# Constants and fixtures
# =========================================================================

CORPUS = "the quick brown fox jumps over the lazy dog"

# deliberately NOT alphabetical: any accidental sort anywhere in the round
# trip changes this, which is exactly what the save/load tests are for.
UNSORTED = ("z", "a", "m", "B", " ")


@pytest.fixture
def tok() -> CharTokenizer:
    return CharTokenizer("abc")


# =========================================================================
# Construction and validation
# =========================================================================


def test_rejects_empty_vocabulary() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        CharTokenizer([])


@pytest.mark.parametrize("bad", [["ab", "c"], ["a", ""], ["abc"]])
def test_rejects_non_single_characters(bad: list[str]) -> None:
    with pytest.raises(ValueError, match="single characters"):
        CharTokenizer(bad)


def test_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicated"):
        CharTokenizer("aab")


def test_duplicate_message_reports_both_counts() -> None:
    # "3 chars, 2 unique" says how bad it is; "duplicate found" does not.
    with pytest.raises(ValueError) as excinfo:
        CharTokenizer("aab")
    message = str(excinfo.value)
    assert "3 chars" in message
    assert "2 unique" in message
    assert "'a'" in message


def test_accepts_str_and_list_identically() -> None:
    assert CharTokenizer("abc").itos == CharTokenizer(["a", "b", "c"]).itos


def test_init_preserves_given_order() -> None:
    # __init__ must NOT sort: the caller's order is the id assignment, and
    # load() depends on this to reproduce a saved vocabulary verbatim.
    assert CharTokenizer("cba").itos == ("c", "b", "a")


def test_itos_is_an_immutable_tuple() -> None:
    assert isinstance(CharTokenizer("abc").itos, tuple)


def test_vocab_size_matches_itos(tok: CharTokenizer) -> None:
    assert tok.vocab_size == len(tok.itos) == 3


def test_ids_are_contiguous_from_zero() -> None:
    tokenizer = CharTokenizer.from_text(CORPUS)
    ids = tokenizer.encode("".join(tokenizer.itos))
    assert ids == list(range(tokenizer.vocab_size))


# =========================================================================
# from_text
# =========================================================================


def test_from_text_sorts_and_dedupes() -> None:
    assert CharTokenizer.from_text("banana").itos == ("a", "b", "n")


def test_from_text_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        CharTokenizer.from_text("")


_HASH_SEED_SCRIPT = (
    "import json;"
    "from rlvr_from_scratch.tokenizer.char import CharTokenizer;"
    f"print(json.dumps(CharTokenizer.from_text({CORPUS!r}).itos))"
)


def test_from_text_is_stable_across_hash_seeds() -> None:
    """The sort in from_text is load-bearing, not cosmetic.

    Python randomises str hashing per process, so an unsorted vocabulary
    would differ between runs. A checkpoint trained on Monday would then
    decode to noise on Tuesday with nothing in the loss curve to explain it.
    """
    outputs = set()
    for seed in ("0", "1", "42"):
        result = subprocess.run(
            [sys.executable, "-c", _HASH_SEED_SCRIPT],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        outputs.add(result.stdout.strip())
    assert len(outputs) == 1


# =========================================================================
# encode / decode
# =========================================================================


def test_encode_maps_to_ids(tok: CharTokenizer) -> None:
    assert tok.encode("cab") == [2, 0, 1]


def test_encode_length_matches_input(tok: CharTokenizer) -> None:
    assert len(tok.encode("abcabc")) == 6


def test_roundtrip_over_the_corpus() -> None:
    tokenizer = CharTokenizer.from_text(CORPUS)
    assert tokenizer.decode(tokenizer.encode(CORPUS)) == CORPUS


def test_empty_string_roundtrips(tok: CharTokenizer) -> None:
    assert tok.encode("") == []
    assert tok.decode([]) == ""


def test_encode_rejects_unknown_character(tok: CharTokenizer) -> None:
    with pytest.raises(ValueError) as excinfo:
        tok.encode("abZ")
    message = str(excinfo.value)
    assert "'Z'" in message  # which character
    assert "index 2" in message  # and where


def test_decode_accepts_any_sequence(tok: CharTokenizer) -> None:
    assert tok.decode((2, 0, 1)) == "cab"


@pytest.mark.parametrize("bad_id", [3, 99, -1])
def test_decode_rejects_out_of_range_ids(tok: CharTokenizer, bad_id: int) -> None:
    # -1 is the interesting one: itos[-1] would happily return the last
    # character, so a pad token or an off-by-one decodes to something
    # plausible and only surfaces as mysteriously bad samples days later.
    with pytest.raises(ValueError, match="outside the vocabulary"):
        tok.decode([bad_id])


def test_decode_error_names_the_offending_id(tok: CharTokenizer) -> None:
    with pytest.raises(ValueError) as excinfo:
        tok.decode([0, 7])
    message = str(excinfo.value)
    assert "id 7" in message
    assert "index 1" in message


# =========================================================================
# save / load
# =========================================================================


def test_save_load_roundtrip_preserves_order(tmp_path: Path) -> None:
    # the whole point: an unsorted vocabulary must come back unsorted. a
    # stray sort in load() would remap every id in a checkpoint at once,
    # raising nothing and generating confident nonsense.
    original = CharTokenizer(UNSORTED)
    path = tmp_path / "vocab.json"
    original.save(path)
    assert CharTokenizer.load(path).itos == UNSORTED


def test_saved_file_is_readable_json(tmp_path: Path) -> None:
    path = tmp_path / "vocab.json"
    CharTokenizer(UNSORTED).save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["itos"] == list(UNSORTED)
    assert "version" in payload


def test_save_load_roundtrip_with_non_ascii(tmp_path: Path) -> None:
    original = CharTokenizer.from_text("olá, coração — ação")
    path = tmp_path / "vocab.json"
    original.save(path)
    assert CharTokenizer.load(path).itos == original.itos


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "vocab.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        CharTokenizer.load(path)


def test_load_rejects_non_object(tmp_path: Path) -> None:
    path = _write(tmp_path / "vocab.json", ["a", "b"])
    with pytest.raises(ValueError, match="must contain a JSON object"):
        CharTokenizer.load(path)


@pytest.mark.parametrize(
    "payload",
    [{"itos": ["a", "b"]}, {"version": 1}, {}],
)
def test_load_rejects_missing_keys(tmp_path: Path, payload: dict) -> None:
    path = _write(tmp_path / "vocab.json", payload)
    with pytest.raises(ValueError, match="missing required key"):
        CharTokenizer.load(path)


def test_load_rejects_unknown_version(tmp_path: Path) -> None:
    path = _write(tmp_path / "vocab.json", {"version": 99, "itos": ["a"]})
    with pytest.raises(ValueError, match="format version"):
        CharTokenizer.load(path)


def test_load_rejects_non_list_itos(tmp_path: Path) -> None:
    path = _write(tmp_path / "vocab.json", {"version": 1, "itos": "ab"})
    with pytest.raises(ValueError, match="not a list"):
        CharTokenizer.load(path)


def test_load_rejects_non_string_entries(tmp_path: Path) -> None:
    path = _write(tmp_path / "vocab.json", {"version": 1, "itos": ["a", 2]})
    with pytest.raises(ValueError, match="non-string entries"):
        CharTokenizer.load(path)


def test_load_propagates_init_validation(tmp_path: Path) -> None:
    # a file can be structurally fine and still describe an impossible
    # vocabulary. __init__ is the single gate, so load must go through it.
    path = _write(tmp_path / "vocab.json", {"version": 1, "itos": ["a", "a"]})
    with pytest.raises(ValueError, match="duplicated"):
        CharTokenizer.load(path)
