"""
Character-level tokenizer. One token per character, no merges, no subword
vocabulary, nothing outside the standard library.

No torch in here on purpose. encode returns a plain list[int] and the tensor
conversion happens one layer up in the data package, which keeps this file
readable on its own and its tests free of a torch import.

A char vocab is the boring choice, which is the point for Cycle 3: it makes the
training loop the object of study instead of the tokenizer. BPE later, if it
earns it.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

# bumped only if the meaning of an existing field changes, not when one is added
_FORMAT_VERSION = 1


class CharTokenizer:
    """
    Characters to ints and back.

    The vocab is a tuple of characters and a character's index in it is the row
    it gets in the embedding table, so the *order* is the entire contract. It is
    sorted in exactly one place (from_text) and passed through untouched
    everywhere else.
    """

    def __init__(self, chars: Sequence[str]) -> None:
        # chars is the vocab already in id order. a plain str works and iterates
        # character-wise, so CharTokenizer("abc") is fine.
        if len(chars) == 0:
            msg = "vocabulary must not be empty"
            raise ValueError(msg)

        multi_char = [e for e in chars if len(e) != 1]
        if multi_char:
            msg = f"vocabulary must only contain single characters, got {multi_char!r}"
            raise ValueError(msg)

        # dupes raise instead of getting deduped. silently collapsing them leaves
        # vocab_size overcounting, which sizes the embedding table wrong and
        # doesn't complain until much later.
        # both counts go in the message. "65 chars, 64 unique" tells you how bad
        # it is and roughly where to look; "duplicate found" does not.
        if len(chars) != len(set(chars)):
            repeated = sorted(c for c, n in Counter(chars).items() if n > 1)
            msg = (
                f"vocabulary cannot contain duplicated elements: "
                f"{len(chars)} chars, {len(set(chars))} unique, "
                f"repeated {repeated!r}"
            )
            raise ValueError(msg)

        self._itos = tuple(chars)
        # built from the stored tuple, not from `chars`. building from the
        # argument means the two structures can disagree if `chars` was a
        # one-shot iterable or a lazy sequence that reads differently twice.
        self._stoi = {c: i for i, c in enumerate(self._itos)}

    @property
    def itos(self) -> tuple[str, ...]:
        """The vocab in id order. Index i is the character with id i."""
        return self._itos

    @property
    def vocab_size(self) -> int:
        """Size of the embedding table and the width of the lm head."""
        return len(self._itos)

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        """Vocab = the sorted set of characters in text."""
        # the sort matters. set iteration order for strings depends on the hash
        # seed, so without it the vocab differs run to run and a checkpoint
        # trained on Monday decodes to noise on Tuesday, with nothing in the loss
        # curve to hint at why.
        return cls(sorted(set(text)))

    def encode(self, s: str) -> list[int]:
        """Text to ids. One id per character, so len(out) == len(s), always."""
        encoded: list[int] = []
        for idx, char in enumerate(s):
            try:
                encoded.append(self._stoi[char])
            except KeyError:
                # no <unk> to fall back on. a tokenizer built by from_text over
                # the training corpus has seen every character it will ever need,
                # so an unknown one means we're encoding text the model never saw.
                # better to say so here than to map it somewhere arbitrary and
                # wonder later why samples got worse.
                msg = f"character {char!r} at index {idx} is not in the vocabulary"
                raise ValueError(msg) from None
        return encoded

    def decode(self, ids: Sequence[int]) -> str:
        """Ids back to text. decode(encode(s)) == s for anything the vocab covers."""
        # takes any Sequence[int], so a list from encode or a .tolist() off a
        # tensor both work. not a tensor itself, which is how torch stays out.
        chars: list[str] = []
        for idx, i in enumerate(ids):
            # the range check is here mostly for negatives: itos[-1] happily
            # returns the last character instead of raising, so an off-by-one or
            # a -1 pad token decodes to something plausible and only shows up as
            # mysteriously bad samples days later.
            if not 0 <= i < self.vocab_size:
                msg = (
                    f"id {i} at index {idx} is outside the "
                    f"vocabulary of size {self.vocab_size}"
                )
                raise ValueError(msg)
            chars.append(self._itos[i])
        return "".join(chars)

    def save(self, path: Path) -> None:
        """Write the vocab to path as json. Parent directory must exist."""
        # only itos goes out. stoi is derived, and writing it too would put the
        # same fact on disk twice with the option of disagreeing with itself.
        # ensure_ascii=False keeps non-ascii readable instead of \uXXXX soup.
        payload = {"version": _FORMAT_VERSION, "itos": list(self._itos)}
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> CharTokenizer:
        """Read back a vocab written by save, in exactly the order it was saved."""
        raw = path.read_text(encoding="utf-8")

        # everything below is checked rather than trusted. the alternative
        # failure is a KeyError or a wrong-shaped embedding table thousands of
        # lines from the malformed file that caused it.
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            msg = f"{path} is not valid JSON: {exc}"
            raise ValueError(msg) from exc

        if not isinstance(payload, dict):
            msg = f"{path} must contain a JSON object, got {type(payload).__name__}"
            raise ValueError(msg)

        missing = {"version", "itos"} - payload.keys()
        if missing:
            msg = f"{path} is missing required key(s): {sorted(missing)}"
            raise ValueError(msg)

        version = payload["version"]
        if version != _FORMAT_VERSION:
            msg = (
                f"{path} has format version {version!r}, "
                f"but this code reads version {_FORMAT_VERSION}"
            )
            raise ValueError(msg)

        itos = payload["itos"]
        if not isinstance(itos, list):
            msg = f"{path} has an 'itos' that is not a list, got {type(itos).__name__}"
            raise ValueError(msg)

        non_str = [e for e in itos if not isinstance(e, str)]
        if non_str:
            msg = f"{path} has non-string entries in 'itos': {non_str!r}"
            raise ValueError(msg)

        # order goes through untouched. re-deriving or re-sorting it here would
        # remap every id in the checkpoint at once and nothing would raise; the
        # model would just generate confident nonsense.
        return cls(itos)
