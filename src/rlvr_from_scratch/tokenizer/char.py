"""Character-level tokenizer.

The smallest honest tokenizer: one token per character, no merges, no subword
vocabulary, no dependencies beyond the standard library.

Torch deliberately never enters this module. ``encode`` produces a plain
``list[int]`` and the conversion to tensors happens one layer up, in the data
package. That boundary keeps this file reasonable on its own and keeps its
tests free of any torch import.

Scope note for Cycle 3: a character vocabulary makes the training loop the
object of study rather than the tokenizer. BPE is a separate piece, later, if
it earns its place.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


class CharTokenizer:
    """Maps single characters to contiguous integer ids and back.

    Vocabulary order is authoritative: a character's index in ``itos`` is the
    row it will occupy in the model's embedding table. Because of that,
    ``__init__`` preserves the order it is given rather than canonicalising it.
    Sorting happens in exactly one place — ``from_text`` — and ``load`` must
    reproduce a saved order verbatim, since re-sorting would silently remap
    every id a checkpoint was trained with.

    Attributes:
        itos: The vocabulary in id order. Index ``i`` holds the character
            encoded as id ``i``. Immutable, so it cannot desync from the
            reverse mapping built alongside it.

    Example:
        >>> tok = CharTokenizer.from_text("hello")
        >>> tok.vocab_size
        4
        >>> tok.decode(tok.encode("hell"))
        'hell'
    """

    def __init__(self, chars: Sequence[str]) -> None:
        """Build a tokenizer over an explicit, ordered vocabulary.

        Args:
            chars: The vocabulary, in the exact id order to use. Each element
                must be a single character and the sequence must contain no
                duplicates. A plain ``str`` is accepted and iterates
                character-wise, so ``CharTokenizer("abc")`` is valid.

        Raises:
            ValueError: If ``chars`` is empty, contains an element that is not
                exactly one character, or contains duplicates. Duplicates are
                fatal rather than deduplicated: silently collapsing them would
                leave ``vocab_size`` reporting the longer count and size the
                embedding table wrongly, with no error until much later.
        """
        # 1. Reject the empty vocabulary. A tokenizer with nothing in it is a
        #    bug at the call site, not a degenerate-but-valid object.
        if len(chars)

        # 2. Reject any element whose length is not exactly 1. This is what
        #    makes the "char" in CharTokenizer true, and it catches the
        #    ["ab", "c"] mistake at construction rather than at decode time.

        # 3. Reject duplicates: compare len(chars) against len(set(chars)).
        #    Put BOTH numbers in the message — "65 chars, 64 unique" tells you
        #    how bad it is and roughly where to look; "duplicate found" does not.

        # 4. Store the vocabulary as a tuple. Immutability is the whole point:
        #    a list could be mutated by a caller after construction and go
        #    silently out of sync with the dict built in step 5.

        # 5. Build the char -> id dict from your OWN stored tuple, not from the
        #    `chars` argument. Building from the stored copy means the two
        #    structures cannot disagree even if `chars` was a one-shot iterator
        #    or some lazy sequence that behaves differently on second read.

        raise NotImplementedError

    @property
    def itos(self) -> tuple[str, ...]:
        """The vocabulary in id order.

        Returns:
            The characters as an immutable tuple, where index ``i`` is the
            character with id ``i``.
        """
        # 1. Return the stored tuple directly. No defensive copy is needed —
        #    tuples are immutable, so handing this out cannot corrupt state.

        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        """Number of distinct characters this tokenizer knows.

        This is the value to pass as the model's embedding-table size and as
        the output width of the language-model head.

        Returns:
            The vocabulary size, always >= 1.
        """
        # 1. Derive it from the length of the stored vocabulary tuple. Never
        #    store a separate count — a cached number is a number that can
        #    drift from the thing it counts.

        raise NotImplementedError

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        """Derive a vocabulary from a corpus.

        This is the only place sorting happens, and the sort is load-bearing:
        ``set`` iteration order for strings depends on the process hash seed,
        so an unsorted vocabulary would differ between runs. A checkpoint
        trained on Monday would decode to noise on Tuesday, with nothing in the
        loss curve to suggest anything was wrong.

        Args:
            text: The corpus to derive the vocabulary from. Every distinct
                character appearing in it becomes a token.

        Returns:
            A tokenizer whose vocabulary is the sorted set of characters in
            ``text``.

        Raises:
            ValueError: If ``text`` is empty, propagated from ``__init__``.

        Example:
            >>> CharTokenizer.from_text("banana").itos
            ('a', 'b', 'n')
        """
        # 1. Take the set of characters in `text`, sort it, hand it to cls().
        #    One line. Resist adding knobs here — min frequency, max vocab
        #    size, special tokens. This box is deliberately the smallest honest
        #    version, and every knob is another thing that needs a test.

        raise NotImplementedError

    # ------------------------------------------------------------------
    # Next chunk — leave these alone until the trio above is green.
    # ------------------------------------------------------------------

    def encode(self, s: str) -> list[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> CharTokenizer: ...
