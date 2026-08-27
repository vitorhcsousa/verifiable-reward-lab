"""Tokenizers.

Training foundation: a character-level tokenizer with no
dependencies, so the training loop is the object of study rather than the
tokenizer. BPE later, if it earns it.
"""

from __future__ import annotations

from rlvr_from_scratch.tokenizer.char import CharTokenizer

__all__ = ["CharTokenizer"]
