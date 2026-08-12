"""Training data path.

A corpus encoded once, split by position, and sampled into (x, y) batches
where y is x shifted one token left. Every entry point takes an explicit
torch.Generator, so a run is reproducible from its seed alone.
"""

from __future__ import annotations

from rlvr_from_scratch.data.dataset import Corpus, get_batch, load_corpus

__all__ = ["Corpus", "get_batch", "load_corpus"]
