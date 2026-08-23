"""Training data path.

A corpus encoded once, split by position, and sampled into (x, y) batches
where y is x shifted one token left. Every entry point takes an explicit
torch.Generator, so a run is reproducible from its seed alone.

The corpus itself is never versioned. `rlvr_from_scratch.data.fetch` obtains
it from a pinned URL and checksum, so a clean clone gets the right bytes in
one command. It is deliberately *not* re-exported here: it runs as
`python -m rlvr_from_scratch.data.fetch`, and importing it from this
__init__ would load the module once as a package attribute and again as
__main__, which Python warns about and which would give the two copies
separate module state.
"""

from __future__ import annotations

from rlvr_from_scratch.data.dataset import Corpus, get_batch, load_corpus

__all__ = ["Corpus", "get_batch", "load_corpus"]
