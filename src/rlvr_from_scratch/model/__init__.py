"""Transformer model components, built from raw tensors.

Model foundations: attention, positional encodings,
normalization, feed-forward networks, the transformer block, and the
fully assembled decoder-only language model.
"""

from __future__ import annotations

from rlvr_from_scratch.model.attention import (
    MultiHeadAttention,
    causal_mask,
    scaled_dot_product_attention,
)
from rlvr_from_scratch.model.block import TransformerBlock
from rlvr_from_scratch.model.ffn import GeluFFN, SwiGLU
from rlvr_from_scratch.model.norm import LayerNorm, RMSNorm
from rlvr_from_scratch.model.positional import (
    ALiBi,
    LearnedPositionalEmbedding,
    RotaryPositionalEmbedding,
    SinusoidalPositionalEncoding,
)
from rlvr_from_scratch.model.transformer import (
    DecoderTransformer,
    TransformerConfig,
)

__all__ = [
    "ALiBi",
    "DecoderTransformer",
    "GeluFFN",
    "LayerNorm",
    "LearnedPositionalEmbedding",
    "MultiHeadAttention",
    "RMSNorm",
    "RotaryPositionalEmbedding",
    "SinusoidalPositionalEncoding",
    "SwiGLU",
    "TransformerBlock",
    "TransformerConfig",
    "causal_mask",
    "scaled_dot_product_attention",
]
