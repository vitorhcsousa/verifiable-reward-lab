"""Transformer model components, built from raw tensors.

Model foundations: attention, positional encodings,
normalization, feed-forward networks, the transformer block, and the
fully assembled decoder-only language model.
"""

from __future__ import annotations

from verifiable_reward_lab.model.attention import (
    MultiHeadAttention,
    causal_mask,
    scaled_dot_product_attention,
)
from verifiable_reward_lab.model.block import TransformerBlock
from verifiable_reward_lab.model.ffn import GeluFFN, SwiGLU
from verifiable_reward_lab.model.norm import LayerNorm, RMSNorm
from verifiable_reward_lab.model.positional import (
    ALiBi,
    LearnedPositionalEmbedding,
    RotaryPositionalEmbedding,
    SinusoidalPositionalEncoding,
)
from verifiable_reward_lab.model.transformer import (
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
