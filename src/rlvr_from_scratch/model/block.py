"""Transformer block built from scratch.

Assembles a single transformer block from the from-scratch components:
multi-head attention, two normalization layers, and a feed-forward
network, tied together with residual connections.

The default configuration is modern (pre-norm + RMSNorm + SwiGLU),
matching Llama / Qwen / Mistral. Flip ``pre_norm=False`` and pass
``norm_cls=LayerNorm`` / ``ffn_cls=GeluFFN`` to recover the classical
"Attention Is All You Need" block — the same switches that the Bits
b3 (norm), b4 (pre/post-norm) and b5 (FFN) discuss.

Note: positional information (RoPE) is applied inside attention, not in
the block. The current MultiHeadAttention does not yet apply RoPE — see
the project notes for the planned additive hook.

Reference: "The Complete Forward Pass"
    https://www.vitorsousa.com/foundations/
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from rlvr_from_scratch.model.attention import MultiHeadAttention
from rlvr_from_scratch.model.ffn import SwiGLU
from rlvr_from_scratch.model.norm import RMSNorm
from rlvr_from_scratch.model.positional import RotaryPositionalEmbedding


# =========================================================================
# Transformer Block
# =========================================================================


class TransformerBlock(nn.Module):
    """A single transformer block: attention + FFN with residuals.

    Pre-norm (default, modern):
        x = x + attn(norm1(x))
        x = x + ffn(norm2(x))

    Post-norm (classical, "Attention Is All You Need"):
        x = norm1(x + attn(x))
        x = norm2(x + ffn(x))

    Pre-norm keeps an unbroken residual highway from input to output,
    which is why deep transformers train stably without the learning-rate
    warmup gymnastics that post-norm needs. Post-norm is kept switchable
    here purely as the classical contrast.

    The block holds two independent norm layers (one per sublayer) and is
    agnostic to which norm/FFN classes are used, as long as they follow
    the project signatures:
        norm_cls(d_model)
        ffn_cls(d_model, d_ff, *, bias=...)

    Args:
        d_model:  Model dimension.
        n_heads:  Number of attention heads. Must divide d_model evenly.
        d_ff:     Hidden dimension of the feed-forward network.
        pre_norm: If True (default), normalize before each sublayer.
        norm_cls: Normalization layer class. Default RMSNorm.
        ffn_cls:  Feed-forward network class. Default SwiGLU.
        rope:     Optional rotary position embedding, injected into the
                  attention sublayer. Built once at the model level and
                  shared across all blocks. Default None.
        bias:     Whether linear layers use bias. Default False.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        *,
        pre_norm: bool = True,
        norm_cls: type[nn.Module] = RMSNorm,
        ffn_cls: type[nn.Module] = SwiGLU,
        rope: RotaryPositionalEmbedding | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.pre_norm = pre_norm

        # =========================================
        # Sublayers: attention, FFN, and one norm each
        # =========================================
        self.attn = MultiHeadAttention(d_model, n_heads, bias=bias, rope=rope)
        self.ffn = ffn_cls(d_model, d_ff, bias=bias)
        self.norm1 = norm_cls(d_model)  # before/after attention
        self.norm2 = norm_cls(d_model)  # before/after FFN

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        kv_cache: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        """Forward pass.

        Args:
            x:        (B, T, d_model)
            mask:     Additive attention mask (B|1, 1|H, T, T).
                      0.0 = allowed, -inf = blocked.
            kv_cache: Optional cached (K, V) from previous steps, each
                      (B, H, T_prev, d_k), for incremental decoding.

        Returns:
            output:       (B, T, d_model)
            new_kv_cache: Updated (K, V) cache, or None if none was passed.
        """
        if self.pre_norm:
            # =========================================
            # Pre-norm: norm -> sublayer -> residual add
            # =========================================
            normed = self.norm1(x)  # (B, T, d_model)
            attn_out, _, new_kv_cache = self.attn(
                normed, normed, normed, mask, kv_cache
            )  # attn_out: (B, T, d_model)
            x = x + attn_out  # (B, T, d_model)

            normed = self.norm2(x)  # (B, T, d_model)
            x = x + self.ffn(normed)  # (B, T, d_model)
        else:
            # =========================================
            # Post-norm: sublayer -> residual add -> norm
            # =========================================
            attn_out, _, new_kv_cache = self.attn(
                x, x, x, mask, kv_cache
            )  # attn_out: (B, T, d_model)
            x = self.norm1(x + attn_out)  # (B, T, d_model)
            x = self.norm2(x + self.ffn(x))  # (B, T, d_model)

        return x, new_kv_cache
