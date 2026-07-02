"""The full decoder-only transformer, assembled from scratch.

Wires the Phase 1 components into a complete language model:

    tokens
      -> token embedding            (B, T)      -> (B, T, d_model)
      -> (optional additive pos)    sinusoidal / learned, when not RoPE
      -> N x TransformerBlock       attention + FFN with residuals
      -> final norm                 RMSNorm by default
      -> LM head                    (B, T, d_model) -> (B, T, vocab)

The default configuration is the modern stack (pre-norm + RMSNorm +
SwiGLU + RoPE + tied embeddings), matching Llama / Qwen / Mistral. The
classical "Attention Is All You Need" decoder is recoverable by flipping
config knobs (``positional="sinusoidal"``, ``norm="layernorm"``,
``ffn="gelu"``, ``pre_norm=False``) — no second implementation.

Positional information is handled in one of two places, never both:
  * RoPE rotates Q/K *inside* attention (shared module, applied per head);
    no vector is added to the residual stream.
  * Sinusoidal / learned encodings are *added* to token embeddings once,
    before the first block.

Initialization follows the GPT-2 / nanoGPT recipe rather than the
component-level He init: linear and embedding weights ~ N(0, 0.02), and
the residual output projections (attention W_O, FFN down-projection) are
additionally scaled by 1/sqrt(2 * n_layers). That scaling keeps the
variance of the residual stream from growing with depth, which is what
gives deep pre-norm transformers their clean early-training loss curves.

Reference: "The Complete Forward Pass"
    https://www.vitorsousa.com/foundations/
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from rlvr_from_scratch.model.attention import causal_mask
from rlvr_from_scratch.model.block import TransformerBlock
from rlvr_from_scratch.model.ffn import GeluFFN, SwiGLU
from rlvr_from_scratch.model.norm import LayerNorm, RMSNorm
from rlvr_from_scratch.model.positional import (
    LearnedPositionalEmbedding,
    RotaryPositionalEmbedding,
    SinusoidalPositionalEncoding,
)
from rlvr_from_scratch.model.sampling import sample

# Per-layer key/value cache: (K, V), each (B, H, T_prev, d_k).
KVCache = tuple[Tensor, Tensor]

_NORMS: dict[str, type[nn.Module]] = {"rmsnorm": RMSNorm, "layernorm": LayerNorm}
_FFNS: dict[str, type[nn.Module]] = {"swiglu": SwiGLU, "gelu": GeluFFN}


# =========================================================================
# Config
# =========================================================================


@dataclass
class TransformerConfig:
    """Hyperparameters for the decoder-only transformer.

    Defaults describe the modern stack (RoPE + RMSNorm + SwiGLU + pre-norm
    + tied embeddings). Architecture variants are plain config changes,
    not new code paths.

    Args:
        vocab_size:     Size of the token vocabulary.
        d_model:        Model / residual-stream dimension.
        n_layers:       Number of stacked transformer blocks.
        n_heads:        Attention heads. Must divide d_model evenly.
        d_ff:           FFN hidden dimension. If None, a sensible default
                        is derived from d_model and the FFN type.
        max_seq_len:    Maximum context length (sets positional caches).
        positional:     "rope" (default), "sinusoidal", "learned", "none".
        norm:           "rmsnorm" (default) or "layernorm".
        ffn:            "swiglu" (default) or "gelu".
        pre_norm:       Pre-norm residuals (default) vs classical post-norm.
        bias:           Whether linear layers carry bias. Default False.
        tie_embeddings: Share the token-embedding weight with the LM head.
        rope_base:      RoPE frequency base (ignored unless positional=rope).
        init_std:       Std of the base normal init for weights.
    """

    vocab_size: int
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int | None = None
    max_seq_len: int = 1024
    positional: Literal["rope", "sinusoidal", "learned", "none"] = "rope"
    norm: Literal["rmsnorm", "layernorm"] = "rmsnorm"
    ffn: Literal["swiglu", "gelu"] = "swiglu"
    pre_norm: bool = True
    bias: bool = False
    tie_embeddings: bool = True
    rope_base: float = 10000.0
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            msg = f"vocab_size must be positive, got {self.vocab_size}"
            raise ValueError(msg)
        if self.d_model % self.n_heads != 0:
            msg = (
                f"d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )
            raise ValueError(msg)
        if self.positional == "rope" and (self.d_model // self.n_heads) % 2 != 0:
            d_k = self.d_model // self.n_heads
            msg = f"RoPE needs an even head dim, got d_k={d_k}"
            raise ValueError(msg)
        if self.norm not in _NORMS:
            msg = f"unknown norm {self.norm!r}, expected one of {list(_NORMS)}"
            raise ValueError(msg)
        if self.ffn not in _FFNS:
            msg = f"unknown ffn {self.ffn!r}, expected one of {list(_FFNS)}"
            raise ValueError(msg)

    @property
    def d_k(self) -> int:
        """Per-head dimension."""
        return self.d_model // self.n_heads

    def ffn_dim(self) -> int:
        """Resolve the FFN hidden dimension.

        SwiGLU carries three matrices instead of two, so its hidden size is
        scaled by ~2/3 (and rounded to a multiple of 64) to match the
        parameter count of a 4*d_model vanilla FFN — the Llama convention.
        """
        if self.d_ff is not None:
            return self.d_ff
        if self.ffn == "swiglu":
            raw = int(8 * self.d_model / 3)
            return max(64, ((raw + 63) // 64) * 64)
        return 4 * self.d_model


# =========================================================================
# Decoder-only transformer
# =========================================================================


class DecoderTransformer(nn.Module):
    """A complete decoder-only transformer language model.

    Args:
        config: A :class:`TransformerConfig`.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        norm_cls = _NORMS[config.norm]
        ffn_cls = _FFNS[config.ffn]
        d_ff = config.ffn_dim()

        # =========================================
        # 1. Token embedding
        # =========================================
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        # =========================================
        # 2. Positional information
        # =========================================
        # RoPE lives inside attention (shared across blocks); additive
        # encodings are added to the residual stream before block 0.
        self.rope: RotaryPositionalEmbedding | None = None
        self.pos_emb: nn.Module | None = None
        if config.positional == "rope":
            self.rope = RotaryPositionalEmbedding(
                config.d_k, max_len=config.max_seq_len, base=config.rope_base
            )
        elif config.positional == "sinusoidal":
            self.pos_emb = SinusoidalPositionalEncoding(
                config.d_model, max_len=config.max_seq_len
            )
        elif config.positional == "learned":
            self.pos_emb = LearnedPositionalEmbedding(
                config.d_model, max_len=config.max_seq_len
            )

        # =========================================
        # 3. Stack of transformer blocks
        # =========================================
        self.blocks = nn.ModuleList(
            TransformerBlock(
                config.d_model,
                config.n_heads,
                d_ff,
                pre_norm=config.pre_norm,
                norm_cls=norm_cls,
                ffn_cls=ffn_cls,
                rope=self.rope,
                bias=config.bias,
            )
            for _ in range(config.n_layers)
        )

        # =========================================
        # 4. Final norm + LM head
        # =========================================
        self.final_norm = norm_cls(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            # Weight tying: one matrix serves as both the input lookup and
            # the output projection (GPT-2 / Llama). Halves the largest
            # parameter block and couples the two roles.
            self.lm_head.weight = self.token_emb.weight

        # =========================================
        # 5. Initialize
        # =========================================
        self.apply(self._init_weights)
        self._scale_residual_projections()

    # ---- initialization -------------------------------------------------

    def _init_weights(self, module: nn.Module) -> None:
        """GPT-2 base init: N(0, init_std) for linear/embedding weights."""
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def _scale_residual_projections(self) -> None:
        """Scale the projections that write into the residual stream.

        Each block adds two residual contributions (attention W_O and the
        FFN down-projection). Scaling their init by 1/sqrt(2 * n_layers)
        keeps the residual variance roughly constant with depth.
        """
        scale = self.config.init_std / math.sqrt(2 * self.config.n_layers)
        for name, param in self.named_parameters():
            if name.endswith(("W_O.weight", "W_down.weight", "W_2.weight")):
                nn.init.normal_(param, mean=0.0, std=scale)

    # ---- introspection --------------------------------------------------

    def num_params(self, *, non_embedding: bool = True) -> int:
        """Count parameters.

        Args:
            non_embedding: If True (default), exclude the token-embedding
                table — the conventional "non-embedding" parameter count.
                With tied weights the LM head shares that table and is
                already excluded.
        """
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.token_emb.weight.numel()
        return total

    # ---- forward --------------------------------------------------------

    def forward(
        self,
        idx: Tensor,
        *,
        kv_caches: list[KVCache] | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, list[KVCache] | None]:
        """Forward pass.

        Args:
            idx:       Token ids (B, T), dtype long.
            kv_caches: Optional per-layer (K, V) caches from previous steps.
                       When provided, ``idx`` holds only the new tokens and
                       attention attends over cached + new positions.
            use_cache: If True, return updated per-layer caches (seeding
                       empty ones when ``kv_caches`` is None).

        Returns:
            logits:     (B, T, vocab_size)
            new_caches: Updated per-layer caches, or None when caching off.
        """
        B, T = idx.shape
        past_len = (
            kv_caches[0][0].size(2)
            if kv_caches is not None and len(kv_caches) > 0
            else 0
        )
        total_len = past_len + T
        if total_len > self.config.max_seq_len:
            msg = (
                f"sequence length {total_len} exceeds "
                f"max_seq_len ({self.config.max_seq_len})"
            )
            raise ValueError(msg)

        # =========================================
        # 1. Embed tokens (+ additive positional)
        # =========================================
        x = self.token_emb(idx)  # (B, T, d_model)
        if self.pos_emb is not None:
            if past_len > 0:
                msg = (
                    "incremental decoding with additive positional encodings "
                    "is not supported; use positional='rope'"
                )
                raise NotImplementedError(msg)
            x = self.pos_emb(x)  # (B, T, d_model)

        # =========================================
        # 2. Causal mask for queries [past_len, total_len) over keys [0, total_len)
        # =========================================
        # Slicing the full causal mask gives the rectangular mask the cache
        # path needs; for a single new token the row is all-allowed (None-equiv).
        mask = causal_mask(total_len, device=idx.device)[:, :, past_len:total_len, :]

        # =========================================
        # 3. Seed caches if requested, then run the stack
        # =========================================
        if use_cache and kv_caches is None:
            kv_caches = self._empty_caches(B, idx.device, idx.dtype)

        new_caches: list[KVCache] | None = [] if kv_caches is not None else None
        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, updated = block(x, mask, layer_cache)
            if new_caches is not None and updated is not None:
                new_caches.append(updated)

        # =========================================
        # 4. Final norm + projection to vocabulary
        # =========================================
        x = self.final_norm(x)  # (B, T, d_model)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        return logits, (new_caches if use_cache else None)

    def _empty_caches(
        self, batch: int, device: torch.device, dtype: torch.dtype
    ) -> list[KVCache]:
        """Per-layer zero-length (K, V) caches to start incremental decoding."""
        cfg = self.config
        param_dtype = self.token_emb.weight.dtype
        del dtype  # ids dtype is irrelevant; caches follow the model dtype
        empty = lambda: torch.zeros(  # noqa: E731
            batch, cfg.n_heads, 0, cfg.d_k, device=device, dtype=param_dtype
        )
        return [(empty(), empty()) for _ in range(cfg.n_layers)]

    # ---- generation -----------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: int | None = None,
        top_p: float | None = None,
        generator: torch.Generator | None = None,
        use_cache: bool = True,
    ) -> Tensor:
        """Autoregressively extend a prompt, one token at a time.

        Greedy by default (``do_sample=False``). With ``do_sample=True``
        the next token is drawn via :func:`sample`, which handles
        temperature, top-k and top-p (nucleus) filtering — this method
        stays the minimal autoregressive loop.

        Args:
            idx:            Prompt token ids (B, T).
            max_new_tokens: Number of tokens to append.
            temperature:    Softmax temperature when sampling (>0).
            do_sample:      Sample from the distribution vs take the argmax.
            top_k:          Keep only the k largest logits before sampling.
            top_p:          Nucleus filter: keep the smallest prefix with
                            cumulative probability >= top_p.
            generator:      Optional RNG for reproducible sampling.
            use_cache:      Use an incremental KV-cache (RoPE/none only).
                            Falls back to full recompute otherwise.

        Returns:
            (B, T + max_new_tokens) token ids, prompt included.
        """
        was_training = self.training
        self.eval()

        # Additive positional encodings have no incremental path -> recompute.
        cache_ok = use_cache and self.pos_emb is None
        caches: list[KVCache] | None = None

        for step in range(max_new_tokens):
            if cache_ok:
                # First step consumes the whole prompt and seeds the cache;
                # later steps feed only the previous token.
                step_input = idx if step == 0 else idx[:, -1:]
                logits, caches = self(step_input, kv_caches=caches, use_cache=True)
            else:
                cond = idx[:, -self.config.max_seq_len :]
                logits, _ = self(cond)

            next_logits = logits[:, -1, :]  # (B, vocab)
            # Greedy == temperature 0 in sample(); filters are irrelevant
            # then, so the default path stays byte-identical to before.
            next_id = sample(
                next_logits,
                temperature=temperature if do_sample else 0.0,
                top_k=top_k,
                top_p=top_p,
                generator=generator,
            )  # (B, 1)
            idx = torch.cat([idx, next_id], dim=1)

        if was_training:
            self.train()
        return idx
