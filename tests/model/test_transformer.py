"""Tests for the assembled decoder-only transformer.

Covers:
- Config: defaults, derived dims, validation errors
- Wiring: token emb -> N blocks -> final norm -> LM head; weight tying
- Shape: forward returns (B, T, vocab); logits finite
- Numerical: causality (future tokens never change earlier logits);
  centred finite-difference gradient check end-to-end
- Gradient flow: backward reaches embedding, a block, and the LM head
- Architecture variants: positional / norm / FFN knobs all run
- KV-cache: incremental decoding == full forward at atol=1e-5
- Generation: autoregressive greedy loop; determinism; cache equivalence
"""

from __future__ import annotations

import pytest
import torch

from rlvr_from_scratch.model.block import TransformerBlock
from rlvr_from_scratch.model.ffn import GeluFFN, SwiGLU
from rlvr_from_scratch.model.norm import LayerNorm, RMSNorm
from rlvr_from_scratch.model.transformer import (
    DecoderTransformer,
    TransformerConfig,
)

VOCAB, B, T = 50, 2, 7


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


def _config(**kw) -> TransformerConfig:
    base = dict(
        vocab_size=VOCAB,
        d_model=16,
        n_layers=3,
        n_heads=4,
        max_seq_len=32,
    )
    base.update(kw)
    return TransformerConfig(**base)


def _model(**kw) -> DecoderTransformer:
    return DecoderTransformer(_config(**kw))


def _ids(t: int = T, b: int = B) -> torch.Tensor:
    return torch.randint(0, VOCAB, (b, t))


# =========================================================================
# Config
# =========================================================================


def test_defaults_are_modern() -> None:
    cfg = _config()
    assert cfg.positional == "rope"
    assert cfg.norm == "rmsnorm"
    assert cfg.ffn == "swiglu"
    assert cfg.pre_norm is True
    assert cfg.tie_embeddings is True


def test_d_k_and_ffn_dim_derived() -> None:
    cfg = _config(d_model=24, n_heads=4)
    assert cfg.d_k == 6
    # SwiGLU: ~8/3 * d_model, rounded up to a multiple of 64.
    assert cfg.ffn_dim() % 64 == 0
    assert _config(ffn="gelu", d_model=24).ffn_dim() == 96  # 4 * d_model


def test_config_rejects_indivisible_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        _config(d_model=17, n_heads=4)


def test_config_rejects_odd_rope_head_dim() -> None:
    with pytest.raises(ValueError, match="even head dim"):
        _config(d_model=12, n_heads=4, positional="rope")  # d_k = 3


def test_config_rejects_bad_vocab() -> None:
    with pytest.raises(ValueError, match="vocab_size must be positive"):
        _config(vocab_size=0)


# =========================================================================
# Wiring & shape
# =========================================================================


def test_forward_shape() -> None:
    model = _model()
    logits, caches = model(_ids())
    assert logits.shape == (B, T, VOCAB)
    assert caches is None
    assert torch.isfinite(logits).all()


def test_stack_depth_and_components() -> None:
    model = _model(n_layers=3)
    assert len(model.blocks) == 3
    assert all(isinstance(b, TransformerBlock) for b in model.blocks)
    assert isinstance(model.final_norm, RMSNorm)
    assert isinstance(model.blocks[0].ffn, SwiGLU)


def test_weight_tying_shares_storage() -> None:
    model = _model(tie_embeddings=True)
    assert model.lm_head.weight is model.token_emb.weight


def test_untied_head_is_independent() -> None:
    model = _model(tie_embeddings=False)
    assert model.lm_head.weight is not model.token_emb.weight


def test_num_params_excludes_embedding() -> None:
    model = _model()
    full = model.num_params(non_embedding=False)
    non_emb = model.num_params(non_embedding=True)
    assert non_emb < full
    assert non_emb == full - model.token_emb.weight.numel()


# =========================================================================
# Numerical: causality
# =========================================================================


def test_causality_future_does_not_leak() -> None:
    """Changing a token at position t must not alter logits at positions < t.

    This is the defining property of a causal decoder; it fails loudly if
    the causal mask is dropped or mis-shaped.
    """
    model = _model().eval()
    ids = _ids()
    with torch.no_grad():
        base, _ = model(ids)
        perturbed = ids.clone()
        perturbed[:, -1] = (perturbed[:, -1] + 1) % VOCAB  # change last token
        changed, _ = model(perturbed)
    # All positions strictly before the last must be identical.
    assert torch.allclose(base[:, :-1], changed[:, :-1], atol=1e-6)
    # The last position should react to its own change.
    assert not torch.allclose(base[:, -1], changed[:, -1], atol=1e-6)


# =========================================================================
# Numerical: centred finite-difference gradient check (end-to-end)
# =========================================================================


def test_finite_difference_gradient_end_to_end() -> None:
    """Analytic grad of a scalar loss matches centred finite differences.

    Runs in float64 through the whole stack so the autograd gradient and
    the numerical estimate agree tightly. Probes a handful of LM-head
    entries — the gradient there flows back through every component.
    """
    torch.manual_seed(0)
    model = _model(tie_embeddings=False).double().eval()
    ids = _ids(t=5, b=1)
    target = _ids(t=5, b=1)

    def loss_fn() -> torch.Tensor:
        logits, _ = model(ids)
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB), target.reshape(-1)
        )

    weight = model.lm_head.weight  # (vocab, d_model)
    model.zero_grad()
    loss_fn().backward()
    analytic = weight.grad
    assert analytic is not None

    eps = 1e-6
    coords = [(0, 0), (3, 2), (VOCAB - 1, weight.shape[1] - 1)]
    with torch.no_grad():
        for i, j in coords:
            orig = weight[i, j].item()
            weight[i, j] = orig + eps
            plus = loss_fn().item()
            weight[i, j] = orig - eps
            minus = loss_fn().item()
            weight[i, j] = orig
            numeric = (plus - minus) / (2 * eps)
            assert abs(numeric - analytic[i, j].item()) < 1e-5


# =========================================================================
# Gradient flow
# =========================================================================


def test_gradients_reach_every_stage() -> None:
    model = _model()
    logits, _ = model(_ids())
    logits.sum().backward()
    grads = {
        "token_emb": model.token_emb.weight.grad,
        "attn_W_Q": model.blocks[0].attn.W_Q.weight.grad,
        "ffn": model.blocks[0].ffn.W_gate.weight.grad,
        "final_norm": model.final_norm.gamma.grad,
    }
    for name, g in grads.items():
        assert g is not None, f"no grad for {name}"
        assert torch.isfinite(g).all(), f"non-finite grad for {name}"
        assert (g != 0).any(), f"all-zero grad for {name}"


# =========================================================================
# Architecture variants
# =========================================================================


@pytest.mark.parametrize("positional", ["rope", "sinusoidal", "learned", "none"])
def test_positional_variants_run(positional: str) -> None:
    model = _model(positional=positional)
    logits, _ = model(_ids())
    assert logits.shape == (B, T, VOCAB)
    assert torch.isfinite(logits).all()
    if positional == "rope":
        assert model.rope is not None and model.pos_emb is None
    elif positional == "none":
        assert model.rope is None and model.pos_emb is None
    else:
        assert model.rope is None and model.pos_emb is not None


def test_classical_stack_runs() -> None:
    """Post-norm + LayerNorm + GELU + sinusoidal recovers the 2017 decoder."""
    model = _model(
        positional="sinusoidal",
        norm="layernorm",
        ffn="gelu",
        pre_norm=False,
    )
    assert isinstance(model.final_norm, LayerNorm)
    assert isinstance(model.blocks[0].ffn, GeluFFN)
    logits, _ = model(_ids())
    assert logits.shape == (B, T, VOCAB)
    assert torch.isfinite(logits).all()


# =========================================================================
# KV-cache equivalence
# =========================================================================


@pytest.mark.parametrize("positional", ["rope", "none"])
def test_kv_cache_matches_full_forward(positional: str) -> None:
    """Incremental decoding with a KV-cache equals the full forward pass.

    The signature correctness test for caching: feed the prompt, then one
    token at a time, and require the per-step logits to match a single full
    forward over the whole sequence at atol=1e-5.
    """
    model = _model(positional=positional).eval()
    ids = _ids(t=6, b=2)

    with torch.no_grad():
        full, _ = model(ids)  # (B, 6, vocab)

        # Incremental: prime on the first 2 tokens, then step the rest.
        caches = None
        step_logits = []
        primed, _caches = model(ids[:, :2], use_cache=True)
        caches = _caches
        step_logits.append(primed)  # logits for positions 0, 1
        for t in range(2, ids.size(1)):
            out, caches = model(ids[:, t : t + 1], kv_caches=caches, use_cache=True)
            step_logits.append(out)
        incremental = torch.cat(step_logits, dim=1)  # (B, 6, vocab)

    assert incremental.shape == full.shape
    assert torch.allclose(incremental, full, atol=1e-5)


# =========================================================================
# Generation
# =========================================================================


def test_generate_appends_tokens() -> None:
    model = _model()
    prompt = _ids(t=3, b=B)
    out = model.generate(prompt, max_new_tokens=5)
    assert out.shape == (B, 8)
    assert torch.equal(out[:, :3], prompt)  # prompt preserved
    assert (out >= 0).all() and (out < VOCAB).all()


def test_generate_greedy_is_deterministic() -> None:
    model = _model().eval()
    prompt = _ids(t=3, b=1)
    a = model.generate(prompt, max_new_tokens=6)
    b = model.generate(prompt, max_new_tokens=6)
    assert torch.equal(a, b)


def test_generate_cache_equals_no_cache() -> None:
    """Greedy decoding yields identical tokens with and without the cache."""
    model = _model().eval()
    prompt = _ids(t=4, b=2)
    cached = model.generate(prompt, max_new_tokens=6, use_cache=True)
    full = model.generate(prompt, max_new_tokens=6, use_cache=False)
    assert torch.equal(cached, full)


def test_generate_respects_max_seq_len() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        model = _model(max_seq_len=8)
        model(_ids(t=9, b=1))


# =========================================================================
# Generation: sampling wiring (640.5 / 640.6)
# =========================================================================


def test_generate_sampling_reproducible_with_generator() -> None:
    """Seeded generator -> identical sampled continuations across runs."""
    model = _model().eval()
    prompt = _ids(t=3, b=2)
    kw = dict(do_sample=True, temperature=0.9, top_k=10, top_p=0.9)
    a = model.generate(
        prompt, max_new_tokens=5, generator=torch.Generator().manual_seed(11), **kw
    )
    b = model.generate(
        prompt, max_new_tokens=5, generator=torch.Generator().manual_seed(11), **kw
    )
    assert torch.equal(a, b)


def test_generate_greedy_default_ignores_filters() -> None:
    """do_sample=False stays pure greedy: top_k/top_p must not change it."""
    model = _model().eval()
    prompt = _ids(t=3, b=1)
    plain = model.generate(prompt, max_new_tokens=5)
    filtered = model.generate(prompt, max_new_tokens=5, top_k=3, top_p=0.5)
    assert torch.equal(plain, filtered)


def test_generate_sampled_output_is_valid() -> None:
    """Sampled generation keeps the prompt and emits in-vocab ids."""
    model = _model().eval()
    prompt = _ids(t=3, b=B)
    out = model.generate(
        prompt,
        max_new_tokens=5,
        do_sample=True,
        top_k=7,
        top_p=0.95,
        generator=torch.Generator().manual_seed(3),
    )
    assert out.shape == (B, 8)
    assert torch.equal(out[:, :3], prompt)
    assert (out >= 0).all() and (out < VOCAB).all()
