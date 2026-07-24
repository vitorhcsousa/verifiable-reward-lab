# rlvr-from-scratch

[![CI](https://github.com/vitorhcsousa/rlvr-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/vitorhcsousa/rlvr-from-scratch/actions)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**From transformer internals to RLVR alignment in one codebase.**

The full pipeline, equation by equation, one file per concept:

```text
transformer  →  pretraining  →  SFT     →  GRPO      →  GDPO
attention       AdamW +         GSM8K      group        multi-
RoPE            cosine          baseline   rollouts     reward
RMSNorm         warmup                     + reward     fix
```

Single GPU, 10M–50M params, results in hours. No `trl`, no `accelerate` magic. Pure PyTorch.

nanoGPT showed you can pretrain in one file. This shows you can align in one file too.

## status

| #   | Phase                 | Deliverable                                 | Status      |
| --- | --------------------- | ------------------------------------------- | ----------- |
| 1   | Transformer internals | decoder model + KV-cache + sampling + tests | ✅ complete |
| 2   | Pretraining           | 50M model on TinyStories                    | ⏸ parked    |
| 3   | SFT                   | GSM8K fine-tune baseline                    | ⏸ planned   |
| 4   | **GRPO** _(flagship)_ | training run + ablations                    | ⏸ planned   |
| 5   | GDPO                  | multi-reward fix                            | ⏸ planned   |

Phase 1 ships a complete decoder-only transformer: attention with RoPE, RMSNorm/LayerNorm, SwiGLU/GELU FFN, incremental **KV-cache that matches the full forward pass at `atol=1e-5`**, and **greedy / temperature / top-k / top-p sampling** — all shape-, numerical- and gradient-tested (112 tests).

> Next up is not automatically Phase 2. The active plan is: Transformer Internals (done) → Research Engineering Core → Eval / Verifiable System + Failure Analysis. Later phases of this repo open when the plan's checkpoint chooses them.

## quickstart

```bash
git clone https://github.com/vitorhcsousa/rlvr-from-scratch.git
cd rlvr-from-scratch
uv sync --group dev
make check   # ruff + ty + pytest
```

Generate with a (randomly initialized — pretraining is Phase 2) model:

```python
import torch

from rlvr_from_scratch.model.transformer import DecoderTransformer, TransformerConfig

config = TransformerConfig(
    vocab_size=256, d_model=128, n_layers=4, n_heads=4, max_seq_len=256
)
model = DecoderTransformer(config)

prompt = torch.randint(0, 256, (1, 8))  # (B, T) token ids

greedy = model.generate(prompt, max_new_tokens=32)  # argmax, deterministic
sampled = model.generate(
    prompt,
    max_new_tokens=32,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    generator=torch.Generator().manual_seed(0),  # reproducible draws
)
```

Decoding lives in one pure function — [`model/sampling.py`](src/rlvr_from_scratch/model/sampling.py) — applied on logits, with the nucleus off-by-one handled and tested (`python -m rlvr_from_scratch.model.sampling` runs a tiny demo).

## what's tested

- **Shapes** for every component (attention, MHA, RoPE/sinusoidal, norms, FFN, block, model).
- **Numerics**: causality (future tokens never change past logits), centred finite-difference gradient checks end-to-end, pre/post-norm variants.
- **KV-cache correctness**: incremental decoding ≡ full forward at `atol=1e-5` — the signature test of Phase 1.
- **Sampling acceptance**: greedy == argmax; `top_k=1` == greedy; `top_p=1.0` == full-softmax sampling; nucleus renormalizes with out-of-set probability exactly 0; seeded draws reproducible.

## repo layout

```text
src/rlvr_from_scratch/
├── model/          # attention, positional, norm, ffn, block, transformer, sampling
├── tokenizer/      # BPE from scratch            (Phase 2)
├── data/           # TinyStories, GSM8K          (Phase 2+)
├── training/       # pretrain, sft, grpo, gdpo   (Phase 2+)
├── rollout/        # batched generation          (Phase 4)
├── rewards/        # verifiers, format, length   (Phase 4)
└── evaluation/     # GSM8K eval, diagnostics     (Phase 3+)
```

## development

```text
make check      # lint + type check + tests (what CI runs)
make test       # tests only
make test-cov   # tests with coverage
make format     # auto-format
make ci         # full pipeline
```

## writing

Each phase ships with long-form articles documenting the math and the code:

- **Part 1** · [Attention Is All You Need to Implement](https://www.vitorsousa.com/foundations/attention-from-scratch/) — scaled dot-product and multi-head attention
- **Part 2** · [Positional Encoding: Teaching Transformers to Count](https://www.vitorsousa.com/foundations/positional-encoding/) — sinusoidal, learned, RoPE, ALiBi
- **Part 3** · [Building a Transformer: The Complete Forward Pass](https://www.vitorsousa.com/foundations/building-a-transformer/) — norms, residuals, FFN, and the assembled decoder
- _Part 4: Training a Transformer — ships with Phase 2_
- **Phase 1 close-out** · [A Transformer from Raw Tensors: What 112 Tests Taught Me](https://www.vitorsousa.com/blog/transformer-from-raw-tensors/) — the three properties that fail silently, and the tests that pin them

## performance

Tokens/s with vs without the KV-cache — greedy decoding, batch 1, 16-token prompt.

`d_model=256`, 4 layers, 4 heads, vocab 1000, RoPE, float32, CPU, seed 0.

Median of 5 runs (1 warmup excluded):

| n_new | use_cache | median tok/s |    min-max    |
| :---: | :-------: | :----------: | :-----------: |
|  128  |   False   |    317.9     | 312.2 - 326.8 |
|  128  |   True    |    814.3     | 785.9 - 837.3 |
|  256  |   False   |    185.2     | 175.3 - 185.7 |
|  256  |   True    |    699.5     | 683.7 - 707.2 |

Without the cache every step re-processes the whole sequence, so per-generation cost grows as O(T²) — doubling `n_new` costs 42% of throughput. With the cache each step processes one token against cached K/V, O(T) — the same doubling costs only 14%. The speedup widens from 2.6× at 128 tokens to 3.8× at 256, and keeps growing with context length.

Reproduce: `uv run python benchmarks/bench_kv_cache.py`

---

## references

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) (Holtzman et al., 2019)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) (DeepSeek, 2025)
- [GDPO](https://arxiv.org/abs/2601.05242) (NVlabs, Jan 2026) · [reference implementation](https://github.com/NVlabs/GDPO)
- [GSM8K](https://arxiv.org/abs/2110.14168) (Cobbe et al., 2021)
- [TinyStories](https://arxiv.org/abs/2305.07759) (Eldan & Li, 2023)

## license

MIT
