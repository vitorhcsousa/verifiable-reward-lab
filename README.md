
# rlvr-from-scratch

[![CI](https://github.com/vitorhcsousa/rlvr-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/vitorhcsousa/rlvr-from-scratch/actions)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**From transformer internals to RLVR alignment in one codebase.**

The full pipeline, equation by equation, one file per concept:
```

transformer → pretraining → SFT → GRPO → GDPO │            │           │      │       │ attention    AdamW +      GSM8K  group    multi- RoPE         cosine       base-  rollouts reward RMSNorm      warmup       line            fix

```
Single GPU, 10M–50M params, results in hours. No `trl`, no `accelerate` magic. Pure PyTorch.

nanoGPT showed you can pretrain in one file. This shows you can align in one file too.

## status — phase 1 of 5

| # | Phase                 | Deliverable                           | Status          |
|---|-----------------------|---------------------------------------|-----------------|
| 1 | Transformer internals | decoder model + tests                 | 🚧 in progress  |
| 2 | Pretraining           | 50M model on TinyStories              | ⏸ planned       |
| 3 | SFT                   | GSM8K fine-tune baseline              | ⏸ planned       |
| 4 | **GRPO** *(flagship)* | training run + ablations              | ⏸ planned       |
| 5 | GDPO                  | multi-reward fix                      | ⏸ planned       |

**Live now:**
- [`model/attention.py`](src/rlvr_from_scratch/model/attention.py) — scaled dot-product + multi-head + causal mask + KV cache
- Foundations articles: [Attention from scratch](https://www.vitorsousa.com/foundations/attention-from-scratch/) · [Positional encoding](https://www.vitorsousa.com/foundations/positional-encoding/)
- Test infrastructure + CI (tests landing this week)

**Target ship:** 2026-07-17 (v1.0). Progress tracked via commits and release tags.

## install

```bash
git clone https://github.com/vitorhcsousa/rlvr-from-scratch.git
cd rlvr-from-scratch
uv sync --group dev
make ci
```

## repo layout

```
src/rlvr_from_scratch/
├── model/          # attention, positional, norm, ffn, transformer, config
├── tokenizer/      # BPE from scratch
├── data/           # TinyStories, GSM8K
├── training/       # pretrain, sft, grpo, gdpo, optimizer, scheduler
├── rollout/        # sampling + batched generation
├── rewards/        # correctness verifier, format, length, conditioned
└── evaluation/     # GSM8K eval, metrics, advantage diagnostics
```

## development

```
make check      # lint + type check
make test       # run tests
make test-cov   # tests with coverage
make format     # auto-format
make ci         # full pipeline
```

## writing

Each phase ships with long-form articles documenting the math and the code:

- **Part 1** · [Attention Is All You Need to Implement](https://www.vitorsousa.com/foundations/attention-from-scratch/) — scaled dot-product and multi-head attention
- **Part 2** · [Positional Encoding: Teaching Transformers to Count](https://www.vitorsousa.com/foundations/positional-encoding/) — sinusoidal, learned, RoPE, ALiBi
- *Part 3: Building a Transformer — Phase 1 exit*
- *Part 4: Training a Transformer — Phase 2 exit*
- *Shipping with each phase completion*

## references

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) (DeepSeek, 2025)
- [GDPO](https://arxiv.org/abs/2601.05242) (NVlabs, Jan 2026) · [reference implementation](https://github.com/NVlabs/GDPO)
- [GSM8K](https://arxiv.org/abs/2110.14168) (Cobbe et al., 2021)
- [TinyStories](https://arxiv.org/abs/2305.07759) (Eldan & Li, 2023)

## license

MIT

