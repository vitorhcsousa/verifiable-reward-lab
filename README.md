# rlvr-from-scratch

From transformer internals to RLVR alignment in one codebase. The simplest, cleanest implementation of the full pipeline: transformer → pretraining → SFT → GRPO → GDPO. Everything built from scratch in PyTorch, single GPU, small models, one file per concept.

nanoGPT showed you can pretrain in one file. This shows you can align in one file too.

## status

Work in progress. The repo structure and tooling are set up. No code yet — implementation starts with the attention module.

## what this will be

The full alignment pipeline, equation by equation:

1. **Transformer** — attention, RoPE, RMSNorm, FFN, full decoder-only model. No `torch.nn.MultiheadAttention`.
2. **Pretraining** — AdamW and cosine warmup from scratch (not `torch.optim`), trained on TinyStories.
3. **SFT** — supervised fine-tuning on GSM8K math reasoning. This is the baseline.
4. **GRPO** — Group Relative Policy Optimization. The algorithm behind DeepSeek-R1. No value network — advantages come from group statistics.
5. **GDPO** — the multi-reward fix. GRPO collapses when you combine rewards naively. GDPO normalizes each reward independently before combining. Per [NVlabs/GDPO](https://github.com/NVlabs/GDPO).

Target model size: 10M–50M params. Results in hours, not days.

## install

```bash
git clone https://github.com/vitorhcsousa/rlvr-from-scratch.git
cd rlvr-from-scratch
uv sync --group dev
make ci
```

## repo structure

```
src/rlvr_from_scratch/
├── model/          # transformer architecture
├── tokenizer/      # BPE from scratch
├── data/           # TinyStories, GSM8K
├── training/       # pretrain, sft, grpo, gdpo
├── rollout/        # sampling + batched generation
├── rewards/        # correctness verifier, format reward
└── evaluation/     # GSM8K eval, metrics
```

## development

```
make check      # lint + type check
make test       # run tests
make test-cov   # tests with coverage
make format     # auto-format
make ci         # full pipeline
```

## references

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) (DeepSeek, 2025)
- [GDPO](https://arxiv.org/abs/2504.12104) (NVlabs, 2025)
- [NVlabs/GDPO](https://github.com/NVlabs/GDPO) — reference implementation
- [GSM8K](https://arxiv.org/abs/2110.14168) (Cobbe et al., 2021)
- [TinyStories](https://arxiv.org/abs/2305.07759) (Eldan & Li, 2023)

## license

MIT
