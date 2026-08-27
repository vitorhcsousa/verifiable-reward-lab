# rlvr-from-scratch

[![CI](https://github.com/vitorhcsousa/rlvr-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/vitorhcsousa/rlvr-from-scratch/actions)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**A decoder-only language model, built from raw tensors.**

Attention through to a model that trains — one file per concept, every component
shape-, numerical- and gradient-tested. Single GPU, small models, pure PyTorch:
no `trl`, no `accelerate` magic.

In the spirit of nanoGPT: small enough to read end to end, tested enough to trust.

## status

**Built and tested.** The complete decoder-only transformer: scaled dot-product
and multi-head attention, RoPE / sinusoidal / learned / ALiBi positional
encodings, RMSNorm and LayerNorm, SwiGLU and GELU feed-forward, pre- and
post-norm blocks, an incremental **KV-cache that matches the full forward pass at
`atol=1e-5`**, and **greedy / temperature / top-k / top-p sampling** — 112 tests
across the model layer.

**Built and tested.** The training path: a character-level tokenizer, corpus
fetching with a pinned checksum, encoding and batching, token-level
cross-entropy, a frozen run config, and the training loop itself — warmup plus
cosine schedule, decoupled weight decay, gradient clipping, deterministic
evaluation, and a run directory that describes itself. One command reproduces
the reference run on CPU.

**Next.** The RLVR study: verifier, rollouts, GRPO, and the empirical work on
GSM8K. `evaluation/`, `rewards/` and `rollout/` are placeholders until then.

## quickstart

```bash
git clone https://github.com/vitorhcsousa/rlvr-from-scratch.git
cd rlvr-from-scratch
uv sync --group dev
make data    # download + checksum-verify the training corpus
make check   # ruff + ty + pytest
```

The corpus is not versioned. `make data` fetches it from a pinned URL and
verifies a sha256 recorded in `src/rlvr_from_scratch/data/fetch.py`, failing
loudly if the bytes differ. It is idempotent — safe to re-run.

Then train:

```bash
uv run rlvr-train --config configs/tiny.yaml
```

That is the whole run. It fetches and checksums its own corpus, so `make data`
above is a convenience rather than a prerequisite; there is no notebook, no
manual data step, and no flag that has to be remembered for the numbers to come
out right. It writes `runs/tiny/` containing `config.yaml`, `tokenizer.json`,
`metrics.jsonl`, `summary.json` and `ckpt.pt` — the config beside the metrics is
the run that actually happened, overrides included.

Generate with a randomly initialized model (the training loop is in progress):

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

## reproducibility

The claim this repository makes about itself, stated so it can be checked:

| | |
|---|---|
| Clean clone | `git clone` + `uv sync` — no hidden local state |
| One command | `uv run rlvr-train --config configs/tiny.yaml` |
| Data | fetched from a pinned URL, sha256-verified, never fetched by hand |
| Hardware | the reference run is a **CPU** run |
| Tolerance | the same seed reproduces the final validation loss within **±0.05 nats** |

Check the last one:

```bash
make reference   # runs configs/tiny.yaml twice and compares the two summaries
```

The tolerance is a property of this run — model size, step count, hardware —
not a universal constant, and it is meant to be validated on a second machine
before being frozen. What must not happen is widening it later to rescue a
reproduction that failed; that turns a test into a decoration.

For scale: on one x86-64 Linux CPU the run takes about 80 seconds and ends at
1.768 nats, and two runs of the same seed agreed to four decimal places. Wall
clock on other hardware will differ — what must not differ is the two runs from
one seed.

## what's tested

- **Shapes** for every component (attention, MHA, RoPE/sinusoidal, norms, FFN, block, model).
- **Numerics**: causality (future tokens never change past logits), centred finite-difference gradient checks end-to-end, pre/post-norm variants.
- **KV-cache correctness**: incremental decoding ≡ full forward at `atol=1e-5` — the signature correctness test of the model layer.
- **Sampling acceptance**: greedy == argmax; `top_k=1` == greedy; `top_p=1.0` == full-softmax sampling; nucleus renormalizes with out-of-set probability exactly 0; seeded draws reproducible.
- **Data path**: encode/decode round-trips, vocabulary ordering, the (x, y) shift, and a corpus checksum that fails loudly on the wrong bytes.
- **Objective**: an untrained model scores exactly ln(V), the reduction is a mean, and one optimizer step provably decreases the loss.
- **Config**: a run that cannot happen is rejected on construction, and a config survives a YAML round trip as the same object — including the tuple that YAML would quietly hand back as a list.
- **Training loop**: the schedule is continuous at the warmup handover and never leaves its band; weight decay reaches matrices and not norm gains; evaluation returns the same number twice; the same seed reproduces a run and a different seed does not.

## repo layout

```text
src/rlvr_from_scratch/
├── model/          # attention, positional, norm, ffn, block, transformer, sampling
├── tokenizer/      # character-level tokenizer
├── data/           # corpus fetch + checksum, encoding, batching
├── training/       # run config, objective, the loop
├── evaluation/     # held-out task evaluation          (RLVR, not yet written)
├── rewards/        # verifier and format reward        (RLVR, not yet written)
├── rollout/        # group sampling and advantages     (RLVR, not yet written)
├── cli.py          # rlvr-train
└── reproduce.py    # rlvr-compare
configs/
└── tiny.yaml       # the reference run
```

## development

```text
make data       # download + verify the training corpus
make train      # the reference run
make reference  # the reference run twice, then check the two agree
make check      # lint + type check + tests (what CI runs)
make test       # tests only
make test-cov   # tests with coverage
make format     # auto-format
make ci         # full pipeline
```

## writing

Long-form articles documenting the math and the code:

- **Part 1** · [Attention Is All You Need to Implement](https://www.vitorsousa.com/foundations/attention-from-scratch/) — scaled dot-product and multi-head attention
- **Part 2** · [Positional Encoding: Teaching Transformers to Count](https://www.vitorsousa.com/foundations/positional-encoding/) — sinusoidal, learned, RoPE, ALiBi
- **Part 3** · [Building a Transformer: The Complete Forward Pass](https://www.vitorsousa.com/foundations/building-a-transformer/) — norms, residuals, FFN, and the assembled decoder
- _Part 4: Training a Transformer — in progress_
- [A Transformer from Raw Tensors: What 112 Tests Taught Me](https://www.vitorsousa.com/blog/transformer-from-raw-tensors/) — the three properties that fail silently, and the tests that pin them

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
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (Su et al., 2021)
- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) (Holtzman et al., 2019)

## license

MIT
