# Learning with Verifiable Rewards

[![CI](https://github.com/vitorhcsousa/verifiable-reward-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/vitorhcsousa/verifiable-reward-lab/actions)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> An empirical research engineering project exploring reinforcement learning with
> verifiable rewards through reproducible baselines, controlled GRPO experiments,
> uncertainty analysis, and systematic failure analysis.

The question is whether GRPO earns its cost against a properly built SFT baseline
at this scale, and where it fails. Everything here exists to answer that honestly:
enough infrastructure to run the comparison, a protocol frozen before any result
is seen, seed variability reported next to point estimates, and a failure taxonomy
derived from inspected examples rather than assumed.

This is a study, not a tutorial and not an RL library. It implements one algorithm
and varies it along three axes. It is finite by design.

## research questions

| | |
| :-- | :-- |
| **RQ1** | Does GRPO materially improve held-out GSM8K accuracy over the chosen SFT baseline at this scale? |
| **RQ2** | How sensitive is the result to group size? |
| **RQ3** | What effect does KL regularization have? |
| **RQ4** | How does format reward change optimization behavior? |
| **RQ5** | Where does reward improve without a corresponding improvement in held-out task accuracy? |

## the study

| | |
| :-- | :-- |
| Base model | Qwen 2.5-0.5B |
| Task | GSM8K |
| Pipeline | SFT baseline → verifier → GRPO → held-out evaluation → ablations → failure analysis |
| Reported | per-seed accuracy, seed variability, reward curves, compute, limitations |

Three controlled ablations, one variable at a time: **group size**, **KL**,
**format reward**. A fourth is not added because an interesting idea appears.

## status

### Model foundations — built and tested

The technical foundation the study rests on, not its purpose. A decoder-only
transformer implemented from raw PyTorch tensors so that attention, positional
encodings, normalization, feed-forward layers, blocks, the KV-cache and sampling
are understood at the level of shapes and gradients rather than taken on trust.

**Built and tested.** The complete decoder-only transformer: scaled dot-product
and multi-head attention, RoPE / sinusoidal / learned / ALiBi positional
encodings, RMSNorm and LayerNorm, SwiGLU and GELU feed-forward, pre- and
post-norm blocks, an incremental **KV-cache that matches the full forward pass at
`atol=1e-5`**, and **greedy / temperature / top-k / top-p sampling** — 112 tests
across the model layer.

### Training foundation — built and tested

Configuration, data path, training loop, a tiny reproducible pretraining run,
metrics, evaluation, CLI.

**Built and tested.** The training path: a character-level tokenizer, corpus
fetching with a pinned checksum, encoding and batching, token-level
cross-entropy, a frozen run config, and the training loop itself — warmup plus
cosine schedule, decoupled weight decay, gradient clipping, deterministic
evaluation, and a run directory that describes itself. One command reproduces
the reference run on CPU.

### RLVR study — next

SFT baseline on GSM8K, the verifier, group sampling, GRPO, and held-out
evaluation against Qwen 2.5-0.5B. `evaluation/`, `rewards/` and `rollout/` are
placeholders until then.

### Empirical analysis — the point of the repository

GRPO against the SFT baseline; group-size sensitivity; KL regularization; format
reward; seed variability; reward improvement measured against held-out accuracy
rather than assumed to track it; and a taxonomy of failure modes derived from
inspected examples.

## quickstart

```bash
git clone https://github.com/vitorhcsousa/verifiable-reward-lab.git
cd verifiable-reward-lab
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

Generate directly from the model layer:

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
| :-- | :-- |
| Clean clone | `git clone` — no hidden local state |
| Install | `uv sync` — nothing else |
| CI | `ruff`, `ty` and `pytest` green on `master` |
| One command | `uv run rlvr-train --config configs/tiny.yaml`, from a committed config |
| Data | pinned URL, sha256-verified, never fetched by hand |
| Hardware | CPU — the acceptance run needs no accelerator |
| Runtime | **≤15 minutes** on CPU |
| Tolerance | the same seed reproduces the final validation loss within **±0.05 nats** |

Check the last one:

```bash
make reference   # runs configs/tiny.yaml twice and compares the two summaries
```

The tolerance is a property of this run — model size, step count, hardware —
not a universal constant, and it is meant to be validated on a second machine
before being frozen. What must not happen is widening it later to rescue a
reproduction that failed; that turns a test into a decoration.

Measured, same config and same seed on two machines:

| | torch | wall clock | final val loss |
| :-- | :-- | --: | --: |
| Apple Silicon, macOS, CPU | 2.11.0 | 37 s | 1.7677 nats |
| x86-64, Linux, CPU | 2.13.0 | 80 s | 1.7677 nats |

Two runs on one machine agree exactly. The two machines — different architecture,
different PyTorch build — agree to four decimal places, far inside the ±0.05
tolerance. Wall clock on other hardware will differ; the loss should not.

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

## out of scope

Deliberately, so the study stays finite and answerable:

- DAPO, GDPO, or any further algorithm added for breadth
- a generic PPO/RL framework or RLHF library
- large-scale pretraining
- distributed training infrastructure
- agent frameworks unrelated to the question

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
