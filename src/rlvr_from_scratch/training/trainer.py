"""
The training loop. One config in, one directory of evidence out.

Nothing here decides anything. Every number that moves the loss curve comes
from TrainConfig, and this file is the machinery that turns that object into
a run: seed the world, fetch and verify the corpus, build the model, step the
optimiser, measure, and write down what happened.

Three properties are worth more than the loop itself.

*Reproducible.* Every source of randomness is seeded from config.seed, and
batches are drawn from explicit torch.Generators rather than the global RNG.
Same config, same machine, same numbers — which is what makes "the val loss
moved" a claim rather than an impression.

*Comparable.* Evaluation re-seeds its own generator on every call, so the val
number at step 100 and the one at step 2000 are measured on exactly the same
windows. Resampling the eval set each time adds a variance that has nothing
to do with the model and swamps small real differences.

*Self-describing.* A run writes its config, its tokenizer, its metrics and a
summary next to each other. A directory you can hand to someone else is the
unit of work here; a number in a terminal that has since scrolled away is not.

Reference: nanoGPT's train.py, minus the distributed path, plus the record
keeping.
"""

from __future__ import annotations

import json
import math
import platform
import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from rlvr_from_scratch.data import fetch
from rlvr_from_scratch.data.dataset import get_batch, load_corpus
from rlvr_from_scratch.model.transformer import DecoderTransformer
from rlvr_from_scratch.training.losses import cross_entropy_loss

if TYPE_CHECKING:
    from pathlib import Path

    from rlvr_from_scratch.data.dataset import Corpus
    from rlvr_from_scratch.training.config import TrainConfig

# Offset so the eval generator can never coincide with the training one.
# Same seed on both would evaluate on windows the model just trained on.
_EVAL_SEED_OFFSET = 1_000_003


# =========================================================================
# Determinism
# =========================================================================


def set_seed(seed: int) -> None:
    """Seed every RNG a run touches.

    torch alone is not enough: the model init path is torch, but anything
    reaching for `random` or numpy — now or three files from now — would
    silently reintroduce run-to-run variation that no config records.

    Args:
        seed: The run's seed, straight from TrainConfig.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str) -> torch.device:
    """Turn a config's device string into a device, or say why not.

    Args:
        name: "cpu", "mps", "cuda", or any torch device string.

    Returns:
        The resolved device.

    Raises:
        ValueError: If the requested accelerator is not available here. A
            run that silently falls back to CPU produces different numbers
            than the config it was launched with claims.
    """
    if name.startswith("cuda") and not torch.cuda.is_available():
        msg = f"device {name!r} requested but CUDA is not available here"
        raise ValueError(msg)
    if name == "mps" and not torch.backends.mps.is_available():
        msg = "device 'mps' requested but the MPS backend is not available here"
        raise ValueError(msg)
    return torch.device(name)


# =========================================================================
# Learning-rate schedule
# =========================================================================


def lr_at(step: int, config: TrainConfig) -> float:
    """Learning rate for `step`: linear warmup, then cosine decay.

    The warmup exists because Adam's second-moment estimate is garbage for
    the first few steps — it has seen almost no gradients — and a full-size
    step taken on a garbage denominator is how a run diverges in the first
    fifty iterations and never recovers.

    The cosine floor is `min_lr_ratio * lr`, not zero. A schedule that
    decays to nothing spends its last steps not learning; one that decays
    to a tenth keeps making small corrections to the end.

    Args:
        step:   0-indexed training step.
        config: The run's config; reads lr, warmup_steps, max_steps and
                min_lr_ratio.

    Returns:
        The learning rate to set on the optimiser before this step.
    """
    min_lr = config.lr * config.min_lr_ratio

    # ---- warmup: ramp over the first warmup_steps ----
    # (step + 1) rather than step, so the very first step is not taken at a
    # learning rate of exactly zero, which wastes it.
    if config.warmup_steps > 0 and step < config.warmup_steps:
        return config.lr * (step + 1) / config.warmup_steps

    # ---- cosine: lr -> min_lr over what is left ----
    decay_steps = max(1, config.max_steps - config.warmup_steps)
    progress = (step - config.warmup_steps) / decay_steps
    progress = min(1.0, max(0.0, progress))
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (config.lr - min_lr)


# =========================================================================
# Optimiser
# =========================================================================


def configure_optimizer(
    model: DecoderTransformer, config: TrainConfig
) -> torch.optim.AdamW:
    """AdamW with weight decay applied only where it means something.

    Decay is a pull towards zero. On a weight matrix that is regularisation;
    on a norm gain or a bias it is a pull towards deleting the layer's
    ability to scale or shift at all. The split is by dimensionality because
    that is exactly what separates the two: matrices and embeddings are 2-D,
    gains and biases are 1-D.

    Args:
        model:  The model whose parameters to optimise.
        config: Reads lr, betas and weight_decay.

    Returns:
        A configured AdamW over two parameter groups.
    """
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]

    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=config.lr, betas=config.betas)


# =========================================================================
# Evaluation
# =========================================================================


@torch.no_grad()
def estimate_loss(
    model: DecoderTransformer,
    corpus: Corpus,
    *,
    config: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    """Mean loss over a fixed sample of both splits.

    The generator is re-seeded here on every call, which is the whole point:
    the same windows are scored at every evaluation, so a change in the
    number is a change in the model. Drawing fresh windows each time adds
    sampling noise of the same order as the differences being looked for.

    A single batch is not a measurement — eval_iters of them, averaged, is.

    Args:
        model:  The model, left in whatever mode it arrived in.
        corpus: The encoded corpus.
        config: Reads batch_size, block_size, eval_iters and seed.
        device: Where to run.

    Returns:
        {"train": float, "val": float}, both in nats per token.
    """
    was_training = model.training
    model.eval()

    out: dict[str, float] = {}
    for split in ("train", "val"):
        # one generator per split, both derived from the run's seed, so the
        # eval sample is a property of the config and nothing else.
        generator = torch.Generator().manual_seed(config.seed + _EVAL_SEED_OFFSET)
        total = 0.0
        for _ in range(config.eval_iters):
            # (B, T), (B, T)
            x, y = get_batch(
                corpus,
                split,
                batch_size=config.batch_size,
                block_size=config.block_size,
                generator=generator,
            )
            x, y = x.to(device), y.to(device)
            # (B, T) -> (B, T, V)
            logits, _ = model(x)
            total += cross_entropy_loss(logits, y).item()
        out[split] = total / config.eval_iters

    if was_training:
        model.train()
    return out


# =========================================================================
# Result
# =========================================================================


@dataclass(frozen=True)
class TrainResult:
    """What a finished run is, reduced to numbers.

    Returned by train() and written to summary.json, so the object a test
    asserts on and the file a reader opens carry the same fields.
    """

    steps: int
    final_train_loss: float
    final_val_loss: float
    best_val_loss: float
    best_step: int
    wall_clock_s: float
    num_params: int

    def to_dict(self) -> dict[str, float | int]:
        """Plain-data view, ready for json.dump."""
        return {
            "steps": self.steps,
            "final_train_loss": self.final_train_loss,
            "final_val_loss": self.final_val_loss,
            "best_val_loss": self.best_val_loss,
            "best_step": self.best_step,
            "wall_clock_s": self.wall_clock_s,
            "num_params": self.num_params,
        }


# =========================================================================
# The loop
# =========================================================================


def resolve_corpus_path(config: TrainConfig) -> Path:
    """Ensure the corpus named by the config is on disk and verified.

    This is what keeps "one command" honest: the run downloads its own data
    from the pinned URL and checks the sha256 before training on it. There
    is no step where a reader is told to go and fetch a file.

    Args:
        config: Reads `corpus`, a key into fetch.SOURCES.

    Returns:
        Path to the verified corpus file.

    Raises:
        ValueError: If the corpus name is unknown, or the bytes on disk do
            not match the pinned hash.
    """
    if config.corpus not in fetch.SOURCES:
        msg = (
            f"unknown corpus {config.corpus!r}; "
            f"fetch.SOURCES has {sorted(fetch.SOURCES)}"
        )
        raise ValueError(msg)
    return fetch.fetch(fetch.SOURCES[config.corpus])


def train(
    config: TrainConfig,
    *,
    out_dir: Path,
    corpus_path: Path | None = None,
    verbose: bool = True,
) -> TrainResult:
    """Run one training job and write everything it produced to `out_dir`.

    Args:
        config:      The frozen description of the run.
        out_dir:     Directory to create and fill with config.yaml,
                     tokenizer.json, metrics.jsonl, summary.json and ckpt.pt.
        corpus_path: Override the corpus file. Tests pass a small local file
                     so that nothing here touches the network; a real run
                     leaves this None and the pinned corpus is fetched.
        verbose:     Print one line per evaluation.

    Returns:
        The run's TrainResult.

    Raises:
        ValueError: If the model's vocab_size disagrees with the corpus, or
            the requested device is unavailable.
    """
    started = time.perf_counter()

    # =========================================
    # 1. Fix the world before anything samples from it
    # =========================================
    set_seed(config.seed)
    device = resolve_device(config.device)

    # =========================================
    # 2. Data
    # =========================================
    path = corpus_path if corpus_path is not None else resolve_corpus_path(config)
    corpus = load_corpus(path, train_frac=config.train_frac)

    # The one disagreement that produces a plausible-looking loss curve for
    # the wrong reason: a model whose embedding table is a different width
    # than the vocabulary it is being fed. Too small and encode() emits ids
    # past the end of the table; too large and rows are dead weight the loss
    # never sees.
    if config.model.vocab_size != corpus.vocab_size:
        msg = (
            f"config.model.vocab_size is {config.model.vocab_size} but the "
            f"corpus at {path} has {corpus.vocab_size} distinct characters"
        )
        raise ValueError(msg)

    # =========================================
    # 3. Model and optimiser
    # =========================================
    model = DecoderTransformer(config.model).to(device)
    model.train()
    optimizer = configure_optimizer(model, config)
    num_params = model.num_params()

    # =========================================
    # 4. Open the record
    # =========================================
    out_dir.mkdir(parents=True, exist_ok=True)
    config.save(out_dir / "config.yaml")
    corpus.tokenizer.save(out_dir / "tokenizer.json")
    metrics_path = out_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")  # a rerun starts a new record

    if verbose:
        print(
            f"{num_params:,} non-embedding params | "
            f"vocab {corpus.vocab_size} | "
            f"train {len(corpus.train):,} tok | val {len(corpus.val):,} tok | "
            f"device {device}"
        )

    # Training batches come from one generator advanced across the whole
    # run, so step k sees the same windows on every rerun of this config.
    train_generator = torch.Generator().manual_seed(config.seed)

    best_val = math.inf
    best_step = -1
    last = {"train": math.nan, "val": math.nan}

    # =========================================
    # 5. Step
    # =========================================
    for step in range(config.max_steps):
        lr = lr_at(step, config)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # (B, T), (B, T)
        x, y = get_batch(
            corpus,
            "train",
            batch_size=config.batch_size,
            block_size=config.block_size,
            generator=train_generator,
        )
        x, y = x.to(device), y.to(device)

        # (B, T) -> (B, T, V)
        logits, _ = model(x)
        loss = cross_entropy_loss(logits, y)

        # set_to_none frees the buffers instead of filling them with zeros:
        # cheaper, and a parameter that receives no gradient this step ends
        # up with grad None rather than a stale zero that AdamW still steps.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Clip the global norm, not per-parameter. One exploding layer
        # should scale the whole update down, not be flattened on its own
        # while every other layer keeps its direction.
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        # ---- measure ----
        is_last = step == config.max_steps - 1
        if (step + 1) % config.eval_interval == 0 or is_last:
            last = estimate_loss(model, corpus, config=config, device=device)
            elapsed = time.perf_counter() - started

            record = {
                "step": step + 1,
                "train_loss": last["train"],
                "val_loss": last["val"],
                "lr": lr,
                "elapsed_s": round(elapsed, 3),
            }
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            if verbose:
                print(
                    f"step {step + 1:>6} | train {last['train']:.4f} | "
                    f"val {last['val']:.4f} | lr {lr:.2e} | {elapsed:6.1f}s"
                )

            if last["val"] < best_val:
                best_val, best_step = last["val"], step + 1
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": config.to_dict(),
                        "step": best_step,
                        "val_loss": best_val,
                    },
                    out_dir / "ckpt.pt",
                )

    # =========================================
    # 6. Close the record
    # =========================================
    result = TrainResult(
        steps=config.max_steps,
        final_train_loss=last["train"],
        final_val_loss=last["val"],
        best_val_loss=best_val,
        best_step=best_step,
        wall_clock_s=time.perf_counter() - started,
        num_params=num_params,
    )

    summary = {
        **result.to_dict(),
        "config": config.to_dict(),
        "corpus_path": str(path),
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    return result
