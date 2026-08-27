"""The training loop: one config in, one directory of evidence out.

uv run rlvr-train --config configs/tiny.yaml
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

# keeps the eval generator off the training one, which would otherwise
# evaluate on the windows the model just trained on
_EVAL_SEED_OFFSET = 1_000_003


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch from the run's seed.

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
        ValueError: If the requested accelerator is not available here.
    """
    # no silent fallback: a run on cpu produces different numbers than the
    # config it was launched with claims
    if name.startswith("cuda") and not torch.cuda.is_available():
        msg = f"device {name!r} requested but CUDA is not available here"
        raise ValueError(msg)
    if name == "mps" and not torch.backends.mps.is_available():
        msg = "device 'mps' requested but the MPS backend is not available here"
        raise ValueError(msg)
    return torch.device(name)


def lr_at(step: int, config: TrainConfig) -> float:
    """Learning rate for `step`: linear warmup, then cosine decay.

    Args:
        step:   0-indexed training step.
        config: Reads lr, warmup_steps, max_steps and min_lr_ratio.

    Returns:
        The learning rate to set before this step.
    """
    min_lr = config.lr * config.min_lr_ratio

    # 1) warmup. adam's second moment is garbage for the first few steps and
    #    a full-size step on a garbage denominator is how a run diverges.
    #    (step + 1) so the very first step is not taken at lr 0.
    if config.warmup_steps > 0 and step < config.warmup_steps:
        return config.lr * (step + 1) / config.warmup_steps

    # 2) cosine down to min_lr, not to zero: a schedule that decays to
    #    nothing spends its last steps not learning
    decay_steps = max(1, config.max_steps - config.warmup_steps)
    progress = (step - config.warmup_steps) / decay_steps
    progress = min(1.0, max(0.0, progress))
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (config.lr - min_lr)


def configure_optimizer(
    model: DecoderTransformer, config: TrainConfig
) -> torch.optim.AdamW:
    """AdamW with weight decay on matrices only.

    Args:
        model:  The model whose parameters to optimise.
        config: Reads lr, betas and weight_decay.

    Returns:
        A configured AdamW over two parameter groups.
    """
    # any parameter that is 2D is weight decayed, otherwise not: matmul
    # weights and embeddings decay, biases and norm gains don't. decaying a
    # norm gain pulls the layer's ability to scale towards zero.
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]

    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=config.lr, betas=config.betas)


@torch.no_grad()
def estimate_loss(
    model: DecoderTransformer,
    corpus: Corpus,
    *,
    config: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    """Mean loss over a fixed sample of both splits.

    Args:
        model:  The model, left in whatever mode it arrived in.
        config: Reads batch_size, block_size, eval_iters and seed.
        device: Where to run.

    Returns:
        {"train": float, "val": float}, in nats per token.
    """
    was_training = model.training
    model.eval()

    out: dict[str, float] = {}
    for split in ("train", "val"):
        # re-seeded per call, so every evaluation scores the same windows.
        # resampling adds noise of the same order as the differences being
        # looked for.
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


@dataclass(frozen=True)
class TrainResult:
    """What a finished run is, reduced to numbers. Also written to summary.json."""

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


def resolve_corpus_path(config: TrainConfig) -> Path:
    """Ensure the corpus named by the config is on disk and verified.

    Args:
        config: Reads `corpus`, a key into fetch.SOURCES.

    Returns:
        Path to the verified corpus file.

    Raises:
        ValueError: On an unknown corpus name, or bytes that do not match
            the pinned hash.
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
        out_dir:     Directory to fill with config.yaml, tokenizer.json,
                     metrics.jsonl, summary.json and ckpt.pt.
        corpus_path: Override the corpus file. Tests pass a small local one
                     so nothing here touches the network.
        verbose:     Print one line per evaluation.

    Returns:
        The run's TrainResult.

    Raises:
        ValueError: On a vocab_size that disagrees with the corpus, or an
            unavailable device.
    """
    started = time.perf_counter()

    set_seed(config.seed)
    device = resolve_device(config.device)

    path = corpus_path if corpus_path is not None else resolve_corpus_path(config)
    corpus = load_corpus(path, train_frac=config.train_frac)

    # the one disagreement that yields a plausible curve for a wrong reason:
    # too small and encode() emits ids past the end of the table, too large
    # and rows are dead weight the loss never sees
    if config.model.vocab_size != corpus.vocab_size:
        msg = (
            f"config.model.vocab_size is {config.model.vocab_size} but the "
            f"corpus at {path} has {corpus.vocab_size} distinct characters"
        )
        raise ValueError(msg)

    model = DecoderTransformer(config.model).to(device)
    model.train()
    optimizer = configure_optimizer(model, config)
    num_params = model.num_params()

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

    # one generator advanced across the whole run, so step k sees the same
    # windows on every rerun of this config
    train_generator = torch.Generator().manual_seed(config.seed)

    best_val = math.inf
    best_step = -1
    last = {"train": math.nan, "val": math.nan}

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

        # set_to_none, so a parameter with no gradient this step ends up
        # None rather than a stale zero that AdamW still steps
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # global norm, not per-parameter: one exploding layer should scale
        # the whole update down, not be flattened on its own
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

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
