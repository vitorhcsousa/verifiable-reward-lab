"""Tests for the training loop: the schedule, the plumbing, and reproducibility."""

from __future__ import annotations

import dataclasses
import json
import math
from typing import TYPE_CHECKING

import pytest
import torch

from rlvr_from_scratch.data.dataset import load_corpus
from rlvr_from_scratch.model.transformer import DecoderTransformer
from rlvr_from_scratch.training.config import TrainConfig
from rlvr_from_scratch.training.trainer import (
    configure_optimizer,
    estimate_loss,
    lr_at,
    resolve_device,
    set_seed,
    train,
)
from tests.conftest import TINY_VOCAB_SIZE, tiny_config

if TYPE_CHECKING:
    from pathlib import Path


# ---- learning-rate schedule ---------------------------------------------


def test_warmup_ramps_linearly() -> None:
    cfg = tiny_config(lr=1.0, warmup_steps=10, max_steps=100)
    assert lr_at(0, cfg) == pytest.approx(0.1)
    assert lr_at(4, cfg) == pytest.approx(0.5)
    assert lr_at(9, cfg) == pytest.approx(1.0)


def test_warmup_does_not_start_at_zero() -> None:
    """A first step taken at lr exactly 0 is a step thrown away."""
    cfg = tiny_config(lr=1.0, warmup_steps=100, max_steps=100)
    assert lr_at(0, cfg) > 0.0


def test_schedule_is_continuous_at_the_handover() -> None:
    """The last warmup step and the first cosine step must agree."""
    cfg = tiny_config(lr=1.0, warmup_steps=10, max_steps=100)
    assert lr_at(9, cfg) == pytest.approx(lr_at(10, cfg))


def test_zero_warmup_starts_at_full_lr() -> None:
    cfg = tiny_config(lr=1.0, warmup_steps=0, max_steps=100)
    assert lr_at(0, cfg) == pytest.approx(1.0)


def test_cosine_reaches_the_floor_and_not_zero() -> None:
    cfg = tiny_config(lr=1.0, warmup_steps=10, max_steps=100, min_lr_ratio=0.1)
    assert lr_at(100, cfg) == pytest.approx(0.1)


def test_lr_never_leaves_the_band() -> None:
    cfg = tiny_config(lr=1.0, warmup_steps=10, max_steps=100, min_lr_ratio=0.1)
    floor = cfg.min_lr_ratio * cfg.lr
    values = [lr_at(s, cfg) for s in range(cfg.max_steps + 20)]
    # past max_steps the cosine is clamped, so the floor holds forever
    assert min(values) >= floor - 1e-12
    assert max(values) <= cfg.lr + 1e-12


def test_lr_decays_monotonically_after_warmup() -> None:
    cfg = tiny_config(lr=1.0, warmup_steps=10, max_steps=100)
    after = [lr_at(s, cfg) for s in range(cfg.warmup_steps, cfg.max_steps + 1)]
    pairs = zip(after[:-1], after[1:], strict=True)
    assert all(nxt <= cur + 1e-12 for cur, nxt in pairs)


# ---- optimiser ----------------------------------------------------------


def test_decay_is_applied_only_to_matrices() -> None:
    """Decaying a norm gain pulls the layer's ability to scale towards zero."""
    cfg = tiny_config()
    model = DecoderTransformer(cfg.model)
    optimizer = configure_optimizer(model, cfg)

    decay, no_decay = optimizer.param_groups
    assert decay["weight_decay"] == cfg.weight_decay
    assert no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() < 2 for p in no_decay["params"])


def test_every_parameter_lands_in_exactly_one_group() -> None:
    cfg = tiny_config()
    model = DecoderTransformer(cfg.model)
    optimizer = configure_optimizer(model, cfg)

    grouped = sum(len(g["params"]) for g in optimizer.param_groups)
    assert grouped == sum(1 for p in model.parameters() if p.requires_grad)


# ---- seeding ------------------------------------------------------------


def test_same_seed_gives_identical_initial_weights() -> None:
    cfg = tiny_config()
    set_seed(123)
    a = DecoderTransformer(cfg.model)
    set_seed(123)
    b = DecoderTransformer(cfg.model)

    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.equal(pa, pb)


def test_resolve_device_rejects_an_absent_accelerator() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available here, so this cannot be tested")
    with pytest.raises(ValueError, match="CUDA"):
        resolve_device("cuda")


# ---- evaluation ---------------------------------------------------------


def test_untrained_loss_is_about_ln_vocab(tiny_corpus: Path) -> None:
    """An untrained model must be exactly as confused as a uniform guess.

    If this number is wrong, every curve downstream measures the wrong
    thing — and it would still look like learning.
    """
    cfg = tiny_config()
    set_seed(cfg.seed)
    corpus = load_corpus(tiny_corpus, train_frac=cfg.train_frac)
    model = DecoderTransformer(cfg.model)

    losses = estimate_loss(model, corpus, config=cfg, device=torch.device("cpu"))
    assert losses["val"] == pytest.approx(math.log(TINY_VOCAB_SIZE), abs=0.2)


def test_evaluation_is_repeatable(tiny_corpus: Path) -> None:
    """Same model, same config, same number — or a moved curve means nothing."""
    cfg = tiny_config()
    corpus = load_corpus(tiny_corpus, train_frac=cfg.train_frac)
    model = DecoderTransformer(cfg.model)
    device = torch.device("cpu")

    first = estimate_loss(model, corpus, config=cfg, device=device)
    second = estimate_loss(model, corpus, config=cfg, device=device)
    assert first == second


def test_evaluation_leaves_the_model_in_training_mode(tiny_corpus: Path) -> None:
    cfg = tiny_config()
    corpus = load_corpus(tiny_corpus, train_frac=cfg.train_frac)
    model = DecoderTransformer(cfg.model)
    model.train()

    estimate_loss(model, corpus, config=cfg, device=torch.device("cpu"))
    assert model.training


# ---- the loop -----------------------------------------------------------


def test_train_writes_the_whole_record(tmp_path: Path, tiny_corpus: Path) -> None:
    """A run directory has to be handable to someone else on its own."""
    out = tmp_path / "run"
    train(tiny_config(), out_dir=out, corpus_path=tiny_corpus, verbose=False)

    for name in ("config.yaml", "tokenizer.json", "metrics.jsonl", "summary.json"):
        assert (out / name).exists(), f"{name} missing from the run directory"
    assert (out / "ckpt.pt").exists()


def test_saved_config_round_trips(tmp_path: Path, tiny_corpus: Path) -> None:
    """The config beside the metrics must be the run that happened."""
    out = tmp_path / "run"
    cfg = tiny_config(seed=5)
    train(cfg, out_dir=out, corpus_path=tiny_corpus, verbose=False)
    assert TrainConfig.load(out / "config.yaml") == cfg


def test_metrics_hold_one_line_per_evaluation(
    tmp_path: Path, tiny_corpus: Path
) -> None:
    out = tmp_path / "run"
    cfg = tiny_config(max_steps=40, eval_interval=10)
    train(cfg, out_dir=out, corpus_path=tiny_corpus, verbose=False)

    lines = (out / "metrics.jsonl").read_text(encoding="utf-8").strip().split("\n")
    records = [json.loads(line) for line in lines]
    assert [r["step"] for r in records] == [10, 20, 30, 40]
    assert all("train_loss" in r and "val_loss" in r for r in records)


def test_training_actually_reduces_the_loss(tmp_path: Path, tiny_corpus: Path) -> None:
    """Forty steps on one repeated sentence has to beat a uniform guess."""
    result = train(
        tiny_config(),
        out_dir=tmp_path / "run",
        corpus_path=tiny_corpus,
        verbose=False,
    )
    assert result.final_val_loss < math.log(TINY_VOCAB_SIZE) - 0.3


def test_result_matches_the_summary_on_disk(tmp_path: Path, tiny_corpus: Path) -> None:
    out = tmp_path / "run"
    result = train(tiny_config(), out_dir=out, corpus_path=tiny_corpus, verbose=False)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["final_val_loss"] == pytest.approx(result.final_val_loss)
    assert summary["num_params"] == result.num_params


def test_same_seed_reproduces_the_run(tmp_path: Path, tiny_corpus: Path) -> None:
    """The claim the whole repository rests on, at smoke-test scale."""
    cfg = tiny_config(seed=11)
    a = train(cfg, out_dir=tmp_path / "a", corpus_path=tiny_corpus, verbose=False)
    b = train(cfg, out_dir=tmp_path / "b", corpus_path=tiny_corpus, verbose=False)
    assert a.final_val_loss == pytest.approx(b.final_val_loss, abs=1e-9)


def test_a_different_seed_changes_the_run(tmp_path: Path, tiny_corpus: Path) -> None:
    """Otherwise 'reproducible' would only mean 'independent of the seed'."""
    a = train(
        tiny_config(seed=1),
        out_dir=tmp_path / "a",
        corpus_path=tiny_corpus,
        verbose=False,
    )
    b = train(
        tiny_config(seed=2),
        out_dir=tmp_path / "b",
        corpus_path=tiny_corpus,
        verbose=False,
    )
    assert a.final_val_loss != pytest.approx(b.final_val_loss, abs=1e-9)


def test_checkpoint_carries_the_config_that_produced_it(
    tmp_path: Path, tiny_corpus: Path
) -> None:
    """A state_dict with no config beside it is a bag of numbers."""
    out = tmp_path / "run"
    cfg = tiny_config()
    train(cfg, out_dir=out, corpus_path=tiny_corpus, verbose=False)

    ckpt = torch.load(out / "ckpt.pt", weights_only=False)
    assert TrainConfig.from_dict(ckpt["config"]) == cfg
    assert ckpt["step"] > 0


def test_vocab_mismatch_names_both_numbers(tmp_path: Path, tiny_corpus: Path) -> None:
    """The one disagreement that yields a plausible curve for a wrong reason."""
    cfg = tiny_config()
    wrong = dataclasses.replace(
        cfg, model=dataclasses.replace(cfg.model, vocab_size=TINY_VOCAB_SIZE + 5)
    )
    with pytest.raises(ValueError) as exc:
        train(wrong, out_dir=tmp_path / "run", corpus_path=tiny_corpus, verbose=False)

    assert str(TINY_VOCAB_SIZE) in str(exc.value)
    assert str(TINY_VOCAB_SIZE + 5) in str(exc.value)
