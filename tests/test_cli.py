"""
Tests for the one command a stranger runs.

The interesting property is not that the CLI trains — trainer.py's tests
cover that. It is that the config written *next to the metrics* is the run
that actually happened. An override that changes a run without changing its
record is worse than no override at all: it makes the directory lie.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from rlvr_from_scratch.cli import main
from rlvr_from_scratch.training.config import TrainConfig
from tests.conftest import tiny_config

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """A saved tiny config, written the way `save` writes one."""
    path = tmp_path / "tiny.yaml"
    tiny_config().save(path)
    return path


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch, tiny_corpus: Path) -> None:
    """Point every run in this module at the local corpus.

    Patched at the trainer, not the CLI, precisely so the CLI still goes
    through its real path — argument parsing, loading, overrides, train().
    """
    monkeypatch.setattr(
        "rlvr_from_scratch.training.trainer.resolve_corpus_path",
        lambda config: tiny_corpus,  # noqa: ARG005
    )


def test_a_run_completes_and_reports_success(tmp_path: Path, config_file: Path) -> None:
    out = tmp_path / "run"
    assert main(["--config", str(config_file), "--out", str(out), "--quiet"]) == 0
    assert (out / "summary.json").exists()


def test_default_output_directory_is_named_after_the_config(
    tmp_path: Path, config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """runs/<stem>, so two configs cannot silently overwrite each other."""
    monkeypatch.chdir(tmp_path)
    assert main(["--config", str(config_file), "--quiet"]) == 0
    assert (tmp_path / "runs" / "tiny" / "summary.json").exists()


def test_overrides_are_written_into_the_record(
    tmp_path: Path, config_file: Path
) -> None:
    out = tmp_path / "run"
    code = main(
        [
            "--config",
            str(config_file),
            "--out",
            str(out),
            "--seed",
            "7",
            "--max-steps",
            "5",
            "--quiet",
        ]
    )
    assert code == 0

    saved = TrainConfig.load(out / "config.yaml")
    assert saved.seed == 7
    assert saved.max_steps == 5


def test_shortening_a_run_clamps_the_warmup(tmp_path: Path, config_file: Path) -> None:
    """--max-steps 2 is how a smoke run is written; it must not be rejected."""
    out = tmp_path / "run"
    assert (
        main(
            [
                "--config",
                str(config_file),
                "--out",
                str(out),
                "--max-steps",
                "2",
                "--quiet",
            ]
        )
        == 0
    )
    assert TrainConfig.load(out / "config.yaml").warmup_steps <= 2


def test_a_missing_config_fails_loudly(tmp_path: Path) -> None:
    assert main(["--config", str(tmp_path / "nope.yaml"), "--quiet"]) == 1


def test_a_config_that_is_not_a_mapping_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    assert main(["--config", str(path), "--quiet"]) == 1


def test_an_impossible_override_fails_before_training(
    tmp_path: Path, config_file: Path
) -> None:
    """Validation happens on replace, so this cannot get halfway through."""
    out = tmp_path / "run"
    assert (
        main(["--config", str(config_file), "--out", str(out), "--max-steps", "0"]) == 1
    )
    assert not (out / "summary.json").exists()
