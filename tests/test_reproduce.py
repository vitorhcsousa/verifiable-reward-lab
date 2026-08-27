"""Tests for the reproducibility check, in both directions."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from rlvr_from_scratch.reproduce import DEFAULT_TOLERANCE, main, read_val_loss

if TYPE_CHECKING:
    from pathlib import Path


def write_run(path: Path, val_loss: float) -> Path:
    """A minimal run directory: just the summary the checker reads."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(
        json.dumps({"final_val_loss": val_loss}), encoding="utf-8"
    )
    return path


def test_reads_the_final_val_loss(tmp_path: Path) -> None:
    assert read_val_loss(write_run(tmp_path / "a", 1.234)) == pytest.approx(1.234)


def test_agreement_inside_the_tolerance_passes(tmp_path: Path) -> None:
    a = write_run(tmp_path / "a", 1.500)
    b = write_run(tmp_path / "b", 1.500 + DEFAULT_TOLERANCE / 2)
    assert main([str(a), str(b)]) == 0


def test_disagreement_outside_the_tolerance_fails(tmp_path: Path) -> None:
    """The half that matters: this is the test that can actually fail a run."""
    a = write_run(tmp_path / "a", 1.500)
    b = write_run(tmp_path / "b", 1.500 + DEFAULT_TOLERANCE * 2)
    assert main([str(a), str(b)]) == 1


def test_a_missing_summary_fails_rather_than_passing_vacuously(
    tmp_path: Path,
) -> None:
    a = write_run(tmp_path / "a", 1.5)
    assert main([str(a), str(tmp_path / "not-a-run")]) == 1


def test_a_summary_without_the_field_fails(tmp_path: Path) -> None:
    a = write_run(tmp_path / "a", 1.5)
    b = tmp_path / "b"
    b.mkdir()
    (b / "summary.json").write_text(json.dumps({"steps": 10}), encoding="utf-8")
    assert main([str(a), str(b)]) == 1
