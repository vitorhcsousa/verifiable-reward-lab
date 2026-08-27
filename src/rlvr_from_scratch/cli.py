"""
The one command a stranger runs.

    uv run rlvr-train --config configs/tiny.yaml

That line has to be the whole story: it downloads and checksums its own
corpus, trains, evaluates, and leaves a directory that describes what
happened. No notebook, no manual data step, no flag that has to be
remembered to make the numbers come out right.

Overrides exist for the three things that are genuinely about *this*
invocation rather than about the experiment — where to run, which seed, how
long. Every one of them is folded into the config before training starts, so
the config.yaml written next to the metrics is the run that actually
happened, not the file it was launched from. An override that changed the
run without changing its record would defeat the point of having a record.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from rlvr_from_scratch.training.config import TrainConfig
from rlvr_from_scratch.training.trainer import train


def build_parser() -> argparse.ArgumentParser:
    """The command-line surface, kept deliberately small."""
    parser = argparse.ArgumentParser(
        prog="rlvr-train",
        description="Train the from-scratch decoder-only transformer.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="path to a run config, e.g. configs/tiny.yaml",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: runs/<config stem>)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="override the config's device, e.g. cpu, mps, cuda",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override the config's seed; use to run the same config again",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="override the config's max_steps; useful for a smoke run",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="do not print a line per evaluation",
    )
    return parser


def apply_overrides(config: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    """Fold the command-line overrides into the config.

    Args:
        config: The config as read from disk.
        args:   Parsed arguments; None means "leave it alone".

    Returns:
        A new config. TrainConfig is frozen, so this is a replace, not a
        mutation — and it is re-validated on construction, which means an
        override that produces an impossible run fails here rather than
        halfway through training.
    """
    changes: dict[str, object] = {}
    if args.device is not None:
        changes["device"] = args.device
    if args.seed is not None:
        changes["seed"] = args.seed
    if args.max_steps is not None:
        changes["max_steps"] = args.max_steps
        # A shortened run whose warmup is longer than the run itself is not
        # a shortened run, it is an all-warmup one. Clamp rather than fail:
        # --max-steps 5 is how a smoke test is written.
        changes["warmup_steps"] = min(config.warmup_steps, args.max_steps)

    if not changes:
        return config
    return dataclasses.replace(config, **changes)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code.

    Args:
        argv: Argument list, or None to read sys.argv.

    Returns:
        0 on a completed run, 1 on any expected failure — a bad config, a
        corpus whose bytes do not match the pin, an unavailable device.
        Nonzero so `make` stops here instead of carrying on.
    """
    args = build_parser().parse_args(argv)

    out_dir = args.out if args.out is not None else Path("runs") / args.config.stem

    try:
        config = TrainConfig.load(args.config)
        config = apply_overrides(config, args)
        result = train(config, out_dir=out_dir, verbose=not args.quiet)
    except (ValueError, OSError) as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 1

    print(
        f"best val {result.best_val_loss:.4f} nats at step {result.best_step} "
        f"| {result.wall_clock_s:.1f}s | {out_dir}"
    )
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
