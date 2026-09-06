"""Entry point for one evaluation run.

uv run rlvr-eval --config configs/eval_gsm8k.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from rlvr_from_scratch.evaluation.config import EvalConfig
from rlvr_from_scratch.evaluation.gsm8k_eval import run


def build_parser() -> argparse.ArgumentParser:
    """The command-line surface, kept deliberately small."""
    parser = argparse.ArgumentParser(
        prog="rlvr-eval",
        description="Evaluate a checkpoint on GSM8K from a committed config.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="path to an eval config, e.g. configs/eval_gsm8k.yaml",
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
        "--n-examples",
        type=int,
        default=None,
        help="override how many examples to score, e.g. for a smoke run",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="do not print the model line before generating",
    )
    return parser


def apply_overrides(config: EvalConfig, args: argparse.Namespace) -> EvalConfig:
    """Fold the command-line overrides into the config.

    The result is what gets saved beside the numbers, so an override is
    recorded rather than lost.
    """
    changes: dict[str, object] = {}
    if args.device is not None:
        changes["device"] = args.device
    if args.n_examples is not None:
        changes["n_examples"] = args.n_examples

    if not changes:
        return config
    return dataclasses.replace(config, **changes)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Returns:
        0 on a completed evaluation, 1 on a bad config, missing corpus, or
        an unavailable device.
    """
    args = build_parser().parse_args(argv)

    out_dir = args.out if args.out is not None else Path("runs") / args.config.stem

    try:
        config = EvalConfig.load(args.config)
        config = apply_overrides(config, args)
        result = run(config, out_dir, verbose=not args.quiet)
    except (ValueError, OSError) as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 1

    print(
        f"accuracy {result.accuracy:.3f} ({result.n_correct}/{result.n}) | "
        f"no answer {result.no_answer_rate:.3f} | "
        f"{result.wall_clock_s:.1f}s | {result.tokens_per_s:.1f} tok/s | {out_dir}"
    )
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
