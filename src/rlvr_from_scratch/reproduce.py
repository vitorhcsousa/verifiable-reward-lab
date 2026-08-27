"""Check that two runs of the same config landed within tolerance.

uv run rlvr-compare runs/ref-a runs/ref-b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# nats per token, stated in the README next to the claim it backs
DEFAULT_TOLERANCE = 0.05


def read_val_loss(run_dir: Path) -> float:
    """Pull the final validation loss out of a run directory.

    Args:
        run_dir: A directory written by `train`, holding summary.json.

    Returns:
        The run's final validation loss, in nats per token.

    Raises:
        ValueError: If the summary is missing or lacks the field.
    """
    path = run_dir / "summary.json"
    if not path.exists():
        msg = f"{path} not found; is {run_dir} a finished run directory?"
        raise ValueError(msg)

    payload = json.loads(path.read_text(encoding="utf-8"))
    if "final_val_loss" not in payload:
        msg = f"{path} has no 'final_val_loss'; keys are {sorted(payload)}"
        raise ValueError(msg)
    return float(payload["final_val_loss"])


def main(argv: list[str] | None = None) -> int:
    """Entry point. 0 if the two runs agree within tolerance, 1 if not."""
    parser = argparse.ArgumentParser(
        prog="rlvr-compare",
        description="Check that two runs of the same config agree.",
    )
    parser.add_argument("first", type=Path, help="a run directory")
    parser.add_argument("second", type=Path, help="another run directory")
    # an argument, not a constant: the tolerance is a property of the run,
    # and is meant to be validated on a second machine before being frozen
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"max |delta| in nats (default {DEFAULT_TOLERANCE})",
    )
    args = parser.parse_args(argv)

    try:
        a = read_val_loss(args.first)
        b = read_val_loss(args.second)
    except (ValueError, OSError, json.JSONDecodeError) as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1

    delta = abs(a - b)
    print(f"{args.first}: final val {a:.4f} nats")
    print(f"{args.second}: final val {b:.4f} nats")
    print(f"|delta| = {delta:.4f}, tolerance {args.tolerance}")

    if delta > args.tolerance:
        print("FAILED: the two runs do not reproduce each other", file=sys.stderr)
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
