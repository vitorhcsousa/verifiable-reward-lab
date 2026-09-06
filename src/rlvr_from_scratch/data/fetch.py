"""
Download a pinned corpus and check it against a pinned hash.

data/ is gitignored, so a fresh clone has no corpus and no way to get one.
This is the other half of that decision: one command, pinned URL, pinned
sha256, loud failure if the bytes aren't the ones we built against.

nanoGPT's prepare.py just downloads. We also hash, because "it downloaded
something" and "it downloaded the right thing" are different claims, and
only the second one makes a run reproducible.

    make data
    python -m rlvr_from_scratch.data.fetch --force   # re-download
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# 1MB chunks. the corpora here are ~1MB so this is overkill today, but it's
# the code path that eventually points at a multi-GB pretraining shard.
CHUNK = 1 << 20

# src/rlvr_from_scratch/data/fetch.py -> repo root
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"

DEFAULT = "shakespeare"

# pinned to a commit, not to master. a branch url is a moving pin: the hash
# check would still fail loudly, but it would fail on a day we changed
# nothing, and a reproducibility check that flakes stops being read.
GSM8K_REV = "2909d34ef28520753df82a2234c357259d254aa8"
GSM8K_BASE = (
    f"https://raw.githubusercontent.com/openai/grade-school-math/{GSM8K_REV}"
    "/grade_school_math/data"
)
GSM8K_NOTE = (
    f"openai/grade-school-math @ {GSM8K_REV[:12]} (MIT); "
    "the split boundary is upstream's, not ours"
)


@dataclass(frozen=True)
class Source:
    """Where a corpus comes from and what it must hash to."""

    url: str
    sha256: str
    fname: str
    nbytes: int
    note: str  # provenance + license, recorded so we don't re-litigate it


SOURCES: dict[str, Source] = {
    "shakespeare": Source(
        url="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        sha256="86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed",
        fname="input.txt",
        nbytes=1_115_394,
        note="tinyshakespeare via karpathy/char-rnn (MIT); the text is public domain",
    ),
    # the two gsm8k files stay separate entries because they are separate
    # claims: test.jsonl is the held-out set, and folding it into one
    # "gsm8k" download would make it that much easier to read by accident.
    "gsm8k-train": Source(
        url=f"{GSM8K_BASE}/train.jsonl",
        sha256="17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465",
        fname="gsm8k_train.jsonl",
        nbytes=4_166_206,
        note=GSM8K_NOTE,
    ),
    "gsm8k-test": Source(
        url=f"{GSM8K_BASE}/test.jsonl",
        sha256="3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
        fname="gsm8k_test.jsonl",
        nbytes=749_738,
        note=GSM8K_NOTE,
    ),
}


def sha256(path: Path) -> str:
    """Hex digest of a file, read incrementally."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def check(path: Path, src: Source) -> None:
    """Raise unless path holds exactly the bytes src pins."""
    # size first, it's free. "5000 bytes, expected 1115394" points you at a
    # truncated download or an html error page; a bare hash mismatch doesn't.
    n = path.stat().st_size
    if n != src.nbytes:
        msg = f"{path} is {n} bytes, expected {src.nbytes} (truncated download? error page?)"
        raise ValueError(msg)

    got = sha256(path)
    if got != src.sha256:
        msg = (
            f"{path} hashes to {got}, pinned is {src.sha256} "
            f"(upstream changed, or the file was edited locally - use --force)"
        )
        raise ValueError(msg)


def fetch(src: Source, out_dir: Path = DATA_DIR, *, force: bool = False) -> Path:
    """Ensure src is present and correct at out_dir/fname. Idempotent."""
    path = out_dir / src.fname

    if path.exists() and not force:
        check(path, src)  # verify what's there, don't just trust it
        print(f"{path} already present and verified")
        return path

    out_dir.mkdir(parents=True, exist_ok=True)

    # download next to the target, never onto it. a download killed halfway
    # that landed on input.txt would look complete to every later step and
    # surface days later as a loss curve that's just slightly worse.
    tmp = path.with_suffix(path.suffix + ".part")
    print(f"downloading {src.url}")
    try:
        with urllib.request.urlopen(src.url) as r, tmp.open("wb") as f:  # noqa: S310
            while chunk := r.read(CHUNK):
                f.write(chunk)
    except urllib.error.URLError:
        tmp.unlink(missing_ok=True)
        raise

    try:
        check(tmp, src)
    except ValueError:
        tmp.unlink(missing_ok=True)  # don't leave it lying around to be renamed
        raise

    tmp.replace(path)  # atomic, same filesystem, which is why tmp lives here
    print(f"{path} ok ({src.nbytes} bytes, sha256 {src.sha256})")
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="download and verify a pinned corpus")
    p.add_argument("--name", default=DEFAULT, choices=sorted(SOURCES))
    p.add_argument("--all", action="store_true", help="fetch every pinned corpus")
    p.add_argument("--dest", type=Path, default=DATA_DIR)
    p.add_argument("--force", action="store_true", help="re-download even if present")
    args = p.parse_args(argv)

    names = sorted(SOURCES) if args.all else [args.name]

    try:
        for name in names:
            fetch(SOURCES[name], args.dest, force=args.force)
    except (ValueError, OSError, urllib.error.URLError) as e:
        # nonzero so make stops here, instead of letting the next target
        # train on a corpus that isn't there
        print(f"FAILED: {e}", file=sys.stderr)
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
