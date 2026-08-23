"""
Tests for corpus download + verification.

Nothing here touches the network - every download goes through a fake
urlopen. A suite that reaches for raw.githubusercontent.com fails on a
plane, in CI without egress, and on the day GitHub is slow, and would be
testing GitHub rather than this module.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Self

import pytest

from rlvr_from_scratch.data.fetch import (
    DEFAULT,
    SOURCES,
    Source,
    check,
    fetch,
    sha256,
)

BLOB = b"To be, or not to be, that is the question.\n" * 64


def make_src(blob: bytes) -> Source:
    """A Source pinned to exactly blob."""
    return Source(
        url="https://example.invalid/input.txt",
        sha256=hashlib.sha256(blob).hexdigest(),
        fname="input.txt",
        nbytes=len(blob),
        note="synthetic, for tests",
    )


class FakeResponse:
    """The slice of the urlopen contract fetch actually uses."""

    def __init__(self, blob: bytes) -> None:
        self.buf = blob

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self, n: int) -> bytes:
        chunk, self.buf = self.buf[:n], self.buf[n:]
        return chunk


class FakeUpstream:
    """Serves fixed bytes and counts calls.

    The count is the only way to tell "verified what was already there"
    apart from "downloaded it again", which is the whole idempotence claim.
    """

    def __init__(self, blob: bytes) -> None:
        self.blob = blob
        self.calls = 0

    def urlopen(self, url: str, *_: object, **__: object) -> FakeResponse:
        self.calls += 1
        return FakeResponse(self.blob)


@pytest.fixture
def upstream(monkeypatch: pytest.MonkeyPatch) -> FakeUpstream:
    up = FakeUpstream(BLOB)
    monkeypatch.setattr(
        "rlvr_from_scratch.data.fetch.urllib.request.urlopen", up.urlopen
    )
    return up


# ---- sha256 -------------------------------------------------------------


def test_sha256_matches_hashlib(tmp_path: Path) -> None:
    p = tmp_path / "f.bin"
    p.write_bytes(BLOB)
    assert sha256(p) == hashlib.sha256(BLOB).hexdigest()


def test_sha256_spans_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The chunked read must not depend on the blob fitting in one chunk."""
    monkeypatch.setattr("rlvr_from_scratch.data.fetch.CHUNK", 7)
    p = tmp_path / "f.bin"
    p.write_bytes(BLOB)
    assert sha256(p) == hashlib.sha256(BLOB).hexdigest()


# ---- check --------------------------------------------------------------


def test_check_accepts_pinned_bytes(tmp_path: Path) -> None:
    p = tmp_path / "input.txt"
    p.write_bytes(BLOB)
    check(p, make_src(BLOB))  # must not raise


def test_check_rejects_short_file(tmp_path: Path) -> None:
    p = tmp_path / "input.txt"
    p.write_bytes(BLOB[:10])
    with pytest.raises(ValueError, match="bytes, expected"):
        check(p, make_src(BLOB))


def test_check_rejects_same_size_edit(tmp_path: Path) -> None:
    """Size alone isn't enough - a one-byte flip keeps the length."""
    edited = bytearray(BLOB)
    edited[5] ^= 0xFF
    p = tmp_path / "input.txt"
    p.write_bytes(bytes(edited))
    with pytest.raises(ValueError, match="hashes to"):
        check(p, make_src(BLOB))


def test_check_message_names_both_numbers(tmp_path: Path) -> None:
    # a mismatch message missing either half sends you to the wrong place
    src = make_src(BLOB)
    other = BLOB + b"!"
    p = tmp_path / "input.txt"
    p.write_bytes(other)
    with pytest.raises(ValueError) as exc:
        check(p, src)
    assert str(len(other)) in str(exc.value)
    assert str(src.nbytes) in str(exc.value)


# ---- fetch --------------------------------------------------------------


def test_fetch_downloads(tmp_path: Path, upstream: FakeUpstream) -> None:
    out = fetch(make_src(BLOB), tmp_path)
    assert out == tmp_path / "input.txt"
    assert out.read_bytes() == BLOB
    assert upstream.calls == 1


def test_fetch_makes_missing_dir(tmp_path: Path, upstream: FakeUpstream) -> None:
    out = fetch(make_src(BLOB), tmp_path / "nested" / "data")
    assert out.exists()


def test_fetch_is_idempotent(tmp_path: Path, upstream: FakeUpstream) -> None:
    """Second run verifies what's there instead of downloading again."""
    src = make_src(BLOB)
    fetch(src, tmp_path)
    fetch(src, tmp_path)
    assert upstream.calls == 1


def test_force_redownloads(tmp_path: Path, upstream: FakeUpstream) -> None:
    src = make_src(BLOB)
    fetch(src, tmp_path)
    fetch(src, tmp_path, force=True)
    assert upstream.calls == 2


def test_fetch_rejects_existing_wrong_file(
    tmp_path: Path, upstream: FakeUpstream
) -> None:
    (tmp_path / "input.txt").write_bytes(b"not the corpus")
    with pytest.raises(ValueError, match="bytes, expected"):
        fetch(make_src(BLOB), tmp_path)
    assert upstream.calls == 0


def test_force_overwrites_wrong_file(tmp_path: Path, upstream: FakeUpstream) -> None:
    (tmp_path / "input.txt").write_bytes(b"not the corpus")
    out = fetch(make_src(BLOB), tmp_path, force=True)
    assert out.read_bytes() == BLOB


def test_bad_download_leaves_nothing(tmp_path: Path, upstream: FakeUpstream) -> None:
    """The failure this module exists for: upstream serves the wrong bytes."""
    upstream.blob = b"an error page, not a corpus"
    with pytest.raises(ValueError):
        fetch(make_src(BLOB), tmp_path)
    # no target that looks finished, no .part someone might rename
    assert list(tmp_path.iterdir()) == []


def test_bad_download_keeps_good_copy(tmp_path: Path, upstream: FakeUpstream) -> None:
    """--force onto a broken upstream must not destroy what already works."""
    src = make_src(BLOB)
    fetch(src, tmp_path)
    upstream.blob = b"garbage"
    with pytest.raises(ValueError):
        fetch(src, tmp_path, force=True)
    assert (tmp_path / "input.txt").read_bytes() == BLOB


# ---- the pinned registry ------------------------------------------------


def test_default_is_registered() -> None:
    assert DEFAULT in SOURCES


def test_pins_are_well_formed() -> None:
    for src in SOURCES.values():
        assert len(src.sha256) == 64
        assert src.sha256 == src.sha256.lower()
        assert src.nbytes > 0
        assert src.url.startswith("https://")
        assert src.note
