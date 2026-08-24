#!/usr/bin/env python3
"""Scan source and packaged artefacts without disclosing matched values.

Google-shaped third-party page tokens are blocking by default.  The only
exception is an exact three-way match between token digest, repository-relative
raw-page path, and the frozen digest of that complete downloaded page.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
from typing import BinaryIO, Iterable
import zipfile


PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    # The left boundary prevents URL slugs such as ``...risk-...`` from
    # matching at the embedded ``sk-`` substring.
    ("openai_api_key", re.compile(rb"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{20,}")),
    ("aws_access_key", re.compile(rb"AKIA[0-9A-Z]{16}")),
    ("github_token", re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}")),
    ("google_api_key", re.compile(rb"AIza[0-9A-Za-z_-]{30,}")),
    (
        "private_key",
        re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    ),
)

ARCHIVE_SUFFIXES = {".zip", ".xlsx", ".xlsm", ".docx", ".pptx"}
SKIP_SUFFIXES = {
    ".parquet",
    ".pkl",
    ".pickle",
    ".npy",
    ".npz",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".png",
    ".jpg",
    ".jpeg",
    ".heic",
    ".gif",
    ".webp",
    ".mp3",
    ".mp4",
    ".mov",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
    ".pyc",
}
SKIP_DIRS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    "playwright-report",
    "test-results",
    "chroma",
}
RAW_PAGE_PREFIX = PurePosixPath("thesis/bth-v2/.work/documents/raw")
READ_CHUNK_SIZE = 1024 * 1024
MATCH_OVERLAP = 256
MAX_NESTED_ARCHIVE_DEPTH = 2


class ScanConfigurationError(ValueError):
    """The allowlist is malformed or attempts to broaden its narrow scope."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(READ_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_digest(value: object) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def load_allowlist(path: Path) -> set[tuple[str, str, str, str]]:
    if not path.exists():
        raise ScanConfigurationError(f"allowlist file is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScanConfigurationError(f"cannot read allowlist: {exc}") from exc
    if payload.get("version") != 1 or not isinstance(payload.get("entries"), list):
        raise ScanConfigurationError("allowlist must use version 1 and an entries list")

    allowed: set[tuple[str, str, str, str]] = set()
    for index, entry in enumerate(payload["entries"]):
        if not isinstance(entry, dict):
            raise ScanConfigurationError(f"allowlist entry {index} is not an object")
        kind = entry.get("kind")
        relative = entry.get("path")
        token_digest = entry.get("token_sha256")
        file_digest = entry.get("file_sha256")
        classification = entry.get("classification")
        if kind != "google_api_key":
            raise ScanConfigurationError(
                f"allowlist entry {index} may only classify a Google-shaped page token"
            )
        if classification != "third_party_downloaded_page_token":
            raise ScanConfigurationError(
                f"allowlist entry {index} has an unsupported classification"
            )
        if not isinstance(relative, str):
            raise ScanConfigurationError(f"allowlist entry {index} has no path")
        parsed = PurePosixPath(relative)
        if (
            parsed.is_absolute()
            or ".." in parsed.parts
            or parsed.parent != RAW_PAGE_PREFIX
            or parsed.suffix.lower() not in {".html", ".htm"}
        ):
            raise ScanConfigurationError(
                f"allowlist entry {index} is outside the downloaded raw-page directory"
            )
        if not _valid_digest(token_digest) or not _valid_digest(file_digest):
            raise ScanConfigurationError(
                f"allowlist entry {index} must contain lowercase SHA-256 digests"
            )
        record = (kind, relative, token_digest, file_digest)
        if record in allowed:
            raise ScanConfigurationError(f"allowlist entry {index} is duplicated")
        allowed.add(record)
    return allowed


def iter_stream_matches(handle: BinaryIO) -> Iterable[tuple[str, bytes]]:
    """Yield unique matches while retaining enough overlap across chunks."""

    tail = b""
    absolute_offset = 0
    seen: set[tuple[str, int]] = set()
    while True:
        chunk = handle.read(READ_CHUNK_SIZE)
        if not chunk:
            break
        block = tail + chunk
        block_offset = absolute_offset - len(tail)
        for kind, pattern in PATTERNS:
            for match in pattern.finditer(block):
                key = (kind, block_offset + match.start())
                if key in seen:
                    continue
                seen.add(key)
                yield kind, match.group(0)
        absolute_offset += len(chunk)
        tail = block[-MATCH_OVERLAP:]


def matches_bytes(value: bytes) -> Iterable[tuple[str, bytes]]:
    return iter_stream_matches(io.BytesIO(value))


class Scanner:
    def __init__(
        self,
        root: Path,
        allowed: set[tuple[str, str, str, str]],
    ) -> None:
        self.root = root.resolve()
        self.allowed = allowed
        self.findings: set[tuple[str, str]] = set()
        self._file_digests: dict[Path, str] = {}
        self._pdftotext = shutil.which("pdftotext")

    def validate_allowlist_sources(self) -> None:
        """Fail closed when a classified page or its content has drifted."""

        for kind, relative, token_digest, expected_file_digest in self.allowed:
            path = self.root / relative
            if not path.is_file() or path.is_symlink():
                raise ScanConfigurationError(
                    f"classified raw page is missing or is not a regular file: {relative}"
                )
            if self.file_digest(path) != expected_file_digest:
                raise ScanConfigurationError(
                    f"classified raw page digest changed: {relative}"
                )
            with path.open("rb") as handle:
                present = any(
                    found_kind == kind and sha256_bytes(token) == token_digest
                    for found_kind, token in iter_stream_matches(handle)
                )
            if not present:
                raise ScanConfigurationError(
                    f"classified token digest is absent from raw page: {relative}"
                )

    def relative_path(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()

    def file_digest(self, path: Path) -> str:
        resolved = path.resolve()
        if resolved not in self._file_digests:
            self._file_digests[resolved] = sha256_path(resolved)
        return self._file_digests[resolved]

    def is_allowed_raw_page_match(self, path: Path, kind: str, token: bytes) -> bool:
        if kind != "google_api_key":
            return False
        relative = self.relative_path(path)
        record = (
            kind,
            relative,
            sha256_bytes(token),
            self.file_digest(path),
        )
        return record in self.allowed

    def record_matches(
        self,
        matches: Iterable[tuple[str, bytes]],
        display_path: str,
        *,
        allow_source_path: Path | None = None,
    ) -> None:
        for kind, token in matches:
            if (
                allow_source_path is not None
                and self.is_allowed_raw_page_match(allow_source_path, kind, token)
            ):
                continue
            self.findings.add((display_path, kind))

    def scan_pdf_path(self, path: Path, display_path: str) -> None:
        if self._pdftotext is None:
            raise ScanConfigurationError(
                "pdftotext is required to inspect text embedded in PDF artefacts"
            )
        completed = subprocess.run(
            [self._pdftotext, "-q", str(path), "-"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if completed.returncode != 0:
            self.findings.add((display_path, "pdf_text_scan_error"))
            return
        self.record_matches(matches_bytes(completed.stdout), f"{display_path}::text")

    def scan_pdf_bytes(self, value: bytes, display_path: str) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pdf") as handle:
            handle.write(value)
            handle.flush()
            self.scan_pdf_path(Path(handle.name), display_path)

    def scan_archive_bytes(self, value: bytes, display_path: str, depth: int) -> None:
        # A legacy zero-byte package placeholder contains no payload to inspect.
        # Package-integrity validation belongs to the packaging gate, not the
        # credential detector.
        if not value:
            return
        try:
            archive = zipfile.ZipFile(io.BytesIO(value))
        except (OSError, zipfile.BadZipFile):
            self.findings.add((display_path, "archive_scan_error"))
            return
        with archive:
            for info in archive.infolist():
                if info.is_dir() or info.flag_bits & 0x1:
                    continue
                member_name = PurePosixPath(info.filename).as_posix()
                member_display = f"{display_path}::{member_name}"
                try:
                    member = archive.read(info)
                except (OSError, RuntimeError, zipfile.BadZipFile):
                    self.findings.add((member_display, "archive_member_scan_error"))
                    continue
                self.record_matches(matches_bytes(member), member_display)
                suffix = PurePosixPath(member_name).suffix.lower()
                if suffix == ".pdf":
                    self.scan_pdf_bytes(member, member_display)
                elif suffix in ARCHIVE_SUFFIXES and depth < MAX_NESTED_ARCHIVE_DEPTH:
                    self.scan_archive_bytes(member, member_display, depth + 1)

    def scan_path(self, path: Path) -> None:
        relative = self.relative_path(path)
        suffix = path.suffix.lower()
        try:
            with path.open("rb") as handle:
                self.record_matches(
                    iter_stream_matches(handle),
                    relative,
                    allow_source_path=path,
                )
        except OSError:
            return

        if suffix == ".pdf":
            self.scan_pdf_path(path, relative)
        elif suffix in ARCHIVE_SUFFIXES:
            try:
                value = path.read_bytes()
            except OSError:
                return
            self.scan_archive_bytes(value, relative, 0)

    def iter_paths(self) -> Iterable[Path]:
        for directory, subdirs, files in os.walk(self.root):
            subdirs[:] = sorted(name for name in subdirs if name not in SKIP_DIRS)
            base = Path(directory)
            for name in sorted(files):
                path = base / name
                if name == ".env" or path.suffix.lower() in SKIP_SUFFIXES:
                    continue
                if path.is_symlink() or not path.is_file():
                    continue
                yield path

    def scan(self) -> set[tuple[str, str]]:
        self.validate_allowlist_sources()
        for path in self.iter_paths():
            self.scan_path(path)
        return self.findings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--allowlist", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    if not root.is_dir():
        print("Secret scan root is not a directory.", file=sys.stderr)
        return 2
    try:
        allowed = load_allowlist(args.allowlist.resolve())
        findings = Scanner(root, allowed).scan()
    except ScanConfigurationError as exc:
        print(f"Secret scan configuration error: {exc}", file=sys.stderr)
        return 2

    if findings:
        print("Potential secret material found (values withheld):", file=sys.stderr)
        for display_path, kind in sorted(findings):
            print(f"  {display_path} [{kind}]", file=sys.stderr)
        return 1
    print("Secret scan passed, including source and packaged artefacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
