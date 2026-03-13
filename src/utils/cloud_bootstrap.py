"""Helpers to bootstrap cloud runtime assets from hosted bundles."""

from __future__ import annotations

import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, Tuple

import requests


def _has_required_files(target_dir: Path, required_files: Iterable[str]) -> bool:
    return all((target_dir / rel_path).exists() for rel_path in required_files)


def ensure_bundle(
    url: str,
    target_dir: str | Path,
    required_files: Iterable[str],
    timeout_seconds: int = 600,
) -> Tuple[bool, str]:
    """
    Download and extract a bundle into target_dir if required files are missing.
    Returns: (changed, message)
    """
    target = Path(target_dir)
    required = list(required_files)
    if _has_required_files(target, required):
        return False, f"Bundle already available at {target}."

    if not url.strip():
        return False, "Bundle URL is empty."

    target.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="eagle-eye-bundle-") as temp_dir:
        archive_path = Path(temp_dir) / "bundle"
        with requests.get(url, stream=True, timeout=timeout_seconds) as response:
            response.raise_for_status()
            suffix = ".tar.gz"
            content_type = str(response.headers.get("content-type", "")).lower()
            if ".zip" in url.lower() or "zip" in content_type:
                suffix = ".zip"
            archive_path = archive_path.with_suffix(suffix)
            with archive_path.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)

        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(target)
        else:
            with tarfile.open(archive_path, "r:*") as tf:
                tf.extractall(target)

    if _has_required_files(target, required):
        return True, f"Downloaded and extracted bundle into {target}."

    # Clean incomplete extraction to avoid broken partial state.
    for child in target.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)
    return False, f"Bundle extracted but required files are still missing in {target}."


def ensure_file_manifest(
    url: str,
    target_dir: str | Path,
    required_files: Iterable[str],
    timeout_seconds: int = 1800,
) -> Tuple[bool, str]:
    """
    Download individual files listed in a JSON manifest into target_dir.
    Manifest format:
    {
      "files": [
        {"path": "chroma.sqlite3", "url": "https://..."},
        {"path": "uuid/data_level0.bin", "url": "https://..."},
        {"path": "uuid/link_lists.bin", "empty": true}
      ]
    }
    """
    target = Path(target_dir)
    required = list(required_files)
    if _has_required_files(target, required):
        return False, f"Manifest assets already exist in {target}."

    if not url.strip():
        return False, "Manifest URL is empty."

    target.mkdir(parents=True, exist_ok=True)

    with requests.get(url, timeout=timeout_seconds) as response:
        response.raise_for_status()
        manifest = response.json()

    files = manifest.get("files") or []
    if not isinstance(files, list) or not files:
        return False, "Manifest is missing a non-empty 'files' list."

    for item in files:
        rel_path = str(item.get("path", "")).strip()
        file_url = str(item.get("url", "")).strip()
        empty = bool(item.get("empty", False))
        if not rel_path or not file_url:
            if not rel_path or not empty:
                continue
        dest = target / rel_path
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if empty:
            dest.touch()
            continue
        with requests.get(file_url, stream=True, timeout=timeout_seconds) as response:
            response.raise_for_status()
            with dest.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)

    if _has_required_files(target, required):
        return True, f"Downloaded manifest assets into {target}."

    return False, f"Manifest downloaded, but required files are still missing in {target}."
