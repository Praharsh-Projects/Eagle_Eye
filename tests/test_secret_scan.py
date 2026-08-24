from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SCANNER = ROOT / "scripts" / "secret_scan.py"
RAW_PREFIX = Path("thesis/bth-v2/.work/documents/raw")


def _token(prefix: str = "google") -> str:
    if prefix == "google":
        return "AI" + "za" + ("A" * 35)
    return "s" + "k-" + ("B" * 24)


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_allowlist(root: Path, entries: list[dict[str, str]]) -> Path:
    path = root / "allowlist.json"
    path.write_text(
        json.dumps({"version": 1, "entries": entries}),
        encoding="utf-8",
    )
    return path


def _entry(path: Path, token: str, content: bytes) -> dict[str, str]:
    return {
        "kind": "google_api_key",
        "classification": "third_party_downloaded_page_token",
        "path": path.as_posix(),
        "token_sha256": _digest(token.encode()),
        "file_sha256": _digest(content),
    }


def _scan(root: Path, allowlist: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCANNER),
            "--root",
            str(root),
            "--allowlist",
            str(allowlist),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_left_boundary_rejects_real_key_but_not_url_slug(tmp_path: Path) -> None:
    allowlist = _write_allowlist(tmp_path, [])
    harmless = tmp_path / "links.txt"
    harmless.write_text(
        "risk-management-framework-ai-rmf-10 and task-sk-abcdefghijklmnopqrstuv",
        encoding="utf-8",
    )
    assert _scan(tmp_path, allowlist).returncode == 0

    secret = _token("openai")
    harmful = tmp_path / "secret.txt"
    harmful.write_text(secret, encoding="utf-8")
    result = _scan(tmp_path, allowlist)
    assert result.returncode == 1
    assert "secret.txt [openai_api_key]" in result.stderr
    assert secret not in result.stderr


def test_google_token_is_blocking_by_default_and_value_is_withheld(
    tmp_path: Path,
) -> None:
    token = _token()
    (tmp_path / "page.html").write_text(token, encoding="utf-8")
    result = _scan(tmp_path, _write_allowlist(tmp_path, []))
    assert result.returncode == 1
    assert "page.html [google_api_key]" in result.stderr
    assert token not in result.stderr


def test_exact_raw_page_allowlist_requires_token_path_and_file_digest(
    tmp_path: Path,
) -> None:
    token = _token()
    relative = RAW_PREFIX / "classified.html"
    source = tmp_path / relative
    source.parent.mkdir(parents=True)
    content = f"<script>{token}</script>".encode()
    source.write_bytes(content)
    allowlist = _write_allowlist(tmp_path, [_entry(relative, token, content)])
    assert _scan(tmp_path, allowlist).returncode == 0

    copied = tmp_path / "public" / "copied.html"
    copied.parent.mkdir()
    copied.write_bytes(content)
    assert _scan(tmp_path, allowlist).returncode == 1
    copied.unlink()

    source.write_bytes(content + b"<!-- changed -->")
    assert _scan(tmp_path, allowlist).returncode != 0

    source.write_bytes(content.replace(token.encode(), _token()[:-1].encode() + b"Z"))
    assert _scan(tmp_path, allowlist).returncode != 0

    source.write_bytes(content)
    moved = source.with_name("moved.html")
    source.rename(moved)
    assert _scan(tmp_path, allowlist).returncode != 0


def test_archive_and_pdf_like_artifacts_are_scanned(tmp_path: Path) -> None:
    token = _token()
    allowlist = _write_allowlist(tmp_path, [])
    package = tmp_path / "public.zip"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("nested/result.txt", token)
    result = _scan(tmp_path, allowlist)
    assert result.returncode == 1
    assert "public.zip::nested/result.txt [google_api_key]" in result.stderr
    assert token not in result.stderr

    package.unlink()
    workbook = tmp_path / "public.xlsx"
    with zipfile.ZipFile(workbook, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("xl/sharedStrings.xml", token)
    result = _scan(tmp_path, allowlist)
    assert result.returncode == 1
    assert "public.xlsx::xl/sharedStrings.xml [google_api_key]" in result.stderr
    assert token not in result.stderr

    workbook.unlink()
    (tmp_path / "result.pdf").write_bytes(b"%PDF-1.4\n" + token.encode())
    result = _scan(tmp_path, allowlist)
    assert result.returncode == 1
    assert "result.pdf [google_api_key]" in result.stderr
    assert token not in result.stderr


def test_repository_allowlist_contains_only_two_tokens_in_six_frozen_pages() -> None:
    payload = json.loads(
        (ROOT / "scripts" / "secret_scan_allowlist.json").read_text(
            encoding="utf-8"
        )
    )
    entries = payload["entries"]
    assert len(entries) == 6
    assert len({entry["token_sha256"] for entry in entries}) == 2
    assert all(
        entry["classification"] == "third_party_downloaded_page_token"
        for entry in entries
    )
    assert all("token" not in entry for entry in entries)


def test_release_gate_pins_fixed_router_and_blocks_moderate_findings() -> None:
    package = json.loads((ROOT / "web" / "package.json").read_text(encoding="utf-8"))
    lock = json.loads(
        (ROOT / "web" / "package-lock.json").read_text(encoding="utf-8")
    )
    assert package["dependencies"]["react-router-dom"] == "7.18.2"
    assert lock["packages"]["node_modules/react-router-dom"]["version"] == "7.18.2"
    verify_script = (ROOT / "scripts" / "verify_all.sh").read_text(encoding="utf-8")
    assert "npm --prefix \"$EAGLE_EYE_ROOT/web\" audit --audit-level=moderate" in verify_script
