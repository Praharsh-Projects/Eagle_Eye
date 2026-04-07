from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.voyage_lab.reference_imports import (
    import_rec20_metadata,
    import_swagger_catalog,
    import_unlocode_reference,
)


def test_import_unlocode_reference_from_csv_parts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "fake_home"))
    project_root = tmp_path / "project"
    data_root = project_root / "data" / "raw" / "reference" / "unlocode"
    data_root.mkdir(parents=True, exist_ok=True)

    # UN/LOCODE part files are headerless and use 12 columns.
    (data_root / "UNLOCODE CodeListPart1.csv").write_text(
        ",SE,KAA,Karlskrona,Karlskrona,,1-3----,AA,2401,,,Test row\n"
        ",PL,GDN,Gdansk,Gdansk,,1-3----,AA,2401,,,Test row\n",
        encoding="latin1",
    )
    (data_root / "UNLOCODE CodeListPart2.csv").write_text(
        ",SE,GOT,Goteborg,Goteborg,,1-3----,AA,2401,,,Test row\n",
        encoding="latin1",
    )

    result = import_unlocode_reference(project_root, roots=[data_root])
    assert result.status == "ok"
    assert result.records == 3

    ports_path = project_root / "data" / "curated" / "reference" / "ports_reference.parquet"
    aliases_path = project_root / "data" / "curated" / "reference" / "ports_aliases.csv"
    assert ports_path.exists()
    assert aliases_path.exists()

    ports = pd.read_parquet(ports_path)
    assert set(ports["locode"].tolist()) == {"SEKAA", "PLGDN", "SEGOT"}

    aliases = pd.read_csv(aliases_path)
    assert {"locode", "alias"}.issubset(set(aliases.columns))
    assert (aliases["locode"] == "SEKAA").any()


def test_import_rec20_metadata_collects_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "fake_home"))
    project_root = tmp_path / "project"
    rec_root = tmp_path / "rec20"
    rec_root.mkdir(parents=True, exist_ok=True)
    (rec_root / "rec20_rev13_2024.csv").write_text("unit,code\nkilogram,KGM\n", encoding="utf-8")

    result = import_rec20_metadata(project_root, roots=[rec_root])
    assert result.status == "ok"
    assert result.records == 1

    out_path = project_root / "data" / "curated" / "reference" / "measurement_reference_rec20.json"
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["files"][0]["revision_hint"] == "13"


def test_import_swagger_catalog_builds_endpoint_catalog(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "fake_home"))
    project_root = tmp_path / "project"
    swagger_root = tmp_path / "swagger"
    swagger_root.mkdir(parents=True, exist_ok=True)
    (swagger_root / "swagger.json").write_text(
        json.dumps(
            {
                "openapi": "3.0.0",
                "servers": [{"url": "https://api.example.com"}],
                "paths": {
                    "/ships": {"get": {"operationId": "listShips", "summary": "List ships", "tags": ["ships"]}},
                    "/voyages/{id}": {"post": {"operationId": "createVoyage", "summary": "Create voyage"}},
                },
            }
        ),
        encoding="utf-8",
    )

    result = import_swagger_catalog(project_root, roots=[swagger_root])
    assert result.status == "ok"
    assert result.records == 2

    out_path = project_root / "data" / "curated" / "reference" / "provider_endpoint_catalog.json"
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["endpoint_count"] == 2
    assert any(ep["operation_id"] == "listShips" for ep in payload["endpoints"])
