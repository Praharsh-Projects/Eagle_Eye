from __future__ import annotations

from pathlib import Path

from src.kpi.query import KPIQueryEngine
from src.predict.data_prep import DEFAULT_DEST_ALIASES, normalize_destination


def test_karlskrona_uses_actual_port_call_locode_everywhere() -> None:
    engine = KPIQueryEngine("data/processed")
    assert engine.resolve_port_token("Karlskrona") == "SEKAA"
    assert engine.resolve_port_token("Port of Karlskrona") == "SEKAA"
    assert normalize_destination("SE KAR", DEFAULT_DEST_ALIASES) == "SEKAA"
    assert normalize_destination("SE KAA", DEFAULT_DEST_ALIASES) == "SEKAA"

    streamlit_source = Path("src/app/streamlit_app.py").read_text(encoding="utf-8")
    assert '"karlskrona": "SEKAA"' in streamlit_source
    assert '"port of karlskrona": "SEKAA"' in streamlit_source
    assert '"karlskrona": "SEKAR"' not in streamlit_source
