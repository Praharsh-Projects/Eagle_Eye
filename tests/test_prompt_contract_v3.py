from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "evaluation/latest/prompt_contract_v3.json"


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_prompt_contract_has_75_unique_explicit_cases() -> None:
    contract = _contract()
    cases = contract["cases"]

    assert contract["schema_version"] == "3.0"
    assert contract["case_count"] == 75
    assert len(cases) == 75
    assert len({case["normalized_prompt"] for case in cases}) == 75

    required_expectation_fields = {
        "intents",
        "metric",
        "entities",
        "dates",
        "result_states",
        "coverage",
        "boundary",
        "required_values",
        "required_terms",
        "forbidden_claims",
        "forbid_numeric_answer",
    }
    for case in cases:
        assert case["case_id"]
        assert case["prompt"]
        assert case["source_tags"]
        assert set(case["expectations"]) == required_expectation_fields


def test_prompt_contract_preserves_all_source_inventories() -> None:
    cases = _contract()["cases"]
    source_counts = Counter(
        tag.split(":", maxsplit=1)[0]
        for case in cases
        for tag in case["source_tags"]
    )

    assert source_counts == {
        "ui": 57,
        "catalog": 40,
        "legacy": 43,
        "paper": 6,
    }


def test_paper_critical_numeric_values_are_locked() -> None:
    cases = {
        case["normalized_prompt"]: case
        for case in _contract()["cases"]
    }

    expected = {
        "how many vessel arrivals were recorded at gothenburg in march 2022?": [488, 31],
        "total vessel arrivals at gothenburg in march 2022?": [488, 31],
        "how many vessel arrivals were recorded at karlshamn and gothenburg in march 2022?": [560],
        "according to port-call records, show daily arrival counts at lvvnt between 2022-02-01 and 2022-02-28.": [58, 25],
        "according to port-call records, how many tanker arrivals were recorded at lvvnt between 2022-03-01 and 2022-03-10?": [7, 5],
    }

    for prompt, values in expected.items():
        actual = [
            item["value"]
            for item in cases[prompt]["expectations"]["required_values"]
        ]
        assert actual == values


def test_retired_voyage_prompts_are_explicitly_unsupported() -> None:
    cases = _contract()["cases"]
    retired = [
        case
        for case in cases
        if any(
            token in case["normalized_prompt"]
            for token in ("resolve voyage", "segment timeline", "voyage evidence")
        )
    ]

    assert len(retired) == 5
    assert all(case["expectations"]["intents"] == ["G"] for case in retired)
    assert all(case["expectations"]["result_states"] == ["UNSUPPORTED"] for case in retired)
