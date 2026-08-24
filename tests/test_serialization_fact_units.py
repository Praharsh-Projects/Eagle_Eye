from __future__ import annotations

from src.query.serialization import extract_answer_facts


def _numeric_facts(answer: str):
    return [fact for fact in extract_answer_facts(answer) if fact.name.startswith("answer_number_")]


def test_shared_clause_unit_does_not_match_inside_ordinary_words() -> None:
    facts = _numeric_facts(
        "The port pressure index is 0.99, which is below the 1.00 historical baseline."
    )
    assert [fact.value for fact in facts] == [0.99, 1]
    assert [fact.unit for fact in facts] == [None, None]


def test_shared_clause_still_types_both_comparison_values() -> None:
    facts = _numeric_facts("Monday=55 vs Friday=57 arrivals.")
    assert [fact.value for fact in facts] == [55, 57]
    assert [fact.unit for fact in facts] == ["arrivals", "arrivals"]
