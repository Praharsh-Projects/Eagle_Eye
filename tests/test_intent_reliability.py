import unittest
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.qa.intent import classify_question
from src.app.streamlit_app import SAMPLE_QUERIES_BY_CATEGORY, _resolve_scope_with_aggressive_port_fallback


@dataclass
class _DummyKPI:
    port_catalog: pd.DataFrame


class IntentReliabilityTests(unittest.TestCase):
    def test_false_port_tokens_are_not_selected(self) -> None:
        q = "Show daily arrival counts at LVVNT between 2022-02-01 and 2022-02-28."
        parsed = classify_question(q)
        ports = [str(p).upper() for p in parsed.entities.get("ports", [])]
        self.assertIn("LVVNT", ports)
        self.assertNotIn("DAILY", ports)

    def test_trend_query_does_not_parse_trend_as_port(self) -> None:
        q = "Show monthly WTW CO2e trend for SETRG in 2022."
        parsed = classify_question(q)
        ports = [str(p).upper() for p in parsed.entities.get("ports", [])]
        self.assertIn("SETRG", ports)
        self.assertNotIn("TREND", ports)
        self.assertEqual(parsed.entities.get("date_from"), "2022-01-01")
        self.assertEqual(parsed.entities.get("date_to"), "2022-12-31")

    def test_year_only_scope_is_parsed_without_reading_call_id_years(self) -> None:
        parsed = classify_question("Carbon emissions for SEGOT by month in 2022.")
        self.assertEqual(parsed.entities.get("date_from"), "2022-01-01")
        self.assertEqual(parsed.entities.get("date_to"), "2022-12-31")

        call_query = classify_question(
            "What are call-level emissions for MMSI 123456789 and "
            "call_id_123456789_2022-03-01T10-00-00_SEGOT?"
        )
        self.assertIsNone(call_query.entities.get("date_from"))
        self.assertIsNone(call_query.entities.get("date_to"))

    def test_call_id_parsing_strips_leading_separator(self) -> None:
        q = "What are call-level emissions for MMSI 123456789 and call_id_123456789_2022-03-01T10-00-00_SEGOT?"
        parsed = classify_question(q)
        self.assertEqual(
            parsed.entities.get("call_id"),
            "123456789_2022-03-01T10-00-00_SEGOT",
        )

    def test_multi_port_phrase_parses_two_ports(self) -> None:
        q = "How many vessel arrivals were recorded at Karlshamn and Karlskrona in March 2022?"
        parsed = classify_question(q)
        ports = [str(p).lower() for p in parsed.entities.get("ports", [])]
        self.assertIn("karlshamn", ports)
        self.assertIn("karlskrona", ports)

    def test_first_arrival_query_maps_to_first_arrival_aggregation(self) -> None:
        q = "The first arrival seen at Karlshamn on 2022-03-22"
        parsed = classify_question(q)
        self.assertEqual(parsed.intent, "A")
        self.assertEqual(parsed.entities.get("aggregation"), "first_arrival")
        ports = [str(p).lower() for p in parsed.entities.get("ports", [])]
        self.assertIn("karlshamn", ports)
        self.assertNotIn("first", ports)

    def test_first_route_query_extracts_origin_destination(self) -> None:
        q = "The first vessel from Karlshamn to Klaipeda in March 2022"
        parsed = classify_question(q)
        self.assertEqual(parsed.intent, "A")
        self.assertEqual(parsed.entities.get("aggregation"), "first_route_vessel")
        self.assertEqual(str(parsed.entities.get("origin_port", "")).lower(), "karlshamn")
        self.assertEqual(str(parsed.entities.get("destination_port", "")).lower(), "klaipeda")

    def test_last_arrival_query_maps_to_last_arrival(self) -> None:
        q = "What is the last arrival at Karlshamn on 2022-03-22?"
        parsed = classify_question(q)
        self.assertEqual(parsed.intent, "A")
        self.assertEqual(parsed.entities.get("aggregation"), "last_arrival")

    def test_first_departure_query_maps_to_first_departure(self) -> None:
        q = "Show first departure from Karlshamn in March 2022"
        parsed = classify_question(q)
        self.assertEqual(parsed.intent, "A")
        self.assertEqual(parsed.entities.get("aggregation"), "first_departure")

    def test_route_travel_time_summary_query_maps_aggregation(self) -> None:
        q = "What is median and p90 route travel time from Karlshamn to Klaipeda in March 2022?"
        parsed = classify_question(q)
        self.assertEqual(parsed.intent, "A")
        self.assertEqual(parsed.entities.get("aggregation"), "route_travel_time_summary")

    def test_multi_route_compare_extracts_route_pairs(self) -> None:
        q = (
            "Compare routes from Karlshamn to Klaipeda and Karlskrona to Gdynia "
            "in March 2022."
        )
        parsed = classify_question(q)
        self.assertEqual(parsed.intent, "D")
        route_pairs = parsed.entities.get("route_pairs") or []
        self.assertGreaterEqual(len(route_pairs), 2)

    def test_mixed_port_and_route_comparison_keeps_port_scope_separate(self) -> None:
        q = (
            "Compare arrivals at PLSZZ and PLSWI and route durations from PLSZZ to PLSWI "
            "and PLSWI to SEMMA in 2021-02."
        )
        parsed = classify_question(q)
        self.assertEqual(parsed.entities.get("ports"), ["PLSZZ", "PLSWI"])
        self.assertGreaterEqual(len(parsed.entities.get("route_pairs") or []), 2)

    def test_mmsi_anomaly_count_routes_to_ais_jump_detection(self) -> None:
        parsed = classify_question(
            "How many anomaly events were detected for MMSI 255806245 in 2022-03?"
        )
        self.assertEqual(parsed.intent, "F")
        self.assertEqual(parsed.entities.get("metric"), "ais_jump")

    def test_unsupported_variants_are_classified_as_unsupported(self) -> None:
        self.assertEqual(
            classify_question("What is truck turn-time at the gate for SEGOT today?").intent,
            "G",
        )
        self.assertEqual(
            classify_question("Give exact berth-level queue length for vessel arrivals at GDANSK.").intent,
            "G",
        )

    def test_aggressive_port_scope_fallback_prefers_valid_candidate(self) -> None:
        kpi = _DummyKPI(
            port_catalog=pd.DataFrame(
                [
                    {
                        "port_key": "LVVNT",
                        "locode_norm": "LVVNT",
                        "port_label": "Ventspils",
                        "port_name_norm": "ventspils",
                        "arrivals_total": 100,
                        "source_kind": "port_call",
                    },
                    {
                        "port_key": "SEGOT",
                        "locode_norm": "SEGOT",
                        "port_label": "Gothenburg",
                        "port_name_norm": "gothenburg",
                        "arrivals_total": 90,
                        "source_kind": "port_call",
                    },
                ]
            )
        )
        question = "Show daily arrival counts at LVVNT between 2022-02-01 and 2022-02-28."
        entities = {
            "port": "DAILY",
            "ports": ["DAILY", "LVVNT"],
            "date_from": "2022-02-01",
            "date_to": "2022-02-28",
        }
        scope = _resolve_scope_with_aggressive_port_fallback(
            question=question,
            entities=entities,
            user_filters={"port": None, "date_from": None, "date_to": None},
            kpi=kpi,
        )
        self.assertEqual(scope.get("port"), "LVVNT")
        self.assertTrue(scope.get("correction_applied"))

    def test_named_port_alias_resolves_to_canonical_locode(self) -> None:
        kpi = _DummyKPI(
            port_catalog=pd.DataFrame(
                [
                    {
                        "port_key": "SEGOT",
                        "locode_norm": "SEGOT",
                        "port_label": "Port of Gothenburg (SEGOT)",
                        "port_name_norm": "gothenburg",
                        "arrivals_total": 90,
                        "source_kind": "port_call",
                    },
                    {
                        "port_key": "SEKAN",
                        "locode_norm": "SEKAN",
                        "port_label": "Karlshamn (SEKAN)",
                        "port_name_norm": "karlshamn",
                        "arrivals_total": 40,
                        "source_kind": "port_call",
                    },
                ]
            )
        )
        question = "Total vessel arrivals at Gothenburg in March 2022?"
        entities = {
            "port": "Gothenburg",
            "ports": ["Gothenburg"],
            "date_from": "2022-03-01",
            "date_to": "2022-03-31",
        }
        scope = _resolve_scope_with_aggressive_port_fallback(
            question=question,
            entities=entities,
            user_filters={"port": None, "date_from": None, "date_to": None},
            kpi=kpi,
        )
        self.assertEqual(scope.get("port"), "SEGOT")
        self.assertTrue(scope.get("correction_applied"))

    def test_sample_query_status_and_carbon_call_pair_integrity(self) -> None:
        statuses = {"A", "B", "C", "D", "E", "F", "G", "H"}
        for category, queries in SAMPLE_QUERIES_BY_CATEGORY.items():
            for q in queries:
                parsed = classify_question(q)
                self.assertIn(parsed.intent, statuses, msg=f"unexpected intent for sample: {category} :: {q}")

        call_level = [q for q in SAMPLE_QUERIES_BY_CATEGORY.get("Carbon Emissions", []) if "call-level emissions" in q.lower()]
        self.assertTrue(call_level, msg="expected at least one call-level carbon sample query")
        sample = call_level[0]
        parsed = classify_question(sample)
        mmsi = str(parsed.entities.get("mmsi") or "").strip()
        call_id = str(parsed.entities.get("call_id") or "").strip()
        self.assertTrue(mmsi and call_id, msg="call-level sample query must include parseable mmsi and call_id")

        call_table = Path("data/processed/carbon_emissions_call.parquet")
        if call_table.exists():
            df = pd.read_parquet(call_table)
            if not df.empty:
                df["mmsi"] = df["mmsi"].fillna("").astype(str)
                df["call_id"] = df["call_id"].fillna("").astype(str)
                match = df[(df["mmsi"] == mmsi) & (df["call_id"] == call_id)]
                self.assertFalse(match.empty, msg="call-level sample query points to missing data in current dataset")


if __name__ == "__main__":
    unittest.main()
