"""Deterministic question intent classification and entity extraction (A-G taxonomy)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd


DOW_NAMES = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]
DOW_TITLE = {name: name.title() for name in DOW_NAMES}
SPECIAL_DOW_NAMES = {"weekend": "Weekend", "weekday": "Weekday"}

UNSUPPORTED_KEYWORDS = (
    "crane",
    "berth utilization",
    "berth productivity",
    "berth-level",
    "berth level",
    "berth queue",
    "teu",
    "teu throughput",
    "throughput",
    "gate queue",
    "queue length",
    "turn-time",
    "turn time",
    "truck turn-time",
    "truck turn time",
    "yard occupancy",
    "terminal gate",
    "container stack",
)

UNSUPPORTED_REGEX_PATTERNS = (
    r"\bberth[\s-]*level\b",
    r"\bturn[\s-]*time\b",
    r"\bqueue[\s-]*length\b",
    r"\bgate[\s-]*queue\b",
    r"\bteu[\s-]*throughput\b",
)

ANOMALY_KEYWORDS = (
    "anomaly",
    "anomalies",
    "unusual",
    "spike",
    "suspicious",
    "spoof",
    "jump",
    "impossible",
    "teleport",
)

FORECAST_KEYWORDS = (
    "forecast",
    "predict",
    "expected",
    "expect",
    "next",
    "coming",
    "future",
    "will",
    "tomorrow",
)

CARBON_KEYWORDS = (
    "carbon",
    "emission",
    "emissions",
    "co2",
    "co2e",
    "nox",
    "sox",
    "pm",
    "tank-to-wake",
    "ttw",
    "well-to-wake",
    "wtw",
)

COMPARE_KEYWORDS = (
    "compare",
    "vs",
    "versus",
    "more than",
    "less than",
    "which port",
)

DIAGNOSTIC_KEYWORDS = (
    "why",
    "cause",
    "reason",
    "dominated",
    "contributing",
    "contributors",
    "breakdown",
)

TEMPORAL_PATTERN_KEYWORDS = (
    "busiest",
    "usually",
    "pattern",
    "seasonal",
    "weekday",
    "day-of-week",
    "hour",
    "busier",
    "quieter",
)

DESCRIPTIVE_KEYWORDS = (
    "how many",
    "how long",
    "how much time",
    "above baseline",
    "below baseline",
    "against baseline",
    "relative to",
    "count",
    "top",
    "average",
    "mean",
    "median",
    "list",
    "show",
    "plot",
    "graph",
    "what is",
)

PORT_TOKEN_STOPWORDS = {
    "TTW",
    "WTW",
    "CO2",
    "CO2E",
    "NOX",
    "SOX",
    "PM",
    "CARBON",
    "EMISSION",
    "EMISSIONS",
    "POLLUTANT",
    "POLLUTANTS",
    "GHG",
    "AIS",
    "IMO",
    "MMSI",
    "ETA",
    "UTC",
    "CSV",
    "RAG",
    "NIS2",
    "ISPS",
    "BASED",
    "PATTERNS",
    "EXPECTED",
    "PREDICT",
    "PREDICTED",
    "CONGESTION",
    "LIKELY",
    "WHICH",
    "SHOW",
    "SUSPICIOUS",
    "JUMPS",
    "MOVEMENT",
    "CHANGES",
    "HOUR",
    "HOURS",
    "KNOT",
    "KNOTS",
    "MODE",
    "DAILY",
    "TREND",
    "INDEX",
    "LEVEL",
    "FIRST",
    "EARLIEST",
    "ISTHE",
    "MONTHLY",
    "FOR",
    "WITH",
}
NON_PORT_CODE_TOKENS = {
    "TTW",
    "WTW",
    "CO2",
    "CO2E",
    "NOX",
    "SOX",
    "PM",
    "CARBON",
    "EMISSION",
    "EMISSIONS",
    "JANUARY",
    "FEBRUARY",
    "MARCH",
    "APRIL",
    "MAY",
    "JUNE",
    "JULY",
    "AUGUST",
    "SEPTEMBER",
    "OCTOBER",
    "NOVEMBER",
    "DECEMBER",
    "MONDAY",
    "TUESDAY",
    "WEDNESDAY",
    "THURSDAY",
    "FRIDAY",
    "SATURDAY",
    "SUNDAY",
}

LOCODE_RE = re.compile(r"\b([A-Z]{2})\s?([A-Z]{3})\b")
ISO_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
MONTH_YEAR_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(?:of\s+)?(20\d{2})\b",
    re.IGNORECASE,
)
YEAR_MONTH_RE = re.compile(r"\b(20\d{2})-(0[1-9]|1[0-2])\b")
YEAR_ONLY_RE = re.compile(r"\b(?:in|during|for)\s+(20\d{2})\b", re.IGNORECASE)
LAST_WEEKS_RE = re.compile(r"\blast\s+(\d{1,2})\s+weeks?\b", re.IGNORECASE)
HORIZON_WEEKS_RE = re.compile(r"\b(\d{1,2})\s+weeks?\b", re.IGNORECASE)
TOP_N_RE = re.compile(r"\btop\s+(\d{1,2})\b", re.IGNORECASE)
NUMBER_WORD_LIMIT_RE = re.compile(
    r"\b(?:top\s+)?(one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?:(?:swedish|finnish|estonian|latvian|lithuanian|polish|danish|german)\s+)?ports?\b",
    re.IGNORECASE,
)
MMSI_RE = re.compile(r"\bmmsi\s*[:#]?\s*(\d{6,9})\b", re.IGNORECASE)
IMO_RE = re.compile(r"\bimo\s*[:#]?\s*(\d{6,8})\b", re.IGNORECASE)
CALL_ID_RE = re.compile(r"\bcall[_\-\s]?id[\s:=_\-]*([A-Za-z0-9_\-:.]+)\b", re.IGNORECASE)
LONG_DATE_RE = re.compile(
    r"\b(?:on\s+)?(?:(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s*,?\s*)?"
    r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+"
    r"(\d{1,2}),?\s+(20\d{2})\b",
    re.IGNORECASE,
)
RELATIVE_DOW_RE = re.compile(
    r"\b(next|this)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
CURRENT_TIME_RE = re.compile(r"\b(today|now|currently|current|live|real[ -]?time)\b", re.IGNORECASE)
RELATIVE_DAY_RE = re.compile(r"\b(yesterday|tomorrow)\b", re.IGNORECASE)

BALTIC_LOCODE_PREFIXES = {"SE", "FI", "LV", "LT", "PL", "EE", "DK", "DE", "NO", "RU"}
PORT_PHRASE_RE = re.compile(
    r"\b(?:at|in|near|to|for|from)\s+"
    r"([^\W\d_][^\d,;?.!()]{1,48}?)"
    r"(?=\s+(?:throughout|during|within|between|from|to|on|for|with|by|over|"
    r"today|tomorrow|now|currently|this|next|last)\b|\s*\([A-Z]{2}\s?[A-Z]{3}\)|[,;?.!()]|$)",
    flags=re.IGNORECASE,
)
BETWEEN_PORTS_RE = re.compile(
    r"\bbetween\s+([A-Za-z0-9\- ]{2,24})\s+and\s+([A-Za-z0-9\- ]{2,24})",
    flags=re.IGNORECASE,
)
FROM_TO_PORTS_RE = re.compile(
    r"\bfrom\s+([A-Za-z0-9\- ]{2,32}?)\s+to\s+([A-Za-z0-9\- ]{2,32}?)(?=\s+(?:and|,|;|$)|[.?!]|$)",
    flags=re.IGNORECASE,
)
KNOWN_PORT_ALIASES = {
    "gothenburg",
    "goteborg",
    "goteborgs",
    "helsinki",
    "sodertalje",
    "södertälje",
    "karlshamn",
    "karlskrona",
    "gdansk",
    "gdynia",
    "klaipeda",
    "riga",
    "kotka",
    "swinoujscie",
    "szczecin",
    "lubeck",
    "ventspils",
}

COUNTRY_NAME_TO_CODE = {
    "sweden": "SE",
    "swedish": "SE",
    "finland": "FI",
    "finnish": "FI",
    "estonia": "EE",
    "estonian": "EE",
    "latvia": "LV",
    "latvian": "LV",
    "lithuania": "LT",
    "lithuanian": "LT",
    "poland": "PL",
    "polish": "PL",
    "denmark": "DK",
    "danish": "DK",
    "germany": "DE",
    "german": "DE",
    "norway": "NO",
    "norwegian": "NO",
}
NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


@dataclass
class IntentResult:
    intent: str
    entities: Dict[str, Any]
    reason: str


def _extract_days_of_week(question: str) -> List[str]:
    q = question.lower()
    hits: List[tuple[int, str]] = []
    for day in DOW_NAMES:
        idx = q.find(day)
        if idx >= 0:
            hits.append((idx, DOW_TITLE[day]))
    hits.sort(key=lambda x: x[0])
    return [name for _, name in hits]


def _extract_special_dow(question: str) -> Optional[str]:
    q = question.lower()
    for token, label in SPECIAL_DOW_NAMES.items():
        if token in q:
            return label
    return None


def _extract_metric(question: str) -> Optional[str]:
    q = question.lower()
    explicit_port_stay = any(
        token in q
        for token in (
            "dwell",
            "port stay",
            "port-stay",
            "stayed in port",
            "stay duration",
            "time in port",
            "time at port",
        )
    )
    duration_cue = bool(
        re.search(
            r"\b(?:how\s+long|how\s+much\s+time|how\s+many\s+(?:minutes?|hours?)|time\s+spent|duration)\b",
            q,
        )
    )
    port_presence_cue = bool(
        re.search(
            r"\b(?:in|at)\s+(?:the\s+)?port\b|\b(?:remain(?:ed)?|stay(?:ed)?|spend|spent)\b.{0,32}\bport\b",
            q,
        )
    )
    if explicit_port_stay or (duration_cue and port_presence_cue):
        return "dwell_minutes"
    if any(token in q for token in ("travel time", "route time", "duration", "eta")):
        return "route_duration_h"
    if "congestion" in q or "congested" in q or "pressure" in q or "busy" in q:
        return "congestion_index"
    if "occupancy" in q:
        return "occupancy_vessels"
    if any(token in q for token in ("arrival", "arrive", "arriving", "ship call", "vessel call", "port call")):
        return "arrival_count"
    if "spike" in q or "anomaly" in q:
        return "arrivals_spike"
    return None


def _extract_aggregation(question: str) -> Optional[str]:
    q = question.lower()
    if any(token in q for token in ("share", "composition", "proportion", "percentage")) and any(
        token in q for token in ("vessel type", "ship type", "arrivals by type")
    ):
        return "vessel_type_composition"
    if "divided between" in q and any(
        token in q for token in ("cargo", "tanker", "vessel", "ship")
    ):
        return "vessel_type_composition"
    if "mean" in q or "average" in q:
        if any(token in q for token in ("dwell", "port stay", "time at port", "time in port")):
            return "mean"
    if "median" in q and any(
        token in q for token in ("dwell", "port stay", "time at port", "time in port")
    ):
        return "median"
    if "distribution" in q and any(token in q for token in ("dwell", "port stay", "stay duration")):
        return "dwell_distribution"
    if any(token in q for token in ("median", "p90", "percentile")) and any(
        token in q for token in ("route", "travel time", "duration", "voyage")
    ):
        return "route_travel_time_summary"
    if any(token in q for token in ("travel time", "route time", "route duration")) and (
        ("from" in q and "to" in q) or "route" in q
    ):
        return "route_travel_time_summary"
    if any(token in q for token in ("first", "earliest")):
        if "from" in q and "to" in q and any(token in q for token in ("vessel", "ship", "voyage", "arrival", "depart")):
            return "first_route_vessel"
        if "depart" in q:
            return "first_departure"
        if any(token in q for token in ("arrival", "arrive", "arrival seen", "first seen")):
            return "first_arrival"
    if any(token in q for token in ("last", "latest", "most recent")) and any(
        token in q for token in ("arrival", "arrive")
    ):
        return "last_arrival"
    if any(token in q for token in ("highest", "maximum", "max", "peak")) and any(
        token in q for token in ("day", "date")
    ):
        return "peak_day"
    if TOP_N_RE.search(question) and any(token in q for token in ("pressure", "congestion")):
        return "peak_day"
    return None


def _extract_limit(question: str) -> int:
    hit = TOP_N_RE.search(question)
    if hit:
        return max(1, min(int(hit.group(1)), 20))
    word_hit = NUMBER_WORD_LIMIT_RE.search(question)
    if word_hit:
        return NUMBER_WORDS[word_hit.group(1).lower()]
    return 1


def _extract_country_codes(question: str) -> List[str]:
    hits: List[tuple[int, str]] = []
    for token, code in COUNTRY_NAME_TO_CODE.items():
        for match in re.finditer(rf"\b{re.escape(token)}\b", question, re.IGNORECASE):
            hits.append((match.start(), code))
    output: List[str] = []
    for _, code in sorted(hits):
        if code not in output:
            output.append(code)
    return output


def _extract_vessel_type(question: str) -> Optional[str]:
    q = question.lower()
    if "tanker" in q:
        return "tanker"
    if "cargo" in q:
        return "cargo ship"
    if "container" in q:
        return "container ship"
    return None


def _extract_carbon_boundary(question: str) -> str:
    q = question.lower()
    has_ttw = any(token in q for token in ("ttw", "tank-to-wake", "tank to wake"))
    has_wtw = any(token in q for token in ("wtw", "well-to-wake", "well to wake", "lifecycle"))
    if has_ttw and has_wtw:
        return "TTW_WTW"
    if has_wtw:
        return "WTW"
    return "TTW"


def _extract_carbon_pollutants(question: str) -> List[str]:
    q = question.lower()
    out: List[str] = []
    if "co2e" in q or "ghg" in q:
        out.append("CO2e")
    if re.search(r"\bco2\b", q) and "CO2" not in out:
        out.append("CO2")
    if "nox" in q:
        out.append("NOx")
    if "sox" in q or "sulfur" in q:
        out.append("SOx")
    if re.search(r"\bpm\b", q) or "particulate" in q:
        out.append("PM")
    if any(token in q for token in ("pollutant", "pollutants", "emissions")) and not out:
        out = ["CO2e", "NOx", "SOx", "PM"]
    if not out:
        out = ["CO2e", "NOx", "SOx", "PM"]
    return out


def _extract_source_scope(question: str) -> Optional[str]:
    q = (question or "").lower()
    if any(token in q for token in ("port-call", "port call", "port calls", "according to port-call", "according to port call")):
        return "port_call"
    if any(token in q for token in ("ais-derived", "ais derived", "ais destination proxy", "ais proxy")):
        return "ais_destination_proxy"
    return None


def _clean_source_scope_artifacts(entities: Dict[str, Any]) -> Dict[str, Any]:
    source_scope = entities.get("source_scope")
    if not source_scope:
        return entities

    bad_tokens = {
        "port-call records",
        "port call records",
        "port-call",
        "port call",
        "port calls",
        "according",
        "records",
        "ais-derived",
        "ais derived",
        "ais proxy",
    }

    clean_ports = []
    for item in entities.get("ports") or []:
        token = str(item or "").strip()
        if token.lower() in bad_tokens:
            continue
        clean_ports.append(token)
    entities["ports"] = clean_ports
    if entities.get("port") and str(entities.get("port")).strip().lower() in bad_tokens:
        entities["port"] = clean_ports[0] if clean_ports else None

    clean_pairs = []
    for pair in entities.get("route_pairs") or []:
        origin = str(pair.get("origin") or "").strip()
        destination = str(pair.get("destination") or "").strip()
        if not origin or not destination:
            continue
        if origin.lower() in bad_tokens or destination.lower() in bad_tokens:
            continue
        clean_pairs.append({"origin": origin, "destination": destination})
    entities["route_pairs"] = clean_pairs
    return entities


def _is_locode_like(token: str) -> bool:
    t = (token or "").upper().replace(" ", "")
    if not re.fullmatch(r"[A-Z]{5}", t):
        return False
    if t in NON_PORT_CODE_TOKENS or t in PORT_TOKEN_STOPWORDS:
        return False
    return t[:2] in BALTIC_LOCODE_PREFIXES


def _clean_port_phrase(raw: str) -> str:
    text = (raw or "").strip(" ,.;:()[]")
    if not text:
        return ""
    text = re.split(
        r"\b(?:between|from|to|on|during|next|last|this|for|with|where|when|which|in|by|above|below|against|baseline)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" ,.;:")
    return text


def _split_port_candidates(raw: str) -> List[str]:
    text = (raw or "").strip()
    if not text:
        return []
    chunks = re.split(r"\s+(?:and|&)\s+|[,/]", text, flags=re.IGNORECASE)
    out: List[str] = []
    for chunk in chunks:
        cleaned = _clean_port_phrase(chunk)
        if cleaned:
            out.append(cleaned)
    return out


def _looks_like_port_name(value: str) -> bool:
    if not value:
        return False
    low = value.lower().strip()
    if low.upper() in NON_PORT_CODE_TOKENS or low.upper() in PORT_TOKEN_STOPWORDS:
        return False
    if not low or low in {"port", "index", "level", "daily", "trend", "monthly"}:
        return False
    if low in KNOWN_PORT_ALIASES:
        return True
    words = [w for w in re.split(r"\s+", low) if w]
    if not words:
        return False
    if any(ch.isdigit() for ch in low):
        return False
    bad_tokens = {
        "mmsi",
        "imo",
        "call",
        "call_id",
        "id",
        "the",
        "gate",
        "queue",
        "length",
        "turn",
        "time",
        "vessel",
        "ship",
        "tanker",
        "tankers",
        "cargo",
        "cargos",
        "container",
        "hours",
        "hour",
        "route",
        "routes",
        "duration",
        "durations",
        "arrival",
        "arrivals",
        "departure",
        "departures",
        "knots",
        "knot",
        "mode",
        "emissions",
        "emission",
        "co2",
        "co2e",
        "nox",
        "sox",
        "pm",
        "poland",
        "sweden",
        "finland",
        "estonia",
        "latvia",
        "lithuania",
        "germany",
        "denmark",
        "norway",
    }
    if any(tok in bad_tokens for tok in words):
        return False
    if len("".join(words)) < 4:
        return False
    return True


def _extract_ports(question: str) -> List[str]:
    ports: List[str] = []
    upper = question.upper()

    for c1, c2 in LOCODE_RE.findall(upper):
        locode = f"{c1}{c2}"
        if _is_locode_like(locode):
            ports.append(locode)

    word_tokens = re.findall(r"\b[A-Za-z]{4,}\b", question)
    for raw in word_tokens:
        token = raw.upper()
        if token in PORT_TOKEN_STOPWORDS or token in NON_PORT_CODE_TOKENS:
            continue
        if _is_locode_like(token):
            if token not in ports:
                ports.append(token)
            continue
        if raw.lower() in KNOWN_PORT_ALIASES and raw not in ports:
            ports.append(raw)

    for phrase in PORT_PHRASE_RE.findall(question):
        for cleaned in _split_port_candidates(phrase):
            if not cleaned:
                continue
            cleaned_code = cleaned.upper().replace(" ", "")
            if _is_locode_like(cleaned_code):
                if cleaned_code not in ports:
                    ports.append(cleaned_code)
                continue
            if _looks_like_port_name(cleaned) and cleaned not in ports:
                ports.append(cleaned)

    if "divided between" not in question.lower():
        for a_raw, b_raw in BETWEEN_PORTS_RE.findall(question):
            for item in (_clean_port_phrase(a_raw), _clean_port_phrase(b_raw)):
                if not item:
                    continue
                item_code = item.upper().replace(" ", "")
                if _is_locode_like(item_code):
                    if item_code not in ports:
                        ports.append(item_code)
                    continue
                if _looks_like_port_name(item) and item not in ports:
                    ports.append(item)

    for origin_raw, destination_raw in FROM_TO_PORTS_RE.findall(question):
        for item in (_clean_port_phrase(origin_raw), _clean_port_phrase(destination_raw)):
            if not item:
                continue
            item_code = item.upper().replace(" ", "")
            if _is_locode_like(item_code):
                if item_code not in ports:
                    ports.append(item_code)
                continue
            if _looks_like_port_name(item) and item not in ports:
                ports.append(item)

    deduped: List[str] = []
    for port in ports:
        if port not in deduped:
            deduped.append(port)
    # ``Name (LOCODE)`` is one entity, not a two-port comparison.  Retain the
    # canonical code and remove only the exact adjacent name token; unrelated
    # names elsewhere in the question remain untouched.
    explicit_codes = [port for port in deduped if _is_locode_like(port)]
    if explicit_codes:
        deduped = [
            port
            for port in deduped
            if _is_locode_like(port)
            or not any(
                re.search(
                    rf"\b{re.escape(port)}\s*\(\s*{re.escape(code[:2])}\s?{re.escape(code[2:])}\s*\)",
                    question,
                    re.IGNORECASE,
                )
                for code in explicit_codes
            )
        ]
    return deduped[:4]


def _extract_origin_destination(question: str) -> tuple[Optional[str], Optional[str]]:
    hit = FROM_TO_PORTS_RE.search(question)
    if not hit:
        return None, None
    origin_raw = _clean_port_phrase(hit.group(1))
    destination_raw = _clean_port_phrase(hit.group(2))

    def _normalize(value: str) -> Optional[str]:
        if not value:
            return None
        code = value.upper().replace(" ", "")
        if _is_locode_like(code):
            return code
        if _looks_like_port_name(value):
            return value
        return None

    return _normalize(origin_raw), _normalize(destination_raw)


def _extract_route_pairs(question: str) -> List[Dict[str, str]]:
    pairs: List[Dict[str, str]] = []
    for origin_raw, destination_raw in FROM_TO_PORTS_RE.findall(question):
        origin_clean = _clean_port_phrase(origin_raw)
        destination_clean = _clean_port_phrase(destination_raw)
        if not origin_clean or not destination_clean:
            continue
        if origin_clean.lower() == destination_clean.lower():
            continue
        pair = {"origin": origin_clean, "destination": destination_clean}
        if pair not in pairs:
            pairs.append(pair)

    lower_q = question.lower()
    if " to " in lower_q:
        route_text = question
        if "from " in lower_q:
            route_text = re.split(r"\bfrom\b", question, maxsplit=1, flags=re.IGNORECASE)[1]
        route_parts = re.split(r"\s+\band\b\s+|[,/]", route_text, flags=re.IGNORECASE)
        for part in route_parts:
            hit = re.search(r"([A-Za-z0-9\- ]{2,32})\s+\bto\b\s+([A-Za-z0-9\- ]{2,32})", part, flags=re.IGNORECASE)
            if not hit:
                continue
            origin_clean = _clean_port_phrase(hit.group(1))
            destination_clean = _clean_port_phrase(hit.group(2))
            if not origin_clean or not destination_clean:
                continue
            if origin_clean.lower() == destination_clean.lower():
                continue
            pair = {"origin": origin_clean, "destination": destination_clean}
            if pair not in pairs:
                pairs.append(pair)
    return pairs[:6]


def _month_start_end(month_name: str, year: int) -> tuple[str, str]:
    ts = pd.Timestamp(year=year, month=pd.Timestamp(month_name).month, day=1)
    month_end = ts + pd.offsets.MonthEnd(0)
    return ts.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d")


def _extract_date_range(question: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    matches = ISO_DATE_RE.findall(question)
    if len(matches) >= 2:
        start, end = matches[0], matches[1]
        if start > end:
            start, end = end, start
        return start, end, None
    if len(matches) == 1:
        return matches[0], matches[0], None

    month_match = MONTH_YEAR_RE.search(question)
    if month_match:
        month = month_match.group(1)
        year = int(month_match.group(2))
        start, end = _month_start_end(month, year)
        return start, end, None

    ym_match = YEAR_MONTH_RE.search(question)
    if ym_match:
        year = int(ym_match.group(1))
        month = int(ym_match.group(2))
        start = pd.Timestamp(year=year, month=month, day=1)
        end = start + pd.offsets.MonthEnd(0)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), None

    year_match = YEAR_ONLY_RE.search(question)
    if year_match:
        year = int(year_match.group(1))
        return f"{year:04d}-01-01", f"{year:04d}-12-31", None

    last_weeks_match = LAST_WEEKS_RE.search(question)
    if last_weeks_match:
        weeks = int(last_weeks_match.group(1))
        return None, None, f"last_{weeks}_weeks"

    relative_hit = RELATIVE_DAY_RE.search(question)
    if relative_hit:
        now = pd.Timestamp.now(tz="UTC").floor("D")
        offset = -1 if relative_hit.group(1).lower() == "yesterday" else 1
        target = now + pd.Timedelta(days=offset)
        value = target.strftime("%Y-%m-%d")
        return value, value, None

    if CURRENT_TIME_RE.search(question):
        today = pd.Timestamp.now(tz="UTC").floor("D").strftime("%Y-%m-%d")
        return today, today, None

    return None, None, None


def _extract_horizon_weeks(question: str) -> int:
    m = HORIZON_WEEKS_RE.search(question.lower())
    if not m:
        return 4
    value = int(m.group(1))
    return max(1, min(12, value))


def _extract_target_date(question: str) -> Optional[str]:
    # Explicit ISO date like "on 2026-02-20".
    iso_matches = ISO_DATE_RE.findall(question)
    if len(iso_matches) == 1:
        return iso_matches[0]

    # Explicit long date like "on Friday, February 20, 2026".
    long_hit = LONG_DATE_RE.search(question)
    if long_hit:
        month = long_hit.group(2)
        day = long_hit.group(3)
        year = long_hit.group(4)
        ts = pd.to_datetime(f"{month} {day} {year}", errors="coerce")
        if pd.notna(ts):
            return pd.Timestamp(ts).strftime("%Y-%m-%d")

    # Relative weekday like "next Friday" or "this Friday".
    rel_hit = RELATIVE_DOW_RE.search(question)
    if rel_hit:
        mode = rel_hit.group(1).lower()
        day = rel_hit.group(2).lower()
        now = pd.Timestamp.now(tz="UTC").floor("D")
        target_idx = DOW_NAMES.index(day)
        delta = (target_idx - now.weekday()) % 7
        if mode == "next":
            if delta == 0:
                delta = 7
        else:  # "this"
            if delta == 0:
                delta = 0
        target = now + pd.Timedelta(days=delta)
        return target.strftime("%Y-%m-%d")

    return None


def _normalize_call_id(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    normalized = re.sub(r"^[\s:_\-]+", "", raw)
    return normalized.strip()


def _unsupported_hits(question: str) -> List[str]:
    q = question.lower()
    hits: List[str] = []
    for token in UNSUPPORTED_KEYWORDS:
        if token in q and token not in hits:
            hits.append(token)
    for pattern in UNSUPPORTED_REGEX_PATTERNS:
        if re.search(pattern, q):
            label = pattern.replace(r"\b", "").replace(r"[\s-]*", "-").strip("()")
            if label not in hits:
                hits.append(label)
    return hits


def classify_question(question: str) -> IntentResult:
    q = question.lower()

    start_date, end_date, window = _extract_date_range(question)
    target_date = _extract_target_date(question)
    if target_date and not start_date and not end_date:
        start_date = target_date
        end_date = target_date
    dows = _extract_days_of_week(question)
    special_dow = _extract_special_dow(question)
    origin_port, destination_port = _extract_origin_destination(question)
    route_pairs = _extract_route_pairs(question)
    ports = _extract_ports(question)
    if route_pairs and "arrival" in q:
        # In mixed queries, route endpoints are not automatically part of the
        # separately requested port comparison.
        port_clause = re.split(
            r"\b(?:and\s+)?route(?:\s+travel)?\s+durations?\b",
            question,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        explicit_arrival_ports = _extract_ports(port_clause)
        if explicit_arrival_ports:
            ports = explicit_arrival_ports
    entities: Dict[str, Any] = {
        "ports": ports,
        "country_codes": _extract_country_codes(question),
        "port": None,
        "origin_port": origin_port,
        "destination_port": destination_port,
        "route_pairs": route_pairs,
        "date_from": start_date,
        "date_to": end_date,
        "target_date": target_date,
        "window": window,
        "vessel_type": _extract_vessel_type(question),
        "dow": dows[0] if dows else special_dow,
        "dow_compare": dows[1] if len(dows) > 1 else None,
        "metric": _extract_metric(question),
        "aggregation": _extract_aggregation(question),
        "limit": _extract_limit(question),
        "horizon_weeks": _extract_horizon_weeks(question),
        "mmsi": None,
        "imo": None,
        "call_id": None,
        "boundary": _extract_carbon_boundary(question),
        "pollutants": _extract_carbon_pollutants(question),
        "source_scope": _extract_source_scope(question),
        "requires_current_data": bool(CURRENT_TIME_RE.search(question)),
        "temporal_reference": (
            "current"
            if CURRENT_TIME_RE.search(question)
            else ("relative" if RELATIVE_DAY_RE.search(question) or RELATIVE_DOW_RE.search(question) else "absolute")
        ),
        "visual_requested": bool(re.search(r"\b(?:plot|graph|chart|visuali[sz]e|show)\b", q)),
    }

    if entities["ports"]:
        entities["port"] = entities["ports"][0]

    mmsi_hit = MMSI_RE.search(question)
    if mmsi_hit:
        entities["mmsi"] = mmsi_hit.group(1)
    imo_hit = IMO_RE.search(question)
    if imo_hit:
        entities["imo"] = imo_hit.group(1)
    call_hit = CALL_ID_RE.search(question)
    if call_hit:
        entities["call_id"] = _normalize_call_id(call_hit.group(1))

    entities = _clean_source_scope_artifacts(entities)

    extraction_diag = {
        "ports_parsed": list(entities.get("ports") or []),
        "port_selected": entities.get("port"),
        "route_pairs_parsed": list(entities.get("route_pairs") or []),
        "call_id_parsed": entities.get("call_id"),
        "date_from": entities.get("date_from"),
        "date_to": entities.get("date_to"),
        "target_date": entities.get("target_date"),
    }
    entities["extraction_diagnostics"] = extraction_diag

    retired_voyage_hits = [
        token
        for token in ("resolve voyage", "segment timeline", "voyage evidence")
        if token in q
    ]
    if retired_voyage_hits:
        entities["metric"] = "voyage"
        extraction_diag["unsupported_hits"] = retired_voyage_hits
        return IntentResult(
            intent="G",
            entities=entities,
            reason=(
                "The retired Voyage Lab workflow is not available. Ask for supported historical port-call, "
                "AIS-event, route-duration, forecast, pressure, or carbon analytics instead."
            ),
        )

    unsupported_hits = _unsupported_hits(question)
    if unsupported_hits:
        entities["metric"] = "unsupported"
        extraction_diag["unsupported_hits"] = unsupported_hits
        return IntentResult(
            intent="G",
            entities=entities,
            reason="Requested metric requires terminal operational data outside AIS/port-call scope.",
        )

    if any(token in q for token in CARBON_KEYWORDS):
        entities["metric"] = "emissions"
        return IntentResult(
            intent="H",
            entities=entities,
            reason="Carbon/emissions inventory request detected.",
        )

    if any(token in q for token in ANOMALY_KEYWORDS):
        if entities.get("mmsi") or any(
            token in q
            for token in ("jump", "spoof", "teleport", "impossible", "movement anomal", "position anomal")
        ):
            entities["metric"] = "ais_jump"
        else:
            entities["metric"] = "arrivals_spike"
        return IntentResult(intent="F", entities=entities, reason="Anomaly/suspicious pattern request.")

    if any(token in q for token in FORECAST_KEYWORDS):
        return IntentResult(intent="C", entities=entities, reason="Forecasting language detected.")

    if len(dows) >= 2 or any(token in q for token in COMPARE_KEYWORDS):
        if entities.get("dow") and entities.get("dow_compare"):
            return IntentResult(
                intent="B",
                entities=entities,
                reason="Temporal weekday comparison detected.",
            )

    if any(token in q for token in COMPARE_KEYWORDS):
        return IntentResult(intent="D", entities=entities, reason="Comparative phrasing detected.")

    if entities.get("aggregation") is not None:
        return IntentResult(intent="A", entities=entities, reason="Supported deterministic aggregation detected.")

    if any(token in q for token in DIAGNOSTIC_KEYWORDS):
        return IntentResult(intent="E", entities=entities, reason="Diagnostic/explanatory phrasing detected.")

    if any(token in q for token in TEMPORAL_PATTERN_KEYWORDS):
        return IntentResult(intent="B", entities=entities, reason="Temporal pattern question detected.")

    if any(token in q for token in DESCRIPTIVE_KEYWORDS) and (
        entities.get("metric") is not None or entities.get("aggregation") is not None
    ):
        return IntentResult(intent="A", entities=entities, reason="Descriptive aggregation request.")

    return IntentResult(
        intent="G",
        entities=entities,
        reason="No supported deterministic maritime metric was identified; route to app help, research, or general assistance.",
    )


def describe_intent(intent: str) -> str:
    names = {
        "A": "Descriptive",
        "B": "Temporal Pattern",
        "C": "Forecasting",
        "D": "Comparative",
        "E": "Diagnostic",
        "F": "Anomaly",
        "G": "Unsupported",
        "H": "Carbon Inventory",
    }
    return names.get(intent, "Unknown")


def required_data_for_intent(intent: str) -> List[str]:
    mapping = {
        "A": ["arrivals_daily.parquet"],
        "B": ["arrivals_daily.parquet", "arrivals_hourly.parquet"],
        "C": ["arrivals_daily.parquet", "congestion_daily.parquet"],
        "D": ["arrivals_daily.parquet", "congestion_daily.parquet"],
        "E": ["arrivals_daily.parquet", "dwell_time.parquet"],
        "F": ["arrivals_daily.parquet"],
        "G": [],
        "H": [
            "carbon_segments.parquet",
            "carbon_emissions_segment.parquet",
            "carbon_emissions_daily_port.parquet",
            "carbon_emissions_call.parquet",
            "carbon_evidence.parquet",
            "carbon_params_version.json",
        ],
    }
    return mapping.get(intent, [])
