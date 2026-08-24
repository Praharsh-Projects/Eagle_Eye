"""Query planning with deterministic safety rails and optional structured AI help."""

from __future__ import annotations

import calendar
import re
import unicodedata
from datetime import datetime
from typing import Optional

from openai import OpenAI

from src.live_eta.fintraffic import BALTIC_PORT_ALIASES
from src.qa.intent import IntentResult, classify_question

from .context import ConversationContext
from .models import (
    DateScope,
    ETAWatchIntent,
    QueryFiltersPayload,
    QueryMode,
    QueryOperation,
    QueryPlan,
    RoutePair,
    VisualizationIntent,
)


_MONTH_OF_YEAR_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+of\s+(20\d{2})\b",
    re.IGNORECASE,
)
_DAY_MONTH_YEAR_RE = re.compile(
    r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(20\d{2})\b",
    re.IGNORECASE,
)
_MONTH_DAY_YEAR_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?[,]?\s+(20\d{2})\b",
    re.IGNORECASE,
)
_EXPLAIN_PREVIOUS_RE = re.compile(
    r"\b(explain|simplif(?:y|ied)|simpler|plain\s+english|what\s+does\s+that\s+mean)\b",
    re.IGNORECASE,
)
_FOLLOW_UP_RE = re.compile(r"^\s*(?:what|how)\s+about\b|^\s*and\b|\bsame\s+(?:period|query|metric)\b", re.IGNORECASE)
_FOLLOW_UP_PREFIX_RE = re.compile(r"^\s*(?:(?:what|how)\s+about|and)\s+", re.IGNORECASE)
_CURRENT_RE = re.compile(
    r"\b(today|tomorrow|now|currently|current|live|real[- ]time|latest|this\s+week|next\s+week|coming\s+week)\b",
    re.IGNORECASE,
)
_MONTH_ONLY_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b"
    r"(?!\s+(?:of\s+)?20\d{2})",
    re.IGNORECASE,
)
_ALL_VESSEL_TYPES_RE = re.compile(r"\b(?:all|any)\s+(?:vessel|ship)\s+types?\b|\ball\s+(?:vessels|ships)\b", re.IGNORECASE)
_GREETING_RE = re.compile(r"^\s*(hi|hello|hey|good\s+(morning|afternoon|evening))\b[\s!,.?]*$", re.IGNORECASE)

_APP_HELP_PHRASES = (
    "what can you do",
    "how do i use",
    "how can you help",
    "supported queries",
    "supported questions",
    "what questions",
    "your capabilities",
    "explain this app",
)
_MARITIME_RESEARCH_TERMS = (
    "solas",
    "marpol",
    "isps code",
    "imo convention",
    "ilo convention",
    "emsa guidance",
    "eur-lex",
    "regulation",
    "regulatory requirement",
    "chapter v",
    "collision regulations",
    "colregs",
)
_DOCUMENTARY_AIS_RE = re.compile(
    r"\b(?:imo|ais|automatic\s+identification\s+systems?)\b.{0,120}"
    r"\b(?:purpose|safety|guidance|prudent\s+navigation|other\s+available\s+information|"
    r"complete\s+official\s+port[- ]arrival\s+board|authoritative\s+scheduled\s+etas?)\b|"
    r"\b(?:prudent\s+navigation|complete\s+official\s+port[- ]arrival\s+board|"
    r"authoritative\s+scheduled\s+etas?)\b.{0,120}\bais\b",
    re.IGNORECASE,
)
_ANALYTICS_TERMS = (
    "arrival",
    "arrivals",
    "vessel traffic",
    "ship calls",
    "port calls",
    "port-call",
    "dwell",
    "congestion",
    "pressure",
    "occupancy",
    "route time",
    "travel time",
    "voyage duration",
    "carbon",
    "emission",
    "co2",
    "co2e",
    "nox",
    "sox",
    "anomaly",
    "anomalies",
    "ais jump",
    "mmsi",
    "forecast",
    "busy",
    "busier",
    "busiest",
    "busiest port",
    "busiest day",
    "correlation",
    "relationship",
)
_PLOT_TERMS = ("plot", "chart", "graph", "visualize", "trend")
_PRESSURE_BASELINE_RE = re.compile(r"\b(?:baseline|1\.00)\b", re.IGNORECASE)
_VESSEL_TYPE_SCOPE_RE = re.compile(r"\b(?:vessel|ship)\s+(?:type|category)\b", re.IGNORECASE)
_LIVE_ETA_RE = re.compile(
    r"\b(?:etas?|estimated\s+times?\s+of\s+arrival|announced\s+(?:eta|delay|variance)|"
    r"arrival\s+board|scheduled\s+(?:arrival|to\s+arrive)|official\s+arrivals?|"
    r"upcoming\s+(?:official\s+)?arrivals?|vessels?\s+due|ships?\s+due)\b",
    re.IGNORECASE,
)
_ETA_WATCH_REQUEST_RE = re.compile(
    r"\b(?:shift\s+handover|handover|inbound\s+watchlist|"
    r"ais(?:-visible|\s+visible|-reported|\s+reported)|vessel-reported|reported\s+etas?|"
    r"vessels?\s+due|ships?\s+due|due\s+in\s+the\s+next|"
    r"eta\s+(?:change|changes|changed|revision|revisions)|"
    r"stale\s+(?:position|signal|signals)|no\s+valid\s+(?:reported\s+)?eta|"
    r"last\s+observed|where\s+is\s+the\s+next|locate\s+(?:vessel|ship)|"
    r"sweden-bound|baltic-bound)\b",
    re.IGNORECASE,
)
_UNSUPPORTED_LIVE_OPERATION_RE = re.compile(
    r"\b(?:official\s+(?:arrival|arrivals|schedule|schedules|eta|etas)|"
    r"scheduled\s+(?:arrival|arrivals|eta|etas)|"
    r"confirmed\s+(?:delay|delays|arrival|arrivals)|"
    r"actual\s+(?:delay|delays)|berth(?:ing)?(?:\s+(?:assignment|assignments))?|"
    r"crane\s+(?:assignment|availability|status)|"
    r"behind\s+schedule|how\s+delay(?:ed)?|delay(?:ed|s)?)\b",
    re.IGNORECASE,
)
_AIS_DESTINATION_RE = re.compile(
    r"\b(?:ais(?:-visible|\s+visible|-reported|\s+reported)|vessel-reported)\b.{0,80}"
    r"\b(?:destinations?|bound\s+for|heading\s+to|etas?|positions?)\b|"
    r"\breport(?:ing|s|ed)\b.{0,60}\b(?:as\s+(?:its|their)\s+destinations?|etas?)\b",
    re.IGNORECASE,
)
_SWEDISH_DESTINATION_SCOPE_RE = re.compile(
    r"\b(?:sweden|swedish|sweden-bound|swedish-destination)\b.{0,60}"
    r"\b(?:ports?|destinations?|eta|vessels?)\b|"
    r"\b(?:ports?|destinations?|eta|vessels?)\b.{0,60}\b(?:sweden|swedish)\b",
    re.IGNORECASE,
)
_BALTIC_DESTINATION_SCOPE_RE = re.compile(
    r"\bbaltic\b.{0,60}\b(?:ports?|destinations?|eta|vessels?)\b|"
    r"\b(?:ports?|destinations?|eta|vessels?)\b.{0,60}\bbaltic\b",
    re.IGNORECASE,
)
_DELAY_RE = re.compile(
    r"\b(?:delay(?:ed)?|late|lateness|behind\s+schedule|announced\s+(?:eta\s+)?variance)\b|"
    r"\b(?:later|earlier)\s+than\b.{0,40}\b(?:official\s+)?schedule\b",
    re.IGNORECASE,
)
_ETA_COMPARISON_RE = re.compile(
    r"\b(?:compare|comparison|versus|vs\.?)\b.*\b(?:eta|arrivals?|schedule)\b|"
    r"\b(?:eta|arrivals?|schedule)\b.*\b(?:compare|comparison|versus|vs\.?)\b",
    re.IGNORECASE,
)
_VESSEL_NAME_RE = re.compile(
    r"\b(?:vessel|ship)\s+(?:named|name(?:d)?\s+is)\s+"
    r"[\"']?([A-Za-z][A-Za-z0-9 .-]{1,39}?)[\"']?"
    r"(?=\s+(?:at|to|for|in|on|with|today|now|eta|delay|arriv)\b|[,?.!]|$)",
    re.IGNORECASE,
)
_LIVE_ETA_OPERATIONS = {
    QueryOperation.LIVE_PORT_ARRIVALS,
    QueryOperation.VESSEL_ETA,
    QueryOperation.VESSEL_DELAY,
    QueryOperation.ETA_COMPARISON,
}
_ETA_WATCH_COUNTRY_PREFIXES = frozenset(
    {"SE", "FI", "EE", "LV", "LT", "PL", "DE", "DK"}
)
_LIVE_PORT_ALIASES = {
    **{
        name: locode
        for name, locode in BALTIC_PORT_ALIASES.items()
        if locode[:2] in _ETA_WATCH_COUNTRY_PREFIXES
    },
    "gothenburg": "SEGOT",
    "goteborg": "SEGOT",
}


def _normalize_question(question: str) -> str:
    return _MONTH_OF_YEAR_RE.sub(lambda m: f"{m.group(1)} {m.group(2)}", question.strip())


def _explicit_calendar_date(question: str) -> Optional[str]:
    match = _DAY_MONTH_YEAR_RE.search(question)
    if match:
        token = f"{match.group(1)} {match.group(2)} {match.group(3)}"
        try:
            return datetime.strptime(token.title(), "%d %B %Y").date().isoformat()
        except ValueError:
            return None
    match = _MONTH_DAY_YEAR_RE.search(question)
    if match:
        token = f"{match.group(1)} {match.group(2)} {match.group(3)}"
        try:
            return datetime.strptime(token.title(), "%B %d %Y").date().isoformat()
        except ValueError:
            return None
    return None


def _requested_visual(question: str) -> VisualizationIntent:
    q = question.lower()
    if any(token in q for token in ("without a chart", "no chart", "text only")):
        return VisualizationIntent.NONE
    if "heatmap" in q:
        return VisualizationIntent.HEATMAP
    if "area chart" in q or "area graph" in q:
        return VisualizationIntent.AREA
    if any(token in q for token in ("box plot", "boxplot")):
        return VisualizationIntent.BOXPLOT
    if any(token in q for token in ("histogram", "distribution")):
        return VisualizationIntent.DISTRIBUTION
    if "map" in q:
        return VisualizationIntent.MAP
    if "timeline" in q:
        return VisualizationIntent.TIMELINE
    if "table" in q:
        return VisualizationIntent.TABLE
    if "kpi" in q or "single number" in q:
        return VisualizationIntent.KPI
    if any(
        token in q
        for token in (
            "stacked",
            "composition",
            "share",
            "breakdown by vessel type",
            "breakdown by pollutant",
            "pollutant breakdown",
            "emissions breakdown",
        )
    ):
        return VisualizationIntent.STACKED_BAR
    if any(token in q for token in ("bar chart", "bar graph")) or (
        re.search(r"\bbars?\b", q) and any(token in q for token in ("chart", "graph", "plot"))
    ):
        return VisualizationIntent.BAR
    if any(token in q for token in _PLOT_TERMS):
        return VisualizationIntent.LINE
    return VisualizationIntent.AUTO


def _looks_like_analytics(question: str, parsed: IntentResult) -> bool:
    q = question.lower()
    if any(token in q for token in _ANALYTICS_TERMS):
        return True
    entities = parsed.entities or {}
    diagnostics = entities.get("extraction_diagnostics") or {}
    if parsed.intent == "G" and diagnostics.get("unsupported_hits"):
        return True
    if entities.get("mmsi") or entities.get("call_id"):
        return True
    if entities.get("ports") and any(token in q for token in ("how many", "compare", "show", "top", "first", "last")):
        return True
    if entities.get("country_codes") and any(token in q for token in ("most", "top", "rank", "busiest", "arrivals")):
        return True
    return False


def _looks_like_documentary_research(question: str) -> bool:
    q = question.lower()
    imo_ais_purpose = (
        "imo" in q
        and ("automatic identification system" in q or re.search(r"\bais\b", q))
        and any(token in q for token in ("purpose", "safety", "guidance"))
    )
    return (
        any(term in q for term in _MARITIME_RESEARCH_TERMS)
        or bool(_DOCUMENTARY_AIS_RE.search(question))
        or bool(imo_ais_purpose)
    )


def _extract_vessel_name(question: str) -> Optional[str]:
    match = _VESSEL_NAME_RE.search(question)
    if not match:
        return None
    return re.sub(r"\s+", " ", match.group(1)).strip() or None


def _extract_live_ports(question: str) -> list[str]:
    hits: list[tuple[int, str]] = []
    country_pattern = "|".join(sorted(_ETA_WATCH_COUNTRY_PREFIXES))
    for match in re.finditer(
        rf"\b(?:{country_pattern})[A-Z0-9]{{3}}\b",
        question.upper(),
    ):
        hits.append((match.start(), match.group(0)))
    ascii_question = unicodedata.normalize("NFKD", question).encode(
        "ascii", "ignore"
    ).decode("ascii")
    for name, locode in _LIVE_PORT_ALIASES.items():
        for match in re.finditer(
            rf"\b{re.escape(name)}\b",
            ascii_question,
            re.IGNORECASE,
        ):
            hits.append((match.start(), locode))
    output: list[str] = []
    for _, locode in sorted(hits):
        if locode not in output:
            output.append(locode)
    return output


class QueryPlanner:
    """Build a validated plan without ever treating an unknown prompt as arrivals."""

    def __init__(
        self,
        *,
        openai_client: Optional[OpenAI] = None,
        model: str = "gpt-5.6-terra",
        reasoning_effort: str = "medium",
        enable_openai: bool = False,
    ) -> None:
        self.openai_client = openai_client
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.enable_openai = bool(enable_openai and openai_client is not None)

    def plan(
        self,
        question: str,
        *,
        filters: Optional[QueryFiltersPayload] = None,
        context: Optional[ConversationContext] = None,
    ) -> QueryPlan:
        normalized = _normalize_question(question)
        context_plan = self._contextual_plan(normalized, context)
        if context_plan is not None:
            return self._apply_filters(context_plan, filters)

        deterministic = self._deterministic_plan(normalized)
        deterministic = self._apply_filters(deterministic, filters)
        # A deterministic unsupported decision is a hard capability boundary,
        # not an invitation for a probabilistic planner to reinterpret the
        # request as a supported metric.  The optional model may only refine a
        # genuinely general request into another non-analytics route.
        if not self.enable_openai or deterministic.mode != QueryMode.GENERAL_CHAT:
            return deterministic

        planned = self._openai_plan(normalized, deterministic)
        constrained = self._constrain_model_plan(deterministic, planned)
        return self._apply_filters(constrained, filters)

    @staticmethod
    def _constrain_model_plan(deterministic: QueryPlan, planned: Optional[QueryPlan]) -> QueryPlan:
        if planned is None:
            return deterministic
        safe_pairs = {
            (QueryMode.GENERAL_CHAT, QueryOperation.GENERAL_RESPONSE),
            (QueryMode.MARITIME_RESEARCH, QueryOperation.RESEARCH),
            (QueryMode.APP_HELP, QueryOperation.HELP),
            (QueryMode.CLARIFICATION, QueryOperation.UNSUPPORTED),
            (QueryMode.UNSUPPORTED, QueryOperation.UNSUPPORTED),
        }
        if (planned.mode, planned.operation) not in safe_pairs:
            return deterministic

        constrained = planned.model_copy(deep=True)
        if deterministic.date_scope.is_current:
            # Preserve the deterministic temporal boundary even when the
            # model omits or contradicts it.  This prevents a current request
            # from being executed against historical analytics by accident.
            constrained.date_scope = deterministic.date_scope.model_copy(deep=True)
            constrained.date_scope.is_current = True
        return constrained

    def _contextual_plan(
        self,
        question: str,
        context: Optional[ConversationContext],
    ) -> Optional[QueryPlan]:
        if context is None or context.previous_plan is None or context.previous_envelope is None:
            return None
        previous = context.previous_plan
        if _EXPLAIN_PREVIOUS_RE.search(question) and len(question.split()) <= 14:
            inherited = previous.model_copy(deep=True)
            inherited.mode = QueryMode.ANALYTICS
            inherited.operation = QueryOperation.EXPLAIN_PREVIOUS
            inherited.requested_visual = VisualizationIntent.NONE
            inherited.reason = "User requested a simpler explanation of the preceding validated answer."
            inherited.context_inherited = [
                "previous_answer",
                "metric",
                "ports",
                "date_scope",
                "aggregation",
            ]
            inherited.planner_source = "context"
            inherited.planner_model = None
            return inherited

        if not _FOLLOW_UP_RE.search(question) or previous.mode != QueryMode.ANALYTICS:
            return None

        # Parse the meaningful follow-up clause rather than allowing the
        # conversational prefix to pollute route extraction (for example,
        # "What about Gothenburg to Karlshamn?").
        follow_up_clause = _FOLLOW_UP_PREFIX_RE.sub("", _normalize_question(question), count=1)
        parsed = classify_question(follow_up_clause)
        entities = dict(parsed.entities or {})
        diagnostics = entities.get("extraction_diagnostics") or {}
        new_ports = [str(item).strip() for item in entities.get("ports") or [] if str(item).strip()]
        date_scope, date_changed, date_ambiguity = self._contextual_date_scope(
            question=question,
            entities=entities,
            previous=previous.date_scope,
        )

        # Terminal telemetry and other explicitly unsupported metrics stay
        # unsupported even inside an otherwise valid analytics conversation.
        if entities.get("metric") == "unsupported" or diagnostics.get("unsupported_hits"):
            return QueryPlan(
                mode=QueryMode.UNSUPPORTED,
                operation=QueryOperation.UNSUPPORTED,
                metric="unsupported",
                ports=new_ports,
                date_scope=date_scope,
                requested_visual=VisualizationIntent.NONE,
                reason="The follow-up explicitly requests terminal operational data outside Eagle Eye's supported datasets.",
                planner_source="context",
            )

        # Weather is a complete general request, not a request to reuse the
        # preceding analytics metric with a new date.
        if "weather" in question.lower():
            return self._deterministic_plan(question)

        if date_ambiguity:
            return QueryPlan(
                mode=QueryMode.CLARIFICATION,
                operation=QueryOperation.UNSUPPORTED,
                requested_visual=VisualizationIntent.NONE,
                ambiguities=[date_ambiguity],
                clarification=date_ambiguity,
                reason="The follow-up contains a date change whose year cannot be inherited safely.",
                planner_source="context",
            )

        inherited = previous.model_copy(deep=True)
        changed_slots: list[str] = []

        route_pairs = [
            RoutePair(origin=str(item.get("origin", "")).strip(), destination=str(item.get("destination", "")).strip())
            for item in entities.get("route_pairs") or []
            if str(item.get("origin", "")).strip() and str(item.get("destination", "")).strip()
        ]
        origin_port = str(entities.get("origin_port") or "").strip() or None
        destination_port = str(entities.get("destination_port") or "").strip() or None
        if route_pairs and not (origin_port and destination_port):
            origin_port = route_pairs[0].origin
            destination_port = route_pairs[0].destination
        route_changed = bool(route_pairs or (origin_port and destination_port))
        route_operations = {
            QueryOperation.FIRST_ROUTE_VESSEL,
            QueryOperation.ROUTE_TRAVEL_TIME,
            QueryOperation.MIXED_PORT_ROUTE_COMPARISON,
        }

        metric = str(entities.get("metric") or "").strip() or None
        aggregation = str(entities.get("aggregation") or "").strip() or None
        metric_explicit = metric is not None or aggregation is not None
        mapped_operation = self._context_operation(
            previous=inherited,
            metric=metric,
            aggregation=aggregation,
            entities=entities,
            question=question,
            resulting_ports=new_ports or inherited.ports,
        ) if metric_explicit else inherited.operation

        if metric_explicit and mapped_operation is None:
            return QueryPlan(
                mode=QueryMode.CLARIFICATION,
                operation=QueryOperation.UNSUPPORTED,
                requested_visual=VisualizationIntent.NONE,
                ambiguities=[f"The metric change to {metric or aggregation} does not identify a supported analytics operation."],
                clarification="Which supported metric should replace the previous one?",
                reason="A metric change was detected but cannot be mapped without guessing.",
                planner_source="context",
            )

        if route_changed and mapped_operation not in route_operations:
            return QueryPlan(
                mode=QueryMode.CLARIFICATION,
                operation=QueryOperation.UNSUPPORTED,
                requested_visual=VisualizationIntent.NONE,
                ambiguities=["A route was supplied, but the requested route metric is ambiguous."],
                clarification="Should I calculate route travel time or find the first vessel on that route?",
                reason="The follow-up changes route scope without a route-compatible operation.",
                planner_source="context",
            )

        if new_ports:
            if inherited.operation in route_operations and not route_changed and len(new_ports) != 2:
                return QueryPlan(
                    mode=QueryMode.CLARIFICATION,
                    operation=QueryOperation.UNSUPPORTED,
                    requested_visual=VisualizationIntent.NONE,
                    ambiguities=["A single port does not identify which route endpoint should change."],
                    clarification="Which origin and destination should replace the previous route?",
                    reason="The route follow-up changes an ambiguous endpoint.",
                    planner_source="context",
                )
            inherited.ports = new_ports
            changed_slots.append("ports")

        if route_changed:
            inherited.origin_port = origin_port
            inherited.destination_port = destination_port
            inherited.route_pairs = route_pairs or (
                [RoutePair(origin=origin_port, destination=destination_port)]
                if origin_port and destination_port
                else []
            )
            changed_slots.append("route")

        if date_changed:
            inherited.date_scope = date_scope
            changed_slots.append("date_scope")

        vessel_type = str(entities.get("vessel_type") or "").strip() or None
        vessel_explicit = vessel_type is not None or bool(_ALL_VESSEL_TYPES_RE.search(question))
        if vessel_explicit:
            inherited.vessel_type = vessel_type
            changed_slots.append("vessel_type")

        if metric_explicit:
            inherited.metric = metric
            inherited.aggregation = aggregation
            inherited.operation = mapped_operation or inherited.operation
            changed_slots.append("metric")

        if entities.get("mmsi"):
            inherited.mmsi = str(entities["mmsi"])
            changed_slots.append("mmsi")
        if entities.get("imo"):
            inherited.imo = str(entities["imo"])
            changed_slots.append("imo")
        vessel_name = _extract_vessel_name(question)
        if vessel_name:
            inherited.vessel_name = vessel_name
            changed_slots.append("vessel_name")
        if entities.get("source_scope"):
            inherited.source_scope = str(entities["source_scope"])
            changed_slots.append("source_scope")

        if not changed_slots and not re.search(r"\bsame\s+(?:period|query|metric)\b", question, re.IGNORECASE):
            return QueryPlan(
                mode=QueryMode.CLARIFICATION,
                operation=QueryOperation.UNSUPPORTED,
                requested_visual=VisualizationIntent.NONE,
                ambiguities=["The follow-up does not identify which scope should change."],
                clarification="Which port, vessel, route, date, or metric should replace the previous scope?",
                reason="Contextual follow-up was detected but the changed slot is ambiguous.",
                planner_source="context",
            )

        if inherited.operation == QueryOperation.ARRIVALS_MULTI and len(inherited.ports) == 1:
            inherited.operation = QueryOperation.ARRIVALS
        elif inherited.operation == QueryOperation.ARRIVALS and len(inherited.ports) >= 2:
            inherited.operation = QueryOperation.ARRIVALS_MULTI

        inherited.mode = QueryMode.ANALYTICS
        inherited.reason = (
            "Inherited the previous validated analytics plan and replaced only the explicit follow-up slots: "
            + (", ".join(changed_slots) if changed_slots else "none")
            + "."
        )
        inherited.context_inherited = [
            slot
            for slot in ("operation", "metric", "ports", "date_scope", "vessel_type", "route", "aggregation")
            if slot not in changed_slots
        ]
        inherited.planner_source = "context"
        inherited.planner_model = None
        return inherited

    @staticmethod
    def _contextual_date_scope(
        *,
        question: str,
        entities: dict,
        previous: DateScope,
    ) -> tuple[DateScope, bool, Optional[str]]:
        current = bool(entities.get("requires_current_data")) or bool(_CURRENT_RE.search(question))
        date_from = str(entities.get("date_from") or "").strip() or None
        date_to = str(entities.get("date_to") or "").strip() or None
        target_date = str(entities.get("target_date") or "").strip() or None
        relative_window = str(entities.get("window") or "").strip() or None
        if current or date_from or date_to or target_date or relative_window:
            return (
                DateScope(
                    date_from=date_from,
                    date_to=date_to,
                    target_date=target_date,
                    relative_window=relative_window,
                    is_current=current,
                ),
                True,
                None,
            )

        month_match = _MONTH_ONLY_RE.search(question)
        if month_match:
            previous_dates = [previous.date_from, previous.date_to, previous.target_date]
            years = {
                datetime.fromisoformat(value[:10]).year
                for value in previous_dates
                if value
            }
            if len(years) != 1:
                return (
                    previous.model_copy(deep=True),
                    False,
                    f"Which year should I use for {month_match.group(1).title()}?",
                )
            year = next(iter(years))
            month = datetime.strptime(month_match.group(1).title(), "%B").month
            final_day = calendar.monthrange(year, month)[1]
            return (
                DateScope(
                    date_from=f"{year:04d}-{month:02d}-01",
                    date_to=f"{year:04d}-{month:02d}-{final_day:02d}",
                    is_current=False,
                ),
                True,
                None,
            )
        return previous.model_copy(deep=True), False, None

    @staticmethod
    def _context_operation(
        *,
        previous: QueryPlan,
        metric: Optional[str],
        aggregation: Optional[str],
        entities: dict,
        question: str,
        resulting_ports: list[str],
    ) -> Optional[QueryOperation]:
        q = question.lower()
        if previous.operation in _LIVE_ETA_OPERATIONS:
            if _DELAY_RE.search(question):
                return QueryOperation.VESSEL_DELAY
            if _ETA_COMPARISON_RE.search(question):
                return QueryOperation.ETA_COMPARISON
            if re.search(r"\b(?:arrival\s+board|upcoming\s+arrivals?|vessels?\s+due|ships?\s+due)\b", question, re.IGNORECASE):
                return QueryOperation.LIVE_PORT_ARRIVALS
            if _LIVE_ETA_RE.search(question):
                return QueryOperation.VESSEL_ETA
            return previous.operation
        aggregation_map = {
            "first_arrival": QueryOperation.FIRST_ARRIVAL,
            "last_arrival": QueryOperation.LAST_ARRIVAL,
            "first_departure": QueryOperation.FIRST_DEPARTURE,
            "first_route_vessel": QueryOperation.FIRST_ROUTE_VESSEL,
            "route_travel_time_summary": QueryOperation.ROUTE_TRAVEL_TIME,
            "vessel_type_composition": QueryOperation.VESSEL_TYPE_COMPOSITION,
            "dwell_distribution": QueryOperation.DWELL_DISTRIBUTION,
        }
        if aggregation in aggregation_map:
            return aggregation_map[aggregation]
        if metric == previous.metric and aggregation is None:
            return previous.operation
        if metric == "arrival_count":
            if any(token in q for token in ("forecast", "predict", "future", "next", "coming")):
                return QueryOperation.FORECAST_ARRIVALS
            return QueryOperation.ARRIVALS_MULTI if len(resulting_ports) >= 2 else QueryOperation.ARRIVALS
        if metric == "dwell_minutes":
            if entities.get("mmsi"):
                return QueryOperation.MMSI_PORT_STAYS
            if any(token in q for token in ("distribution", "histogram", "box plot", "boxplot", "percentile")):
                return QueryOperation.DWELL_DISTRIBUTION
            return QueryOperation.DWELL_SUMMARY
        if metric == "congestion_index":
            if any(token in q for token in ("forecast", "predict", "future", "next", "coming")):
                return QueryOperation.FORECAST_CONGESTION
            if any(token in q for token in ("why", "cause", "reason", "contributor")):
                return QueryOperation.DIAGNOSTIC
            return QueryOperation.CONGESTION
        if metric == "route_duration_h":
            return QueryOperation.ROUTE_TRAVEL_TIME
        if metric == "arrivals_spike":
            return QueryOperation.ARRIVAL_ANOMALY
        if metric == "ais_jump":
            return QueryOperation.AIS_JUMP
        if metric == "emissions":
            return QueryOperation.CARBON
        return None

    def _deterministic_plan(self, question: str) -> QueryPlan:
        q = question.lower()
        current = bool(_CURRENT_RE.search(question))
        if _GREETING_RE.search(question) or any(phrase in q for phrase in _APP_HELP_PHRASES):
            return QueryPlan(
                mode=QueryMode.APP_HELP,
                operation=QueryOperation.HELP,
                requested_visual=VisualizationIntent.NONE,
                reason="Greeting or Eagle Eye capability question detected.",
            )

        if _looks_like_documentary_research(question):
            return QueryPlan(
                mode=QueryMode.MARITIME_RESEARCH,
                operation=QueryOperation.RESEARCH,
                requested_visual=VisualizationIntent.NONE,
                reason="Maritime regulation or domain-research question detected.",
            )

        live_eta = self._live_eta_plan(question)
        if live_eta is not None:
            return live_eta

        if "weather" in q:
            return QueryPlan(
                mode=QueryMode.GENERAL_CHAT,
                operation=QueryOperation.GENERAL_RESPONSE,
                date_scope=DateScope(is_current=current),
                requested_visual=VisualizationIntent.NONE,
                reason="Weather is not present in Eagle Eye's historical analytics datasets.",
            )

        if re.search(
            r"\bwhere\s+(?:are|is)\b.{0,80}\b(?:vessels?|ships?)\b.{0,80}"
            r"\b(?:located|positioned|now|currently|right\s+now)\b|"
            r"\b(?:current|live|real[- ]time)\b.{0,60}\b(?:vessel|ship)\s+positions?\b",
            question,
            re.IGNORECASE,
        ):
            return QueryPlan(
                mode=QueryMode.ANALYTICS,
                operation=QueryOperation.CURRENT_POSITIONS,
                metric="current_positions",
                date_scope=DateScope(is_current=True),
                requested_visual=VisualizationIntent.NONE,
                reason="Current vessel-position request detected; historical data cannot satisfy it.",
            )

        parsed = classify_question(question)
        if not _looks_like_analytics(question, parsed):
            return QueryPlan(
                mode=QueryMode.GENERAL_CHAT,
                operation=QueryOperation.GENERAL_RESPONSE,
                date_scope=DateScope(is_current=current),
                requested_visual=VisualizationIntent.NONE,
                reason="No supported maritime analytics operation was identified.",
            )

        plan = self._analytics_plan(question, parsed)
        plan.date_scope.is_current = current
        return plan

    def _live_eta_plan(self, question: str) -> Optional[QueryPlan]:
        """Route explicit vessel-broadcast monitoring requests to ETA Watch.

        AISStream supplies AIS broadcasts, not an official schedule, confirmed
        delay, berth plan, or complete arrival board.  Those unsupported
        requests are rejected here before any live provider is called.
        Historical route-duration questions remain on the historical planner.
        """

        ais_destination_request = bool(_AIS_DESTINATION_RE.search(question))
        live_watch_request = bool(_ETA_WATCH_REQUEST_RE.search(question))
        unsupported_live_request = bool(_UNSUPPORTED_LIVE_OPERATION_RE.search(question))
        if not (live_watch_request or ais_destination_request or unsupported_live_request):
            return None
        if (
            re.search(r"\bfrom\b.{1,80}\bto\b", question, re.IGNORECASE)
            and not (ais_destination_request or live_watch_request)
            and not _CURRENT_RE.search(question)
        ):
            return None

        parsed = classify_question(question)
        entities = dict(parsed.entities or {})
        ports = _extract_live_ports(question)
        destination_group = (
            "baltic_destination_signals"
            if _BALTIC_DESTINATION_SCOPE_RE.search(question)
            else "swedish_destination_signals"
            if _SWEDISH_DESTINATION_SCOPE_RE.search(question)
            or re.search(r"\b(?:sweden-bound|shift\s+handover|handover)\b", question, re.IGNORECASE)
            else None
        )
        mmsi = str(entities.get("mmsi") or "").strip() or None
        imo = str(entities.get("imo") or "").strip() or None
        vessel_name = _extract_vessel_name(question)
        explicit_live_scope = bool(
            mmsi
            or imo
            or vessel_name
            or ports
            or destination_group
        )

        if unsupported_live_request:
            return QueryPlan(
                mode=QueryMode.UNSUPPORTED,
                operation=QueryOperation.UNSUPPORTED,
                metric="unsupported_live_port_operation",
                ports=ports,
                date_scope=DateScope(is_current=True),
                mmsi=mmsi,
                imo=imo,
                vessel_name=vessel_name,
                requested_visual=VisualizationIntent.NONE,
                reason=(
                    "AISStream does not provide official schedules, confirmed delays, "
                    "or berth assignments. Ask for vessel-reported ETA changes, an "
                    "AIS-visible inbound watchlist, or the last observed vessel position."
                ),
            )
        if not explicit_live_scope:
            return None

        q = question.lower()
        if re.search(r"\b(?:shift\s+handover|handover)\b", q):
            eta_watch_intent = ETAWatchIntent.SHIFT_HANDOVER
            metric = "operational_exceptions"
        elif re.search(
            r"\b(?:below|under|less\s+than)\s+\d+(?:\.\d+)?\s*(?:kn|knot|knots)\b",
            q,
        ):
            eta_watch_intent = ETAWatchIntent.LOW_SPEED_EXCEPTIONS
            metric = "sog_kn"
        elif re.search(
            r"\beta\s+(?:change|changes|changed|revision|revisions)\b|"
            r"\bchang(?:e|ed|es|ing)\b.{0,40}\b(?:reported\s+)?eta\b",
            q,
        ):
            eta_watch_intent = ETAWatchIntent.ETA_REVISIONS
            metric = "eta_change_minutes"
        elif re.search(
            r"\b(?:stale\s+(?:position|signal|signals)|no\s+valid\s+(?:reported\s+)?eta)\b",
            q,
        ):
            eta_watch_intent = ETAWatchIntent.SIGNAL_QUALITY
            metric = "signal_status"
        elif re.search(
            r"\b(?:most|rank|ranking|busiest)\b.{0,60}\b(?:destination|destinations|ports?)\b|"
            r"\b(?:destination|destinations|ports?)\b.{0,60}\b(?:most|rank|ranking|busiest)\b",
            q,
        ):
            eta_watch_intent = ETAWatchIntent.DESTINATION_LOAD
            metric = "inbound_vessels"
        elif mmsi or imo or vessel_name or re.search(r"\bwhere\s+is\s+the\s+next\b", q):
            eta_watch_intent = ETAWatchIntent.VESSEL_STATUS
            metric = "ais_eta_utc"
        else:
            eta_watch_intent = ETAWatchIntent.INBOUND_WATCHLIST
            metric = "ais_eta_utc"

        horizon_match = re.search(
            r"\b(?:within|next|for\s+the\s+next)\s+(\d{1,3})\s+hours?\b",
            question,
            re.IGNORECASE,
        )
        if horizon_match is None:
            horizon_match = re.search(
                r"\b(\d{1,3})[\s-]+hour\b",
                question,
                re.IGNORECASE,
            )
        limit_match = re.search(
            r"\bnext\s+(\d{1,2})\s+(?:(?:vessel-reported|reported)\s+)?"
            r"(?:etas?|vessels?|ships?|arrivals?)\b",
            question,
            re.IGNORECASE,
        )
        if re.search(
            r"\bnext\s+(?:ais(?:-visible|\s+visible)\s+)?vessel\b",
            question,
            re.IGNORECASE,
        ):
            limit = 1
        elif limit_match:
            limit = max(1, min(int(limit_match.group(1)), 20))
        else:
            limit = 20
        horizon_hours = (
            max(1, min(int(horizon_match.group(1)), 48))
            if horizon_match
            else 12
            if eta_watch_intent == ETAWatchIntent.SHIFT_HANDOVER
            else 48
            if (
                eta_watch_intent
                in {
                    ETAWatchIntent.ETA_REVISIONS,
                    ETAWatchIntent.SIGNAL_QUALITY,
                }
                or (
                    eta_watch_intent == ETAWatchIntent.INBOUND_WATCHLIST
                    and limit_match is not None
                )
            )
            else 24
        )

        speed_match = re.search(
            r"\b(?:below|under|less\s+than)\s+(\d+(?:\.\d+)?)\s*"
            r"(?:kn|knot|knots)\b",
            question,
            re.IGNORECASE,
        )
        speed_threshold_kn = (
            min(float(speed_match.group(1)), 100.0)
            if speed_match
            else 2.0
            if eta_watch_intent in {
                ETAWatchIntent.SHIFT_HANDOVER,
                ETAWatchIntent.LOW_SPEED_EXCEPTIONS,
            }
            else None
        )
        eta_change_match = re.search(
            r"\b(?:more\s+than|over|above)\s+(\d{1,4})\s+minutes?\b",
            question,
            re.IGNORECASE,
        )
        eta_change_threshold_minutes = (
            min(int(eta_change_match.group(1)), 1440)
            if eta_change_match
            else 30
            if eta_watch_intent in {
                ETAWatchIntent.SHIFT_HANDOVER,
                ETAWatchIntent.ETA_REVISIONS,
            }
            else None
        )
        change_window_match = re.search(
            r"\blast\s+(\d{1,3})\s+(minutes?|hours?)\b",
            question,
            re.IGNORECASE,
        )
        if change_window_match:
            change_window_minutes = int(change_window_match.group(1))
            if change_window_match.group(2).lower().startswith("hour"):
                change_window_minutes *= 60
            change_window_minutes = max(1, min(change_window_minutes, 1440))
        elif re.search(r"\blast\s+hour\b", question, re.IGNORECASE):
            change_window_minutes = 60
        else:
            change_window_minutes = (
                60
                if eta_watch_intent
                in {ETAWatchIntent.SHIFT_HANDOVER, ETAWatchIntent.ETA_REVISIONS}
                else None
            )

        explicit_date = _explicit_calendar_date(question)
        target_date = explicit_date or entities.get("target_date")
        position_requested = bool(
            re.search(
                r"\b(?:positions?|where\s+(?:is|are|were)|located|locations?|last\s+observed)\b",
                question,
                re.IGNORECASE,
            )
            or eta_watch_intent
            in {
                ETAWatchIntent.SHIFT_HANDOVER,
                ETAWatchIntent.LOW_SPEED_EXCEPTIONS,
                ETAWatchIntent.VESSEL_STATUS,
            }
        )
        dimensions = [metric]
        if position_requested:
            dimensions.append("position")
        if eta_watch_intent == ETAWatchIntent.DESTINATION_LOAD:
            dimensions.append("destination")

        return QueryPlan(
            mode=QueryMode.ANALYTICS,
            operation=QueryOperation.VESSEL_ETA,
            metric=metric,
            dimensions=dimensions,
            ports=ports,
            date_scope=DateScope(
                date_from=target_date,
                date_to=target_date,
                target_date=target_date,
                is_current=True,
            ),
            mmsi=mmsi,
            imo=imo,
            vessel_name=vessel_name,
            aggregation=destination_group,
            horizon_hours=horizon_hours,
            limit=limit,
            eta_watch_intent=eta_watch_intent,
            speed_threshold_kn=speed_threshold_kn,
            eta_change_threshold_minutes=eta_change_threshold_minutes,
            change_window_minutes=change_window_minutes,
            include_stale=eta_watch_intent
            in {ETAWatchIntent.SHIFT_HANDOVER, ETAWatchIntent.SIGNAL_QUALITY},
            source_scope="aisstream",
            requested_visual=_requested_visual(question),
            reason=(
                "Explicit vessel-reported ETA Watch language was routed to the "
                "AISStream-backed deterministic operational query path."
            ),
        )

    def _analytics_plan(self, question: str, parsed: IntentResult) -> QueryPlan:
        q = question.lower()
        requested_visual = _requested_visual(question)
        entities = dict(parsed.entities or {})
        explicit_date = _explicit_calendar_date(question)
        if explicit_date:
            entities["date_from"] = explicit_date
            entities["date_to"] = explicit_date
            entities["target_date"] = explicit_date
        ports = [str(item).strip() for item in entities.get("ports") or [] if str(item).strip()]
        country_codes = [
            str(item).strip().upper()
            for item in entities.get("country_codes") or []
            if str(item).strip()
        ]
        route_pairs = [
            RoutePair(origin=str(item.get("origin", "")).strip(), destination=str(item.get("destination", "")).strip())
            for item in entities.get("route_pairs") or []
            if str(item.get("origin", "")).strip() and str(item.get("destination", "")).strip()
        ]
        operation = QueryOperation.UNSUPPORTED

        if country_codes and "port" in q and any(
            token in q for token in ("most", "top", "rank", "busiest")
        ):
            operation = QueryOperation.TOP_PORTS
        elif "heatmap" in q and "hour" in q and any(token in q for token in ("weekday", "day of week", "daily")):
            operation = QueryOperation.ARRIVAL_PATTERN
        elif any(token in q for token in ("correlation", "relationship between")) and any(
            metric in q for metric in ("arrival", "dwell", "congestion", "pressure")
        ):
            operation = QueryOperation.CORRELATION
        elif "vessel type" in q and "arrival" in q and any(
            token in q for token in ("share", "composition", "breakdown", "break down")
        ):
            operation = QueryOperation.VESSEL_TYPE_COMPOSITION
        elif (
            entities.get("metric") == "congestion_index"
            and _VESSEL_TYPE_SCOPE_RE.search(question)
            and parsed.intent not in {"C", "F"}
        ):
            operation = QueryOperation.PRESSURE_BY_VESSEL_TYPE
        elif (
            entities.get("metric") == "congestion_index"
            and _PRESSURE_BASELINE_RE.search(question)
            and parsed.intent not in {"C", "F"}
        ):
            operation = QueryOperation.CONGESTION
        elif entities.get("metric") == "dwell_minutes" and entities.get("mmsi"):
            operation = QueryOperation.MMSI_PORT_STAYS
        elif entities.get("metric") == "dwell_minutes" and any(
            token in q for token in ("distribution", "histogram", "box plot", "boxplot", "percentile")
        ):
            operation = QueryOperation.DWELL_DISTRIBUTION
        elif entities.get("metric") == "dwell_minutes":
            operation = QueryOperation.DWELL_SUMMARY
        elif any(token in q for token in ("arrival", "arrive", "arriving")) and any(
            token in q for token in ("today", "now", "current", "live", "real-time")
        ):
            operation = QueryOperation.CURRENT_ARRIVALS
        elif entities.get("dow") and entities.get("dow_compare") and any(token in q for token in ("busy", "busier", "busiest")):
            operation = QueryOperation.WEEKDAY_COMPARISON
        elif (
            "arrival" in q
            and requested_visual not in {VisualizationIntent.AUTO, VisualizationIntent.NONE}
            and len(ports) <= 1
            and not any(token in q for token in ("forecast", "predict", "future", "anomal"))
        ):
            operation = QueryOperation.ARRIVALS
        elif parsed.intent == "H":
            operation = QueryOperation.CARBON
        elif parsed.intent == "G":
            operation = QueryOperation.UNSUPPORTED
        elif parsed.intent == "F":
            operation = QueryOperation.AIS_JUMP if entities.get("metric") == "ais_jump" else QueryOperation.ARRIVAL_ANOMALY
        elif parsed.intent == "C":
            explicit_forecast = any(token in q for token in ("forecast", "predict", "future", "next", "coming", "horizon"))
            historical_scope = bool(entities.get("date_from") and entities.get("date_to"))
            if historical_scope and not explicit_forecast and any(token in q for token in ("arrival", "arrivals", "ship calls")):
                operation = QueryOperation.ARRIVALS_MULTI if len(ports) >= 2 else QueryOperation.ARRIVALS
            elif entities.get("metric") == "arrival_count" and len(ports) < 2:
                operation = QueryOperation.FORECAST_ARRIVALS
            else:
                operation = QueryOperation.FORECAST_COMPARISON if len(ports) >= 2 or entities.get("dow_compare") else QueryOperation.FORECAST_CONGESTION
        elif parsed.intent == "D":
            operation = (
                QueryOperation.MIXED_PORT_ROUTE_COMPARISON
                if route_pairs and ports
                else QueryOperation.PORT_COMPARISON
            )
        elif parsed.intent == "E":
            operation = QueryOperation.DIAGNOSTIC
        elif parsed.intent == "B":
            if entities.get("dow") and entities.get("dow_compare"):
                operation = (
                    QueryOperation.CONGESTION_WEEKDAY_COMPARISON
                    if entities.get("metric") == "congestion_index"
                    else QueryOperation.WEEKDAY_COMPARISON
                )
            elif "hour" in q:
                operation = QueryOperation.BUSIEST_HOUR
            else:
                operation = QueryOperation.BUSIEST_WEEKDAY
        elif parsed.intent == "A":
            aggregation = entities.get("aggregation")
            if aggregation == "first_arrival":
                operation = QueryOperation.FIRST_ARRIVAL
            elif aggregation == "last_arrival":
                operation = QueryOperation.LAST_ARRIVAL
            elif aggregation == "first_departure":
                operation = QueryOperation.FIRST_DEPARTURE
            elif aggregation == "first_route_vessel":
                operation = QueryOperation.FIRST_ROUTE_VESSEL
            elif aggregation == "route_travel_time_summary":
                operation = QueryOperation.ROUTE_TRAVEL_TIME
            elif aggregation == "peak_day" and entities.get("metric") == "congestion_index":
                operation = QueryOperation.PEAK_CONGESTION_DAY
            elif aggregation == "peak_day":
                operation = QueryOperation.PEAK_ARRIVAL_DAY
            elif "pressure" in q and _VESSEL_TYPE_SCOPE_RE.search(question):
                operation = QueryOperation.PRESSURE_BY_VESSEL_TYPE
            elif entities.get("metric") == "arrival_count" and any(
                token in q for token in ("share of arrivals", "composition", "by vessel type", "breakdown by vessel type")
            ):
                operation = QueryOperation.VESSEL_TYPE_COMPOSITION
            elif aggregation == "vessel_type_composition":
                operation = QueryOperation.VESSEL_TYPE_COMPOSITION
            elif entities.get("metric") == "dwell_minutes" and any(
                token in q for token in ("distribution", "histogram", "box plot", "boxplot", "percentile")
            ):
                operation = QueryOperation.DWELL_DISTRIBUTION
            elif entities.get("mmsi") and entities.get("metric") == "dwell_minutes":
                operation = QueryOperation.MMSI_PORT_STAYS
            elif entities.get("metric") == "dwell_minutes":
                operation = QueryOperation.DWELL_SUMMARY
            elif "top" in q and "port" in q:
                operation = QueryOperation.TOP_PORTS
            elif "congestion" in q or "pressure" in q:
                operation = QueryOperation.CONGESTION
            elif len(ports) >= 2:
                operation = QueryOperation.ARRIVALS_MULTI
            else:
                operation = QueryOperation.ARRIVALS

        mode = QueryMode.ANALYTICS if operation != QueryOperation.UNSUPPORTED else QueryMode.UNSUPPORTED
        clarification = None
        ambiguities = []
        if operation in {
            QueryOperation.WEEKDAY_COMPARISON,
            QueryOperation.CONGESTION_WEEKDAY_COMPARISON,
        } and not (entities.get("dow") and entities.get("dow_compare")):
            mode = QueryMode.CLARIFICATION
            ambiguities.append("A weekday comparison requires two weekdays.")
            clarification = "Which two weekdays should I compare?"

        route_operation = operation in {
            QueryOperation.FIRST_ROUTE_VESSEL,
            QueryOperation.ROUTE_TRAVEL_TIME,
            QueryOperation.MIXED_PORT_ROUTE_COMPARISON,
        }
        return QueryPlan(
            mode=mode,
            operation=operation,
            metric=entities.get("metric"),
            dimensions=(
                ["day_of_week", "hour"]
                if operation == QueryOperation.ARRIVAL_PATTERN
                else ["date"]
                if any(token in q for token in _PLOT_TERMS)
                or (
                    requested_visual in {VisualizationIntent.LINE, VisualizationIntent.AREA, VisualizationIntent.BAR}
                    and any(token in q for token in ("daily", "over time", "by date", "time series"))
                )
                or (
                    operation == QueryOperation.CARBON
                    and any(
                        token in q
                        for token in ("daily", "monthly", "per day", "per month", "by day", "by month")
                    )
                )
                or (
                    operation == QueryOperation.ARRIVALS
                    and any(
                        token in q
                        for token in ("daily", "per day", "by day", "by date", "time series", "over time")
                    )
                )
                else []
            ),
            ports=ports,
            country_codes=country_codes,
            origin_port=entities.get("origin_port") if route_operation else None,
            destination_port=entities.get("destination_port") if route_operation else None,
            route_pairs=route_pairs if route_operation else [],
            date_scope=DateScope(
                date_from=entities.get("date_from"),
                date_to=entities.get("date_to"),
                target_date=entities.get("target_date"),
                relative_window=entities.get("window"),
            ),
            vessel_type=entities.get("vessel_type"),
            mmsi=entities.get("mmsi"),
            imo=entities.get("imo"),
            call_id=entities.get("call_id"),
            aggregation=entities.get("aggregation"),
            day_of_week=entities.get("dow"),
            compare_day_of_week=entities.get("dow_compare"),
            horizon_weeks=int(entities.get("horizon_weeks") or 4),
            limit=int(entities.get("limit") or 1),
            source_scope=entities.get("source_scope"),
            carbon_boundary=str(entities.get("boundary") or "TTW"),
            pollutants=[str(item) for item in entities.get("pollutants") or []],
            requested_visual=requested_visual,
            ambiguities=ambiguities,
            clarification=clarification,
            reason=parsed.reason,
        )

    def _apply_filters(
        self,
        plan: QueryPlan,
        filters: Optional[QueryFiltersPayload],
    ) -> QueryPlan:
        if filters is None:
            return plan
        updated = plan.model_copy(deep=True)
        if filters.port:
            updated.ports = [filters.port]
        if filters.date_from:
            updated.date_scope.date_from = filters.date_from
        if filters.date_to:
            updated.date_scope.date_to = filters.date_to
        if filters.vessel_type:
            updated.vessel_type = filters.vessel_type
        if filters.vessel_name:
            updated.vessel_name = filters.vessel_name
        if filters.mmsi:
            updated.mmsi = filters.mmsi
        if filters.imo:
            updated.imo = filters.imo
        return updated

    def _openai_plan(self, question: str, fallback: QueryPlan) -> Optional[QueryPlan]:
        if self.openai_client is None:
            return None
        instructions = (
            "Classify one Eagle Eye request into the supplied QueryPlan schema. "
            "Never choose analytics unless the user asks for a metric supported by historical AIS, port-call, "
            "congestion, dwell, forecast, route, anomaly, or carbon data. Unknown requests are general_chat, "
            "not arrivals. Regulation and maritime-domain explanations are maritime_research. Do not calculate "
            "numbers or invent ports, dates, entities, or capabilities. Use clarification when a required scope "
            "is genuinely ambiguous."
        )
        try:
            response = self.openai_client.responses.parse(
                model=self.model,
                reasoning={"effort": self.reasoning_effort},
                instructions=instructions,
                input=question,
                text_format=QueryPlan,
            )
            parsed = getattr(response, "output_parsed", None)
            if not isinstance(parsed, QueryPlan):
                return None
            parsed.planner_source = "openai_structured"
            parsed.planner_model = str(getattr(response, "model", None) or self.model)
            return parsed
        except Exception:
            return fallback
