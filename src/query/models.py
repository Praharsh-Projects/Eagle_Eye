"""Strict public contracts for the canonical Eagle Eye query API."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    """Forbid silent wire-contract drift."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class QueryMode(str, Enum):
    ANALYTICS = "analytics"
    MARITIME_RESEARCH = "maritime_research"
    GENERAL_CHAT = "general_chat"
    APP_HELP = "app_help"
    CLARIFICATION = "clarification"
    UNSUPPORTED = "unsupported"


class QueryOperation(str, Enum):
    ARRIVALS = "arrivals"
    ARRIVALS_MULTI = "arrivals_multi"
    TOP_PORTS = "top_ports"
    PEAK_ARRIVAL_DAY = "peak_arrival_day"
    BUSIEST_WEEKDAY = "busiest_weekday"
    BUSIEST_HOUR = "busiest_hour"
    ARRIVAL_PATTERN = "arrival_pattern"
    WEEKDAY_COMPARISON = "weekday_comparison"
    VESSEL_TYPE_COMPOSITION = "vessel_type_composition"
    DWELL_SUMMARY = "dwell_summary"
    DWELL_DISTRIBUTION = "dwell_distribution"
    MMSI_PORT_STAYS = "mmsi_port_stays"
    CONGESTION = "congestion"
    PEAK_CONGESTION_DAY = "peak_congestion_day"
    CONGESTION_WEEKDAY_COMPARISON = "congestion_weekday_comparison"
    PRESSURE_BY_VESSEL_TYPE = "pressure_by_vessel_type"
    PORT_COMPARISON = "port_comparison"
    FORECAST_ARRIVALS = "forecast_arrivals"
    FORECAST_CONGESTION = "forecast_congestion"
    FORECAST_COMPARISON = "forecast_comparison"
    LIVE_PORT_ARRIVALS = "live_port_arrivals"
    VESSEL_ETA = "vessel_eta"
    VESSEL_DELAY = "vessel_delay"
    ETA_COMPARISON = "eta_comparison"
    DIAGNOSTIC = "diagnostic"
    CORRELATION = "correlation"
    ARRIVAL_ANOMALY = "arrival_anomaly"
    AIS_JUMP = "ais_jump"
    CARBON = "carbon"
    FIRST_ARRIVAL = "first_arrival"
    LAST_ARRIVAL = "last_arrival"
    FIRST_DEPARTURE = "first_departure"
    FIRST_ROUTE_VESSEL = "first_route_vessel"
    ROUTE_TRAVEL_TIME = "route_travel_time"
    MIXED_PORT_ROUTE_COMPARISON = "mixed_port_route_comparison"
    EXPLAIN_PREVIOUS = "explain_previous"
    RESEARCH = "research"
    CURRENT_ARRIVALS = "current_arrivals"
    CURRENT_POSITIONS = "current_positions"
    GENERAL_RESPONSE = "general_response"
    HELP = "help"
    UNSUPPORTED = "unsupported"


class AnswerState(str, Enum):
    COMPUTED = "COMPUTED"
    PARTIAL = "PARTIAL"
    RETRIEVED = "RETRIEVED"
    GENERAL = "GENERAL"
    NO_DATA = "NO_DATA"
    NO_CURRENT_DATA = "NO_CURRENT_DATA"
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    UNSUPPORTED = "UNSUPPORTED"
    ASSURANCE_UNAVAILABLE = "ASSURANCE_UNAVAILABLE"
    ERROR = "ERROR"


class VisualizationIntent(str, Enum):
    AUTO = "auto"
    NONE = "none"
    KPI = "kpi"
    LINE = "line"
    AREA = "area"
    BAR = "bar"
    STACKED_BAR = "stacked_bar"
    DISTRIBUTION = "distribution"
    BOXPLOT = "boxplot"
    HEATMAP = "heatmap"
    MAP = "map"
    TIMELINE = "timeline"
    TABLE = "table"


class ETAWatchIntent(str, Enum):
    """Deterministic operational intent for vessel-reported AIS monitoring."""

    SHIFT_HANDOVER = "shift_handover"
    INBOUND_WATCHLIST = "inbound_watchlist"
    LOW_SPEED_EXCEPTIONS = "low_speed_exceptions"
    DESTINATION_LOAD = "destination_load"
    ETA_REVISIONS = "eta_revisions"
    SIGNAL_QUALITY = "signal_quality"
    VESSEL_STATUS = "vessel_status"


class QueryFiltersPayload(StrictModel):
    port: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    vessel_type: Optional[str] = None
    vessel_name: Optional[str] = None
    mmsi: Optional[str] = None
    imo: Optional[str] = None
    anomaly: Optional[bool] = None

    @field_validator(
        "port",
        "date_from",
        "date_to",
        "vessel_type",
        "vessel_name",
        "mmsi",
        "imo",
    )
    @classmethod
    def _strip_optional_strings(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class QueryRequest(StrictModel):
    question: str = Field(min_length=1, max_length=4000)
    conversation_id: Optional[str] = Field(default=None, max_length=128)
    top_k_evidence: int = Field(
        default=5,
        ge=0,
        le=10,
        description="Maximum local/vector evidence rows to retrieve; 0 disables evidence retrieval.",
    )
    filters: QueryFiltersPayload = Field(default_factory=QueryFiltersPayload)

    @field_validator("question")
    @classmethod
    def _strip_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must not be blank")
        return cleaned

    @field_validator("conversation_id")
    @classmethod
    def _strip_conversation_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class DateScope(StrictModel):
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    target_date: Optional[str] = None
    relative_window: Optional[str] = None
    is_current: bool = False

    @model_validator(mode="after")
    def _ordered_dates(self) -> "DateScope":
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must be on or before date_to")
        return self


class RoutePair(StrictModel):
    origin: str
    destination: str


class QueryPlan(StrictModel):
    mode: QueryMode
    operation: QueryOperation
    metric: Optional[str] = None
    dimensions: List[str] = Field(default_factory=list)
    ports: List[str] = Field(default_factory=list)
    country_codes: List[str] = Field(default_factory=list)
    origin_port: Optional[str] = None
    destination_port: Optional[str] = None
    route_pairs: List[RoutePair] = Field(default_factory=list)
    date_scope: DateScope = Field(default_factory=DateScope)
    vessel_type: Optional[str] = None
    mmsi: Optional[str] = None
    imo: Optional[str] = None
    vessel_name: Optional[str] = None
    call_id: Optional[str] = None
    aggregation: Optional[str] = None
    day_of_week: Optional[str] = None
    compare_day_of_week: Optional[str] = None
    horizon_weeks: int = Field(default=4, ge=1, le=12)
    horizon_hours: Optional[int] = Field(default=None, ge=1, le=336)
    limit: int = Field(default=1, ge=1, le=20)
    eta_watch_intent: Optional[ETAWatchIntent] = None
    speed_threshold_kn: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    eta_change_threshold_minutes: Optional[int] = Field(
        default=None,
        ge=1,
        le=1440,
    )
    change_window_minutes: Optional[int] = Field(default=None, ge=1, le=1440)
    include_stale: bool = False
    source_scope: Optional[str] = None
    carbon_boundary: str = "TTW"
    pollutants: List[str] = Field(default_factory=list)
    requested_visual: VisualizationIntent = VisualizationIntent.AUTO
    ambiguities: List[str] = Field(default_factory=list)
    clarification: Optional[str] = None
    reason: str
    context_inherited: List[str] = Field(default_factory=list)
    planner_source: Literal["deterministic", "openai_structured", "context"] = "deterministic"
    planner_model: Optional[str] = None

    @field_validator("country_codes")
    @classmethod
    def _normalize_country_codes(cls, values: List[str]) -> List[str]:
        output: List[str] = []
        for value in values:
            code = str(value or "").strip().upper()
            if not code:
                continue
            if len(code) != 2 or not code.isalpha():
                raise ValueError("country_codes must contain ISO alpha-2 codes")
            if code not in output:
                output.append(code)
        return output


class FactSlot(StrictModel):
    name: str
    value: Any
    unit: Optional[str] = None
    entity: Optional[str] = None
    source: Literal["computed", "retrieved", "model", "system"]
    immutable: bool = True


class AppliedScope(StrictModel):
    ports: List[str] = Field(default_factory=list)
    country_codes: List[str] = Field(default_factory=list)
    origin_port: Optional[str] = None
    destination_port: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    target_date: Optional[str] = None
    vessel_type: Optional[str] = None
    mmsi: Optional[str] = None
    imo: Optional[str] = None
    vessel_name: Optional[str] = None
    horizon_hours: Optional[int] = Field(default=None, ge=1, le=336)
    source_scope: Optional[str] = None


class ColumnSpec(StrictModel):
    field: str
    label: str
    data_type: Literal["string", "number", "integer", "boolean", "datetime"]
    unit: Optional[str] = None


class DatasetSpec(StrictModel):
    id: str
    columns: List[ColumnSpec]
    rows: List[Dict[str, Any]]
    row_count: int = Field(ge=0)


class EvidenceItem(StrictModel):
    id: str
    source_type: Literal["computed", "traffic_event", "local_document", "web", "system"]
    title: str
    excerpt: Optional[str] = None
    url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FreshnessInfo(StrictModel):
    data_from: Optional[str] = None
    data_to: Optional[str] = None
    as_of: Optional[str] = None
    historical: bool = True
    message: str


class AssuranceAssessment(StrictModel):
    status: Literal["verified", "unavailable", "not_applicable"]
    level: Literal["high", "not_applicable"]
    basis: Literal[
        "direct_computation",
        "official_live_source",
        "validated_model",
        "source_grounded_research",
        "system_response",
    ]
    reason: str
    checks: List[str] = Field(default_factory=list)


class AvailabilityInfo(StrictModel):
    code: Literal[
        "available",
        "no_data",
        "source_unavailable",
        "source_stale",
        "coverage_unavailable",
        "ambiguous_match",
        "not_applicable",
    ]
    provider: Optional[str] = None
    retryable: bool = False


class TraceInfo(StrictModel):
    trace_id: str
    route: str
    operation: str
    planner_source: str
    planner_model: Optional[str] = None
    model: str = "deterministic"
    reasoning_effort: Optional[str] = None
    sources: List[str] = Field(default_factory=list)
    retrieval_mode: str = "none"
    retrieval_backend: Optional[str] = None
    retrieval_status: str = "not_applicable"
    retrieval_top_k: int = Field(default=0, ge=0)
    result_state: str = "UNKNOWN"
    failure_state: Optional[str] = None
    result_hash: str = ""
    data_manifest_version: Optional[str] = None
    dataset_rows: int = 0
    visualization_decision: str
    visualization_contract_version: Literal["2.1"] = "2.1"
    chart_profile: List[str] = Field(default_factory=list)
    visualization_dataset_ids: List[str] = Field(default_factory=list)
    visualization_fallback_reasons: List[str] = Field(default_factory=list)
    latency_ms: float = Field(ge=0)
    warnings: List[str] = Field(default_factory=list)


class VisualizationCommon(StrictModel):
    id: str
    title: str
    dataset_id: Optional[str] = None
    accessible_summary: str
    table_fallback_dataset_id: Optional[str] = None
    citations: List[str] = Field(default_factory=list)
    row_id_field: Optional[str] = None


class KPIThresholdSpec(StrictModel):
    id: str
    label: str
    value: float
    unit: Optional[str] = None


class KPIVisualization(VisualizationCommon):
    kind: Literal["kpi"] = "kpi"
    value_field: str
    label: str
    unit: Optional[str] = None
    trend_field: Optional[str] = None
    comparison_field: Optional[str] = None
    baseline_value: Optional[float] = None
    thresholds: List[KPIThresholdSpec] = Field(default_factory=list)


class CartesianHighlight(StrictModel):
    condition_field: str = Field(min_length=1)
    value_field: str = Field(min_length=1)
    label: str = Field(min_length=1)


class ReferenceLineSpec(StrictModel):
    id: str
    label: str
    axis: Literal["x", "y"] = "y"
    value: Union[str, float]
    unit: Optional[str] = None
    line_style: Literal["solid", "dashed", "dotted"] = "dashed"


class IntervalBandSpec(StrictModel):
    id: str
    label: str
    lower_field: str
    upper_field: str
    unit: Optional[str] = None
    point_field: Optional[str] = None
    display: Literal["band", "whisker"] = "band"


class FlaggedPointAnnotation(StrictModel):
    id: str
    label: str
    condition_field: str
    x_field: str
    y_field: str


class FittedSeriesBinding(StrictModel):
    id: str
    label: str
    x_field: str
    y_field: str
    method: Literal["ols", "rolling_median"]
    association_only: bool = False
    slope: Optional[float] = None
    intercept: Optional[float] = None
    r_squared: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class CartesianVisualization(VisualizationCommon):
    kind: Literal["cartesian"] = "cartesian"
    chart_type: Literal["line", "area", "bar", "grouped_bar", "stacked_bar", "scatter"]
    x_field: str
    y_fields: List[str] = Field(min_length=1)
    series_field: Optional[str] = None
    orientation: Literal["vertical", "horizontal"] = "vertical"
    sort: Literal["none", "ascending", "descending", "calendar"] = "none"
    x_unit: Optional[str] = None
    y_unit: Optional[str] = None
    stacked: bool = False
    highlight: Optional[CartesianHighlight] = None
    reference_lines: List[ReferenceLineSpec] = Field(default_factory=list)
    interval_bands: List[IntervalBandSpec] = Field(default_factory=list)
    annotations: List[FlaggedPointAnnotation] = Field(default_factory=list)
    fitted_series: List[FittedSeriesBinding] = Field(default_factory=list)


class ForecastQualitySpec(StrictModel):
    mase: float = Field(ge=0.0)
    interval_coverage: float = Field(ge=0.0, le=1.0)
    interval_level: float = Field(default=0.8, gt=0.0, lt=1.0)
    gate_passed: Literal[True] = True


class ForecastVisualization(VisualizationCommon):
    kind: Literal["forecast"] = "forecast"
    date_field: str
    predicted_field: str
    lower_field: str
    upper_field: str
    actual_field: Optional[str] = None
    unit: Optional[str] = None
    interval_level: float = Field(default=0.8, gt=0.0, lt=1.0)
    forecast_boundary: Optional[str] = None
    quality_metrics: Optional[ForecastQualitySpec] = None


class FiveNumberSummary(StrictModel):
    minimum: float
    q1: float
    median: float
    q3: float
    maximum: float
    lower_whisker: float
    upper_whisker: float
    p90: Optional[float] = None
    count: int = Field(ge=1)


class DistributionVisualization(VisualizationCommon):
    kind: Literal["distribution"] = "distribution"
    chart_type: Literal["histogram", "boxplot", "percentile"]
    value_field: str
    count_field: Optional[str] = None
    category_field: Optional[str] = None
    bins: Optional[int] = Field(default=None, ge=5, le=100)
    unit: Optional[str] = None
    bin_lower_field: Optional[str] = None
    bin_upper_field: Optional[str] = None
    five_number_summary: Optional[FiveNumberSummary] = None
    summary_dataset_id: Optional[str] = None
    outlier_dataset_id: Optional[str] = None
    outlier_condition_field: Optional[str] = None
    outlier_value_field: Optional[str] = None


class HeatmapVisualization(VisualizationCommon):
    kind: Literal["heatmap"] = "heatmap"
    x_field: str
    y_field: str
    value_field: str
    unit: Optional[str] = None


class MapVisualization(VisualizationCommon):
    kind: Literal["map"] = "map"
    latitude_field: str
    longitude_field: str
    label_field: Optional[str] = None
    value_field: Optional[str] = None
    geometry_mode: Literal["points", "segments", "ordered_path"] = "points"
    start_latitude_field: Optional[str] = None
    start_longitude_field: Optional[str] = None
    end_latitude_field: Optional[str] = None
    end_longitude_field: Optional[str] = None
    path_field: Optional[str] = None
    sequence_field: Optional[str] = None
    timestamp_field: Optional[str] = None


class TimelineVisualization(VisualizationCommon):
    kind: Literal["timeline"] = "timeline"
    time_field: str
    label_field: str
    detail_fields: List[str] = Field(default_factory=list)
    end_time_field: Optional[str] = None
    lane_field: Optional[str] = None


class TableVisualization(VisualizationCommon):
    kind: Literal["table"] = "table"
    visible_fields: List[str] = Field(default_factory=list)


class OmittedVisualization(VisualizationCommon):
    kind: Literal["omitted"] = "omitted"
    reason_code: Literal[
        "not_requested",
        "insufficient_data",
        "not_meaningful",
        "unsupported_visual",
        "validation_failed",
        "stale_data",
        "coverage_unavailable",
        "missing_route_dataset",
    ]
    reason: str


VisualizationSpec = Annotated[
    Union[
        KPIVisualization,
        CartesianVisualization,
        ForecastVisualization,
        DistributionVisualization,
        HeatmapVisualization,
        MapVisualization,
        TimelineVisualization,
        TableVisualization,
        OmittedVisualization,
    ],
    Field(discriminator="kind"),
]


class ChartInsight(StrictModel):
    id: str
    visualization_id: str
    insight_type: Literal[
        "peak",
        "period_median",
        "trend",
        "baseline_deviation",
        "boundary_delta",
        "distribution_summary",
        "association",
        "forecast_quality",
        "threshold_exceedance",
        "ranking_margin",
        "dominant_share",
        "movement_anomaly",
    ]
    statement: str = Field(min_length=1)
    fact_names: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)


class OperationalBriefItem(StrictModel):
    row_id: str
    vessel_label: Optional[str] = None
    priority: Literal["attention", "monitor", "information"]
    status: Literal[
        "due_soon",
        "low_speed",
        "eta_changed",
        "stale_position",
        "missing_eta",
        "observed",
    ]
    reason: str
    actions: List[
        Literal[
            "locate_vessel",
            "watch_next_six_hours",
            "inspect_eta_changes",
        ]
    ] = Field(default_factory=list)


class OperationalExceptionSummary(StrictModel):
    code: Literal[
        "due_soon",
        "low_speed",
        "eta_changed",
        "stale_position",
        "missing_eta",
    ]
    count: int = Field(ge=0)
    summary: str


class OperationalBrief(StrictModel):
    intent: ETAWatchIntent
    headline: str
    window_start_utc: Optional[str] = None
    window_end_utc: Optional[str] = None
    matched_count: int = Field(ge=0)
    displayed_count: int = Field(ge=0)
    prioritized_items: List[OperationalBriefItem] = Field(default_factory=list)
    exceptions: List[OperationalExceptionSummary] = Field(default_factory=list)
    source_health: Literal["connecting", "warming", "live", "stale", "unavailable"]
    source_observed_at: Optional[str] = None
    coverage: str

    @model_validator(mode="after")
    def _displayed_within_matches(self) -> "OperationalBrief":
        if self.displayed_count > self.matched_count:
            raise ValueError("displayed_count cannot exceed matched_count")
        return self


class AnswerEnvelope(StrictModel):
    api_version: Literal["2.0"] = "2.0"
    visualization_contract_version: Literal["2.1"] = "2.1"
    conversation_id: str
    turn_id: str
    question: str
    mode: QueryMode
    state: AnswerState
    answer: str
    plan: QueryPlan
    facts: List[FactSlot] = Field(default_factory=list)
    applied_scope: AppliedScope = Field(default_factory=AppliedScope)
    datasets: List[DatasetSpec] = Field(default_factory=list)
    visualizations: List[VisualizationSpec] = Field(default_factory=list)
    chart_insights: List[ChartInsight] = Field(default_factory=list, max_length=3)
    operational_brief: Optional[OperationalBrief] = None
    evidence: List[EvidenceItem] = Field(default_factory=list)
    freshness: FreshnessInfo
    confidence: Literal["high", "medium", "low", "not_applicable"]
    assurance: Optional[AssuranceAssessment] = None
    availability: Optional[AvailabilityInfo] = None
    caveats: List[str] = Field(default_factory=list)
    trace: TraceInfo


class ExportRequest(StrictModel):
    conversation_id: str = Field(min_length=1, max_length=128)
    turn_id: str = Field(min_length=1, max_length=128)
    dataset_id: str = Field(min_length=1, max_length=128)
    format: Literal["csv", "json"] = "csv"


class ExportResponse(StrictModel):
    export_id: str
    format: Literal["csv", "json"]
    dataset_id: str
    row_count: int
    path: str


class FeedbackRequest(StrictModel):
    prompt: str = Field(min_length=1, max_length=4000)
    trace_id: str = Field(min_length=1, max_length=128)
    note: Optional[str] = Field(default=None, max_length=2000)

    @field_validator("prompt", "trace_id")
    @classmethod
    def _strip_required_feedback_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("value must not be blank")
        return cleaned

    @field_validator("note")
    @classmethod
    def _strip_optional_feedback_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return value.strip() or None


class FeedbackResponse(StrictModel):
    feedback_id: str
    status: Literal["accepted"] = "accepted"
