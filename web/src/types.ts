export type QueryMode =
  | "analytics"
  | "maritime_research"
  | "general_chat"
  | "app_help"
  | "clarification"
  | "unsupported";

export type QueryOperation =
  | "arrivals"
  | "arrivals_multi"
  | "top_ports"
  | "peak_arrival_day"
  | "busiest_weekday"
  | "busiest_hour"
  | "arrival_pattern"
  | "weekday_comparison"
  | "vessel_type_composition"
  | "dwell_summary"
  | "dwell_distribution"
  | "mmsi_port_stays"
  | "congestion"
  | "peak_congestion_day"
  | "congestion_weekday_comparison"
  | "pressure_by_vessel_type"
  | "port_comparison"
  | "forecast_arrivals"
  | "forecast_congestion"
  | "forecast_comparison"
  | "current_arrivals"
  | "current_positions"
  | "live_port_arrivals"
  | "vessel_eta"
  | "vessel_delay"
  | "eta_comparison"
  | "diagnostic"
  | "correlation"
  | "arrival_anomaly"
  | "ais_jump"
  | "carbon"
  | "first_arrival"
  | "last_arrival"
  | "first_departure"
  | "first_route_vessel"
  | "route_travel_time"
  | "mixed_port_route_comparison"
  | "explain_previous"
  | "research"
  | "general_response"
  | "help"
  | "unsupported";

export type ResultState =
  | "COMPUTED"
  | "PARTIAL"
  | "RETRIEVED"
  | "GENERAL"
  | "NO_DATA"
  | "NO_CURRENT_DATA"
  | "CLARIFICATION_REQUIRED"
  | "UNSUPPORTED"
  | "ERROR"
  | "ASSURANCE_UNAVAILABLE";

export type VisualizationIntent =
  | "auto"
  | "none"
  | "kpi"
  | "line"
  | "area"
  | "bar"
  | "stacked_bar"
  | "distribution"
  | "boxplot"
  | "heatmap"
  | "map"
  | "timeline"
  | "table";

export type VisualizationKind =
  | "kpi"
  | "cartesian"
  | "forecast"
  | "distribution"
  | "heatmap"
  | "map"
  | "timeline"
  | "table"
  | "omitted";

export interface QueryFilters {
  port?: string | null;
  date_from?: string | null;
  date_to?: string | null;
  vessel_type?: string | null;
  vessel_name?: string | null;
  mmsi?: string | null;
  imo?: string | null;
  anomaly?: boolean | null;
}

/**
 * The public request accepted by POST /api/v2/query.
 *
 * The optional fields have backend defaults, so callers may omit them. The
 * operations workspace supplies all four fields to keep each analysis
 * reproducible.
 */
export interface QueryRequestPayload {
  question: string;
  conversation_id?: string | null;
  top_k_evidence?: number;
  filters?: QueryFilters;
}

/** @deprecated Prefer QueryRequestPayload for the public API request body. */
export type QueryRequest = QueryRequestPayload;

export interface DateScope {
  date_from?: string | null;
  date_to?: string | null;
  target_date?: string | null;
  relative_window?: string | null;
  is_current: boolean;
}

export interface RoutePair {
  origin: string;
  destination: string;
}

export interface QueryPlan {
  mode: QueryMode;
  operation: QueryOperation;
  metric?: string | null;
  dimensions: string[];
  ports: string[];
  country_codes: string[];
  origin_port?: string | null;
  destination_port?: string | null;
  route_pairs: RoutePair[];
  date_scope: DateScope;
  vessel_type?: string | null;
  vessel_name?: string | null;
  mmsi?: string | null;
  imo?: string | null;
  call_id?: string | null;
  aggregation?: string | null;
  day_of_week?: string | null;
  compare_day_of_week?: string | null;
  horizon_weeks: number;
  horizon_hours?: number | null;
  limit: number;
  eta_watch_intent?: ETAWatchIntent | null;
  speed_threshold_kn?: number | null;
  eta_change_threshold_minutes?: number | null;
  change_window_minutes?: number | null;
  include_stale?: boolean;
  source_scope?: string | null;
  carbon_boundary: string;
  pollutants: string[];
  requested_visual: VisualizationIntent;
  ambiguities: string[];
  clarification?: string | null;
  reason: string;
  context_inherited: string[];
  planner_source: "deterministic" | "openai_structured" | "context";
  planner_model?: string | null;
}

export type DataRow = Record<string, unknown>;

export interface ColumnSpec {
  field: string;
  label: string;
  data_type: "string" | "number" | "integer" | "boolean" | "datetime";
  unit?: string | null;
}

export interface Dataset {
  id: string;
  columns: ColumnSpec[];
  rows: DataRow[];
  row_count: number;
}

export interface VisualizationCommon {
  id: string;
  title: string;
  dataset_id?: string | null;
  row_id_field?: string | null;
  accessible_summary: string;
  table_fallback_dataset_id?: string | null;
  citations: string[];
}

export interface KPIThresholdSpec {
  id: string;
  label: string;
  value: number;
  unit?: string | null;
}

export interface KPIVisualization extends VisualizationCommon {
  kind: "kpi";
  value_field: string;
  label: string;
  unit?: string | null;
  trend_field?: string | null;
  comparison_field?: string | null;
  baseline_value?: number | null;
  thresholds?: KPIThresholdSpec[];
}

export interface ReferenceLineSpec {
  id: string;
  label: string;
  axis: "x" | "y";
  value: string | number;
  unit?: string | null;
  line_style: "solid" | "dashed" | "dotted";
}

export interface IntervalBandSpec {
  id: string;
  label: string;
  lower_field: string;
  upper_field: string;
  point_field?: string | null;
  display?: "band" | "whisker";
  unit?: string | null;
}

export interface FlaggedPointAnnotation {
  id: string;
  label: string;
  condition_field: string;
  x_field: string;
  y_field: string;
}

export interface FittedSeriesBinding {
  id: string;
  label: string;
  x_field: string;
  y_field: string;
  method: "ols" | "rolling_median";
  association_only: boolean;
  slope?: number | null;
  intercept?: number | null;
  r_squared?: number | null;
}

export interface CartesianVisualization extends VisualizationCommon {
  kind: "cartesian";
  chart_type: "line" | "area" | "bar" | "grouped_bar" | "stacked_bar" | "scatter";
  x_field: string;
  y_fields: string[];
  series_field?: string | null;
  orientation: "vertical" | "horizontal";
  sort: "none" | "ascending" | "descending" | "calendar";
  x_unit?: string | null;
  y_unit?: string | null;
  stacked: boolean;
  reference_lines?: ReferenceLineSpec[];
  interval_bands?: IntervalBandSpec[];
  annotations?: FlaggedPointAnnotation[];
  fitted_series?: FittedSeriesBinding[];
  highlight?: {
    condition_field: string;
    value_field: string;
    label: string;
  } | null;
}

export interface ForecastVisualization extends VisualizationCommon {
  kind: "forecast";
  date_field: string;
  predicted_field: string;
  lower_field: string;
  upper_field: string;
  actual_field?: string | null;
  unit?: string | null;
  interval_level?: number;
  forecast_boundary?: string | null;
  quality_metrics?: {
    mase: number;
    interval_coverage: number;
    interval_level: number;
    gate_passed: true;
  } | null;
}

export interface FiveNumberSummary {
  minimum: number;
  q1: number;
  median: number;
  q3: number;
  maximum: number;
  lower_whisker: number;
  upper_whisker: number;
  p90?: number | null;
  count: number;
}

export interface DistributionVisualization extends VisualizationCommon {
  kind: "distribution";
  chart_type: "histogram" | "boxplot" | "percentile";
  value_field: string;
  count_field?: string | null;
  category_field?: string | null;
  bins?: number | null;
  unit?: string | null;
  bin_lower_field?: string | null;
  bin_upper_field?: string | null;
  summary_dataset_id?: string | null;
  five_number_summary?: FiveNumberSummary | null;
  outlier_dataset_id?: string | null;
  outlier_condition_field?: string | null;
  outlier_value_field?: string | null;
}

export interface HeatmapVisualization extends VisualizationCommon {
  kind: "heatmap";
  x_field: string;
  y_field: string;
  value_field: string;
  unit?: string | null;
}

export interface MapVisualization extends VisualizationCommon {
  kind: "map";
  latitude_field: string;
  longitude_field: string;
  label_field?: string | null;
  value_field?: string | null;
  geometry_mode?: "points" | "segments" | "ordered_path";
  start_latitude_field?: string | null;
  start_longitude_field?: string | null;
  end_latitude_field?: string | null;
  end_longitude_field?: string | null;
  path_field?: string | null;
  sequence_field?: string | null;
  timestamp_field?: string | null;
}

export interface TimelineVisualization extends VisualizationCommon {
  kind: "timeline";
  time_field: string;
  label_field: string;
  detail_fields: string[];
  end_time_field?: string | null;
  lane_field?: string | null;
}

export interface TableVisualization extends VisualizationCommon {
  kind: "table";
  visible_fields: string[];
}

export type VisualizationOmissionReason =
  | "not_requested"
  | "insufficient_data"
  | "not_meaningful"
  | "unsupported_visual"
  | "validation_failed"
  | "stale_data"
  | "coverage_unavailable"
  | "missing_route_dataset";

export interface OmittedVisualization extends VisualizationCommon {
  kind: "omitted";
  reason_code: VisualizationOmissionReason;
  reason: string;
}

export type VisualizationSpec =
  | KPIVisualization
  | CartesianVisualization
  | ForecastVisualization
  | DistributionVisualization
  | HeatmapVisualization
  | MapVisualization
  | TimelineVisualization
  | TableVisualization
  | OmittedVisualization;

export interface EvidenceItem {
  id: string;
  source_type: "computed" | "traffic_event" | "local_document" | "web" | "system";
  title: string;
  excerpt?: string | null;
  url?: string | null;
  metadata: Record<string, unknown>;
}

export interface AppliedScope {
  ports: string[];
  country_codes: string[];
  origin_port?: string | null;
  destination_port?: string | null;
  date_from?: string | null;
  date_to?: string | null;
  target_date?: string | null;
  vessel_type?: string | null;
  vessel_name?: string | null;
  mmsi?: string | null;
  imo?: string | null;
  horizon_hours?: number | null;
  source_scope?: string | null;
}

export interface AssuranceAssessment {
  status: "verified" | "unavailable" | "not_applicable";
  level: "high" | "not_applicable";
  basis:
    | "direct_computation"
    | "official_live_source"
    | "validated_model"
    | "source_grounded_research"
    | "system_response";
  reason: string;
  checks: string[];
}

export interface AvailabilityInfo {
  code:
    | "available"
    | "no_data"
    | "source_unavailable"
    | "source_stale"
    | "coverage_unavailable"
    | "ambiguous_match"
    | "not_applicable";
  provider?: string | null;
  retryable?: boolean;
}

export interface FreshnessInfo {
  data_from?: string | null;
  data_to?: string | null;
  as_of?: string | null;
  historical: boolean;
  message: string;
}

export interface TraceInfo {
  trace_id: string;
  route: string;
  operation: string;
  planner_source: string;
  planner_model?: string | null;
  model: string;
  reasoning_effort?: string | null;
  sources: string[];
  retrieval_mode: string;
  retrieval_backend?: string | null;
  retrieval_status: string;
  retrieval_top_k: number;
  result_state: string;
  failure_state?: string | null;
  result_hash: string;
  data_manifest_version?: string | null;
  dataset_rows: number;
  visualization_decision: string;
  visualization_contract_version?: "2.1";
  chart_profile?: string[];
  visualization_dataset_ids?: string[];
  visualization_fallback_reasons?: string[];
  latency_ms: number;
  warnings: string[];
}

export interface FactSlot {
  name: string;
  value: unknown;
  unit?: string | null;
  entity?: string | null;
  source: "computed" | "retrieved" | "model" | "system";
  immutable: boolean;
}

export interface ChartInsight {
  id: string;
  visualization_id: string;
  insight_type:
    | "peak"
    | "period_median"
    | "trend"
    | "baseline_deviation"
    | "boundary_delta"
    | "distribution_summary"
    | "association"
    | "forecast_quality"
    | "threshold_exceedance"
    | "ranking_margin"
    | "dominant_share"
    | "movement_anomaly";
  statement: string;
  fact_names: string[];
  evidence_ids: string[];
}

export type ETAWatchIntent =
  | "shift_handover"
  | "inbound_watchlist"
  | "low_speed_exceptions"
  | "destination_load"
  | "eta_revisions"
  | "signal_quality"
  | "vessel_status";

export type OperationalPriority = "attention" | "monitor" | "information";

export type OperationalStatus =
  | "due_soon"
  | "low_speed"
  | "eta_changed"
  | "stale_position"
  | "missing_eta"
  | "observed";

export type OperationalAction =
  | "locate_vessel"
  | "watch_next_six_hours"
  | "inspect_eta_changes";

export interface OperationalBriefItem {
  row_id: string;
  vessel_label?: string | null;
  priority: OperationalPriority;
  status: OperationalStatus;
  reason: string;
  actions: OperationalAction[];
}

export interface OperationalExceptionSummary {
  code: Exclude<OperationalStatus, "observed">;
  count: number;
  summary: string;
}

export interface OperationalBrief {
  intent: ETAWatchIntent;
  headline: string;
  window_start_utc?: string | null;
  window_end_utc?: string | null;
  matched_count: number;
  displayed_count: number;
  prioritized_items: OperationalBriefItem[];
  exceptions: OperationalExceptionSummary[];
  source_health: "connecting" | "warming" | "live" | "stale" | "unavailable";
  source_observed_at?: string | null;
  coverage: string;
}

export interface AnswerEnvelope {
  api_version: "2.0";
  visualization_contract_version?: "2.0" | "2.1";
  conversation_id: string;
  turn_id: string;
  question: string;
  mode: QueryMode;
  state: ResultState;
  answer: string;
  plan: QueryPlan;
  facts: FactSlot[];
  applied_scope: AppliedScope;
  datasets: Dataset[];
  visualizations: VisualizationSpec[];
  chart_insights?: ChartInsight[];
  operational_brief?: OperationalBrief | null;
  evidence: EvidenceItem[];
  freshness: FreshnessInfo;
  confidence: "high" | "medium" | "low" | "not_applicable";
  assurance?: AssuranceAssessment | null;
  availability?: AvailabilityInfo | null;
  caveats: string[];
  trace: TraceInfo;
}

export interface DataManifest extends Record<string, unknown> {
  schema_version?: string;
  built_at_utc?: string;
  available_ports?: string[];
  enabled_operations?: string[];
  row_counts?: Record<string, number>;
  tables?: Record<string, Record<string, unknown>>;
  coverage_dates?: Record<string, unknown>;
  model_validation?: Record<string, unknown>;
  source_hashes?: Record<string, string>;
}

export interface LiveETACapabilities {
  provider?: string;
  available: boolean;
  api_key_configured?: boolean;
  /**
   * AISStream subscription rectangles in provider order:
   * [[south latitude, west longitude], [north latitude, east longitude]].
   */
  bounding_boxes?: Array<[[number, number], [number, number]]>;
  country_scope?: string[];
  default_scope?: string;
  source_health?: "connecting" | "warming" | "live" | "stale" | "unavailable";
  maximum_horizon_hours?: number;
  operational_intents?: ETAWatchIntent[];
  eta_watch_intents?: ETAWatchIntent[];
  source_kind?: "vessel_reported_ais";
  history_hours?: number;
  official_schedule?: boolean;
  official_schedule_country_scope?: string[];
  ais_destination_country_scope?: string[];
  timezone?: string;
  maximum_horizon_days?: number;
  operations?: QueryOperation[];
  official_eta_authority?: string;
  regional_ais_scope?: string;
  announced_variance?: string;
  prediction?: boolean;
  reason?: string | null;
}

export interface ModelRoutes {
  planner: string;
  general: string;
  research: string;
  reasoning_effort: string;
  enabled: boolean;
}

export interface RetrievalCapabilities {
  available: boolean;
  backend?: string | null;
  reason?: string | null;
  default_top_k: number;
  recommended_top_k: number;
  disable_value: number;
  local_synthesis: {
    available: boolean;
    provider?: string | null;
    model?: string | null;
    scope: string;
    analytics_fact_rewriting: boolean;
  };
}

export interface CapabilityRegistry {
  product_label?: string;
  historical_analytics?: string[];
  research?: string;
  general_assistance?: string;
  not_supported?: string[];
  [key: string]: unknown;
}

export interface CapabilityResponse {
  api_version: "2.0";
  visualization_contract_version?: "2.1";
  modes: QueryMode[];
  operations: QueryOperation[];
  visualization_kinds: VisualizationKind[];
  freshness: FreshnessInfo;
  data_manifest: DataManifest;
  kpi_capabilities: Record<string, unknown>;
  model_routes: ModelRoutes;
  retrieval: RetrievalCapabilities;
  live_eta?: LiveETACapabilities;
  capability_registry: CapabilityRegistry;
  conversation_store: string;
}

export interface ExportRequest {
  conversation_id: string;
  turn_id: string;
  dataset_id: string;
  format: "csv" | "json";
}

export interface ExportResponse {
  export_id: string;
  format: "csv" | "json";
  dataset_id: string;
  row_count: number;
  path: string;
}

export interface FeedbackResponse {
  feedback_id: string;
  status: "accepted";
}

export type WorkspaceRoute =
  | "/overview"
  | "/analysis"
  | "/traffic-monitoring"
  | "/vessel-investigation"
  | "/eta-delay"
  | "/port-pressure"
  | "/carbon-emissions";

/**
 * New route/conversation fields are optional so v1 local-history records can
 * be migrated without deleting or invalidating a user's saved analyses.
 */
export interface AnalysisHistoryItem {
  id: string;
  question: string;
  createdAt: string;
  result: AnswerEnvelope;
  route?: WorkspaceRoute;
  conversationId?: string;
  /** Exact body submitted to POST /api/v2/query for reproducible restoration. */
  request?: QueryRequestPayload;
  schemaVersion?: 1 | 2 | 3;
}
