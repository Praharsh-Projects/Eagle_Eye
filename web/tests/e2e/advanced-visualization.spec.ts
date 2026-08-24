import { readFileSync } from "node:fs";
import { expect, test, type Page, type Route } from "@playwright/test";
import type {
  AnswerEnvelope,
  CapabilityResponse,
  Dataset,
  IntervalBandSpec,
  QueryRequestPayload,
  VisualizationSpec,
} from "../../src/types";

type AdvancedIntervalBandSpec = IntervalBandSpec & {
  point_field?: string | null;
  display?: "band" | "whisker";
};

type AdvancedRenderObservation = {
  visualization_id: string;
  chart_profile: string;
  semantic_profile?: string;
  visualization_contract_version: "2.0" | "2.1";
  render_latency_ms: number | null;
  fallback_reason: string | null;
  interaction_mode: string;
  source_dataset_ids: string[];
  analytical_series_names?: string[];
  analytical_series_types?: string[];
  rendered_point_count?: number;
  interval_band_count?: number;
  reference_line_count?: number;
  annotation_count?: number;
  fitted_series_count?: number;
  summary_marker_count?: number;
  quality_annotation_count?: number;
  null_value_count?: number;
  preserved_null_count?: number;
  null_to_zero_count?: number;
  inspected_series_name?: string;
  render_diagnostics?: Record<string, unknown>;
  encoding_diagnostics?: Record<string, unknown>;
  inspection?: Record<string, unknown>;
};

const rendererSource = readFileSync(
  new URL("../../src/components/Visualization.tsx", import.meta.url),
  "utf8",
);
const stylesSource = readFileSync(
  new URL("../../src/styles.css", import.meta.url),
  "utf8",
);

const carbonDataset: Dataset = {
  id: "carbon-uncertainty-series",
  columns: [
    { field: "row_id", label: "Row ID", data_type: "string" },
    { field: "date", label: "Date", data_type: "datetime" },
    {
      field: "co2e_ttw",
      label: "CO2e TTW",
      data_type: "number",
      unit: "tCO2e",
    },
    {
      field: "co2e_lower",
      label: "Lower bound",
      data_type: "number",
      unit: "tCO2e",
    },
    {
      field: "co2e_upper",
      label: "Upper bound",
      data_type: "number",
      unit: "tCO2e",
    },
  ],
  rows: [
    {
      row_id: "carbon-01",
      date: "2022-03-01",
      co2e_ttw: 12.1,
      co2e_lower: 10.8,
      co2e_upper: 13.6,
    },
    {
      row_id: "carbon-02",
      date: "2022-03-02",
      co2e_ttw: 13.4,
      co2e_lower: 12.2,
      co2e_upper: 14.9,
    },
    {
      row_id: "carbon-03",
      date: "2022-03-03",
      co2e_ttw: 11.8,
      co2e_lower: 10.5,
      co2e_upper: 13.1,
    },
  ],
  row_count: 3,
};

const kpiWithoutBulletDataset: Dataset = {
  id: "kpi-without-bullet-markers",
  columns: [
    {
      field: "arrivals",
      label: "Arrivals",
      data_type: "integer",
      unit: "vessels",
    },
  ],
  rows: [{ arrivals: 54 }],
  row_count: 1,
};

const kpiWithoutBulletVisualization: VisualizationSpec = {
  id: "kpi-without-bullet-markers",
  kind: "kpi",
  title: "KPI value without bullet markers",
  dataset_id: kpiWithoutBulletDataset.id,
  table_fallback_dataset_id: kpiWithoutBulletDataset.id,
  accessible_summary: "A validated total of 54 vessel arrivals.",
  citations: [],
  value_field: "arrivals",
  label: "Arrivals",
  unit: "vessels",
  thresholds: [],
};

const rankingDataset: Dataset = {
  id: "port-ranking-series",
  columns: [
    { field: "row_id", label: "Row ID", data_type: "string" },
    { field: "port", label: "Port", data_type: "string" },
    {
      field: "arrivals",
      label: "Arrivals",
      data_type: "integer",
      unit: "vessels",
    },
  ],
  rows: [
    { row_id: "rank-01", port: "Gothenburg", arrivals: 124 },
    { row_id: "rank-02", port: "Karlshamn", arrivals: 88 },
    { row_id: "rank-03", port: "Ventspils", arrivals: 61 },
  ],
  row_count: 3,
};

const dwellBins: Dataset = {
  id: "dwell-histogram-bins",
  columns: [
    { field: "row_id", label: "Row ID", data_type: "string" },
    {
      field: "bin_midpoint",
      label: "Bin midpoint",
      data_type: "number",
      unit: "minutes",
    },
    {
      field: "bin_lower",
      label: "Bin lower",
      data_type: "number",
      unit: "minutes",
    },
    {
      field: "bin_upper",
      label: "Bin upper",
      data_type: "number",
      unit: "minutes",
    },
    {
      field: "count",
      label: "Observations",
      data_type: "integer",
      unit: "port calls",
    },
  ],
  rows: [
    {
      row_id: "bin-01",
      bin_midpoint: 30,
      bin_lower: 0,
      bin_upper: 60,
      count: 8,
    },
    {
      row_id: "bin-02",
      bin_midpoint: 90,
      bin_lower: 60,
      bin_upper: 120,
      count: 14,
    },
    {
      row_id: "bin-03",
      bin_midpoint: 150,
      bin_lower: 120,
      bin_upper: 180,
      count: 6,
    },
  ],
  row_count: 3,
};

const dwellSummary: Dataset = {
  id: "dwell-five-number-summary",
  columns: [
    { field: "minimum", label: "Minimum", data_type: "number", unit: "minutes" },
    { field: "q1", label: "Q1", data_type: "number", unit: "minutes" },
    { field: "median", label: "Median", data_type: "number", unit: "minutes" },
    { field: "q3", label: "Q3", data_type: "number", unit: "minutes" },
    { field: "maximum", label: "Maximum", data_type: "number", unit: "minutes" },
    {
      field: "lower_whisker",
      label: "Lower whisker",
      data_type: "number",
      unit: "minutes",
    },
    {
      field: "upper_whisker",
      label: "Upper whisker",
      data_type: "number",
      unit: "minutes",
    },
    { field: "p90", label: "P90", data_type: "number", unit: "minutes" },
    { field: "count", label: "Count", data_type: "integer" },
  ],
  rows: [
    {
      minimum: 12,
      q1: 48,
      median: 82,
      q3: 126,
      maximum: 240,
      lower_whisker: 12,
      upper_whisker: 210,
      p90: 171,
      count: 28,
    },
  ],
  row_count: 1,
};

const dwellOutliers: Dataset = {
  id: "dwell-outliers",
  columns: [
    { field: "row_id", label: "Row ID", data_type: "string" },
    {
      field: "dwell_minutes",
      label: "Dwell",
      data_type: "number",
      unit: "minutes",
    },
  ],
  rows: [{ row_id: "outlier-01", dwell_minutes: 240 }],
  row_count: 1,
};

const forecastDataset: Dataset = {
  id: "pressure-forecast-series",
  columns: [
    { field: "row_id", label: "Row ID", data_type: "string" },
    { field: "date", label: "Date", data_type: "datetime" },
    {
      field: "actual",
      label: "Actual pressure",
      data_type: "number",
      unit: "index",
    },
    {
      field: "predicted",
      label: "Predicted pressure",
      data_type: "number",
      unit: "index",
    },
    {
      field: "lower",
      label: "Lower 80%",
      data_type: "number",
      unit: "index",
    },
    {
      field: "upper",
      label: "Upper 80%",
      data_type: "number",
      unit: "index",
    },
  ],
  rows: [
    {
      row_id: "forecast-01",
      date: "2022-04-24",
      actual: 1.02,
      predicted: 1.01,
      lower: 0.87,
      upper: 1.15,
    },
    {
      row_id: "forecast-02",
      date: "2022-05-01",
      actual: null,
      predicted: 1.12,
      lower: 0.94,
      upper: 1.31,
    },
    {
      row_id: "forecast-03",
      date: "2022-05-08",
      actual: null,
      predicted: 1.18,
      lower: 0.97,
      upper: 1.39,
    },
  ],
  row_count: 3,
};

const correlationDataset: Dataset = {
  id: "arrivals-pressure-correlation",
  columns: [
    { field: "row_id", label: "Row ID", data_type: "string" },
    {
      field: "arrivals",
      label: "Arrivals",
      data_type: "integer",
      unit: "vessels",
    },
    {
      field: "pressure",
      label: "Pressure",
      data_type: "number",
      unit: "index",
    },
    {
      field: "ols_fitted_pressure",
      label: "OLS fitted pressure",
      data_type: "number",
      unit: "index",
    },
  ],
  rows: [
    {
      row_id: "correlation-01",
      arrivals: 16,
      pressure: 0.86,
      ols_fitted_pressure: 0.9,
    },
    {
      row_id: "correlation-02",
      arrivals: 25,
      pressure: 1.12,
      ols_fitted_pressure: 1.08,
    },
    {
      row_id: "correlation-03",
      arrivals: 34,
      pressure: 1.24,
      ols_fitted_pressure: 1.26,
    },
  ],
  row_count: 3,
};

const heatmapDataset: Dataset = {
  id: "weekday-hour-pattern",
  columns: [
    { field: "row_id", label: "Row ID", data_type: "string" },
    { field: "weekday", label: "Weekday", data_type: "string" },
    { field: "hour", label: "Hour", data_type: "string" },
    {
      field: "arrivals",
      label: "Arrivals",
      data_type: "integer",
      unit: "vessels",
    },
  ],
  rows: [
    { row_id: "heat-01", weekday: "Monday", hour: "00:00", arrivals: 2 },
    { row_id: "heat-02", weekday: "Monday", hour: "06:00", arrivals: null },
    { row_id: "heat-03", weekday: "Friday", hour: "12:00", arrivals: 5 },
  ],
  row_count: 3,
};

const visualizations: VisualizationSpec[] = [
  {
    id: "carbon-uncertainty-trend",
    kind: "cartesian",
    chart_type: "line",
    title: "Carbon estimate with uncertainty",
    dataset_id: carbonDataset.id,
    table_fallback_dataset_id: carbonDataset.id,
    row_id_field: "row_id",
    accessible_summary:
      "Validated tank-to-wake carbon estimates with lower and upper uncertainty bounds.",
    citations: ["evidence-advanced"],
    x_field: "date",
    y_fields: ["co2e_ttw"],
    orientation: "vertical",
    sort: "calendar",
    y_unit: "tCO2e",
    stacked: false,
    interval_bands: [
      {
        id: "carbon-uncertainty",
        label: "Validated uncertainty",
        lower_field: "co2e_lower",
        upper_field: "co2e_upper",
        point_field: "co2e_ttw",
        display: "whisker",
        unit: "tCO2e",
      },
    ] satisfies AdvancedIntervalBandSpec[],
    reference_lines: [
      {
        id: "carbon-baseline",
        label: "Period baseline",
        axis: "y",
        value: 12.1,
        unit: "tCO2e",
        line_style: "dashed",
      },
    ],
  },
  {
    id: "port-arrivals-ranking",
    kind: "cartesian",
    chart_type: "bar",
    title: "Port arrivals ranking",
    dataset_id: rankingDataset.id,
    table_fallback_dataset_id: rankingDataset.id,
    row_id_field: "row_id",
    accessible_summary:
      "Three ports ranked from highest to lowest validated arrival count.",
    citations: ["evidence-advanced"],
    x_field: "port",
    y_fields: ["arrivals"],
    orientation: "horizontal",
    sort: "descending",
    y_unit: "vessels",
    stacked: false,
  },
  {
    id: "dwell-histogram-summary",
    kind: "distribution",
    chart_type: "histogram",
    title: "Dwell distribution and percentile summary",
    dataset_id: dwellBins.id,
    table_fallback_dataset_id: dwellBins.id,
    row_id_field: "row_id",
    accessible_summary:
      "Explicit dwell-time bins with a supplied five-number summary and one outlier.",
    citations: ["evidence-advanced"],
    value_field: "bin_midpoint",
    count_field: "count",
    unit: "minutes",
    bin_lower_field: "bin_lower",
    bin_upper_field: "bin_upper",
    summary_dataset_id: dwellSummary.id,
    outlier_dataset_id: dwellOutliers.id,
    outlier_value_field: "dwell_minutes",
    five_number_summary: {
      minimum: 12,
      q1: 48,
      median: 82,
      q3: 126,
      maximum: 240,
      lower_whisker: 12,
      upper_whisker: 210,
      p90: 171,
      count: 28,
    },
  },
  {
    id: "pressure-forecast-fan",
    kind: "forecast",
    title: "Pressure forecast with validated 80% interval",
    dataset_id: forecastDataset.id,
    table_fallback_dataset_id: forecastDataset.id,
    row_id_field: "row_id",
    accessible_summary:
      "Actual and predicted pressure with a validated eighty percent forecast interval.",
    citations: ["evidence-advanced"],
    date_field: "date",
    actual_field: "actual",
    predicted_field: "predicted",
    lower_field: "lower",
    upper_field: "upper",
    unit: "index",
    interval_level: 0.8,
    forecast_boundary: "2022-05-01",
    quality_metrics: {
      mase: 0.74,
      interval_coverage: 0.81,
      interval_level: 0.8,
      gate_passed: true,
    },
  },
  {
    id: "arrivals-pressure-fit",
    kind: "cartesian",
    chart_type: "scatter",
    title: "Arrivals and pressure association",
    dataset_id: correlationDataset.id,
    table_fallback_dataset_id: correlationDataset.id,
    row_id_field: "row_id",
    accessible_summary:
      "Validated arrival and pressure observations with a backend-produced OLS fit; association is not causation.",
    citations: ["evidence-advanced"],
    x_field: "arrivals",
    y_fields: ["pressure"],
    orientation: "vertical",
    sort: "ascending",
    x_unit: "vessels",
    y_unit: "index",
    stacked: false,
    fitted_series: [
      {
        id: "ols-pressure-fit",
        label: "OLS fit · association only",
        x_field: "arrivals",
        y_field: "ols_fitted_pressure",
        method: "ols",
        association_only: true,
        slope: 0.02,
        intercept: 0.58,
        r_squared: 0.89,
      },
    ],
  },
  {
    id: "weekday-hour-heatmap",
    kind: "heatmap",
    title: "Weekday and hour arrival pattern",
    dataset_id: heatmapDataset.id,
    table_fallback_dataset_id: heatmapDataset.id,
    row_id_field: "row_id",
    accessible_summary:
      "Calendar-ordered weekday and hour observations; one cell has no reported value.",
    citations: ["evidence-advanced"],
    x_field: "hour",
    y_field: "weekday",
    value_field: "arrivals",
    unit: "vessels",
  },
];

const datasets = [
  carbonDataset,
  rankingDataset,
  dwellBins,
  dwellSummary,
  dwellOutliers,
  forecastDataset,
  correlationDataset,
  heatmapDataset,
];

function legacyCarbonDataset(validUnits: boolean): Dataset {
  return {
    ...carbonDataset,
    id: validUnits
      ? "legacy-carbon-uncertainty-series"
      : "legacy-carbon-invalid-uncertainty-series",
    columns: carbonDataset.columns.map((column) => {
      if (column.field === "co2e_lower") {
        return { ...column, field: "co2e_ttw_lower" };
      }
      if (column.field === "co2e_upper") {
        return {
          ...column,
          field: "co2e_ttw_upper",
          unit: validUnits ? column.unit : "kg",
        };
      }
      return { ...column };
    }),
    rows: carbonDataset.rows.map((row) => {
      const {
        co2e_lower: lower,
        co2e_upper: upper,
        ...remaining
      } = row;
      return {
        ...remaining,
        co2e_ttw_lower: lower,
        co2e_ttw_upper: upper,
      };
    }),
  };
}

function legacyCarbonVisualization(datasetId: string): VisualizationSpec {
  return {
    id: "legacy-carbon-trend",
    kind: "cartesian",
    chart_type: "line",
    title: "Legacy carbon estimate",
    dataset_id: datasetId,
    table_fallback_dataset_id: datasetId,
    row_id_field: "row_id",
    accessible_summary:
      "Legacy carbon estimates backed by validated central, lower, and upper fields.",
    citations: ["evidence-legacy-carbon"],
    x_field: "date",
    y_fields: ["co2e_ttw"],
    orientation: "vertical",
    sort: "calendar",
    y_unit: "tCO2e",
    stacked: false,
  };
}

const capabilities: CapabilityResponse = {
  api_version: "2.0",
  visualization_contract_version: "2.1",
  modes: [
    "analytics",
    "maritime_research",
    "general_chat",
    "app_help",
    "clarification",
    "unsupported",
  ],
  operations: [
    "carbon",
    "top_ports",
    "dwell_distribution",
    "forecast_congestion",
    "correlation",
    "arrival_pattern",
  ],
  visualization_kinds: [
    "cartesian",
    "distribution",
    "forecast",
    "heatmap",
    "table",
    "omitted",
  ],
  freshness: {
    data_from: "2021-01-01",
    data_to: "2022-04-30",
    historical: true,
    message: "Historical coverage through 2022-04-30",
  },
  data_manifest: {
    schema_version: "2.1-advanced-test",
    built_at_utc: "2026-07-27T10:00:00Z",
    available_ports: ["SEGOT", "SEKAN", "LVVNT"],
    enabled_operations: [
      "carbon",
      "top_ports",
      "dwell_distribution",
      "forecast_congestion",
      "correlation",
      "arrival_pattern",
    ],
    row_counts: Object.fromEntries(
      datasets.map((dataset) => [dataset.id, dataset.row_count]),
    ),
  },
  kpi_capabilities: {},
  model_routes: {
    planner: "deterministic-test",
    general: "disabled",
    research: "disabled",
    reasoning_effort: "none",
    enabled: false,
  },
  retrieval: {
    available: true,
    backend: "test",
    default_top_k: 5,
    recommended_top_k: 5,
    disable_value: 0,
    local_synthesis: {
      available: false,
      scope: "test",
      analytics_fact_rewriting: false,
    },
  },
  capability_registry: {
    product_label: "Eagle Eye",
    historical_analytics: capabilitiesOperations(),
  },
  conversation_store: "sqlite",
};

function capabilitiesOperations(): string[] {
  return [
    "carbon",
    "top_ports",
    "dwell_distribution",
    "forecast_congestion",
    "correlation",
    "arrival_pattern",
  ];
}

function envelope(
  question: string,
  conversationId: string,
): AnswerEnvelope {
  return {
    api_version: "2.0",
    visualization_contract_version: "2.1",
    conversation_id: conversationId,
    turn_id: "turn-advanced-visual-profiles",
    question,
    mode: "analytics",
    state: "COMPUTED",
    answer:
      "Six validated analytical views are shown without changing the canonical answer.",
    plan: {
      mode: "analytics",
      operation: "carbon",
      metric: "validated_profiles",
      dimensions: ["date", "port", "weekday", "hour"],
      ports: ["SEGOT", "SEKAN", "LVVNT"],
      origin_port: null,
      destination_port: null,
      route_pairs: [],
      date_scope: {
        date_from: "2022-03-01",
        date_to: "2022-05-08",
        target_date: null,
        relative_window: null,
        is_current: false,
      },
      vessel_type: null,
      mmsi: null,
      imo: null,
      call_id: null,
      aggregation: "day",
      day_of_week: null,
      compare_day_of_week: null,
      horizon_weeks: 2,
      limit: 100,
      source_scope: "historical",
      carbon_boundary: "TTW",
      pollutants: ["CO2e"],
      requested_visual: "auto",
      ambiguities: [],
      clarification: null,
      reason: "Frozen advanced-visualization browser fixture.",
      context_inherited: [],
      planner_source: "deterministic",
      planner_model: null,
    },
    facts: [
      {
        name: "profile_count",
        value: visualizations.length,
        unit: "visualizations",
        source: "computed",
        immutable: true,
      },
    ],
    applied_scope: {
      ports: ["SEGOT", "SEKAN", "LVVNT"],
      date_from: "2022-03-01",
      date_to: "2022-05-08",
      source_scope: "historical",
    },
    datasets,
    visualizations,
    evidence: [
      {
        id: "evidence-advanced",
        source_type: "computed",
        title: "Frozen advanced visualization fixture",
        excerpt:
          "All values and enriched bindings are supplied by deterministic test data.",
        metadata: {
          dataset_ids: datasets.map((dataset) => dataset.id),
        },
      },
    ],
    freshness: {
      data_from: "2021-01-01",
      data_to: "2022-04-30",
      historical: true,
      message: "Historical coverage through 2022-04-30",
    },
    confidence: "high",
    caveats: [
      "The OLS series represents association, not causation.",
      "The null heatmap cell must remain unavailable rather than become zero.",
    ],
    trace: {
      trace_id: "trace-advanced-visual-profiles",
      route: "analytics",
      operation: "carbon",
      planner_source: "deterministic",
      planner_model: null,
      model: "deterministic",
      reasoning_effort: "none",
      sources: datasets.map((dataset) => dataset.id),
      retrieval_mode: "none",
      retrieval_backend: "test",
      retrieval_status: "not_applicable",
      retrieval_top_k: 5,
      result_state: "COMPUTED",
      failure_state: null,
      result_hash: "result-hash-advanced-visual-profiles",
      data_manifest_version: "2.1-advanced-test",
      dataset_rows: datasets.reduce(
        (total, dataset) => total + dataset.row_count,
        0,
      ),
      visualization_decision: "advanced-profile-suite",
      visualization_contract_version: "2.1",
      chart_profile: [
        "cartesian:line",
        "cartesian:bar",
        "distribution:histogram",
        "forecast",
        "cartesian:scatter",
        "heatmap",
      ],
      visualization_dataset_ids: datasets.map((dataset) => dataset.id),
      visualization_fallback_reasons: [],
      latency_ms: 18,
      warnings: [],
    },
  };
}

function legacyCarbonEnvelope(
  question: string,
  conversationId: string,
  validUnits: boolean,
): AnswerEnvelope {
  const base = envelope(question, conversationId);
  const dataset = legacyCarbonDataset(validUnits);
  const visualization = legacyCarbonVisualization(dataset.id);
  return {
    ...base,
    visualization_contract_version: "2.0",
    turn_id: validUnits
      ? "turn-legacy-carbon-enriched"
      : "turn-legacy-carbon-retained",
    answer: "Legacy carbon answer remains byte-for-byte unchanged.",
    facts: [
      {
        name: "carbon_observation_count",
        value: dataset.row_count,
        unit: "days",
        entity: "SEGOT",
        source: "computed",
        immutable: true,
      },
    ],
    datasets: [dataset],
    visualizations: [visualization],
    chart_insights: undefined,
    evidence: [
      {
        id: "evidence-legacy-carbon",
        source_type: "computed",
        title: "Validated legacy carbon dataset",
        excerpt:
          "The canonical 2.0 result supplies central, lower, and upper carbon fields.",
        metadata: { dataset_id: dataset.id },
      },
    ],
    trace: {
      ...base.trace,
      trace_id: validUnits
        ? "trace-legacy-carbon-enriched"
        : "trace-legacy-carbon-retained",
      result_hash: validUnits
        ? "result-hash-legacy-carbon-enriched"
        : "result-hash-legacy-carbon-retained",
      sources: [dataset.id],
      dataset_rows: dataset.row_count,
      visualization_decision: "cartesian:line",
      visualization_contract_version: undefined,
      chart_profile: ["cartesian:line"],
      visualization_dataset_ids: [dataset.id],
      visualization_fallback_reasons: [],
    },
  };
}

function kpiWithoutBulletEnvelope(
  question: string,
  conversationId: string,
): AnswerEnvelope {
  const base = envelope(question, conversationId);
  return {
    ...base,
    turn_id: "turn-kpi-without-bullet-markers",
    answer: "The validated arrival total is 54 vessels.",
    datasets: [kpiWithoutBulletDataset],
    visualizations: [kpiWithoutBulletVisualization],
    facts: [
      {
        name: "arrival_total",
        value: 54,
        unit: "vessels",
        source: "computed",
        immutable: true,
      },
    ],
    trace: {
      ...base.trace,
      trace_id: "trace-kpi-without-bullet-markers",
      result_hash: "result-hash-kpi-without-bullet-markers",
      sources: [kpiWithoutBulletDataset.id],
      dataset_rows: 1,
      visualization_decision: "kpi",
      chart_profile: ["kpi"],
      visualization_dataset_ids: [kpiWithoutBulletDataset.id],
    },
  };
}

function fulfillJson(route: Route, body: unknown) {
  return route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify(body),
  });
}

async function installMocks(page: Page) {
  await page.route("**/api/v2/capabilities", (route) =>
    fulfillJson(route, capabilities),
  );
  await page.route("**/api/v2/query", async (route, request) => {
    const payload = request.postDataJSON() as QueryRequestPayload;
    return fulfillJson(
      route,
      envelope(
        payload.question,
        payload.conversation_id || "conversation-advanced-visual-profiles",
      ),
    );
  });
}

async function submitAdvancedFixture(page: Page) {
  await page.goto("/analysis");
  await page
    .getByTestId("query-input")
    .fill("Render the frozen advanced visualization profiles.");
  await page.getByTestId("analyze-button").click();
  await expect(page.getByTestId("result-answer")).toHaveText(
    "Six validated analytical views are shown without changing the canonical answer.",
  );
  const chartCanvases = page.locator(".chart-canvas");
  await expect(chartCanvases).toHaveCount(
    visualizations.length,
  );
  for (let index = 0; index < visualizations.length; index += 1) {
    await expect
      .poll(
        () => chartCanvases.nth(index).locator("canvas").count(),
        {
          message: `${visualizations[index].id} must render at least one analytical canvas layer`,
        },
      )
      .toBeGreaterThanOrEqual(1);
  }
}

async function observations(page: Page): Promise<AdvancedRenderObservation[]> {
  return page.evaluate(() => {
    const target = window as unknown as {
      __EAGLE_EYE_VISUALIZATION_OBSERVABILITY__?: AdvancedRenderObservation[];
    };
    return target.__EAGLE_EYE_VISUALIZATION_OBSERVABILITY__ || [];
  });
}

async function waitForRender(
  page: Page,
  visualizationId: string,
): Promise<AdvancedRenderObservation> {
  await expect
    .poll(async () => {
      const matching = (await observations(page)).filter(
        (observation) =>
          observation.visualization_id === visualizationId &&
          observation.interaction_mode === "render",
      );
      return matching.at(-1);
    })
    .toBeTruthy();
  return (await observations(page)).filter(
    (observation) =>
      observation.visualization_id === visualizationId &&
      observation.interaction_mode === "render",
  ).at(-1)!;
}

function nestedDiagnostic(
  observation: AdvancedRenderObservation,
  ...names: string[]
): unknown {
  for (const name of names) {
    const direct = observation[name as keyof AdvancedRenderObservation];
    if (direct !== undefined) return direct;
    const renderValue = observation.render_diagnostics?.[name];
    if (renderValue !== undefined) return renderValue;
    const encodingValue = observation.encoding_diagnostics?.[name];
    if (encodingValue !== undefined) return encodingValue;
    const inspectionValue = observation.inspection?.[name];
    if (inspectionValue !== undefined) return inspectionValue;
  }
  return undefined;
}

function expectProfile(
  observation: AdvancedRenderObservation,
  pattern: RegExp,
) {
  expect(
    observation.semantic_profile,
    `${observation.visualization_id} must expose its advanced semantic profile`,
  ).toEqual(expect.any(String));
  expect(observation.semantic_profile!).toMatch(pattern);
  expect(observation.fallback_reason).toBeNull();
  expect(observation.render_latency_ms).toEqual(expect.any(Number));
}

test.beforeEach(async ({ page }) => {
  await page.clock.setFixedTime(new Date("2026-07-27T10:00:00Z"));
  await installMocks(page);
});

test("render diagnostics describe executed null and keyboard encodings", () => {
  expect(rendererSource).not.toMatch(
    /asNumber\(row\[spec\.value_field\]\)\s*\|\|\s*0/,
  );
  expect(rendererSource).not.toMatch(
    /dispatchAction\(\{\s*type:\s*"(?:highlight|showTip)",\s*seriesIndex:\s*0\b/,
  );
  expect(rendererSource).toContain("inspectableSeriesId");
  expect(rendererSource).toContain("null_to_zero_count");
  expect(rendererSource).toContain("inspected_series_name");
  expect(rendererSource).toContain("encode: { x: 0, y: [1, 2] }");
  expect(rendererSource).toContain("useUTC: true");
  expect(rendererSource).toMatch(
    /const hasBullet = numericValue !== null[\s\S]*\(enhanced\.thresholds\?\.length \?\? 0\) > 0/,
  );
  expect(rendererSource).toMatch(/const scaleValues = \[\s*0,/);
  expect(rendererSource).toContain(
    'if (spec.chart_type !== "line" && spec.chart_type !== "area") return null;',
  );
  expect(rendererSource).toContain("focusedPositiveCarbonAxis");
  expect(rendererSource).toContain(
    'data-axis-policy="positive-carbon-focus"',
  );
});

test("KPI without baseline or thresholds does not leak a literal zero", async ({
  page,
}) => {
  await page.unroute("**/api/v2/query");
  await page.route("**/api/v2/query", async (route, request) => {
    const payload = request.postDataJSON() as QueryRequestPayload;
    return fulfillJson(
      route,
      kpiWithoutBulletEnvelope(
        payload.question,
        payload.conversation_id || "conversation-kpi-without-bullet-markers",
      ),
    );
  });

  await page.goto("/analysis");
  await page
    .getByTestId("query-input")
    .fill("Show a KPI without comparison markers.");
  await page.getByTestId("analyze-button").click();

  const kpi = page.getByRole("region", {
    name: "KPI value without bullet markers",
  });
  await expect(kpi.locator(".kpi-bullet")).toHaveCount(0);
  await expect(kpi.locator(".kpi-canvas")).toHaveText(
    /^Arrivals\s*54 vessels\s*$/,
  );
});

test("reduced-motion preference disables chart and interface animation paths", async ({
  page,
}) => {
  expect(rendererSource).toMatch(
    /animation:\s*!window\.matchMedia\("\(prefers-reduced-motion: reduce\)"\)\.matches/,
  );
  expect(stylesSource).toMatch(
    /@media\s*\(prefers-reduced-motion:\s*reduce\)[\s\S]*animation-duration:\s*0\.01ms\s*!important/,
  );

  await page.emulateMedia({ reducedMotion: "reduce" });
  await submitAdvancedFixture(page);
  expect(
    await page.evaluate(() =>
      window.matchMedia("(prefers-reduced-motion: reduce)").matches,
    ),
  ).toBe(true);
  await expect(page.getByRole("alert")).toHaveCount(0);
});

test("legacy 2.0 Carbon uses only validated uncertainty fields and preserves the answer", async ({
  page,
}) => {
  await page.unroute("**/api/v2/query");
  await page.route("**/api/v2/query", async (route, request) => {
    const payload = request.postDataJSON() as QueryRequestPayload;
    return fulfillJson(
      route,
      legacyCarbonEnvelope(
        payload.question,
        payload.conversation_id || "conversation-legacy-carbon-enriched",
        true,
      ),
    );
  });

  await page.goto("/carbon-emissions");
  await page
    .getByTestId("query-input")
    .fill("Show the legacy validated carbon uncertainty range.");
  await page.getByTestId("analyze-button").click();
  await expect(page.getByTestId("result-answer")).toHaveText(
    "Legacy carbon answer remains byte-for-byte unchanged.",
  );

  const enrichment = page.locator(
    '[data-legacy-enrichment="carbon-uncertainty"]',
  );
  await expect(enrichment).toBeVisible();
  await expect(enrichment).toContainText(
    "Additional uncertainty fields from this saved result are displayed.",
  );
  await expect
    .poll(() => enrichment.locator(".chart-canvas canvas").count())
    .toBeGreaterThanOrEqual(1);

  const observation = await waitForRender(page, "legacy-carbon-trend");
  expect(observation.semantic_profile).toBe("uncertainty-range");
  expect(observation.interval_band_count).toBe(1);
  expect(observation.visualization_contract_version).toBe("2.0");
  expect(observation.source_dataset_ids).toEqual([
    "legacy-carbon-uncertainty-series",
  ]);
  await expect(page.getByRole("alert")).toHaveCount(0);
});

test("legacy 2.0 Carbon refuses presentation enrichment when bound units conflict", async ({
  page,
}) => {
  await page.unroute("**/api/v2/query");
  await page.route("**/api/v2/query", async (route, request) => {
    const payload = request.postDataJSON() as QueryRequestPayload;
    return fulfillJson(
      route,
      legacyCarbonEnvelope(
        payload.question,
        payload.conversation_id || "conversation-legacy-carbon-retained",
        false,
      ),
    );
  });

  await page.goto("/carbon-emissions");
  await page
    .getByTestId("query-input")
    .fill("Show the invalid legacy carbon uncertainty fixture.");
  await page.getByTestId("analyze-button").click();
  await expect(page.getByTestId("result-answer")).toHaveText(
    "Legacy carbon answer remains byte-for-byte unchanged.",
  );

  const retained = page.getByRole("region", {
    name: "Legacy carbon estimate",
  });
  await expect(retained).toBeVisible();
  await expect(retained).not.toHaveAttribute(
    "data-legacy-enrichment",
    "carbon-uncertainty",
  );
  await expect(retained).toContainText(
    "This saved chart uses its original data fields.",
  );
  await expect
    .poll(() => retained.locator(".chart-canvas canvas").count())
    .toBeGreaterThanOrEqual(1);

  const observation = await waitForRender(page, "legacy-carbon-trend");
  expect(observation.semantic_profile).not.toBe("uncertainty-range");
  expect(observation.interval_band_count).toBe(0);
  expect(observation.visualization_contract_version).toBe("2.0");
  expect(observation.source_dataset_ids).toEqual([
    "legacy-carbon-invalid-uncertainty-series",
  ]);
  await expect(page.getByRole("alert")).toHaveCount(0);
});

test("advanced profiles expose deterministic render semantics and exact source bindings", async ({
  page,
}) => {
  await submitAdvancedFixture(page);

  const expected = [
    {
      id: "carbon-uncertainty-trend",
      profile: /uncertainty|range/i,
      sources: [carbonDataset.id],
    },
    {
      id: "port-arrivals-ranking",
      profile: /ranking|lollipop/i,
      sources: [rankingDataset.id],
    },
    {
      id: "dwell-histogram-summary",
      profile: /distribution|histogram|summary/i,
      sources: [dwellBins.id, dwellSummary.id, dwellOutliers.id],
    },
    {
      id: "pressure-forecast-fan",
      profile: /forecast|fan|interval/i,
      sources: [forecastDataset.id],
    },
    {
      id: "arrivals-pressure-fit",
      profile: /correlation|association|fit/i,
      sources: [correlationDataset.id],
    },
  ] as const;

  for (const item of expected) {
    const observation = await waitForRender(page, item.id);
    expectProfile(observation, item.profile);
    expect(observation.visualization_contract_version).toBe("2.1");
    expect(observation.source_dataset_ids).toEqual(item.sources);
  }

  const carbon = await waitForRender(page, "carbon-uncertainty-trend");
  expect(
    nestedDiagnostic(carbon, "interval_band_count", "intervalBands"),
  ).toBe(1);
  expect(
    nestedDiagnostic(carbon, "reference_line_count", "referenceLines"),
  ).toBe(1);
  const carbonVisual = page.getByRole("region", {
    name: "Carbon estimate with uncertainty",
  });
  const truncatedAxisNotice = carbonVisual.getByRole("note", {
    name: "Truncated Carbon y-axis",
  });
  await expect(truncatedAxisNotice).toContainText(
    "Positive-only Carbon scale starts at",
  );
  await expect(truncatedAxisNotice).toContainText(
    "Exact values and uncertainty bounds are unchanged.",
  );
  const axisMinimum = Number(
    await truncatedAxisNotice.getAttribute("data-axis-min"),
  );
  const axisMaximum = Number(
    await truncatedAxisNotice.getAttribute("data-axis-max"),
  );
  expect(axisMinimum).toBeGreaterThan(0);
  expect(axisMinimum).toBeLessThan(10.5);
  expect(axisMaximum).toBeGreaterThan(14.9);

  const ranking = await waitForRender(page, "port-arrivals-ranking");
  expect(
    nestedDiagnostic(ranking, "analytical_series_types", "seriesTypes"),
  ).toEqual(expect.arrayContaining([expect.stringMatching(/bar|custom|scatter/)]));
  const rankingVisual = page.getByRole("region", {
    name: "Port arrivals ranking",
  });
  await expect(
    rankingVisual.locator('[data-axis-policy="positive-carbon-focus"]'),
  ).toHaveCount(0);

  const distribution = await waitForRender(
    page,
    "dwell-histogram-summary",
  );
  expect(
    nestedDiagnostic(
      distribution,
      "analytical_series_names",
      "seriesNames",
    ),
  ).toEqual(expect.arrayContaining(["Observations"]));
  expect(
    nestedDiagnostic(
      distribution,
      "analytical_series_types",
      "seriesTypes",
    ),
  ).toEqual(expect.arrayContaining(["bar"]));
  expect(distribution.rendered_point_count).toBe(dwellBins.row_count);

  const forecast = await waitForRender(page, "pressure-forecast-fan");
  expect(
    nestedDiagnostic(forecast, "interval_band_count", "intervalBands"),
  ).toBe(1);
  expect(
    nestedDiagnostic(forecast, "analytical_series_names", "seriesNames"),
  ).toEqual(
    expect.arrayContaining([
      "80% interval",
      "Actual",
      "Predicted",
    ]),
  );

  const correlation = await waitForRender(page, "arrivals-pressure-fit");
  expect(
    nestedDiagnostic(correlation, "analytical_series_names", "seriesNames"),
  ).toEqual(
    expect.arrayContaining([
      "Arrivals vs Pressure",
      "OLS fit · association only",
    ]),
  );

  const heatmap = await waitForRender(page, "weekday-hour-heatmap");
  expectProfile(heatmap, /heatmap|pattern/i);
  expect(heatmap.visualization_contract_version).toBe("2.1");
  expect(heatmap.source_dataset_ids).toEqual([heatmapDataset.id]);

  await expect(page.getByRole("alert")).toHaveCount(0);
});

test("keyboard inspection selects an analytical series and preserves a null heatmap cell", async ({
  page,
}) => {
  await submitAdvancedFixture(page);

  const carbonVisual = page.getByRole("region", {
    name: "Carbon estimate with uncertainty",
  });
  await carbonVisual
    .getByRole("button", { name: "Inspect chart points with the keyboard" })
    .click();
  const carbonInspector = carbonVisual.locator(".chart-inspector-focus");
  await expect(carbonInspector).toBeFocused();
  await carbonInspector.press("ArrowRight");
  await expect(carbonVisual.locator(".chart-live-region")).toContainText(
    "CO2e TTW: 13.4 tCO2e",
  );

  await expect
    .poll(async () => {
      const matching = (await observations(page)).filter(
        (observation) =>
          observation.visualization_id === "carbon-uncertainty-trend" &&
          observation.interaction_mode === "keyboard_inspection",
      );
      return matching.at(-1);
    })
    .toBeTruthy();
  const inspection = (await observations(page)).filter(
    (observation) =>
      observation.visualization_id === "carbon-uncertainty-trend" &&
      observation.interaction_mode === "keyboard_inspection",
  ).at(-1)!;
  const inspectedSeries = String(
    nestedDiagnostic(
      inspection,
      "inspected_series_name",
      "series_name",
      "seriesName",
    ) || "",
  );
  expect(inspectedSeries).toMatch(/co2e|carbon/i);
  expect(inspectedSeries).not.toMatch(/^__/);
  expect(inspectedSeries).not.toMatch(/interval base/i);

  const heatmapVisual = page.getByRole("region", {
    name: "Weekday and hour arrival pattern",
  });
  await heatmapVisual
    .getByRole("button", { name: "Inspect chart points with the keyboard" })
    .click();
  const heatmapInspector = heatmapVisual.locator(".chart-inspector-focus");
  await heatmapInspector.press("ArrowRight");
  const heatmapAnnouncement = heatmapVisual.locator(".chart-live-region");
  await expect(heatmapAnnouncement).toContainText("Point 2 of 3");
  await expect(heatmapAnnouncement).not.toContainText("Arrivals: 0 vessels");

  const heatmap = await waitForRender(page, "weekday-hour-heatmap");
  expect(
    nestedDiagnostic(heatmap, "preserved_null_count", "preservedNulls"),
  ).toBe(1);
  expect(heatmap.rendered_point_count).toBe(2);
  expect(
    nestedDiagnostic(
      heatmap,
      "null_to_zero_count",
      "nullToZeroCount",
      "coercedNullCount",
    ),
  ).toBe(0);

  await expect(page.getByRole("alert")).toHaveCount(0);
});
