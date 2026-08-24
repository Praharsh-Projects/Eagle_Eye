import AxeBuilder from "@axe-core/playwright";
import { readFileSync } from "node:fs";
import {
  expect,
  test,
  type Page,
  type Request,
  type Route,
} from "@playwright/test";
import type {
  AnswerEnvelope,
  CapabilityResponse,
  Dataset,
  QueryRequestPayload,
  VisualizationSpec,
} from "../../src/types";

const queryCatalog = JSON.parse(
  readFileSync(
    new URL("../../src/data/queryCatalog.json", import.meta.url),
    "utf8",
  ),
) as Record<string, string[]>;

const pages = [
  ["/overview", "Overview"],
  ["/analysis", "Analysis Desk"],
  ["/traffic-monitoring", "Traffic Monitoring"],
  ["/vessel-investigation", "Vessel Investigation"],
  ["/eta-delay", "ETA Watch"],
  ["/port-pressure", "Port Pressure"],
  ["/carbon-emissions", "Carbon Emissions"],
] as const;

const categoryRoutes = [
  ["/traffic-monitoring", "Traffic Monitoring"],
  ["/vessel-investigation", "Vessel Investigation"],
  ["/eta-delay", "ETA & Delay"],
  ["/port-pressure", "Port Pressure"],
  ["/carbon-emissions", "Carbon Emissions"],
] as const;

type CatalogCategory = keyof typeof queryCatalog;

const catalogOrder: CatalogCategory[] = [
  "Traffic Monitoring",
  "Vessel Investigation",
  "ETA & Delay",
  "Port Pressure",
  "Carbon Emissions",
  "Unsupported Scope",
];

const allPrompts = catalogOrder.flatMap((category) => queryCatalog[category]);
const adHocNextTenQuestion =
  "What are the next 10 vessel-reported ETAs for Stockholm, Gothenburg, Nynäshamn, Malmö, and Trelleborg?";

const capabilities: CapabilityResponse = {
  api_version: "2.0",
  modes: [
    "analytics",
    "maritime_research",
    "general_chat",
    "app_help",
    "clarification",
    "unsupported",
  ],
  operations: [
    "arrivals",
    "dwell_distribution",
    "ais_jump",
    "forecast_congestion",
    "diagnostic",
    "carbon",
  ],
  visualization_kinds: [
    "kpi",
    "cartesian",
    "forecast",
    "distribution",
    "heatmap",
    "map",
    "timeline",
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
    schema_version: "2.0-test",
    built_at_utc: "2026-07-23T08:00:00Z",
    available_ports: ["SEGOT", "SEKAN", "LVVNT"],
    enabled_operations: ["arrivals", "carbon", "ais_jump"],
    tables: {
      "arrivals_daily.parquet": {
        readable: true,
        rows: 473,
        coverage: {
          date: { min: "2022-03-01", max: "2022-03-31" },
        },
      },
      "events.parquet": {
        readable: true,
        rows: 3,
        coverage: {
          timestamp_full: { min: "2022-03-01", max: "2022-03-02" },
        },
      },
    },
  },
  kpi_capabilities: {},
  model_routes: {
    planner: "test-planner",
    general: "test-general",
    research: "test-research",
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
  live_eta: {
    provider: "aisstream",
    available: true,
    country_scope: ["SE", "FI", "EE", "LV", "LT", "PL", "DK", "DE"],
    default_scope: "SE",
    source_health: "live",
    timezone: "UTC",
    maximum_horizon_hours: 48,
    operations: [
      "live_port_arrivals",
      "vessel_eta",
      "vessel_delay",
      "eta_comparison",
    ],
    source_kind: "vessel_reported_ais",
    prediction: false,
    bounding_boxes: [
      [
        [53, 9],
        [66, 31.5],
      ],
    ],
  } as CapabilityResponse["live_eta"] & {
    bounding_boxes: Array<Array<[number, number]>>;
  },
  capability_registry: {
    product_label: "Eagle Eye",
    retrieval: "Local evidence index ready",
    historical_analytics: ["arrivals", "carbon"],
  },
  conversation_store: "sqlite",
};

function fixtureLocodes(country: string, count: number): string[] {
  return Array.from({ length: count }, (_, index) =>
    `${country}${index.toString(36).toUpperCase().padStart(3, "0")}`,
  );
}

const overviewCapabilities: CapabilityResponse = {
  ...capabilities,
  data_manifest: {
    ...capabilities.data_manifest,
    // Live-shaped fixture: 124 unique manifest ports across the six countries
    // represented in the current historical archive. Duplicates and malformed
    // values prove that the UI derives counts from valid, deduplicated LOCODEs.
    available_ports: [
      ...fixtureLocodes("SE", 73),
      ...fixtureLocodes("FI", 29),
      ...fixtureLocodes("PL", 9),
      ...fixtureLocodes("EE", 7),
      ...fixtureLocodes("LV", 5),
      ...fixtureLocodes("LT", 1),
      "SE000",
      "not-a-locode",
    ],
    tables: {
      "arrivals_daily.parquet": {
        readable: true,
        rows: 103_851,
        coverage: { date: { min: "2021-01-01", max: "2022-04-30" } },
      },
      "arrivals_hourly.parquet": {
        readable: true,
        rows: 355_688,
        coverage: {
          datetime_hour: { min: "2021-01-01", max: "2022-04-30" },
        },
      },
      "carbon_emissions_call.parquet": {
        readable: true,
        rows: 87,
        coverage: null,
      },
      "carbon_emissions_daily_port.parquet": {
        readable: true,
        rows: 9_117,
        coverage: { date: { min: "2021-01-01", max: "2022-04-30" } },
      },
      "congestion_daily.parquet": {
        readable: true,
        rows: 85_605,
        coverage: { date: { min: "2021-01-01", max: "2022-04-30" } },
      },
      "dwell_time.parquet": {
        readable: true,
        rows: 82_423,
        coverage: {
          arrival_time: { min: "2021-01-01", max: "2022-04-30" },
          departure_time: { min: "2021-01-01", max: "2022-05-05" },
        },
      },
      "events.parquet": {
        readable: true,
        rows: 1_417_791,
        coverage: {
          timestamp_full: { min: "2021-01-01", max: "2022-04-30" },
        },
      },
      "occupancy_hourly.parquet": {
        readable: true,
        rows: 754_131,
        coverage: {
          datetime_hour: { min: "2021-01-01", max: "2022-05-05" },
        },
      },
      "voyages.parquet": {
        readable: true,
        rows: 70_513,
        coverage: {
          arrival_time: { min: "2021-01-01", max: "2022-04-30" },
          departure_time: { min: "2021-01-01", max: "2022-04-30" },
        },
      },
    },
  },
};

const lineDataset: Dataset = {
  id: "arrivals-series",
  columns: [
    {
      field: "date",
      label: "Date",
      data_type: "datetime",
    },
    {
      field: "arrivals",
      label: "Arrivals",
      data_type: "integer",
      unit: "vessels",
    },
  ],
  rows: [
    { date: "2022-03-01", arrivals: 18 },
    { date: "2022-03-02", arrivals: 21 },
    { date: "2022-03-03", arrivals: 15 },
  ],
  row_count: 3,
};

const lineVisualization: VisualizationSpec = {
  id: "arrivals-line",
  kind: "cartesian",
  chart_type: "line",
  title: "Arrivals over time",
  dataset_id: lineDataset.id,
  table_fallback_dataset_id: lineDataset.id,
  accessible_summary:
    "Daily historical arrivals for Gothenburg in chronological order.",
  citations: ["evidence-computed"],
  x_field: "date",
  y_fields: ["arrivals"],
  orientation: "vertical",
  sort: "calendar",
  y_unit: "vessels",
  stacked: false,
};

let responseSequence = 0;

function answerEnvelope(
  question: string,
  conversationId = "conversation-test",
  overrides: Partial<AnswerEnvelope> = {},
): AnswerEnvelope {
  responseSequence += 1;
  return {
    api_version: "2.0",
    conversation_id: conversationId,
    turn_id: `turn-${responseSequence}`,
    question,
    mode: "analytics",
    state: "COMPUTED",
    answer: `Canonical answer: ${question}`,
    plan: {
      mode: "analytics",
      operation: "arrivals",
      metric: "arrivals",
      dimensions: ["date"],
      ports: ["SEGOT"],
      origin_port: null,
      destination_port: null,
      route_pairs: [],
      date_scope: {
        date_from: "2022-03-01",
        date_to: "2022-03-31",
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
      horizon_weeks: 0,
      limit: 10,
      source_scope: "historical",
      carbon_boundary: "TTW",
      pollutants: [],
      requested_visual: "line",
      ambiguities: [],
      clarification: null,
      reason: "Deterministic test fixture.",
      context_inherited: [],
      planner_source: "deterministic",
      planner_model: null,
    },
    facts: [
      {
        name: "arrival_total",
        value: 54,
        unit: "vessels",
        entity: "SEGOT",
        source: "computed",
        immutable: true,
      },
    ],
    applied_scope: {
      ports: ["SEGOT"],
      date_from: "2022-03-01",
      date_to: "2022-03-31",
    },
    datasets: [lineDataset],
    visualizations: [lineVisualization],
    evidence: [
      {
        id: "evidence-computed",
        source_type: "computed",
        title: "Validated arrivals dataset",
        excerpt: "Three historical daily arrival buckets.",
        metadata: { dataset_id: lineDataset.id },
      },
    ],
    freshness: {
      data_from: "2021-01-01",
      data_to: "2022-04-30",
      historical: true,
      message: "Historical coverage through 2022-04-30",
    },
    confidence: "high",
    assurance: {
      status: "verified",
      level: "high",
      basis: "direct_computation",
      reason: "All result-specific high-assurance publication checks passed.",
      checks: ["high_assurance_gate=passed"],
    },
    availability: {
      code: "available",
      provider: "structured_datasets",
      retryable: false,
    },
    caveats: ["Historical data must not be presented as current traffic."],
    trace: {
      trace_id: `trace-${responseSequence}`,
      route: "analytics",
      operation: "arrivals",
      planner_source: "deterministic",
      planner_model: null,
      model: "none",
      reasoning_effort: "none",
      sources: [lineDataset.id],
      retrieval_mode: "local",
      retrieval_backend: "test",
      retrieval_status: "ready",
      retrieval_top_k: 5,
      result_state: "COMPUTED",
      failure_state: null,
      result_hash: `result-hash-${responseSequence}`,
      data_manifest_version: "2.0-test",
      dataset_rows: lineDataset.row_count,
      visualization_decision: "line",
      latency_ms: 4,
      warnings: [],
    },
    ...overrides,
  };
}

function json(route: Route, body: unknown, status = 200) {
  return route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(body),
  });
}

async function installApiMocks(
  page: Page,
  queryResponder: (
    payload: QueryRequestPayload,
    request: Request,
  ) => AnswerEnvelope | Promise<AnswerEnvelope> = (payload) =>
    answerEnvelope(
      payload.question,
      payload.conversation_id || "conversation-test",
    ),
) {
  await page.route("**/api/v2/capabilities", (route) =>
    json(route, capabilities),
  );
  await page.route("**/api/v2/query", async (route, request) => {
    const payload = request.postDataJSON() as QueryRequestPayload;
    return json(route, await queryResponder(payload, request));
  });
  await page.route("**/api/v2/exports", async (route, request) => {
    const payload = request.postDataJSON() as {
      dataset_id: string;
      format: "csv" | "json";
    };
    return json(route, {
      export_id: "export-test",
      format: payload.format,
      dataset_id: payload.dataset_id,
      row_count: 3,
      path: `/tmp/eagle-eye/${payload.dataset_id}.${payload.format}`,
    });
  });
  await page.route("**/api/v2/feedback", (route) =>
    json(route, { feedback_id: "feedback-test", status: "accepted" }),
  );
}

async function mockOverviewCapabilities(
  page: Page,
  fixture: CapabilityResponse = overviewCapabilities,
) {
  await page.unroute("**/api/v2/capabilities");
  await page.route("**/api/v2/capabilities", (route) =>
    json(route, fixture),
  );
}

async function expectNoSeriousA11yViolations(page: Page) {
  const results = await new AxeBuilder({ page }).analyze();
  const blocking = results.violations.filter(
    (violation) =>
      violation.impact === "serious" || violation.impact === "critical",
  );
  expect(
    blocking,
    blocking.map((item) => `${item.id}: ${item.help}`).join("\n"),
  ).toEqual([]);
}

async function runAnalysis(
  page: Page,
  question: string,
  expectedAnswer = question,
) {
  const input = page.getByTestId("query-input");
  await expect(input).toBeVisible();
  await input.fill(question);
  await page.getByTestId("analyze-button").click();
  await expect(page.getByTestId("result-answer")).toContainText(expectedAnswer);
}

function readableEtaPresentationEnvelope(
  question: string,
  conversationId = "eta-readable-answer-test",
): AnswerEnvelope {
  const rows = [
    {
      row_id: "eta-priority-1",
      mmsi: "265100001",
      vessel_label: "STEN ARNOLD",
      destination_name: "Malmö",
      destination_locode: "SEMMA",
      reported_eta_utc: "2026-07-28T13:30:00Z",
      latitude: 55.7345,
      longitude: 12.8821,
      speed_kn: 9.3,
      observation_time_utc: "2026-07-28T13:18:53Z",
    },
    {
      row_id: "eta-priority-2",
      mmsi: "265100002",
      vessel_label: "STI HAMMERSMITH",
      destination_name: "Gothenburg",
      destination_locode: "SEGOT",
      reported_eta_utc: "2026-07-28T14:00:00Z",
      latitude: 57.69,
      longitude: 11.879,
      speed_kn: 0,
      observation_time_utc: "2026-07-28T13:11:56Z",
    },
    {
      row_id: "eta-priority-3",
      mmsi: "265100003",
      vessel_label: "ALBATROS",
      destination_name: "Stockholm",
      destination_locode: "SESTO",
      reported_eta_utc: null,
      latitude: 59.3425,
      longitude: 18.4351,
      speed_kn: 0,
      observation_time_utc: "2026-07-28T11:07:10Z",
    },
    {
      row_id: "eta-information-4",
      mmsi: "265100004",
      vessel_label: "AMBITION",
      destination_name: "Stockholm",
      destination_locode: "SESTO",
      reported_eta_utc: null,
      latitude: 54.1763,
      longitude: 12.0938,
      speed_kn: 11.2,
      observation_time_utc: "2026-07-28T12:47:47Z",
    },
    {
      row_id: "eta-information-5",
      mmsi: "265100005",
      vessel_label: "ANINA",
      destination_name: "Gothenburg",
      destination_locode: "SEGOT",
      reported_eta_utc: null,
      latitude: 53.8798,
      longitude: 9.1024,
      speed_kn: 12.4,
      observation_time_utc: "2026-07-28T00:21:42Z",
    },
  ];
  const columns: Dataset["columns"] = [
    { field: "row_id", label: "Row", data_type: "string" },
    { field: "mmsi", label: "MMSI", data_type: "string" },
    { field: "vessel_label", label: "Vessel", data_type: "string" },
    {
      field: "destination_name",
      label: "Destination",
      data_type: "string",
    },
    {
      field: "destination_locode",
      label: "UN/LOCODE",
      data_type: "string",
    },
    {
      field: "reported_eta_utc",
      label: "Reported ETA",
      data_type: "datetime",
    },
    { field: "latitude", label: "Latitude", data_type: "number" },
    { field: "longitude", label: "Longitude", data_type: "number" },
    { field: "speed_kn", label: "Speed", data_type: "number", unit: "kn" },
    {
      field: "observation_time_utc",
      label: "Observed",
      data_type: "datetime",
    },
  ];
  const statusDataset: Dataset = {
    id: "table",
    columns,
    rows,
    row_count: rows.length,
  };
  const timelineDataset: Dataset = {
    id: "eta_timeline",
    columns,
    rows: rows.filter((row) => row.reported_eta_utc),
    row_count: 2,
  };
  const timeline: VisualizationSpec = {
    id: "eta_readable_answer_timeline",
    kind: "timeline",
    title: "Due-soon vessel-reported ETAs",
    dataset_id: timelineDataset.id,
    table_fallback_dataset_id: statusDataset.id,
    row_id_field: "row_id",
    accessible_summary:
      "Two valid vessel-reported ETAs are plotted on separate vessel lanes.",
    citations: ["aisstream-readable-answer"],
    time_field: "reported_eta_utc",
    label_field: "vessel_label",
    lane_field: "vessel_label",
    detail_fields: ["destination_name", "speed_kn"],
  };
  const response = answerEnvelope(question, conversationId);
  response.answer =
    "Sweden-bound 12-hour shift watch: 96 vessel signals reviewed, including 2 due soon; attention: 1 low-speed, 1 ETA revision, 7 stale-position, 18 missing-ETA. STEN ARNOLD to Malmö (SEMMA), reported ETA 2026-07-28T13:30:00Z, position 55.7345, 12.8821, speed 9.3 knots, observed 2026-07-28T13:18:53Z; STI HAMMERSMITH to Gothenburg (SEGOT), reported ETA 2026-07-28T14:00:00Z, position 57.6900, 11.8790, speed 0.0 knots observed 2026-07-28T13:11:56Z; ALBATROS to Stockholm (SESTO), reported ETA unavailable, position 59.3425, 18.4351, speed 0.0 knots, observed 2026-07-28T11:07:10Z; AMBITION to Stockholm (SESTO), reported ETA unavailable, position 54.1763, 12.0938, speed 11.2 knots, observed 2026-07-28T12:47:47Z; ANINA to Gothenburg (SEGOT), reported ETA unavailable, position 53.8798, 9.1024, speed 12.4 knots, observed 2026-07-28T00:21:42Z. Showing 5 of 96; source observed 2026-07-28T13:21:22Z. These are vessel-reported AIS signals, not an official schedule, confirmed delay, arrival confirmation, or prediction.";
  response.plan = {
    ...response.plan,
    operation: "vessel_eta",
    source_scope: "aisstream",
    eta_watch_intent: "shift_handover",
    horizon_hours: 12,
    limit: 20,
  };
  response.datasets = [statusDataset, timelineDataset];
  response.visualizations = [timeline];
  response.trace = {
    ...response.trace,
    operation: "vessel_eta",
    sources: ["aisstream"],
    dataset_rows: rows.length,
    visualization_decision: "timeline",
  };
  response.evidence = [
    {
      id: "aisstream-readable-answer",
      source_type: "traffic_event",
      title: "AISStream observation snapshot",
      url: "https://aisstream.io/",
      metadata: {},
    },
  ];
  response.operational_brief = {
    intent: "shift_handover",
    headline: "Two due-soon vessels; two signals need immediate attention.",
    window_start_utc: "2026-07-28T13:21:22Z",
    window_end_utc: "2026-07-29T01:21:22Z",
    matched_count: 96,
    displayed_count: 5,
    prioritized_items: [
      {
        row_id: "eta-priority-2",
        vessel_label: "STI HAMMERSMITH",
        priority: "attention",
        status: "low_speed",
        reason: "Reported speed is below the 2 kn attention threshold.",
        actions: ["locate_vessel", "watch_next_six_hours"],
      },
      {
        row_id: "eta-priority-3",
        vessel_label: "ALBATROS",
        priority: "attention",
        status: "missing_eta",
        reason: "No valid vessel-reported ETA is available.",
        actions: ["locate_vessel"],
      },
      {
        row_id: "eta-priority-1",
        vessel_label: "STEN ARNOLD",
        priority: "monitor",
        status: "due_soon",
        reason: "The reported ETA falls inside the shift window.",
        actions: ["locate_vessel"],
      },
    ],
    exceptions: [
      {
        code: "due_soon",
        count: 2,
        summary: "Two vessels with valid reported ETAs are due this shift.",
      },
      {
        code: "low_speed",
        count: 1,
        summary: "One due-soon vessel is moving below 2 kn.",
      },
      {
        code: "missing_eta",
        count: 18,
        summary: "Eighteen signals have no valid reported ETA.",
      },
      {
        code: "stale_position",
        count: 7,
        summary: "Seven signals have stale positions.",
      },
    ],
    source_health: "live",
    source_observed_at: "2026-07-28T13:21:22Z",
    coverage:
      "Vessel-reported AIS observations; not an official arrival schedule.",
  };
  return response;
}

test.beforeEach(async ({ page }) => {
  responseSequence = 0;
  await installApiMocks(page);
});

test("root and wildcard routes open Overview while all seven deep routes retain their exact labels", async ({
  page,
}) => {
  await mockOverviewCapabilities(page);
  await page.goto("/");
  await expect(page).toHaveURL(/\/overview$/);
  await expect(page.getByTestId("app-shell")).toBeVisible();
  await expect(page.getByTestId("operational-overview")).toBeVisible();
  await expect(
    page.getByRole("heading", { level: 1, name: "Baltic archive footprint" }),
  ).toBeVisible();
  await expect(page.getByText("Chat Assistant", { exact: true })).toHaveCount(0);

  const navigation = page.getByRole("navigation", {
    name: "Operational areas",
  });
  await expect(page.getByTestId("nav-rail")).toBeVisible();
  await expect(page.getByTestId("nav-rail")).toHaveClass(/nav-rail-collapsed/);
  await expect(
    page.getByTestId("nav-rail").getByRole("button", {
      name: /Collapse navigation|Expand navigation/,
    }),
  ).toHaveCount(0);
  await expect(page.locator(".utility-bar")).toContainText("Overview");
  await expect(navigation.getByRole("link")).toHaveCount(7);
  for (const [, label] of pages) {
    await expect(
      navigation.getByRole("link", { name: label, exact: true }),
    ).toHaveAccessibleName(label);
  }

  for (const [path, label] of pages) {
    await page.goto(path);
    await expect(page).toHaveURL(new RegExp(`${path}$`));
    if (path === "/overview") {
      await expect(page.getByTestId("operational-overview")).toBeVisible();
    } else {
      await expect(
        page.getByRole("heading", { level: 1, name: label }),
      ).toBeVisible();
      await expect(page.getByTestId("nav-rail")).not.toHaveClass(
        /nav-rail-collapsed/,
      );
      await expect(
        page.getByTestId("nav-rail").getByRole("button", {
          name: "Collapse navigation",
        }),
      ).toBeVisible();
    }
    await expect(page).toHaveTitle(`${label} | Eagle Eye`);
  }

  await page.goto("/not-a-route");
  await expect(page).toHaveURL(/\/overview$/);
  await expect(page.getByTestId("operational-overview")).toBeVisible();
});

test("the Overview gateway always opens a fresh Analysis Desk workspace", async ({
  page,
}) => {
  await mockOverviewCapabilities(page);
  await page.goto("/overview");
  const newAnalysis = page.getByTestId("overview-enter-analysis");
  await expect(newAnalysis).toHaveCount(1);
  await newAnalysis.click();

  await expect(page).toHaveURL(/\/analysis$/);
  await expect(
    page.getByRole("heading", { level: 1, name: "Analysis Desk" }),
  ).toBeVisible();
  await expect(page.getByTestId("query-input")).toHaveValue("");
});

test("Overview presents a historical-only Baltic situation sheet without fabricated activity", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await mockOverviewCapabilities(page);
  await page.goto("/overview");

  const overview = page.getByTestId("operational-overview");
  await expect(overview).toContainText("Eagle Eye / Baltic situation sheet");
  await expect(
    overview.getByRole("heading", {
      level: 1,
      name: "Baltic archive footprint",
    }),
  ).toBeVisible();
  await expect(overview).toContainText(
    "124 verified ports · 6 represented countries",
  );

  const readiness = overview.locator('[aria-label="Workspace readiness"]');
  await expect(readiness).toContainText("AIS watch");
  await expect(readiness).toContainText("Live");
  await expect(readiness).toContainText("Evidence");
  await expect(readiness).toContainText("Ready");
  await expect(readiness).toContainText("01 Jan 2021 — 30 Apr 2022");
  await expect(readiness).toContainText("Historical—not current");
  await expect(readiness).toContainText("9 validated datasets");
  await expect(readiness).toContainText("2,879,206 records");

  const atlas = overview.getByTestId("overview-historical-atlas");
  await expect(atlas).toBeVisible();
  await expect(atlas.locator('[data-country-code]')).toHaveCount(8);
  const expectedCoverage = [
    ["SE", "Sweden", "73", "4"],
    ["FI", "Finland", "29", "3"],
    ["PL", "Poland", "9", "2"],
    ["EE", "Estonia", "7", "2"],
    ["LV", "Latvia", "5", "1"],
    ["LT", "Lithuania", "1", "1"],
  ] as const;
  for (const [code, name, count, tier] of expectedCoverage) {
    const country = atlas.locator(`[data-country-code="${code}"]`);
    await expect(country).toHaveAttribute("data-coverage-tier", tier);
    await expect(country).toContainText(name);
    await expect(country).toContainText(count);
  }
  for (const [code, name] of [
    ["DE", "Germany"],
    ["DK", "Denmark"],
  ] as const) {
    const country = atlas.locator(`[data-country-code="${code}"]`);
    await expect(country).toHaveAttribute("data-coverage-tier", "0");
    await expect(country).toContainText(name);
    await expect(country).toContainText(/not represented/i);
  }
  await expect(overview).toContainText(
    "Historical coverage—not traffic volume or current vessel activity",
  );
  const coverageDisclosure = overview.getByTestId(
    "overview-coverage-disclosure",
  );
  await expect(coverageDisclosure).toContainText("View coverage data");
  await coverageDisclosure.getByText("View coverage data", { exact: true }).click();
  await expect(coverageDisclosure).toContainText("Sweden");
  await expect(coverageDisclosure).toContainText("73");
  await expect(overview.getByTestId("overview-enter-analysis")).toContainText(
    "Start analysis",
  );
  await expect(overview.getByTestId("overview-enter-analysis")).toHaveCount(1);

  for (const rejectedSelector of [
    "canvas",
    ".immersive-overview-webgl",
    '[data-testid="overview-theatre"]',
    '[data-testid="overview-stage-nav"]',
    '[data-testid="overview-webgl-fallback"]',
    ".operational-status-ledger",
    ".operational-country-ledger",
    ".operational-source-register",
  ]) {
    await expect(overview.locator(rejectedSelector)).toHaveCount(0);
  }
});

test("Overview readiness ribbon reports loading, live, stale, warming, unavailable, and failed capability states", async ({
  page,
}) => {
  await page.unroute("**/api/v2/capabilities");
  let releaseCapabilities: (() => void) | undefined;
  await page.route("**/api/v2/capabilities", async (route) => {
    await new Promise<void>((resolve) => {
      releaseCapabilities = resolve;
    });
    return json(route, overviewCapabilities);
  });

  await page.goto("/overview");
  let readiness = page.locator('[aria-label="Workspace readiness"]');
  await expect(readiness).toContainText("Loading");
  await expect(readiness).toHaveAttribute("aria-busy", "true");
  await expect.poll(() => Boolean(releaseCapabilities)).toBe(true);
  releaseCapabilities?.();
  await expect(readiness).toContainText("Live");
  await expect(readiness).toContainText("Ready");
  await expect(readiness).toHaveAttribute("aria-busy", "false");

  const sourceStates = [
    {
      state: "unavailable",
      available: false,
      reason: "AIS coverage is temporarily unavailable.",
    },
    {
      state: "stale",
      available: true,
      reason: "The last AIS observation is outside the freshness window.",
    },
    {
      state: "warming",
      available: true,
      reason: "The Baltic collector is building its first snapshot.",
    },
  ] as const;

  for (const sourceState of sourceStates) {
    await test.step(sourceState.state, async () => {
      await page.unroute("**/api/v2/capabilities");
      await page.route("**/api/v2/capabilities", (route) =>
        json(route, {
          ...overviewCapabilities,
          live_eta: {
            ...overviewCapabilities.live_eta,
            available: sourceState.available,
            source_health: sourceState.state,
            reason: sourceState.reason,
          },
        }),
      );
      await page.reload();
      readiness = page.locator('[aria-label="Workspace readiness"]');
      await expect(readiness).toContainText(
        new RegExp(sourceState.state, "i"),
      );
      await expect(readiness).toContainText(sourceState.reason);
    });
  }

  await test.step("capability request unavailable", async () => {
    await page.unroute("**/api/v2/capabilities");
    await page.route("**/api/v2/capabilities", (route) =>
      json(route, { detail: "Capability service unavailable" }, 503),
    );
    await page.reload();
    readiness = page.locator('[aria-label="Workspace readiness"]');
    await expect(readiness).toContainText("Workspace status unavailable");
    await expect(readiness).toContainText("Evidence status unavailable");
    await expect(page.locator(".situation-atlas-fallback")).toBeVisible();
    await expect(page.getByText("AISStream", { exact: true })).toHaveCount(0);
  });

  await test.step("incomplete manifest does not publish an aggregate row total", async () => {
    await page.unroute("**/api/v2/capabilities");
    await page.route("**/api/v2/capabilities", (route) =>
      json(route, {
        ...overviewCapabilities,
        data_manifest: {
          ...overviewCapabilities.data_manifest,
          tables: {
            ...overviewCapabilities.data_manifest.tables,
            "voyages.parquet": {
              readable: true,
              coverage: {
                departure_time: {
                  min: "2021-01-01",
                  max: "2022-04-30",
                },
              },
            },
          },
        },
      }),
    );
    await page.reload();
    readiness = page.locator('[aria-label="Workspace readiness"]');
    await expect(readiness).toContainText("Row total unavailable");
    await expect(readiness).not.toContainText("2,808,693");
  });

  await test.step("non-regional manifest ports use the exact coverage fallback", async () => {
    await page.unroute("**/api/v2/capabilities");
    await page.route("**/api/v2/capabilities", (route) =>
      json(route, {
        ...overviewCapabilities,
        data_manifest: {
          ...overviewCapabilities.data_manifest,
          available_ports: ["GBLON", "not-a-locode"],
        },
      }),
    );
    await page.reload();
    const fallback = page.locator(".situation-atlas-fallback");
    await expect(fallback).toBeVisible();
    await expect(fallback).toContainText(
      "Atlas geometry or coverage is unavailable",
    );
    await expect(fallback).toContainText("Germany");
    await expect(fallback).toContainText("Not represented");
  });
});

test("Overview opens capability-derived data provenance in a focus-trapped dialog", async ({
  page,
}) => {
  await mockOverviewCapabilities(page);
  await page.goto("/overview");

  const provenanceButton = page.getByTestId("overview-provenance-button");
  await expect(provenanceButton).toContainText("Data provenance");
  await expect(provenanceButton).toHaveAttribute("aria-haspopup", "dialog");
  await provenanceButton.focus();
  await provenanceButton.click();

  const provenanceDialog = page.getByRole("dialog", {
    name: "Data provenance",
  });
  await expect(provenanceDialog).toBeVisible();
  expect(
    await page.evaluate(
      () => document.activeElement?.closest('[role="dialog"]') !== null,
    ),
  ).toBe(true);
  const inventory = provenanceDialog.getByTestId("dataset-inventory");
  await expect(inventory).toContainText(
    "9 validated datasets · 2,879,206 historical records",
  );
  await expect(
    inventory.getByRole("table", { name: "Dataset inventory" }).getByRole("row"),
  ).toHaveCount(10);
  await expect(
    inventory.getByText("Arrivals Daily", { exact: true }),
  ).toBeVisible();
  await expect(
    inventory.getByText("2021-01-01 to 2022-04-30", { exact: true }).first(),
  ).toBeVisible();
  await expect(inventory.getByText("103,851", { exact: true })).toBeVisible();
  await expect(inventory.getByText("Events", { exact: true })).toBeVisible();
  await expect(inventory.getByText("1,417,791", { exact: true })).toBeVisible();

  await page.keyboard.press("Escape");
  await expect(provenanceDialog).toBeHidden();
  await expect(provenanceButton).toBeFocused();
});

test("sample library exposes all 57 frozen prompts in category and source order", async ({
  page,
}) => {
  expect(allPrompts).toHaveLength(57);
  await page.goto("/analysis");
  await page.getByTestId("sample-library-button").click();

  const dialog = page.getByRole("dialog", { name: "Sample query library" });
  await expect(dialog).toBeVisible();
  await expect(
    dialog.getByRole("searchbox", { name: "Search all 57 prompts" }),
  ).toBeVisible();
  await expect(
    dialog.locator(".sample-group h3").allTextContents(),
  ).resolves.toEqual(
    catalogOrder.map((category) =>
      category === "ETA & Delay" ? "ETA Watch" : category,
    ),
  );
  await expect(
    dialog.locator(".sample-groups li button").allTextContents(),
  ).resolves.toEqual(allPrompts);

  const unsupported = queryCatalog["Unsupported Scope"].at(-1)!;
  await dialog.getByRole("button", { name: unsupported, exact: true }).click();
  await expect(page.getByTestId("query-input")).toHaveValue(unsupported);
  await expect(page.getByTestId("result-answer")).toHaveCount(0);
});

test("each analytical page preserves its category-specific prompt order", async ({
  page,
}) => {
  for (const [path, category] of categoryRoutes) {
    await page.goto(path);
    const selector = page.getByLabel("Sample query");
    const options = await selector.locator("option").allTextContents();
    expect(options.slice(1)).toEqual(queryCatalog[category]);
    expect(options[0]).toBe(
      `Select one of ${queryCatalog[category].length} samples`,
    );
  }
});

test("ETA question controls and all ten current samples remain visible after a result", async ({
  page,
}) => {
  await page.goto("/eta-delay");
  await runAnalysis(
    page,
    queryCatalog["ETA & Delay"][0],
  );

  await expect(
    page.getByRole("heading", { name: "Ask a question", exact: true }),
  ).toBeVisible();
  await expect(page.getByTestId("query-input")).toBeVisible();
  await expect(page.getByTestId("analyze-button")).toBeVisible();
  await expect(page.getByLabel("Sample query")).toBeVisible();
  await expect(page.getByLabel("Sample query").locator("option")).toHaveCount(11);
  await expect(
    page.getByText("Submitted operational question", { exact: true }),
  ).toHaveCount(0);
  await expect(
    page.getByText("Expand to edit or rerun", { exact: true }),
  ).toHaveCount(0);
});

test("ETA Watch renders a decision brief and prepares vessel follow-ups without resubmitting", async ({
  page,
}) => {
  const etaDataset: Dataset = {
    id: "eta-watch",
    columns: [
      { field: "row_id", label: "Row", data_type: "string" },
      { field: "vessel_label", label: "Vessel", data_type: "string" },
      { field: "destination_label", label: "Destination", data_type: "string" },
      {
        field: "reported_eta_utc",
        label: "Reported ETA",
        data_type: "datetime",
      },
      { field: "speed_kn", label: "Speed", data_type: "number", unit: "kn" },
      {
        field: "observed_at_utc",
        label: "Observed",
        data_type: "datetime",
      },
    ],
    rows: [
      {
        row_id: "eta-1",
        vessel_label: "BALTIC TESTER",
        destination_label: "Stockholm (SESTO)",
        reported_eta_utc: "2026-07-28T01:00:00Z",
        speed_kn: 1.4,
        observed_at_utc: "2026-07-27T20:25:00Z",
      },
    ],
    row_count: 1,
  };
  let submitted = 0;
  await installApiMocks(page, (payload) => {
    submitted += 1;
    const response = answerEnvelope(
      payload.question,
      payload.conversation_id || "eta-watch-test",
    );
    response.plan = {
      ...response.plan,
      operation: "vessel_eta",
      source_scope: "aisstream",
      eta_watch_intent: "low_speed_exceptions",
      horizon_hours: 6,
    };
    response.datasets = [etaDataset];
    response.trace = {
      ...response.trace,
      operation: "vessel_eta",
      sources: ["aisstream"],
      dataset_rows: 1,
    };
    response.evidence = [
      {
        id: "aisstream-snapshot",
        source_type: "traffic_event",
        title: "AISStream observation snapshot",
        url: "https://aisstream.io/",
        metadata: {},
      },
    ];
    response.operational_brief = {
      intent: "low_speed_exceptions",
      headline: "One Sweden-bound vessel needs attention.",
      window_start_utc: "2026-07-27T20:30:00Z",
      window_end_utc: "2026-07-28T02:30:00Z",
      matched_count: 1,
      displayed_count: 1,
      prioritized_items: [
        {
          row_id: "eta-1",
          vessel_label: "BALTIC TESTER",
          priority: "attention",
          status: "low_speed",
          reason: "Reported speed is below the requested 2 kn threshold.",
          actions: [
            "locate_vessel",
            "watch_next_six_hours",
            "inspect_eta_changes",
          ],
        },
      ],
      exceptions: [
        {
          code: "low_speed",
          count: 1,
          summary: "One due-soon vessel is moving below 2 kn.",
        },
      ],
      source_health: "live",
      source_observed_at: "2026-07-27T20:25:00Z",
      coverage:
        "Vessel-reported AIS observations; not an official arrival schedule.",
    };
    return response;
  });

  await page.goto("/eta-delay");
  await runAnalysis(
    page,
    "Which Sweden-bound vessels due in the next 6 hours are moving below 2 knots?",
  );

  const brief = page.getByTestId("operational-brief");
  await expect(brief).toContainText("One Sweden-bound vessel needs attention.");
  await expect(brief).toContainText("BALTIC TESTER");
  await expect(brief).toContainText("Stockholm (SESTO)");
  await expect(brief).toContainText("1.4 kn");
  await brief.getByRole("button", { name: "Locate vessel" }).click();
  await expect(page.getByTestId("query-input")).toHaveValue(
    "Where is BALTIC TESTER now, and what ETA is it transmitting?",
  );
  expect(submitted).toBe(1);
});

test("ETA Watch leads with a concise finding and keeps the verbatim canonical response collapsed", async ({
  page,
}) => {
  const question = queryCatalog["ETA & Delay"][0];
  await installApiMocks(page, (payload) =>
    readableEtaPresentationEnvelope(
      payload.question,
      payload.conversation_id || "eta-readable-answer-desktop",
    ),
  );

  await page.goto("/eta-delay");
  await runAnalysis(page, question, "Sweden-bound 12-hour shift watch");

  const summary = page.getByTestId("eta-answer-summary");
  await expect(summary).toBeVisible();
  await expect(summary).toContainText(
    "Two due-soon vessels; two signals need immediate attention.",
  );

  const stats = page.getByTestId("eta-key-stats");
  await expect(stats).toBeVisible();
  await expect(stats).toContainText("Signals reviewed");
  await expect(stats).toContainText("96");
  await expect(stats).toContainText("Due soon");
  await expect(stats).toContainText("2");
  await expect(stats).toContainText("Needs attention");
  await expect(stats).toContainText("2");

  const attention = page.getByRole("region", { name: "Needs attention" });
  await expect(attention).not.toContainText(
    "Two vessels with valid reported ETAs are due this shift.",
  );
  await expect(attention).toContainText(
    "One due-soon vessel is moving below 2 kn.",
  );

  const priorityVessels = page.getByTestId("eta-priority-vessels");
  await expect(priorityVessels).toBeVisible();
  await expect(
    priorityVessels.getByTestId("eta-priority-vessel"),
  ).toHaveCount(3);
  await expect(priorityVessels).toContainText("STI HAMMERSMITH");
  await expect(priorityVessels).toContainText(/below.*2 kn/i);
  await expect(priorityVessels).toContainText("ALBATROS");
  await expect(priorityVessels).toContainText(/no valid.*ETA/i);
  await expect(priorityVessels).toContainText("STEN ARNOLD");

  const canonicalDisclosure = page.getByTestId(
    "canonical-response-disclosure",
  );
  await expect(canonicalDisclosure).toBeVisible();
  await expect(canonicalDisclosure).toHaveJSProperty("open", false);
  await expect(page.getByTestId("result-answer")).toBeHidden();

  const answerRecord = page.locator(".answer-record");
  const compactBox = await answerRecord.boundingBox();
  expect(compactBox).not.toBeNull();
  expect(compactBox!.height).toBeLessThan(420);

  await canonicalDisclosure
    .getByText("Full response", { exact: true })
    .click();
  await expect(canonicalDisclosure).toHaveJSProperty("open", true);
  await expect(page.getByTestId("result-answer")).toBeVisible();
  await expect(page.getByTestId("result-answer")).toHaveText(
    readableEtaPresentationEnvelope(question).answer,
  );
  await expectNoSeriousA11yViolations(page);
});

test("ETA destination rankings never masquerade as vessel watchlist rows", async ({
  page,
}) => {
  await installApiMocks(page, (payload) => {
    const response = readableEtaPresentationEnvelope(
      payload.question,
      payload.conversation_id || "eta-destination-load-readable",
    );
    response.answer =
      "Gothenburg has the highest validated inbound signal count.";
    response.plan = {
      ...response.plan,
      eta_watch_intent: "destination_load",
    };
    response.datasets = [
      {
        id: "table",
        columns: [
          { field: "row_id", label: "Row", data_type: "string" },
          {
            field: "destination_name",
            label: "Destination",
            data_type: "string",
          },
          {
            field: "inbound_vessels",
            label: "Inbound vessels",
            data_type: "number",
          },
        ],
        rows: [
          {
            row_id: "destination-gothenburg",
            destination_name: "Gothenburg",
            inbound_vessels: 12,
          },
        ],
        row_count: 1,
      },
    ];
    response.visualizations = [];
    response.operational_brief = {
      ...response.operational_brief!,
      intent: "destination_load",
      headline: "Inbound destination load ranked.",
      matched_count: 12,
      displayed_count: 1,
      exceptions: [],
      prioritized_items: [
        {
          row_id: "destination-gothenburg",
          priority: "information",
          status: "observed",
          reason: "Highest validated destination count.",
          actions: [],
        },
      ],
    };
    return response;
  });

  await page.goto("/eta-delay");
  await runAnalysis(
    page,
    "Which Swedish destinations have the most AIS-reported inbound vessels in the next 24 hours?",
    "Gothenburg has the highest",
  );

  await expect(page.getByTestId("eta-answer-summary")).toContainText(
    "12 validated vessel signals contribute to the destination ranking.",
  );
  await expect(page.getByTestId("eta-priority-vessels")).toHaveCount(0);
  await expect(page.getByText("Selected vessel", { exact: true })).toHaveCount(
    0,
  );
});

test("ETA stale-only results never appear as current reported ETAs", async ({
  page,
}) => {
  await installApiMocks(page, (payload) => {
    const response = readableEtaPresentationEnvelope(
      payload.question,
      payload.conversation_id || "eta-stale-only-readable",
    );
    response.answer =
      "No current vessel-reported ETA passed the freshness check.";
    response.plan = {
      ...response.plan,
      eta_watch_intent: "inbound_watchlist",
      limit: 1,
    };
    response.datasets = [
      {
        id: "eta_freshness_candidates",
        columns: [
          { field: "row_id", label: "Row", data_type: "string" },
          { field: "vessel_label", label: "Vessel", data_type: "string" },
          {
            field: "last_reported_eta_utc",
            label: "Last ETA—not current",
            data_type: "datetime",
          },
        ],
        rows: [
          {
            row_id: "stale-only-1",
            vessel_label: "STALE ONLY",
            last_reported_eta_utc: "2026-07-28T14:00:00Z",
            eta_observation_age_minutes: 42,
            validation_reason: "The ETA broadcast is too old.",
          },
        ],
        row_count: 1,
      },
    ];
    response.visualizations = [];
    response.operational_brief = {
      ...response.operational_brief!,
      intent: "inbound_watchlist",
      headline: "No current ETA signal passed validation.",
      matched_count: 0,
      displayed_count: 0,
      prioritized_items: [],
      exceptions: [],
    };
    return response;
  });

  await page.goto("/eta-delay");
  await runAnalysis(
    page,
    "Show the next current vessel-reported ETA.",
    "No current vessel-reported ETA",
  );

  await expect(page.getByTestId("validated-inbound-list")).toHaveCount(0);
  await expect(page.getByTestId("eta-freshness-candidates")).toContainText(
    "STALE ONLY",
  );
  await expect(page.getByText("Current reported ETAs")).toHaveCount(0);
});

test("ETA revision priority rows show the previous ETA and signed change", async ({
  page,
}) => {
  await installApiMocks(page, (payload) => {
    const response = readableEtaPresentationEnvelope(
      payload.question,
      payload.conversation_id || "eta-revision-readable",
    );
    response.answer = "REVISION TEST moved its reported ETA 45 minutes later.";
    response.plan = {
      ...response.plan,
      eta_watch_intent: "eta_revisions",
    };
    response.datasets = [
      {
        id: "table",
        columns: [
          { field: "row_id", label: "Row", data_type: "string" },
          { field: "vessel_label", label: "Vessel", data_type: "string" },
          {
            field: "reported_eta_utc",
            label: "Reported ETA",
            data_type: "datetime",
          },
          {
            field: "previous_reported_eta_utc",
            label: "Previous ETA",
            data_type: "datetime",
          },
          {
            field: "eta_change_minutes",
            label: "ETA revision",
            data_type: "number",
            unit: "min",
          },
        ],
        rows: [
          {
            row_id: "eta-revision-1",
            vessel_label: "REVISION TEST",
            reported_eta_utc: "2026-07-28T15:45:00Z",
            previous_reported_eta_utc: "2026-07-28T15:00:00Z",
            eta_change_minutes: 45,
            eta_change_observed_at_utc: "2026-07-28T13:25:00Z",
          },
        ],
        row_count: 1,
      },
    ];
    response.visualizations = [];
    response.operational_brief = {
      ...response.operational_brief!,
      intent: "eta_revisions",
      headline: "Material vessel-reported ETA revisions identified.",
      matched_count: 1,
      displayed_count: 1,
      exceptions: [
        {
          code: "eta_changed",
          count: 1,
          summary: "One vessel crossed the revision threshold.",
        },
      ],
      prioritized_items: [
        {
          row_id: "eta-revision-1",
          vessel_label: "REVISION TEST",
          priority: "attention",
          status: "eta_changed",
          reason: "The vessel-reported ETA crossed the threshold.",
          actions: ["inspect_eta_changes"],
        },
      ],
    };
    return response;
  });

  await page.goto("/eta-delay");
  await runAnalysis(
    page,
    "Which vessels changed their reported ETA?",
    "REVISION TEST moved",
  );

  const priority = page.getByTestId("eta-priority-vessel");
  await expect(priority).toContainText("Previous ETA");
  await expect(priority).toContainText("28 Jul · 15:00 UTC");
  await expect(priority).toContainText("+45 min later");
});

test("ETA Watch keeps its readable answer hierarchy and disclosure keyboard-accessible on mobile", async ({
  page,
}) => {
  const question = queryCatalog["ETA & Delay"][0];
  await installApiMocks(page, (payload) =>
    readableEtaPresentationEnvelope(
      payload.question,
      payload.conversation_id || "eta-readable-answer-mobile",
    ),
  );
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/eta-delay");
  await runAnalysis(page, question, "Sweden-bound 12-hour shift watch");

  const orderedSections = [
    page.getByTestId("eta-answer-summary"),
    page.getByTestId("eta-key-stats"),
    page.getByTestId("eta-priority-vessels"),
    page.getByTestId("result-visualizations"),
    page.getByTestId("result-metadata"),
  ];
  const boxes = await Promise.all(
    orderedSections.map((locator) => locator.boundingBox()),
  );
  expect(boxes.every(Boolean)).toBe(true);
  expect(boxes.map((box) => box!.y)).toEqual(
    [...boxes].map((box) => box!.y).sort((a, b) => a - b),
  );

  const canonicalDisclosure = page.getByTestId(
    "canonical-response-disclosure",
  );
  const disclosureSummary = canonicalDisclosure.getByText(
    "Full response",
    { exact: true },
  );
  await disclosureSummary.focus();
  await expect(disclosureSummary).toBeFocused();
  await disclosureSummary.press("Enter");
  await expect(page.getByTestId("result-answer")).toBeVisible();
  await disclosureSummary.press("Enter");
  await expect(page.getByTestId("result-answer")).toBeHidden();

  const viewport = await page.evaluate(() => ({
    client: document.documentElement.clientWidth,
    scroll: document.documentElement.scrollWidth,
  }));
  expect(viewport.scroll).toBeLessThanOrEqual(viewport.client);
  await expectNoSeriousA11yViolations(page);
});

test("ETA Watch keeps all ten returned inbound vessels visible in the operational brief", async ({
  page,
}) => {
  const vesselNames = Array.from(
    { length: 10 },
    (_, index) => `BALTIC WATCH ${String(index + 1).padStart(2, "0")}`,
  );
  const ports = [
    ["Stockholm", "SESTO"],
    ["Gothenburg", "SEGOT"],
    ["Nynäshamn", "SENYN"],
    ["Malmö", "SEMMA"],
    ["Trelleborg", "SETRG"],
  ] as const;
  const rows = vesselNames.map((vesselName, index) => {
    const [destinationName, destinationLocode] = ports[index % ports.length];
    return {
      row_id: `eta-next-ten-${index + 1}`,
      mmsi: String(265100001 + index),
      vessel_label: vesselName,
      vessel_name: vesselName,
      destination_name: destinationName,
      destination_locode: destinationLocode,
      reported_eta_utc: `2026-07-28T${String(13 + index).padStart(2, "0")}:00:00Z`,
      latitude: 55.4 + index * 0.2,
      longitude: 12.1 + index * 0.3,
      speed_kn: 7.5 + index * 0.4,
      observation_time_utc: "2026-07-28T11:55:00Z",
    };
  });
  const etaDataset: Dataset = {
    id: "table",
    columns: [
      { field: "row_id", label: "Row", data_type: "string" },
      { field: "mmsi", label: "MMSI", data_type: "string" },
      { field: "vessel_label", label: "Vessel", data_type: "string" },
      {
        field: "destination_name",
        label: "Destination",
        data_type: "string",
      },
      {
        field: "destination_locode",
        label: "UN/LOCODE",
        data_type: "string",
      },
      {
        field: "reported_eta_utc",
        label: "Reported ETA",
        data_type: "datetime",
      },
      { field: "latitude", label: "Latitude", data_type: "number" },
      { field: "longitude", label: "Longitude", data_type: "number" },
      { field: "speed_kn", label: "Speed", data_type: "number", unit: "kn" },
      {
        field: "observation_time_utc",
        label: "Observed",
        data_type: "datetime",
      },
    ],
    rows,
    row_count: rows.length,
  };
  const etaTimeline: VisualizationSpec = {
    id: "eta-next-ten-timeline",
    kind: "timeline",
    title: "Next vessel-reported ETAs",
    dataset_id: etaDataset.id,
    table_fallback_dataset_id: etaDataset.id,
    row_id_field: "row_id",
    accessible_summary:
      "Ten validated vessel-reported ETAs ordered chronologically.",
    citations: ["aisstream-snapshot"],
    time_field: "reported_eta_utc",
    label_field: "vessel_label",
    detail_fields: ["destination_name", "destination_locode", "speed_kn"],
  };

  await installApiMocks(page, (payload) => {
    const response = answerEnvelope(
      payload.question,
      payload.conversation_id || "eta-next-ten-test",
    );
    response.answer =
      "Twelve source candidates were observed; ten validated vessel signals are returned. Showing 10 of 12.";
    response.plan = {
      ...response.plan,
      operation: "vessel_eta",
      source_scope: "aisstream",
      eta_watch_intent: "inbound_watchlist",
      horizon_hours: 48,
      limit: 10,
    };
    response.datasets = [etaDataset];
    response.visualizations = [etaTimeline];
    response.trace = {
      ...response.trace,
      operation: "vessel_eta",
      sources: ["aisstream"],
      dataset_rows: 10,
      visualization_decision: "timeline",
    };
    response.evidence = [
      {
        id: "aisstream-snapshot",
        source_type: "traffic_event",
        title: "AISStream observation snapshot",
        url: "https://aisstream.io/",
        metadata: {},
      },
    ];
    response.operational_brief = {
      intent: "inbound_watchlist",
      headline: "Inbound watchlist ordered by vessel-reported ETA.",
      window_start_utc: "2026-07-28T12:00:00Z",
      window_end_utc: "2026-07-30T12:00:00Z",
      matched_count: 12,
      displayed_count: 10,
      prioritized_items: rows.slice(0, 5).map((row) => ({
        row_id: row.row_id,
        vessel_label: row.vessel_label,
        priority: "monitor" as const,
        status: "due_soon",
        reason:
          "The validated vessel-reported ETA falls inside the requested watch window.",
        actions: ["locate_vessel" as const],
      })),
      exceptions: [
        {
          code: "due_soon",
          count: 10,
          summary: "Ten validated reported ETAs fall inside the watch window.",
        },
      ],
      source_health: "live",
      source_observed_at: "2026-07-28T11:55:00Z",
      coverage:
        "Vessel-reported AIS observations; not an official arrival schedule.",
    };
    return response;
  });

  await page.goto("/eta-delay");
  await runAnalysis(
    page,
    adHocNextTenQuestion,
    "Showing 10 of 12",
  );

  const inboundList = page.getByTestId("validated-inbound-list");
  await expect(inboundList).toBeVisible();
  await expect(inboundList.getByTestId("validated-inbound-row")).toHaveCount(10);
  for (let index = 0; index < vesselNames.length; index += 1) {
    const row = inboundList.getByTestId("validated-inbound-row").nth(index);
    await expect(row).toBeVisible();
    await expect(row).toContainText(vesselNames[index]);
    await expect(row).toContainText(ports[index % ports.length][0]);
  }
  await expect(page.getByTestId("eta-priority-vessels")).toHaveCount(0);
  await expectNoSeriousA11yViolations(page);
});

test("ETA Watch shows one current vessel, eight stale candidates, and one missing next-10 slot", async ({
  page,
}) => {
  const question = adHocNextTenQuestion;
  const currentRow = {
    row_id: "eta-current-1",
    mmsi: "255915983",
    vessel_label: "CURRENT ETA VESSEL",
    vessel_name: "CURRENT ETA VESSEL",
    destination_name: "Malmö",
    destination_locode: "SEMMA",
    reported_eta_utc: "2026-07-28T13:30:00Z",
    latitude: 55.8969,
    longitude: 12.7358,
    speed_kn: 11.3,
    observation_time_utc: "2026-07-28T12:10:53Z",
  };
  const currentDataset: Dataset = {
    id: "table",
    columns: [
      { field: "row_id", label: "Row", data_type: "string" },
      { field: "mmsi", label: "MMSI", data_type: "string" },
      { field: "vessel_label", label: "Vessel", data_type: "string" },
      {
        field: "destination_name",
        label: "Destination",
        data_type: "string",
      },
      {
        field: "destination_locode",
        label: "UN/LOCODE",
        data_type: "string",
      },
      {
        field: "reported_eta_utc",
        label: "Reported ETA",
        data_type: "datetime",
      },
      { field: "latitude", label: "Latitude", data_type: "number" },
      { field: "longitude", label: "Longitude", data_type: "number" },
      { field: "speed_kn", label: "Speed", data_type: "number", unit: "kn" },
      {
        field: "observation_time_utc",
        label: "Observed",
        data_type: "datetime",
      },
    ],
    rows: [currentRow],
    row_count: 1,
  };
  const stalePorts = [
    ["Stockholm", "SESTO"],
    ["Malmö", "SEMMA"],
    ["Gothenburg", "SEGOT"],
    ["Trelleborg", "SETRG"],
    ["Gothenburg", "SEGOT"],
    ["Stockholm", "SESTO"],
    ["Stockholm", "SESTO"],
    ["Gothenburg", "SEGOT"],
  ] as const;
  const staleRows = stalePorts.map(
    ([destinationName, destinationLocode], index) => ({
      row_id: `eta-stale-${index + 1}`,
      mmsi: String(265500640 + index),
      vessel_label: `STALE ETA ${String(index + 1).padStart(2, "0")}`,
      vessel_name: `STALE ETA ${String(index + 1).padStart(2, "0")}`,
      destination_name: destinationName,
      destination_locode: destinationLocode,
      last_reported_eta_utc: `2026-07-${index < 6 ? "28" : "29"}T${String(
        14 + index,
      ).padStart(2, "0")}:00:00Z`,
      last_eta_observed_at_utc: `2026-07-28T${String(
        11 - Math.floor(index / 4),
      ).padStart(2, "0")}:${String(48 - index * 3).padStart(2, "0")}:00Z`,
      eta_observation_age_minutes: 12 + index * 15,
      validation_reason:
        "The latest ETA transmission is older than the 10-minute current-publication limit.",
    }),
  );
  const freshnessDataset: Dataset = {
    id: "eta_freshness_candidates",
    columns: [
      { field: "row_id", label: "Row", data_type: "string" },
      { field: "mmsi", label: "MMSI", data_type: "string" },
      { field: "vessel_label", label: "Vessel", data_type: "string" },
      {
        field: "destination_name",
        label: "Destination",
        data_type: "string",
      },
      {
        field: "destination_locode",
        label: "UN/LOCODE",
        data_type: "string",
      },
      {
        field: "last_reported_eta_utc",
        label: "Last ETA—not current",
        data_type: "datetime",
      },
      {
        field: "last_eta_observed_at_utc",
        label: "Last ETA received",
        data_type: "datetime",
      },
      {
        field: "eta_observation_age_minutes",
        label: "ETA report age",
        data_type: "number",
        unit: "min",
      },
      {
        field: "validation_reason",
        label: "Publication status",
        data_type: "string",
      },
    ],
    rows: staleRows,
    row_count: staleRows.length,
  };
  let submitted = 0;

  await installApiMocks(page, (payload) => {
    submitted += 1;
    const response = answerEnvelope(
      payload.question,
      payload.conversation_id || "eta-freshness-test",
    );
    response.answer =
      `Live ETA refresh ${submitted}: 1 current reported ETA is available; ` +
      "8 destination-matched candidates are too old to publish as current, " +
      "and 1 requested slot has no matching source signal.";
    response.plan = {
      ...response.plan,
      operation: "vessel_eta",
      source_scope: "aisstream",
      eta_watch_intent: "inbound_watchlist",
      horizon_hours: 48,
      limit: 10,
    };
    response.datasets = [currentDataset, freshnessDataset];
    response.visualizations = [];
    response.trace = {
      ...response.trace,
      operation: "vessel_eta",
      sources: ["aisstream"],
      dataset_rows: 1,
      visualization_decision: "timeline",
    };
    response.operational_brief = {
      intent: "inbound_watchlist",
      headline: "Inbound watchlist ordered by vessel-reported ETA.",
      window_start_utc: "2026-07-28T12:00:00Z",
      window_end_utc: "2026-07-30T12:00:00Z",
      matched_count: 1,
      displayed_count: 1,
      prioritized_items: [
        {
          row_id: currentRow.row_id,
          vessel_label: currentRow.vessel_label,
          priority: "monitor",
          status: "due_soon",
          reason:
            "The validated vessel-reported ETA falls inside the requested watch window.",
          actions: ["locate_vessel"],
        },
      ],
      exceptions: [
        {
          code: "due_soon",
          count: 1,
          summary: "One validated reported ETA falls inside the watch window.",
        },
      ],
      source_health: "live",
      source_observed_at: "2026-07-28T12:13:06Z",
      coverage:
        "Vessel-reported AIS observations; not an official arrival schedule.",
    };
    return response;
  });

  await page.goto("/eta-delay");
  await runAnalysis(page, question, "1 current reported ETA is available");

  const currentList = page.getByTestId("validated-inbound-list");
  await expect(currentList).toContainText("Current reported ETAs");
  await expect(currentList).toContainText("1 current of 10 requested");
  await expect(currentList.getByTestId("validated-inbound-row")).toHaveCount(1);
  await expect(currentList).toContainText("CURRENT ETA VESSEL");

  const freshnessCandidates = page.getByTestId("eta-freshness-candidates");
  await expect(freshnessCandidates).toContainText(
    "Awaiting a fresh ETA broadcast",
  );
  await expect(freshnessCandidates).toContainText("8 excluded signals");
  await expect(
    freshnessCandidates.getByTestId("eta-freshness-candidate-row"),
  ).toHaveCount(8);
  for (let index = 0; index < staleRows.length; index += 1) {
    const row = freshnessCandidates
      .getByTestId("eta-freshness-candidate-row")
      .nth(index);
    await expect(row).toBeVisible();
    await expect(row).toContainText(staleRows[index].vessel_label);
    await expect(row).toContainText(stalePorts[index][0]);
    await expect(row).toContainText(
      `${staleRows[index].eta_observation_age_minutes} min`,
    );
    await expect(row).toContainText("Last ETA—not current");
    await expect(row).not.toContainText(staleRows[index].last_reported_eta_utc);
    await expect(row).not.toContainText(
      staleRows[index].last_eta_observed_at_utc,
    );
  }
  await expect(
    freshnessCandidates.getByTestId("eta-freshness-candidate-row").first(),
  ).toContainText("28 Jul · 14:00 UTC");
  await expect(freshnessCandidates).toContainText(
    "No additional matching AIS vessel signal is present for 1 requested slot",
  );

  await page.getByRole("button", { name: "Refresh live signals" }).click();
  await expect.poll(() => submitted).toBe(2);
  await expect(page.getByTestId("result-answer")).toContainText(
    "Live ETA refresh 2",
  );
  await expectNoSeriousA11yViolations(page);
});

test("ETA timeline plots only valid timestamps on separate vessel lanes", async ({
  page,
}) => {
  const validRows = Array.from({ length: 3 }, (_, index) => ({
    row_id: `eta-valid-${index + 1}`,
    vessel_label: `VALID ETA ${index + 1}`,
    destination_name: ["Malmö", "Gothenburg", "Stockholm"][index],
    reported_eta_utc: `2026-07-28T${String(13 + index).padStart(2, "0")}:30:00Z`,
    is_missing_eta: false,
  }));
  const missingRows = Array.from({ length: 17 }, (_, index) => ({
    row_id: `eta-missing-${index + 1}`,
    vessel_label: `MISSING ETA ${index + 1}`,
    destination_name: "Sweden",
    reported_eta_utc: null,
    is_missing_eta: true,
  }));
  const columns: Dataset["columns"] = [
    { field: "row_id", label: "Row", data_type: "string" },
    { field: "vessel_label", label: "Vessel", data_type: "string" },
    {
      field: "destination_name",
      label: "Destination",
      data_type: "string",
    },
    {
      field: "reported_eta_utc",
      label: "Reported ETA",
      data_type: "datetime",
    },
    {
      field: "is_missing_eta",
      label: "Missing ETA",
      data_type: "boolean",
    },
  ];
  const statusDataset: Dataset = {
    id: "table",
    columns,
    rows: [...validRows, ...missingRows],
    row_count: 20,
  };
  const timelineDataset: Dataset = {
    id: "eta_timeline",
    columns,
    rows: validRows,
    row_count: validRows.length,
  };
  const timeline: VisualizationSpec = {
    id: "eta_watch_timeline",
    kind: "timeline",
    title: "Due-soon vessel-reported ETAs",
    dataset_id: "eta_timeline",
    table_fallback_dataset_id: "table",
    row_id_field: "row_id",
    accessible_summary:
      "Only valid vessel-reported ETAs are plotted on separate vessel lanes.",
    citations: ["aisstream-snapshot"],
    time_field: "reported_eta_utc",
    label_field: "vessel_label",
    lane_field: "vessel_label",
    detail_fields: ["destination_name"],
  };

  await installApiMocks(page, (payload) => {
    const response = answerEnvelope(
      payload.question,
      payload.conversation_id || "eta-readable-timeline",
    );
    response.answer =
      "Three valid vessel-reported ETAs are plotted; seventeen missing ETA signals remain in the status table.";
    response.plan = {
      ...response.plan,
      operation: "vessel_eta",
      source_scope: "aisstream",
      eta_watch_intent: "shift_handover",
      horizon_hours: 12,
      limit: 20,
    };
    response.datasets = [statusDataset, timelineDataset];
    response.visualizations = [timeline];
    response.trace = {
      ...response.trace,
      operation: "vessel_eta",
      sources: ["aisstream"],
      dataset_rows: 20,
      visualization_decision: "timeline",
    };
    return response;
  });

  await page.goto("/eta-delay");
  await runAnalysis(
    page,
    queryCatalog["ETA & Delay"][0],
    "Three valid vessel-reported ETAs are plotted",
  );

  const visualization = page.locator(
    'section[aria-labelledby="visual-title-eta_watch_timeline"]',
  );
  await expect(visualization).toContainText(
    "Chronological event rail · 3 rows",
  );
  await expect(visualization.getByTestId("timeline-scope-notice")).toContainText(
    "3 valid ETAs plotted",
  );
  await expect(visualization.getByTestId("timeline-scope-notice")).toContainText(
    "17 rows without a valid timestamp",
  );
  await expect(visualization.locator(".chart-canvas canvas")).toBeVisible();
  await visualization
    .getByRole("button", { name: "Inspect chart points with the keyboard" })
    .click();
  await expect(visualization.getByText(/Point 1 of 3\./)).toHaveCount(1);
  await expect(visualization).not.toContainText("MISSING ETA 1");
  await expectNoSeriousA11yViolations(page);
});

test("ETA timeline with no valid timestamps shows its verified table fallback", async ({
  page,
}) => {
  const rows = [
    {
      row_id: "eta-invalid-1",
      vessel_label: "MISSING ETA ONE",
      reported_eta_utc: null,
    },
    {
      row_id: "eta-invalid-2",
      vessel_label: "INVALID ETA TWO",
      reported_eta_utc: "not-a-timestamp",
    },
  ];
  const columns: Dataset["columns"] = [
    { field: "row_id", label: "Row", data_type: "string" },
    { field: "vessel_label", label: "Vessel", data_type: "string" },
    {
      field: "reported_eta_utc",
      label: "Reported ETA",
      data_type: "datetime",
    },
  ];
  const fallback: Dataset = {
    id: "table",
    columns,
    rows,
    row_count: rows.length,
  };
  const invalidTimeline: Dataset = {
    id: "legacy_eta_timeline",
    columns,
    rows,
    row_count: rows.length,
  };
  const timeline: VisualizationSpec = {
    id: "legacy_eta_watch_timeline",
    kind: "timeline",
    title: "Vessel-reported ETA schedule",
    dataset_id: "legacy_eta_timeline",
    table_fallback_dataset_id: "table",
    row_id_field: "row_id",
    accessible_summary: "Only valid vessel-reported ETA timestamps are plotted.",
    citations: [],
    time_field: "reported_eta_utc",
    label_field: "vessel_label",
    lane_field: "vessel_label",
    detail_fields: [],
  };

  await installApiMocks(page, (payload) => answerEnvelope(
    payload.question,
    payload.conversation_id || "eta-empty-timeline",
    {
      answer:
        "No valid vessel-reported ETA timestamp is available; verified source rows remain listed.",
      datasets: [fallback, invalidTimeline],
      visualizations: [timeline],
    },
  ));

  await page.goto("/eta-delay");
  await runAnalysis(
    page,
    queryCatalog["ETA & Delay"][6],
    "No valid vessel-reported ETA timestamp is available",
  );

  const visualization = page.locator(
    'section[aria-labelledby="visual-title-legacy_eta_watch_timeline"]',
  );
  await expect(visualization.getByTestId("timeline-scope-notice")).toContainText(
    "0 valid ETAs plotted",
  );
  const tableFallback = visualization.getByTestId("timeline-table-fallback");
  await expect(tableFallback).toContainText(
    "No valid timestamps can be plotted",
  );
  await expect(tableFallback).toContainText("MISSING ETA ONE");
  await expect(tableFallback).toContainText("INVALID ETA TWO");
  await expect(visualization.locator(".chart-canvas canvas")).toHaveCount(0);
});

test("filters and Evidence Top K are sent through the unchanged query request", async ({
  page,
}) => {
  let submitted: QueryRequestPayload | undefined;
  await page.unroute("**/api/v2/query");
  await page.route("**/api/v2/query", async (route, request) => {
    submitted = request.postDataJSON() as QueryRequestPayload;
    return json(
      route,
      answerEnvelope(
        submitted.question,
        submitted.conversation_id || "conversation-filter",
      ),
    );
  });

  await page.goto("/traffic-monitoring");
  await page.getByTestId("filter-button").click();
  const filterDialog = page.getByRole("dialog", { name: "Analysis filters" });
  await filterDialog.getByLabel("Port").fill("SEGOT");
  await filterDialog.getByLabel("Vessel type").fill("tanker");
  await filterDialog.getByLabel("Vessel name").fill("NORDIC STAR");
  await filterDialog.getByLabel("MMSI").fill("230123456");
  await filterDialog.getByLabel("IMO").fill("9074729");
  await filterDialog.getByLabel("Include anomaly filter").check();
  await filterDialog.getByLabel("Apply an explicit date range").check();
  await filterDialog.getByLabel("From").fill("2022-03-01");
  await filterDialog.getByLabel("To").fill("2022-03-31");
  await filterDialog.getByRole("button", { name: "Apply" }).click();
  await expect(page.getByTestId("filter-button")).toHaveText("Filters (7)");

  await page.getByTestId("settings-button").click();
  const settingsDialog = page.getByRole("dialog", {
    name: "Workspace settings",
  });
  await expect(
    settingsDialog.getByLabel("Screenshot mode"),
  ).toHaveCount(0);
  await expect(
    settingsDialog.getByRole("heading", { name: "Runtime details" }),
  ).toHaveCount(0);
  await settingsDialog.getByRole("slider").fill("8");
  await expect(settingsDialog.getByText("Evidence top K: 8")).toBeVisible();
  await settingsDialog.getByRole("button", { name: "Done" }).click();

  await runAnalysis(page, "Apply every explicit filter.");
  expect(submitted).toMatchObject({
    question: "Apply every explicit filter.",
    top_k_evidence: 8,
    filters: {
      port: "SEGOT",
      vessel_type: "tanker",
      vessel_name: "NORDIC STAR",
      mmsi: "230123456",
      imo: "9074729",
      anomaly: true,
      date_from: "2022-03-01",
      date_to: "2022-03-31",
    },
  });
  expect(submitted?.conversation_id).toMatch(/^web-traffic-/);
});

test("conversation IDs are page-scoped, New analysis isolates context, and history restores it", async ({
  page,
}) => {
  const requests: QueryRequestPayload[] = [];
  await page.unroute("**/api/v2/query");
  await page.route("**/api/v2/query", async (route, request) => {
    const payload = request.postDataJSON() as QueryRequestPayload;
    requests.push(payload);
    return json(
      route,
      answerEnvelope(
        payload.question,
        payload.conversation_id || "conversation-fallback",
      ),
    );
  });

  await page.goto("/analysis");
  await runAnalysis(page, "First analysis question");
  await runAnalysis(page, "Follow-up analysis question");
  expect(requests[1].conversation_id).toBe(requests[0].conversation_id);

  await page.getByRole("link", { name: "Traffic Monitoring" }).click();
  await expect(page).toHaveURL(/\/traffic-monitoring$/);
  await expect(
    page.getByRole("heading", { level: 1, name: "Traffic Monitoring" }),
  ).toBeVisible();
  await runAnalysis(page, "Traffic workspace question");
  expect(requests[2].conversation_id).not.toBe(requests[0].conversation_id);

  await page.getByRole("button", { name: "New analysis" }).click();
  await expect(page).toHaveURL(/\/analysis$/);
  await expect(page.getByTestId("query-input")).toHaveValue("");
  await expect(page.getByTestId("result-answer")).toHaveCount(0);
  await runAnalysis(page, "Fresh analysis question");
  expect(requests[3].conversation_id).not.toBe(requests[0].conversation_id);

  await page
    .getByRole("button", { name: /Traffic workspace question/ })
    .click();
  await expect(page).toHaveURL(/\/traffic-monitoring$/);
  await expect(page.getByTestId("query-input")).toHaveValue(
    "Traffic workspace question",
  );
  await expect(page.getByTestId("result-answer")).toContainText(
    "Traffic workspace question",
  );

  await page
    .getByRole("button", { name: /Follow-up analysis question/ })
    .click();
  await expect(page).toHaveURL(/\/analysis$/);
  await expect(page.getByTestId("query-input")).toHaveValue(
    "Follow-up analysis question",
  );
  await expect(page.getByTestId("result-answer")).toContainText(
    "Follow-up analysis question",
  );
  await runAnalysis(page, "Follow-up after restoring history");
  expect(requests[4].conversation_id).toBe(requests[0].conversation_id);
});

test("Overview recent analyses restore the saved route, result, and conversation", async ({
  page,
}) => {
  const requests: QueryRequestPayload[] = [];
  await page.unroute("**/api/v2/query");
  await page.route("**/api/v2/query", async (route, request) => {
    const payload = request.postDataJSON() as QueryRequestPayload;
    requests.push(payload);
    return json(
      route,
      answerEnvelope(
        payload.question,
        payload.conversation_id || "conversation-fallback",
      ),
    );
  });

  const etaQuestion =
    "Which AIS-visible vessels reporting Swedish destinations are due in the next 6 hours?";
  await page.goto("/eta-delay");
  await runAnalysis(page, etaQuestion);

  await mockOverviewCapabilities(page);
  await page.goto("/overview");
  await page.reload();
  await expect(page.getByTestId("overview-command-log")).toHaveCount(0);
  const recentAnalyses = page.locator(".situation-activity-tape");
  await expect(recentAnalyses).toContainText(etaQuestion);
  await expect(recentAnalyses).toContainText("ETA Watch");
  await page.getByTestId("overview-provenance-button").click();
  const provenanceDialog = page.getByRole("dialog", {
    name: "Data provenance",
  });
  await expect(provenanceDialog).toBeVisible();
  await expect(
    provenanceDialog.getByRole("table", { name: "Dataset inventory" }),
  ).toBeVisible();
  await expect(provenanceDialog).toContainText("Arrivals Daily");
  await expect(provenanceDialog).toContainText("103,851");
  await expect(provenanceDialog).toContainText("Events");
  await expect(provenanceDialog).toContainText(
    "2021-01-01 to 2022-04-30",
  );
  await page.keyboard.press("Escape");
  await expect(provenanceDialog).toBeHidden();

  await page
    .getByRole("button", {
      name: `Restore analysis: ${etaQuestion}`,
    })
    .click();

  await expect(page).toHaveURL(/\/eta-delay$/);
  await expect(
    page.getByRole("heading", { level: 1, name: "ETA Watch" }),
  ).toBeVisible();
  await expect(page.getByTestId("query-input")).toHaveValue(etaQuestion);
  await expect(page.getByTestId("result-answer")).toContainText(etaQuestion);

  await runAnalysis(page, "Show the reported ETA again.");
  expect(requests[1].conversation_id).toBe(requests[0].conversation_id);
});

test("all visualization families render in canonical order with tables and explicit omission", async ({
  page,
}) => {
  const datasets: Dataset[] = [
    {
      id: "kpi",
      columns: [
        {
          field: "total",
          label: "Total",
          data_type: "integer",
          unit: "vessels",
        },
      ],
      rows: [{ total: 54 }],
      row_count: 1,
    },
    lineDataset,
    {
      id: "forecast",
      columns: [
        { field: "date", label: "Date", data_type: "datetime" },
        {
          field: "predicted",
          label: "Predicted",
          data_type: "number",
          unit: "index",
        },
        {
          field: "lower",
          label: "Lower",
          data_type: "number",
          unit: "index",
        },
        {
          field: "upper",
          label: "Upper",
          data_type: "number",
          unit: "index",
        },
        {
          field: "actual",
          label: "Actual",
          data_type: "number",
          unit: "index",
        },
      ],
      rows: [
        {
          date: "2022-04-01",
          predicted: 1.1,
          lower: 0.8,
          upper: 1.4,
          actual: 1,
        },
        {
          date: "2022-04-08",
          predicted: 1.2,
          lower: 0.9,
          upper: 1.5,
          actual: null,
        },
      ],
      row_count: 2,
    },
    {
      id: "distribution",
      columns: [
        {
          field: "dwell",
          label: "Dwell",
          data_type: "number",
          unit: "minutes",
        },
      ],
      rows: [
        { dwell: 20 },
        { dwell: 35 },
        { dwell: 50 },
        { dwell: 85 },
        { dwell: 120 },
      ],
      row_count: 5,
    },
    {
      id: "heatmap",
      columns: [
        { field: "weekday", label: "Weekday", data_type: "string" },
        { field: "hour", label: "Hour", data_type: "string" },
        {
          field: "count",
          label: "Count",
          data_type: "integer",
          unit: "vessels",
        },
      ],
      rows: [
        { weekday: "Monday", hour: "00:00", count: 2 },
        { weekday: "Friday", hour: "12:00", count: 5 },
      ],
      row_count: 2,
    },
    {
      id: "positions",
      columns: [
        {
          field: "latitude",
          label: "Latitude",
          data_type: "number",
          unit: "degrees",
        },
        {
          field: "longitude",
          label: "Longitude",
          data_type: "number",
          unit: "degrees",
        },
        { field: "vessel", label: "Vessel", data_type: "string" },
      ],
      rows: [
        { latitude: 57.7, longitude: 11.9, vessel: "A" },
        { latitude: 57.8, longitude: 12, vessel: "B" },
      ],
      row_count: 2,
    },
    {
      id: "timeline",
      columns: [
        { field: "time", label: "Time", data_type: "datetime" },
        { field: "event", label: "Event", data_type: "string" },
        { field: "port", label: "Port", data_type: "string" },
      ],
      rows: [
        {
          time: "2022-03-01T08:00:00Z",
          event: "Arrival",
          port: "SEGOT",
        },
        {
          time: "2022-03-01T16:00:00Z",
          event: "Departure",
          port: "SEGOT",
        },
      ],
      row_count: 2,
    },
    {
      id: "records",
      columns: [
        { field: "port", label: "Port", data_type: "string" },
        {
          field: "value",
          label: "Value",
          data_type: "number",
          unit: "index",
        },
      ],
      rows: [
        { port: "SEGOT", value: 1.2 },
        { port: "LVVNT", value: 0.9 },
      ],
      row_count: 2,
    },
  ];

  const visualizations: VisualizationSpec[] = [
    {
      id: "total-kpi",
      kind: "kpi",
      title: "Arrival total",
      dataset_id: "kpi",
      table_fallback_dataset_id: "kpi",
      accessible_summary: "A total of 54 vessels.",
      citations: [],
      value_field: "total",
      label: "Arrivals",
      unit: "vessels",
    },
    lineVisualization,
    {
      id: "forecast-band",
      kind: "forecast",
      title: "Forecast with interval",
      dataset_id: "forecast",
      table_fallback_dataset_id: "forecast",
      accessible_summary:
        "Actual and predicted pressure with an eighty percent interval.",
      citations: [],
      date_field: "date",
      predicted_field: "predicted",
      lower_field: "lower",
      upper_field: "upper",
      actual_field: "actual",
      unit: "index",
    },
    {
      id: "dwell-box",
      kind: "distribution",
      chart_type: "boxplot",
      title: "Dwell distribution",
      dataset_id: "distribution",
      table_fallback_dataset_id: "distribution",
      accessible_summary: "A five-number dwell-time distribution.",
      citations: [],
      value_field: "dwell",
      unit: "minutes",
    },
    {
      id: "weekday-heatmap",
      kind: "heatmap",
      title: "Weekday by hour",
      dataset_id: "heatmap",
      table_fallback_dataset_id: "heatmap",
      accessible_summary: "Weekday rows and UTC hour columns.",
      citations: [],
      x_field: "hour",
      y_field: "weekday",
      value_field: "count",
      unit: "vessels",
    },
    {
      id: "position-map",
      kind: "map",
      title: "Vessel positions",
      dataset_id: "positions",
      table_fallback_dataset_id: "positions",
      accessible_summary:
        "Two historical vessel positions on an offline Natural Earth map.",
      citations: [],
      latitude_field: "latitude",
      longitude_field: "longitude",
      label_field: "vessel",
    },
    {
      id: "event-timeline",
      kind: "timeline",
      title: "Port-call timeline",
      dataset_id: "timeline",
      table_fallback_dataset_id: "timeline",
      accessible_summary: "Arrival followed by departure.",
      citations: [],
      time_field: "time",
      label_field: "event",
      detail_fields: ["port"],
    },
    {
      id: "records-table",
      kind: "table",
      title: "Validated records",
      dataset_id: "records",
      table_fallback_dataset_id: "records",
      accessible_summary: "Two validated port records.",
      citations: [],
      visible_fields: ["port", "value"],
    },
    {
      id: "no-current-chart",
      kind: "omitted",
      title: "Current traffic",
      dataset_id: null,
      table_fallback_dataset_id: null,
      accessible_summary: "No current traffic chart is available.",
      citations: [],
      reason_code: "stale_data",
      reason:
        "Historical coverage cannot support a current operational chart.",
    },
  ];

  await page.unroute("**/api/v2/query");
  await page.route("**/api/v2/query", async (route, request) => {
    const payload = request.postDataJSON() as QueryRequestPayload;
    return json(
      route,
      answerEnvelope(
        payload.question,
        payload.conversation_id || "visual-suite",
        {
          datasets,
          visualizations,
        },
      ),
    );
  });

  await page.goto("/analysis");
  await runAnalysis(page, "Render every validated visualization.");

  const stack = page.getByTestId("result-visualizations");
  await expect(stack.locator(".visualization")).toHaveCount(
    visualizations.length,
  );
  await expect(
    stack.locator(".visualization h3").allTextContents(),
  ).resolves.toEqual(visualizations.map((item) => item.title));
  await expect(stack.locator(".kpi-canvas")).toContainText("54");
  await expect(stack.locator(".chart-canvas")).toHaveCount(5);
  expect(await stack.locator(".chart-canvas canvas").count()).toBeGreaterThanOrEqual(
    5,
  );
  await expect(stack.locator(".map-canvas")).toHaveCount(1);
  await expect(
    stack.getByRole("table", { name: "Underlying analysis data" }),
  ).toHaveCount(1);
  await expect(stack.getByText("No visualization is available.")).toBeVisible();
  await expect(
    stack.getByText(
      "Historical coverage cannot support a current operational chart.",
    ),
  ).toBeVisible();

  await stack.getByRole("button", { name: "View data" }).first().click();
  await expect(
    stack.getByRole("table", { name: "Underlying analysis data" }),
  ).toHaveCount(2);
});

test("an empty visualization list and a canonical omission both remain explicit", async ({
  page,
}) => {
  await page.unroute("**/api/v2/query");
  let count = 0;
  await page.route("**/api/v2/query", async (route, request) => {
    count += 1;
    const payload = request.postDataJSON() as QueryRequestPayload;
    const result =
      count === 1
        ? answerEnvelope(payload.question, "empty-contract", {
            state: "NO_DATA",
            datasets: [],
            visualizations: [],
            answer: "No rows matched the requested historical scope.",
          })
        : answerEnvelope(payload.question, "omitted-contract", {
            state: "NO_CURRENT_DATA",
            datasets: [],
            visualizations: [
              {
                id: "omitted-current",
                kind: "omitted",
                title: "Current traffic unavailable",
                dataset_id: null,
                table_fallback_dataset_id: null,
                accessible_summary: "No current traffic chart is available.",
                citations: [],
                reason_code: "stale_data",
                reason:
                  "The latest historical observation is 2022-04-30.",
              },
            ],
            answer: "Current traffic is unavailable from the historical data.",
          });
    return json(route, result);
  });

  await page.goto("/analysis");
  await runAnalysis(
    page,
    "No matching rows",
    "No rows matched the requested historical scope.",
  );
  await expect(
    page.getByText("No visualization contract was returned."),
  ).toBeVisible();

  await runAnalysis(
    page,
    "What is happening right now?",
    "Current traffic is unavailable from the historical data.",
  );
  await expect(page.getByText("No visualization is available.")).toBeVisible();
  await expect(
    page.getByText("The latest historical observation is 2022-04-30."),
  ).toBeVisible();
});

test("chart and map engine failures expose notices and verified tables instead of blanks", async ({
  page,
}) => {
  await page.route(/\/assets\/echarts-[^/]+\.js(?:\?.*)?$/, (route) =>
    route.abort(),
  );
  await page.goto("/analysis");
  await runAnalysis(page, "Force chart fallback");
  await expect(page.getByRole("alert")).toContainText(
    "Visualization could not be rendered",
  );
  await expect(page.getByRole("alert")).toContainText(
    "chart engine could not be loaded",
  );
  await expect(
    page.getByRole("table", { name: "Underlying analysis data" }),
  ).toBeVisible();
  await expect(page.locator(".chart-canvas canvas")).toHaveCount(0);

  const mapResult = answerEnvelope("Force map fallback", "map-failure", {
    datasets: [
      {
        id: "positions",
        columns: [
          {
            field: "latitude",
            label: "Latitude",
            data_type: "number",
            unit: "degrees",
          },
          {
            field: "longitude",
            label: "Longitude",
            data_type: "number",
            unit: "degrees",
          },
          { field: "vessel", label: "Vessel", data_type: "string" },
        ],
        rows: [{ latitude: 57.7, longitude: 11.9, vessel: "A" }],
        row_count: 1,
      },
    ],
    visualizations: [
      {
        id: "positions-map",
        kind: "map",
        title: "Vessel positions",
        dataset_id: "positions",
        table_fallback_dataset_id: "positions",
        accessible_summary: "One historical vessel position.",
        citations: [],
        latitude_field: "latitude",
        longitude_field: "longitude",
        label_field: "vessel",
      },
    ],
  });
  await page.unroute("**/api/v2/query");
  await page.route("**/api/v2/query", (route) => json(route, mapResult));
  await page.route(/\/assets\/maplibre-[^/]+\.js(?:\?.*)?$/, (route) =>
    route.abort(),
  );
  await page.reload();
  await runAnalysis(page, "Force map fallback");
  await expect(page.getByRole("alert")).toContainText(
    "map engine could not be loaded",
  );
  await expect(
    page.getByRole("table", { name: "Underlying analysis data" }),
  ).toBeVisible();
  await expect(page.locator(".map-canvas")).toHaveCount(0);
});

test("dark-only settings ignore legacy light values and keyboard workflows retain focus traps", async ({
  page,
}) => {
  await page.goto("/analysis");
  await page.evaluate(() => {
    localStorage.setItem(
      "eagle-eye-ui-settings-v1",
      JSON.stringify({
        theme: "light",
        topK: 6,
        railCollapsed: false,
      }),
    );
  });
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await expect(page.locator("html")).toHaveCSS("color-scheme", "dark");

  await page.getByTestId("settings-button").click();
  const settings = page.getByRole("dialog", { name: "Workspace settings" });
  await expect(settings.getByTestId("theme-select")).toHaveCount(0);
  await expect(settings.getByText("Evidence top K: 6")).toBeVisible();
  await settings.getByRole("button", { name: "Done" }).click();
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  expect(
    await page.evaluate(() => {
      const value = localStorage.getItem("eagle-eye-ui-settings-v1");
      return value ? JSON.parse(value) : null;
    }),
  ).toEqual({ topK: 6, railCollapsed: false });

  await page.keyboard.press("/");
  await expect(page.getByTestId("query-input")).toBeFocused();
  await page.getByTestId("query-input").fill("Keyboard submitted question");
  await page.getByTestId("query-input").press("Control+Enter");
  await expect(page.getByTestId("result-answer")).toContainText(
    "Keyboard submitted question",
  );

  await expect(page.getByRole("heading", { name: "Ask a question" })).toBeVisible();
  await expect(page.getByTestId("query-input")).toBeVisible();
  await page.getByTestId("sample-library-button").focus();
  await page.getByTestId("sample-library-button").press("Enter");
  const dialog = page.getByRole("dialog", { name: "Sample query library" });
  await expect(dialog).toBeVisible();
  await expect(
    dialog.getByRole("button", { name: "Close", exact: true }),
  ).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();
  await expect(page.getByTestId("sample-library-button")).toBeFocused();
});

test("mobile order is readable answer, charts, metadata, then evidence with no horizontal overflow", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/analysis");
  await expect(page.getByTestId("nav-toggle")).toBeVisible();

  await page.getByTestId("nav-toggle").click();
  await expect(page.getByTestId("nav-rail")).toHaveClass(/nav-rail-open/);
  await expect(
    page.getByRole("button", { name: "Close navigation", exact: true }),
  ).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(page.getByTestId("nav-rail")).not.toHaveClass(/nav-rail-open/);
  await expect(page.getByTestId("nav-toggle")).toBeFocused();

  await runAnalysis(page, "Mobile ordered result");
  const boxes = await Promise.all(
    [
      page.getByTestId("global-answer-summary"),
      page.getByTestId("global-answer-detail"),
      page.getByTestId("result-visualizations"),
      page.getByTestId("result-metadata"),
      page.getByTestId("evidence-inspector"),
    ].map((locator) => locator.boundingBox()),
  );
  expect(boxes.every(Boolean)).toBe(true);
  const positions = boxes.map((box) => box!.y);
  expect(positions).toEqual([...positions].sort((a, b) => a - b));

  const viewport = await page.evaluate(() => ({
    client: document.documentElement.clientWidth,
    scroll: document.documentElement.scrollWidth,
  }));
  expect(viewport.scroll).toBeLessThanOrEqual(viewport.client);
  await expectNoSeriousA11yViolations(page);
});

test("results without evidence use the full result canvas without an empty inspector", async ({
  page,
}) => {
  await page.unroute("**/api/v2/query");
  await page.route("**/api/v2/query", async (route, request) => {
    const payload = request.postDataJSON() as QueryRequestPayload;
    return json(
      route,
      answerEnvelope(
        payload.question,
        payload.conversation_id || "conversation-no-evidence",
        { evidence: [] },
      ),
    );
  });

  await page.goto("/analysis");
  await runAnalysis(page, "Computed result without external evidence");

  await expect(page.getByTestId("evidence-inspector")).toHaveCount(0);
  await expect(page.locator(".result-grid")).toHaveClass(
    /result-grid-without-evidence/,
  );
  await expect(page.getByTestId("global-answer-detail")).toBeVisible();
  await expect(
    page.getByTestId("canonical-response-disclosure"),
  ).toHaveJSProperty("open", false);
  await expect(page.getByTestId("result-answer")).toBeHidden();
  await expect(page.getByTestId("result-visualizations")).toBeVisible();
});

test("legacy confidence metadata is not exposed or used to block a current result", async ({
  page,
}) => {
  let requestCount = 0;
  await page.unroute("**/api/v2/query");
  await page.route("**/api/v2/query", async (route, request) => {
    requestCount += 1;
    const payload = request.postDataJSON() as QueryRequestPayload;
    return json(
      route,
      answerEnvelope(payload.question, payload.conversation_id || "legacy", {
        confidence: "low",
        assurance: undefined,
        availability: undefined,
      }),
    );
  });

  await page.goto("/analysis");
  await runAnalysis(page, "Stored legacy result");
  await expect(page.getByText("Analysis result", { exact: true })).toBeVisible();
  await expect(page.locator(".legacy-assurance-notice")).toHaveCount(0);
  await expect(page.locator(".readable-answer-state")).toHaveCount(0);
  await expect(page.getByTestId("result-metadata")).not.toContainText(
    /confidence|assurance|verification|basis/i,
  );
  expect(requestCount).toBe(1);
});

test("successful generated-result surfaces use direct neutral presentation", async ({
  page,
}) => {
  await installApiMocks(page, (payload) =>
    answerEnvelope(
      payload.question,
      payload.conversation_id || "neutral-result",
      {
        answer: "The requested route duration is 2.76 hours.",
        assurance: {
          status: "verified",
          level: "high",
          basis: "direct_computation",
          reason: "A publication gate and assurance check passed.",
          checks: ["high_assurance_gate=passed"],
        },
        confidence: "low",
        caveats: [
          "This reconstructed estimate is partial and uses a proxy fallback.",
        ],
      },
    ),
  );

  await page.goto("/traffic-monitoring");
  await runAnalysis(
    page,
    "Estimate the reconstructed route duration with a proxy fallback",
    "The requested route duration is 2.76 hours.",
  );

  await expect(page.getByText("Analysis result", { exact: true })).toBeVisible();
  await expect(page.locator(".readable-answer-state")).toHaveCount(0);
  await expect(page.getByTestId("result-metadata")).not.toContainText(
    /assurance|confidence|verification|basis|caveat|publication/i,
  );
  await expect(page.getByTestId("global-answer-detail")).not.toContainText(
    /reconstructed|estimated|proxy|fallback|partial/i,
  );
  await expect(page.getByTestId("result-visualizations")).not.toContainText(
    /verified data|verification|assurance|confidence|caveat|publication/i,
  );
  await expect(page.locator(".result-details")).not.toContainText(
    /method\s*&\s*audit|assurance|confidence|caveat|publication/i,
  );
});

test("history v3 stores the exact request and refreshes a saved unavailable record once", async ({
  page,
}) => {
  const question = "Refresh this saved route analysis";
  const oldResult = answerEnvelope(question, "saved-conversation", {
    state: "ASSURANCE_UNAVAILABLE",
    answer: "Obsolete policy result must never be displayed.",
    datasets: [],
    visualizations: [],
    facts: [],
  });
  const savedRequest: QueryRequestPayload = {
    question,
    conversation_id: "saved-conversation",
    top_k_evidence: 7,
    filters: {
      port: "SEGOT",
      vessel_type: "cargo",
      anomaly: true,
      date_from: "2022-03-01",
      date_to: "2022-03-31",
    },
  };
  const savedRecord = {
    id: "saved-v3-record",
    question,
    createdAt: "2026-08-08T18:00:00.000Z",
    route: "/traffic-monitoring",
    conversationId: "saved-conversation",
    result: oldResult,
    request: savedRequest,
    schemaVersion: 3,
  };
  await page.addInitScript((record) => {
    localStorage.setItem(
      "eagle-eye-analysis-history-v3",
      JSON.stringify([record]),
    );
  }, savedRecord);

  let requestCount = 0;
  let submitted: QueryRequestPayload | undefined;
  await page.unroute("**/api/v2/query");
  await page.route("**/api/v2/query", async (route, request) => {
    requestCount += 1;
    submitted = request.postDataJSON() as QueryRequestPayload;
    await new Promise((resolve) => setTimeout(resolve, 75));
    return json(
      route,
      answerEnvelope(question, "saved-conversation", {
        answer: "The refreshed route result is available.",
      }),
    );
  });

  await page.goto("/analysis");
  expect(requestCount).toBe(0);
  await page
    .getByRole("button", { name: new RegExp(question) })
    .dblclick({ delay: 10 });

  await expect(page).toHaveURL(/\/traffic-monitoring$/);
  await expect.poll(() => requestCount).toBe(1);
  expect(submitted).toEqual(savedRequest);
  await expect(page.getByTestId("result-answer")).toContainText(
    "The refreshed route result is available.",
  );
  await expect(
    page.getByText("Obsolete policy result must never be displayed."),
  ).toHaveCount(0);
  await expect(page.getByTestId("query-input")).toHaveValue(question);

  await expect
    .poll(async () =>
      page.evaluate(() => {
        const records = JSON.parse(
          localStorage.getItem("eagle-eye-analysis-history-v3") || "[]",
        );
        return records[0];
      }),
    )
    .toMatchObject({
      id: "saved-v3-record",
      route: "/traffic-monitoring",
      conversationId: "saved-conversation",
      schemaVersion: 3,
      request: savedRequest,
      result: { state: "COMPUTED" },
    });
});

test("saved history routes are restricted to the fixed workspace allowlist", async ({
  page,
}) => {
  const question = "Restore a record with an untrusted route";
  const savedRecord = {
    id: "untrusted-route-record",
    question,
    createdAt: "2026-08-09T09:00:00.000Z",
    route: "https://example.invalid/redirect",
    conversationId: "untrusted-route-conversation",
    result: answerEnvelope(question, "untrusted-route-conversation"),
    request: {
      question,
      conversation_id: "untrusted-route-conversation",
      top_k_evidence: 5,
    },
    schemaVersion: 3,
  };
  await page.addInitScript((record) => {
    localStorage.setItem(
      "eagle-eye-analysis-history-v3",
      JSON.stringify([record]),
    );
  }, savedRecord);

  await page.goto("/analysis");
  await page.getByRole("button", { name: new RegExp(question) }).click();
  await expect(page).toHaveURL(/\/analysis$/);
  await expect(page.getByTestId("query-input")).toHaveValue(question);
  await expect(page.getByTestId("result-answer")).toContainText(question);
  await expect
    .poll(async () =>
      page.evaluate(() => {
        const records = JSON.parse(
          localStorage.getItem("eagle-eye-analysis-history-v3") || "[]",
        );
        return records[0]?.route;
      }),
    )
    .toBe("/analysis");
});

test("v2 unavailable history derives its request and leaves a normal error when refresh fails", async ({
  page,
}) => {
  const question = "Restore a v2 saved analysis";
  const oldResult = answerEnvelope(question, "v2-conversation", {
    state: "ASSURANCE_UNAVAILABLE",
    answer: "Old blocked response must remain hidden.",
    applied_scope: {
      ports: ["SEGOT"],
      date_from: "2022-03-01",
      date_to: "2022-03-31",
      vessel_type: "tanker",
      mmsi: "230123456",
    },
    datasets: [],
    visualizations: [],
    facts: [],
  });
  await page.addInitScript(({ result, savedQuestion }) => {
    localStorage.setItem(
      "eagle-eye-analysis-history-v2",
      JSON.stringify([
        {
          id: "saved-v2-record",
          question: savedQuestion,
          createdAt: "2026-08-08T17:00:00.000Z",
          route: "/traffic-monitoring",
          conversationId: "v2-conversation",
          result,
        },
      ]),
    );
    localStorage.setItem(
      "eagle-eye-ui-settings-v1",
      JSON.stringify({ topK: 8, railCollapsed: false }),
    );
  }, { result: oldResult, savedQuestion: question });

  let submitted: QueryRequestPayload | undefined;
  await page.unroute("**/api/v2/query");
  await page.route("**/api/v2/query", async (route, request) => {
    submitted = request.postDataJSON() as QueryRequestPayload;
    return json(route, { detail: "The service is temporarily unavailable." }, 503);
  });

  await page.goto("/analysis");
  await page.getByRole("button", { name: new RegExp(question) }).click();
  await expect(page).toHaveURL(/\/traffic-monitoring$/);
  await expect(page.getByRole("alert")).toContainText(
    "The service is temporarily unavailable.",
  );
  await expect(page.getByTestId("query-input")).toHaveValue(question);
  await expect(page.getByText("Old blocked response must remain hidden.")).toHaveCount(0);
  expect(submitted).toEqual({
    question,
    conversation_id: "v2-conversation",
    top_k_evidence: 8,
    filters: {
      port: "SEGOT",
      date_from: "2022-03-01",
      date_to: "2022-03-31",
      vessel_type: "tanker",
      mmsi: "230123456",
    },
  });
});

test("exports and Report Issue preserve the canonical result identifiers", async ({
  page,
}) => {
  let exportPayload: Record<string, unknown> | undefined;
  let feedbackPayload: Record<string, unknown> | undefined;
  await page.unroute("**/api/v2/exports");
  await page.route("**/api/v2/exports", async (route, request) => {
    exportPayload = request.postDataJSON() as Record<string, unknown>;
    return json(route, {
      export_id: "export-confirmed",
      format: "csv",
      dataset_id: lineDataset.id,
      row_count: 3,
      path: "/tmp/eagle-eye/arrivals-series.csv",
    });
  });
  await page.unroute("**/api/v2/feedback");
  await page.route("**/api/v2/feedback", async (route, request) => {
    feedbackPayload = request.postDataJSON() as Record<string, unknown>;
    return json(route, {
      feedback_id: "feedback-confirmed",
      status: "accepted",
    });
  });

  await page.goto("/analysis");
  await runAnalysis(page, "Export and review this result");
  const traceId = "trace-1";

  await page.getByText("Data & scope", { exact: true }).click();
  await page.getByRole("button", { name: "Export CSV" }).click();
  await expect(page.getByRole("status")).toContainText(
    "3 rows exported to /tmp/eagle-eye/arrivals-series.csv",
  );
  expect(exportPayload).toMatchObject({
    conversation_id: expect.any(String),
    turn_id: "turn-1",
    dataset_id: lineDataset.id,
    format: "csv",
  });

  await page
    .getByRole("button", {
      name: "Report a possible issue with this analysis",
    })
    .click();
  const report = page.getByRole("dialog", { name: "Report a possible issue" });
  await report
    .getByRole("textbox", { name: "Issue detail" })
    .fill("The displayed comparison needs review.");
  await report.getByRole("button", { name: "Send report" }).click();
  await expect(report.getByRole("status")).toHaveText(
    "The issue was recorded for review.",
  );
  expect(feedbackPayload).toEqual({
    trace_id: traceId,
    prompt: "Export and review this result",
    note: "The displayed comparison needs review.",
  });
});

test("Baltic Situation Sheet remains readable, touch-sized, overflow-free, and motion-safe on mobile", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await mockOverviewCapabilities(page);
  await page.goto("/overview");

  const overview = page.getByTestId("operational-overview");
  await expect(overview).toBeVisible();
  await expect(
    page.getByRole("heading", { level: 1, name: "Baltic archive footprint" }),
  ).toBeVisible();
  await expect(page.getByTestId("nav-rail")).not.toHaveClass(/nav-rail-open/);
  expect((await page.getByTestId("nav-rail").boundingBox())!.x).toBeLessThan(0);
  const touchTargets = [
    page.getByTestId("nav-toggle"),
    page.getByTestId("overview-enter-analysis"),
    page.getByTestId("overview-provenance-button"),
    page
      .getByTestId("overview-coverage-disclosure")
      .getByText("View coverage data", { exact: true }),
  ];
  for (const button of touchTargets) {
    const buttonBox = await button.boundingBox();
    expect(buttonBox).not.toBeNull();
    expect(buttonBox!.height).toBeGreaterThanOrEqual(44);
  }

  const mastheadBox = await overview
    .getByRole("heading", { level: 1, name: "Baltic archive footprint" })
    .boundingBox();
  const statusBox = await overview
    .locator('[aria-label="Workspace readiness"]')
    .boundingBox();
  const coverageBox = await overview
    .getByTestId("overview-historical-atlas")
    .boundingBox();
  const disclosureBox = await overview
    .getByTestId("overview-coverage-disclosure")
    .boundingBox();
  const recentBox = await overview
    .locator(".situation-activity-tape")
    .boundingBox();
  expect(mastheadBox).not.toBeNull();
  expect(statusBox).not.toBeNull();
  expect(coverageBox).not.toBeNull();
  expect(disclosureBox).not.toBeNull();
  expect(recentBox).not.toBeNull();
  expect(statusBox!.y).toBeGreaterThan(mastheadBox!.y);
  expect(coverageBox!.y).toBeGreaterThan(statusBox!.y);
  expect(disclosureBox!.y).toBeGreaterThan(coverageBox!.y);
  expect(recentBox!.y).toBeGreaterThan(disclosureBox!.y);

  const documentWidth = await page.evaluate(() => ({
    client: document.documentElement.clientWidth,
    scroll: document.documentElement.scrollWidth,
  }));
  expect(documentWidth.scroll).toBeLessThanOrEqual(documentWidth.client);
  expect(
    await page.evaluate(
      () => window.matchMedia("(prefers-reduced-motion: reduce)").matches,
    ),
  ).toBe(true);

  const motionOffenders = await overview.evaluate((root) =>
    Array.from(root.querySelectorAll("*"))
      .map((element) => {
        const style = getComputedStyle(element);
        const durations = `${style.animationDuration},${style.transitionDuration}`
          .split(",")
          .map((value) => {
            const trimmed = value.trim();
            if (trimmed.endsWith("ms")) return Number.parseFloat(trimmed);
            if (trimmed.endsWith("s"))
              return Number.parseFloat(trimmed) * 1000;
            return 0;
          });
        return {
          tag: element.tagName,
          animationIterationCount: style.animationIterationCount,
          maximumDurationMs: Math.max(0, ...durations),
        };
      })
      .filter(
        (entry) =>
          entry.animationIterationCount === "infinite" ||
          entry.maximumDurationMs > 20,
      )
      .slice(0, 10),
  );
  expect(motionOffenders).toEqual([]);

  for (const rejectedId of [
    "overview-map",
    "overview-source-spine",
    "overview-command-log",
    "overview-theatre",
    "overview-stage-nav",
  ]) {
    await expect(page.getByTestId(rejectedId)).toHaveCount(0);
  }
  await expect(overview.locator("canvas")).toHaveCount(0);
  await expect(overview.locator('[class*="card"]')).toHaveCount(0);

  const navToggle = page.getByTestId("nav-toggle");
  await navToggle.focus();
  await navToggle.click();
  const rail = page.getByTestId("nav-rail");
  await expect(rail).toHaveClass(/nav-rail-open/);
  await expect(rail.getByRole("button", { name: "Close navigation" })).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(rail).not.toHaveClass(/nav-rail-open/);
  await expect(navToggle).toBeFocused();
  await expectNoSeriousA11yViolations(page);
});

test("desktop and mobile shells have no serious accessibility violations", async ({
  page,
}) => {
  await page.goto("/overview");
  await expectNoSeriousA11yViolations(page);

  await page.goto("/analysis");
  await runAnalysis(page, "Accessible analysis result");
  await expectNoSeriousA11yViolations(page);

  await page.setViewportSize({ width: 1024, height: 1366 });
  await page.goto("/carbon-emissions");
  await expectNoSeriousA11yViolations(page);
});

test("dark-only Baltic Situation Sheet baselines cover ready and unavailable states", async ({
  page,
}) => {
  await page.clock.setFixedTime(new Date("2026-07-28T14:00:00Z"));
  const viewports = [
    { label: "desktop-1440x900", width: 1440, height: 900 },
    { label: "tablet-1024x1366", width: 1024, height: 1366 },
    { label: "mobile-390x844", width: 390, height: 844 },
  ] as const;

  for (const viewport of viewports) {
    await page.setViewportSize({
      width: viewport.width,
      height: viewport.height,
    });
    await page.unroute("**/api/v2/capabilities");
    await page.route("**/api/v2/capabilities", (route) =>
      json(route, overviewCapabilities),
    );
    await page.addInitScript(() => {
      localStorage.setItem(
        "eagle-eye-ui-settings-v1",
        JSON.stringify({
          theme: "light",
          topK: 5,
          railCollapsed: false,
        }),
      );
      localStorage.removeItem("eagle-eye-analysis-history-v2");
      localStorage.removeItem("eagle-eye-analysis-history-v3");
      sessionStorage.clear();
    });
    await page.goto("/overview");
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
    await expect(page.getByTestId("operational-overview")).toBeVisible();
    await expect(page.locator('[aria-label="Workspace readiness"]')).toContainText(
      "Live",
    );
    await expect(page.getByTestId("overview-historical-atlas")).toBeVisible();
    await expect(page).toHaveScreenshot(
      `overview-dark-${viewport.label}-ready.png`,
      {
        animations: "disabled",
        caret: "hide",
        fullPage: true,
        maxDiffPixels: 120,
      },
    );

    await page.unroute("**/api/v2/capabilities");
    await page.route("**/api/v2/capabilities", (route) =>
      json(route, { detail: "Capability service unavailable" }, 503),
    );
    await page.reload();
    await expect(page.locator('[aria-label="Workspace readiness"]')).toContainText(
      "Workspace status unavailable",
    );
    await expect(page.locator(".situation-atlas-fallback")).toBeVisible();
    await expect(page).toHaveScreenshot(
      `overview-dark-${viewport.label}-unavailable.png`,
      {
        animations: "disabled",
        caret: "hide",
        fullPage: true,
        maxDiffPixels: 120,
      },
    );
  }
});

test("approved dark-only populated and empty-state baselines cover all target viewports", async ({
  page,
}) => {
  await page.unroute("**/api/v2/query");
  await page.route("**/api/v2/query", async (route, request) => {
    const payload = request.postDataJSON() as QueryRequestPayload;
    const conversationId = payload.conversation_id || "visual-baseline";
    if (payload.question === "Empty-state baseline") {
      return json(
        route,
        answerEnvelope(payload.question, conversationId, {
          state: "NO_DATA",
          answer: "No rows matched the requested historical scope.",
          datasets: [],
          visualizations: [],
          facts: [],
          evidence: [],
          confidence: "not_applicable",
          caveats: ["No chart is shown because no validated rows were returned."],
        }),
      );
    }
    return json(route, answerEnvelope(payload.question, conversationId));
  });

  const viewports = [
    { label: "desktop-1440x900", width: 1440, height: 900 },
    { label: "tablet-1024x1366", width: 1024, height: 1366 },
    { label: "mobile-390x844", width: 390, height: 844 },
  ] as const;

  for (const viewport of viewports) {
    await page.setViewportSize({
      width: viewport.width,
      height: viewport.height,
    });
    await page.goto("/analysis");
    await page.evaluate(() => {
      localStorage.setItem(
        "eagle-eye-ui-settings-v1",
        JSON.stringify({
          theme: "light",
          topK: 5,
          railCollapsed: false,
        }),
      );
      localStorage.removeItem("eagle-eye-analysis-history-v2");
      localStorage.removeItem("eagle-eye-analysis-history-v3");
      sessionStorage.clear();
    });
    await page.reload();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");

    await runAnalysis(page, "Populated-state baseline");
    await expect(page.locator(".chart-canvas canvas")).toHaveCount(1);
    await expect(page).toHaveScreenshot(
      `dark-${viewport.label}-populated.png`,
      {
        animations: "disabled",
        caret: "hide",
        fullPage: false,
        // ECharts canvas antialiasing can vary by a few pixels between
        // otherwise identical Chromium frames. Keep the approved layout
        // strict while ignoring that sub-pixel raster noise.
        maxDiffPixels: 50,
      },
    );

    await page.evaluate(() => {
      localStorage.removeItem("eagle-eye-analysis-history-v2");
      localStorage.removeItem("eagle-eye-analysis-history-v3");
      sessionStorage.clear();
    });
    await page.reload();
    await runAnalysis(
      page,
      "Empty-state baseline",
      "No rows matched the requested historical scope.",
    );
    await expect(
      page.getByText("No visualization contract was returned."),
    ).toBeVisible();
    await expect(page).toHaveScreenshot(
      `dark-${viewport.label}-empty.png`,
      {
        animations: "disabled",
        caret: "hide",
        fullPage: false,
        maxDiffPixels: 50,
      },
    );
  }
});

const readableAnswerRoutes = [
  ["/analysis", "Analysis Desk"],
  ["/traffic-monitoring", "Traffic Monitoring"],
  ["/vessel-investigation", "Vessel Investigation"],
  ["/eta-delay", "ETA Watch"],
  ["/port-pressure", "Port Pressure"],
  ["/carbon-emissions", "Carbon Emissions"],
] as const;

function projectWideReadableAnswer(
  question: string,
  label: string,
  conversationId: string,
): AnswerEnvelope {
  if (label === "ETA Watch") {
    return readableEtaPresentationEnvelope(question, conversationId);
  }
  const lead = `${label}: 54 validated vessel arrivals were recorded.`;
  const supporting =
    "The supporting breakdown covers three chronological daily buckets and reconciles to the immutable total.";
  const boundary =
    "Historical coverage ends on 30 April 2022; this is not current traffic.";
  const response = answerEnvelope(question, conversationId, {
    answer: `${lead}\n\n${supporting}\n\n${boundary}`,
    operational_brief: undefined,
  });

  if (label === "Analysis Desk") {
    return {
      ...response,
      mode: "maritime_research",
      state: "RETRIEVED",
      plan: {
        ...response.plan,
        mode: "maritime_research",
        operation: "research",
        source_scope: "authoritative_sources",
        requested_visual: "none",
      },
      facts: [
        {
          name: "grounded_source_count",
          value: 2,
          unit: "sources",
          source: "retrieved",
          immutable: true,
        },
      ],
      datasets: [],
      visualizations: [
        {
          id: "research-no-chart",
          kind: "omitted",
          title: "Research evidence",
          dataset_id: null,
          accessible_summary:
            "No chart is shown because the source-grounded response has no compatible structured dataset.",
          citations: ["research-local", "research-web"],
          reason_code: "not_meaningful",
          reason:
            "The research response is supported by evidence rather than a numeric chart dataset.",
        },
      ],
      evidence: [
        {
          id: "research-local",
          source_type: "local_document",
          title: "Local maritime source",
          excerpt: "A source-grounded local excerpt.",
          metadata: {},
        },
        {
          id: "research-web",
          source_type: "web",
          title: "Authoritative maritime source",
          url: "https://www.imo.org/",
          metadata: {},
        },
      ],
      trace: {
        ...response.trace,
        route: "maritime_research",
        operation: "research",
        sources: ["research-local", "research-web"],
        dataset_rows: 0,
        visualization_decision: "omitted",
      },
    };
  }

  if (label === "Vessel Investigation") {
    const vesselDataset: Dataset = {
      id: "vessel-events",
      columns: [
        { field: "row_id", label: "Row", data_type: "string" },
        { field: "mmsi", label: "MMSI", data_type: "string" },
        {
          field: "observed_at",
          label: "Observed",
          data_type: "datetime",
        },
        {
          field: "distance_km",
          label: "Distance",
          data_type: "number",
          unit: "km",
        },
      ],
      rows: [
        {
          row_id: "vessel-event-1",
          mmsi: "265000001",
          observed_at: "2022-03-10T11:00:00Z",
          distance_km: 12.5,
        },
      ],
      row_count: 1,
    };
    return {
      ...response,
      plan: {
        ...response.plan,
        operation: "ais_jump",
        metric: "distance_km",
        dimensions: ["mmsi", "observed_at"],
        mmsi: "265000001",
        requested_visual: "table",
      },
      applied_scope: {
        ...response.applied_scope,
        mmsi: "265000001",
      },
      datasets: [vesselDataset],
      visualizations: [
        {
          id: "vessel-events-table",
          kind: "table",
          title: "Vessel events",
          dataset_id: vesselDataset.id,
          table_fallback_dataset_id: vesselDataset.id,
          row_id_field: "row_id",
          accessible_summary: "A table of vessel-event observations.",
          citations: ["vessel-event-evidence"],
          visible_fields: ["mmsi", "observed_at", "distance_km"],
        },
      ],
      evidence: [
        {
          id: "vessel-event-evidence",
          source_type: "traffic_event",
          title: "AIS event record",
          metadata: { dataset_id: vesselDataset.id },
        },
      ],
      trace: {
        ...response.trace,
        operation: "ais_jump",
        sources: [vesselDataset.id],
        dataset_rows: 1,
        visualization_decision: "table",
      },
    };
  }

  return response;
}

test("every non-Overview workspace uses the same readable answer hierarchy and preserves its canonical response", async ({
  page,
}) => {
  let activeLabel = "Analysis Desk";
  await page.unroute("**/api/v2/query");
  await page.route("**/api/v2/query", async (route, request) => {
    const payload = request.postDataJSON() as QueryRequestPayload;
    return json(
      route,
      projectWideReadableAnswer(
        payload.question,
        activeLabel,
        payload.conversation_id || `readable-${activeLabel}`,
      ),
    );
  });

  for (const [path, label] of readableAnswerRoutes) {
    await test.step(label, async () => {
      activeLabel = label;
      const question = `Summarize the validated result for ${label}.`;
      const expected = projectWideReadableAnswer(
        question,
        label,
        `expected-${label}`,
      ).answer;

      await page.goto(path);
      await page.getByTestId("query-input").fill(question);
      await page.getByTestId("analyze-button").click();

      const summary = page.getByTestId("global-answer-summary");
      const detail = page.getByTestId("global-answer-detail");
      const canonical = page.getByTestId("canonical-response-disclosure");
      const canonicalAnswer = page.getByTestId("result-answer");

      await expect(summary).toBeVisible();
      await expect(summary).toContainText(question);
      await expect(detail).toBeVisible();
      if (label === "ETA Watch") {
        await expect(detail).toContainText(
          "Two due-soon vessels; two signals need immediate attention.",
        );
        await expect(detail).toContainText("Priority watchlist");
      } else {
        await expect(detail).toContainText(
          `${label}: 54 validated vessel arrivals were recorded.`,
        );
        await expect(detail).toContainText(
          "The supporting breakdown covers three chronological daily buckets",
        );
        await expect(detail).toContainText(
          "Historical coverage ends on 30 April 2022",
        );
      }

      if (label === "Analysis Desk") {
        await expect(page.getByTestId("evidence-inspector")).toContainText(
          "Authoritative maritime source",
        );
        await expect(
          page.getByText("No visualization is available.", { exact: true }),
        ).toBeVisible();
        await expect(
          page.getByText(
            "The research response is supported by evidence rather than a numeric chart dataset.",
            { exact: true },
          ),
        ).toBeVisible();
      }
      if (label === "Vessel Investigation") {
        await expect(
          page.getByRole("heading", { name: "Vessel events" }),
        ).toBeVisible();
      }

      const summaryBox = await summary.boundingBox();
      expect(summaryBox).not.toBeNull();
      expect(summaryBox!.height).toBeLessThan(180);

      await expect(canonical).toBeVisible();
      await expect(canonical).toHaveJSProperty("open", false);
      await expect(canonicalAnswer).toBeHidden();
      expect(await canonicalAnswer.evaluate((node) => node.textContent)).toBe(
        expected,
      );

      await canonical.locator("summary").click();
      await expect(canonical).toHaveJSProperty("open", true);
      await expect(canonicalAnswer).toBeVisible();
      expect(await canonicalAnswer.evaluate((node) => node.textContent)).toBe(
        expected,
      );
      await canonical.locator("summary").click();
    });
  }
});

const conciseResultStates = [
  {
    mode: "app_help",
    state: "GENERAL",
    answer:
      "Eagle Eye can analyze historical traffic, vessels, port pressure, and Carbon Emissions.",
  },
  {
    mode: "clarification",
    state: "CLARIFICATION_REQUIRED",
    answer: "Which Baltic port and UTC date range should this analysis use?",
  },
  {
    mode: "unsupported",
    state: "UNSUPPORTED",
    answer:
      "Confirmed commercial delay is unavailable from vessel-reported AIS. Eagle Eye has no official schedule baseline for this scope. No historical result was substituted. Ask for reported ETA changes instead. Add a vessel identity or supported destination to narrow the request.",
  },
  {
    mode: "maritime_research",
    state: "RETRIEVED",
    answer:
      "The source-grounded maritime research answer is available with its cited evidence.",
  },
  {
    mode: "general_chat",
    state: "GENERAL",
    answer: "The requested general maritime explanation is available.",
  },
  {
    mode: "analytics",
    state: "NO_DATA",
    answer:
      "No validated historical rows match this scope. Broaden the port or date filters and run the request again.",
  },
  {
    mode: "analytics",
    state: "ASSURANCE_UNAVAILABLE",
    answer:
      "A supported answer is unavailable for this request. Narrow the scope and run the request again.",
  },
] as const;

test("short help, clarification, unsupported, research, and general states remain direct and readable", async ({
  page,
}) => {
  let activeState: (typeof conciseResultStates)[number] =
    conciseResultStates[0];
  await page.unroute("**/api/v2/query");
  await page.route("**/api/v2/query", async (route, request) => {
    const payload = request.postDataJSON() as QueryRequestPayload;
    return json(
      route,
      answerEnvelope(
        payload.question,
        payload.conversation_id || `state-${activeState.mode}`,
        {
          mode: activeState.mode,
          state: activeState.state,
          answer: activeState.answer,
          facts: [],
          datasets: [],
          visualizations: [],
          evidence: [],
          confidence: "not_applicable",
          assurance: {
            status: "not_applicable",
            level: "not_applicable",
            basis: "system_response",
            reason: "No analytical assurance score applies to this response.",
            checks: [],
          },
          availability: {
            code: "not_applicable",
            provider: null,
            retryable: false,
          },
        },
      ),
    );
  });

  await page.goto("/analysis");
  for (const resultState of conciseResultStates) {
    await test.step(resultState.mode, async () => {
      activeState = resultState;
      const question = `Exercise the ${resultState.mode} result state.`;
      await page.getByTestId("query-input").fill(question);
      await page.getByTestId("analyze-button").click();

      await expect(page.getByTestId("global-answer-summary")).toContainText(
        question,
      );
      const detail = page.getByTestId("global-answer-detail");
      await expect(detail).toBeVisible();
      const visibleSegments = resultState.answer.split(
        /(?<=[.!?])\s+(?=[A-Z0-9])/,
      );
      await expect(detail).toContainText(visibleSegments[0]);
      await expect(detail).toContainText(
        visibleSegments[visibleSegments.length - 1],
      );
      await expect(detail).not.toBeEmpty();
      await expect(
        detail.getByText(/Show \d+ more answer point/),
      ).toHaveCount(0);
      await expect(
        page.getByTestId("canonical-response-disclosure"),
      ).toHaveJSProperty("open", false);
      expect(
        await page
          .getByTestId("result-answer")
          .evaluate((node) => node.textContent),
      ).toBe(resultState.answer);
    });
  }
});

test("the project-wide answer hierarchy keeps its mobile reading order and accessible disclosure", async ({
  page,
}) => {
  const question = "Show the mobile-readable validated traffic result.";
  const expected = projectWideReadableAnswer(
    question,
    "Traffic Monitoring",
    "mobile-readable",
  ).answer;
  await page.unroute("**/api/v2/query");
  await page.route("**/api/v2/query", async (route, request) => {
    const payload = request.postDataJSON() as QueryRequestPayload;
    return json(
      route,
      projectWideReadableAnswer(
        payload.question,
        "Traffic Monitoring",
        payload.conversation_id || "mobile-readable",
      ),
    );
  });

  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/traffic-monitoring");
  await page.getByTestId("query-input").fill(question);
  await page.getByTestId("analyze-button").click();

  const orderedSections = [
    page.getByTestId("global-answer-summary"),
    page.getByTestId("global-answer-detail"),
    page.getByTestId("result-visualizations"),
    page.getByTestId("result-metadata"),
    page.getByTestId("evidence-inspector"),
  ];
  const boxes = await Promise.all(
    orderedSections.map((locator) => locator.boundingBox()),
  );
  expect(boxes.every(Boolean)).toBe(true);
  expect(boxes.map((box) => box!.y)).toEqual(
    [...boxes].map((box) => box!.y).sort((a, b) => a - b),
  );

  const disclosure = page.getByTestId("canonical-response-disclosure");
  const disclosureControl = disclosure.locator("summary");
  await expect(disclosure).toHaveJSProperty("open", false);
  await disclosureControl.focus();
  await expect(disclosureControl).toBeFocused();
  await disclosureControl.press("Enter");
  await expect(disclosure).toHaveJSProperty("open", true);
  await expect(page.getByTestId("result-answer")).toBeVisible();
  expect(
    await page.getByTestId("result-answer").evaluate((node) => node.textContent),
  ).toBe(expected);
  await disclosureControl.press("Enter");
  await expect(disclosure).toHaveJSProperty("open", false);

  const viewport = await page.evaluate(() => ({
    client: document.documentElement.clientWidth,
    scroll: document.documentElement.scrollWidth,
  }));
  expect(viewport.scroll).toBeLessThanOrEqual(viewport.client);
  await expectNoSeriousA11yViolations(page);
});
