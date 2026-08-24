import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page, type Route } from "@playwright/test";
import type {
  AnswerEnvelope,
  CapabilityResponse,
  Dataset,
  QueryRequestPayload,
  VisualizationSpec,
} from "../../src/types";

const SERIES_ROWS = [
  ["2022-03-01", 18, 19],
  ["2022-03-02", 20, 19.5],
  ["2022-03-03", 19, 20],
  ["2022-03-04", 21, 20],
  ["2022-03-05", 22, 20],
  ["2022-03-06", 23, 20.5],
  ["2022-03-07", 21, 21],
  ["2022-03-08", 24, 21],
  ["2022-03-09", 20, 21],
  ["2022-03-10", 22, 22],
  ["2022-03-11", 25, 22],
  ["2022-03-12", 24, 22],
  ["2022-03-13", 19, 22],
  ["2022-03-14", 21, 22],
  ["2022-03-15", 23, 22],
  ["2022-03-16", 22, 22],
  ["2022-03-17", 26, 23],
  ["2022-03-18", 24, 23],
  ["2022-03-19", 20, 23],
  ["2022-03-20", 18, 22],
  ["2022-03-21", 21, 22],
  ["2022-03-22", 22, 22],
  ["2022-03-23", 27, 23],
  ["2022-03-24", 25, 23],
  ["2022-03-25", 23, 23],
  ["2022-03-26", 24, 23],
  ["2022-03-27", 22, 23],
  ["2022-03-28", 21, 23],
  ["2022-03-29", 20, 22],
  ["2022-03-30", 19, 22],
  ["2022-03-31", 25, 23],
  ["2022-04-01", 26, 24],
  ["2022-04-02", 28, 25],
  ["2022-04-03", 27, 25],
  ["2022-04-04", 24, 25],
  ["2022-04-05", 23, 25],
  ["2022-04-06", 22, 24],
  ["2022-04-07", 21, 23],
  ["2022-04-08", 29, 24],
  ["2022-04-09", 34, 26],
  ["2022-04-10", 30, 27],
  ["2022-04-11", 28, 27],
  ["2022-04-12", 26, 27],
  ["2022-04-13", 25, 26],
  ["2022-04-14", 24, 25],
  ["2022-04-15", 23, 24],
  ["2022-04-16", 22, 23],
  ["2022-04-17", 21, 23],
] as const;

const arrivalsDataset: Dataset = {
  id: "arrivals-daily-v21",
  columns: [
    { field: "row_id", label: "Row ID", data_type: "string" },
    { field: "date", label: "Date", data_type: "datetime" },
    {
      field: "arrivals",
      label: "Arrivals",
      data_type: "integer",
      unit: "vessels",
    },
    {
      field: "rolling_median_7",
      label: "7-day rolling median",
      data_type: "number",
      unit: "vessels",
    },
    { field: "is_peak", label: "Peak event", data_type: "boolean" },
  ],
  rows: SERIES_ROWS.map(([date, arrivals, rollingMedian], index) => ({
    row_id: `arrival-${String(index + 1).padStart(2, "0")}`,
    date,
    arrivals,
    rolling_median_7: rollingMedian,
    is_peak: date === "2022-04-09",
  })),
  row_count: SERIES_ROWS.length,
};

const arrivalsVisualization: VisualizationSpec = {
  id: "arrivals-daily-line-v21",
  kind: "cartesian",
  chart_type: "line",
  title: "Gothenburg daily arrivals",
  dataset_id: arrivalsDataset.id,
  table_fallback_dataset_id: arrivalsDataset.id,
  row_id_field: "row_id",
  accessible_summary:
    "Forty-eight validated daily arrival observations with a period median reference, rolling median overlay, and one flagged peak.",
  citations: ["evidence-arrivals-daily"],
  x_field: "date",
  y_fields: ["arrivals"],
  orientation: "vertical",
  sort: "calendar",
  y_unit: "vessels",
  stacked: false,
  reference_lines: [
    {
      id: "period-median",
      label: "Period median",
      axis: "y",
      value: 23,
      unit: "vessels",
      line_style: "dashed",
    },
  ],
  interval_bands: [],
  annotations: [
    {
      id: "peak-annotation",
      label: "Validated peak",
      condition_field: "is_peak",
      x_field: "date",
      y_field: "arrivals",
    },
  ],
  fitted_series: [
    {
      id: "rolling-median-overlay",
      label: "7-day rolling median",
      x_field: "date",
      y_field: "rolling_median_7",
      method: "rolling_median",
      association_only: false,
      slope: null,
      intercept: null,
      r_squared: null,
    },
  ],
};

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
  operations: ["arrivals"],
  visualization_kinds: ["cartesian", "table", "omitted"],
  freshness: {
    data_from: "2021-01-01",
    data_to: "2022-04-30",
    historical: true,
    message: "Historical coverage through 2022-04-30",
  },
  data_manifest: {
    schema_version: "2.1-test",
    built_at_utc: "2026-07-23T12:00:00Z",
    available_ports: ["SEGOT"],
    enabled_operations: ["arrivals"],
    row_counts: { arrivals: 48 },
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
    historical_analytics: ["arrivals"],
  },
  conversation_store: "sqlite",
};

function answerEnvelope(
  question: string,
  conversationId: string,
): AnswerEnvelope {
  return {
    api_version: "2.0",
    visualization_contract_version: "2.1",
    conversation_id: conversationId,
    turn_id: "turn-visualization-v21",
    question,
    mode: "analytics",
    state: "COMPUTED",
    answer:
      "Historical arrivals are shown for 48 validated daily observations.",
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
        date_to: "2022-04-17",
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
      limit: 48,
      source_scope: "historical",
      carbon_boundary: "TTW",
      pollutants: [],
      requested_visual: "line",
      ambiguities: [],
      clarification: null,
      reason: "Frozen visualization 2.1 browser fixture.",
      context_inherited: [],
      planner_source: "deterministic",
      planner_model: null,
    },
    facts: [
      {
        name: "observation_count",
        value: 48,
        unit: "days",
        entity: "SEGOT",
        source: "computed",
        immutable: true,
      },
      {
        name: "peak_arrivals",
        value: 34,
        unit: "vessels",
        entity: "SEGOT",
        source: "computed",
        immutable: true,
      },
      {
        name: "peak_date",
        value: "2022-04-09",
        entity: "SEGOT",
        source: "computed",
        immutable: true,
      },
      {
        name: "rolling_median_end",
        value: 23,
        unit: "vessels",
        entity: "SEGOT",
        source: "computed",
        immutable: true,
      },
    ],
    applied_scope: {
      ports: ["SEGOT"],
      date_from: "2022-03-01",
      date_to: "2022-04-17",
    },
    datasets: [arrivalsDataset],
    visualizations: [arrivalsVisualization],
    chart_insights: [
      {
        id: "insight-peak",
        visualization_id: arrivalsVisualization.id,
        insight_type: "peak",
        statement:
          "The validated peak was 34 vessels on 2022-04-09.",
        fact_names: ["peak_arrivals", "peak_date"],
        evidence_ids: ["evidence-arrivals-daily"],
      },
      {
        id: "insight-trend",
        visualization_id: arrivalsVisualization.id,
        insight_type: "trend",
        statement:
          "The 7-day rolling median ended at 23 vessels.",
        fact_names: ["rolling_median_end"],
        evidence_ids: ["evidence-arrivals-daily"],
      },
    ],
    evidence: [
      {
        id: "evidence-arrivals-daily",
        source_type: "computed",
        title: "Validated daily arrivals",
        excerpt:
          "Forty-eight deterministic historical rows from the analytics dataset.",
        metadata: { dataset_id: arrivalsDataset.id },
      },
    ],
    freshness: {
      data_from: "2021-01-01",
      data_to: "2022-04-30",
      historical: true,
      message: "Historical coverage through 2022-04-30",
    },
    confidence: "high",
    caveats: ["Historical data is not current operational traffic."],
    trace: {
      trace_id: "trace-visualization-v21",
      route: "analytics",
      operation: "arrivals",
      planner_source: "deterministic",
      planner_model: null,
      model: "deterministic",
      reasoning_effort: "none",
      sources: [arrivalsDataset.id],
      retrieval_mode: "none",
      retrieval_backend: "test",
      retrieval_status: "not_applicable",
      retrieval_top_k: 5,
      result_state: "COMPUTED",
      failure_state: null,
      result_hash: "result-hash-visualization-v21",
      data_manifest_version: "2.1-test",
      dataset_rows: 48,
      visualization_decision: "cartesian:line",
      visualization_contract_version: "2.1",
      chart_profile: ["cartesian:line"],
      visualization_dataset_ids: [arrivalsDataset.id],
      visualization_fallback_reasons: [],
      latency_ms: 12,
      warnings: [],
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
      answerEnvelope(
        payload.question,
        payload.conversation_id || "conversation-visualization-v21",
      ),
    );
  });
}

async function submitFixtureQuery(page: Page) {
  await page.goto("/analysis");
  await page
    .getByTestId("query-input")
    .fill("Show the validated Gothenburg daily arrival trend.");
  await page.getByTestId("analyze-button").click();
  await expect(page.getByTestId("result-answer")).toHaveText(
    "Historical arrivals are shown for 48 validated daily observations.",
  );
  await expect(page.locator(".chart-canvas canvas")).toHaveCount(1);
}

async function expectNoBlockingAxeViolations(page: Page) {
  const results = await new AxeBuilder({ page }).analyze();
  const blocking = results.violations.filter(
    (violation) =>
      violation.impact === "serious" || violation.impact === "critical",
  );
  expect(
    blocking,
    blocking.map((violation) => `${violation.id}: ${violation.help}`).join("\n"),
  ).toEqual([]);
}

test.beforeEach(async ({ page }) => {
  await installMocks(page);
});

test("visualization 2.1 exposes analyst controls, inspection, insights, fullscreen, data, and PNG", async ({
  page,
}) => {
  await submitFixtureQuery(page);

  const visual = page.locator(".visualization-primary");
  await expect
    .poll(() =>
      page.evaluate(() => {
        const observations =
          window.__EAGLE_EYE_VISUALIZATION_OBSERVABILITY__ || [];
        return observations.find(
          (observation) =>
            observation.visualization_id === "arrivals-daily-line-v21" &&
            observation.interaction_mode === "render",
        );
      }),
    )
    .toMatchObject({
      visualization_id: "arrivals-daily-line-v21",
      chart_profile: "cartesian:line",
      visualization_contract_version: "2.1",
      fallback_reason: null,
      interaction_mode: "render",
      source_dataset_ids: ["arrivals-daily-v21"],
    });
  const renderLatency = await page.evaluate(() => {
    const observations =
      window.__EAGLE_EYE_VISUALIZATION_OBSERVABILITY__ || [];
    return observations.find(
      (observation) =>
        observation.visualization_id === "arrivals-daily-line-v21" &&
        observation.interaction_mode === "render",
    )?.render_latency_ms;
  });
  expect(renderLatency).toEqual(expect.any(Number));
  expect(renderLatency).toBeGreaterThanOrEqual(0);

  await expect(visual).toHaveAttribute(
    "data-profile",
    "temporal-focus-context",
  );
  await expect(visual).toContainText(
    "Time series + focus/context · 48 rows · vessels",
  );
  await expect(page.getByTestId("chart-insights")).toContainText(
    "The validated peak was 34 vessels on 2022-04-09.",
  );
  await expect(page.getByTestId("chart-insights")).toContainText(
    "The 7-day rolling median ended at 23 vessels.",
  );

  const zoomIn = visual.getByRole("button", { name: "Zoom in on chart" });
  const reset = visual.getByRole("button", { name: "Reset chart view" });
  await expect(zoomIn).toBeEnabled();
  await zoomIn.click();
  await reset.click();
  await expect(visual.getByRole("alert")).toHaveCount(0);

  const inspector = visual.locator(".chart-inspector-focus");
  const liveRegion = visual.locator(".chart-live-region");
  await inspector.focus();
  await expect(liveRegion).toContainText("Point 1 of 48");
  await inspector.press("ArrowRight");
  await expect(liveRegion).toContainText("Point 2 of 48");
  await expect(liveRegion).toContainText("Arrivals: 20 vessels");

  await visual.getByRole("button", { name: "View data" }).click();
  const dataTable = visual.getByRole("table", {
    name: "Underlying analysis data",
  });
  await expect(dataTable).toBeVisible();
  await expect(dataTable.getByRole("row")).toHaveCount(49);

  const downloadPromise = page.waitForEvent("download");
  await visual.getByRole("button", { name: "Download chart as PNG" }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe("gothenburg-daily-arrivals.png");

  await visual.getByRole("button", { name: "Expand chart" }).click();
  const dialog = page.getByRole("dialog", {
    name: "Gothenburg daily arrivals",
  });
  await expect(dialog).toBeVisible();
  const close = dialog.getByRole("button", { name: "Close", exact: true });
  await expect(close).toBeFocused();
  await page.keyboard.press("Shift+Tab");
  await expect(dialog.locator(".chart-inspector-focus")).toBeFocused();
  await page.keyboard.press("Tab");
  await expect(close).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();

  const interactionModes = await page.evaluate(() =>
    (window.__EAGLE_EYE_VISUALIZATION_OBSERVABILITY__ || []).map(
      (observation) => observation.interaction_mode,
    ),
  );
  expect(interactionModes).toEqual(
    expect.arrayContaining([
      "zoom_in",
      "reset",
      "keyboard_inspection",
      "view_data",
      "png_download",
      "expand",
    ]),
  );
});

test("mobile keeps answer, chart, insights, metadata, and evidence in order with a 310px canvas", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await submitFixtureQuery(page);

  const ordered = await Promise.all(
    [
      page.getByTestId("result-answer"),
      page.getByTestId("result-visualizations"),
      page.getByTestId("chart-insights"),
      page.getByTestId("result-metadata"),
      page.getByTestId("evidence-inspector"),
    ].map((locator) => locator.boundingBox()),
  );
  expect(ordered.every(Boolean)).toBe(true);
  const positions = ordered.map((box) => box!.y);
  expect(positions).toEqual([...positions].sort((a, b) => a - b));

  const chartBox = await page.locator(".chart-canvas").boundingBox();
  expect(chartBox).not.toBeNull();
  expect(chartBox!.height).toBeGreaterThanOrEqual(309);
  expect(chartBox!.height).toBeLessThanOrEqual(311);

  const width = await page.evaluate(() => ({
    clientWidth: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth,
  }));
  expect(width.scrollWidth).toBeLessThanOrEqual(width.clientWidth);
  await expectNoBlockingAxeViolations(page);
});
