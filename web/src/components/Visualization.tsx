import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { ECharts, EChartsOption } from "echarts";
import type { StyleSpecification } from "maplibre-gl";
import type { ColumnSpec, DataRow, Dataset, VisualizationSpec } from "../types";
import naturalEarthLandRaw from "../assets/maps/ne_110m_land.geojson?raw";

type EChartsModule = typeof import("echarts");
let echartsModulePromise: Promise<EChartsModule> | null = null;

function loadEChartsModule(): Promise<EChartsModule> {
  if (!echartsModulePromise) echartsModulePromise = import("echarts");
  return echartsModulePromise;
}

// Begin fetching the renderer with the application shell so a submitted
// analysis does not have to wait for the chart chunk before creating canvases.
if (typeof window !== "undefined") void loadEChartsModule();

interface VisualizationProps {
  visualization: VisualizationSpec;
  datasets: Dataset[];
  primary?: boolean;
  contractVersion?: "2.0" | "2.1";
}

interface VisualizationObservation {
  recorded_at: string;
  visualization_id: string;
  chart_profile: string;
  semantic_profile: string;
  visualization_contract_version: "2.0" | "2.1";
  render_latency_ms: number | null;
  fallback_reason: string | null;
  interaction_mode: string;
  source_dataset_ids: string[];
  analytical_series_names: string[];
  analytical_series_types: string[];
  rendered_point_count: number;
  reference_line_count: number;
  interval_band_count: number;
  annotation_count: number;
  preserved_null_count: number;
  fitted_series_count: number;
  summary_marker_count: number;
  quality_annotation_count: number;
  null_value_count: number;
  null_to_zero_count: number;
  inspected_series_name: string | null;
}

declare global {
  interface Window {
    __EAGLE_EYE_VISUALIZATION_OBSERVABILITY__?: VisualizationObservation[];
  }
}

function recordVisualizationObservation(
  observation: Omit<VisualizationObservation, "recorded_at">,
) {
  const entry: VisualizationObservation = {
    recorded_at: new Date().toISOString(),
    ...observation,
  };
  const buffer = window.__EAGLE_EYE_VISUALIZATION_OBSERVABILITY__ || [];
  window.__EAGLE_EYE_VISUALIZATION_OBSERVABILITY__ = [...buffer, entry].slice(-200);
  window.dispatchEvent(
    new CustomEvent("eagleeye:visualization-observability", {
      detail: entry,
    }),
  );
}

interface ChartCommandProps {
  canZoom: boolean;
  expanded: boolean;
  showData: boolean;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onReset: () => void;
  onExpand: () => void;
  onInspect: () => void;
  onToggleData: () => void;
  onDownload: () => void;
}

interface MapGeometryFields {
  geometry_mode?: "points" | "segments" | "ordered_path";
  start_latitude_field?: string | null;
  start_longitude_field?: string | null;
  end_latitude_field?: string | null;
  end_longitude_field?: string | null;
  path_field?: string | null;
  sequence_field?: string | null;
  timestamp_field?: string | null;
}

interface CartesianReferenceLine {
  id: string;
  label: string;
  axis: "x" | "y";
  value: string | number;
  unit?: string | null;
  line_style?: "solid" | "dashed" | "dotted";
}

interface CartesianIntervalBand {
  id: string;
  label: string;
  lower_field: string;
  upper_field: string;
  point_field?: string | null;
  display?: "band" | "whisker";
  unit?: string | null;
}

interface CartesianAnnotation {
  id: string;
  label: string;
  condition_field: string;
  x_field: string;
  y_field: string;
}

interface CartesianFittedSeries {
  id: string;
  label: string;
  x_field: string;
  y_field: string;
  method: "ols" | "rolling_median";
  association_only: boolean;
  slope: number;
  intercept: number;
  r_squared?: number | null;
}

interface CartesianEnhancements {
  reference_lines?: CartesianReferenceLine[];
  interval_bands?: CartesianIntervalBand[];
  annotations?: CartesianAnnotation[];
  fitted_series?: CartesianFittedSeries[];
}

interface DistributionEnhancements {
  bin_lower_field?: string | null;
  bin_upper_field?: string | null;
  summary_dataset_id?: string | null;
  outlier_dataset_id?: string | null;
  five_number_summary?: {
    minimum: number;
    q1: number;
    median: number;
    q3: number;
    maximum: number;
    lower_whisker: number;
    upper_whisker: number;
    p90?: number | null;
    count: number;
  } | null;
  outlier_condition_field?: string | null;
  outlier_value_field?: string | null;
}

interface ForecastEnhancements {
  interval_level?: number;
  forecast_boundary?: string | null;
  quality_metrics?: {
    mase: number;
    interval_coverage: number;
    interval_level: number;
    gate_passed: true;
  } | null;
}

interface KpiEnhancements {
  comparison_field?: string | null;
  baseline_value?: number | null;
  thresholds?: Array<{ id: string; label: string; value: number; unit?: string | null }>;
}

interface FocusedPositiveAxisScale {
  minimum: number;
  maximum: number;
}

type PresentationProfile =
  | "metric-bullet"
  | "temporal-focus-context"
  | "ranking-lollipop"
  | "category-comparison"
  | "composition-ribbon"
  | "correlation-fit"
  | "uncertainty-range"
  | "forecast-fan-interval"
  | "distribution-histogram-summary"
  | "distribution-box"
  | "distribution-percentile"
  | "calendar-pattern-heatmap"
  | "geospatial-investigation"
  | "event-rail"
  | "verified-table"
  | "explicit-omission";

interface VisualizationDiagnosticFields {
  semantic_profile: PresentationProfile;
  analytical_series_names: string[];
  analytical_series_types: string[];
  rendered_point_count: number;
  reference_line_count: number;
  interval_band_count: number;
  annotation_count: number;
  preserved_null_count: number;
  fitted_series_count: number;
  summary_marker_count: number;
  quality_annotation_count: number;
  null_value_count: number;
  null_to_zero_count: number;
  inspected_series_name: string | null;
}

type ResolvedTheme = "dark";
interface ChartTheme {
  text: string;
  muted: string;
  subtle: string;
  grid: string;
  tooltipBackground: string;
  tooltipBorder: string;
  surface: string;
  ocean: string;
  land: string;
  coast: string;
  interval: string;
  danger: string;
  palette: string[];
}

const numberFormatter = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });
const preciseDateFormatter = new Intl.DateTimeFormat("en-CA", {
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
  timeZone: "UTC",
});
const preciseDateTimeFormatter = new Intl.DateTimeFormat("en-GB", {
  year: "numeric",
  month: "short",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
  hourCycle: "h23",
  timeZone: "UTC",
});
const weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
const lineSymbols = ["circle", "rect", "diamond", "triangle", "roundRect"] as const;
const lineTypes = ["solid", "dashed", "dotted"] as const;
const decalSymbols = ["rect", "circle", "triangle", "diamond"] as const;
const naturalEarthLand = JSON.parse(naturalEarthLandRaw) as GeoJSON.FeatureCollection;

const chartThemes: Record<ResolvedTheme, ChartTheme> = {
  dark: {
    text: "#E7ECE9",
    muted: "#9DAAA6",
    subtle: "#64767B",
    grid: "#26363D",
    tooltipBackground: "#101C21",
    tooltipBorder: "#43545A",
    surface: "#142228",
    ocean: "#0A1820",
    land: "#25343A",
    coast: "#66777A",
    interval: "#6E9DD4",
    danger: "#D97975",
    palette: ["#4BAE9E", "#E3A13B", "#6E9DD4", "#C17AA4", "#D8C45A", "#9B8BD3", "#D97975"],
  },
};

function useResolvedTheme(): ResolvedTheme {
  return "dark";
}

function datasetFor(spec: VisualizationSpec, datasets: Dataset[], fallback = false): Dataset | undefined {
  const id = fallback ? spec.table_fallback_dataset_id : spec.dataset_id;
  return datasets.find((dataset) => dataset.id === id) || (!fallback ? undefined : datasets.find((dataset) => dataset.id === spec.dataset_id));
}

function timelinePresentationDataset(
  spec: VisualizationSpec,
  dataset?: Dataset,
): Dataset | undefined {
  if (!dataset || spec.kind !== "timeline") return dataset;
  const rows = dataset.rows.filter((row) =>
    Number.isFinite(Date.parse(text(row[spec.time_field]))));
  if (rows.length === dataset.rows.length) return dataset;
  return {
    ...dataset,
    rows,
    row_count: rows.length,
  };
}

function unitFor(columns: ColumnSpec[], field: string): string | undefined { return columns.find((column) => column.field === field)?.unit || undefined; }
function asNumber(value: unknown): number | null {
  if (value == null || (typeof value === "string" && !value.trim())) return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}
function text(value: unknown): string { return value == null ? "—" : String(value); }
function escapeHtml(value: unknown): string { return text(value).replace(/[&<>"']/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character]!); }
function parsedDate(value: unknown): Date | null {
  const numeric = typeof value === "number" ? value : Date.parse(text(value));
  if (!Number.isFinite(numeric)) return null;
  const date = new Date(numeric);
  return Number.isNaN(date.valueOf()) ? null : date;
}
function axisText(value: unknown): string {
  const dateValue = parsedDate(value);
  if (dateValue && dateValue.valueOf() >= Date.UTC(1900, 0, 1) && dateValue.valueOf() <= Date.UTC(2200, 0, 1)) {
    const source = text(value);
    const includesTime = typeof value === "number" || source.includes("T") || dateValue.getUTCHours() !== 0 || dateValue.getUTCMinutes() !== 0;
    return includesTime
      ? dateValue.toISOString().replace(".000Z", "Z")
      : dateValue.toISOString().slice(0, 10);
  }
  const raw = text(value);
  return raw;
}
function timeAxisText(value: unknown): string {
  const date = parsedDate(value);
  if (!date) return text(value);
  const includesTime = date.getUTCHours() !== 0 || date.getUTCMinutes() !== 0;
  return includesTime
    ? preciseDateTimeFormatter.format(date).replace(",", "\n")
    : preciseDateFormatter.format(date).replaceAll("-", "\n");
}
function visualSummary(spec: VisualizationSpec, rows: DataRow[]): string { return spec.accessible_summary || `${spec.title}. ${rows.length} data rows are available in the table.`; }
function centeredAxisName(name?: string | null, nameGap = 42) { return name ? { name, nameLocation: "middle" as const, nameGap } : {}; }

function focusedPositiveCarbonAxis(
  spec: Extract<VisualizationSpec, { kind: "cartesian" }>,
  rows: DataRow[],
): FocusedPositiveAxisScale | null {
  if (spec.chart_type !== "line" && spec.chart_type !== "area") return null;
  const carbonIdentity = [spec.title, spec.y_unit, ...spec.y_fields].filter(Boolean).join(" ");
  if (!/(?:carbon|co2e?|greenhouse)/i.test(carbonIdentity)) return null;

  const fields = new Set(spec.y_fields);
  for (const band of spec.interval_bands || []) {
    fields.add(band.lower_field);
    fields.add(band.upper_field);
    if (band.point_field) fields.add(band.point_field);
  }
  for (const binding of spec.fitted_series || []) fields.add(binding.y_field);
  for (const annotation of spec.annotations || []) fields.add(annotation.y_field);
  if (spec.highlight?.value_field) fields.add(spec.highlight.value_field);

  const values = rows.flatMap((row) =>
    Array.from(fields).flatMap((field) => {
      const value = asNumber(row[field]);
      return value === null ? [] : [value];
    }),
  );
  for (const line of spec.reference_lines || []) {
    if (line.axis !== "y") continue;
    const value = asNumber(line.value);
    if (value !== null) values.push(value);
  }
  if (!values.length || values.some((value) => value <= 0)) return null;

  const dataMinimum = Math.min(...values);
  const dataMaximum = Math.max(...values);
  const dataSpan = dataMaximum - dataMinimum;
  const padding = dataSpan > 0
    ? Math.max(dataSpan * 0.12, dataMaximum * 0.02)
    : Math.max(Math.abs(dataMaximum) * 0.08, Number.EPSILON);
  return {
    minimum: Math.max(dataMinimum * 0.5, dataMinimum - padding),
    maximum: dataMaximum + padding,
  };
}

function axisTitle(field: string, unit?: string | null): string {
  const words = field.replaceAll("_", " ").trim().split(/\s+/);
  const normalizedUnit = unit?.trim().toLowerCase();
  const unitAliases: Record<string, string[]> = {
    hours: ["h", "hr", "hrs", "hour", "hours"],
    minutes: ["min", "mins", "minute", "minutes"],
    vessels: ["vessel", "vessels"],
    arrivals: ["arrival", "arrivals"],
  };
  if (normalizedUnit) {
    const aliases = unitAliases[normalizedUnit] || [normalizedUnit, normalizedUnit.replace(/s$/, "")];
    while (words.length > 1 && aliases.includes(words[words.length - 1].toLowerCase())) words.pop();
  }
  const label = words.join(" ").replace(/^./, (letter) => letter.toUpperCase());
  return normalizedUnit ? `${label} (${unit})` : label;
}
function calendarRank(value: unknown): number {
  const label = text(value).trim();
  const normalizedLabel = label.replace(/\.$/, "").toLowerCase();
  const weekday = weekdays.findIndex((day) => day.toLowerCase() === normalizedLabel || (normalizedLabel.length === 3 && day.slice(0, 3).toLowerCase() === normalizedLabel));
  if (weekday >= 0) return weekday;
  const hour = label.match(/^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?(?:\s*utc)?$/i);
  if (hour) {
    let valueHour = Number(hour[1]);
    if (hour[3]?.toLowerCase() === "pm" && valueHour < 12) valueHour += 12;
    if (hour[3]?.toLowerCase() === "am" && valueHour === 12) valueHour = 0;
    return valueHour + Number(hour[2] || 0) / 60;
  }
  const timestamp = Date.parse(label);
  return Number.isNaN(timestamp) ? Number.POSITIVE_INFINITY : timestamp;
}
function calendarValues(values: string[]): string[] {
  return [...values].sort((a, b) => calendarRank(a) - calendarRank(b) || a.localeCompare(b));
}
function quantile(values: number[], percentile: number): number {
  const sorted = [...values].sort((a, b) => a - b);
  const position = (sorted.length - 1) * percentile;
  const lower = Math.floor(position), upper = Math.ceil(position);
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

function rowIdentity(spec: VisualizationSpec, row: DataRow, index: number): string {
  const supplied = spec.row_id_field ? row[spec.row_id_field] : null;
  return supplied == null || text(supplied) === "—" ? `${spec.id}-row-${index}` : text(supplied);
}

interface LegacyCarbonPresentation {
  visualization: VisualizationSpec;
  dataset?: Dataset;
  state: "enriched" | "retained" | null;
  notice: string | null;
}

const legacyCarbonEnhancedNotice = "Additional uncertainty fields from this saved result are displayed.";
const legacyCarbonRetainedNotice = "This saved chart uses its original data fields.";

function normalizedUnit(value?: string | null): string | null {
  const normalized = value?.trim().toLowerCase();
  return normalized || null;
}

function nearlyEqual(left: number, right: number): boolean {
  const scale = Math.max(1, Math.abs(left), Math.abs(right));
  return Math.abs(left - right) <= scale * 1e-9;
}

function legacyCarbonPresentation(
  contractVersion: "2.0" | "2.1",
  visualization: VisualizationSpec,
  datasets: Dataset[],
): LegacyCarbonPresentation {
  const originalDataset = datasetFor(visualization, datasets);
  if (
    contractVersion !== "2.0"
    || visualization.kind !== "cartesian"
    || !originalDataset
  ) {
    return { visualization, dataset: originalDataset, state: null, notice: null };
  }
  const carbonFields = visualization.y_fields.filter((field) =>
    /(^|_)(?:co2|co2e|carbon)(?:_|$)/i.test(field));
  if (!carbonFields.length) {
    return { visualization, dataset: originalDataset, state: null, notice: null };
  }
  const candidateIds = [
    visualization.dataset_id,
    visualization.table_fallback_dataset_id,
    ...datasets.map((dataset) => dataset.id),
  ].filter((value): value is string => Boolean(value));
  const candidates = Array.from(new Set(candidateIds))
    .map((id) => datasets.find((dataset) => dataset.id === id))
    .filter((dataset): dataset is Dataset => Boolean(dataset));
  const originalFields = new Set(originalDataset.columns.map((column) => column.field));
  const joinField = visualization.row_id_field && originalFields.has(visualization.row_id_field)
    ? visualization.row_id_field
    : visualization.x_field;

  for (const candidate of candidates) {
    const candidateColumns = new Map(candidate.columns.map((column) => [column.field, column]));
    if (!candidateColumns.has(joinField)) continue;
    const bindings = carbonFields.map((field) => {
      const supplied = visualization.interval_bands?.find((band) =>
        band.point_field === field
        || (visualization.y_fields.length === 1 && visualization.interval_bands?.length === 1));
      if (
        supplied
        && candidateColumns.has(supplied.lower_field)
        && candidateColumns.has(supplied.upper_field)
      ) {
        return {
          pointField: field,
          lowerField: supplied.lower_field,
          upperField: supplied.upper_field,
        };
      }
      const bases = Array.from(new Set([
        field,
        field.replace(/_(?:ttw|wtw|wtt)$/i, ""),
      ]));
      const base = bases.find((candidateBase) =>
        candidateColumns.has(`${candidateBase}_lower`)
        && candidateColumns.has(`${candidateBase}_upper`));
      return base ? {
        pointField: field,
        lowerField: `${base}_lower`,
        upperField: `${base}_upper`,
      } : null;
    });
    if (bindings.some((binding) => binding === null) || !carbonFields.every((field) => candidateColumns.has(field))) continue;
    const resolvedBindings = bindings.filter((binding): binding is NonNullable<typeof binding> => binding !== null);

    let unitsAreCompatible = true;
    for (const binding of resolvedBindings) {
      const pointUnit = normalizedUnit(candidateColumns.get(binding.pointField)?.unit);
      const lowerUnit = normalizedUnit(candidateColumns.get(binding.lowerField)?.unit);
      const upperUnit = normalizedUnit(candidateColumns.get(binding.upperField)?.unit);
      const declaredUnit = normalizedUnit(visualization.y_unit || unitFor(originalDataset.columns, binding.pointField));
      if (!pointUnit || pointUnit !== lowerUnit || pointUnit !== upperUnit || (declaredUnit && declaredUnit !== pointUnit)) {
        unitsAreCompatible = false;
        break;
      }
    }
    if (!unitsAreCompatible) continue;

    const candidateByKey = new Map<string, DataRow>();
    let duplicateKey = false;
    for (const row of candidate.rows) {
      const rawKey = row[joinField];
      if (rawKey == null) {
        duplicateKey = true;
        break;
      }
      const key = text(rawKey);
      if (candidateByKey.has(key)) {
        duplicateKey = true;
        break;
      }
      candidateByKey.set(key, row);
    }
    if (duplicateKey) continue;

    const mergedRows: DataRow[] = [];
    let rowsAreCompatible = true;
    for (const row of originalDataset.rows) {
      const rawKey = row[joinField];
      const candidateRow = rawKey == null ? undefined : candidateByKey.get(text(rawKey));
      if (!candidateRow) {
        rowsAreCompatible = false;
        break;
      }
      const additions: DataRow = {};
      for (const binding of resolvedBindings) {
        const chartPoint = asNumber(row[binding.pointField]);
        const tablePoint = asNumber(candidateRow[binding.pointField]);
        const lower = asNumber(candidateRow[binding.lowerField]);
        const upper = asNumber(candidateRow[binding.upperField]);
        if (
          chartPoint === null
          || tablePoint === null
          || lower === null
          || upper === null
          || !nearlyEqual(chartPoint, tablePoint)
          || lower > chartPoint
          || chartPoint > upper
        ) {
          rowsAreCompatible = false;
          break;
        }
        additions[binding.lowerField] = lower;
        additions[binding.upperField] = upper;
      }
      if (!rowsAreCompatible) break;
      mergedRows.push({ ...row, ...additions });
    }
    if (!rowsAreCompatible || mergedRows.length !== originalDataset.rows.length) continue;

    const extraColumns = resolvedBindings.flatMap((binding) => [
      candidateColumns.get(binding.lowerField)!,
      candidateColumns.get(binding.upperField)!,
    ]).filter((column, index, columns) =>
      !originalFields.has(column.field)
      && columns.findIndex((candidateColumn) => candidateColumn.field === column.field) === index);
    const intervals = resolvedBindings.map((binding) => ({
      id: `legacy-${binding.pointField}-uncertainty`,
      label: `${axisTitle(binding.pointField)} uncertainty`,
      lower_field: binding.lowerField,
      upper_field: binding.upperField,
      point_field: binding.pointField,
      display: (mergedRows.length <= 5 ? "whisker" : "band") as "whisker" | "band",
      unit: candidateColumns.get(binding.pointField)?.unit || visualization.y_unit || null,
    }));
    return {
      visualization: { ...visualization, interval_bands: intervals },
      dataset: {
        ...originalDataset,
        columns: [...originalDataset.columns, ...extraColumns],
        rows: mergedRows,
      },
      state: "enriched",
      notice: legacyCarbonEnhancedNotice,
    };
  }
  return {
    visualization: visualization.interval_bands?.length
      ? { ...visualization, interval_bands: [] }
      : visualization,
    dataset: originalDataset,
    state: "retained",
    notice: legacyCarbonRetainedNotice,
  };
}

function DataTable({ dataset, fields }: { dataset?: Dataset; fields?: string[] }) {
  if (!dataset || !dataset.rows.length) return <p className="table-empty">No table data is available for this result.</p>;
  const available = dataset.columns.filter((column) => !fields?.length || fields.includes(column.field));
  const columns = available.length ? available : dataset.columns;
  return <div className="data-table-wrap" tabIndex={0} aria-label="Scrollable analysis data"><table aria-label="Underlying analysis data"><thead><tr>{columns.map((column) => <th key={column.field}>{column.label}</th>)}</tr></thead><tbody>{dataset.rows.slice(0, 200).map((row, index) => <tr key={index}>{columns.map((column) => <td key={column.field}>{text(row[column.field])}{column.unit && typeof row[column.field] === "number" ? ` ${column.unit}` : ""}</td>)}</tr>)}</tbody></table>{dataset.row_count > 200 && <p className="table-note">Showing the first 200 of {dataset.row_count} rows.</p>}</div>;
}

function tooltipValue(value: unknown): string {
  const numericValue = Number(Array.isArray(value) ? value[0] : value);
  return Number.isFinite(numericValue) ? numberFormatter.format(numericValue) : text(value);
}

function formattedValue(value: unknown, unit?: string | null): string {
  const numeric = asNumber(value);
  const rendered = numeric === null ? text(value) : numberFormatter.format(numeric);
  return unit ? `${rendered} ${unit}` : rendered;
}

function seriesUnit(columns: ColumnSpec[], field: string, explicit?: string | null): string | undefined {
  return explicit || unitFor(columns, field);
}

function highDensitySeries(rows: DataRow[]) {
  return rows.length > 1_000
    ? { progressive: 2_000, progressiveThreshold: 1_000, large: true, largeThreshold: 1_000 }
    : {};
}

function chartZoom(rows: DataRow[], axis: "x" | "y" = "x"): EChartsOption["dataZoom"] {
  if (rows.length <= 45) return undefined;
  const axisIndex = 0;
  const controls: NonNullable<EChartsOption["dataZoom"]> = [{
    type: "inside",
    [axis === "x" ? "xAxisIndex" : "yAxisIndex"]: axisIndex,
    filterMode: "none",
    zoomOnMouseWheel: "shift",
    moveOnMouseWheel: true,
    moveOnMouseMove: false,
  }];
  if (rows.length > 45) {
    controls.push({
      type: "slider",
      [axis === "x" ? "xAxisIndex" : "yAxisIndex"]: axisIndex,
      height: axis === "x" ? 18 : undefined,
      width: axis === "y" ? 14 : undefined,
      right: axis === "y" ? 6 : undefined,
      bottom: axis === "x" ? 12 : undefined,
      showDetail: false,
      brushSelect: false,
      borderColor: "transparent",
    });
  }
  return controls;
}

function baseOption(spec: VisualizationSpec, rows: DataRow[], theme: ChartTheme): EChartsOption {
  return {
    backgroundColor: "transparent",
    useUTC: true,
    color: theme.palette,
    textStyle: { color: theme.text, fontFamily: '"IBM Plex Sans", "Avenir Next", system-ui, sans-serif' },
    animation: !window.matchMedia("(prefers-reduced-motion: reduce)").matches,
    animationDuration: 240,
    aria: { enabled: true, description: visualSummary(spec, rows) },
    grid: { left: 64, right: 28, top: 54, bottom: 68, containLabel: true },
    tooltip: {
      trigger: "axis",
      axisPointer: {
        type: "cross",
        snap: true,
        lineStyle: { color: theme.muted, width: 1, type: "dashed" },
        crossStyle: { color: theme.muted, width: 1, type: "dashed" },
        label: { color: theme.surface, backgroundColor: theme.text },
      },
      backgroundColor: theme.tooltipBackground,
      borderColor: theme.tooltipBorder,
      borderWidth: 1,
      textStyle: { color: theme.text, fontSize: 12 },
      extraCssText: "box-shadow:none;border-radius:2px;",
      valueFormatter: tooltipValue,
    },
    axisPointer: {
      link: [],
      label: { color: theme.surface, backgroundColor: theme.text },
    },
  };
}

function tooltipAppearance(theme: ChartTheme) {
  return {
    backgroundColor: theme.tooltipBackground,
    borderColor: theme.tooltipBorder,
    borderWidth: 1,
    textStyle: { color: theme.text, fontSize: 12 },
    extraCssText: "box-shadow:none;border-radius:2px;",
  };
}

function categoryAxis(theme: ChartTheme, data: string[], name?: string | null, rotate = 0) {
  return {
    type: "category" as const,
    data,
    ...centeredAxisName(name),
    nameTextStyle: { color: theme.muted, fontSize: 11 },
    axisLine: { lineStyle: { color: theme.subtle, width: 1 } },
    axisTick: { lineStyle: { color: theme.subtle } },
    axisLabel: { color: theme.muted, fontSize: 11, rotate, hideOverlap: true, formatter: axisText },
  };
}

function valueAxis(
  theme: ChartTheme,
  name?: string | null,
  nameGap = 42,
  focusedScale?: FocusedPositiveAxisScale | null,
) {
  return {
    type: "value" as const,
    ...centeredAxisName(name, nameGap),
    ...(focusedScale
      ? {
        scale: true,
        min: focusedScale.minimum,
        max: focusedScale.maximum,
      }
      : {}),
    nameTextStyle: { color: theme.muted, fontSize: 11 },
    axisLine: { show: false },
    axisTick: { show: false },
    axisLabel: { color: theme.muted, fontSize: 11 },
    splitLine: { lineStyle: { color: theme.grid, width: 1, type: "dashed" as const } },
  };
}

function timeAxis(theme: ChartTheme, name?: string | null) {
  return {
    type: "time" as const,
    min: "dataMin" as const,
    max: "dataMax" as const,
    boundaryGap: [0, 0] as [number, number],
    ...centeredAxisName(name),
    nameTextStyle: { color: theme.muted, fontSize: 11 },
    axisLine: { lineStyle: { color: theme.subtle, width: 1 } },
    axisTick: { show: false },
    axisLabel: {
      color: theme.muted,
      fontSize: 10,
      hideOverlap: true,
      formatter: timeAxisText,
      lineHeight: 14,
    },
    splitLine: { show: false },
    axisPointer: {
      label: {
        formatter: ({ value }: { value?: unknown }) => axisText(value),
      },
    },
  };
}

function seriesDecal(index: number, theme: ChartTheme) {
  const patterns = [
    { dashArrayX: [1, 0], dashArrayY: [1, 0] },
    { dashArrayX: [3, 2], dashArrayY: [2, 2] },
    { dashArrayX: [1, 2], dashArrayY: [4, 2] },
    { dashArrayX: [4, 2], dashArrayY: [1, 2] },
  ];
  return {
    symbol: decalSymbols[index % decalSymbols.length],
    symbolSize: 0.7,
    color: theme.surface,
    backgroundColor: "transparent",
    ...patterns[index % patterns.length],
  };
}

function cartesianOption(
  spec: Extract<VisualizationSpec, { kind: "cartesian" }>,
  rows: DataRow[],
  columns: ColumnSpec[],
  theme: ChartTheme,
): EChartsOption {
  const enhancements = spec as typeof spec & CartesianEnhancements;
  const sorted = [...rows];
  if (spec.sort === "ascending" || spec.sort === "descending") {
    const direction = spec.sort === "ascending" ? 1 : -1;
    sorted.sort((a, b) => {
      const left = asNumber(a[spec.y_fields[0]]);
      const right = asNumber(b[spec.y_fields[0]]);
      if (left === null && right === null) return 0;
      if (left === null) return 1;
      if (right === null) return -1;
      return (left - right) * direction;
    });
  } else if (spec.sort === "calendar") {
    sorted.sort((a, b) => calendarRank(a[spec.x_field]) - calendarRank(b[spec.x_field]) || text(a[spec.x_field]).localeCompare(text(b[spec.x_field])));
  }
  const focusedScale = focusedPositiveCarbonAxis(spec, sorted);
  const referenceLines = enhancements.reference_lines || [];
  const annotations = enhancements.annotations || [];
  const fittedSeries = enhancements.fitted_series || [];
  if (spec.chart_type === "scatter") {
    const yField = spec.y_fields[0];
    const xTitle = axisTitle(spec.x_field, spec.x_unit || unitFor(columns, spec.x_field));
    const yTitle = axisTitle(yField, spec.y_unit || unitFor(columns, yField));
    const points = sorted.flatMap((row, index) => {
      const xValue = asNumber(row[spec.x_field]);
      const yValue = asNumber(row[yField]);
      return xValue === null || yValue === null
        ? []
        : [{ id: rowIdentity(spec, row, index), value: [xValue, yValue] as [number, number] }];
    });
    const scatterTooltip = (raw: unknown) => {
      const param = (Array.isArray(raw) ? raw[0] : raw) as { seriesName?: string; value?: unknown } | undefined;
      const supplied = Array.isArray(param?.value) ? param?.value : [];
      return [
        `<strong>${escapeHtml(param?.seriesName || "Observation")}</strong>`,
        `${escapeHtml(axisTitle(spec.x_field))}: ${escapeHtml(formattedValue(supplied[0], seriesUnit(columns, spec.x_field, spec.x_unit)))}`,
        `${escapeHtml(axisTitle(yField))}: ${escapeHtml(formattedValue(supplied[1], seriesUnit(columns, yField, spec.y_unit)))}`,
      ].join("<br/>");
    };
    const fitLines = fittedSeries.map((binding, index) => {
      const fitPoints = sorted
        .map((row) => [asNumber(row[binding.x_field]), asNumber(row[binding.y_field])])
        .filter((point): point is [number, number] => point[0] !== null && point[1] !== null)
        .sort((a, b) => a[0] - b[0]);
      return {
        id: `scatter-fit-${binding.id}`,
        name: binding.label,
        type: "line" as const,
        data: fitPoints,
        smooth: false,
        showSymbol: false,
        symbol: "none",
        lineStyle: { color: theme.palette[(index + 1) % theme.palette.length], width: 2, type: "dashed" as const },
        emphasis: { focus: "series" as const },
        tooltip: { show: false },
        z: 5,
      };
    }).filter((series) => series.data.length);
    const flagged = annotations.flatMap((annotation) => sorted
      .filter((row) => row[annotation.condition_field] === true || row[annotation.condition_field] === 1)
      .map((row) => ({
        name: annotation.label,
        value: [asNumber(row[annotation.x_field]), asNumber(row[annotation.y_field])],
      }))
      .filter((point) => point.value[0] !== null && point.value[1] !== null));
    const rugSeries = points.length <= 500 ? [{
      id: "scatter-rugs",
      name: "Observation rugs",
      type: "custom" as const,
      data: points.map((point) => point.value),
      silent: true,
      tooltip: { show: false },
      renderItem: (params: { coordSys?: { x: number; y: number; width: number; height: number } }, api: {
        value: (dimension: number) => number;
        coord: (value: [number, number]) => [number, number];
      }) => {
        const bounds = params.coordSys;
        if (!bounds) return null;
        const point = api.coord([api.value(0), api.value(1)]);
        return {
          type: "group",
          children: [
            {
              type: "line",
              shape: { x1: point[0], y1: bounds.y + bounds.height - 5, x2: point[0], y2: bounds.y + bounds.height },
              style: { stroke: theme.muted, opacity: .28, lineWidth: 1 },
            },
            {
              type: "line",
              shape: { x1: bounds.x, y1: point[1], x2: bounds.x + 5, y2: point[1] },
              style: { stroke: theme.muted, opacity: .28, lineWidth: 1 },
            },
          ],
        };
      },
      z: 1,
    }] : [];
    return {
      ...baseOption(spec, rows, theme),
      grid: { left: 84, right: 36, top: 46, bottom: 82, containLabel: true },
      tooltip: {
        ...tooltipAppearance(theme),
        trigger: "item",
        formatter: scatterTooltip,
        axisPointer: { type: "cross", snap: false },
      },
      legend: fitLines.length
        ? { type: "scroll", top: 4, textStyle: { color: theme.muted, fontSize: 11 }, pageTextStyle: { color: theme.muted } }
        : undefined,
      dataZoom: chartZoom(rows),
      xAxis: valueAxis(theme, xTitle, 52),
      yAxis: valueAxis(theme, yTitle, 62),
      series: [
        {
          id: "scatter-main",
          name: `${axisTitle(spec.x_field)} vs ${axisTitle(yField)}`,
          type: "scatter",
          data: points,
          symbol: "circle",
          symbolSize: points.length > 250 ? 6 : 9,
          itemStyle: { color: theme.palette[0], opacity: points.length > 250 ? .58 : .78, borderColor: theme.surface, borderWidth: 1 },
          emphasis: { focus: "series", scale: 1.35 },
          ...highDensitySeries(rows),
          markLine: referenceLines.length ? {
            silent: true,
            symbol: "none",
            label: { color: theme.text, backgroundColor: theme.surface, padding: [3, 5], formatter: "{b}" },
            data: referenceLines.map((line) => ({
              name: line.label,
              [line.axis === "x" ? "xAxis" : "yAxis"]: line.value,
              lineStyle: { color: theme.palette[3], type: line.line_style || "dashed", width: 1.5 },
            })),
          } : undefined,
        },
        ...fitLines,
        ...rugSeries,
        ...(flagged.length ? [{
          id: "scatter-flagged",
          name: "Flagged observations",
          type: "scatter" as const,
          data: flagged,
          symbol: "diamond",
          symbolSize: 15,
          itemStyle: { color: theme.danger, borderColor: theme.surface, borderWidth: 2 },
          z: 10,
        }] : []),
      ] as EChartsOption["series"],
    };
  }
  const horizontal = spec.orientation === "horizontal";
  const isLine = spec.chart_type === "line" || spec.chart_type === "area";
  const type = isLine ? "line" : "bar";
  const groups = spec.series_field ? Array.from(new Set(sorted.map((row) => text(row[spec.series_field!])))) : [null];
  const baseCategories = Array.from(new Set(sorted.map((row) => text(row[spec.x_field]))));
  const categoryOccurrences = spec.series_field
    ? new Map(baseCategories.map((category) => [
      category,
      Math.max(1, ...groups.map((group) => sorted.filter((row) => text(row[spec.x_field]) === category && text(row[spec.series_field!]) === group).length)),
    ]))
    : new Map(baseCategories.map((category) => [category, sorted.filter((row) => text(row[spec.x_field]) === category).length]));
  const categorySlots = baseCategories.flatMap((category) => Array.from({ length: categoryOccurrences.get(category) || 1 }, (_, occurrence) => ({ category, occurrence })));
  const categories = categorySlots.map((slot) => slot.category);
  const temporal = isLine && sorted.length > 1 && sorted.every((row) => Number.isFinite(Date.parse(text(row[spec.x_field]))));
  const resolveValue = (field: string, group: string | null, category: string, occurrence: number) => {
    const matching = sorted.filter((row) => text(row[spec.x_field]) === category && (!group || text(row[spec.series_field!]) === group));
    return asNumber(matching[occurrence]?.[field]);
  };
  const yUnit = spec.y_unit || unitFor(columns, spec.y_fields[0]);
  const isValidatedShare = spec.stacked && /%|percent/i.test(yUnit || "");
  const standardTooltip = (raw: unknown) => {
    const params = (Array.isArray(raw) ? raw : [raw]) as Array<{ axisValue?: unknown; seriesName?: string; value?: unknown; marker?: string }>;
    const header = axisText(params[0]?.axisValue ?? (Array.isArray(params[0]?.value) ? params[0]?.value?.[0] : ""));
    const lines = [`<strong>${escapeHtml(header)}</strong>`];
    params.forEach((param) => {
      if (param.seriesName?.startsWith("__")) return;
      const supplied = Array.isArray(param.value) ? param.value[1] : param.value;
      if (supplied == null || !Number.isFinite(Number(supplied))) return;
      lines.push(`${param.marker || ""}${escapeHtml(param.seriesName || "Value")}: ${escapeHtml(formattedValue(supplied, yUnit))}`);
    });
    return lines.join("<br/>");
  };
  const rankingProfile = horizontal
    && !isLine
    && !spec.stacked
    && spec.y_fields.length === 1
    && groups.length === 1
    && !spec.series_field;
  if (rankingProfile) {
    const field = spec.y_fields[0];
    const rankingValues = categorySlots.map((slot, index) => {
      const value = resolveValue(field, null, slot.category, slot.occurrence);
      const sourceRow = sorted.find((row) => text(row[spec.x_field]) === slot.category);
      return {
        id: sourceRow ? rowIdentity(spec, sourceRow, index) : `${spec.id}-ranking-${index}`,
        category: slot.category,
        value,
      };
    });
    const rankingTooltip = (raw: unknown) => {
      const param = (Array.isArray(raw) ? raw[0] : raw) as { value?: unknown; name?: string } | undefined;
      const supplied = Array.isArray(param?.value) ? param?.value : [];
      return [
        `<strong>${escapeHtml(param?.name || supplied[1])}</strong>`,
        `${escapeHtml(axisTitle(field))}: ${escapeHtml(formattedValue(supplied[0], yUnit))}`,
      ].join("<br/>");
    };
    return {
      ...baseOption(spec, rows, theme),
      grid: { left: 54, right: 112, top: 34, bottom: rows.length > 45 ? 58 : 38, containLabel: true },
      tooltip: { ...tooltipAppearance(theme), trigger: "item", formatter: rankingTooltip },
      dataZoom: chartZoom(rows, "y"),
      xAxis: valueAxis(theme, axisTitle(field, yUnit), 52),
      yAxis: {
        ...categoryAxis(theme, rankingValues.map((item) => item.category)),
        inverse: true,
        axisTick: { show: false },
      },
      series: [
        {
          id: "ranking-stems-helper",
          name: "__ranking-stems",
          type: "bar",
          data: rankingValues.map((item) => item.value),
          barWidth: 3,
          silent: true,
          itemStyle: { color: theme.grid },
          z: 1,
        },
        {
          id: "ranking-points",
          name: axisTitle(field),
          type: "scatter",
          data: rankingValues
            .filter((item) => item.value !== null)
            .map((item, index) => ({
              id: item.id,
              name: item.category,
              value: [item.value, item.category],
              symbolSize: index === 0 && spec.sort === "descending" ? 13 : 10,
              itemStyle: {
                color: index === 0 && spec.sort === "descending" ? theme.palette[1] : theme.palette[0],
                borderColor: theme.surface,
                borderWidth: 2,
              },
              label: rows.length <= 40 ? {
                show: true,
                position: (item.value || 0) < 0 ? "left" : "right",
                color: theme.text,
                fontSize: 10,
                formatter: formattedValue(item.value, yUnit),
              } : undefined,
            })),
          symbol: "circle",
          emphasis: { focus: "series", scale: 1.25 },
          markLine: referenceLines.length ? {
            silent: true,
            symbol: "none",
            label: { color: theme.text, backgroundColor: theme.surface, padding: [3, 5], formatter: "{b}" },
            data: referenceLines.filter((line) => line.axis === "y").map((line) => ({
              name: line.label,
              xAxis: line.value,
              lineStyle: { color: theme.palette[3], type: line.line_style || "dashed", width: 1.5 },
            })),
          } : undefined,
          z: 4,
        },
      ] as EChartsOption["series"],
    };
  }
  const baseSeries = groups.flatMap((group, groupIndex) => spec.y_fields.map((field, fieldIndex) => {
    const index = groupIndex * spec.y_fields.length + fieldIndex;
    const color = theme.palette[index % theme.palette.length];
    const values = temporal
      ? sorted
        .filter((row) => !group || text(row[spec.series_field!]) === group)
        .map((row) => [Date.parse(text(row[spec.x_field])), asNumber(row[field])])
      : categorySlots.map((slot) => resolveValue(field, group, slot.category, slot.occurrence));
    const exactLabel = !isLine && (horizontal ? rows.length <= 40 : rows.length <= 14);
    return {
      id: index === 0 ? "cartesian-main" : `cartesian-series-${index}`,
      name: group ? `${group} · ${field.replaceAll("_", " ")}` : field.replaceAll("_", " "),
      type,
      smooth: false,
      stack: spec.stacked ? "total" : undefined,
      barWidth: !isLine && spec.stacked ? 34 : undefined,
      symbol: isLine ? lineSymbols[index % lineSymbols.length] : undefined,
      symbolSize: isLine ? 7 : undefined,
      showSymbol: isLine && rows.length <= 24,
      lineStyle: isLine ? { color, width: 2, type: lineTypes[index % lineTypes.length] } : undefined,
      areaStyle: spec.chart_type === "area" ? { color, opacity: .12 } : undefined,
      data: values,
      endLabel: isLine && groups.length * spec.y_fields.length <= 4 ? {
        show: true,
        color,
        fontSize: 10,
        fontWeight: 600,
        formatter: ({ value }: { value?: unknown }) => formattedValue(Array.isArray(value) ? value[1] : value, yUnit),
      } : undefined,
      labelLayout: isLine ? { moveOverlap: "shiftY" as const } : undefined,
      itemStyle: {
        color,
        borderColor: isLine || spec.stacked ? theme.surface : theme.tooltipBorder,
        borderWidth: isLine || spec.stacked ? 1 : 0,
        decal: !isLine && groups.length * spec.y_fields.length > 1 ? seriesDecal(index, theme) : undefined,
      },
      label: exactLabel || spec.stacked ? {
        show: true,
        position: spec.stacked ? "inside" as const : horizontal ? "right" as const : "top" as const,
        color: theme.text,
        fontSize: 10,
        formatter: ({ value }: { value?: unknown }) => {
          const supplied = Array.isArray(value) ? value[1] : value;
          const numeric = asNumber(supplied);
          if (isValidatedShare && numeric !== null && numeric < 6) return "";
          return formattedValue(supplied, yUnit);
        },
      } : undefined,
      emphasis: { focus: "series" as const },
      clip: false,
      ...(rows.length > 1_000 ? { progressive: 2_000, progressiveThreshold: 1_000, ...(isLine ? {} : { large: true, largeThreshold: 1_000 }) } : {}),
    };
  }));
  const intervalSeries = ((enhancements.interval_bands || []) as CartesianIntervalBand[]).flatMap((band, bandIndex) => {
    const bandRows = temporal ? sorted : categorySlots.map((slot) => sorted.find((row) => text(row[spec.x_field]) === slot.category));
    const lower = bandRows.map((row) => row ? asNumber(row[band.lower_field]) : null);
    const upper = bandRows.map((row) => row ? asNumber(row[band.upper_field]) : null);
    const width = bandRows.map((row, index) => {
      if (!row || lower[index] === null) return null;
      return upper[index] === null ? null : Math.max(0, upper[index]! - lower[index]!);
    });
    const withDomain = (values: Array<number | null>) => temporal
      ? values.map((value, index) => [Date.parse(text(sorted[index]?.[spec.x_field])), value])
      : values;
    const rangeData = bandRows.flatMap((row, index) => {
      if (!row || lower[index] === null || upper[index] === null) return [];
      return [[
        temporal ? Date.parse(text(row[spec.x_field])) : text(row[spec.x_field]),
        lower[index],
        upper[index],
      ]];
    });
    const whiskers = {
      id: `interval-whiskers-${band.id}`,
      name: `${band.label} range`,
      type: "custom" as const,
      data: rangeData,
      encode: { x: 0, y: [1, 2] },
      silent: true,
      tooltip: { show: false },
      renderItem: (params: { coordSys?: { x: number; y: number; width: number; height: number } }, api: {
        value: (dimension: number) => string | number;
        coord: (value: [string | number, number]) => [number, number];
      }) => {
        if (!params.coordSys) return null;
        const low = api.coord([api.value(0), Number(api.value(1))]);
        const high = api.coord([api.value(0), Number(api.value(2))]);
        const cap = band.display === "whisker" ? 5 : 3;
        return {
          type: "group",
          children: [
            { type: "line", shape: { x1: low[0], y1: low[1], x2: high[0], y2: high[1] }, style: { stroke: theme.palette[(bandIndex + 2) % theme.palette.length], opacity: band.display === "whisker" ? .95 : .42, lineWidth: band.display === "whisker" ? 1.5 : 1 } },
            { type: "line", shape: { x1: low[0] - cap, y1: low[1], x2: low[0] + cap, y2: low[1] }, style: { stroke: theme.palette[(bandIndex + 2) % theme.palette.length], opacity: .78, lineWidth: 1 } },
            { type: "line", shape: { x1: high[0] - cap, y1: high[1], x2: high[0] + cap, y2: high[1] }, style: { stroke: theme.palette[(bandIndex + 2) % theme.palette.length], opacity: .78, lineWidth: 1 } },
          ],
        };
      },
      z: 4,
    };
    const pointSeries = band.point_field ? [{
      id: `interval-point-${band.id}`,
      name: band.label,
      type: "scatter" as const,
      data: bandRows.flatMap((row) => {
        if (!row) return [];
        const point = asNumber(row[band.point_field!]);
        return point === null ? [] : [[temporal ? Date.parse(text(row[spec.x_field])) : text(row[spec.x_field]), point]];
      }),
      symbol: "circle",
      symbolSize: 9,
      itemStyle: { color: theme.palette[(bandIndex + 2) % theme.palette.length], borderColor: theme.surface, borderWidth: 2 },
      emphasis: { focus: "series" as const },
      z: 6,
    }] : [];
    if (band.display === "whisker") return [whiskers, ...pointSeries];
    return [
      {
        id: `interval-base-${band.id}`,
        name: `__${band.id}-base`,
        type: "line" as const,
        stack: `interval-${band.id}`,
        data: withDomain(lower),
        symbol: "none",
        lineStyle: { opacity: 0 },
        areaStyle: { opacity: 0 },
        silent: true,
        tooltip: { show: false },
        z: 1,
      },
      {
        id: `interval-fill-${band.id}`,
        name: band.label,
        type: "line" as const,
        stack: `interval-${band.id}`,
        data: withDomain(width),
        symbol: "none",
        lineStyle: { opacity: 0 },
        areaStyle: { color: theme.palette[(bandIndex + 2) % theme.palette.length], opacity: .14 },
        silent: true,
        tooltip: { show: false },
        z: 1,
      },
      {
        id: `interval-lower-${band.id}`,
        name: `${band.label} lower`,
        type: "line" as const,
        data: withDomain(lower),
        symbol: "none",
        showSymbol: false,
        lineStyle: { color: theme.palette[(bandIndex + 2) % theme.palette.length], opacity: .42, width: 1 },
        silent: true,
        tooltip: { show: false },
        z: 2,
      },
      {
        id: `interval-upper-${band.id}`,
        name: `${band.label} upper`,
        type: "line" as const,
        data: withDomain(upper),
        symbol: "none",
        showSymbol: false,
        lineStyle: { color: theme.palette[(bandIndex + 2) % theme.palette.length], opacity: .42, width: 1 },
        silent: true,
        tooltip: { show: false },
        z: 2,
      },
      ...(rangeData.length <= 24 ? [whiskers] : []),
      ...pointSeries,
    ];
  });
  const overlaySeries = fittedSeries.map((binding, index) => ({
    id: `cartesian-fit-${binding.id}`,
    name: binding.label,
    type: "line" as const,
    smooth: false,
    showSymbol: false,
    symbol: "none",
    data: temporal
      ? sorted.map((row) => [Date.parse(text(row[binding.x_field])), asNumber(row[binding.y_field])])
      : sorted.map((row) => asNumber(row[binding.y_field])),
    lineStyle: { color: theme.palette[(baseSeries.length + index) % theme.palette.length], width: 2, type: "dashed" as const },
    emphasis: { focus: "series" as const },
    z: 6,
  })).filter((series) => series.data.some((entry) => Array.isArray(entry) ? entry[1] !== null : entry !== null));
  const highlightSeries = spec.highlight ? [{
    id: "cartesian-highlight",
    name: spec.highlight.label,
    type: "scatter" as const,
    data: sorted.filter((row) => row[spec.highlight!.condition_field] === true || row[spec.highlight!.condition_field] === 1 || text(row[spec.highlight!.condition_field]).toLowerCase() === "true").map((row) => horizontal
      ? [asNumber(row[spec.highlight!.value_field]), text(row[spec.x_field])]
      : [temporal ? Date.parse(text(row[spec.x_field])) : text(row[spec.x_field]), asNumber(row[spec.highlight!.value_field])]),
    symbol: "diamond",
    symbolSize: 14,
    itemStyle: { color: theme.danger, borderColor: theme.surface, borderWidth: 2 },
    z: 10,
  }] : [];
  const annotationSeries = annotations.map((annotation) => ({
    id: `cartesian-annotation-${annotation.id}`,
    name: annotation.label,
    type: "scatter" as const,
    data: sorted
      .filter((row) => row[annotation.condition_field] === true || row[annotation.condition_field] === 1)
      .map((row) => [
        temporal ? Date.parse(text(row[annotation.x_field])) : text(row[annotation.x_field]),
        asNumber(row[annotation.y_field]),
      ]),
    symbol: "diamond",
    symbolSize: 15,
    itemStyle: { color: theme.danger, borderColor: theme.surface, borderWidth: 2 },
    z: 11,
  }));
  const mainMarkLines = referenceLines.map((line) => ({
    name: line.label,
    [line.axis === "x" ? "xAxis" : "yAxis"]: line.axis === "x" && temporal && typeof line.value === "string"
      ? Date.parse(line.value)
      : line.value,
    lineStyle: { color: theme.palette[3], type: line.line_style || "dashed", width: 1.5 },
  }));
  if (mainMarkLines.length && baseSeries.length) {
    Object.assign(baseSeries[0], {
      markLine: {
        silent: true,
        symbol: "none",
        label: { color: theme.text, backgroundColor: theme.surface, padding: [3, 5], formatter: "{b}" },
        data: mainMarkLines,
      },
    });
  }
  const series = [...intervalSeries, ...baseSeries, ...overlaySeries, ...highlightSeries, ...annotationSeries];
  const hasContextTrack = temporal && rows.length > 45 && baseSeries.length > 0;
  const contextSeries = hasContextTrack ? [{
    id: "cartesian-context-helper",
    name: "__context",
    type: "line" as const,
    xAxisIndex: 1,
    yAxisIndex: 1,
    data: baseSeries[0].data,
    smooth: false,
    showSymbol: false,
    symbol: "none",
    silent: true,
    tooltip: { show: false },
    lineStyle: { color: theme.palette[0], width: 1, opacity: .72 },
    areaStyle: { color: theme.palette[0], opacity: .08 },
    z: 1,
  }] : [];
  const legendNames = series
    .filter((item) => {
      const id = "id" in item ? text(item.id) : "";
      return !id.includes("interval-base") && !id.includes("interval-lower") && !id.includes("interval-upper") && !id.includes("interval-whiskers");
    })
    .map((item) => item.name)
    .filter((name) => !name.startsWith("__"));
  const zoom = chartZoom(rows, horizontal ? "y" : "x");
  const linkedZoom = hasContextTrack && Array.isArray(zoom)
    ? zoom.map((control) => ({ ...control, xAxisIndex: [0, 1] }))
    : zoom;
  return {
    ...baseOption(spec, rows, theme),
    legend: series.length > 1
      ? {
        type: "scroll",
        top: 4,
        left: 16,
        right: 16,
        itemWidth: 18,
        itemHeight: 8,
        textStyle: { color: theme.muted, fontSize: 11 },
        pageTextStyle: { color: theme.muted },
        data: legendNames,
      }
      : undefined,
    tooltip: {
      ...tooltipAppearance(theme),
      trigger: "axis",
      formatter: standardTooltip,
      axisPointer: {
        type: "cross",
        snap: true,
        lineStyle: { color: theme.muted, width: 1, type: "dashed" },
        crossStyle: { color: theme.muted, width: 1, type: "dashed" },
      },
    },
    dataZoom: linkedZoom,
    grid: hasContextTrack
      ? [
        {
          left: 64,
          right: 104,
          top: series.length > 1 ? 64 : 48,
          bottom: 122,
          containLabel: true,
        },
        { left: 64, right: 104, bottom: 48, height: 34, containLabel: false },
      ]
      : {
        left: horizontal ? 52 : 64,
        right: horizontal ? 82 : temporal ? 104 : 36,
        top: series.length > 1 ? 64 : 48,
        bottom: rows.length > 45 && !horizontal ? 92 : 68,
        containLabel: true,
      },
    xAxis: hasContextTrack
      ? [
        timeAxis(theme, spec.x_unit),
        {
          ...timeAxis(theme),
          gridIndex: 1,
          axisLabel: { show: false },
          axisTick: { show: false },
          axisLine: { lineStyle: { color: theme.grid, width: 1 } },
        },
      ]
      : horizontal
        ? valueAxis(theme, spec.y_unit)
        : temporal
          ? timeAxis(theme, spec.x_unit)
          : categoryAxis(theme, categories, spec.x_unit, categories.length > 8 ? 32 : 0),
    yAxis: hasContextTrack
      ? [
        valueAxis(theme, spec.y_unit, 42, focusedScale),
        {
          type: "value",
          gridIndex: 1,
          scale: true,
          show: false,
        },
      ]
      : horizontal
        ? categoryAxis(theme, categories)
        : valueAxis(theme, spec.y_unit, 42, focusedScale),
    series: [...series, ...contextSeries] as EChartsOption["series"],
  };
}

function forecastOption(spec: Extract<VisualizationSpec, { kind: "forecast" }>, rows: DataRow[], theme: ChartTheme): EChartsOption {
  const enhancements = spec as typeof spec & ForecastEnhancements;
  const dates = rows.map((row) => text(row[spec.date_field]));
  const temporal = dates.every((value) => Number.isFinite(Date.parse(value)));
  const lower = rows.map((row) => asNumber(row[spec.lower_field]));
  const upper = rows.map((row) => asNumber(row[spec.upper_field]));
  const intervalWidth = upper.map((value, index) => value === null || lower[index] === null ? null : Math.max(0, value - lower[index]!));
  const withDomain = (values: Array<number | null>) => temporal
    ? values.map((value, index) => [Date.parse(dates[index]), value])
    : values;
  const field = (name: string, color: string, index: number, id: string) => ({
    id,
    name: axisTitle(name),
    type: "line" as const,
    smooth: false,
    connectNulls: false,
    data: withDomain(rows.map((row) => asNumber(row[name]))),
    symbol: lineSymbols[index % lineSymbols.length],
    symbolSize: 7,
    showSymbol: rows.length <= 24,
    lineStyle: { color, width: 2, type: lineTypes[index % lineTypes.length] },
    itemStyle: { color, borderColor: theme.surface, borderWidth: 1 },
    endLabel: {
      show: true,
      color,
      fontSize: 10,
      fontWeight: 600,
      formatter: ({ value }: { value?: unknown }) => formattedValue(Array.isArray(value) ? value[1] : value, spec.unit),
    },
    labelLayout: { moveOverlap: "shiftY" as const },
    emphasis: { focus: "series" as const },
    ...(rows.length > 1_000 ? { progressive: 2_000, progressiveThreshold: 1_000 } : {}),
    z: 4,
  });
  const intervalLabel = `${Math.round((enhancements.interval_level || .8) * 100)}% interval`;
  const rangeData = rows.flatMap((row, index) => lower[index] === null || upper[index] === null
    ? []
    : [[temporal ? Date.parse(dates[index]) : dates[index], lower[index], upper[index]]]);
  const rangeWhiskers = rangeData.length <= 24 ? [{
    id: "forecast-range-whiskers",
    name: `${intervalLabel} whiskers`,
    type: "custom" as const,
    data: rangeData,
    silent: true,
    tooltip: { show: false },
    renderItem: (params: { coordSys?: { x: number; y: number; width: number; height: number } }, api: {
      value: (dimension: number) => string | number;
      coord: (value: [string | number, number]) => [number, number];
    }) => {
      if (!params.coordSys) return null;
      const low = api.coord([api.value(0), Number(api.value(1))]);
      const high = api.coord([api.value(0), Number(api.value(2))]);
      return {
        type: "group",
        children: [
          { type: "line", shape: { x1: low[0], y1: low[1], x2: high[0], y2: high[1] }, style: { stroke: theme.interval, opacity: .48, lineWidth: 1 } },
          { type: "line", shape: { x1: low[0] - 3, y1: low[1], x2: low[0] + 3, y2: low[1] }, style: { stroke: theme.interval, opacity: .72, lineWidth: 1 } },
          { type: "line", shape: { x1: high[0] - 3, y1: high[1], x2: high[0] + 3, y2: high[1] }, style: { stroke: theme.interval, opacity: .72, lineWidth: 1 } },
        ],
      };
    },
    z: 3,
  }] : [];
  const series = [
    { id: "forecast-interval-base-helper", name: "Interval base", type: "line" as const, stack: "forecast-interval", data: withDomain(lower), symbol: "none", lineStyle: { opacity: 0 }, areaStyle: { opacity: 0 }, silent: true, z: 1 },
    { id: "forecast-interval-fill", name: intervalLabel, type: "line" as const, stack: "forecast-interval", data: withDomain(intervalWidth), symbol: "none", lineStyle: { opacity: 0 }, areaStyle: { color: theme.interval, opacity: .16 }, silent: true, z: 1 },
    { id: "forecast-interval-lower", name: `${intervalLabel} lower`, type: "line" as const, data: withDomain(lower), symbol: "none", showSymbol: false, lineStyle: { color: theme.interval, width: 1, opacity: .48 }, silent: true, tooltip: { show: false }, z: 2 },
    { id: "forecast-interval-upper", name: `${intervalLabel} upper`, type: "line" as const, data: withDomain(upper), symbol: "none", showSymbol: false, lineStyle: { color: theme.interval, width: 1, opacity: .48 }, silent: true, tooltip: { show: false }, z: 2 },
    ...rangeWhiskers,
    spec.actual_field && field(spec.actual_field, theme.palette[2], 0, "forecast-actual"),
    field(spec.predicted_field, theme.palette[0], 1, "forecast-predicted"),
  ].filter(Boolean);
  const firstAnalyticalSeries = series.find((item) => item && !item.name.startsWith("Interval"));
  if (enhancements.forecast_boundary && firstAnalyticalSeries) {
    Object.assign(firstAnalyticalSeries, {
      markLine: {
        silent: true,
        symbol: "none",
        label: {
          color: theme.text,
          backgroundColor: theme.surface,
          padding: [3, 5],
          formatter: "Forecast begins",
        },
        data: [{
          name: "Forecast begins",
          xAxis: temporal ? Date.parse(enhancements.forecast_boundary) : enhancements.forecast_boundary,
          lineStyle: { color: theme.palette[1], type: "dashed", width: 1.5 },
        }],
      },
      markArea: enhancements.forecast_boundary ? {
        silent: true,
        label: {
          show: true,
          position: "insideTopLeft",
          color: theme.muted,
          fontSize: 9,
          fontWeight: 600,
          formatter: "FORECAST REGION",
        },
        itemStyle: { color: theme.interval, opacity: .055 },
        data: [[
          { xAxis: temporal ? Date.parse(enhancements.forecast_boundary) : enhancements.forecast_boundary },
          { xAxis: temporal ? Date.parse(dates[dates.length - 1]) : dates[dates.length - 1] },
        ]],
      } : undefined,
    });
  }
  const tooltip = (raw: unknown) => {
    const entries = Array.isArray(raw) ? raw as Array<{ dataIndex?: number }> : [];
    const index = entries[0]?.dataIndex ?? 0;
    const suffix = spec.unit ? ` ${escapeHtml(spec.unit)}` : "";
    const lines = [`<strong>${escapeHtml(axisText(dates[index]))}</strong>`];
    if (spec.actual_field) {
      const actual = asNumber(rows[index]?.[spec.actual_field]);
      if (actual !== null) lines.push(`${escapeHtml(axisTitle(spec.actual_field))}: ${numberFormatter.format(actual)}${suffix}`);
    }
    const predicted = asNumber(rows[index]?.[spec.predicted_field]);
    if (predicted !== null) lines.push(`${escapeHtml(axisTitle(spec.predicted_field))}: ${numberFormatter.format(predicted)}${suffix}`);
    if (lower[index] !== null && upper[index] !== null) lines.push(`${intervalLabel}: ${numberFormatter.format(lower[index]!)}–${numberFormatter.format(upper[index]!)}${suffix}`);
    return lines.join("<br/>");
  };
  return {
    ...baseOption(spec, rows, theme),
    tooltip: { ...tooltipAppearance(theme), trigger: "axis", formatter: tooltip },
    legend: {
      type: "scroll",
      top: 4,
      data: [spec.actual_field && axisTitle(spec.actual_field), axisTitle(spec.predicted_field), intervalLabel].filter(Boolean) as string[],
      textStyle: { color: theme.muted, fontSize: 11 },
      pageTextStyle: { color: theme.muted },
    },
    dataZoom: chartZoom(rows),
    grid: { left: 72, right: 104, top: 58, bottom: rows.length > 45 ? 92 : 68, containLabel: true },
    xAxis: temporal
      ? timeAxis(theme)
      : categoryAxis(theme, dates, undefined, dates.length > 8 ? 32 : 0),
    yAxis: valueAxis(theme, axisTitle(spec.predicted_field, spec.unit), 58),
    series: series as EChartsOption["series"],
  };
}

function distributionOption(
  spec: Extract<VisualizationSpec, { kind: "distribution" }>,
  rows: DataRow[],
  theme: ChartTheme,
  outlierRows: DataRow[] = [],
): EChartsOption {
  const enhancements = spec as typeof spec & DistributionEnhancements;
  const values = rows.map((row) => asNumber(row[spec.value_field])).filter((value): value is number => value !== null);
  if (!values.length && !enhancements.five_number_summary) return { ...baseOption(spec, rows, theme), title: { text: "No valid distribution values", left: "center", top: "middle", textStyle: { color: theme.muted, fontSize: 13, fontWeight: 500 } } };
  if (spec.chart_type === "boxplot") {
    const grouped = new Map<string, number[]>();
    rows.forEach((row) => {
      const value = asNumber(row[spec.value_field]);
      if (value === null) return;
      const category = spec.category_field ? text(row[spec.category_field]) : "Distribution";
      grouped.set(category, [...(grouped.get(category) || []), value]);
    });
    let summaries = Array.from(grouped, ([category, groupValues]) => ({
      category,
      values: [Math.min(...groupValues), quantile(groupValues, .25), quantile(groupValues, .5), quantile(groupValues, .75), Math.max(...groupValues)],
    }));
    if (enhancements.five_number_summary) {
      const summary = enhancements.five_number_summary;
      summaries = [{
        category: "Distribution",
        values: [summary.lower_whisker, summary.q1, summary.median, summary.q3, summary.upper_whisker],
      }];
    }
    const tooltip = (raw: unknown) => {
      const params = raw as { name?: string; value?: unknown };
      const supplied = Array.isArray(params.value) ? params.value.map(Number).slice(-5) : [];
      const suffix = spec.unit ? ` ${escapeHtml(spec.unit)}` : "";
      if (supplied.length !== 5 || supplied.some((value) => !Number.isFinite(value))) return escapeHtml(params.name);
      if (enhancements.five_number_summary) {
        const summary = enhancements.five_number_summary;
        return `<strong>${escapeHtml(params.name)}</strong><br/>Minimum: ${numberFormatter.format(summary.minimum)}${suffix}<br/>Lower whisker: ${numberFormatter.format(supplied[0])}${suffix}<br/>Q1: ${numberFormatter.format(supplied[1])}${suffix}<br/>Median: ${numberFormatter.format(supplied[2])}${suffix}<br/>Q3: ${numberFormatter.format(supplied[3])}${suffix}<br/>Upper whisker: ${numberFormatter.format(supplied[4])}${suffix}<br/>Maximum: ${numberFormatter.format(summary.maximum)}${suffix}`;
      }
      return `<strong>${escapeHtml(params.name)}</strong><br/>Minimum: ${numberFormatter.format(supplied[0])}${suffix}<br/>Q1: ${numberFormatter.format(supplied[1])}${suffix}<br/>Median: ${numberFormatter.format(supplied[2])}${suffix}<br/>Q3: ${numberFormatter.format(supplied[3])}${suffix}<br/>Maximum: ${numberFormatter.format(supplied[4])}${suffix}`;
    };
    const categoryIndex = (row: DataRow) => {
      if (!spec.category_field || summaries.length === 1) return 0;
      const suppliedCategory = text(row[spec.category_field]);
      const index = summaries.findIndex((summary) => summary.category === suppliedCategory);
      return index >= 0 ? index : 0;
    };
    const explicitOutliers = outlierRows
      .flatMap((row) => {
        const value = asNumber(row[enhancements.outlier_value_field || spec.value_field]);
        return value === null ? [] : [[categoryIndex(row), value]];
      });
    const localOutliers = enhancements.outlier_condition_field
      ? rows
        .filter((row) => row[enhancements.outlier_condition_field!] === true || row[enhancements.outlier_condition_field!] === 1)
        .flatMap((row) => {
          const value = asNumber(row[enhancements.outlier_value_field || spec.value_field]);
          return value === null ? [] : [[categoryIndex(row), value]];
        })
      : [];
    return {
      ...baseOption(spec, rows, theme),
      tooltip: { ...tooltipAppearance(theme), trigger: "item", formatter: tooltip },
      xAxis: categoryAxis(theme, summaries.map((summary) => summary.category)),
      yAxis: valueAxis(theme, axisTitle(spec.value_field, spec.unit), 62),
      series: [
        {
          id: "distribution-main",
          name: axisTitle(spec.value_field),
          type: "boxplot",
          data: summaries.map((summary) => summary.values),
          itemStyle: { color: `${theme.palette[0]}30`, borderColor: theme.palette[0], borderWidth: 2 },
          emphasis: { focus: "series" },
        },
        ...((explicitOutliers.length || localOutliers.length) ? [{
          id: "distribution-outliers",
          name: "Outliers",
          type: "scatter" as const,
          data: explicitOutliers.length ? explicitOutliers : localOutliers,
          symbol: "diamond",
          symbolSize: 9,
          itemStyle: { color: theme.danger, borderColor: theme.surface, borderWidth: 1 },
          z: 7,
        }] : []),
      ] as EChartsOption["series"],
    };
  }
  if (spec.chart_type === "percentile" && spec.category_field) {
    const suppliedMarkers = rows.map((row) => ({ category: text(row[spec.category_field!]), value: asNumber(row[spec.value_field]) })).filter((marker): marker is { category: string; value: number } => marker.value !== null);
    return {
      ...baseOption(spec, rows, theme),
      dataZoom: chartZoom(rows),
      xAxis: categoryAxis(theme, suppliedMarkers.map((marker) => marker.category)),
      yAxis: valueAxis(theme, spec.unit),
      series: [{
        id: "distribution-main",
        name: spec.value_field.replaceAll("_", " "),
        type: "bar",
        data: suppliedMarkers.map((marker) => marker.value),
        itemStyle: { color: theme.palette[0], borderColor: theme.tooltipBorder, borderWidth: 1 },
        emphasis: { focus: "series" },
        label: {
          show: true,
          position: "top",
          formatter: ({ value }: { value?: unknown }) => `${numberFormatter.format(Number(value))}${spec.unit ? ` ${spec.unit}` : ""}`,
          color: theme.text,
          fontSize: 11,
        },
      }],
    };
  }
  if (spec.count_field) {
    const suppliedBins = rows.map((row) => ({
      midpoint: asNumber(row[spec.value_field]),
      lower: enhancements.bin_lower_field ? asNumber(row[enhancements.bin_lower_field]) : null,
      upper: enhancements.bin_upper_field ? asNumber(row[enhancements.bin_upper_field]) : null,
      count: asNumber(row[spec.count_field!]),
    })).filter((bin): bin is { midpoint: number; lower: number | null; upper: number | null; count: number } => bin.midpoint !== null && bin.count !== null).sort((a, b) => a.midpoint - b.midpoint);
    const labels = suppliedBins.map((bin) => bin.lower !== null && bin.upper !== null
      ? `${numberFormatter.format(bin.lower)}–${numberFormatter.format(bin.upper)}`
      : numberFormatter.format(bin.midpoint));
    const summary = enhancements.five_number_summary;
    const containingBin = (value: number) => {
      const index = suppliedBins.findIndex((bin) => bin.lower !== null && bin.upper !== null && value >= bin.lower && value <= bin.upper);
      return labels[index >= 0 ? index : suppliedBins.reduce((best, bin, candidate) =>
        Math.abs(bin.midpoint - value) < Math.abs(suppliedBins[best].midpoint - value) ? candidate : best, 0)];
    };
    if (summary && suppliedBins.length) {
      const inferredWidth = suppliedBins.length > 1
        ? Math.max(Number.EPSILON, Math.abs(suppliedBins[1].midpoint - suppliedBins[0].midpoint))
        : 1;
      const continuousBins = suppliedBins.map((bin) => ({
        ...bin,
        lower: bin.lower ?? bin.midpoint - inferredWidth / 2,
        upper: bin.upper ?? bin.midpoint + inferredWidth / 2,
      }));
      const outlierValues = (outlierRows.length ? outlierRows : rows.filter((row) =>
        enhancements.outlier_condition_field
        && (row[enhancements.outlier_condition_field] === true || row[enhancements.outlier_condition_field] === 1)))
        .map((row) => asNumber(row[enhancements.outlier_value_field || spec.value_field]))
        .filter((value): value is number => value !== null);
      const histogramTooltip = (raw: unknown) => {
        const param = (Array.isArray(raw) ? raw[0] : raw) as { seriesName?: string; value?: unknown } | undefined;
        const supplied = Array.isArray(param?.value) ? param.value.map(Number) : [];
        if (param?.seriesName === "Tukey summary") {
          return [
            "<strong>Tukey summary</strong>",
            `Minimum: ${escapeHtml(formattedValue(summary.minimum, spec.unit))}`,
            `Lower whisker: ${escapeHtml(formattedValue(summary.lower_whisker, spec.unit))}`,
            `Q1: ${escapeHtml(formattedValue(summary.q1, spec.unit))}`,
            `Median: ${escapeHtml(formattedValue(summary.median, spec.unit))}`,
            `Q3: ${escapeHtml(formattedValue(summary.q3, spec.unit))}`,
            `Upper whisker: ${escapeHtml(formattedValue(summary.upper_whisker, spec.unit))}`,
            `Maximum: ${escapeHtml(formattedValue(summary.maximum, spec.unit))}`,
          ].join("<br/>");
        }
        if (supplied.length >= 3) {
          return [
            `<strong>${escapeHtml(formattedValue(supplied[0], spec.unit))}–${escapeHtml(formattedValue(supplied[1], spec.unit))}</strong>`,
            `Observations: ${numberFormatter.format(supplied[2])}`,
          ].join("<br/>");
        }
        return escapeHtml(param?.seriesName);
      };
      return {
        ...baseOption(spec, rows, theme),
        tooltip: { ...tooltipAppearance(theme), trigger: "item", formatter: histogramTooltip },
        grid: [
          { left: 72, right: 38, top: 48, bottom: 152, containLabel: true },
          { left: 72, right: 38, height: 48, bottom: 66, containLabel: false },
        ],
        xAxis: [
          {
            ...valueAxis(theme),
            min: "dataMin",
            max: "dataMax",
            axisLabel: { show: false },
          },
          {
            ...valueAxis(theme, axisTitle(spec.value_field, spec.unit), 46),
            gridIndex: 1,
            min: "dataMin",
            max: "dataMax",
          },
        ],
        yAxis: [
          valueAxis(theme, "Observations"),
          {
            ...categoryAxis(theme, ["Tukey summary"]),
            gridIndex: 1,
            axisTick: { show: false },
            splitLine: { show: false },
          },
        ],
        series: [
          {
            id: "distribution-histogram",
            name: "Observations",
            type: "custom",
            xAxisIndex: 0,
            yAxisIndex: 0,
            data: continuousBins.map((bin) => [bin.lower, bin.upper, bin.count, bin.midpoint]),
            encode: { x: [0, 1], y: 2 },
            renderItem: (params: { coordSys?: { x: number; y: number; width: number; height: number } }, api: {
              value: (dimension: number) => number;
              coord: (value: [number, number]) => [number, number];
            }) => {
              const bounds = params.coordSys;
              if (!bounds) return null;
              const lowerPoint = api.coord([api.value(0), 0]);
              const upperPoint = api.coord([api.value(1), api.value(2)]);
              const left = Math.max(bounds.x, lowerPoint[0] + .75);
              const right = Math.min(bounds.x + bounds.width, upperPoint[0] - .75);
              const top = Math.max(bounds.y, upperPoint[1]);
              const bottom = Math.min(bounds.y + bounds.height, lowerPoint[1]);
              return {
                type: "rect",
                shape: { x: left, y: top, width: Math.max(1, right - left), height: Math.max(0, bottom - top) },
                style: { fill: theme.palette[0], opacity: .74, stroke: theme.tooltipBorder, lineWidth: .7 },
                emphasis: { style: { fill: theme.palette[0], opacity: 1 } },
              };
            },
            z: 3,
          },
          {
            id: "distribution-summary",
            name: "Tukey summary",
            type: "custom",
            xAxisIndex: 1,
            yAxisIndex: 1,
            data: [[summary.lower_whisker, summary.q1, summary.median, summary.q3, summary.upper_whisker, summary.minimum, summary.maximum]],
            encode: { x: [0, 1, 2, 3, 4], y: -1 },
            renderItem: (_params: unknown, api: {
              value: (dimension: number) => number;
              coord: (value: [number, number]) => [number, number];
            }) => {
              const lower = api.coord([api.value(0), 0]);
              const q1 = api.coord([api.value(1), 0]);
              const median = api.coord([api.value(2), 0]);
              const q3 = api.coord([api.value(3), 0]);
              const upper = api.coord([api.value(4), 0]);
              const minimum = api.coord([api.value(5), 0]);
              const maximum = api.coord([api.value(6), 0]);
              return {
                type: "group",
                children: [
                  { type: "line", shape: { x1: lower[0], y1: lower[1], x2: upper[0], y2: upper[1] }, style: { stroke: theme.muted, lineWidth: 1.5 } },
                  { type: "rect", shape: { x: q1[0], y: q1[1] - 10, width: Math.max(1, q3[0] - q1[0]), height: 20 }, style: { fill: theme.surface, stroke: theme.palette[0], lineWidth: 2 } },
                  { type: "line", shape: { x1: median[0], y1: median[1] - 10, x2: median[0], y2: median[1] + 10 }, style: { stroke: theme.palette[1], lineWidth: 2 } },
                  { type: "line", shape: { x1: lower[0], y1: lower[1] - 6, x2: lower[0], y2: lower[1] + 6 }, style: { stroke: theme.muted, lineWidth: 1.5 } },
                  { type: "line", shape: { x1: upper[0], y1: upper[1] - 6, x2: upper[0], y2: upper[1] + 6 }, style: { stroke: theme.muted, lineWidth: 1.5 } },
                  { type: "circle", shape: { cx: minimum[0], cy: minimum[1], r: 2.5 }, style: { fill: theme.muted } },
                  { type: "circle", shape: { cx: maximum[0], cy: maximum[1], r: 2.5 }, style: { fill: theme.muted } },
                ],
              };
            },
            z: 5,
          },
          ...(outlierValues.length ? [{
            id: "distribution-outliers",
            name: "Outliers",
            type: "scatter" as const,
            xAxisIndex: 1,
            yAxisIndex: 1,
            data: outlierValues.map((value) => [value, 0]),
            symbol: "diamond",
            symbolSize: 8,
            itemStyle: { color: theme.danger, borderColor: theme.surface, borderWidth: 1 },
            z: 7,
          }] : []),
          {
            id: "distribution-summary-markers",
            name: "Summary markers",
            type: "scatter",
            xAxisIndex: 0,
            yAxisIndex: 0,
            data: [],
            silent: true,
            tooltip: { show: false },
            markLine: {
              silent: true,
              symbol: "none",
              label: { color: theme.text, backgroundColor: theme.surface, padding: [3, 5], formatter: "{b}" },
              data: [
                { name: `Median ${formattedValue(summary.median, spec.unit)}`, xAxis: summary.median, lineStyle: { color: theme.palette[1], type: "solid" } },
                ...(summary.p90 == null ? [] : [{ name: `p90 ${formattedValue(summary.p90, spec.unit)}`, xAxis: summary.p90, lineStyle: { color: theme.palette[2], type: "dashed" as const } }]),
              ],
            },
          },
        ] as EChartsOption["series"],
      };
    }
    return {
      ...baseOption(spec, rows, theme),
      dataZoom: chartZoom(rows),
      grid: { left: 64, right: 32, top: 54, bottom: rows.length > 45 ? 98 : 76, containLabel: true },
      xAxis: categoryAxis(theme, labels, spec.unit, suppliedBins.length > 8 ? 32 : 0),
      yAxis: valueAxis(theme, "Observations"),
      series: [{
        id: "distribution-histogram",
        name: "Observations",
        type: "bar",
        data: suppliedBins.map((bin) => bin.count),
        itemStyle: { color: theme.palette[0], borderColor: theme.tooltipBorder, borderWidth: 1 },
        emphasis: { focus: "series" },
        ...(rows.length > 1_000 ? { progressive: 2_000, progressiveThreshold: 1_000, large: true, largeThreshold: 1_000 } : {}),
        markLine: summary ? {
          silent: true,
          symbol: "none",
          label: { color: theme.text, backgroundColor: theme.surface, padding: [3, 5], formatter: "{b}" },
          data: [
            { name: `Median ${formattedValue(summary.median, spec.unit)}`, xAxis: containingBin(summary.median), lineStyle: { color: theme.palette[0], type: "solid" as const } },
            ...(summary.p90 == null ? [] : [{ name: `p90 ${formattedValue(summary.p90, spec.unit)}`, xAxis: containingBin(summary.p90), lineStyle: { color: theme.palette[1], type: "dashed" as const } }]),
          ],
        } : undefined,
      }],
    };
  }
  const min = Math.min(...values), max = Math.max(...values), bins = spec.bins || Math.max(5, Math.min(16, Math.ceil(Math.sqrt(values.length))));
  const width = max === min ? 1 : (max - min) / bins;
  const counts = Array.from({ length: bins }, (_, index) => ({ label: `${numberFormatter.format(min + index * width)}–${numberFormatter.format(min + (index + 1) * width)}`, count: 0 }));
  values.forEach((value) => { counts[Math.min(bins - 1, Math.floor((value - min) / width))].count += 1; });
  const p50 = quantile(values, .5), p90 = quantile(values, .9);
  const nearestBin = (value: number) => counts[Math.min(counts.length - 1, Math.max(0, Math.floor((value - min) / width)))].label;
  return {
    ...baseOption(spec, rows, theme),
    dataZoom: chartZoom(rows),
    xAxis: categoryAxis(theme, counts.map((item) => item.label), spec.unit, 32),
    yAxis: valueAxis(theme, "Observations"),
    series: [{
      id: "distribution-main",
      name: "Observations",
      type: "bar",
      data: counts.map((item) => item.count),
      itemStyle: { color: theme.palette[0], borderColor: theme.tooltipBorder, borderWidth: 1 },
      markLine: spec.chart_type === "percentile" ? {
        symbol: "none",
        label: { color: theme.text, formatter: "{b}", backgroundColor: theme.surface, padding: [3, 5] },
        data: [
          { name: `p50: ${numberFormatter.format(p50)}${spec.unit ? ` ${spec.unit}` : ""}`, xAxis: nearestBin(p50), lineStyle: { color: theme.palette[0], type: "solid" } },
          { name: `p90: ${numberFormatter.format(p90)}${spec.unit ? ` ${spec.unit}` : ""}`, xAxis: nearestBin(p90), lineStyle: { color: theme.palette[1], type: "dashed" } },
        ],
      } : undefined,
    }],
  };
}

function heatmapOption(spec: Extract<VisualizationSpec, { kind: "heatmap" }>, rows: DataRow[], theme: ChartTheme, resolvedTheme: ResolvedTheme): EChartsOption {
  const x = calendarValues(Array.from(new Set(rows.map((row) => text(row[spec.x_field])))));
  const y = calendarValues(Array.from(new Set(rows.map((row) => text(row[spec.y_field])))));
  const data = rows.flatMap((row, index) => {
    const value = asNumber(row[spec.value_field]);
    return value === null ? [] : [{
      id: rowIdentity(spec, row, index),
      value: [x.indexOf(text(row[spec.x_field])), y.indexOf(text(row[spec.y_field])), value],
    }];
  });
  const values = data.map((item) => Number(item.value[2]));
  if (!values.length) {
    return {
      ...baseOption(spec, rows, theme),
      title: {
        text: "No valid heatmap values",
        left: "center",
        top: "middle",
        textStyle: { color: theme.muted, fontSize: 13, fontWeight: 500 },
      },
    };
  }
  const scale = resolvedTheme === "dark"
    ? ["#172B35", "#246A73", "#54A66B", "#D6C95F"]
    : ["#EEF2EA", "#B5D3BB", "#599B77", "#17594D"];
  return {
    ...baseOption(spec, rows, theme),
    grid: { left: 64, right: 28, top: 36, bottom: 84, containLabel: true },
    tooltip: {
      ...tooltipAppearance(theme),
      trigger: "item",
      formatter: (raw: unknown) => {
        const param = (Array.isArray(raw) ? raw[0] : raw) as { value?: unknown } | undefined;
        const supplied = Array.isArray(param?.value) ? param.value : [];
        const xValue = x[Number(supplied[0])] || "—";
        const yValue = y[Number(supplied[1])] || "—";
        return [
          `<strong>${escapeHtml(xValue)} · ${escapeHtml(yValue)}</strong>`,
          `${escapeHtml(axisTitle(spec.value_field))}: ${escapeHtml(formattedValue(supplied[2], spec.unit))}`,
        ].join("<br/>");
      },
    },
    xAxis: categoryAxis(theme, x),
    yAxis: categoryAxis(theme, y),
    visualMap: {
      min: Math.min(...values),
      max: Math.max(...values),
      calculable: true,
      orient: "horizontal",
      left: "center",
      bottom: 6,
      textStyle: { color: theme.muted, fontSize: 11 },
      inRange: { color: scale },
    },
    series: [{
      id: "heatmap-main",
      name: axisTitle(spec.value_field),
      type: "heatmap",
      data,
      label: {
        show: data.length <= 48,
        color: theme.text,
        fontSize: 10,
        textBorderColor: resolvedTheme === "dark" ? "#071014" : "#FFFFFF",
        textBorderWidth: 2,
        formatter: ({ value }: { value?: unknown }) => Array.isArray(value) ? numberFormatter.format(Number(value[2])) : "",
      },
      itemStyle: { borderColor: theme.surface, borderWidth: 1 },
      select: { itemStyle: { borderColor: theme.palette[1], borderWidth: 3 } },
      selectedMode: "single",
      emphasis: { itemStyle: { borderColor: theme.text, borderWidth: 2 }, label: { show: true } },
      ...(rows.length > 1_000 ? { progressive: 2_000, progressiveThreshold: 1_000 } : {}),
    }],
  };
}

function timelineOption(
  spec: Extract<VisualizationSpec, { kind: "timeline" }>,
  rows: DataRow[],
  columns: ColumnSpec[],
  theme: ChartTheme,
): EChartsOption {
  const enhanced = spec as typeof spec & { end_time_field?: string | null; lane_field?: string | null };
  const times = rows.map((row) => text(row[spec.time_field]));
  const isTemporal = times.every((value) => Number.isFinite(Date.parse(value)));
  const lanes = Array.from(new Set(rows.map((row) => enhanced.lane_field ? text(row[enhanced.lane_field]) : "Events")));
  const tooltip = {
    ...tooltipAppearance(theme),
    trigger: "item" as const,
    formatter: (raw: unknown) => {
      const params = (Array.isArray(raw) ? raw[0] : raw) as { dataIndex?: number; value?: unknown } | undefined;
      const supplied = Array.isArray(params?.value) ? params?.value : [];
      const indexFromValue = Number(supplied[3]);
      const index = Number.isInteger(indexFromValue) ? indexFromValue : params?.dataIndex ?? 0;
      const row = rows[index];
      const details = spec.detail_fields
        .filter((field) => row?.[field] != null)
        .map((field) => {
          const column = columns.find((candidate) => candidate.field === field);
          const value = column?.data_type === "boolean"
            ? text(row[field])
            : formattedValue(row[field], column?.unit);
          return `${escapeHtml(column?.label || axisTitle(field))}: ${escapeHtml(value)}`;
        });
      const interval = enhanced.end_time_field && row?.[enhanced.end_time_field] != null
        ? `${escapeHtml(axisText(row?.[spec.time_field]))} – ${escapeHtml(axisText(row?.[enhanced.end_time_field]))}`
        : escapeHtml(axisText(row?.[spec.time_field]));
      return [`<strong>${escapeHtml(row?.[spec.label_field])}</strong>`, interval, ...details].join("<br/>");
    },
  };
  if (isTemporal) {
    const numericTimes = times.map((value) => Date.parse(value));
    const timeMinimum = Math.min(...numericTimes);
    const timeMaximum = Math.max(...numericTimes);
    const timeSpan = Math.max(0, timeMaximum - timeMinimum);
    const timePadding = Math.max(30 * 60 * 1_000, timeSpan * .08);
    const longestLane = lanes.reduce(
      (length, lane) => Math.max(length, lane.length),
      0,
    );
    const laneAxisWidth = Math.min(180, Math.max(86, longestLane * 7));
    const laneGridLeft = enhanced.lane_field
      ? Math.min(220, laneAxisWidth + 34)
      : 54;
    const laneZoom = lanes.length > 10
      ? [
        {
          type: "inside" as const,
          yAxisIndex: 0,
          startValue: 0,
          endValue: 9,
          filterMode: "none" as const,
          zoomOnMouseWheel: false,
          moveOnMouseWheel: true,
          moveOnMouseMove: true,
        },
        {
          type: "slider" as const,
          yAxisIndex: 0,
          startValue: 0,
          endValue: 9,
          filterMode: "none" as const,
          right: 8,
          width: 12,
          showDetail: false,
          brushSelect: false,
        },
      ]
      : chartZoom(rows);
    const eventData = rows.map((row, index) => [
      Date.parse(text(row[spec.time_field])),
      lanes.indexOf(enhanced.lane_field ? text(row[enhanced.lane_field]) : "Events"),
      text(row[spec.label_field]),
      index,
    ]);
    const intervalData = enhanced.end_time_field
      ? rows.map((row, index) => [
        Date.parse(text(row[spec.time_field])),
        Date.parse(text(row[enhanced.end_time_field!])),
        lanes.indexOf(enhanced.lane_field ? text(row[enhanced.lane_field]) : "Events"),
        index,
      ]).filter((entry) => Number.isFinite(Number(entry[0])) && Number.isFinite(Number(entry[1])))
      : [];
    const series = intervalData.length
      ? [{
        id: "timeline-main",
        name: "Intervals",
        type: "custom" as const,
        renderItem: (params: { coordSys?: { x: number; y: number; width: number; height: number } }, api: {
          value: (dimension: number) => number;
          coord: (value: [number, number]) => [number, number];
          size: (value: [number, number]) => [number, number];
          style: () => Record<string, unknown>;
        }) => {
          const start = api.coord([api.value(0), api.value(2)]);
          const end = api.coord([api.value(1), api.value(2)]);
          const height = Math.max(8, api.size([0, 1])[1] * .34);
          const raw = { x: start[0], y: start[1] - height / 2, width: Math.max(2, end[0] - start[0]), height };
          const bounds = params.coordSys;
          if (!bounds) return { type: "rect", shape: raw, style: api.style() };
          const x = Math.max(raw.x, bounds.x);
          const right = Math.min(raw.x + raw.width, bounds.x + bounds.width);
          return {
            type: "rect",
            shape: { x, y: raw.y, width: Math.max(0, right - x), height: raw.height },
            style: { ...api.style(), fill: theme.palette[0], stroke: theme.surface, lineWidth: 1 },
          };
        },
        encode: { x: [0, 1], y: 2 },
        data: intervalData,
        emphasis: { focus: "series" as const },
      }]
      : [{
        id: "timeline-main",
        name: "Events",
        type: "scatter" as const,
        symbol: "diamond",
        symbolSize: 13,
        data: eventData,
        itemStyle: { color: theme.palette[0], borderColor: theme.surface, borderWidth: 2 },
        label: {
          show: lanes.length === 1 && rows.length <= 8,
          formatter: (raw: { value?: unknown }) => Array.isArray(raw.value) ? text(raw.value[2]) : "",
          position: "top" as const,
          color: theme.text,
          fontSize: 11,
        },
      }];
    const timelineAxis = timeAxis(theme);
    return {
      ...baseOption(spec, rows, theme),
      grid: {
        left: laneGridLeft,
        right: lanes.length > 10 ? 62 : 42,
        top: 46,
        bottom: rows.length > 45 ? 92 : 68,
        containLabel: false,
      },
      tooltip,
      dataZoom: laneZoom,
      xAxis: {
        ...timelineAxis,
        min: timeMinimum - timePadding,
        max: timeMaximum + timePadding,
      },
      yAxis: {
        ...categoryAxis(theme, lanes),
        inverse: true,
        axisLabel: {
          color: theme.muted,
          fontSize: 10,
          width: laneAxisWidth,
          overflow: "truncate",
          hideOverlap: false,
          formatter: axisText,
        },
        axisTick: { show: false },
        splitArea: {
          show: lanes.length > 1,
          areaStyle: { color: [theme.surface, `${theme.grid}22`] },
        },
      },
      series: series as EChartsOption["series"],
    };
  }
  return {
    ...baseOption(spec, rows, theme),
    grid: { left: 36, right: 36, top: 68, bottom: 72, containLabel: true },
    tooltip,
    xAxis: categoryAxis(theme, times, undefined, times.length > 8 ? 32 : 0),
    yAxis: { type: "value", show: false, min: 0, max: 1 },
    series: [{
      id: "timeline-main",
      type: "scatter",
      symbol: "diamond",
      symbolSize: 13,
      data: rows.map((row) => ({ value: 0.5, name: text(row[spec.label_field]) })),
      itemStyle: { color: theme.palette[0], borderColor: theme.surface, borderWidth: 2 },
      label: { show: true, formatter: "{b}", position: "top", color: theme.text, fontSize: 11 },
    }],
  };
}

function RenderFailure({ message, dataset }: { message: string; dataset?: Dataset }) {
  return <div className="visual-fallback"><div className="visual-error" role="alert"><strong>Visualization could not be rendered.</strong><span>{message} The data remains available below.</span></div><DataTable dataset={dataset} /></div>;
}

function ChartCommandStrip({
  canZoom,
  expanded,
  showData,
  onZoomIn,
  onZoomOut,
  onReset,
  onExpand,
  onInspect,
  onToggleData,
  onDownload,
}: ChartCommandProps) {
  return (
    <div className="chart-command-strip" role="toolbar" aria-label="Chart commands">
      <button className="chart-command" type="button" onClick={onZoomIn} disabled={!canZoom} aria-label="Zoom in on chart">Zoom in</button>
      <button className="chart-command" type="button" onClick={onZoomOut} disabled={!canZoom} aria-label="Zoom out of chart">Zoom out</button>
      <button className="chart-command" type="button" onClick={onReset} aria-label="Reset chart view">Reset</button>
      <button className="chart-command" type="button" onClick={onInspect} aria-label="Inspect chart points with the keyboard">Inspect points</button>
      <button className="chart-command" type="button" onClick={onExpand} aria-label={expanded ? "Close expanded chart" : "Expand chart"}>
        {expanded ? "Close" : "Expand"}
      </button>
      <button className="chart-command" type="button" onClick={onToggleData} aria-expanded={showData}>
        {showData ? "Hide data" : "View data"}
      </button>
      <button className="chart-command" type="button" onClick={onDownload} aria-label="Download chart as PNG">PNG</button>
    </div>
  );
}

function useDialogFocusTrap(
  expanded: boolean,
  dialogRef: React.RefObject<HTMLDivElement | null>,
  close: () => void,
) {
  useEffect(() => {
    if (!expanded || !dialogRef.current) return;
    const dialog = dialogRef.current;
    const previous = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const focusable = () => Array.from(dialog.querySelectorAll<HTMLElement>(
      'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
    )).filter((element) => !element.hasAttribute("hidden"));
    focusable()[0]?.focus();
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        close();
        return;
      }
      if (event.key !== "Tab") return;
      const elements = focusable();
      if (!elements.length) {
        event.preventDefault();
        dialog.focus();
        return;
      }
      const first = elements[0];
      const last = elements[elements.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    dialog.addEventListener("keydown", handleKeyDown);
    return () => {
      dialog.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = previousOverflow;
      previous?.focus();
    };
  }, [close, dialogRef, expanded]);
}

function inspectableFields(
  spec: Exclude<VisualizationSpec, { kind: "kpi" | "map" | "table" | "omitted" }>,
): string[] {
  if (spec.kind === "cartesian") return [spec.x_field, ...(spec.series_field ? [spec.series_field] : []), ...spec.y_fields];
  if (spec.kind === "forecast") return [spec.date_field, ...(spec.actual_field ? [spec.actual_field] : []), spec.predicted_field, spec.lower_field, spec.upper_field];
  if (spec.kind === "distribution") return [...(spec.category_field ? [spec.category_field] : []), spec.value_field, ...(spec.count_field ? [spec.count_field] : [])];
  if (spec.kind === "heatmap") return [spec.x_field, spec.y_field, spec.value_field];
  return [spec.time_field, spec.label_field, ...spec.detail_fields];
}

function inspectionAnnouncement(
  spec: Exclude<VisualizationSpec, { kind: "kpi" | "map" | "table" | "omitted" }>,
  dataset: Dataset | undefined,
  index: number,
): string {
  const rows = dataset?.rows || [];
  const row = rows[index];
  if (!row) return "No chart point is selected.";
  const fields = Array.from(new Set(inspectableFields(spec)));
  const details = fields
    .filter((field) => row[field] != null)
    .map((field) => {
      const column = dataset?.columns.find((candidate) => candidate.field === field);
      return `${column?.label || axisTitle(field)}: ${formattedValue(row[field], column?.unit)}`;
    });
  if (spec.kind === "heatmap" && asNumber(row[spec.value_field]) === null) {
    return `Point ${index + 1} of ${rows.length}. ${details.join(". ")}. Value unavailable; no heatmap cell is rendered for this source row.`;
  }
  return `Point ${index + 1} of ${rows.length}. ${details.join(". ")}.`;
}

function temporalDomainSignature(
  spec: Exclude<VisualizationSpec, { kind: "kpi" | "map" | "table" | "omitted" }>,
  rows: DataRow[],
): string | null {
  const field = spec.kind === "cartesian"
    ? spec.x_field
    : spec.kind === "forecast"
      ? spec.date_field
      : spec.kind === "timeline"
        ? spec.time_field
        : null;
  if (!field || !rows.length) return null;
  const values = rows.map((row) => text(row[field]));
  if (!values.every((value) => Number.isFinite(Date.parse(value)))) return null;
  let hash = 2166136261;
  for (const character of values.join("|")) {
    hash ^= character.charCodeAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return `eagle-eye-time-${(hash >>> 0).toString(16)}`;
}

function semanticProfile(spec: VisualizationSpec, rows: DataRow[] = []): PresentationProfile {
  if (spec.kind === "kpi") return "metric-bullet";
  if (spec.kind === "cartesian") {
    if (spec.chart_type === "scatter") return "correlation-fit";
    if (spec.interval_bands?.length) return "uncertainty-range";
    if (spec.stacked || spec.chart_type === "stacked_bar") return "composition-ribbon";
    const temporal = (spec.chart_type === "line" || spec.chart_type === "area")
      && rows.length > 1
      && rows.every((row) => Number.isFinite(Date.parse(text(row[spec.x_field]))));
    if (temporal) return "temporal-focus-context";
    if (
      spec.orientation === "horizontal"
      && spec.y_fields.length === 1
      && !spec.series_field
      && spec.chart_type === "bar"
    ) return "ranking-lollipop";
    return "category-comparison";
  }
  if (spec.kind === "forecast") return "forecast-fan-interval";
  if (spec.kind === "distribution") {
    if (spec.count_field && spec.five_number_summary) return "distribution-histogram-summary";
    if (spec.chart_type === "boxplot") return "distribution-box";
    return "distribution-percentile";
  }
  if (spec.kind === "heatmap") return "calendar-pattern-heatmap";
  if (spec.kind === "map") return "geospatial-investigation";
  if (spec.kind === "timeline") return "event-rail";
  if (spec.kind === "table") return "verified-table";
  return "explicit-omission";
}

function visualizationProfile(spec: VisualizationSpec): string {
  if (spec.kind === "cartesian" || spec.kind === "distribution") {
    return `${spec.kind}:${spec.chart_type}`;
  }
  return spec.kind;
}

function diagnosticFields(
  spec: VisualizationSpec,
  rows: DataRow[],
  series: Array<{ name?: unknown; type?: unknown; id?: unknown }> = [],
): VisualizationDiagnosticFields {
  let fields: string[] = [];
  let renderedPointCount = rows.length;
  if (spec.kind === "kpi") {
    fields = [spec.value_field, ...(spec.comparison_field ? [spec.comparison_field] : [])];
    renderedPointCount = rows.some((row) => asNumber(row[spec.value_field]) !== null) ? 1 : 0;
  } else if (spec.kind === "cartesian") {
    fields = [
      spec.x_field,
      ...(spec.series_field ? [spec.series_field] : []),
      ...spec.y_fields,
      ...(spec.interval_bands || []).flatMap((band) => [band.lower_field, band.upper_field]),
      ...(spec.fitted_series || []).flatMap((binding) => [binding.x_field, binding.y_field]),
    ];
    renderedPointCount = spec.y_fields.reduce(
      (count, field) => count + rows.filter((row) => asNumber(row[field]) !== null).length,
      0,
    );
  } else if (spec.kind === "forecast") {
    fields = [spec.date_field, ...(spec.actual_field ? [spec.actual_field] : []), spec.predicted_field, spec.lower_field, spec.upper_field];
    renderedPointCount = rows.filter((row) => asNumber(row[spec.predicted_field]) !== null || (spec.actual_field && asNumber(row[spec.actual_field]) !== null)).length;
  } else if (spec.kind === "distribution") {
    fields = [spec.value_field, ...(spec.count_field ? [spec.count_field] : []), ...(spec.category_field ? [spec.category_field] : [])];
    renderedPointCount = rows.filter((row) => asNumber(row[spec.value_field]) !== null).length;
  } else if (spec.kind === "heatmap") {
    fields = [spec.x_field, spec.y_field, spec.value_field];
    renderedPointCount = rows.filter((row) => asNumber(row[spec.value_field]) !== null).length;
  } else if (spec.kind === "map") {
    fields = [spec.latitude_field, spec.longitude_field, ...(spec.value_field ? [spec.value_field] : [])];
    renderedPointCount = rows.filter((row) => asNumber(row[spec.latitude_field]) !== null && asNumber(row[spec.longitude_field]) !== null).length;
  } else if (spec.kind === "timeline") {
    fields = [spec.time_field, spec.label_field, ...(spec.end_time_field ? [spec.end_time_field] : []), ...(spec.lane_field ? [spec.lane_field] : [])];
  }
  const preservedNullCount = rows.reduce(
    (count, row) => count + Array.from(new Set(fields)).filter((field) => row[field] == null || (typeof row[field] === "number" && !Number.isFinite(row[field]))).length,
    0,
  );
  const visibleSeries = series.filter((item) => {
    const id = text(item.id);
    const name = text(item.name);
    return !id.includes("helper") && !id.includes("interval-base") && !name.startsWith("__") && name !== "Interval base";
  });
  return {
    semantic_profile: semanticProfile(spec, rows),
    analytical_series_names: visibleSeries.map((item) => text(item.name)).filter((name) => name !== "—"),
    analytical_series_types: visibleSeries.map((item) => {
      const id = text(item.id);
      if (id === "distribution-histogram") return "bar";
      if (id === "distribution-summary") return "boxplot";
      return text(item.type);
    }).filter((type) => type !== "—"),
    rendered_point_count: renderedPointCount,
    reference_line_count: spec.kind === "cartesian" ? (spec.reference_lines || []).length : 0,
    interval_band_count: spec.kind === "cartesian" ? (spec.interval_bands || []).length : spec.kind === "forecast" ? 1 : 0,
    annotation_count: spec.kind === "cartesian"
      ? (spec.annotations || []).length + (spec.highlight ? 1 : 0)
      : 0,
    preserved_null_count: preservedNullCount,
    fitted_series_count: spec.kind === "cartesian" ? (spec.fitted_series || []).length : 0,
    summary_marker_count: spec.kind === "distribution" && spec.five_number_summary
      ? 5 + (spec.five_number_summary.p90 == null ? 0 : 1)
      : 0,
    quality_annotation_count: spec.kind === "forecast" && spec.quality_metrics ? 3 : 0,
    null_value_count: preservedNullCount,
    null_to_zero_count: 0,
    inspected_series_name: null,
  };
}

function seriesDiagnostics(option: EChartsOption, spec: VisualizationSpec, rows: DataRow[]): VisualizationDiagnosticFields {
  const rawSeries = Array.isArray(option.series) ? option.series : option.series ? [option.series] : [];
  const normalizedSeries = rawSeries.map((item) => item as unknown as { name?: unknown; type?: unknown; id?: unknown });
  const diagnostics = diagnosticFields(
    spec,
    rows,
    normalizedSeries,
  );
  if (!["kpi", "map", "table", "omitted"].includes(spec.kind)) {
    const seriesId = inspectableSeriesId(spec as Exclude<VisualizationSpec, { kind: "kpi" | "map" | "table" | "omitted" }>);
    const inspected = normalizedSeries.find((item) => text(item.id) === seriesId);
    diagnostics.inspected_series_name = inspected?.name == null ? null : text(inspected.name);
  }
  return diagnostics;
}

function inspectableSeriesId(
  spec: Exclude<VisualizationSpec, { kind: "kpi" | "map" | "table" | "omitted" }>,
): string {
  if (spec.kind === "forecast") return spec.actual_field ? "forecast-actual" : "forecast-predicted";
  if (spec.kind === "distribution") return spec.count_field ? "distribution-histogram" : "distribution-main";
  if (spec.kind === "heatmap") return "heatmap-main";
  if (spec.kind === "timeline") return "timeline-main";
  if (spec.chart_type === "scatter") return "scatter-main";
  if (semanticProfile(spec) === "ranking-lollipop") return "ranking-points";
  return "cartesian-main";
}

function EChart({
  spec,
  dataset,
  fallbackDataset,
  datasets,
  resolvedTheme,
  showData,
  onToggleData,
  contractVersion,
}: {
  spec: Exclude<VisualizationSpec, { kind: "kpi" | "map" | "table" | "omitted" }>;
  dataset?: Dataset;
  fallbackDataset?: Dataset;
  datasets: Dataset[];
  resolvedTheme: ResolvedTheme;
  showData: boolean;
  onToggleData: () => void;
  contractVersion: "2.0" | "2.1";
}) {
  const chartRef = useRef<HTMLDivElement>(null);
  const inspectorRef = useRef<HTMLDivElement>(null);
  const chartInstanceRef = useRef<ECharts | null>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const zoomRef = useRef({ start: 0, end: 100 });
  const failureObservationRef = useRef<string | null>(null);
  const diagnosticsRef = useRef<VisualizationDiagnosticFields | null>(null);
  const [renderError, setRenderError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [inspectionIndex, setInspectionIndex] = useState(0);
  const rows = dataset?.rows || [];
  const columns = dataset?.columns || [];
  const theme = chartThemes[resolvedTheme];
  const outlierDatasetId = spec.kind === "distribution"
    ? (spec as typeof spec & DistributionEnhancements).outlier_dataset_id
    : null;
  const summaryDatasetId = spec.kind === "distribution"
    ? (spec as typeof spec & DistributionEnhancements).summary_dataset_id
    : null;
  const outlierRows = useMemo(
    () => datasets.find((candidate) => candidate.id === outlierDatasetId)?.rows || [],
    [datasets, outlierDatasetId],
  );
  const scatterAxes = spec.kind === "cartesian" && spec.chart_type === "scatter"
    ? [axisTitle(spec.x_field, spec.x_unit || unitFor(columns, spec.x_field)), axisTitle(spec.y_fields[0], spec.y_unit || unitFor(columns, spec.y_fields[0]))]
    : null;
  const observe = useCallback(
    (
      interactionMode: string,
      renderLatencyMs: number | null = null,
      fallbackReason: string | null = null,
    ) => {
      recordVisualizationObservation({
        visualization_id: spec.id,
        chart_profile: visualizationProfile(spec),
        ...(diagnosticsRef.current || diagnosticFields(spec, rows)),
        visualization_contract_version: contractVersion,
        render_latency_ms: renderLatencyMs,
        fallback_reason: fallbackReason,
        interaction_mode: interactionMode,
        source_dataset_ids: Array.from(
          new Set(
            [dataset?.id, fallbackDataset?.id, summaryDatasetId, outlierDatasetId]
              .filter((value): value is string => Boolean(value)),
          ),
        ),
      });
    },
    [contractVersion, dataset?.id, fallbackDataset?.id, outlierDatasetId, rows, spec, summaryDatasetId],
  );
  const closeExpanded = useCallback(() => {
    observe("collapse");
    setExpanded(false);
  }, [observe]);
  useDialogFocusTrap(expanded, dialogRef, closeExpanded);
  useEffect(() => {
    if (!chartRef.current || !rows.length) return;
    const renderStarted = window.performance.now();
    const node = chartRef.current; let chart: ECharts | undefined; let cancelled = false;
    let renderRecorded = false;
    let firstRenderFrame: number | null = null;
    let fallbackRenderFrame: number | null = null;
    const recordRender = () => {
      if (renderRecorded || cancelled) return;
      renderRecorded = true;
      failureObservationRef.current = null;
      observe(
        "render",
        Math.max(0, window.performance.now() - renderStarted),
      );
    };
    setRenderError(null);
    const fail = (message: string) => {
      if (!cancelled) {
        if (failureObservationRef.current !== message) {
          failureObservationRef.current = message;
          observe(
            "render_failure",
            Math.max(0, window.performance.now() - renderStarted),
            message,
          );
        }
        chart?.dispose();
        chart = undefined;
        chartInstanceRef.current = null;
        setExpanded(false);
        setRenderError(message);
      }
    };
    void loadEChartsModule().then((echarts) => {
      if (cancelled) return;
      try {
        chart = echarts.init(node, undefined, { renderer: "canvas" });
        chartInstanceRef.current = chart;
        const option = spec.kind === "cartesian"
          ? cartesianOption(spec, rows, columns, theme)
          : spec.kind === "forecast"
            ? forecastOption(spec, rows, theme)
            : spec.kind === "distribution"
              ? distributionOption(spec, rows, theme, outlierRows)
              : spec.kind === "heatmap"
                ? heatmapOption(spec, rows, theme, resolvedTheme)
                : timelineOption(spec, rows, columns, theme);
        diagnosticsRef.current = seriesDiagnostics(option, spec, rows);
        chart.on("rendered", recordRender);
        chart.on("finished", recordRender);
        chart.setOption(option, true);
        firstRenderFrame = window.requestAnimationFrame(() => {
          fallbackRenderFrame = window.requestAnimationFrame(recordRender);
        });
        const group = temporalDomainSignature(spec, rows);
        if (group) {
          chart.group = group;
          echarts.connect(group);
        }
        chart.on("datazoom", (event: unknown) => {
          const payload = event as { start?: number; end?: number; batch?: Array<{ start?: number; end?: number }> };
          const zoom = payload.batch?.[0] || payload;
          if (typeof zoom.start === "number" && typeof zoom.end === "number") zoomRef.current = { start: zoom.start, end: zoom.end };
        });
      } catch {
        chart?.dispose(); chart = undefined; fail("The analytical chart engine reported an initialization error.");
      }
    }).catch(() => fail("The analytical chart engine could not be loaded."));
    const observer = new ResizeObserver(() => { try { chart?.resize(); } catch { fail("The analytical chart engine stopped while resizing the view."); } }); observer.observe(node);
    return () => {
      cancelled = true;
      observer.disconnect();
      if (firstRenderFrame !== null) window.cancelAnimationFrame(firstRenderFrame);
      if (fallbackRenderFrame !== null) window.cancelAnimationFrame(fallbackRenderFrame);
      chart?.dispose();
      if (chartInstanceRef.current === chart) chartInstanceRef.current = null;
    };
  }, [columns, expanded, observe, outlierRows, resolvedTheme, rows, spec, theme]);
  const changeZoom = (direction: "in" | "out") => {
    const current = zoomRef.current;
    const span = current.end - current.start;
    const adjustment = Math.max(2, span * .16);
    const start = direction === "in" ? current.start + adjustment : Math.max(0, current.start - adjustment);
    const end = direction === "in" ? current.end - adjustment : Math.min(100, current.end + adjustment);
    if (end - start < 4) return;
    zoomRef.current = { start, end };
    chartInstanceRef.current?.dispatchAction({ type: "dataZoom", start, end });
    observe(direction === "in" ? "zoom_in" : "zoom_out");
  };
  const reset = () => {
    zoomRef.current = { start: 0, end: 100 };
    chartInstanceRef.current?.dispatchAction({ type: "dataZoom", start: 0, end: 100 });
    chartInstanceRef.current?.dispatchAction({ type: "downplay" });
    setInspectionIndex(0);
    observe("reset");
  };
  const download = () => {
    const chart = chartInstanceRef.current;
    if (!chart) return;
    try {
      const anchor = document.createElement("a");
      anchor.download = `${spec.title.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "eagle-eye-chart"}.png`;
      anchor.href = chart.getDataURL({ type: "png", pixelRatio: 2, backgroundColor: theme.surface });
      anchor.click();
      observe("png_download");
    } catch {
      const message = "The PNG export could not be created by the chart engine.";
      observe("png_download_failure", null, message);
      setRenderError(message);
    }
  };
  const inspect = (next: number) => {
    if (!rows.length) return;
    const bounded = Math.max(0, Math.min(rows.length - 1, next));
    setInspectionIndex(bounded);
    const chart = chartInstanceRef.current;
    const seriesId = inspectableSeriesId(spec);
    const renderedDataIndex = spec.kind === "heatmap"
      ? asNumber(rows[bounded]?.[spec.value_field]) === null
        ? null
        : rows.slice(0, bounded).filter((row) => asNumber(row[spec.value_field]) !== null).length
      : bounded;
    chart?.dispatchAction({ type: "downplay" });
    if (renderedDataIndex !== null) {
      chart?.dispatchAction({ type: "highlight", seriesId, dataIndex: renderedDataIndex });
      chart?.dispatchAction({ type: "showTip", seriesId, dataIndex: renderedDataIndex });
    }
    observe("keyboard_inspection");
  };
  const handleInspectionKey = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "ArrowRight" || event.key === "ArrowDown") {
      event.preventDefault();
      inspect(inspectionIndex + 1);
    } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
      event.preventDefault();
      inspect(inspectionIndex - 1);
    } else if (event.key === "Home") {
      event.preventDefault();
      inspect(0);
    } else if (event.key === "End") {
      event.preventDefault();
      inspect(rows.length - 1);
    }
  };
  if (renderError) return <RenderFailure message={renderError} dataset={fallbackDataset || dataset} />;
  const commandProps: ChartCommandProps = {
    canZoom: rows.length > 45,
    expanded,
    showData,
    onZoomIn: () => changeZoom("in"),
    onZoomOut: () => changeZoom("out"),
    onReset: reset,
    onExpand: () => {
      if (expanded) closeExpanded();
      else {
        observe("expand");
        setExpanded(true);
      }
    },
    onInspect: () => {
      observe("inspect_points");
      inspectorRef.current?.focus();
      inspect(inspectionIndex);
    },
    onToggleData: () => {
      observe(showData ? "hide_data" : "view_data");
      onToggleData();
    },
    onDownload: download,
  };
  const associationFit = spec.kind === "cartesian" && spec.chart_type === "scatter"
    ? (spec.fitted_series || []).find((binding) => binding.association_only)
    : null;
  const quality = spec.kind === "forecast" ? spec.quality_metrics : null;
  const focusedScale = spec.kind === "cartesian"
    ? focusedPositiveCarbonAxis(spec, rows)
    : null;
  const focusedScaleUnit = spec.kind === "cartesian"
    ? spec.y_unit || unitFor(columns, spec.y_fields[0])
    : undefined;
  const focusedScaleNoticeId = focusedScale
    ? `focused-positive-axis-${spec.id}`
    : undefined;
  const chartStage = (
    <div className="chart-stage">
      <ChartCommandStrip {...commandProps} />
      {associationFit && (
        <div className="chart-analysis-strip chart-association-strip" role="note">
          <strong>Association view</strong>
          <span>Observed relationship; this visualization does not establish causation.</span>
          {associationFit.r_squared != null && <span>R² {numberFormatter.format(associationFit.r_squared)}</span>}
        </div>
      )}
      {quality && (
        <div className="chart-analysis-strip chart-quality-strip" role="note" aria-label="Forecast validation quality">
          <strong>Validation passed</strong>
          <span>MASE {numberFormatter.format(quality.mase)}</span>
          <span>Interval coverage {numberFormatter.format(quality.interval_coverage * 100)}%</span>
          <span>Target {numberFormatter.format(quality.interval_level * 100)}%</span>
        </div>
      )}
      {focusedScale && (
        <div
          className="chart-analysis-strip chart-axis-scale-notice"
          id={focusedScaleNoticeId}
          role="note"
          aria-label="Truncated Carbon y-axis"
          data-axis-policy="positive-carbon-focus"
          data-axis-min={focusedScale.minimum}
          data-axis-max={focusedScale.maximum}
        >
          <strong>Truncated y-axis</strong>
          <span>
            Positive-only Carbon scale starts at {formattedValue(focusedScale.minimum, focusedScaleUnit)} instead of zero
            and ends at {formattedValue(focusedScale.maximum, focusedScaleUnit)}. Exact values and uncertainty bounds are unchanged.
          </span>
        </div>
      )}
      <div
        className="chart-inspector-focus"
        ref={inspectorRef}
        tabIndex={0}
        onKeyDown={handleInspectionKey}
        aria-label={`${visualSummary(spec, rows)} Use Left and Right Arrow keys to inspect individual points.`}
        aria-describedby={focusedScaleNoticeId}
      >
        <div ref={chartRef} className="chart-canvas" role="img" aria-label={visualSummary(spec, rows)} />
      </div>
      <p className="chart-live-region sr-only" aria-live="polite" aria-atomic="true">
        {inspectionAnnouncement(spec, dataset, inspectionIndex)}
      </p>
      {scatterAxes && <p className="sr-only">Horizontal axis: {scatterAxes[0]}. Vertical axis: {scatterAxes[1]}.</p>}
    </div>
  );
  if (expanded) {
    return createPortal(
      <div className="expanded-chart-backdrop">
        <div
          className="expanded-chart-dialog"
          role="dialog"
          aria-modal="true"
          aria-labelledby={`expanded-chart-title-${spec.id}`}
          ref={dialogRef}
          tabIndex={-1}
        >
          <header className="expanded-chart-header">
            <div>
              <p className="eyebrow">Expanded analysis</p>
              <h3 id={`expanded-chart-title-${spec.id}`}>{spec.title}</h3>
            </div>
            <div className="expanded-chart-actions">
              <button className="chart-command" type="button" onClick={closeExpanded}>Close</button>
            </div>
          </header>
          <div className="expanded-chart-body">{chartStage}</div>
        </div>
      </div>,
      document.body,
    );
  }
  return chartStage;
}

function MapVisualization({
  spec,
  dataset,
  fallbackDataset,
  resolvedTheme,
  showData,
  onToggleData,
  contractVersion,
}: {
  spec: Extract<VisualizationSpec, { kind: "map" }>;
  dataset?: Dataset;
  fallbackDataset?: Dataset;
  resolvedTheme: ResolvedTheme;
  showData: boolean;
  onToggleData: () => void;
  contractVersion: "2.0" | "2.1";
}) {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<import("maplibre-gl").Map | null>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const extentRef = useRef<[[number, number], [number, number]] | null>(null);
  const failureObservationRef = useRef<string | null>(null);
  const diagnosticsRef = useRef<VisualizationDiagnosticFields | null>(null);
  const [renderError, setRenderError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [mapInspection, setMapInspection] = useState("No mapped observation is selected.");
  const rows = dataset?.rows || [];
  const theme = chartThemes[resolvedTheme];
  const geometry = spec as typeof spec & MapGeometryFields;
  const observe = useCallback(
    (
      interactionMode: string,
      renderLatencyMs: number | null = null,
      fallbackReason: string | null = null,
    ) => {
      recordVisualizationObservation({
        visualization_id: spec.id,
        chart_profile: visualizationProfile(spec),
        ...(diagnosticsRef.current || diagnosticFields(spec, rows)),
        visualization_contract_version: contractVersion,
        render_latency_ms: renderLatencyMs,
        fallback_reason: fallbackReason,
        interaction_mode: interactionMode,
        source_dataset_ids: Array.from(
          new Set(
            [dataset?.id, fallbackDataset?.id]
              .filter((value): value is string => Boolean(value)),
          ),
        ),
      });
    },
    [contractVersion, dataset?.id, fallbackDataset?.id, spec],
  );
  const closeExpanded = useCallback(() => {
    observe("collapse");
    setExpanded(false);
  }, [observe]);
  useDialogFocusTrap(expanded, dialogRef, closeExpanded);
  useEffect(() => {
    if (!mapRef.current || !rows.length) return;
    const renderStarted = window.performance.now();
    const features: GeoJSON.Feature<GeoJSON.Point>[] = [];
    rows.forEach((row, index) => {
      const longitude = asNumber(row[spec.longitude_field]);
      const latitude = asNumber(row[spec.latitude_field]);
      if (longitude === null || latitude === null || longitude < -180 || longitude > 180 || latitude < -90 || latitude > 90) return;
      features.push({
        type: "Feature",
        properties: {
          rowIndex: index,
          label: spec.label_field ? text(row[spec.label_field]) : "Mapped observation",
          value: spec.value_field ? row[spec.value_field] : null,
          numericValue: spec.value_field ? asNumber(row[spec.value_field]) : null,
          valueField: spec.value_field || null,
        },
        geometry: { type: "Point", coordinates: [longitude, latitude] },
      });
    });
    const pathFeatures: GeoJSON.Feature<GeoJSON.LineString>[] = [];
    if (
      geometry.geometry_mode === "segments"
      && geometry.start_latitude_field
      && geometry.start_longitude_field
      && geometry.end_latitude_field
      && geometry.end_longitude_field
    ) {
      rows.forEach((row, index) => {
        const startLatitude = asNumber(row[geometry.start_latitude_field!]);
        const startLongitude = asNumber(row[geometry.start_longitude_field!]);
        const endLatitude = asNumber(row[geometry.end_latitude_field!]);
        const endLongitude = asNumber(row[geometry.end_longitude_field!]);
        if (
          startLatitude === null || startLongitude === null || endLatitude === null || endLongitude === null
          || startLongitude < -180 || startLongitude > 180 || endLongitude < -180 || endLongitude > 180
          || startLatitude < -90 || startLatitude > 90 || endLatitude < -90 || endLatitude > 90
        ) return;
        pathFeatures.push({
          type: "Feature",
          properties: {
            rowIndex: index,
            label: spec.label_field ? text(row[spec.label_field]) : "Validated segment",
          },
          geometry: { type: "LineString", coordinates: [[startLongitude, startLatitude], [endLongitude, endLatitude]] },
        });
      });
    } else if (
      geometry.geometry_mode === "ordered_path"
      && geometry.path_field
      && geometry.sequence_field
      && geometry.timestamp_field
    ) {
      const paths = new Map<string, Array<{ row: DataRow; index: number }>>();
      rows.forEach((row, index) => {
        const key = text(row[geometry.path_field!]);
        paths.set(key, [...(paths.get(key) || []), { row, index }]);
      });
      paths.forEach((pathRows, path) => {
        const ordered = [...pathRows].sort((a, b) => {
          const sequenceA = asNumber(a.row[geometry.sequence_field!]);
          const sequenceB = asNumber(b.row[geometry.sequence_field!]);
          if (sequenceA !== null && sequenceB !== null && sequenceA !== sequenceB) return sequenceA - sequenceB;
          return Date.parse(text(a.row[geometry.timestamp_field!])) - Date.parse(text(b.row[geometry.timestamp_field!]));
        });
        const coordinates = ordered.map(({ row }) => [
          asNumber(row[spec.longitude_field]),
          asNumber(row[spec.latitude_field]),
        ]).filter((coordinate): coordinate is [number, number] =>
          coordinate[0] !== null && coordinate[1] !== null
          && coordinate[0] >= -180 && coordinate[0] <= 180
          && coordinate[1] >= -90 && coordinate[1] <= 90);
        if (coordinates.length < 2) return;
        pathFeatures.push({
          type: "Feature",
          properties: { rowIndex: ordered[0].index, label: path },
          geometry: { type: "LineString", coordinates },
        });
      });
    }
    const pathCoordinates = pathFeatures.flatMap((feature) => feature.geometry.coordinates);
    const allCoordinates = [...features.map((feature) => feature.geometry.coordinates), ...pathCoordinates];
    diagnosticsRef.current = {
      ...diagnosticFields(spec, rows),
      analytical_series_names: [
        ...(features.length ? ["Mapped observations"] : []),
        ...(pathFeatures.length ? ["Validated paths"] : []),
      ],
      analytical_series_types: [
        ...(features.length ? ["map-point"] : []),
        ...(pathFeatures.length ? ["map-line"] : []),
      ],
      rendered_point_count: features.length + pathFeatures.length,
    };
    if (!allCoordinates.length) {
      const message = "No valid coordinates were available for the map.";
      if (failureObservationRef.current !== message) {
        failureObservationRef.current = message;
        observe(
          "render_failure",
          Math.max(0, window.performance.now() - renderStarted),
          message,
        );
      }
      setRenderError(message);
      return;
    }
    const longitudes = allCoordinates.map((coordinate) => coordinate[0]);
    const latitudes = allCoordinates.map((coordinate) => coordinate[1]);
    const west = Math.min(...longitudes), east = Math.max(...longitudes);
    const south = Math.min(...latitudes), north = Math.max(...latitudes);
    const hasExtent = west !== east || south !== north;
    extentRef.current = [[west, south], [east, north]];
    const shouldCluster = features.length > 100;
    const numericValues = features.map((feature) => Number(feature.properties?.numericValue)).filter(Number.isFinite);
    const minimumValue = numericValues.length ? Math.min(...numericValues) : null;
    const maximumValue = numericValues.length ? Math.max(...numericValues) : null;
    const pointRadius = minimumValue !== null && maximumValue !== null && minimumValue !== maximumValue
      ? (["interpolate", ["linear"], ["get", "numericValue"], minimumValue, 4, maximumValue, 11] as const)
      : 6;
    let cancelled = false;
    let instance: import("maplibre-gl").Map | undefined;
    setRenderError(null);
    const fail = (message: string) => {
      if (!cancelled) {
        if (failureObservationRef.current !== message) {
          failureObservationRef.current = message;
          observe(
            "render_failure",
            Math.max(0, window.performance.now() - renderStarted),
            message,
          );
        }
        try { instance?.remove(); } catch { /* The verified table fallback remains usable. */ }
        instance = undefined;
        mapInstanceRef.current = null;
        setExpanded(false);
        setRenderError(message);
      }
    };
    void Promise.all([import("maplibre-gl"), import("maplibre-gl/dist/maplibre-gl.css")]).then(([module]) => {
      if (cancelled || !mapRef.current) return;
      try {
        const maplibregl = module.default;
        const offlineStyle: StyleSpecification = {
          version: 8,
          sources: {
            "natural-earth-land": {
              type: "geojson",
              data: naturalEarthLand,
              attribution: "Natural Earth · Public domain",
            },
            "analysis-points": {
              type: "geojson",
              data: { type: "FeatureCollection", features },
              ...(shouldCluster ? { cluster: true, clusterMaxZoom: 13, clusterRadius: 48 } : {}),
            },
            ...(pathFeatures.length ? {
              "analysis-paths": {
                type: "geojson" as const,
                data: { type: "FeatureCollection" as const, features: pathFeatures },
              },
            } : {}),
          },
          layers: [
            { id: "ocean-background", type: "background", paint: { "background-color": theme.ocean } },
            { id: "natural-earth-land", type: "fill", source: "natural-earth-land", paint: { "fill-color": theme.land, "fill-opacity": 1 } },
            { id: "natural-earth-coast", type: "line", source: "natural-earth-land", paint: { "line-color": theme.coast, "line-width": .7, "line-opacity": .9 } },
            ...(pathFeatures.length ? [{
              id: "analysis-paths",
              type: "line" as const,
              source: "analysis-paths",
              paint: {
                "line-color": theme.palette[1],
                "line-width": 2,
                "line-opacity": .9,
                "line-dasharray": [3, 2],
              },
            }] : []),
            ...(shouldCluster ? [
              {
                id: "analysis-clusters",
                type: "circle" as const,
                source: "analysis-points",
                filter: ["has", "point_count"] as ["has", string],
                paint: {
                  "circle-radius": ["step", ["get", "point_count"], 16, 250, 22, 750, 28] as never,
                  "circle-color": theme.palette[0],
                  "circle-stroke-color": theme.surface,
                  "circle-stroke-width": 2,
                },
              },
              {
                id: "analysis-cluster-count",
                type: "symbol" as const,
                source: "analysis-points",
                filter: ["has", "point_count"] as ["has", string],
                layout: {
                  "text-field": "{point_count_abbreviated}",
                  "text-size": 11,
                },
                paint: { "text-color": theme.surface },
              },
              {
                id: "analysis-points",
                type: "circle" as const,
                source: "analysis-points",
                filter: ["!", ["has", "point_count"]] as ["!", ["has", string]],
                paint: {
                  "circle-radius": pointRadius as never,
                  "circle-color": theme.palette[0],
                  "circle-stroke-color": theme.surface,
                  "circle-stroke-width": 2,
                },
              },
            ] : [{
              id: "analysis-points",
              type: "circle" as const,
              source: "analysis-points",
              paint: {
                "circle-radius": pointRadius as never,
                "circle-color": theme.palette[0],
                "circle-stroke-color": theme.surface,
                "circle-stroke-width": 2,
              },
            }]),
          ],
        };
        const created = new maplibregl.Map({
          container: mapRef.current,
          style: offlineStyle,
          ...(hasExtent
            ? { bounds: [[west, south], [east, north]] as [[number, number], [number, number]], fitBoundsOptions: { padding: 48, maxZoom: 9 } }
            : { center: allCoordinates[0] as [number, number], zoom: 5 }),
          attributionControl: { compact: true },
          maplibreLogo: false,
          cooperativeGestures: true,
          preserveDrawingBuffer: true,
        });
        instance = created;
        mapInstanceRef.current = created;
        created.once("load", () => {
          if (cancelled) return;
          failureObservationRef.current = null;
          observe(
            "render",
            Math.max(0, window.performance.now() - renderStarted),
          );
        });
        created.getCanvas().setAttribute("aria-label", visualSummary(spec, rows));
        created.addControl(new maplibregl.NavigationControl({ showCompass: false, visualizePitch: false }), "top-right");
        created.addControl(new maplibregl.ScaleControl({ maxWidth: 120, unit: "nautical" }), "bottom-left");
        created.on("error", () => fail("The map engine reported a runtime or map-style error."));
        created.on("mouseenter", "analysis-points", () => { created.getCanvas().style.cursor = "pointer"; });
        created.on("mouseleave", "analysis-points", () => { created.getCanvas().style.cursor = ""; });
        created.on("click", "analysis-points", (event) => {
          const feature = event.features?.[0];
          if (!feature || feature.geometry.type !== "Point") return;
          const properties = feature.properties || {};
          const details = properties.valueField && properties.value != null
            ? `<br/>${escapeHtml(axisTitle(String(properties.valueField)))}: ${escapeHtml(formattedValue(properties.value, spec.value_field ? unitFor(dataset?.columns || [], spec.value_field) : undefined))}`
            : "";
          const announcement = `${properties.label || "Mapped observation"}${properties.valueField && properties.value != null ? `. ${axisTitle(String(properties.valueField))}: ${formattedValue(properties.value, spec.value_field ? unitFor(dataset?.columns || [], spec.value_field) : undefined)}` : ""}. Longitude ${numberFormatter.format(feature.geometry.coordinates[0])}. Latitude ${numberFormatter.format(feature.geometry.coordinates[1])}.`;
          observe("point_inspection");
          setMapInspection(announcement);
          new maplibregl.Popup({ closeButton: true, closeOnClick: true, offset: 10 })
            .setLngLat(feature.geometry.coordinates.slice() as [number, number])
            .setHTML(`<strong>${escapeHtml(properties.label)}</strong>${details}`)
            .addTo(created);
        });
        created.on("mousemove", "analysis-points", (event) => {
          const feature = event.features?.[0];
          if (!feature || feature.geometry.type !== "Point") return;
          const properties = feature.properties || {};
          created.getCanvas().setAttribute(
            "aria-label",
            `${visualSummary(spec, rows)} Hovered: ${properties.label || "mapped observation"}.`,
          );
        });
        if (shouldCluster) {
          created.on("mouseenter", "analysis-clusters", () => { created.getCanvas().style.cursor = "pointer"; });
          created.on("mouseleave", "analysis-clusters", () => { created.getCanvas().style.cursor = ""; });
          created.on("click", "analysis-clusters", (event) => {
            const feature = event.features?.[0];
            if (!feature || feature.geometry.type !== "Point") return;
            const clusterId = Number(feature.properties?.cluster_id);
            const source = created.getSource("analysis-points") as import("maplibre-gl").GeoJSONSource;
            if (!Number.isFinite(clusterId)) return;
            const clusterCoordinates = feature.geometry.coordinates.slice() as [number, number];
            void source.getClusterExpansionZoom(clusterId).then((zoom) => {
              observe("cluster_expand");
              created.easeTo({ center: clusterCoordinates, zoom });
            });
          });
        }
      } catch { fail("The map engine reported an initialization error."); }
    }).catch(() => fail("The map engine could not be loaded."));
    return () => {
      cancelled = true;
      instance?.remove();
      if (mapInstanceRef.current === instance) mapInstanceRef.current = null;
    };
  }, [dataset?.columns, expanded, geometry, observe, resolvedTheme, rows, spec, theme]);
  const fitToData = (interactionMode = "fit_to_data") => {
    const extent = extentRef.current;
    if (!extent || !mapInstanceRef.current) return;
    const [[west, south], [east, north]] = extent;
    if (west === east && south === north) {
      mapInstanceRef.current.easeTo({ center: [west, south], zoom: 6 });
    } else {
      mapInstanceRef.current.fitBounds(extent, { padding: 48, maxZoom: 9 });
    }
    observe(interactionMode);
  };
  const download = () => {
    const instance = mapInstanceRef.current;
    if (!instance) return;
    try {
      const anchor = document.createElement("a");
      anchor.download = `${spec.title.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "eagle-eye-map"}.png`;
      anchor.href = instance.getCanvas().toDataURL("image/png");
      anchor.click();
      observe("png_download");
    } catch {
      const message = "The PNG export could not be created by the map engine.";
      observe("png_download_failure", null, message);
      setRenderError(message);
    }
  };
  if (renderError) return <RenderFailure message={renderError} dataset={fallbackDataset || dataset} />;
  const mapStage = (
    <div className="map-stage">
      <div className="chart-command-strip map-command-strip" role="toolbar" aria-label="Map commands">
        <button className="chart-command" type="button" onClick={() => { observe("zoom_in"); mapInstanceRef.current?.zoomIn(); }}>Zoom in</button>
        <button className="chart-command" type="button" onClick={() => { observe("zoom_out"); mapInstanceRef.current?.zoomOut(); }}>Zoom out</button>
        <button className="chart-command" type="button" onClick={() => fitToData("fit_to_data")}>Fit to data</button>
        <button className="chart-command" type="button" onClick={() => fitToData("reset")}>Reset</button>
        <button className="chart-command" type="button" onClick={() => {
          if (expanded) closeExpanded();
          else {
            observe("expand");
            setExpanded(true);
          }
        }}>{expanded ? "Close" : "Expand"}</button>
        <button className="chart-command" type="button" onClick={() => { observe(showData ? "hide_data" : "view_data"); onToggleData(); }} aria-expanded={showData}>{showData ? "Hide data" : "View data"}</button>
        <button className="chart-command" type="button" onClick={download}>PNG</button>
      </div>
      <div ref={mapRef} className="map-canvas" role="region" aria-label={visualSummary(spec, rows)} />
      <p className="map-inspection sr-only" aria-live="polite" aria-atomic="true">{mapInspection}</p>
    </div>
  );
  if (expanded) {
    return createPortal(
      <div className="expanded-chart-backdrop">
        <div
          className="expanded-chart-dialog"
          role="dialog"
          aria-modal="true"
          aria-labelledby={`expanded-map-title-${spec.id}`}
          ref={dialogRef}
          tabIndex={-1}
        >
          <header className="expanded-chart-header">
            <div>
              <p className="eyebrow">Expanded map</p>
              <h3 id={`expanded-map-title-${spec.id}`}>{spec.title}</h3>
            </div>
            <div className="expanded-chart-actions">
              <button className="chart-command" type="button" onClick={closeExpanded}>Close</button>
            </div>
          </header>
          <div className="expanded-chart-body">{mapStage}</div>
        </div>
      </div>,
      document.body,
    );
  }
  return mapStage;
}

function Kpi({ spec, dataset }: { spec: Extract<VisualizationSpec, { kind: "kpi" }>; dataset?: Dataset }) {
  const enhanced = spec as typeof spec & KpiEnhancements;
  const value = dataset?.rows[0]?.[spec.value_field];
  const numericValue = asNumber(value);
  const comparison = enhanced.comparison_field ? dataset?.rows[0]?.[enhanced.comparison_field] : null;
  const scaleValues = [
    0,
    numericValue,
    enhanced.baseline_value,
    ...(enhanced.thresholds || []).map((threshold) => threshold.value),
  ].filter((candidate): candidate is number => candidate != null && Number.isFinite(candidate));
  const scaleMinimum = scaleValues.length ? Math.min(...scaleValues, 0) : 0;
  const scaleMaximum = scaleValues.length ? Math.max(...scaleValues, 0) : 1;
  const scaleSpan = scaleMaximum === scaleMinimum ? 1 : scaleMaximum - scaleMinimum;
  const position = (candidate: number) => Math.max(0, Math.min(100, (candidate - scaleMinimum) / scaleSpan * 100));
  const zeroPosition = position(0);
  const valuePosition = numericValue === null ? zeroPosition : position(numericValue);
  const hasBullet = numericValue !== null
    && (enhanced.baseline_value != null || (enhanced.thresholds?.length ?? 0) > 0);
  return (
    <div className="kpi-canvas" role="img" aria-label={visualSummary(spec, dataset?.rows || [])}>
      <span>{spec.label}</span>
      <strong>{typeof value === "number" ? numberFormatter.format(value) : text(value)}{spec.unit ? <small> {spec.unit}</small> : null}</strong>
      {hasBullet && (
        <div className="kpi-bullet" aria-label={`Value ${formattedValue(numericValue, spec.unit)}${enhanced.baseline_value == null ? "" : `; baseline ${formattedValue(enhanced.baseline_value, spec.unit)}`}`}>
          <div className="kpi-bullet-track">
            <span
              className="kpi-bullet-value"
              style={{
                left: `${Math.min(zeroPosition, valuePosition)}%`,
                width: `${Math.max(.35, Math.abs(valuePosition - zeroPosition))}%`,
              }}
            />
            <span className="kpi-bullet-zero" style={{ left: `${zeroPosition}%` }} title="Zero" />
            {enhanced.baseline_value != null && (
              <span className="kpi-bullet-baseline" style={{ left: `${position(enhanced.baseline_value)}%` }} />
            )}
            {(enhanced.thresholds || []).map((threshold) => (
              <span
                key={threshold.id}
                className="kpi-bullet-threshold"
                title={`${threshold.label}: ${formattedValue(threshold.value, threshold.unit || spec.unit)}`}
                style={{ left: `${position(threshold.value)}%` }}
              />
            ))}
          </div>
          <div className="kpi-bullet-labels">
            <span>{formattedValue(scaleMinimum, spec.unit)}</span>
            {enhanced.baseline_value != null && <span>Baseline {formattedValue(enhanced.baseline_value, spec.unit)}</span>}
            {(enhanced.thresholds || []).map((threshold) => <span key={threshold.id}>{threshold.label} {formattedValue(threshold.value, threshold.unit || spec.unit)}</span>)}
            <span>{formattedValue(scaleMaximum, spec.unit)}</span>
          </div>
        </div>
      )}
      {comparison != null && <em>Comparison: {formattedValue(comparison, enhanced.comparison_field ? unitFor(dataset?.columns || [], enhanced.comparison_field) : undefined)}</em>}
      {spec.trend_field && dataset?.rows[0]?.[spec.trend_field] != null && <em>{text(dataset.rows[0][spec.trend_field])}</em>}
    </div>
  );
}

function visualizationKindLabel(spec: VisualizationSpec, rows: DataRow[]): string {
  const profile = semanticProfile(spec, rows);
  const labels: Record<PresentationProfile, string> = {
    "metric-bullet": "Metric + quantitative bullet",
    "temporal-focus-context": "Time series + focus/context",
    "ranking-lollipop": "Ranked lollipop",
    "category-comparison": "Category comparison",
    "composition-ribbon": "Composition ribbon",
    "correlation-fit": "Correlation + fitted association",
    "uncertainty-range": "Uncertainty range",
    "forecast-fan-interval": "Forecast fan + interval",
    "distribution-histogram-summary": "Histogram + Tukey summary",
    "distribution-box": "Tukey distribution",
    "distribution-percentile": "Percentile profile",
    "calendar-pattern-heatmap": "Calendar pattern heatmap",
    "geospatial-investigation": "Geospatial investigation",
    "event-rail": "Chronological event rail",
    "verified-table": "Data table",
    "explicit-omission": "Explicit omission",
  };
  return labels[profile];
}

function visualizationUnit(spec: VisualizationSpec, dataset?: Dataset): string | null {
  if (spec.kind === "kpi" || spec.kind === "forecast" || spec.kind === "distribution" || spec.kind === "heatmap") return spec.unit || null;
  if (spec.kind === "cartesian") {
    const units = Array.from(new Set(spec.y_fields.map((field) => spec.y_unit || unitFor(dataset?.columns || [], field)).filter(Boolean)));
    return units.length === 1 ? units[0]! : null;
  }
  if (spec.kind === "map" && spec.value_field) return unitFor(dataset?.columns || [], spec.value_field) || null;
  return null;
}

export function Visualization({
  visualization,
  datasets,
  primary = false,
  contractVersion = "2.0",
}: VisualizationProps) {
  const staticObservationKeyRef = useRef<string | null>(null);
  const staticRenderStarted = window.performance.now();
  const [showData, setShowData] = useState(false);
  const resolvedTheme = useResolvedTheme();
  const legacyPresentation = useMemo(
    () => legacyCarbonPresentation(contractVersion, visualization, datasets),
    [contractVersion, datasets, visualization],
  );
  const presentedVisualization = legacyPresentation.visualization;
  const canonicalDataset = legacyPresentation.dataset;
  const dataset = useMemo(
    () => timelinePresentationDataset(
      presentedVisualization,
      canonicalDataset,
    ),
    [canonicalDataset, presentedVisualization],
  );
  const fallback = useMemo(() => datasetFor(visualization, datasets, true), [visualization, datasets]);
  const rows = dataset?.rows || [];
  const excludedTimelineRows = (
    presentedVisualization.kind === "timeline"
    && fallback
    && dataset
    && fallback.id !== dataset.id
  )
    ? Math.max(0, fallback.row_count - dataset.row_count)
    : (
      presentedVisualization.kind === "timeline"
      && canonicalDataset
      && dataset
      ? Math.max(0, canonicalDataset.row_count - dataset.row_count)
      : 0
    );
  const isOmitted = presentedVisualization.kind === "omitted" || !rows.length;
  const summaryId = `visual-summary-${presentedVisualization.id}`;
  const dataId = `visual-data-${presentedVisualization.id}`;
  const unit = visualizationUnit(presentedVisualization, dataset);
  const profile = semanticProfile(presentedVisualization, rows);
  const interactiveChart = !isOmitted && !["kpi", "table"].includes(presentedVisualization.kind);
  const sourceDatasetIds = useMemo(() => {
    const distribution = visualization.kind === "distribution" ? visualization : null;
    return Array.from(new Set([
      visualization.dataset_id,
      visualization.table_fallback_dataset_id,
      dataset?.id,
      fallback?.id,
      distribution?.summary_dataset_id,
      distribution?.outlier_dataset_id,
    ].filter((value): value is string => Boolean(value))));
  }, [dataset?.id, fallback?.id, visualization]);
  const staticFallbackReason = presentedVisualization.kind === "omitted"
    ? `${presentedVisualization.reason_code}: ${presentedVisualization.reason}`
    : !rows.length
      ? "missing_or_empty_dataset"
      : null;
  const staticInteractionMode = presentedVisualization.kind === "omitted"
    ? "omitted"
    : !rows.length
      ? "render_fallback"
      : "render";
  useEffect(() => {
    if (interactiveChart) return;
    const observationKey = JSON.stringify({
      id: visualization.id,
      profile,
      contractVersion,
      sourceDatasetIds,
      fallback: staticFallbackReason,
      rows: dataset?.row_count ?? rows.length,
    });
    if (staticObservationKeyRef.current === observationKey) return;
    staticObservationKeyRef.current = observationKey;
    recordVisualizationObservation({
      visualization_id: visualization.id,
      chart_profile: visualizationProfile(visualization),
      ...diagnosticFields(presentedVisualization, rows),
      visualization_contract_version: contractVersion,
      render_latency_ms: Math.max(0, window.performance.now() - staticRenderStarted),
      fallback_reason: staticFallbackReason,
      interaction_mode: staticInteractionMode,
      source_dataset_ids: sourceDatasetIds,
    });
  }, [
    contractVersion,
    dataset?.row_count,
    interactiveChart,
    rows.length,
    sourceDatasetIds,
    staticFallbackReason,
    staticInteractionMode,
    staticRenderStarted,
    visualization,
    presentedVisualization,
    profile,
  ]);
  const observeStaticInteraction = (interactionMode: string) => {
    recordVisualizationObservation({
      visualization_id: visualization.id,
      chart_profile: visualizationProfile(visualization),
      ...diagnosticFields(presentedVisualization, rows),
      visualization_contract_version: contractVersion,
      render_latency_ms: null,
      fallback_reason: staticFallbackReason,
      interaction_mode: interactionMode,
      source_dataset_ids: sourceDatasetIds,
    });
  };
  return (
    <section
      className={`visualization visualization-profile--${profile} ${primary ? "visualization-primary" : "visualization-secondary"}`}
      data-profile={profile}
      data-legacy-enrichment={legacyPresentation.state === "enriched" ? "carbon-uncertainty" : undefined}
      aria-labelledby={`visual-title-${presentedVisualization.id}`}
      aria-describedby={!isOmitted ? summaryId : undefined}
    >
      <header className="visualization-header">
        <div>
          <p className="eyebrow visualization-header-meta">
            {visualizationKindLabel(presentedVisualization, rows)} · {dataset?.row_count ?? rows.length} {dataset?.row_count === 1 ? "row" : "rows"}{unit ? ` · ${unit}` : ""}
          </p>
          <h3 id={`visual-title-${presentedVisualization.id}`}>{presentedVisualization.title}</h3>
        </div>
        {!isOmitted && !interactiveChart && (
          <button
            className="button button-quiet"
            onClick={() => {
              observeStaticInteraction(showData ? "hide_data" : "view_data");
              setShowData((value) => !value);
            }}
            aria-expanded={showData}
            aria-controls={dataId}
          >
            {showData ? "Hide data" : "View data"}
          </button>
        )}
      </header>
      {legacyPresentation.notice && (
        <div className="chart-analysis-strip legacy-visualization-notice" role="note">
          <strong>Saved chart</strong>
          <span>{legacyPresentation.notice}</span>
        </div>
      )}
      {presentedVisualization.kind === "timeline" && excludedTimelineRows > 0 && (
        <div
          className="chart-analysis-strip timeline-scope-notice"
          role="note"
          data-testid="timeline-scope-notice"
        >
          <strong>Timeline scope</strong>
          <span>
            {dataset?.row_count ?? rows.length} valid ETA
            {(dataset?.row_count ?? rows.length) === 1 ? "" : "s"} plotted.
            {" "}{excludedTimelineRows} row{excludedTimelineRows === 1 ? "" : "s"} without a valid timestamp remain in the data table.
          </span>
        </div>
      )}
      {isOmitted
        ? (
          <>
            <div className="empty-visual" role="status">
              <strong>No visualization is available.</strong>
              <span>{presentedVisualization.kind === "omitted" ? presentedVisualization.reason : "This result does not contain sufficient suitable data for a trustworthy chart."}</span>
            </div>
            {presentedVisualization.kind === "timeline" && fallback?.rows.length ? (
              <div className="visual-fallback" data-testid="timeline-table-fallback">
                <p className="chart-summary">
                  No valid timestamps can be plotted. The complete data rows are shown below.
                </p>
                <DataTable dataset={fallback} />
              </div>
            ) : null}
          </>
        )
        : presentedVisualization.kind === "kpi"
          ? <Kpi spec={presentedVisualization} dataset={dataset} />
          : presentedVisualization.kind === "table"
            ? <DataTable dataset={dataset} fields={presentedVisualization.visible_fields} />
            : presentedVisualization.kind === "map"
              ? <MapVisualization spec={presentedVisualization} dataset={dataset} fallbackDataset={fallback} resolvedTheme={resolvedTheme} showData={showData} onToggleData={() => setShowData((value) => !value)} contractVersion={contractVersion} />
              : <EChart spec={presentedVisualization} dataset={dataset} fallbackDataset={fallback} datasets={datasets} resolvedTheme={resolvedTheme} showData={showData} onToggleData={() => setShowData((value) => !value)} contractVersion={contractVersion} />}
      {!isOmitted && <p className="chart-summary" id={summaryId}>{visualSummary(presentedVisualization, rows)}</p>}
      {!!presentedVisualization.citations.length && <p className="sr-only">Evidence references: {presentedVisualization.citations.join(", ")}.</p>}
      {showData && presentedVisualization.kind !== "table" && <div id={dataId}><DataTable dataset={fallback || dataset} /></div>}
    </section>
  );
}
