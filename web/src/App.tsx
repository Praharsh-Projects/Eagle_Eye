import {
  FormEvent,
  type KeyboardEvent as ReactKeyboardEvent,
  type ReactNode,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  Navigate,
  NavLink,
  Route,
  Routes,
  useLocation,
  useNavigate,
} from "react-router-dom";
import {
  ApiError,
  exportDataset,
  loadCapabilities,
  reportIssue,
  submitQuery,
} from "./api";
import { ChartInsights } from "./components/ChartInsights";
import {
  OperationalOverview,
  deriveHistoricalCountryCoverage,
  type OverviewCapabilityState,
  type OverviewStatus,
} from "./components/OperationalOverview";
import { Visualization } from "./components/Visualization";
import {
  ETA_WATCH_SAMPLE_GROUPS,
  QUERY_CATEGORY_HELP,
  QUERY_CATEGORY_PAGE_ORDER,
  SAMPLE_QUERIES_BY_CATEGORY,
  type QueryCategoryHelp,
  type SampleQueryCategory,
} from "./data/sampleQueries";
import { normalizeWorkspaceRoute } from "./data/analysisPages";
import type {
  AnswerEnvelope,
  CapabilityResponse,
  Dataset,
  ExportResponse,
  OperationalAction,
  OperationalBrief as OperationalBriefContract,
  QueryFilters,
  QueryRequestPayload,
  WorkspaceRoute,
} from "./types";

type PageKey =
  | "overview"
  | "analysis"
  | "traffic"
  | "vessels"
  | "eta"
  | "pressure"
  | "carbon";

type ModalName =
  | "samples"
  | "filters"
  | "settings"
  | "report"
  | "provenance"
  | null;

interface PageDefinition {
  key: PageKey;
  path: WorkspaceRoute;
  label: string;
  internalId: string;
  scope: string;
  category?: SampleQueryCategory;
}

interface UiFilters {
  port: string;
  vesselType: string;
  vesselName: string;
  mmsi: string;
  imo: string;
  anomaly: boolean;
  useDateRange: boolean;
  dateFrom: string;
  dateTo: string;
}

interface UiSettings {
  topK: number;
  railCollapsed: boolean;
}

interface HistoryRecord {
  id: string;
  question: string;
  createdAt: string;
  route: WorkspaceRoute;
  conversationId: string;
  result: AnswerEnvelope;
  request?: QueryRequestPayload;
  schemaVersion: 1 | 2 | 3;
}

interface LegacyHistoryRecord {
  id?: string;
  question?: string;
  createdAt?: string;
  result?: AnswerEnvelope;
}

const HISTORY_KEY = "eagle-eye-analysis-history-v3";
const VERSION_TWO_HISTORY_KEY = "eagle-eye-analysis-history-v2";
const LEGACY_HISTORY_KEY = "eagle-eye-analysis-history-v1";
const CONVERSATIONS_KEY = "eagle-eye-conversations-v2";
const LEGACY_CONVERSATION_KEY = "eagle-eye-conversation-id-v1";
const SETTINGS_KEY = "eagle-eye-ui-settings-v1";

const DEFAULT_FILTERS: UiFilters = {
  port: "",
  vesselType: "",
  vesselName: "",
  mmsi: "",
  imo: "",
  anomaly: false,
  useDateRange: false,
  dateFrom: "",
  dateTo: "",
};

const DEFAULT_SETTINGS: UiSettings = {
  topK: 5,
  railCollapsed: false,
};

const PAGE_DEFINITIONS: PageDefinition[] = [
  {
    key: "overview",
    path: "/overview",
    label: "Overview",
    internalId: "Overview",
    scope: "Validated historical coverage, live AIS-source readiness, and recent analytical work.",
  },
  {
    key: "analysis",
    path: "/analysis",
    label: "Analysis Desk",
    internalId: "Chat Assistant",
    scope: "Cross-domain historical analysis, source-grounded maritime research, and general assistance.",
  },
  {
    key: "traffic",
    path: "/traffic-monitoring",
    label: "Traffic Monitoring",
    internalId: "Traffic Monitoring",
    scope: "Arrival volumes, time patterns, port comparisons, and route durations.",
    category: "Traffic Monitoring",
  },
  {
    key: "vessels",
    path: "/vessel-investigation",
    label: "Vessel Investigation",
    internalId: "Vessel Investigation",
    scope: "MMSI-level port stays, movement evidence, and suspicious AIS jump records.",
    category: "Vessel Investigation",
  },
  {
    key: "eta",
    path: "/eta-delay",
    label: "ETA Watch",
    internalId: "ETA & Delay",
    scope: "Sweden-first Baltic inbound watchlists, vessel-reported ETAs, positions, revisions, and signal-quality exceptions.",
    category: "ETA & Delay",
  },
  {
    key: "pressure",
    path: "/port-pressure",
    label: "Port Pressure",
    internalId: "Port Pressure",
    scope: "Observed pressure indices, baselines, contributors, and port comparisons.",
    category: "Port Pressure",
  },
  {
    key: "carbon",
    path: "/carbon-emissions",
    label: "Carbon Emissions",
    internalId: "Carbon Emissions",
    scope: "Deterministic TTW and WTW emissions by port, period, pollutant, or vessel call.",
    category: "Carbon Emissions",
  },
];

function newConversationId(page: PageKey): string {
  return `web-${page}-${crypto.randomUUID()}`;
}

function safeRead<T>(storage: Storage, key: string, fallback: T): T {
  try {
    const value = storage.getItem(key);
    return value ? (JSON.parse(value) as T) : fallback;
  } catch {
    return fallback;
  }
}

function readHistory(): HistoryRecord[] {
  if (localStorage.getItem(HISTORY_KEY) !== null) {
    const current = safeRead<HistoryRecord[]>(localStorage, HISTORY_KEY, []);
    if (!Array.isArray(current)) return [];
    return current
      .filter((item) => Boolean(item?.question && item?.result))
      .map((item) => ({
        ...item,
        route: normalizeWorkspaceRoute(item.route),
        schemaVersion: 3 as const,
      }))
      .slice(0, 24);
  }

  const versionTwo = safeRead<HistoryRecord[]>(
    localStorage,
    VERSION_TWO_HISTORY_KEY,
    [],
  );
  if (Array.isArray(versionTwo) && versionTwo.length) {
    return versionTwo
      .filter((item) => Boolean(item?.question && item?.result))
      .map((item) => ({
        ...item,
        route: normalizeWorkspaceRoute(item.route),
        schemaVersion: 2 as const,
        request: undefined,
      }))
      .slice(0, 24);
  }

  const legacy = safeRead<LegacyHistoryRecord[]>(
    localStorage,
    LEGACY_HISTORY_KEY,
    [],
  );
  if (!Array.isArray(legacy)) return [];
  return legacy
    .filter(
      (item): item is LegacyHistoryRecord & {
        question: string;
        result: AnswerEnvelope;
      } => Boolean(item.question && item.result),
    )
    .map((item) => ({
      id: item.id || item.result.trace.trace_id || crypto.randomUUID(),
      question: item.question,
      createdAt: item.createdAt || new Date().toISOString(),
      route: "/analysis" as const,
      conversationId:
        item.result.conversation_id ||
        sessionStorage.getItem(LEGACY_CONVERSATION_KEY) ||
        newConversationId("analysis"),
      result: item.result,
      request: undefined,
      schemaVersion: 1 as const,
    }))
    .slice(0, 24);
}

function readConversations(): Partial<Record<PageKey, string>> {
  const current = safeRead<Partial<Record<PageKey, string>>>(
    sessionStorage,
    CONVERSATIONS_KEY,
    {},
  );
  const legacy = sessionStorage.getItem(LEGACY_CONVERSATION_KEY);
  if (!current.analysis && legacy) current.analysis = legacy;
  return current;
}

function readSettings(): UiSettings {
  const stored = safeRead<Partial<UiSettings>>(
    localStorage,
    SETTINGS_KEY,
    {},
  );
  return {
    topK:
      typeof stored.topK === "number"
        ? Math.max(0, Math.min(8, Math.round(stored.topK)))
        : DEFAULT_SETTINGS.topK,
    railCollapsed: Boolean(stored.railCollapsed),
  };
}

function routeForPath(pathname: string): PageDefinition {
  return (
    PAGE_DEFINITIONS.find((page) => page.path === pathname) ||
    PAGE_DEFINITIONS[0]
  );
}

function stateLabel(value: string): string {
  return value
    .replaceAll("_", " ")
    .toLowerCase()
    .replace(/^./, (letter) => letter.toUpperCase());
}

function displayKey(value: string): string {
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function sampleCategoryLabel(category: SampleQueryCategory): string {
  return category === "ETA & Delay" ? "ETA Watch" : category;
}

function hasDisplayValue(value: unknown): boolean {
  return !(
    value === null ||
    value === undefined ||
    value === "" ||
    (Array.isArray(value) && value.length === 0)
  );
}

function displayValue(value: unknown): string {
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "object" && value !== null)
    return JSON.stringify(value);
  return String(value ?? "—");
}

function activeFilterCount(filters: UiFilters): number {
  return [
    Boolean(filters.port.trim()),
    Boolean(filters.vesselType.trim()),
    Boolean(filters.vesselName.trim()),
    Boolean(filters.mmsi.trim()),
    Boolean(filters.imo.trim()),
    filters.anomaly,
    filters.useDateRange &&
      Boolean(filters.dateFrom || filters.dateTo),
  ].filter(Boolean).length;
}

function apiFilters(filters: UiFilters): QueryFilters {
  return {
    port: filters.port.trim() || undefined,
    vessel_type: filters.vesselType.trim() || undefined,
    vessel_name: filters.vesselName.trim() || undefined,
    mmsi: filters.mmsi.trim() || undefined,
    imo: filters.imo.trim() || undefined,
    anomaly: filters.anomaly || undefined,
    date_from:
      filters.useDateRange && filters.dateFrom
        ? filters.dateFrom
        : undefined,
    date_to:
      filters.useDateRange && filters.dateTo ? filters.dateTo : undefined,
  };
}

function uiFiltersFromRequest(request: QueryRequestPayload): UiFilters {
  const filters = request.filters || {};
  return {
    port: filters.port || "",
    vesselType: filters.vessel_type || "",
    vesselName: filters.vessel_name || "",
    mmsi: filters.mmsi || "",
    imo: filters.imo || "",
    anomaly: Boolean(filters.anomaly),
    useDateRange: Boolean(filters.date_from || filters.date_to),
    dateFrom: filters.date_from || "",
    dateTo: filters.date_to || "",
  };
}

function requestForHistoryRecord(
  record: HistoryRecord,
  currentTopK: number,
): QueryRequestPayload {
  if (record.schemaVersion === 3 && record.request) {
    return {
      ...record.request,
      filters: record.request.filters
        ? { ...record.request.filters }
        : undefined,
    };
  }

  const scope = record.result.applied_scope;
  const port = scope.ports.length === 1 ? scope.ports[0] : undefined;
  return {
    question: record.question,
    conversation_id:
      record.conversationId || record.result.conversation_id,
    top_k_evidence: currentTopK,
    filters: {
      port,
      date_from: scope.date_from || undefined,
      date_to: scope.date_to || undefined,
      vessel_type: scope.vessel_type || undefined,
      vessel_name: scope.vessel_name || undefined,
      mmsi: scope.mmsi || undefined,
      imo: scope.imo || undefined,
    },
  };
}

function isSuccessfulResultState(state: AnswerEnvelope["state"]): boolean {
  return (
    state === "COMPUTED" ||
    state === "PARTIAL" ||
    state === "RETRIEVED" ||
    state === "GENERAL"
  );
}

function capabilityEvidenceLabel(
  capabilities: CapabilityResponse | null,
): string {
  if (!capabilities) return "Evidence status unavailable";
  return capabilities.retrieval?.available
    ? "Evidence retrieval ready"
    : "Evidence retrieval unavailable";
}

function sourceLabel(result: AnswerEnvelope): string {
  if (result.trace.sources?.length) return result.trace.sources.join(", ");
  const sources = Array.from(
    new Set(result.evidence.map((item) => item.source_type)),
  );
  if (sources.length) return sources.join(", ");
  return result.mode === "analytics"
    ? "Validated structured data"
    : stateLabel(result.mode);
}

function useDarkTheme(): void {
  useEffect(() => {
    document.documentElement.dataset.theme = "dark";
    document.documentElement.style.colorScheme = "dark";
    document
      .querySelector<HTMLMetaElement>('meta[name="theme-color"]')
      ?.setAttribute("content", "#071014");
  }, []);
}

function Modal({
  title,
  children,
  onClose,
  size = "medium",
}: {
  title: string;
  children: ReactNode;
  onClose: () => void;
  size?: "medium" | "wide";
}) {
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const previouslyFocused = document.activeElement as HTMLElement | null;
    const app = document.querySelector<HTMLElement>(".app-frame");
    if (app) app.inert = true;
    document.body.classList.add("modal-open");

    const panel = panelRef.current;
    const focusable = () =>
      Array.from(
        panel?.querySelectorAll<HTMLElement>(
          'button:not([disabled]), a[href], input:not([disabled]), textarea:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])',
        ) || [],
      );
    requestAnimationFrame(() => focusable()[0]?.focus());

    const handleKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
        return;
      }
      if (event.key !== "Tab") return;
      const items = focusable();
      if (!items.length) return;
      const first = items[0];
      const last = items[items.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", handleKey);
    return () => {
      document.removeEventListener("keydown", handleKey);
      document.body.classList.remove("modal-open");
      if (app) app.inert = false;
      previouslyFocused?.focus();
    };
  }, [onClose]);

  return (
    <div className="modal-layer">
      <button
        className="modal-scrim"
        type="button"
        aria-label={`Close ${title}`}
        onClick={onClose}
      />
      <div
        className={`modal-panel modal-panel-${size}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="modal-title"
        ref={panelRef}
      >
        <header className="modal-header">
          <h2 id="modal-title">{title}</h2>
          <button className="icon-button" type="button" onClick={onClose}>
            <span aria-hidden="true">×</span>
            <span className="sr-only">Close</span>
          </button>
        </header>
        {children}
      </div>
    </div>
  );
}

function Navigation({
  currentPage,
  history,
  open,
  collapsed,
  allowCollapse,
  onClose,
  onToggleCollapse,
  onNewAnalysis,
  onSelectHistory,
  onSettings,
}: {
  currentPage: PageDefinition;
  history: HistoryRecord[];
  open: boolean;
  collapsed: boolean;
  allowCollapse: boolean;
  onClose: () => void;
  onToggleCollapse: () => void;
  onNewAnalysis: () => void;
  onSelectHistory: (record: HistoryRecord) => void;
  onSettings: () => void;
}) {
  return (
    <>
      <aside
        className={`nav-rail ${open ? "nav-rail-open" : ""} ${collapsed ? "nav-rail-collapsed" : ""}`}
        data-testid="nav-rail"
        aria-label="Eagle Eye navigation"
      >
        <header className="brand-block">
          <div className="brand-code" aria-hidden="true">
            EE
          </div>
          <div className="brand-copy">
            <strong>Eagle Eye</strong>
            <span>Maritime operations</span>
          </div>
          <button
            type="button"
            className="nav-close"
            onClick={onClose}
            aria-label="Close navigation"
          >
            ×
          </button>
        </header>

        <button
          type="button"
          className="new-analysis-action"
          aria-label="New analysis"
          onClick={onNewAnalysis}
        >
          <span aria-hidden="true">＋</span>
          <span className="nav-label">New analysis</span>
        </button>

        <nav className="primary-nav" aria-label="Operational areas">
          {PAGE_DEFINITIONS.map((page) => (
            <NavLink
              key={page.key}
              to={page.path}
              onClick={onClose}
              className={({ isActive }) =>
                `nav-link ${isActive ? "nav-link-active" : ""}`
              }
              aria-label={page.label}
            >
              <span className="nav-index" aria-hidden="true">
                {String(PAGE_DEFINITIONS.indexOf(page) + 1).padStart(2, "0")}
              </span>
              <span className="nav-label">{page.label}</span>
            </NavLink>
          ))}
        </nav>

        <section className="recent-nav" aria-labelledby="recent-nav-title">
          <h2 id="recent-nav-title" className="rail-heading">
            Recent analyses
          </h2>
          <div className="recent-records">
            {history.slice(0, 6).map((record) => (
              <button
                key={`${record.id}-${record.createdAt}`}
                type="button"
                className="recent-record"
                onClick={() => onSelectHistory(record)}
                title={record.question}
              >
                <span>{record.question}</span>
                <time dateTime={record.createdAt}>
                  {new Date(record.createdAt).toLocaleDateString(undefined, {
                    month: "short",
                    day: "numeric",
                  })}
                </time>
              </button>
            ))}
            {!history.length && (
              <p className="rail-empty">
                Completed analysis records appear here.
              </p>
            )}
          </div>
        </section>

        <footer className="rail-footer">
          <button
            type="button"
            className="rail-control"
            aria-label="Workspace settings"
            onClick={onSettings}
            data-testid="settings-button"
          >
            <span aria-hidden="true">⚙</span>
            <span className="nav-label">Workspace settings</span>
          </button>
          {allowCollapse && (
            <button
              type="button"
              className="rail-control collapse-control"
              onClick={onToggleCollapse}
              aria-label={
                collapsed ? "Expand navigation" : "Collapse navigation"
              }
            >
              <span aria-hidden="true">{collapsed ? "›" : "‹"}</span>
              <span className="nav-label">
                {collapsed ? "Expand" : "Collapse"}
              </span>
            </button>
          )}
        </footer>
      </aside>
      {open && (
        <button
          className="nav-scrim"
          type="button"
          aria-label="Close navigation overlay"
          onClick={onClose}
        />
      )}
      <span className="sr-only" aria-live="polite">
        Current section: {currentPage.label}
      </span>
    </>
  );
}

function UtilityBar({
  page,
  onOpenNavigation,
}: {
  page: PageDefinition;
  onOpenNavigation: () => void;
}) {
  return (
    <header className="utility-bar">
      <button
        className="mobile-nav-toggle"
        type="button"
        onClick={onOpenNavigation}
        aria-label="Open navigation"
        data-testid="nav-toggle"
      >
        <span aria-hidden="true">☰</span>
      </button>
      <div className="utility-current">
        <span>Workspace</span>
        <strong>{page.label}</strong>
      </div>
      <span className="utility-context">Maritime operations workspace</span>
    </header>
  );
}

function PageHeading({ page }: { page: PageDefinition }) {
  return (
    <header className="page-heading">
      <div>
        <p className="section-code">
          {String(PAGE_DEFINITIONS.indexOf(page) + 1).padStart(2, "0")} /{" "}
          {PAGE_DEFINITIONS.length.toString().padStart(2, "0")}
        </p>
        <h1>{page.label}</h1>
      </div>
      <p>{page.scope}</p>
    </header>
  );
}

interface ManifestDatasetSummary {
  id: string;
  label: string;
  rows: number | null;
  coverage: string | null;
  readable: boolean | null;
}

const COUNT_FORMATTER = new Intl.NumberFormat("en-US");

function recordValue(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function finiteRowCount(value: unknown): number | null {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0)
    return null;
  return Math.round(value);
}

function datasetCoverage(value: unknown): string | null {
  const coverage = recordValue(value);
  if (!coverage) return null;
  const minima: string[] = [];
  const maxima: string[] = [];
  Object.values(coverage).forEach((rawRange) => {
    const range = recordValue(rawRange);
    if (!range) return;
    if (typeof range.min === "string") minima.push(range.min);
    if (typeof range.max === "string") maxima.push(range.max);
  });
  if (!minima.length && !maxima.length) return null;
  const minimum = minima.sort()[0] || maxima.sort()[0];
  const maximum = maxima.sort().at(-1) || minima.sort().at(-1);
  return minimum === maximum ? minimum : `${minimum} to ${maximum}`;
}

function manifestDatasetRows(
  capabilities: CapabilityResponse | null,
): ManifestDatasetSummary[] {
  const manifest = capabilities?.data_manifest || {};
  const tables = recordValue(manifest.tables);
  if (tables) {
    return Object.entries(tables)
      .map(([id, rawSummary]) => {
        const summary = recordValue(rawSummary) || {};
        return {
          id,
          label: displayKey(id.replace(/\.[^.]+$/, "")),
          rows: finiteRowCount(summary.rows),
          coverage: datasetCoverage(summary.coverage),
          readable:
            typeof summary.readable === "boolean" ? summary.readable : null,
        };
      })
      .sort((left, right) => left.label.localeCompare(right.label));
  }

  // Older capability fixtures exposed a flat row_counts mapping. Preserve
  // that display path while preferring the live manifest's table summaries.
  const legacyCounts = recordValue(manifest.row_counts);
  return legacyCounts
    ? Object.entries(legacyCounts)
        .map(([id, rawRows]) => ({
          id,
          label: displayKey(id.replace(/\.[^.]+$/, "")),
          rows: finiteRowCount(rawRows),
          coverage: null,
          readable: null,
        }))
        .sort((left, right) => left.label.localeCompare(right.label))
    : [];
}

function OverviewPage({
  capabilities,
  capabilityState,
  history,
  onEnterAnalysis,
  onSelectHistory,
  onOpenProvenance,
}: {
  capabilities: CapabilityResponse | null;
  capabilityState: OverviewCapabilityState;
  history: HistoryRecord[];
  onEnterAnalysis: () => void;
  onSelectHistory: (record: HistoryRecord) => void;
  onOpenProvenance: () => void;
}) {
  const datasets = manifestDatasetRows(capabilities);
  const reportedRows = datasets
    .map((dataset) => dataset.rows)
    .filter((value): value is number => value !== null);
  const hasCompleteRowCounts =
    datasets.length > 0 && reportedRows.length === datasets.length;
  const totalRows = reportedRows.reduce((sum, value) => sum + value, 0);
  const historicalCoverage = deriveHistoricalCountryCoverage(
    capabilities?.data_manifest?.available_ports,
  );
  const reportedLiveProvider = capabilities?.live_eta?.provider?.trim();
  const liveProvider =
    reportedLiveProvider?.toLowerCase() === "aisstream"
      ? "AISStream"
      : reportedLiveProvider
        ? displayKey(reportedLiveProvider)
        : "Not reported";
  const liveHealth = capabilities?.live_eta?.source_health;
  const sourceStatus: OverviewStatus =
    capabilityState === "loading"
      ? {
          label: "Loading",
          detail: "Checking Baltic AIS source readiness",
          tone: "neutral",
        }
      : capabilityState === "unavailable" || !capabilities
        ? {
            label: "Workspace status unavailable",
            detail:
              "The capability service did not return a verified AIS source state.",
            tone: "error",
          }
        : !capabilities.live_eta?.available ||
            liveHealth === "unavailable"
          ? {
              label: "Unavailable",
              detail:
                capabilities.live_eta?.reason ||
                "ETA Watch source unavailable",
              tone: "error",
            }
          : liveHealth === "stale"
            ? {
                label: "Stale",
                detail:
                  capabilities.live_eta.reason ||
                  "The last AIS observation is outside the freshness window.",
                tone: "warning",
              }
            : liveHealth === "connecting" || liveHealth === "warming"
              ? {
                  label:
                    liveHealth === "connecting" ? "Connecting" : "Warming",
                  detail:
                    capabilities.live_eta.reason ||
                    "The Baltic collector is building its current snapshot.",
                  tone: "warning",
                }
              : liveHealth === "live"
                ? {
                    label: "Live",
                    detail:
                      liveProvider === "Not reported"
                        ? "Baltic vessel watch reported ready"
                        : `${liveProvider} Baltic vessel watch ready`,
                    tone: "success",
                  }
                : {
                    label: "Status unverified",
                    detail:
                      capabilities.live_eta.reason ||
                      "The AIS source did not report a verified health state.",
                    tone: "warning",
                  };
  const evidenceStatus: OverviewStatus =
    capabilityState === "loading"
      ? {
          label: "Loading",
          detail: "Checking evidence retrieval readiness",
          tone: "neutral",
        }
      : capabilityState === "unavailable" || !capabilities
        ? {
            label: "Unavailable",
            detail: "Evidence status unavailable",
            tone: "error",
          }
        : capabilities.retrieval?.available
          ? {
              label: "Ready",
              detail: "Evidence retrieval ready",
              tone: "success",
            }
          : {
              label: "Unavailable",
              detail:
                capabilities.retrieval?.reason ||
                "Evidence retrieval unavailable",
              tone: "warning",
            };

  return (
    <div className="overview-page">
      <OperationalOverview
        capabilityState={capabilityState}
        boundingBoxes={capabilities?.live_eta?.bounding_boxes}
        liveCountryCodes={capabilities?.live_eta?.country_scope || []}
        historicalCoverage={historicalCoverage}
        historicalFrom={capabilities?.freshness?.data_from}
        historicalTo={capabilities?.freshness?.data_to}
        provider={liveProvider}
        sourceStatus={sourceStatus}
        evidenceStatus={evidenceStatus}
        horizonHours={capabilities?.live_eta?.maximum_horizon_hours}
        timezone={capabilities?.live_eta?.timezone}
        datasetCount={datasets.length}
        recordCount={hasCompleteRowCounts ? totalRows : null}
        recentRecords={history.slice(0, 6)}
        onEnterAnalysis={onEnterAnalysis}
        onSelectHistory={onSelectHistory}
        onOpenProvenance={onOpenProvenance}
      />
    </div>
  );
}

function DataProvenanceDialog({
  capabilities,
  onClose,
}: {
  capabilities: CapabilityResponse | null;
  onClose: () => void;
}) {
  const datasets = manifestDatasetRows(capabilities);
  const reportedRows = datasets
    .map((dataset) => dataset.rows)
    .filter((value): value is number => value !== null);
  const hasCompleteRowCounts =
    datasets.length > 0 && reportedRows.length === datasets.length;
  const totalRows = reportedRows.reduce((sum, value) => sum + value, 0);
  const inventorySummary = datasets.length
    ? `${datasets.length} validated datasets · ${
        hasCompleteRowCounts
          ? `${COUNT_FORMATTER.format(totalRows)} historical records`
          : "row totals unavailable"
      }`
    : "No table metadata";

  return (
    <Modal title="Data provenance" size="wide" onClose={onClose}>
      <div
        className="modal-content overview-provenance-modal dataset-inventory-body"
        data-testid="dataset-inventory"
      >
        <p className="overview-inventory-summary">{inventorySummary}</p>
        <p className="modal-note">
          Exact capability-manifest inventory. These tables describe the
          historical analytical archive; they are not a live traffic feed.
        </p>
        <div className="compact-table-wrap" tabIndex={0}>
          <table aria-label="Dataset inventory">
            <thead>
              <tr>
                <th>Dataset</th>
                <th>Rows</th>
                <th>Coverage</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {datasets.map((dataset) => (
                <tr key={dataset.id}>
                  <td>
                    <code>{dataset.id}</code>
                    <small>{dataset.label}</small>
                  </td>
                  <td>
                    {dataset.rows === null
                      ? "Not reported"
                      : COUNT_FORMATTER.format(dataset.rows)}
                  </td>
                  <td>{dataset.coverage || "Not reported"}</td>
                  <td>
                    {dataset.readable === true
                      ? "Ready"
                      : dataset.readable === false
                        ? "Unavailable"
                        : "Not reported"}
                  </td>
                </tr>
              ))}
              {!datasets.length && (
                <tr>
                  <td colSpan={4}>
                    Dataset table metadata was not reported.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </Modal>
  );
}

function DatasetTable({ dataset }: { dataset: Dataset }) {
  return (
    <div
      className="detail-table-wrap"
      tabIndex={0}
      aria-label={`Scrollable ${dataset.id} data`}
    >
      <table aria-label={`${dataset.id} dataset`}>
        <thead>
          <tr>
            {dataset.columns.map((column) => (
              <th key={column.field}>{column.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {dataset.rows.slice(0, 100).map((row, rowIndex) => (
            <tr key={rowIndex}>
              {dataset.columns.map((column) => (
                <td key={column.field}>
                  {displayValue(row[column.field])}
                  {column.unit &&
                    typeof row[column.field] === "number" &&
                    ` ${column.unit}`}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {dataset.row_count > 100 && (
        <p>Showing 100 of {dataset.row_count} rows.</p>
      )}
    </div>
  );
}

function isLegacyFintrafficResult(result: AnswerEnvelope): boolean {
  const sourceScope = String(result.plan.source_scope || "").toLowerCase();
  const traceSources = result.trace.sources.map((source) =>
    source.toLowerCase(),
  );
  return (
    sourceScope.includes("fintraffic") ||
    traceSources.some(
      (source) =>
        source.includes("fintraffic") ||
        source.includes("digitraffic") ||
        source.includes("portnet"),
    ) ||
    result.evidence.some((item) =>
      String(item.url || "").includes("meri.digitraffic.fi"),
    )
  );
}

function operationalActionLabel(action: OperationalAction): string {
  if (action === "locate_vessel") return "Locate vessel";
  if (action === "watch_next_six_hours") return "Watch next 6 hours";
  return "Inspect ETA changes";
}

function operationalFollowUp(
  action: OperationalAction,
  vesselLabel: string,
): string {
  if (action === "locate_vessel") {
    return `Where is ${vesselLabel} now, and what ETA is it transmitting?`;
  }
  if (action === "watch_next_six_hours") {
    return `Watch ${vesselLabel} over the next 6 hours and show its current reported ETA, position, speed, and observation time.`;
  }
  return `Show reported ETA changes for ${vesselLabel} in the last hour.`;
}

const operationalUtcFormatter = new Intl.DateTimeFormat("en-GB", {
  day: "2-digit",
  month: "short",
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
  timeZone: "UTC",
});

function formatOperationalUtc(value?: string | null): string {
  if (!value) return "Unavailable";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return `${operationalUtcFormatter.format(parsed).replace(",", " ·")} UTC`;
}

function operationalWindowLabel(
  start?: string | null,
  end?: string | null,
): string {
  if (!start || !end) return "Current source scope";
  const startDate = new Date(start);
  const endDate = new Date(end);
  if (
    Number.isNaN(startDate.getTime()) ||
    Number.isNaN(endDate.getTime()) ||
    endDate <= startDate
  ) {
    return `${formatOperationalUtc(start)}–${formatOperationalUtc(end)}`;
  }
  const hours = (endDate.getTime() - startDate.getTime()) / 3_600_000;
  const duration =
    hours < 1
      ? `${Math.round(hours * 60)} min`
      : Number.isInteger(hours)
        ? `${hours} hr`
        : `${hours.toFixed(1)} hr`;
  return `${duration} · ${formatOperationalUtc(start)}–${formatOperationalUtc(end)}`;
}

function operationalExceptionCount(
  brief: OperationalBriefContract,
  code: string,
): number {
  return brief.exceptions.find((item) => item.code === code)?.count || 0;
}

function operationalFinding(brief: OperationalBriefContract): string {
  const dueSoon = operationalExceptionCount(brief, "due_soon");
  const lowSpeed = operationalExceptionCount(brief, "low_speed");
  const etaChanged = operationalExceptionCount(brief, "eta_changed");
  const stalePosition = operationalExceptionCount(brief, "stale_position");
  const missingEta = operationalExceptionCount(brief, "missing_eta");
  const plural = (count: number, noun: string) =>
    `${count} ${noun}${count === 1 ? "" : "s"}`;

  if (brief.intent === "shift_handover") {
    const dueStatement = dueSoon
      ? `${plural(dueSoon, "vessel")} due in the requested watch window.`
      : "No vessel with a validated ETA is due in the requested watch window.";
    const attention = [
      lowSpeed ? `${lowSpeed} low-speed` : "",
      etaChanged ? `${etaChanged} ETA-change` : "",
      stalePosition ? `${stalePosition} stale-position` : "",
      missingEta ? `${missingEta} missing-ETA` : "",
    ].filter(Boolean);
    return attention.length
      ? `${dueStatement} Attention flags: ${attention.join(" · ")}.`
      : dueStatement;
  }
  if (brief.intent === "inbound_watchlist") {
    return brief.matched_count
      ? `Showing ${brief.displayed_count} of ${brief.matched_count} validated vessel-reported ETA${brief.matched_count === 1 ? "" : "s"}.`
      : "No current vessel-reported ETA passed the requested scope and freshness checks.";
  }
  if (brief.intent === "low_speed_exceptions") {
    return brief.matched_count
      ? `${plural(brief.matched_count, "vessel")} meet the requested low-speed exception criteria.`
      : "No vessel meets the requested low-speed exception criteria.";
  }
  if (brief.intent === "eta_revisions") {
    return brief.matched_count
      ? `${plural(brief.matched_count, "vessel")} crossed the requested reported-ETA change threshold.`
      : "No vessel crossed the requested reported-ETA change threshold.";
  }
  if (brief.intent === "signal_quality") {
    return `${plural(stalePosition, "stale-position signal")} · ${plural(missingEta, "signal without a valid ETA")}.`;
  }
  if (brief.intent === "destination_load") {
    return brief.matched_count
      ? `${plural(brief.matched_count, "validated vessel signal")} contribute to the destination ranking.`
      : "No validated vessel signal contributes to the requested destination ranking.";
  }
  const vessel = brief.prioritized_items[0]?.vessel_label;
  return vessel
    ? `${vessel} is the next matching vessel signal for this request.`
    : "No matching vessel signal passed the requested validation checks.";
}

function OperationalTime({
  value,
  unavailable = "Unavailable",
}: {
  value?: unknown;
  unavailable?: string;
}) {
  if (!hasDisplayValue(value)) return <>{unavailable}</>;
  const text = String(value);
  const parsed = new Date(text);
  if (Number.isNaN(parsed.getTime())) return <>{displayValue(value)}</>;
  return (
    <time dateTime={text} title={text}>
      {formatOperationalUtc(text)}
    </time>
  );
}

function etaRevisionLabel(value: unknown): string {
  const minutes = Number(value);
  if (!Number.isFinite(minutes) || minutes === 0) return "No change";
  const magnitude = Math.abs(minutes);
  const formatted = Number.isInteger(magnitude)
    ? String(magnitude)
    : magnitude.toFixed(1);
  return minutes > 0
    ? `+${formatted} min later`
    : `−${formatted} min earlier`;
}

function OperationalBrief({
  brief,
  datasets,
  onFollowUp,
  onRefresh,
  requestedCount,
}: {
  brief: OperationalBriefContract;
  datasets: Dataset[];
  onFollowUp: (question: string) => void;
  onRefresh: () => void;
  requestedCount?: number;
}) {
  const rowById = new Map<string, Record<string, unknown>>();
  datasets.forEach((dataset) => {
    dataset.rows.forEach((row) => {
      const rowId = row.row_id ?? row.id;
      if (typeof rowId === "string" && !rowById.has(rowId)) {
        rowById.set(rowId, row);
      }
    });
  });
  const activeExceptions = brief.exceptions.filter(
    (item) => item.count > 0 && item.code !== "due_soon",
  );
  const inboundDataset = datasets.find(
    (dataset) => dataset.id === "table" && dataset.rows.length,
  );
  const inboundRows =
    brief.intent === "inbound_watchlist"
      ? inboundDataset?.rows || []
      : [];
  const freshnessCandidateRows =
    brief.intent === "inbound_watchlist"
      ? datasets.find(
          (dataset) => dataset.id === "eta_freshness_candidates",
        )?.rows || []
      : [];
  const missingRequestedCount =
    requestedCount && requestedCount < 20
      ? Math.max(
          0,
          requestedCount - inboundRows.length - freshnessCandidateRows.length,
        )
      : 0;
  const dueSoonCount = operationalExceptionCount(brief, "due_soon");
  const attentionCategoryCount = activeExceptions.length;
  const visiblePriorityItems = brief.prioritized_items.slice(0, 3);
  const additionalPriorityItems = brief.prioritized_items.slice(3);

  const renderPriorityItem = (
    item: OperationalBriefContract["prioritized_items"][number],
  ) => {
    const row = rowById.get(item.row_id) || {};
    const vesselLabel =
      item.vessel_label ||
      String(
        row.vessel_label ||
          row.vessel_name ||
          row.name ||
          row.mmsi ||
          "Selected vessel",
      );
    const destinationName =
      row.destination_label ||
      row.port_label ||
      row.destination ||
      row.destination_name ||
      row.destination_raw;
    const destinationCode = row.destination_locode;
    const destination =
      hasDisplayValue(destinationName) &&
      hasDisplayValue(destinationCode) &&
      !String(destinationName).includes(String(destinationCode))
        ? `${displayValue(destinationName)} (${displayValue(destinationCode)})`
        : destinationName || destinationCode;
    const eta =
      item.status === "missing_eta"
        ? null
        : row.reported_eta_utc || row.eta_utc || row.eta;
    const previousEta = row.previous_reported_eta_utc;
    const etaChange = row.eta_change_minutes;
    const speed = row.speed_kn ?? row.speed_knots ?? row.sog_kn ?? row.sog;
    const observed =
      row.eta_change_observed_at_utc ||
      row.position_time_utc ||
      row.observed_at_utc ||
      row.position_observed_at_utc ||
      row.observation_time_utc;
    return (
      <li
        key={item.row_id}
        className={`priority-item priority-${item.priority}`}
        data-testid="eta-priority-vessel"
      >
        <div className="priority-item-copy">
          <div>
            <strong>{vesselLabel}</strong>
            <span>{displayKey(item.status)}</span>
          </div>
          <p>{item.reason}</p>
          <dl>
            {hasDisplayValue(destination) && (
              <div>
                <dt>Destination</dt>
                <dd>{displayValue(destination)}</dd>
              </div>
            )}
            <div>
              <dt>Reported ETA</dt>
              <dd>
                <OperationalTime
                  value={eta}
                  unavailable={
                    item.status === "missing_eta"
                      ? "No valid ETA"
                      : "Unavailable"
                  }
                />
              </dd>
            </div>
            {item.status === "eta_changed" &&
              hasDisplayValue(previousEta) && (
                <div>
                  <dt>Previous ETA</dt>
                  <dd>
                    <OperationalTime value={previousEta} />
                  </dd>
                </div>
              )}
            {item.status === "eta_changed" && hasDisplayValue(etaChange) && (
              <div>
                <dt>ETA revision</dt>
                <dd>{etaRevisionLabel(etaChange)}</dd>
              </div>
            )}
            {hasDisplayValue(speed) && (
              <div>
                <dt>Speed</dt>
                <dd>{displayValue(speed)} kn</dd>
              </div>
            )}
            {hasDisplayValue(observed) && (
              <div>
                <dt>Observed</dt>
                <dd>
                  <OperationalTime value={observed} />
                </dd>
              </div>
            )}
          </dl>
        </div>
        {item.actions.length > 0 && (
          <div className="priority-actions">
            {item.actions.map((action) => (
              <button
                type="button"
                key={action}
                onClick={() =>
                  onFollowUp(operationalFollowUp(action, vesselLabel))
                }
              >
                {operationalActionLabel(action)}
              </button>
            ))}
          </div>
        )}
      </li>
    );
  };

  return (
    <section
      className="operational-brief"
      aria-labelledby="operational-brief-title"
      data-testid="operational-brief"
    >
      <header
        className="operational-brief-heading"
        data-testid="eta-answer-summary"
      >
        <div>
          <p className="section-code">Operational finding</p>
          <h3 id="operational-brief-title">{operationalFinding(brief)}</h3>
          <p className="operational-finding-copy">{brief.headline}</p>
        </div>
        <span className={`source-health source-health-${brief.source_health}`}>
          Source {stateLabel(brief.source_health)}
        </span>
        {brief.intent === "inbound_watchlist" && (
          <button
            type="button"
            className="refresh-live-signals"
            onClick={onRefresh}
          >
            Refresh live signals
          </button>
        )}
      </header>

      <dl className="operational-brief-ledger" data-testid="eta-key-stats">
        <div>
          <dt>
            {brief.intent === "shift_handover"
              ? "Signals reviewed"
              : "Signals matched"}
          </dt>
          <dd>{brief.matched_count}</dd>
        </div>
        <div>
          <dt>Due soon</dt>
          <dd>{dueSoonCount}</dd>
        </div>
        <div>
          <dt>Needs attention categories</dt>
          <dd>{attentionCategoryCount}</dd>
        </div>
        <div>
          <dt>Last AIS update</dt>
          <dd>
            <OperationalTime
              value={brief.source_observed_at}
              unavailable="Awaiting a validated timestamp"
            />
          </dd>
        </div>
      </dl>
      <p className="operational-scope-line">
        {brief.displayed_count} operational row
        {brief.displayed_count === 1 ? "" : "s"} shown
        <span aria-hidden="true"> · </span>
        <span className="sr-only">. </span>
        {operationalWindowLabel(
          brief.window_start_utc,
          brief.window_end_utc,
        )}
      </p>

      {activeExceptions.length > 0 && (
        <section className="operational-exceptions" aria-label="Needs attention">
          <h4>Needs attention</h4>
          <ul>
            {activeExceptions.map((item) => (
              <li key={item.code}>
                <strong>{item.count}</strong>
                <span>{item.summary}</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {inboundRows.length > 0 && (
        <section
          className="validated-inbound-watchlist"
          aria-labelledby="validated-inbound-title"
          data-testid="validated-inbound-list"
        >
          <header>
            <h4 id="validated-inbound-title">Current reported ETAs</h4>
            <span>
              {inboundRows.length} current
              {requestedCount && requestedCount < 20
                ? ` of ${requestedCount} requested`
                : ""}
            </span>
          </header>
          <ol>
            {inboundRows.map((row, index) => {
              const rowId = String(
                row.row_id || row.id || row.mmsi || `inbound-${index + 1}`,
              );
              const vesselLabel = String(
                row.vessel_label ||
                  row.vessel_name ||
                  row.name ||
                  row.mmsi ||
                  "Unidentified vessel",
              );
              const destinationName =
                row.destination_label ||
                row.port_label ||
                row.destination ||
                row.destination_name ||
                row.destination_raw;
              const destinationCode = row.destination_locode;
              const destination =
                hasDisplayValue(destinationName) &&
                hasDisplayValue(destinationCode) &&
                !String(destinationName).includes(String(destinationCode))
                  ? `${displayValue(destinationName)} (${displayValue(destinationCode)})`
                  : destinationName || destinationCode;
              const eta = row.reported_eta_utc || row.eta_utc || row.eta;
              const speed =
                row.speed_kn ?? row.speed_knots ?? row.sog_kn ?? row.sog;
              const observed =
                row.observed_at_utc ||
                row.position_observed_at_utc ||
                row.observation_time_utc ||
                row.position_time_utc;
              const hasPosition =
                hasDisplayValue(row.latitude) &&
                hasDisplayValue(row.longitude);
              return (
                <li
                  key={rowId}
                  data-testid="validated-inbound-row"
                  data-row-id={rowId}
                >
                  <div className="validated-inbound-identity">
                    <span>{String(index + 1).padStart(2, "0")}</span>
                    <div>
                      <strong>{vesselLabel}</strong>
                      {hasDisplayValue(row.mmsi) && (
                        <small>MMSI {displayValue(row.mmsi)}</small>
                      )}
                    </div>
                  </div>
                  <dl>
                    {hasDisplayValue(destination) && (
                      <div>
                        <dt>Destination</dt>
                        <dd>{displayValue(destination)}</dd>
                      </div>
                    )}
                    {hasDisplayValue(eta) && (
                      <div>
                        <dt>Reported ETA</dt>
                        <dd>
                          <OperationalTime value={eta} />
                        </dd>
                      </div>
                    )}
                    {hasPosition && (
                      <div>
                        <dt>Last position</dt>
                        <dd>
                          {displayValue(row.latitude)},{" "}
                          {displayValue(row.longitude)}
                        </dd>
                      </div>
                    )}
                    {hasDisplayValue(speed) && (
                      <div>
                        <dt>Speed</dt>
                        <dd>{displayValue(speed)} kn</dd>
                      </div>
                    )}
                    {hasDisplayValue(observed) && (
                      <div>
                        <dt>Observed</dt>
                        <dd>
                          <OperationalTime value={observed} />
                        </dd>
                      </div>
                    )}
                  </dl>
                </li>
              );
            })}
          </ol>
        </section>
      )}

      {freshnessCandidateRows.length > 0 && (
        <section
          className="eta-freshness-candidates"
          aria-labelledby="eta-freshness-candidates-title"
          data-testid="eta-freshness-candidates"
        >
          <header>
            <div>
              <h4 id="eta-freshness-candidates-title">
                Awaiting a fresh ETA broadcast
              </h4>
              <p>
                These vessels match the requested destinations, but their last
                ETA transmission is too old for the current total or chart.
              </p>
            </div>
            <span>
              {freshnessCandidateRows.length} excluded signal
              {freshnessCandidateRows.length === 1 ? "" : "s"}
            </span>
          </header>
          <ol>
            {freshnessCandidateRows.map((row, index) => {
              const rowId = String(
                row.row_id ||
                  row.id ||
                  row.mmsi ||
                  `eta-freshness-${index + 1}`,
              );
              const vesselLabel = String(
                row.vessel_label ||
                  row.vessel_name ||
                  row.mmsi ||
                  "Unidentified vessel",
              );
              const destinationName =
                row.destination_name ||
                row.destination_raw ||
                row.destination_locode;
              const destinationCode = row.destination_locode;
              const destination =
                hasDisplayValue(destinationName) &&
                hasDisplayValue(destinationCode) &&
                !String(destinationName).includes(String(destinationCode))
                  ? `${displayValue(destinationName)} (${displayValue(destinationCode)})`
                  : destinationName || destinationCode;
              return (
                <li
                  key={rowId}
                  data-testid="eta-freshness-candidate-row"
                >
                  <div className="validated-inbound-identity">
                    <span>{String(index + 1).padStart(2, "0")}</span>
                    <div>
                      <strong>{vesselLabel}</strong>
                      {hasDisplayValue(row.mmsi) && (
                        <small>MMSI {displayValue(row.mmsi)}</small>
                      )}
                    </div>
                  </div>
                  <div className="eta-candidate-detail">
                    <dl>
                      {hasDisplayValue(destination) && (
                        <div>
                          <dt>Destination</dt>
                          <dd>{displayValue(destination)}</dd>
                        </div>
                      )}
                      {hasDisplayValue(row.last_reported_eta_utc) && (
                        <div>
                          <dt>Last ETA—not current</dt>
                          <dd>
                            <OperationalTime
                              value={row.last_reported_eta_utc}
                            />
                          </dd>
                        </div>
                      )}
                      {hasDisplayValue(
                        row.eta_observation_age_minutes,
                      ) && (
                        <div>
                          <dt>ETA report age</dt>
                          <dd>
                            {Math.round(
                              Number(row.eta_observation_age_minutes),
                            )}{" "}
                            min
                          </dd>
                        </div>
                      )}
                      {hasDisplayValue(row.last_eta_observed_at_utc) && (
                        <div>
                          <dt>Last ETA received</dt>
                          <dd>
                            <OperationalTime
                              value={row.last_eta_observed_at_utc}
                            />
                          </dd>
                        </div>
                      )}
                    </dl>
                    <p>{displayValue(row.validation_reason)}</p>
                  </div>
                </li>
              );
            })}
          </ol>
          {missingRequestedCount > 0 && (
            <p className="eta-candidate-missing">
              No additional matching AIS vessel signal is present for{" "}
              {missingRequestedCount} requested slot
              {missingRequestedCount === 1 ? "" : "s"} in the current source
              window.
            </p>
          )}
        </section>
      )}

      {brief.intent !== "inbound_watchlist" &&
        brief.intent !== "destination_load" &&
        brief.prioritized_items.length > 0 && (
        <section
          className="priority-watchlist"
          aria-label="Priority watchlist"
          data-testid="eta-priority-vessels"
        >
          <h4>Priority watchlist</h4>
          <ol>{visiblePriorityItems.map(renderPriorityItem)}</ol>
          {additionalPriorityItems.length > 0 && (
            <details className="additional-priority-items">
              <summary>
                Show {additionalPriorityItems.length} more priority vessel
                {additionalPriorityItems.length === 1 ? "" : "s"}
              </summary>
              <ol>{additionalPriorityItems.map(renderPriorityItem)}</ol>
            </details>
          )}
        </section>
      )}

      <p className="operational-coverage">{brief.coverage}</p>
    </section>
  );
}

function ResultMetadata({ result }: { result: AnswerEnvelope }) {
  const showResultState = !isSuccessfulResultState(result.state);
  return (
    <section
      className="result-metadata"
      aria-labelledby="result-metadata-title"
      data-testid="result-metadata"
    >
      <h2 id="result-metadata-title" className="sr-only">
        Result metadata
      </h2>
      <dl>
        {showResultState && (
          <div>
            <dt>Result state</dt>
            <dd>
              <span
                className={`state-marker state-${result.state.toLowerCase()}`}
                aria-hidden="true"
              />
              {stateLabel(result.state)}
            </dd>
          </div>
        )}
        <div>
          <dt>Source</dt>
          <dd>{sourceLabel(result)}</dd>
        </div>
        {result.availability &&
          result.availability.code !== "available" &&
          result.availability.code !== "not_applicable" && (
            <div>
              <dt>Availability</dt>
              <dd>{displayKey(result.availability.code)}</dd>
            </div>
          )}
        <div>
          <dt>Freshness</dt>
          <dd>{result.freshness.message}</dd>
        </div>
      </dl>
    </section>
  );
}

function EvidenceInspector({ result }: { result: AnswerEnvelope }) {
  return (
    <aside
      className="evidence-inspector"
      aria-labelledby="evidence-title"
      data-testid="evidence-inspector"
    >
      <header className="inspector-heading">
        <div>
          <p className="section-code">Lineage</p>
          <h2 id="evidence-title">Evidence</h2>
        </div>
        <span>{result.evidence.length} records</span>
      </header>

      <ol className="evidence-records">
        {result.evidence.map((item, index) => (
          <li key={item.id}>
            <span className="evidence-index">
              {String(index + 1).padStart(2, "0")}
            </span>
            <div>
              <small>{displayKey(item.source_type)}</small>
              {item.url ? (
                <a
                  href={item.url}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  {item.title}
                </a>
              ) : (
                <strong>{item.title}</strong>
              )}
              {item.excerpt && <p>{item.excerpt}</p>}
              {Object.keys(item.metadata).length > 0 && (
                <details>
                  <summary>Source metadata</summary>
                  <dl>
                    {Object.entries(item.metadata).map(([key, value]) => (
                      <div key={key}>
                        <dt>{displayKey(key)}</dt>
                        <dd>{displayValue(value)}</dd>
                      </div>
                    ))}
                  </dl>
                </details>
              )}
            </div>
          </li>
        ))}
      </ol>

    </aside>
  );
}

function ResultDetails({
  result,
  onReport,
}: {
  result: AnswerEnvelope;
  onReport: () => void;
}) {
  const [exportStatus, setExportStatus] = useState<
    Record<string, string>
  >({});

  async function runExport(datasetId: string, format: "csv" | "json") {
    const key = `${datasetId}:${format}`;
    setExportStatus((current) => ({
      ...current,
      [key]: "Creating export…",
    }));
    try {
      const response: ExportResponse = await exportDataset({
        conversation_id: result.conversation_id,
        turn_id: result.turn_id,
        dataset_id: datasetId,
        format,
      });
      setExportStatus((current) => ({
        ...current,
        [key]: `${response.row_count} rows exported to ${response.path}`,
      }));
    } catch (error) {
      setExportStatus((current) => ({
        ...current,
        [key]:
          error instanceof ApiError
            ? error.message
            : "The export could not be created.",
      }));
    }
  }

  const scopeEntries = Object.entries(result.applied_scope).filter(([, value]) =>
    hasDisplayValue(value),
  );
  return (
    <section className="result-details" aria-label="Analysis record details">
      <details>
        <summary>Data &amp; scope</summary>
        <div className="details-body">
          <section>
            <h3>Applied scope</h3>
            <dl className="definition-ledger">
              {scopeEntries.map(([key, value]) => (
                <div key={key}>
                  <dt>{displayKey(key)}</dt>
                  <dd>{displayValue(value)}</dd>
                </div>
              ))}
              {!scopeEntries.length && (
                <div>
                  <dt>Scope</dt>
                  <dd>No explicit scope fields were applied.</dd>
                </div>
              )}
            </dl>
          </section>

          {result.facts.length > 0 && (
            <section>
              <h3>Immutable facts</h3>
              <div className="compact-table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Fact</th>
                      <th>Value</th>
                      <th>Source</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.facts.map((fact, index) => (
                      <tr key={`${fact.name}-${index}`}>
                        <td>{displayKey(fact.name)}</td>
                        <td>
                          {displayValue(fact.value)}
                          {fact.unit ? ` ${fact.unit}` : ""}
                        </td>
                        <td>{displayKey(fact.source)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          {result.datasets.map((dataset) => (
            <section key={dataset.id} className="dataset-record">
              <header>
                <div>
                  <h3>{displayKey(dataset.id)}</h3>
                  <span>{dataset.row_count} rows</span>
                </div>
                <div className="dataset-actions">
                  <button
                    type="button"
                    onClick={() => runExport(dataset.id, "csv")}
                  >
                    Export CSV
                  </button>
                  <button
                    type="button"
                    onClick={() => runExport(dataset.id, "json")}
                  >
                    Export JSON
                  </button>
                </div>
              </header>
              <DatasetTable dataset={dataset} />
              {(["csv", "json"] as const).map((format) => {
                const status = exportStatus[`${dataset.id}:${format}`];
                return status ? (
                  <p
                    className="export-status"
                    role="status"
                    key={format}
                  >
                    {format.toUpperCase()}: {status}
                  </p>
                ) : null;
              })}
            </section>
          ))}
        </div>
      </details>

      <div className="result-record-actions">
        <button className="text-action" type="button" onClick={onReport}>
          Report a possible issue with this analysis
        </button>
      </div>
    </section>
  );
}

function resultPresentationTitle(result: AnswerEnvelope): string {
  if (result.state === "CLARIFICATION_REQUIRED") return "Clarification needed";
  if (result.state === "UNSUPPORTED") return "Unsupported request";
  if (
    result.state === "NO_DATA" ||
    result.state === "NO_CURRENT_DATA" ||
    result.state === "ASSURANCE_UNAVAILABLE"
  ) {
    return "Availability result";
  }
  if (result.state === "ERROR") return "Result unavailable";
  return "Analysis result";
}

function splitAnswerForPresentation(answer: string): string[] {
  const blocks = answer
    .trim()
    .split(/\n+/)
    .map((block) => block.trim())
    .filter(Boolean);

  return blocks.flatMap((block) => {
    if (block.length <= 220) return [block];
    const sentences = block
      .split(/(?<=[.!?])\s+(?=[A-Z0-9"'([])/)
      .map((sentence) => sentence.trim())
      .filter(Boolean);
    if (sentences.length > 1) return sentences;

    const semicolonParts = block
      .split(/;\s+/)
      .map((part) => part.trim())
      .filter(Boolean);
    if (semicolonParts.length < 3) return [block];
    return semicolonParts.map((part, index) =>
      index < semicolonParts.length - 1 && !part.endsWith(";")
        ? `${part};`
        : part,
    );
  });
}

function ReadableAnswer({ result }: { result: AnswerEnvelope }) {
  const segments = splitAnswerForPresentation(result.answer);
  const lead = segments[0] || "No answer text was returned.";
  const canCollapse =
    result.state === "COMPUTED" ||
    result.state === "PARTIAL" ||
    result.state === "RETRIEVED" ||
    result.state === "GENERAL";
  const supporting = canCollapse ? segments.slice(1, 4) : segments.slice(1);
  const remaining = canCollapse ? segments.slice(4) : [];

  return (
    <section
      className={`readable-answer readable-answer-${result.state.toLowerCase()}`}
      aria-labelledby="readable-answer-title"
      data-testid="global-answer-detail"
    >
      <header className="readable-answer-heading">
        <div>
          <p className="section-code">Answer</p>
          <h3 id="readable-answer-title">{lead}</h3>
        </div>
        {!isSuccessfulResultState(result.state) && (
          <span className="readable-answer-state">{stateLabel(result.state)}</span>
        )}
      </header>

      {supporting.length > 0 && (
        <ul className="readable-answer-points">
          {supporting.map((segment, index) => (
            <li key={`${index}-${segment}`}>{segment}</li>
          ))}
        </ul>
      )}

      {remaining.length > 0 && (
        <details className="readable-answer-more">
          <summary>
            Show {remaining.length} more answer point
            {remaining.length === 1 ? "" : "s"}
          </summary>
          <ul>
            {remaining.map((segment, index) => (
              <li key={`${index}-${segment}`}>{segment}</li>
            ))}
          </ul>
        </details>
      )}
    </section>
  );
}

function ResultWorkspace({
  result,
  onReport,
  onRerun,
  onFollowUp,
}: {
  result: AnswerEnvelope;
  onReport: () => void;
  onRerun: () => void;
  onFollowUp: (question: string) => void;
}) {
  const primaryVisualizationIndex = result.visualizations.findIndex(
    (visualization) =>
      visualization.kind !== "omitted" && visualization.kind !== "table",
  );
  const hasEvidence = result.evidence.length > 0;
  const legacyFintraffic = isLegacyFintrafficResult(result);

  return (
    <article className="result-workspace" aria-labelledby="result-question">
      {legacyFintraffic && (
        <section className="source-update-notice" role="status">
          <div>
            <strong>ETA Watch source update required</strong>
            <p>
              This saved record uses a retired live-data source. Refresh it
              before using it for current operations.
            </p>
          </div>
          <button type="button" onClick={onRerun}>
            Refresh ETA Watch
          </button>
        </section>
      )}
      <div
        className={`result-grid ${
          hasEvidence
            ? "result-grid-with-evidence"
            : "result-grid-without-evidence"
        }`}
      >
        <div className="result-primary">
          <section
            className="answer-record answer-record-operational"
            data-testid="global-answer-summary"
          >
            <p className="section-code">Result</p>
            <h2 id="result-question">{resultPresentationTitle(result)}</h2>
            <details className="submitted-request">
              <summary>
                <span>Submitted request</span>
                <span className="submitted-request-preview">
                  {result.question}
                </span>
              </summary>
              <p>{result.question}</p>
            </details>
          </section>

          {result.operational_brief ? (
            <div
              className="eta-answer-detail"
              data-testid="global-answer-detail"
            >
              <OperationalBrief
                brief={result.operational_brief}
                datasets={result.datasets}
                onFollowUp={onFollowUp}
                onRefresh={onRerun}
                requestedCount={result.plan.limit}
              />
            </div>
          ) : (
            <ReadableAnswer result={result} />
          )}

          <details
            className="canonical-response-disclosure"
            data-testid="canonical-response-disclosure"
          >
            <summary>Full response</summary>
            <p className="canonical-answer" data-testid="result-answer">
              {result.answer}
            </p>
          </details>

          <section
            className="visualization-stack"
            aria-label="Visualizations"
            data-testid="result-visualizations"
          >
            {result.visualizations.length ? (
              result.visualizations.map((visualization, index) => (
                <Visualization
                  key={visualization.id || `${visualization.kind}-${index}`}
                  visualization={{
                    ...visualization,
                    id:
                      visualization.id ||
                      `${visualization.kind}-${index}`,
                  }}
                  datasets={result.datasets}
                  primary={
                    index ===
                    (primaryVisualizationIndex >= 0
                      ? primaryVisualizationIndex
                      : 0)
                  }
                  contractVersion={
                    result.visualization_contract_version || "2.0"
                  }
                />
              ))
            ) : (
              <section className="visualization">
                <div className="empty-visual" role="status">
                  <strong>No visualization contract was returned.</strong>
                  <span>
                    The answer remains available, but no chart can be displayed.
                  </span>
                </div>
              </section>
            )}
          </section>

          <ChartInsights insights={result.chart_insights} />

          <ResultMetadata result={result} />
        </div>
        {hasEvidence && <EvidenceInspector result={result} />}
      </div>
      <ResultDetails result={result} onReport={onReport} />
    </article>
  );
}

function AboutAnalysis({
  category,
}: {
  category?: SampleQueryCategory;
}) {
  if (!category) {
    return (
      <details className="about-analysis">
        <summary>About this analysis</summary>
        <div className="about-grid">
          <section>
            <h3>Scope</h3>
            <p>
              Analysis Desk routes each request through the canonical query
              service. Structured datasets remain the sole numeric authority
              for analytics.
            </p>
          </section>
          <section>
            <h3>Evidence</h3>
            <p>
              Research checks local documents first. Retrieved evidence is
              shown separately from computed facts and historical datasets.
            </p>
          </section>
        </div>
      </details>
    );
  }

  const help: QueryCategoryHelp | undefined = QUERY_CATEGORY_HELP[category];
  if (!help) return null;
  const groups: Array<[string, string[]]> = [
    ["What to enter", help.what_to_enter],
    ["Expected output", help.expected_output],
    ["How to review", help.test_steps],
    ["Method and data", help.calculation],
  ];
  return (
    <details className="about-analysis">
      <summary>About this analysis</summary>
      <div className="about-intro">{help.overview}</div>
      <div className="about-grid">
        {groups.map(([title, values]) => (
          <section key={title}>
            <h3>{title}</h3>
            <ul>
              {values.map((value) => (
                <li key={value}>{value}</li>
              ))}
            </ul>
          </section>
        ))}
      </div>
    </details>
  );
}

function QueryPage({
  page,
  question,
  filters,
  result,
  error,
  isWorking,
  onQuestion,
  onRun,
  onOpenSamples,
  onOpenFilters,
  onReport,
}: {
  page: PageDefinition;
  question: string;
  filters: UiFilters;
  result?: AnswerEnvelope;
  error?: string;
  isWorking: boolean;
  onQuestion: (value: string) => void;
  onRun: () => void;
  onOpenSamples: () => void;
  onOpenFilters: () => void;
  onReport: () => void;
}) {
  const promptRef = useRef<HTMLTextAreaElement>(null);
  const samples = page.category
    ? SAMPLE_QUERIES_BY_CATEGORY[page.category]
    : [];
  const [selectedSample, setSelectedSample] = useState("");

  useEffect(() => {
    const focusShortcut = (event: KeyboardEvent) => {
      if (
        event.key === "/" &&
        !event.metaKey &&
        !event.ctrlKey &&
        !event.altKey &&
        !["INPUT", "TEXTAREA", "SELECT"].includes(
          (event.target as HTMLElement)?.tagName,
        )
      ) {
        event.preventDefault();
        promptRef.current?.focus();
      }
    };
    document.addEventListener("keydown", focusShortcut);
    return () => document.removeEventListener("keydown", focusShortcut);
  }, []);

  function submit(event: FormEvent) {
    event.preventDefault();
    onRun();
  }

  function keyboardSubmit(event: ReactKeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      onRun();
    }
  }

  return (
    <div className={`page-stack query-page page-${page.key}`}>
      <PageHeading page={page} />
      <section
        className="query-workbench"
        aria-labelledby="request-title"
      >
        <header className="workbench-heading">
          <div>
            <p className="section-code">Analysis request</p>
            <h2 id="request-title">Ask a question</h2>
          </div>
          <span>
            {result
              ? "Enter another question or rerun this request"
              : "⌘/Ctrl + Enter to analyze"}
          </span>
        </header>

        {page.category && (
          <div className="sample-selector">
            <label htmlFor={`sample-${page.key}`}>Sample query</label>
            <select
              id={`sample-${page.key}`}
              value={selectedSample}
              onChange={(event) => setSelectedSample(event.target.value)}
            >
              <option value="">Select one of {samples.length} samples</option>
              {page.key === "eta"
                ? ETA_WATCH_SAMPLE_GROUPS.map((group) => (
                    <optgroup key={group.label} label={group.label}>
                      {group.prompts.map((sample) => (
                        <option key={sample} value={sample}>
                          {sample}
                        </option>
                      ))}
                    </optgroup>
                  ))
                : samples.map((sample) => (
                    <option key={sample} value={sample}>
                      {sample}
                    </option>
                  ))}
            </select>
            <button
              type="button"
              onClick={() => {
                if (selectedSample) onQuestion(selectedSample);
              }}
              disabled={!selectedSample}
            >
              Load sample
            </button>
          </div>
        )}

        <form onSubmit={submit}>
          <label htmlFor={`query-${page.key}`}>Question</label>
          <textarea
            id={`query-${page.key}`}
            ref={promptRef}
            value={question}
            onChange={(event) => onQuestion(event.target.value)}
            onKeyDown={keyboardSubmit}
            placeholder={
              page.key === "analysis"
                ? "Enter a maritime analysis, research, or application question"
                : `Enter a focused ${page.label.toLowerCase()} question`
            }
            rows={3}
            maxLength={4000}
            disabled={isWorking}
            data-testid="query-input"
          />
          <div className="query-actions">
            <div>
              <button
                type="button"
                className="secondary-action"
                onClick={onOpenSamples}
                data-testid="sample-library-button"
              >
                Sample library
              </button>
              <button
                type="button"
                className="secondary-action"
                onClick={onOpenFilters}
                data-testid="filter-button"
              >
                Filters
                {activeFilterCount(filters) > 0 &&
                  ` (${activeFilterCount(filters)})`}
              </button>
            </div>
            <button
              type="submit"
              className="primary-action"
              disabled={isWorking || !question.trim()}
              data-testid="analyze-button"
            >
              {isWorking ? "Analyzing…" : "Analyze"}
            </button>
          </div>
        </form>
      </section>

      <section
        className="query-status"
        aria-live="polite"
        aria-busy={isWorking}
      >
        {isWorking && (
          <div className="processing-record" role="status">
            <span className="processing-indicator" aria-hidden="true" />
            <div>
              <strong>Analysis in progress</strong>
              <p>
                Applying the canonical planner, data authority, and
                visualization validator.
              </p>
            </div>
          </div>
        )}
        {!isWorking && error && (
          <div className="error-record" role="alert">
            <strong>Analysis unavailable</strong>
            <p>{error}</p>
          </div>
        )}
      </section>

      {!isWorking && result && (
        <ResultWorkspace
          result={result}
          onReport={onReport}
          onRerun={onRun}
          onFollowUp={(followUp) => {
            onQuestion(followUp);
            requestAnimationFrame(() => {
              promptRef.current?.focus();
              promptRef.current?.scrollIntoView({
                behavior: "smooth",
                block: "center",
              });
            });
          }}
        />
      )}

      {!isWorking && !result && !error && (
        <section className="empty-workspace">
          <div>
            <span className="empty-index">01</span>
            <strong>Choose a focused scope</strong>
            <p>
              Use a sample or enter a focused port, vessel, route, date, or
              research question.
            </p>
          </div>
          <div>
            <span className="empty-index">02</span>
            <strong>Review the canonical answer</strong>
            <p>
              The result preserves its answer, facts, evidence, freshness, and
              data scope.
            </p>
          </div>
          <div>
            <span className="empty-index">03</span>
            <strong>Inspect every chart</strong>
            <p>
              Unsuitable or unavailable data receives an explicit reason,
              never an empty graph.
            </p>
          </div>
        </section>
      )}

      <AboutAnalysis category={page.category} />
    </div>
  );
}

function SampleLibrary({
  currentCategory,
  onChoose,
  onClose,
}: {
  currentCategory?: SampleQueryCategory;
  onChoose: (sample: string) => void;
  onClose: () => void;
}) {
  const [search, setSearch] = useState("");
  const categories = [
    ...QUERY_CATEGORY_PAGE_ORDER,
    "Unsupported Scope" as SampleQueryCategory,
  ];
  const filtered = categories
    .map((category) => ({
      category,
      prompts: SAMPLE_QUERIES_BY_CATEGORY[category].filter((prompt) =>
        prompt.toLowerCase().includes(search.trim().toLowerCase()),
      ),
    }))
    .filter(({ prompts }) => prompts.length > 0);

  return (
    <Modal title="Sample query library" onClose={onClose} size="wide">
      <div className="modal-content sample-library">
        <label htmlFor="sample-search">Search all 57 prompts</label>
        <input
          id="sample-search"
          type="search"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          placeholder="Search by port, vessel, metric, or date"
          autoComplete="off"
        />
        <p className="modal-note">
          Selecting a sample loads it into the current workspace. It is not
          submitted automatically.
        </p>
        <div className="sample-groups">
          {filtered.map(({ category, prompts }) => (
            <section
              key={category}
              className={
                category === "Unsupported Scope"
                  ? "sample-group sample-group-limits"
                  : "sample-group"
              }
            >
              <header>
                <h3>{sampleCategoryLabel(category)}</h3>
                <span>
                  {prompts.length}
                  {category === currentCategory ? " · current page" : ""}
                </span>
              </header>
              <ol>
                {prompts.map((prompt) => (
                  <li key={prompt}>
                    <button
                      type="button"
                      onClick={() => {
                        onChoose(prompt);
                        onClose();
                      }}
                    >
                      {prompt}
                    </button>
                  </li>
                ))}
              </ol>
            </section>
          ))}
          {!filtered.length && (
            <p className="section-empty">No sample queries match the search.</p>
          )}
        </div>
      </div>
    </Modal>
  );
}

function FilterSettings({
  filters,
  onFilters,
  onClose,
}: {
  filters: UiFilters;
  onFilters: (filters: UiFilters) => void;
  onClose: () => void;
}) {
  return (
    <Modal title="Analysis filters" onClose={onClose}>
      <div className="modal-content form-grid">
        <label>
          Port
          <input
            value={filters.port}
            onChange={(event) =>
              onFilters({ ...filters, port: event.target.value })
            }
            placeholder="LOCODE or port name"
          />
        </label>
        <label>
          Vessel type
          <input
            value={filters.vesselType}
            onChange={(event) =>
              onFilters({ ...filters, vesselType: event.target.value })
            }
            placeholder="For example, tanker"
          />
        </label>
        <label>
          Vessel name
          <input
            value={filters.vesselName}
            onChange={(event) =>
              onFilters({ ...filters, vesselName: event.target.value })
            }
            placeholder="Exact vessel name"
          />
        </label>
        <label>
          MMSI
          <input
            value={filters.mmsi}
            inputMode="numeric"
            onChange={(event) =>
              onFilters({ ...filters, mmsi: event.target.value })
            }
            placeholder="Nine digits"
          />
        </label>
        <label>
          IMO
          <input
            value={filters.imo}
            inputMode="numeric"
            onChange={(event) =>
              onFilters({ ...filters, imo: event.target.value })
            }
            placeholder="Seven digits"
          />
        </label>
        <label className="check-row">
          <input
            type="checkbox"
            checked={filters.anomaly}
            onChange={(event) =>
              onFilters({ ...filters, anomaly: event.target.checked })
            }
          />
          Include anomaly filter
        </label>
        <label className="check-row">
          <input
            type="checkbox"
            checked={filters.useDateRange}
            onChange={(event) =>
              onFilters({
                ...filters,
                useDateRange: event.target.checked,
              })
            }
          />
          Apply an explicit date range
        </label>
        <div className="date-fields">
          <label>
            From
            <input
              type="date"
              value={filters.dateFrom}
              disabled={!filters.useDateRange}
              onChange={(event) =>
                onFilters({ ...filters, dateFrom: event.target.value })
              }
            />
          </label>
          <label>
            To
            <input
              type="date"
              value={filters.dateTo}
              disabled={!filters.useDateRange}
              onChange={(event) =>
                onFilters({ ...filters, dateTo: event.target.value })
              }
            />
          </label>
        </div>
        <div className="modal-actions">
          <button
            type="button"
            className="secondary-action"
            onClick={() => onFilters({ ...DEFAULT_FILTERS })}
          >
            Clear filters
          </button>
          <button type="button" className="primary-action" onClick={onClose}>
            Apply
          </button>
        </div>
      </div>
    </Modal>
  );
}

function WorkspaceSettings({
  settings,
  onSettings,
  onClose,
}: {
  settings: UiSettings;
  onSettings: (settings: UiSettings) => void;
  onClose: () => void;
}) {
  return (
    <Modal title="Workspace settings" onClose={onClose}>
      <div className="modal-content form-grid">
        <label>
          Evidence top K: {settings.topK}
          <input
            type="range"
            min={0}
            max={8}
            step={1}
            value={settings.topK}
            onChange={(event) =>
              onSettings({
                ...settings,
                topK: Number(event.target.value),
              })
            }
          />
          <small>
            Five supporting rows are retrieved by default. Zero disables
            evidence retrieval.
          </small>
        </label>
        <div className="modal-actions">
          <button type="button" className="primary-action" onClick={onClose}>
            Done
          </button>
        </div>
      </div>
    </Modal>
  );
}

function IssueReport({
  result,
  onClose,
}: {
  result: AnswerEnvelope;
  onClose: () => void;
}) {
  const [detail, setDetail] = useState("");
  const [status, setStatus] = useState("");
  const [sending, setSending] = useState(false);

  async function send(event: FormEvent) {
    event.preventDefault();
    if (!detail.trim() || sending) return;
    setSending(true);
    setStatus("Sending issue report…");
    try {
      await reportIssue(
        result.trace.trace_id,
        result.question,
        detail.trim(),
      );
      setDetail("");
      setStatus("The issue was recorded for review.");
    } catch (error) {
      setStatus(
        error instanceof ApiError
          ? error.message
          : "The report could not be sent. Keep the trace ID and try again.",
      );
    } finally {
      setSending(false);
    }
  }

  return (
    <Modal title="Report a possible issue" onClose={onClose}>
      <form className="modal-content form-grid" onSubmit={send}>
        <p className="modal-note">
          Describe what was wrong or missing. The trace ID is attached
          automatically.
        </p>
        <label>
          Issue detail
          <textarea
            rows={6}
            value={detail}
            onChange={(event) => setDetail(event.target.value)}
            required
          />
        </label>
        {status && <p role="status">{status}</p>}
        <div className="modal-actions">
          <button
            type="button"
            className="secondary-action"
            onClick={onClose}
          >
            Cancel
          </button>
          <button
            type="submit"
            className="primary-action"
            disabled={sending || !detail.trim()}
          >
            Send report
          </button>
        </div>
      </form>
    </Modal>
  );
}

export function App() {
  const location = useLocation();
  const navigate = useNavigate();
  const currentPage = routeForPath(location.pathname);
  const [capabilities, setCapabilities] =
    useState<CapabilityResponse | null>(null);
  const [capabilityState, setCapabilityState] =
    useState<OverviewCapabilityState>("loading");
  const [history, setHistory] = useState<HistoryRecord[]>(readHistory);
  const [conversations, setConversations] = useState<
    Partial<Record<PageKey, string>>
  >(readConversations);
  const [settings, setSettings] = useState<UiSettings>(readSettings);
  const [drafts, setDrafts] = useState<Partial<Record<PageKey, string>>>({});
  const [filters, setFilters] = useState<
    Partial<Record<PageKey, UiFilters>>
  >({});
  const [results, setResults] = useState<
    Partial<Record<PageKey, AnswerEnvelope>>
  >({});
  const [errors, setErrors] = useState<Partial<Record<PageKey, string>>>({});
  const [workingPage, setWorkingPage] = useState<PageKey | null>(null);
  const [modal, setModal] = useState<ModalName>(null);
  const [navOpen, setNavOpen] = useState(false);
  const historyRefreshes = useRef<Set<string>>(new Set());

  useDarkTheme();

  useEffect(() => {
    let active = true;
    void loadCapabilities().then((value) => {
      if (!active) return;
      setCapabilities(value);
      setCapabilityState(value ? "ready" : "unavailable");
    });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, 24)));
  }, [history]);

  useEffect(() => {
    sessionStorage.setItem(CONVERSATIONS_KEY, JSON.stringify(conversations));
  }, [conversations]);

  useEffect(() => {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  }, [settings]);

  useEffect(() => {
    document.title = `${currentPage.label} | Eagle Eye`;
    setNavOpen(false);
  }, [currentPage.label, location.pathname]);

  useEffect(() => {
    if (!navOpen) return;
    const previous = document.activeElement as HTMLElement | null;
    const rail = document.querySelector<HTMLElement>(".nav-rail");
    const workspace = document.querySelector<HTMLElement>(".workspace-frame");
    const mobile = window.matchMedia("(max-width: 899px)").matches;
    if (!mobile) return;
    if (workspace) workspace.inert = true;
    const close = rail?.querySelector<HTMLElement>(".nav-close");
    requestAnimationFrame(() => close?.focus());
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setNavOpen(false);
      if (event.key !== "Tab" || !rail) return;
      const items = Array.from(
        rail.querySelectorAll<HTMLElement>(
          'button:not([disabled]), a[href], [tabindex]:not([tabindex="-1"])',
        ),
      );
      if (!items.length) return;
      const first = items[0];
      const last = items[items.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", handleKey);
    return () => {
      document.removeEventListener("keydown", handleKey);
      if (workspace) workspace.inert = false;
      previous?.focus();
    };
  }, [navOpen]);

  const currentFilters =
    filters[currentPage.key] || ({ ...DEFAULT_FILTERS } as UiFilters);
  const currentQuestion = drafts[currentPage.key] || "";
  const currentResult = results[currentPage.key];

  function updateSettings(next: UiSettings) {
    setSettings({
      ...next,
      topK: Math.max(0, Math.min(8, Math.round(next.topK))),
    });
  }

  function updateCurrentFilters(next: UiFilters) {
    setFilters((existing) => ({ ...existing, [currentPage.key]: next }));
  }

  function chooseSample(sample: string) {
    setDrafts((existing) => ({
      ...existing,
      [currentPage.key]: sample,
    }));
  }

  async function executeQuery(
    page: PageDefinition,
    request: QueryRequestPayload,
    replacement?: HistoryRecord,
  ) {
    setWorkingPage(page.key);
    setErrors((existing) => ({ ...existing, [page.key]: undefined }));
    try {
      const result = await submitQuery(request);
      if (replacement && result.state === "ASSURANCE_UNAVAILABLE") {
        throw new ApiError(
          "The saved analysis could not be refreshed. Please try again.",
        );
      }
      setResults((existing) => ({
        ...existing,
        [page.key]: result,
      }));
      setConversations((existing) => ({
        ...existing,
        [page.key]: result.conversation_id,
      }));
      const record: HistoryRecord = {
        id:
          replacement?.id ||
          result.trace.trace_id ||
          crypto.randomUUID(),
        question: request.question,
        createdAt: replacement?.createdAt || new Date().toISOString(),
        route: replacement?.route || page.path,
        conversationId:
          replacement?.conversationId || result.conversation_id,
        result,
        request: {
          ...request,
          filters: request.filters ? { ...request.filters } : undefined,
        },
        schemaVersion: 3,
      };
      setHistory((existing) => {
        if (replacement) {
          return existing
            .map((item) => (item.id === replacement.id ? record : item))
            .slice(0, 24);
        }
        return [
          record,
          ...existing.filter(
            (item) =>
              !(
                item.result.trace.result_hash === result.trace.result_hash &&
                item.route === page.path
              ),
          ),
        ].slice(0, 24);
      });
    } catch (error) {
      if (replacement) {
        setResults((existing) => ({ ...existing, [page.key]: undefined }));
      }
      setErrors((existing) => ({
        ...existing,
        [page.key]:
          error instanceof ApiError
            ? error.message
            : "An unexpected service error occurred. No result was generated.",
      }));
    } finally {
      setWorkingPage(null);
    }
  }

  async function runCurrentQuery() {
    if (currentPage.key === "overview" || workingPage) return;
    const question = (drafts[currentPage.key] || "").trim();
    if (!question) return;
    const conversationId =
      conversations[currentPage.key] || newConversationId(currentPage.key);
    setConversations((existing) => ({
      ...existing,
      [currentPage.key]: conversationId,
    }));

    const request: QueryRequestPayload = {
      question,
      conversation_id: conversationId,
      top_k_evidence: settings.topK,
      filters: apiFilters(currentFilters),
    };
    await executeQuery(currentPage, request);
  }

  function startNewAnalysis() {
    const conversationId = newConversationId("analysis");
    setConversations((existing) => ({
      ...existing,
      analysis: conversationId,
    }));
    setDrafts((existing) => ({ ...existing, analysis: "" }));
    setResults((existing) => ({ ...existing, analysis: undefined }));
    setErrors((existing) => ({ ...existing, analysis: undefined }));
    navigate("/analysis");
    setNavOpen(false);
  }

  function selectHistory(record: HistoryRecord) {
    const page = routeForPath(record.route);
    const request = requestForHistoryRecord(record, settings.topK);
    setDrafts((existing) => ({
      ...existing,
      [page.key]: request.question,
    }));
    setFilters((existing) => ({
      ...existing,
      [page.key]: uiFiltersFromRequest(request),
    }));
    setErrors((existing) => ({ ...existing, [page.key]: undefined }));
    setConversations((existing) => ({
      ...existing,
      [page.key]:
        request.conversation_id ||
        record.conversationId ||
        record.result.conversation_id,
    }));
    navigate(page.path);
    setNavOpen(false);

    if (record.result.state !== "ASSURANCE_UNAVAILABLE") {
      setResults((existing) => ({
        ...existing,
        [page.key]: record.result,
      }));
      return;
    }

    setResults((existing) => ({ ...existing, [page.key]: undefined }));
    if (historyRefreshes.current.has(record.id)) return;
    historyRefreshes.current.add(record.id);
    void executeQuery(page, request, record).finally(() => {
      historyRefreshes.current.delete(record.id);
    });
  }

  function renderPage(page: PageDefinition) {
    if (page.key === "overview") {
      return (
        <OverviewPage
          capabilities={capabilities}
          capabilityState={capabilityState}
          history={history}
          onEnterAnalysis={startNewAnalysis}
          onSelectHistory={selectHistory}
          onOpenProvenance={() => setModal("provenance")}
        />
      );
    }
    return (
      <QueryPage
        key={page.key}
        page={page}
        question={drafts[page.key] || ""}
        filters={filters[page.key] || { ...DEFAULT_FILTERS }}
        result={results[page.key]}
        error={errors[page.key]}
        isWorking={workingPage === page.key}
        onQuestion={(value) =>
          setDrafts((existing) => ({ ...existing, [page.key]: value }))
        }
        onRun={runCurrentQuery}
        onOpenSamples={() => setModal("samples")}
        onOpenFilters={() => setModal("filters")}
        onReport={() => setModal("report")}
      />
    );
  }

  const overviewRailForcedCompact = currentPage.key === "overview";
  const effectiveRailCollapsed =
    overviewRailForcedCompact || settings.railCollapsed;

  return (
    <>
      <div
        className={`app-frame ${
          effectiveRailCollapsed ? "app-frame-collapsed" : ""
        } ${
          overviewRailForcedCompact ? "app-frame-overview" : ""
        }`}
        data-testid="app-shell"
      >
        <Navigation
          currentPage={currentPage}
          history={history}
          open={navOpen}
          collapsed={effectiveRailCollapsed}
          allowCollapse={!overviewRailForcedCompact}
          onClose={() => setNavOpen(false)}
          onToggleCollapse={() =>
            updateSettings({
              ...settings,
              railCollapsed: !settings.railCollapsed,
            })
          }
          onNewAnalysis={startNewAnalysis}
          onSelectHistory={selectHistory}
          onSettings={() => setModal("settings")}
        />
        <div className="workspace-frame">
          <UtilityBar
            page={currentPage}
            onOpenNavigation={() => setNavOpen(true)}
          />
          <main id="main-content" className="main-workspace" tabIndex={-1}>
            <Routes>
              <Route path="/" element={<Navigate to="/overview" replace />} />
              {PAGE_DEFINITIONS.map((page) => (
                <Route
                  key={page.key}
                  path={page.path}
                  element={renderPage(page)}
                />
              ))}
              <Route path="*" element={<Navigate to="/overview" replace />} />
            </Routes>
          </main>
        </div>
      </div>

      {modal === "samples" && (
        <SampleLibrary
          currentCategory={currentPage.category}
          onChoose={chooseSample}
          onClose={() => setModal(null)}
        />
      )}
      {modal === "filters" && currentPage.key !== "overview" && (
        <FilterSettings
          filters={currentFilters}
          onFilters={updateCurrentFilters}
          onClose={() => setModal(null)}
        />
      )}
      {modal === "settings" && (
        <WorkspaceSettings
          settings={settings}
          onSettings={updateSettings}
          onClose={() => setModal(null)}
        />
      )}
      {modal === "report" && currentResult && (
        <IssueReport
          result={currentResult}
          onClose={() => setModal(null)}
        />
      )}
      {modal === "provenance" && (
        <DataProvenanceDialog
          capabilities={capabilities}
          onClose={() => setModal(null)}
        />
      )}
    </>
  );
}
