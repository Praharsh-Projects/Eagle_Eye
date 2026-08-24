import type { WorkspaceRoute } from "../types";
import type { QueryPageCategory } from "./sampleQueries";

export type AnalysisPageId =
  | "overview"
  | "analysis"
  | "traffic-monitoring"
  | "vessel-investigation"
  | "eta-delay"
  | "port-pressure"
  | "carbon-emissions";

export interface AnalysisPageDefinition {
  id: AnalysisPageId;
  route: WorkspaceRoute;
  displayLabel: string;
  internalLabel: string;
  scope: string;
  sampleCategory?: QueryPageCategory;
}

export const ANALYSIS_PAGES: readonly AnalysisPageDefinition[] = [
  {
    id: "overview",
    route: "/overview",
    displayLabel: "Overview",
    internalLabel: "Overview",
    scope: "Validated historical inventory, live-source readiness, and recent analytical work.",
  },
  {
    id: "analysis",
    route: "/analysis",
    displayLabel: "Analysis Desk",
    internalLabel: "Chat Assistant",
    scope: "Cross-domain maritime analysis with evidence and follow-up context.",
  },
  {
    id: "traffic-monitoring",
    route: "/traffic-monitoring",
    displayLabel: "Traffic Monitoring",
    internalLabel: "Traffic Monitoring",
    scope: "Vessel arrivals, temporal patterns, comparisons, and route durations.",
    sampleCategory: "Traffic Monitoring",
  },
  {
    id: "vessel-investigation",
    route: "/vessel-investigation",
    displayLabel: "Vessel Investigation",
    internalLabel: "Vessel Investigation",
    scope: "MMSI-based port stays and AIS movement-anomaly evidence.",
    sampleCategory: "Vessel Investigation",
  },
  {
    id: "eta-delay",
    route: "/eta-delay",
    displayLabel: "ETA Watch",
    internalLabel: "ETA & Delay",
    scope: "Sweden-first Baltic inbound watchlists, vessel-reported ETAs, positions, revisions, and signal-quality exceptions.",
    sampleCategory: "ETA & Delay",
  },
  {
    id: "port-pressure",
    route: "/port-pressure",
    displayLabel: "Port Pressure",
    internalLabel: "Port Pressure",
    scope: "Pressure-v2 trends, baseline comparisons, and ranked pressure days.",
    sampleCategory: "Port Pressure",
  },
  {
    id: "carbon-emissions",
    route: "/carbon-emissions",
    displayLabel: "Carbon Emissions",
    internalLabel: "Carbon Emissions",
    scope: "Deterministic TTW and WTW emissions by port, period, or vessel call.",
    sampleCategory: "Carbon Emissions",
  },
] as const;

export const DEFAULT_WORKSPACE_ROUTE: WorkspaceRoute = "/overview";

/**
 * The complete set of internal locations that may be restored from
 * device-local history. Never pass a persisted string directly to navigation.
 */
export const WORKSPACE_ROUTE_ALLOWLIST: readonly WorkspaceRoute[] = [
  "/overview",
  "/analysis",
  "/traffic-monitoring",
  "/vessel-investigation",
  "/eta-delay",
  "/port-pressure",
  "/carbon-emissions",
] as const;

const WORKSPACE_ROUTE_SET = new Set<string>(WORKSPACE_ROUTE_ALLOWLIST);

export function isWorkspaceRoute(value: unknown): value is WorkspaceRoute {
  return typeof value === "string" && WORKSPACE_ROUTE_SET.has(value);
}

export function normalizeWorkspaceRoute(
  value: unknown,
  fallback: WorkspaceRoute = "/analysis",
): WorkspaceRoute {
  return isWorkspaceRoute(value) ? value : fallback;
}

export const ANALYSIS_PAGE_BY_ROUTE = Object.fromEntries(
  ANALYSIS_PAGES.map((page) => [page.route, page]),
) as Record<WorkspaceRoute, AnalysisPageDefinition>;
