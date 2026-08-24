import rawCatalog from "./queryCatalog.json";
import rawGuidance from "./queryGuidance.json";

export const QUERY_CATEGORY_PAGE_ORDER = [
  "Traffic Monitoring",
  "Vessel Investigation",
  "ETA & Delay",
  "Port Pressure",
  "Carbon Emissions",
] as const;

export const SAMPLE_QUERY_CATEGORIES = [
  ...QUERY_CATEGORY_PAGE_ORDER,
  "Unsupported Scope",
] as const;

export type QueryPageCategory = (typeof QUERY_CATEGORY_PAGE_ORDER)[number];
export type SampleQueryCategory = (typeof SAMPLE_QUERY_CATEGORIES)[number];

export interface QueryCategoryHelp {
  overview: string;
  what_to_enter: string[];
  expected_output: string[];
  test_steps: string[];
  calculation: string[];
}

export const SAMPLE_QUERIES_BY_CATEGORY =
  rawCatalog as Record<SampleQueryCategory, string[]>;

export const QUERY_CATEGORY_HELP =
  rawGuidance as Record<SampleQueryCategory, QueryCategoryHelp>;

export const UNSUPPORTED_SCOPE_PROMPTS =
  SAMPLE_QUERIES_BY_CATEGORY["Unsupported Scope"];

export const ALL_SAMPLE_PROMPTS = SAMPLE_QUERY_CATEGORIES.flatMap(
  (category) => SAMPLE_QUERIES_BY_CATEGORY[category],
);

export const ETA_WATCH_SAMPLE_GROUPS = [
  {
    label: "Shift Briefs",
    prompts: SAMPLE_QUERIES_BY_CATEGORY["ETA & Delay"].slice(0, 1),
  },
  {
    label: "Inbound Watchlists",
    prompts: SAMPLE_QUERIES_BY_CATEGORY["ETA & Delay"].slice(1, 5),
  },
  {
    label: "Exceptions",
    prompts: SAMPLE_QUERIES_BY_CATEGORY["ETA & Delay"].slice(5, 7),
  },
  {
    label: "Vessel Lookup",
    prompts: SAMPLE_QUERIES_BY_CATEGORY["ETA & Delay"].slice(7, 8),
  },
  {
    label: "Baltic Coverage",
    prompts: SAMPLE_QUERIES_BY_CATEGORY["ETA & Delay"].slice(8, 10),
  },
] as const;

export const QUERY_CATALOG_SHA256 =
  "1df9188bbdfc1a6d411d8a9b56426ea7cafaa6838e9424046c727e7b72282a47";
