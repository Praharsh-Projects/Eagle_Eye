import { useMemo } from "react";

import balticCountriesRaw from "../assets/maps/ne_110m_baltic_countries.geojson?raw";
import "./operational-overview.css";

export type OverviewCapabilityState = "loading" | "ready" | "unavailable";
export type OverviewStatusTone =
  | "success"
  | "warning"
  | "error"
  | "neutral";

export interface HistoricalCountryCoverage {
  code: string;
  name: string;
  count: number;
}

export interface OverviewStatus {
  label: string;
  detail: string;
  tone: OverviewStatusTone;
}

export interface ValidCoverageBox {
  south: number;
  west: number;
  north: number;
  east: number;
}

export interface OverviewHistoryRecord {
  id: string;
  question: string;
  createdAt: string;
  route: string;
  state?: string | null;
  area?: string | null;
  result?: {
    state?: string | null;
  };
}

export interface OperationalOverviewProps<
  TRecord extends OverviewHistoryRecord = OverviewHistoryRecord,
> {
  capabilityState: OverviewCapabilityState;
  boundingBoxes?: Array<[[number, number], [number, number]]>;
  liveCountryCodes: string[];
  historicalCoverage: HistoricalCountryCoverage[];
  historicalFrom?: string | null;
  historicalTo?: string | null;
  provider: string;
  sourceStatus: OverviewStatus;
  evidenceStatus: OverviewStatus;
  horizonHours?: number | null;
  timezone?: string | null;
  datasetCount: number;
  recordCount: number | null;
  recentRecords?: TRecord[];
  history?: TRecord[];
  onEnterAnalysis: () => void;
  onSelectHistory: (record: TRecord) => void;
  onOpenProvenance: () => void;
}

type Position = [number, number];
type PolygonCoordinates = Position[][];
type MultiPolygonCoordinates = Position[][][];

interface BalticFeature {
  type: "Feature";
  properties: {
    name?: string;
    iso_a2?: string;
  };
  geometry: {
    type: "Polygon" | "MultiPolygon";
    coordinates: PolygonCoordinates | MultiPolygonCoordinates;
  };
}

interface BalticFeatureCollection {
  type: "FeatureCollection";
  features: BalticFeature[];
}

const COUNTRY_NAMES: Record<string, string> = {
  DE: "Germany",
  DK: "Denmark",
  EE: "Estonia",
  FI: "Finland",
  LT: "Lithuania",
  LV: "Latvia",
  PL: "Poland",
  SE: "Sweden",
};

const COUNTRY_ORDER = ["SE", "FI", "PL", "EE", "LV", "LT", "DK", "DE"];
const REGIONAL_CODES = new Set(COUNTRY_ORDER);

const ROUTE_LABELS: Record<string, string> = {
  "/analysis": "Analysis Desk",
  "/traffic-monitoring": "Traffic Monitoring",
  "/vessel-investigation": "Vessel Investigation",
  "/eta-delay": "ETA Watch",
  "/port-pressure": "Port Pressure",
  "/carbon-emissions": "Carbon Emissions",
};

const LABEL_POSITIONS: Record<string, Position> = {
  DE: [11.3, 53.75],
  DK: [10.15, 56.3],
  EE: [25.65, 58.75],
  FI: [25.25, 64.1],
  LT: [24.25, 55.25],
  LV: [24.65, 56.85],
  PL: [18.7, 53.85],
  SE: [16.45, 62.2],
};

const VIEW = {
  minLongitude: 8,
  maxLongitude: 32,
  minLatitude: 53,
  maxLatitude: 70,
  width: 1120,
  height: 650,
  padding: 36,
};

let parsedBalticCountries: BalticFeatureCollection | null = null;
try {
  parsedBalticCountries = JSON.parse(
    balticCountriesRaw,
  ) as BalticFeatureCollection;
} catch {
  parsedBalticCountries = null;
}

function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function formatCount(value: number | null): string {
  return value === null
    ? "Row total unavailable"
    : `${new Intl.NumberFormat("en-US").format(value)} records`;
}

function formatDate(value?: string | null): string {
  if (!value) return "Not reported";
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return value;
  return new Intl.DateTimeFormat("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  }).format(date);
}

function formatTimestamp(value: string): string {
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return value;
  return `${new Intl.DateTimeFormat("en-GB", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
    timeZone: "UTC",
  }).format(date)} UTC`;
}

function routeLabel(route: string): string {
  return ROUTE_LABELS[route] || "Analysis";
}

function displayState(value?: string | null): string {
  if (!value) return "Saved";
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function coverageTier(count: number): 0 | 1 | 2 | 3 | 4 {
  if (count <= 0) return 0;
  if (count <= 5) return 1;
  if (count <= 20) return 2;
  if (count <= 50) return 3;
  return 4;
}

function project(position: Position): Position {
  const [longitude, latitude] = position;
  const usableWidth = VIEW.width - VIEW.padding * 2;
  const usableHeight = VIEW.height - VIEW.padding * 2;
  return [
    VIEW.padding +
      ((longitude - VIEW.minLongitude) /
        (VIEW.maxLongitude - VIEW.minLongitude)) *
        usableWidth,
    VIEW.padding +
      ((VIEW.maxLatitude - latitude) /
        (VIEW.maxLatitude - VIEW.minLatitude)) *
        usableHeight,
  ];
}

function validPosition(value: unknown): value is Position {
  return (
    Array.isArray(value) &&
    value.length >= 2 &&
    finite(value[0]) &&
    finite(value[1])
  );
}

function ringPath(ring: Position[]): string {
  const points = ring.filter(validPosition).map(project);
  if (points.length < 3) return "";
  return `${points
    .map(([x, y], index) => `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`)
    .join(" ")} Z`;
}

function geometryPath(feature: BalticFeature): string {
  if (feature.geometry.type === "Polygon") {
    return (feature.geometry.coordinates as PolygonCoordinates)
      .map(ringPath)
      .filter(Boolean)
      .join(" ");
  }
  return (feature.geometry.coordinates as MultiPolygonCoordinates)
    .flatMap((polygon) => polygon.map(ringPath))
    .filter(Boolean)
    .join(" ");
}

function regionalFeatures(): BalticFeature[] {
  if (!parsedBalticCountries?.features) return [];
  return parsedBalticCountries.features
    .filter((feature) => {
      const code = feature.properties.iso_a2?.toUpperCase();
      return Boolean(code && REGIONAL_CODES.has(code) && geometryPath(feature));
    })
    .sort(
      (left, right) =>
        COUNTRY_ORDER.indexOf(left.properties.iso_a2?.toUpperCase() || "") -
        COUNTRY_ORDER.indexOf(right.properties.iso_a2?.toUpperCase() || ""),
    );
}

export function validateOverviewCoverageBoxes(
  boxes: OperationalOverviewProps["boundingBoxes"],
): ValidCoverageBox[] {
  if (!Array.isArray(boxes)) return [];

  return boxes.flatMap((box) => {
    if (!Array.isArray(box) || box.length !== 2) return [];
    const [lower, upper] = box;
    if (
      !Array.isArray(lower) ||
      !Array.isArray(upper) ||
      lower.length !== 2 ||
      upper.length !== 2
    ) {
      return [];
    }

    const [south, west] = lower;
    const [north, east] = upper;
    if (
      !finite(south) ||
      !finite(west) ||
      !finite(north) ||
      !finite(east) ||
      south < -90 ||
      north > 90 ||
      west < -180 ||
      east > 180 ||
      south >= north ||
      west >= east
    ) {
      return [];
    }

    return [{ south, west, north, east }];
  });
}

export function deriveHistoricalCountryCoverage(
  ports: unknown,
): HistoricalCountryCoverage[] {
  if (!Array.isArray(ports)) return [];

  const uniquePorts = new Set(
    ports
      .filter((value): value is string => typeof value === "string")
      .map((value) => value.trim().toUpperCase())
      .filter((value) => /^[A-Z]{2}[A-Z0-9]{3}$/.test(value)),
  );
  const counts = new Map<string, number>();

  uniquePorts.forEach((port) => {
    const countryCode = port.slice(0, 2);
    counts.set(countryCode, (counts.get(countryCode) || 0) + 1);
  });

  return Array.from(counts, ([code, count]) => ({
    code,
    name: COUNTRY_NAMES[code] || code,
    count,
  })).sort(
    (left, right) =>
      right.count - left.count || left.name.localeCompare(right.name),
  );
}

function RibbonItem({
  label,
  status,
}: {
  label: string;
  status: OverviewStatus;
}) {
  return (
    <div className="situation-ribbon-item">
      <dt>{label}</dt>
      <dd>
        <span
          className={`situation-status-mark situation-status-mark-${status.tone}`}
          aria-hidden="true"
        />
        <strong>{status.label}</strong>
        <small>{status.detail}</small>
      </dd>
    </div>
  );
}

function HistoricalAtlas({
  coverage,
  historicalFrom,
  historicalTo,
}: {
  coverage: HistoricalCountryCoverage[];
  historicalFrom?: string | null;
  historicalTo?: string | null;
}) {
  const features = useMemo(regionalFeatures, []);
  const coverageByCountry = useMemo(
    () =>
      new Map(
        coverage
          .filter(
            (item) =>
              REGIONAL_CODES.has(item.code.toUpperCase()) &&
              finite(item.count) &&
              item.count >= 0,
          )
          .map((item) => [item.code.toUpperCase(), item]),
      ),
    [coverage],
  );
  const hasRegionalCoverage = Array.from(coverageByCountry.values()).some(
    (country) => country.count > 0,
  );
  const atlasReady =
    features.length === COUNTRY_ORDER.length && hasRegionalCoverage;
  const ledger = COUNTRY_ORDER.map((code) => ({
    code,
    name: COUNTRY_NAMES[code],
    count: coverageByCountry.get(code)?.count || 0,
  }));

  if (!atlasReady) {
    return (
      <div className="situation-atlas-fallback" role="status">
        <div>
          <p className="situation-kicker">Coverage ledger</p>
          <h2>Atlas geometry or coverage is unavailable</h2>
          <p>
            Eagle Eye will not infer a geographic footprint from missing data.
          </p>
        </div>
        <dl>
          {ledger.map((country) => (
            <div key={country.code}>
              <dt>
                {country.name} <span>{country.code}</span>
              </dt>
              <dd>
                {country.count > 0
                  ? `${country.count} verified ports`
                  : "Not represented"}
              </dd>
            </div>
          ))}
        </dl>
      </div>
    );
  }

  const longitudeLines = [10, 15, 20, 25, 30];
  const latitudeLines = [55, 60, 65, 70];

  return (
    <div className="situation-atlas-layout">
      <figure className="situation-atlas-figure">
        <svg
          className="situation-atlas-svg"
          data-testid="overview-historical-atlas"
          viewBox={`0 0 ${VIEW.width} ${VIEW.height}`}
          role="img"
          aria-labelledby="situation-atlas-title situation-atlas-description"
          preserveAspectRatio="xMidYMid meet"
        >
          <title id="situation-atlas-title">
            Historical Baltic port coverage atlas
          </title>
          <desc id="situation-atlas-description">
            Eight Baltic-region countries are shown. Six countries have verified
            historical port coverage. Fill intensity represents deduplicated port
            counts, not traffic volume or current vessel activity.
          </desc>

          <g className="situation-graticule" aria-hidden="true">
            {longitudeLines.map((longitude) => {
              const [x] = project([longitude, VIEW.minLatitude]);
              return (
                <g key={`longitude-${longitude}`}>
                  <line
                    x1={x}
                    y1={VIEW.padding}
                    x2={x}
                    y2={VIEW.height - VIEW.padding}
                  />
                  <text x={x + 7} y={VIEW.height - 12}>
                    {longitude}°E
                  </text>
                </g>
              );
            })}
            {latitudeLines.map((latitude) => {
              const [, y] = project([VIEW.minLongitude, latitude]);
              return (
                <g key={`latitude-${latitude}`}>
                  <line
                    x1={VIEW.padding}
                    y1={y}
                    x2={VIEW.width - VIEW.padding}
                    y2={y}
                  />
                  <text x={8} y={y - 7}>
                    {latitude}°N
                  </text>
                </g>
              );
            })}
          </g>

          <g className="situation-countries">
            {features.map((feature) => {
              const code = feature.properties.iso_a2?.toUpperCase() || "";
              const country = coverageByCountry.get(code);
              const count = country?.count || 0;
              const tier = coverageTier(count);
              return (
                <path
                  key={code}
                  d={geometryPath(feature)}
                  className={`situation-country situation-country-tier-${tier}`}
                  data-country-code={code}
                  data-coverage-count={count}
                  data-coverage-tier={tier}
                >
                  <title>
                    {COUNTRY_NAMES[code] || feature.properties.name || code}: {count > 0 ? `${count} verified historical ports` : "not represented in the historical archive"}
                  </title>
                </path>
              );
            })}
          </g>

          <g className="situation-country-labels" aria-hidden="true">
            {COUNTRY_ORDER.map((code) => {
              const position = LABEL_POSITIONS[code];
              const [x, y] = project(position);
              const count = coverageByCountry.get(code)?.count || 0;
              return (
                <g key={`label-${code}`} transform={`translate(${x} ${y})`}>
                  <text className="situation-label-name" textAnchor="middle">
                    {COUNTRY_NAMES[code].toUpperCase()}
                  </text>
                  <text
                    className="situation-label-value"
                    textAnchor="middle"
                    y={19}
                  >
                    {count > 0 ? `${code} / ${count}` : `${code} / —`}
                  </text>
                </g>
              );
            })}
          </g>
        </svg>
        <figcaption>
          Historical coverage—not traffic volume or current vessel activity.
        </figcaption>
      </figure>

      <aside className="situation-atlas-key" aria-label="Atlas coverage key">
        <div>
          <p className="situation-kicker">Printed archive / 01</p>
          <h2>Coverage key</h2>
          <p>
            Country fill is derived only from unique UN/LOCODE values in the
            validated manifest.
          </p>
        </div>

        <ol className="situation-tier-key">
          <li>
            <span className="situation-key-swatch situation-key-tier-4" />
            <b>51+</b> ports
          </li>
          <li>
            <span className="situation-key-swatch situation-key-tier-3" />
            <b>21–50</b> ports
          </li>
          <li>
            <span className="situation-key-swatch situation-key-tier-2" />
            <b>6–20</b> ports
          </li>
          <li>
            <span className="situation-key-swatch situation-key-tier-1" />
            <b>1–5</b> ports
          </li>
          <li>
            <span className="situation-key-swatch situation-key-tier-0" />
            <b>0</b> represented
          </li>
        </ol>

        <dl className="situation-archive-window">
          <div>
            <dt>Archive opens</dt>
            <dd>{formatDate(historicalFrom)}</dd>
          </div>
          <div>
            <dt>Archive closes</dt>
            <dd>{formatDate(historicalTo)}</dd>
          </div>
        </dl>

        <details
          className="situation-data-disclosure"
          data-testid="overview-coverage-disclosure"
        >
          <summary>View coverage data</summary>
          <div className="situation-coverage-table-wrap" tabIndex={0}>
            <table aria-label="Historical country coverage data">
              <thead>
                <tr>
                  <th>Country</th>
                  <th>Code</th>
                  <th>Ports</th>
                </tr>
              </thead>
              <tbody>
                {ledger.map((country) => (
                  <tr key={`coverage-row-${country.code}`}>
                    <td>{country.name}</td>
                    <td>{country.code}</td>
                    <td>{country.count || "Not represented"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      </aside>
    </div>
  );
}

export function OperationalOverview<
  TRecord extends OverviewHistoryRecord = OverviewHistoryRecord,
>({
  capabilityState,
  historicalCoverage,
  historicalFrom,
  historicalTo,
  sourceStatus,
  evidenceStatus,
  datasetCount,
  recordCount,
  recentRecords,
  history,
  onEnterAnalysis,
  onSelectHistory,
  onOpenProvenance,
}: OperationalOverviewProps<TRecord>) {
  const records = recentRecords ?? history ?? [];
  const historicalCountries = useMemo(
    () =>
      historicalCoverage.filter(
        (country) => REGIONAL_CODES.has(country.code.toUpperCase()) && country.count > 0,
      ),
    [historicalCoverage],
  );
  const representedPorts = historicalCountries.reduce(
    (sum, country) => sum + country.count,
    0,
  );

  const archiveStatus: OverviewStatus = {
    label:
      historicalFrom || historicalTo
        ? `${formatDate(historicalFrom)} — ${formatDate(historicalTo)}`
        : "Boundary unavailable",
    detail: "Historical—not current operational truth",
    tone: historicalFrom || historicalTo ? "neutral" : "warning",
  };
  const dataStatus: OverviewStatus = {
    label: datasetCount > 0 ? `${datasetCount} validated datasets` : "Unavailable",
    detail: formatCount(recordCount),
    tone: datasetCount > 0 && recordCount !== null ? "neutral" : "warning",
  };

  return (
    <section
      className="operational-overview situation-sheet"
      data-testid="operational-overview"
      aria-labelledby="operational-overview-title"
    >
      <header className="situation-masthead">
        <div className="situation-identity">
          <p>Eagle Eye / Baltic situation sheet</p>
          <h1 id="operational-overview-title">Baltic archive footprint</h1>
          <strong>
            {representedPorts > 0
              ? `${representedPorts} verified ports · ${historicalCountries.length} represented countries`
              : "Verified historical coverage unavailable"}
          </strong>
          <span>
            A geographic view of the archive Eagle Eye can calculate from—not a
            live operating picture.
          </span>
        </div>

        <div className="situation-actions">
          <button
            type="button"
            className="situation-primary-action"
            data-testid="overview-enter-analysis"
            onClick={onEnterAnalysis}
          >
            Start analysis
            <span aria-hidden="true">↗</span>
          </button>
          <button
            type="button"
            className="situation-provenance-action"
            data-testid="overview-provenance-button"
            aria-haspopup="dialog"
            onClick={onOpenProvenance}
          >
            Data provenance
          </button>
        </div>
      </header>

      <dl
        className="situation-readiness-ribbon"
        aria-label="Workspace readiness"
        aria-busy={capabilityState === "loading"}
      >
        <RibbonItem label="Archive" status={archiveStatus} />
        <RibbonItem label="AIS watch" status={sourceStatus} />
        <RibbonItem label="Evidence" status={evidenceStatus} />
        <RibbonItem label="Data volume" status={dataStatus} />
      </dl>

      <section
        className="situation-atlas-stage"
        aria-label="Historical Baltic coverage"
      >
        <HistoricalAtlas
          coverage={historicalCoverage}
          historicalFrom={historicalFrom}
          historicalTo={historicalTo}
        />
      </section>

      <section
        className="situation-activity-tape"
        aria-labelledby="recent-analysis-title"
      >
        <header>
          <div>
            <p className="situation-kicker">Local workspace / recent work</p>
            <h2 id="recent-analysis-title">Analysis tape</h2>
          </div>
          <span>{records.length} saved</span>
        </header>

        {records.length ? (
          <ol>
            {records.map((record, index) => {
              const recordState =
                record.state || record.result?.state || "Saved";
              return (
                <li
                  key={record.id}
                  className={index === 0 ? "situation-activity-lead" : undefined}
                >
                  <button
                    type="button"
                    onClick={() => onSelectHistory(record)}
                    aria-label={`Restore analysis: ${record.question}`}
                  >
                    <span className="situation-activity-index">
                      {String(index + 1).padStart(2, "0")}
                    </span>
                    <span className="situation-activity-meta">
                      <b>{record.area || routeLabel(record.route)}</b>
                      <i>{displayState(recordState)}</i>
                    </span>
                    <strong>{record.question}</strong>
                    <time dateTime={record.createdAt}>
                      {formatTimestamp(record.createdAt)}
                    </time>
                  </button>
                </li>
              );
            })}
          </ol>
        ) : (
          <p className="situation-activity-empty">
            No saved analyses yet. Completed work will appear in this tape.
          </p>
        )}
      </section>
    </section>
  );
}
