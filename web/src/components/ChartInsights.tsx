import type { ChartInsight } from "../types";

function insightLabel(value: string): string {
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

export function ChartInsights({
  insights,
}: {
  insights?: ChartInsight[];
}) {
  if (!insights?.length) return null;

  return (
    <section
      className="chart-insights"
      aria-labelledby="chart-insights-title"
      data-testid="chart-insights"
    >
      <header className="chart-insights-heading">
        <div>
          <p className="section-code">Chart observations</p>
          <h2 id="chart-insights-title">Observed in the chart</h2>
        </div>
        <span>{insights.length} {insights.length === 1 ? "item" : "items"}</span>
      </header>
      <ol className="chart-insight-list">
        {insights.map((insight, index) => (
          <li key={insight.id}>
            <span className="chart-insight-index" aria-hidden="true">
              {String(index + 1).padStart(2, "0")}
            </span>
            <div>
              <small>{insightLabel(insight.insight_type)}</small>
              <p>{insight.statement}</p>
              <details>
                <summary>References</summary>
                <dl>
                  <div>
                    <dt>Chart</dt>
                    <dd>{insight.visualization_id}</dd>
                  </div>
                  {insight.fact_names.length > 0 && (
                    <div>
                      <dt>Fact slots</dt>
                      <dd>{insight.fact_names.join(", ")}</dd>
                    </div>
                  )}
                  {insight.evidence_ids.length > 0 && (
                    <div>
                      <dt>Evidence</dt>
                      <dd>{insight.evidence_ids.join(", ")}</dd>
                    </div>
                  )}
                </dl>
              </details>
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}
