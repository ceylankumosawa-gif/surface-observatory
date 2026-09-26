import { useId, useState } from "react";
import { ArrowDownToLine, ChevronRight, RefreshCw } from "lucide-react";
import { solarChartRows, solarRoleLabel, solarTestLabel, type SolarEvidence } from "./solar-season-evidence";

const n = (v: number | null) => v === null ? "—" : v.toFixed(2);
const pct = (v: number | null) => v === null ? "—" : `${(v * 100).toFixed(1)}%`;
const count = (v: number) => v.toLocaleString("en-GB");

export function SolarSeasonEvidence({ data }: { data: SolarEvidence }) {
  const titleId = useId();
  const [phase, setPhase] = useState<"all" | "day" | "night">("all");
  const rows = data.regions.filter(r => phase === "all" || r.phase === phase);
  const comparisons = solarChartRows(data.comparisons);
  const scale = Math.max(5, Math.ceil(Math.max(3, ...comparisons.flatMap(r => [r.baseline_balanced_mae ?? 0, r.candidate_balanced_mae ?? 0])) * 1.12));
  return <section className="gre-expansion gre-solar" aria-labelledby={titleId}>
    <div className="gre-section-heading"><div><span className="gre-kicker">A SEPARATE REPRESENTATION TEST</span><h3 id={titleId}>Describe the season through sunlight</h3></div><span className="gre-status">Experiment checked · no live model change</span></div>
    <p>{data.description}</p>
    <div className="gre-solar-legend" role="group" aria-label="Compared models"><span><i className="gre-solar-baseline" aria-hidden="true" />{data.baseline_label}</span><span><i className="gre-solar-candidate" aria-hidden="true" />{data.candidate_label}</span></div>
    <div className="gre-solar-charts" role="group" aria-label="Solar-season paired balanced mean absolute errors">
      {(["geography", "month", "reference"] as const).map(mode => <figure className="gre-chart" key={mode}>
        <figcaption><strong>{solarTestLabel(mode)}</strong><span>Balanced MAE · lower is better</span></figcaption>
        <div className="gre-chart-axis" aria-hidden="true"><span>0</span><span>{scale} °C</span></div>
        <div className="gre-chart-body"><div className="gre-target" style={{ left: `${3 / scale * 100}%` }} aria-hidden="true"><span>3° target</span></div>
          {comparisons.filter(r => r.test_mode === mode).map(r => <div className="gre-solar-pair" key={r.geography_role}>
            <strong className="gre-solar-population">{solarRoleLabel(r.geography_role)}</strong>
            {([['baseline', data.baseline_label, r.baseline_balanced_mae], ['candidate', data.candidate_label, r.candidate_balanced_mae]] as const).map(([arm, label, value]) => <div className="gre-bar-row" key={arm}>
              <div className="gre-bar-label"><span>{label}</span><strong>{value === null ? "No paired data" : `${n(value)}°`}</strong></div>
              <div className="gre-bar-track" role="img" aria-label={`${solarRoleLabel(r.geography_role)}, ${label}: ${value === null ? "no paired observations" : `${n(value)} degrees Celsius balanced mean absolute error`}`}>
                {value !== null && <div className={`gre-bar gre-solar-${arm}`} style={{ width: `${value / scale * 100}%` }} />}
              </div></div>)}
            <small>{count(r.paired)} paired observations · {count(r.dates)} dates</small>
          </div>)}
        </div>
      </figure>)}
    </div>
    <p className="gre-footnote">The shared scale compares the same held observations. The 3°C line is a target, not a qualification; a lower average can still hide large errors.</p>
    <div className="gre-solar-downloads"><a className="gre-download" href={data.download_url} download><ArrowDownToLine size={15} />All 96 area / phase outcomes</a><a className="gre-download" href={data.cases_download_url} download><ArrowDownToLine size={15} />All 960 requested cases</a></div>
    <details className="gre-expansion-regions"><summary>Every area, including missing and reserved outcomes<ChevronRight className="gre-disclosure-chevron" size={16} aria-hidden="true" /></summary>
      <div className="gre-controls"><div className="gre-phase" role="group" aria-label="Solar-season detail time of day">{(["all", "day", "night"] as const).map(p => <button key={p} aria-pressed={phase === p} onClick={() => setPhase(p)}>{p === "all" ? "Day + night" : p === "day" ? "Day" : "Night"}</button>)}</div><button className="gre-reset" onClick={() => setPhase("all")}><RefreshCw size={13} />Reset</button></div>
      <p className="gre-row-status" role="status">Showing {rows.length} of 96 area/phase groups · {rows.filter(r => r.paired === 0).length} without paired observations in this view · downloads retain all groups</p>
      <div className="gre-table-wrap" tabIndex={0} role="region" aria-label="Solar-season outcomes; scroll horizontally for all measures">
        <table><caption>Each paired cell reads Expanded baseline → Solar-season candidate · errors in °C · fractions count paired observations equally</caption>
          <thead><tr><th>Area / phase</th><th>Role</th><th>Balanced MAE</th><th>Ordinary MAE</th><th>Ordinary signed bias</th><th>Errors &gt;5°C</th><th>Errors &gt;7°C</th><th>Dates</th><th>Paired observations</th><th>Outcome</th></tr></thead>
          <tbody>{rows.map(r => <tr key={`${r.region_id}|${r.phase}`} className={r.paired === 0 ? "gre-missing" : undefined}><th scope="row">{r.region_id.replaceAll("_", " ")}<small>{r.phase === "day" ? "Day" : "Night"} · {solarTestLabel(r.test_mode)}</small></th><td>{solarRoleLabel(r.geography_role)}</td><td>{n(r.baseline_balanced_mae)} → {n(r.candidate_balanced_mae)}</td><td>{n(r.baseline_mae)} → {n(r.candidate_mae)}</td><td>{n(r.baseline_bias)} → {n(r.candidate_bias)}</td><td>{pct(r.baseline_above_5_fraction)} → {pct(r.candidate_above_5_fraction)}</td><td>{pct(r.baseline_above_7_fraction)} → {pct(r.candidate_above_7_fraction)}</td><td>{count(r.dates)}</td><td>{count(r.paired)}</td><td>{r.paired === 0 && r.geography_role === "reserved_new_geography" ? "Reserved · unopened" : r.outcome}</td></tr>)}</tbody>
        </table></div><p className="gre-footnote">— means no score, never zero error. “Improved” and “worsened” describe balanced MAE only. Footprints may overlap; their count is not independent weather-event support.</p>
    </details>
    <details className="gre-receipts gre-solar-receipt"><summary>Candidate source receipts<ChevronRight className="gre-disclosure-chevron" size={16} aria-hidden="true" /></summary><dl><dt>Independent audit SHA-256</dt><dd>{data.source.audit_sha256}</dd><dt>Audited source metrics SHA-256</dt><dd>{data.source.metrics_sha256}</dd><dt>Completed candidate SHA-256</dt><dd>{data.source.trial_sha256}</dd></dl></details>
    <p className="gre-updated">Candidate snapshot: {data.generated_at_utc.replace("T", " ").replace(/\.\d+(Z|\+00:00)$/, " UTC").replace(/(Z|\+00:00)$/, " UTC")}</p>
  </section>;
}
