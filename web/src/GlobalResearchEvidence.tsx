import { useEffect, useId, useState } from "react";
import { ArrowDownToLine, ChevronRight, FlaskConical, RefreshCw } from "lucide-react";
import "./global-research-evidence.css";
import { SolarSeasonEvidence } from "./SolarSeasonEvidence";
import { readSolarEvidence, type SolarEvidence } from "./solar-season-evidence";

type Metric = {
  mae: number | null; balanced_mae: number | null; bias: number | null;
  paired: number; dates: number; sites: number;
  above_5_fraction: number | null; above_7_fraction: number | null;
};
type RegionMetric = Metric & {
  region_id: string; label: string; phase: "day" | "night";
  test_mode: "pilot" | "reference";
};
type Study = {
  id: string; label: string; change: string;
  source: { audit_sha256: string; metrics_sha256: string; download_url: string };
  overall: Record<"pilot" | "month" | "reference", Metric>;
  regions: RegionMetric[];
};
type ExpansionComparison = {
  test_mode: string; geography_role: string; old_balanced_mae: number | null;
  expanded_balanced_mae: number | null; paired: number;
};
type ExpansionRegion = ExpansionComparison & {
  region_id: string; phase: "day" | "night"; dates: number;
  old_mae: number | null; expanded_mae: number | null;
  delta_balanced_mae: number | null; outcome: string;
};
type Evidence = {
  version: 1; generated_at_utc: string; live_model_label: string;
  target_mae_c: number; qualified: boolean; metric_definition: string;
  observation_scope: string; period_label: string; studies: Study[];
  solar_season?: SolarEvidence;
  expansion: {
    status: "preparing_predictors" | "evaluating" | "complete";
    development_areas: number; reserved_areas: number; retained_footprints: number;
    context_footprints: number; description: string;
    result: null | { comparisons: ExpansionComparison[]; download_url: string; regions?: ExpansionRegion[] };
  };
};

const MODES = [
  { key: "pilot", title: "A whole area held out", note: "Geographic transfer" },
  { key: "month", title: "A whole month held out", note: "Seasonal transfer" },
  { key: "reference", title: "Cabauw reference", note: "Never used for fitting" },
] as const;
const count = (n: number) => n.toLocaleString("en-GB");
const number = (n: number | null) => n === null ? "—" : n.toFixed(2);
const percent = (n: number | null) => n === null ? "—" : `${(n * 100).toFixed(1)}%`;
const regionKey = (r: RegionMetric) => `${r.region_id}|${r.phase}`;
const sameOriginDownload = (path: unknown): path is string => {
  if (typeof path !== "string" || !path.startsWith("/research/") || path.includes("\\")) return false;
  const url = new URL(path, window.location.origin);
  return url.origin === window.location.origin && url.pathname.startsWith("/research/");
};

function readEvidence(input: unknown): Evidence {
  const d = input as Evidence;
  const text = (x: unknown) => typeof x === "string" && x.length > 0;
  const integer = (x: unknown) => typeof x === "number" && Number.isInteger(x) && x >= 0;
  const finite = (x: unknown) => x === null || (typeof x === "number" && Number.isFinite(x));
  const fraction = (x: unknown) => x === null || (typeof x === "number" && Number.isFinite(x) && x >= 0 && x <= 1);
  const metric = (m: Metric) => !!m && [m.paired, m.dates, m.sites].every(integer)
    && [m.mae, m.balanced_mae, m.bias].every(finite)
    && [m.mae, m.balanced_mae].every(x => x === null || x >= 0)
    && [m.above_5_fraction, m.above_7_fraction].every(fraction)
    && (m.paired > 0 || [m.mae, m.balanced_mae, m.bias, m.above_5_fraction, m.above_7_fraction].every(x => x === null));
  const fail = () => { throw new Error("The research evidence file is incomplete or has an unsupported format."); };
  if (!d || d.version !== 1 || ![d.generated_at_utc, d.live_model_label, d.metric_definition,
    d.observation_scope, d.period_label].every(text) || !Number.isFinite(d.target_mae_c)
    || d.target_mae_c <= 0 || typeof d.qualified !== "boolean" || !Array.isArray(d.studies)
    || ![5, 6].includes(d.studies.length) || new Set(d.studies.map(s => s?.id)).size !== d.studies.length) fail();
  let keys: string[] | undefined;
  for (const s of d.studies) {
    if (!s || ![s.id, s.label, s.change].every(text) || !s.source
      || ![s.source.audit_sha256, s.source.metrics_sha256].every(h => typeof h === "string" && /^[a-f0-9]{64}$/.test(h))
      || !sameOriginDownload(s.source.download_url) || !s.overall
      || !MODES.every(m => metric(s.overall[m.key])) || !Array.isArray(s.regions) || s.regions.length !== 24) fail();
    if (!s.regions.every(r => metric(r) && text(r.region_id) && text(r.label)
      && ["day", "night"].includes(r.phase) && ["pilot", "reference"].includes(r.test_mode))) fail();
    const current = s.regions.map(regionKey).sort();
    if (new Set(current).size !== 24 || (keys && current.join("\n") !== keys.join("\n"))) fail();
    keys = current;
  }
  const x = d.expansion;
  if (!x || !["preparing_predictors", "evaluating", "complete"].includes(x.status)
    || ![x.development_areas, x.reserved_areas, x.retained_footprints, x.context_footprints].every(integer)
    || !text(x.description) || x.context_footprints > x.retained_footprints) fail();
  if (x.result !== null && (!x.result || x.status !== "complete" || !sameOriginDownload(x.result.download_url)
    || !Array.isArray(x.result.comparisons) || !x.result.comparisons.every(r => text(r.test_mode)
      && text(r.geography_role) && integer(r.paired)
      && [r.old_balanced_mae, r.expanded_balanced_mae].every(v => finite(v) && (v === null || v >= 0))
      && (r.paired > 0 || [r.old_balanced_mae, r.expanded_balanced_mae].every(v => v === null))))) fail();
  if (x.result?.regions !== undefined) {
    const rows = x.result.regions;
    if (!Array.isArray(rows) || rows.length !== 96
      || new Set(rows.map(r => `${r.region_id}|${r.phase}`)).size !== 96
      || !rows.every(r => text(r.region_id) && ["day", "night"].includes(r.phase)
        && text(r.test_mode) && text(r.geography_role) && text(r.outcome)
        && [r.paired, r.dates].every(integer)
        && [r.old_mae, r.expanded_mae, r.old_balanced_mae, r.expanded_balanced_mae].every(v => finite(v) && (v === null || v >= 0))
        && finite(r.delta_balanced_mae)
        && (r.paired > 0 || [r.old_mae, r.expanded_mae, r.old_balanced_mae, r.expanded_balanced_mae, r.delta_balanced_mae].every(v => v === null)))) fail();
  }
  if (d.solar_season !== undefined) readSolarEvidence(d.solar_season, window.location.origin);
  return d;
}

function ComparisonChart({ data }: { data: Evidence }) {
  const values = data.studies.flatMap(s => MODES.map(m => s.overall[m.key].balanced_mae));
  const maximum = Math.max(data.target_mae_c, ...values.filter((x): x is number => x !== null));
  const scale = Math.max(5, Math.ceil(maximum * 1.12));
  const targetPosition = `${data.target_mae_c / scale * 100}%`;
  return <div className="gre-charts" role="group" aria-label="Balanced mean absolute error by held-out test">
    {MODES.map(mode => <figure className="gre-chart" key={mode.key}>
      <figcaption><strong>{mode.title}</strong><span>{mode.note}</span></figcaption>
      <div className="gre-chart-axis" aria-hidden="true"><span>0</span><span>{scale} °C</span></div>
      <div className="gre-chart-body">
        <div className="gre-target" style={{ left: targetPosition }} aria-hidden="true"><span>{data.target_mae_c}° target</span></div>
        {data.studies.map((s, i) => {
          const m = s.overall[mode.key];
          return <div className="gre-bar-row" key={s.id}>
            <div className="gre-bar-label"><span>{s.label}</span><strong>{m.balanced_mae === null ? "No paired data" : `${number(m.balanced_mae)}°`}</strong></div>
            <div className="gre-bar-track" role="img" aria-label={`${s.label}: ${m.balanced_mae === null ? "no paired observations" : `${number(m.balanced_mae)} degrees Celsius balanced mean absolute error`}`}>
              {m.balanced_mae !== null && <div className={`gre-bar gre-series-${i}`} style={{ width: `${m.balanced_mae / scale * 100}%` }} />}
            </div>
          </div>;
        })}
      </div>
    </figure>)}
  </div>;
}

function Expansion({ data }: { data: Evidence["expansion"] }) {
  const status = { preparing_predictors: "Preparing weather & surface inputs", evaluating: "Evaluation in progress", complete: "Experiment complete" }[data.status];
  const roleLabel = (s: string) => ({ original: "Original areas", original_development: "Original development areas", new_development: "New development areas", original_reference: "Separate reference", reference: "Separate reference", new_reserved: "Reserved · unopened", reserved_new_geography: "Reserved · unopened" }[s] ?? s.replaceAll("_", " "));
  const testLabel = (s: string) => ({ pilot: "Whole area held out", geography: "Whole area held out", area: "Whole area held out", month: "Whole month held out", reference: "Cabauw reference" }[s] ?? s.replaceAll("_", " "));
  const comparisons = data.result?.comparisons.filter(r => r.test_mode === "reference"
    ? r.geography_role === "original_reference"
    : r.geography_role !== "original_reference") ?? [];
  return <section className="gre-expansion" aria-labelledby="gre-expansion-title">
    <div className="gre-section-heading"><div><span className="gre-kicker">A SEPARATE EXPERIMENT</span><h3 id="gre-expansion-title">Learn from more places</h3></div><span className="gre-status">{status}</span></div>
    <p>{data.description}</p>
    <dl className="gre-coverage">
      <div><dt>Additional development areas</dt><dd>{count(data.development_areas)}</dd></div>
      <div><dt>Reserved test areas</dt><dd>{count(data.reserved_areas)}</dd></div>
      <div><dt>Retained native footprints</dt><dd>{count(data.retained_footprints)}</dd></div>
      <div><dt>Footprints passing source/context checks</dt><dd>{count(data.context_footprints)}</dd></div>
    </dl>
    <p className="gre-footnote">Footprint counts are not independent dates, complete model inputs or accuracy results.</p>
    {data.result ? <>
      <div className="gre-table-wrap" tabIndex={0} role="region" aria-label="Geographic expansion comparisons">
        <table><caption>Balanced MAE on the same held observations · °C</caption><thead><tr><th>Test</th><th>Areas</th><th>Original data</th><th>Expanded data</th><th>Paired observations</th></tr></thead>
          <tbody>{comparisons.map((r, i) => <tr key={`${r.test_mode}|${r.geography_role}|${i}`}>
            <th scope="row">{testLabel(r.test_mode)}</th><td>{roleLabel(r.geography_role)}</td><td>{number(r.old_balanced_mae)}</td><td>{number(r.expanded_balanced_mae)}</td><td>{count(r.paired)}{r.paired === 0 && <small>No paired observations</small>}</td>
          </tr>)}</tbody></table>
      </div>
      <a className="gre-download" href={data.result.download_url} download><ArrowDownToLine size={15} />Download geography comparison</a>
      {data.result.regions && <details className="gre-expansion-regions"><summary>All 96 area / day–night outcomes, including unavailable and reserved groups<ChevronRight className="gre-disclosure-chevron" size={16} aria-hidden="true" /></summary>
        <div className="gre-table-wrap" tabIndex={0} role="region" aria-label="All geographic expansion area and phase outcomes">
          <table><caption>Balanced MAE · °C · Missing is not zero</caption><thead><tr><th>Area / phase</th><th>Role</th><th>Original data</th><th>Expanded data</th><th>Dates</th><th>Paired observations</th><th>Outcome</th></tr></thead>
            <tbody>{data.result.regions.map(r => <tr key={`${r.region_id}|${r.phase}`} className={r.paired === 0 ? "gre-missing" : undefined}>
              <th scope="row">{r.region_id.replaceAll("_", " ")}<small>{r.phase === "day" ? "Day" : "Night"} · {testLabel(r.test_mode)}</small></th><td>{roleLabel(r.geography_role)}</td><td>{number(r.old_balanced_mae)}</td><td>{number(r.expanded_balanced_mae)}</td><td>{count(r.dates)}</td><td>{count(r.paired)}</td><td>{r.outcome.replaceAll("_", " ")}</td>
            </tr>)}</tbody></table>
        </div>
      </details>}
    </> : <div className="gre-waiting">{data.status === "complete" ? "The result receipt is not available in this snapshot." : "Results will appear after the experiment and its independent checks finish."}</div>}
  </section>;
}

export function GlobalResearchEvidence() {
  const [data, setData] = useState<Evidence | null>(null);
  const [error, setError] = useState("");
  const [reload, setReload] = useState(0);
  const [studyId, setStudyId] = useState("");
  const [phase, setPhase] = useState<"all" | "day" | "night">("all");
  const selectId = useId();
  useEffect(() => {
    const controller = new AbortController();
    setError(""); setData(null);
    fetch("/research/evidence-v2.json", { signal: controller.signal, cache: "no-cache", credentials: "same-origin" })
      .then(r => { if (!r.ok) throw new Error("Research evidence is currently unavailable."); return r.json(); })
      .then(readEvidence).then(d => { setData(d); setStudyId(d.studies.find(s => s.id === "control")?.id ?? d.studies[0].id); })
      .catch(e => { if (!controller.signal.aborted) setError(e instanceof Error ? e.message : "Research evidence could not be loaded."); });
    return () => controller.abort();
  }, [reload]);
  const chosen = data?.studies.find(s => s.id === studyId) ?? data?.studies[0];
  const rows = chosen?.regions.filter(r => phase === "all" || r.phase === phase) ?? [];
  const reset = () => { setPhase("all"); setStudyId(data?.studies.find(s => s.id === "control")?.id ?? data?.studies[0].id ?? ""); };

  return <section className="global-research-evidence" aria-labelledby="gre-title">
    <div className="gre-heading"><FlaskConical size={22} /><span className="gre-kicker">GLOBAL RESEARCH · SEPARATE FROM THE LIVE MAP</span></div>
    <h2 id="gre-title">What has actually improved?</h2>
    {!data ? <div className="gre-loading" role="status">{error || "Loading the checked research snapshot…"}{error && <button onClick={() => setReload(x => x + 1)}><RefreshCw size={14} />Try again</button>}</div> : <>
      <div className="gre-disclosure"><strong>{data.live_model_label}</strong><span>These research experiments start from a 31-input model, separate from the live 40-input model. They do not validate 100m temperatures or every weather condition.</span></div>
      <div className="gre-section-heading"><div><h3>{data.studies.length} fixed variants. The same original observations.</h3><p>{data.period_label} · Balanced MAE · Lower is better</p></div><span className="gre-status">{data.qualified ? "See qualification scope in source" : "Global accuracy target not met"}</span></div>
      <ComparisonChart data={data} />
      <p className="gre-footnote">{data.metric_definition} The dashed line marks the {data.target_mae_c}°C target; crossing it in one aggregate does not establish global accuracy.</p>
      <p className="gre-scope">{data.observation_scope}</p>
      <details className="gre-receipts"><summary>What changed, and the source receipts<ChevronRight className="gre-disclosure-chevron" size={16} aria-hidden="true" /></summary>
        {data.studies.map(s => <div key={s.id}><strong>{s.label}</strong><p>{s.change}</p><a className="gre-download" href={s.source.download_url} download><ArrowDownToLine size={14} />Download checked metrics</a>
          <dl><dt>Independent audit SHA-256</dt><dd>{s.source.audit_sha256}</dd><dt>Audited source metrics SHA-256</dt><dd>{s.source.metrics_sha256}</dd></dl></div>)}
      </details>
      <div className="gre-detail-heading"><div><h3>Every area, day and night</h3><p>Missing observations stay visible. Large errors and limited dates matter too.</p></div></div>
      <div className="gre-controls"><label htmlFor={selectId}>Variant<select id={selectId} value={chosen?.id ?? ""} onChange={e => setStudyId(e.target.value)}>{data.studies.map(s => <option key={s.id} value={s.id}>{s.label}</option>)}</select></label>
        <div className="gre-phase" role="group" aria-label="Time of day">{(["all", "day", "night"] as const).map(p => <button key={p} aria-pressed={phase === p} onClick={() => setPhase(p)}>{p === "all" ? "Day + night" : p === "day" ? "Day" : "Night"}</button>)}</div>
        <button className="gre-reset" onClick={reset}><RefreshCw size={13} />Reset</button>
      </div>
      <p className="gre-row-status" role="status">Showing {rows.length} of 24 area/phase groups · {rows.filter(r => r.paired === 0).length} without paired observations in this view</p>
      <div className="gre-table-wrap" tabIndex={0} role="region" aria-label="Research errors for every area and phase; scroll horizontally for all measures">
        <table><caption>{chosen?.label} · Errors in °C; percentages count paired native observations equally</caption><thead><tr><th>Area / phase</th><th>Balanced MAE</th><th>Ordinary MAE</th><th>Ordinary signed bias</th><th>Errors &gt;5°C</th><th>Errors &gt;7°C</th><th>Dates</th><th>Paired observations</th></tr></thead>
          <tbody>{rows.map(r => <tr key={regionKey(r)} className={r.paired === 0 ? "gre-missing" : undefined}>
            <th scope="row">{r.label}<small>{r.phase === "day" ? "Day" : "Night"}{r.test_mode === "reference" ? " · Separate reference" : " · Area held out"}</small></th>
            <td>{number(r.balanced_mae)}</td><td>{number(r.mae)}</td><td>{number(r.bias)}</td><td>{percent(r.above_5_fraction)}</td><td>{percent(r.above_7_fraction)}</td><td>{count(r.dates)}</td><td>{count(r.paired)}{r.paired === 0 && <small>No paired observations</small>}</td>
          </tr>)}</tbody></table>
      </div>
      <p className="gre-footnote">— means no reported score, never zero error. Native footprints can overlap and are not independent weather events. Cabauw remains a separate reference.</p>
      <Expansion data={data.expansion} />
      {data.solar_season && <SolarSeasonEvidence data={data.solar_season} />}
      <p className="gre-updated">Source snapshot: {data.generated_at_utc.replace("T", " ").replace(/\.\d+(Z|\+00:00)$/, " UTC").replace(/(Z|\+00:00)$/, " UTC")}</p>
    </>}
  </section>;
}
