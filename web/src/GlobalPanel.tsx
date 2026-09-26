import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowUpRight, Check, Download, Info, Layers3, LoaderCircle, MapPin, Moon, Pentagon, Sun, X } from "lucide-react";
import { LocalTimePicker } from "./LocalTimePicker";
import { formatLocal, localParts } from "./local-time";
import { ACTIVE_STATUSES, SUCCESS_STATUSES, dateBound, finiteNumber, globalFile, globalFormat as fmt,
  localCandidates, nearestEarlierStep, progressDescription, requestProblem, selectionEstimate, validZone,
  type AreaPolygon, type GlobalCapabilities, type GlobalJob, type Point } from "./global-explorer";

type Props = {
  polygon: AreaPolygon | null; drawing: boolean; vertices: number; mapReady: boolean; mapCentre: Point;
  job: GlobalJob | null; error: string; submitting: boolean;
  onJob: (job: GlobalJob | null) => void; onSubmitting: (busy: boolean) => void;
  onError: (error: string) => void; onDraw: () => void; onFinish: () => void; onClear: () => void;
  onBox: (centre?: Point) => void; onLocate: (centre: Point) => void;
};
async function readJson(path: string, init?: RequestInit) {
  const response = await fetch(path, init);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : data.message || "The global service could not complete this request.");
  return data;
}
const zones = (() => {
  try { return ["UTC", ...(Intl as any).supportedValuesOf("timeZone")] as string[]; }
  catch { return ["UTC", "Europe/London", "America/New_York", "America/Chicago", "America/Los_Angeles", "Asia/Kolkata", "Asia/Shanghai", "Asia/Tokyo", "Australia/Sydney"]; }
})();

export function GlobalPanel(props: Props) {
  const [cap, setCap] = useState<GlobalCapabilities | null>(null);
  const [capError, setCapError] = useState("");
  const [lookup, setLookup] = useState<{ key: string; zone: string }>({ key: "", zone: "" });
  const [manualZone, setManualZone] = useState("");
  const [date, setDate] = useState("2023-06-21"), [hour, setHour] = useState("12:00"), [occurrence, setOccurrence] = useState("");
  const [resolution, setResolution] = useState(100);
  const [latitude, setLatitude] = useState("51.75"), [longitude, setLongitude] = useState("-1.25");
  const [refresh, setRefresh] = useState(0);
  const restoredJob = useRef("");
  const busy = props.submitting || ACTIVE_STATUSES.includes(props.job?.status || "");
  const done = SUCCESS_STATUSES.includes(props.job?.status || "");
  const estimate = selectionEstimate(props.polygon, resolution);
  const centre = estimate?.centre || props.mapCentre;
  const lookupKey = centre.map(x => x.toFixed(5)).join(",");
  const zone = manualZone || (lookup.key === lookupKey ? lookup.zone : "");
  const zoneValid = validZone(zone);
  const candidates = useMemo(() => localCandidates(date, hour, zone), [date, hour, zone]);
  const selected = candidates.length === 1 ? candidates[0] : candidates.find(x => x.utc === occurrence);
  const problem = requestProblem(cap, props.polygon, resolution, selected?.utc);
  const supportedStep = selected && cap ? nearestEarlierStep(selected.utc, cap.date_range.time_step_minutes) : null;
  const result = props.job?.result;
  const temperatures = result?.summary?.predicted_lst_c || {};
  const counts = result?.counts || {};
  useEffect(() => { if (props.job && !ACTIVE_STATUSES.includes(props.job.status)) setRefresh(n => n + 1); }, [props.job?.status]);
  useEffect(() => {
    if (!props.job?.request?.datetime_utc || !zoneValid || restoredJob.current === props.job.id) return;
    const local = localParts(props.job.request.datetime_utc, zone);
    setDate(local.date); setHour(local.time); setOccurrence(new Date(props.job.request.datetime_utc).toISOString());
    if (props.job.request.resolution_m) setResolution(props.job.request.resolution_m);
    restoredJob.current = props.job.id;
  }, [props.job?.id, zoneValid, zone]);

  useEffect(() => {
    const controller = new AbortController();
    setCapError("");
    const timer = window.setTimeout(() => {
      const query = new URLSearchParams({ lon: String(centre[0]), lat: String(centre[1]) });
      readJson(`/api/global/capabilities?${query}`, { signal: controller.signal }).then((data: GlobalCapabilities) => {
        if (controller.signal.aborted) return;
        if (!data?.date_range?.min || !data?.limits || !Array.isArray(data.resolutions_m)) throw new Error("Global capability details are incomplete.");
        setCap(data);
        setLookup({ key: lookupKey, zone: data.time_zone && validZone(data.time_zone) ? data.time_zone : "" });
        setResolution(current => data.resolutions_m.includes(current) ? current : data.resolutions_m[0]);
      }).catch(error => { if (!controller.signal.aborted) setCapError(error.message); });
    }, 200);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [lookupKey, refresh]);

  function change() { props.onJob(null); props.onError(""); setOccurrence(""); }
  function chooseInstant(utc: string) {
    if (!zoneValid) return;
    const local = localParts(utc, zone);
    setDate(local.date); setHour(local.time); setOccurrence(utc); props.onJob(null); props.onError("");
  }
  async function generate() {
    if (problem || busy || props.drawing || !selected) return;
    props.onSubmitting(true); props.onError("");
    try {
      const created = await readJson("/api/global/jobs", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ polygon: props.polygon, datetime_utc: selected.utc, resolution_m: resolution }) });
      props.onJob(created);
      const link = new URL(location.href); link.searchParams.set("global_job", created.id); history.replaceState(null, "", link);
      setRefresh(n => n + 1);
    } catch (error: any) { props.onError(error.message); setRefresh(n => n + 1); }
    finally { props.onSubmitting(false); }
  }
  const maxArea = cap?.limits.max_area_km2_by_resolution[String(resolution)];
  const timedOutMinutes = cap ? Math.round(cap.limits.job_timeout_seconds / 60) : null;
  const jobProgress = props.job?.progress;
  const numericProgress = finiteNumber(jobProgress) ? Math.max(0, Math.min(100, jobProgress <= 1 ? jobProgress * 100 : jobProgress)) : null;

  return <div className="global-controls">
    <div className="global-disclosure"><FlaskIcon /><div><strong>Global experimental</strong><p>Estimate beyond the pilots. Accuracy of 3°C MAE across regions has not been demonstrated.</p></div></div>
    <section>
      <div className="step-title"><span>01</span><h2>Choose any area</h2><span className="quiet">Map selection</span></div>
      <p className="helper">Pan or zoom to a place, then draw a compact area. Missing source coverage stays transparent.</p>
      <div className="button-row area-actions"><button className={`secondary ${props.drawing ? "selected" : ""}`} disabled={busy || !props.mapReady} onClick={props.onDraw}><Pentagon size={15} />{props.drawing ? "Drawing…" : "Draw an area"}</button>
        <button className="secondary" disabled={busy || !props.mapReady} onClick={() => props.onBox()}><Layers3 size={15} />5 km box here</button></div>
      <button className="quick-area" disabled={busy || !props.mapReady} onClick={() => { props.onBox([-1.25, 51.75]); setManualZone(""); }}>Try Oxford · outside the pilots <ArrowUpRight size={13} /></button>
      <details className="global-coordinate-search"><summary><MapPin size={14} />Go to coordinates</summary><div className="date-row"><div><label htmlFor="global-latitude">Latitude</label><input id="global-latitude" type="number" min="-90" max="90" step="any" value={latitude} onChange={e => setLatitude(e.target.value)} disabled={busy} /></div><div><label htmlFor="global-longitude">Longitude</label><input id="global-longitude" type="number" min="-180" max="180" step="any" value={longitude} onChange={e => setLongitude(e.target.value)} disabled={busy} /></div></div><button className="secondary" disabled={busy || !latitude.trim() || !longitude.trim() || !Number.isFinite(Number(latitude)) || !Number.isFinite(Number(longitude)) || Math.abs(Number(latitude)) > 90 || Math.abs(Number(longitude)) > 180} onClick={() => props.onLocate([Number(longitude), Number(latitude)])}>Show location</button></details>
      {props.drawing ? <div className="draw-hint"><span>{props.vertices} corners selected</span><button onClick={props.onFinish} disabled={props.vertices < 3}>Finish area <Check size={14} /></button></div>
        : props.polygon && <div className={`area-line ${problem.includes("area") || problem.includes("cells") ? "invalid" : ""}`}><span><span className="selection-dot" />≈ {fmt(estimate?.areaKm2)} km² bounding area</span><button aria-label="Clear global selected area" disabled={busy} onClick={props.onClear}><X size={14} /></button></div>}
      {cap?.spatial?.description && <p className="helper small">{cap.spatial.description}</p>}
    </section>
    <section>
      <div className="step-title"><span>02</span><h2>Choose a local moment</h2><span className="quiet">Day or night</span></div>
      <label htmlFor="global-zone">Time zone for this area</label>
      <input id="global-zone" list="global-time-zones" value={manualZone || zone} placeholder="Choose an IANA time zone" onChange={e => { setManualZone(e.target.value); change(); }} disabled={busy} autoComplete="off" />
      <datalist id="global-time-zones">{zones.map(value => <option key={value} value={value} />)}</datalist>
      <p className="helper small">{manualZone ? "Using your chosen clock convention." : zone ? props.polygon ? "Detected at the selected area’s centre." : "Detected at the map centre; draw an area to confirm." : props.polygon ? "Finding the local clock; choose a zone if none is found." : "Draw an area to detect its local clock."} Areas can cross time-zone boundaries.</p>
      {manualZone && <button className="text-action" disabled={busy} onClick={() => { setManualZone(""); change(); }}>Use area-centre time zone</button>}
      {zoneValid ? <>
        <div className="global-phase-buttons"><button className="secondary" disabled={busy} onClick={() => { setHour("12:00"); change(); }}><Sun size={14} />Local noon</button><button className="secondary" disabled={busy} onClick={() => { setHour("00:00"); change(); }}><Moon size={14} />Local midnight</button></div>
        <LocalTimePicker date={date} hour={hour} zone={zone} centre={centre} candidates={candidates} selected={selected} disabled={busy} hasArea={!!props.polygon}
          minDate={cap ? localParts(dateBound(cap.date_range.min), zone).date : undefined} maxDate={cap ? localParts(dateBound(cap.date_range.max, true), zone).date : undefined}
          inputIdPrefix="global-" referenceLabel="At the map centre" supportDescription="Sun times are approximate. Day and night use the same frozen research model; twilight or missing inputs may leave gaps. No cast-shadow geometry is modelled."
          onDate={value => { setDate(value); change(); }} onHour={value => { setHour(value); change(); }} onInstant={chooseInstant} />
        {supportedStep && selected?.utc !== supportedStep && <div className="info-note amber"><Info size={15} /><span>Weather uses whole UTC hours. <button className="text-action" disabled={busy} onClick={() => chooseInstant(supportedStep)}>Use {localParts(supportedStep, zone).time} local ({formatLocal(supportedStep, zone).split(" · ").at(-1)})</button></span></div>}
      </> : <p className="helper small">Choose a valid time-zone name, for example Europe/London or Asia/Kolkata. No clock time is guessed.</p>}
      {cap && <div className="global-available"><strong>Request date range</strong><span>{new Date(dateBound(cap.date_range.min)).toISOString().slice(0, 16).replace("T", " ")} to {new Date(dateBound(cap.date_range.max, true)).toISOString().slice(0, 16).replace("T", " ")} UTC</span><small>Some areas or hours may lack required source data.</small><small>{cap.date_range.description || "Available times follow the service’s verified source coverage."}</small><button className="text-action" disabled={busy || !zoneValid} onClick={() => chooseInstant(nearestEarlierStep(new Date(dateBound(cap.date_range.max, true)).toISOString(), cap.date_range.time_step_minutes))}>Use latest available time</button></div>}
    </section>
    <section>
      <div className="step-title"><span>03</span><h2>Choose output cell size</h2></div>
      <div className="global-resolution" role="group" aria-label="Output cell size">{(cap?.resolutions_m || [100, 250, 500, 1000]).map(value => <button key={value} className={resolution === value ? "selected" : ""} onClick={() => { setResolution(value); change(); }} disabled={busy || !cap}>{value === 1000 ? "1 km" : `${value} m`}</button>)}</div>
      <p className="helper small">Cell size describes the output grid. Weather inputs remain coarser; finer cells do not establish greater accuracy.</p>
      {cap?.resolution_description && <p className="helper small">{cap.resolution_description}</p>}
      <dl className="global-limits"><div><dt>Estimated output cells</dt><dd>{fmt(estimate?.outputPixels, 0)}</dd></div><div><dt>Area limit at {resolution} m</dt><dd>{fmt(maxArea, 0)} km²</dd></div><div><dt>Output cell limit</dt><dd>{fmt(cap?.limits.max_output_pixels, 0)}</dd></div><div><dt>Source tiles per job</dt><dd>{fmt(cap?.limits.max_source_tiles, 0)}</dd></div><div><dt>Pending jobs / limit</dt><dd>{fmt(cap?.queue?.pending, 0)} / {fmt(cap?.limits.max_pending, 0)}</dd></div><div><dt>Task time limit</dt><dd>{timedOutMinutes === null ? "—" : timedOutMinutes} min</dd></div></dl>
      {(finiteNumber(cap?.limits.hourly_per_connection) || finiteNumber(cap?.limits.new_jobs_per_day) || finiteNumber(cap?.limits.source_tiles_per_day)) && <p className="helper small">{finiteNumber(cap?.limits.hourly_per_connection) && <>{cap.limits.hourly_per_connection} new jobs per connection per hour · </>}{finiteNumber(cap?.limits.new_jobs_per_day) && <>{cap.limits.new_jobs_per_day} new jobs per day across the service · </>}{finiteNumber(cap?.limits.source_tiles_per_day) && <>{cap.limits.source_tiles_per_day} source-tile reservations per day</>}</p>}
      {finiteNumber(cap?.limits.output_retention_days) && <p className="helper small">Saved output files are kept for {cap.limits.output_retention_days} days. Download results you need to retain.</p>}
      <button className="text-action" disabled={busy} onClick={() => setRefresh(n => n + 1)}>Refresh availability and queue</button>
      <p className="helper small">Estimates use the bounding area. The server checks the exact grid and source-tile work before accepting the job.</p>
    </section>
    <section className="run-section">
      <button className="generate" disabled={!!problem || busy || props.drawing || !props.mapReady || !!capError} onClick={generate}>{busy ? <LoaderCircle className="spin" size={18} /> : <Layers3 size={18} />}<span>{busy ? "Preparing experimental raster…" : "Generate experimental raster"}</span>{!busy && <ArrowUpRight size={18} />}</button>
      {!busy && problem && <p className="helper small">{problem}</p>}
      {capError && <div className="error" role="alert"><Info size={16} /><span>{capError}<button className="text-action" onClick={() => setRefresh(n => n + 1)}>Retry availability check</button></span></div>}
      {busy && props.job && <div className="global-progress" role="status"><div><LoaderCircle className="spin" size={14} /><span>{progressDescription(props.job)}</span></div>{numericProgress !== null && <progress max="100" value={numericProgress} />}<p className="helper small">Job {props.job.id.slice(0, 12)} · source tiles are cached for reuse.</p></div>}
      {(props.error || props.job?.error) && <div className="error" role="alert"><Info size={16} /><span>{props.error || props.job?.error}</span></div>}
      {props.job?.status === "expired" && <div className="info-note amber" role="status"><Info size={15} /><span>Saved result files have expired. Submit the request again to rebuild them.</span></div>}
      {props.job?.status === "failed" && !props.job.error && <p className="helper">This request could not be completed. Review the area and source dates before trying again.</p>}
    </section>
    {done && result && <section className="result-panel global-result">
      <div className="step-title"><span className="success"><Check size={13} /></span><h2>Experimental result</h2><span className="quiet">{result.grid?.resolution_m ?? resolution} m</span></div>
      <div className="result-stats"><div><strong>{fmt(temperatures.mean)}{finiteNumber(temperatures.mean) ? "°C" : ""}</strong><span>Mean predicted surface temp.</span></div><div><strong>{fmt(counts.predicted_pixels ?? counts.valid_pixels ?? counts.predicted, 0)}</strong><span>Supported output cells</span></div></div>
      <p className="helper">{fmt(temperatures.min)} to {fmt(temperatures.max)} °C<br />{props.job?.request?.datetime_utc && <>{zoneValid ? formatLocal(props.job.request.datetime_utc, zone) : props.job.request.datetime_utc}<br /></>}Transparent cells are missing or unsupported; they are not 0°C.</p>
      <div className="info-note amber"><Info size={15} /><span>Global extrapolation · no validated uncertainty interval. The 3°C regional accuracy target has not been achieved. Night training covers only London and Sioux Falls.</span></div>
      <div className="download-row">{globalFile(result, "prediction_tif") && <a download href={globalFile(result, "prediction_tif")!}><Download size={14} />GeoTIFF</a>}{globalFile(result, "quality_tif") && <a download href={globalFile(result, "quality_tif")!}>Coverage mask</a>}{globalFile(result, "provenance_json") && <a href={globalFile(result, "provenance_json")!} target="_blank" rel="noreferrer">Sources <ArrowUpRight size={13} /></a>}</div>
      <details><summary>Coverage and data used</summary><dl className="global-limits">{Object.entries(counts).filter(([, value]) => finiteNumber(value)).map(([key, value]) => <div key={key}><dt>{key.replaceAll("_", " ")}</dt><dd>{fmt(value, 0)}</dd></div>)}</dl>{result.grid?.grid_note && <p className="helper small">{result.grid.grid_note}</p>}<ul className="notes">{(result.sources || []).map((source: unknown, i: number) => <li key={i}>{typeof source === "string" ? source : JSON.stringify(source)}</li>)}</ul></details>
      {result.warnings?.length > 0 && <details open><summary>Coverage and quality notes</summary><ul className="notes">{result.warnings.map((warning: unknown, i: number) => <li key={i}>{typeof warning === "string" ? warning : JSON.stringify(warning)}</li>)}</ul></details>}
    </section>}
  </div>;
}
function FlaskIcon() { return <Info size={18} />; }
