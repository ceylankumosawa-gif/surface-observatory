import { resolveLocal } from "./local-time.ts";

export type Point = [number, number];
export type AreaPolygon = { type: "Polygon"; coordinates: Point[][] };
export type GlobalCapabilities = {
  status: string;
  resolutions_m: number[];
  date_range: { min: string; max: string; timezone: string; time_step_minutes: number;
    current_conditions_available?: boolean; description?: string };
  limits: { max_area_km2_by_resolution: Record<string, number>; max_output_pixels: number;
    max_source_tiles: number; max_pending: number; job_timeout_seconds: number; max_vertices?: number;
    hourly_per_connection?: number; new_jobs_per_day?: number; source_tiles_per_day?: number; output_retention_days?: number };
  accuracy: { target_mae_c: number; qualified: boolean; description?: string };
  spatial: { description?: string };
  queue?: { pending?: number };
  time_zone?: string | null;
  time_zone_source?: string;
  warnings?: string[];
  resolution_description?: string;
};
export type GlobalJob = { id: string; status: string; stage?: string; progress?: unknown; message?: string;
  result?: any; error?: string; request?: { datetime_utc?: string; resolution_m?: number; polygon?: AreaPolygon }; cached?: boolean };

export const ACTIVE_STATUSES = ["queued", "running", "pending"];
export const SUCCESS_STATUSES = ["completed", "complete", "succeeded"];
export const finiteNumber = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);
export const globalFormat = (value: unknown, digits = 1) => finiteNumber(value)
  ? value.toLocaleString("en-GB", { maximumFractionDigits: digits, minimumFractionDigits: digits }) : "—";

export function validZone(zone: string) {
  try { new Intl.DateTimeFormat("en-GB", { timeZone: zone }).format(); return zone.length > 0; } catch { return false; }
}
export function localCandidates(date: string, hour: string, zone: string) {
  return validZone(zone) ? resolveLocal(date, hour, zone) : [];
}
export function dateBound(value: string, end = false) {
  return Date.parse(/^\d{4}-\d{2}-\d{2}$/.test(value) ? `${value}T${end ? "23:59:59.999" : "00:00:00.000"}Z` : value);
}
/** Spherical bounding rectangle estimate; backend projected-grid limits are authoritative. */
export function selectionEstimate(polygon: AreaPolygon | null, resolution: number) {
  if (!polygon) return null;
  const ring = polygon.coordinates[0];
  if (!ring || ring.length < 4 || ring.some(p => p.length !== 2 || !p.every(Number.isFinite))) return null;
  const xs = ring.map(p => p[0]), ys = ring.map(p => p[1]);
  const west = Math.min(...xs), east = Math.max(...xs), south = Math.min(...ys), north = Math.max(...ys);
  if (east - west > 180 || west < -180 || east > 180 || south < -90 || north > 90) return null;
  const r = Math.PI / 180;
  const areaKm2 = 6371.0088 ** 2 * (east - west) * r * (Math.sin(north * r) - Math.sin(south * r));
  return { areaKm2, outputPixels: Math.ceil(areaKm2 * 1e6 / (resolution ** 2)), centre: [(west + east) / 2, (south + north) / 2] as Point };
}
export function requestProblem(cap: GlobalCapabilities | null, polygon: AreaPolygon | null, resolution: number, utc?: string) {
  if (!cap) return "Service availability is still loading.";
  if (finiteNumber(cap.queue?.pending) && cap.queue!.pending! >= cap.limits.max_pending) return "The request queue is full. Refresh availability in a moment.";
  if (!polygon) return "Draw an area or choose a small box on the map.";
  const estimate = selectionEstimate(polygon, resolution);
  if (!estimate || estimate.areaKm2 <= 0) return "Choose a valid compact area. Split selections that cross the date line.";
  if (!cap.resolutions_m.includes(resolution)) return "Choose an available output cell size.";
  if (estimate.areaKm2 > cap.limits.max_area_km2_by_resolution[String(resolution)]) return "This area exceeds the limit for the selected cell size.";
  if (estimate.outputPixels > cap.limits.max_output_pixels) return "This selection has too many output cells. Draw a smaller area or choose larger cells.";
  if (!utc) return "Choose a valid local time and resolve any clock change.";
  const stamp = Date.parse(utc);
  if (!Number.isFinite(stamp) || stamp < dateBound(cap.date_range.min) || stamp > dateBound(cap.date_range.max, true)) return "This time is outside the currently available source dates.";
  const step = cap.date_range.time_step_minutes * 60_000;
  if (!(step > 0) || stamp % step !== 0) return `Choose a supported weather time (${cap.date_range.time_step_minutes}-minute UTC steps).`;
  return "";
}
export function nearestEarlierStep(utc: string, stepMinutes: number) {
  return new Date(Math.floor(Date.parse(utc) / (stepMinutes * 60_000)) * stepMinutes * 60_000).toISOString();
}
export function globalFile(result: any, key: string): string | null {
  const file = result?.files?.[key];
  const url = typeof file === "string" ? file : file?.url;
  return typeof url === "string" && url.startsWith("/") && !url.startsWith("//") ? url : null;
}
export function normalizeGlobalResult(result: any) {
  if (!result) return result;
  const temperatures = result.summary?.predicted_lst_c || {};
  return { ...result, mode: "global", legends: { ...result.legends, prediction: result.legends?.temperature },
    summary: { ...result.summary, min_c: temperatures.min, max_c: temperatures.max, mean_c: temperatures.mean } };
}
export function progressDescription(job: GlobalJob) {
  const names: Record<string, string> = { queued: "Waiting for a worker", validate: "Checking area and source dates",
    preparing: "Preparing source tiles", optical: "Reading past optical observations", surfaces: "Preparing surface features",
    weather: "Joining weather for the requested time", stations: "Checking nearby station reports",
    radiation: "Reading radiation and snow", inference: "Estimating surface temperature", tiles: "Preparing canonical tiles",
    mosaic: "Combining tiles", aggregate: "Preparing the selected cell size", render: "Writing raster and preview", export: "Writing downloads" };
  return job.message || names[job.stage || job.status] || (job.stage || job.status).replaceAll("_", " ");
}
