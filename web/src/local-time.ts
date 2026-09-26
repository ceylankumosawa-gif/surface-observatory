/** Browser IANA time-zone rules; no host-local Date parsing or fixed UTC offsets.
 * Pilot zone names are from https://data.iana.org/time-zones/tzdb/zone.tab.
 * Solar geometry follows NOAA's Julian-century equations:
 * https://gml.noaa.gov/grad/solcalc/calcdetails.html . This is a UI guide;
 * the API makes the authoritative whole-area daylight check with SPA.
 */
export type Coordinate = [number, number];
export type LocalInstant = { utc: string; offsetSeconds: number };
export type LightPhase = "day" | "twilight" | "night";
export const PILOT_TIME_ZONES: Record<string, string> = {
  greater_london: "Europe/London",
  cabauw: "Europe/Amsterdam",
  darwin_howard_springs: "Australia/Darwin",
  gobabeb: "Africa/Windhoek",
  boulder: "America/Denver",
  sioux_falls: "America/Chicago",
  sodankyla: "Europe/Helsinki",
  utqiagvik: "America/Anchorage",
  singapore_johor: "Asia/Singapore",
  manaus_zf2: "America/Manaus",
  cape_town: "Africa/Johannesburg",
  lhasa: "Asia/Shanghai",
};
const MINUTE = 60_000;
const DAY = 86_400_000;
const formatters = new Map<string, Intl.DateTimeFormat>();
const pad = (n: number) => String(n).padStart(2, "0");
function formatter(zone: string) {
  if (!formatters.has(zone)) {
    formatters.set(zone, new Intl.DateTimeFormat("en-GB", {
      timeZone: zone, calendar: "iso8601", numberingSystem: "latn",
      year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit",
      minute: "2-digit", second: "2-digit", hourCycle: "h23",
    }));
  }
  return formatters.get(zone)!;
}
export function localParts(instant: number | string, zone: string) {
  const parts: Record<string, number> = {};
  for (const p of formatter(zone).formatToParts(new Date(instant))) {
    if (p.type !== "literal") parts[p.type] = Number(p.value);
  }
  return {
    year: parts.year, month: parts.month, day: parts.day,
    hour: parts.hour, minute: parts.minute, second: parts.second,
    date: `${parts.year}-${pad(parts.month)}-${pad(parts.day)}`,
    time: `${pad(parts.hour)}:${pad(parts.minute)}`,
  };
}
function naiveDate(date: string, time = "00:00") {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || !/^([01]\d|2[0-3]):[0-5]\d$/.test(time)) return NaN;
  const stamp = Date.parse(`${date}T${time}:00Z`);
  return Number.isFinite(stamp) && new Date(stamp).toISOString().slice(0, 16) === `${date}T${time}` ? stamp : NaN;
}
function offsetSeconds(stamp: number, zone: string) {
  const p = localParts(stamp, zone);
  const asUtc = Date.UTC(p.year, p.month - 1, p.day, p.hour, p.minute, p.second);
  return (asUtc - Math.floor(stamp / 1000) * 1000) / 1000;
}
/** Return every matching instant. Zero = a gap/invalid date; two = a fold.
 * Round-tripping each candidate prevents silently moving a skipped wall time.
 */
export function resolveLocal(date: string, time: string, zone: string): LocalInstant[] {
  const nominal = naiveDate(date, time);
  if (!Number.isFinite(nominal)) return [];
  const offsets = new Set<number>();
  for (let hours = -36; hours <= 36; hours += 6) {
    offsets.add(offsetSeconds(nominal + hours * 60 * MINUTE, zone));
  }
  return [...offsets].map((offset) => {
    const stamp = nominal - offset * 1000;
    const p = localParts(stamp, zone);
    return p.date === date && p.time === time && p.second === 0
      ? { utc: new Date(stamp).toISOString(), offsetSeconds: offset } : null;
  }).filter((p): p is LocalInstant => p !== null).sort((a, b) => a.utc.localeCompare(b.utc));
}
export function utcOffsetLabel(seconds: number) {
  const magnitude = Math.abs(seconds);
  const hours = Math.floor(magnitude / 3600);
  const minutes = Math.floor((magnitude % 3600) / 60);
  const remainder = magnitude % 60;
  return `UTC${seconds < 0 ? "−" : "+"}${pad(hours)}:${pad(minutes)}${remainder ? `:${pad(remainder)}` : ""}`;
}
export function formatLocal(instant: string, zone: string, seconds = false) {
  const p = localParts(instant, zone);
  return `${p.date} · ${p.time}${seconds ? `:${pad(p.second)}` : ""} · ${utcOffsetLabel(offsetSeconds(Date.parse(instant), zone))}`;
}
/** Find the actual start of a civil date, including midnight DST changes.
 * The current pilot zones have monotonic civil dates throughout 1900–2100.
 */
export function localDayBounds(date: string, zone: string) {
  const nominal = naiveDate(date);
  if (!Number.isFinite(nominal)) return null;
  const boundary = (target: string, centre: number) => {
    let lo = Math.floor((centre - 36 * 60 * MINUTE) / 1000);
    let hi = Math.ceil((centre + 36 * 60 * MINUTE) / 1000);
    while (lo < hi) {
      const middle = Math.floor((lo + hi) / 2);
      if (localParts(middle * 1000, zone).date < target) lo = middle + 1;
      else hi = middle;
    }
    return lo * 1000;
  };
  const start = boundary(date, nominal);
  if (localParts(start, zone).date !== date) return null;
  const next = new Date(nominal + DAY).toISOString().slice(0, 10);
  const end = boundary(next, nominal + DAY);
  return { start, end, durationHours: (end - start) / (60 * MINUTE) };
}
const rad = Math.PI / 180;
const mod = (n: number, d: number) => ((n % d) + d) % d;
/** Geometric elevation, degrees. Atmospheric refraction and terrain are excluded. */
export function solarElevation(instant: number | string, [longitude, latitude]: Coordinate) {
  const stamp = typeof instant === "number" ? instant : Date.parse(instant);
  const t = (stamp / DAY + 2440587.5 - 2451545) / 36525;
  const l = mod(280.46646 + t * (36000.76983 + 0.0003032 * t), 360);
  const m = 357.52911 + t * (35999.05029 - 0.0001537 * t);
  const e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t);
  const c = Math.sin(m * rad) * (1.914602 - t * (0.004817 + 0.000014 * t))
    + Math.sin(2 * m * rad) * (0.019993 - 0.000101 * t) + 0.000289 * Math.sin(3 * m * rad);
  const omega = 125.04 - 1934.136 * t;
  const lambda = l + c - 0.00569 - 0.00478 * Math.sin(omega * rad);
  const epsilon = 23 + (26 + (21.448 - t * (46.815 + t * (0.00059 - 0.001813 * t))) / 60) / 60
    + 0.00256 * Math.cos(omega * rad);
  const declination = Math.asin(Math.sin(epsilon * rad) * Math.sin(lambda * rad));
  const y = Math.tan(epsilon * rad / 2) ** 2;
  const equation = 4 / rad * (y * Math.sin(2 * l * rad) - 2 * e * Math.sin(m * rad)
    + 4 * e * y * Math.sin(m * rad) * Math.cos(2 * l * rad)
    - 0.5 * y * y * Math.sin(4 * l * rad) - 1.25 * e * e * Math.sin(2 * m * rad));
  const utcMinutes = mod(stamp, DAY) / MINUTE;
  const hourAngle = (mod(utcMinutes + equation + 4 * longitude, 1440) / 4 - 180) * rad;
  const sine = Math.sin(latitude * rad) * Math.sin(declination)
    + Math.cos(latitude * rad) * Math.cos(declination) * Math.cos(hourAngle);
  return Math.asin(Math.max(-1, Math.min(1, sine))) / rad;
}
export function lightPhase(elevation: number): LightPhase {
  return elevation > 0 ? "day" : elevation >= -6 ? "twilight" : "night";
}
export function selectionCentre(points: Coordinate[], fallback: Coordinate): Coordinate {
  if (points.length < 3) return fallback;
  let cross = 0, x = 0, y = 0;
  for (let i = 0; i < points.length; i++) {
    const p = points[i], q = points[(i + 1) % points.length];
    const weight = p[0] * q[1] - q[0] * p[1];
    cross += weight; x += (p[0] + q[0]) * weight; y += (p[1] + q[1]) * weight;
  }
  return Math.abs(cross) > 1e-12 ? [x / (3 * cross), y / (3 * cross)] : fallback;
}
export function daylightGuide(instant: string, points: Coordinate[], fallback: Coordinate) {
  const xs = points.map((p) => p[0]), ys = points.map((p) => p[1]);
  const centre: Coordinate = points.length >= 3
    ? [(Math.min(...xs) + Math.max(...xs)) / 2, (Math.min(...ys) + Math.max(...ys)) / 2] : fallback;
  const margin = points.length >= 3
    ? (Math.max(...xs) - Math.min(...xs) + Math.max(...ys) - Math.min(...ys)) / 2 + .001 : .001;
  // Only definite night is disabled here. Near the horizon, precise SPA in
  // the API decides; approximate browser geometry must not reject valid hours.
  const elevation = solarElevation(instant, centre);
  return { elevation, definitelyNight: elevation < -.5, nearHorizon: elevation <= margin + .5 };
}
export function solarDay(date: string, zone: string, centre: Coordinate) {
  const bounds = localDayBounds(date, zone);
  if (!bounds) return null;
  const count = Math.ceil((bounds.end - bounds.start) / (10 * MINUTE));
  const samples = Array.from({ length: count + 1 }, (_, i) => {
    const stamp = bounds.start + (bounds.end - bounds.start) * i / count;
    const elevation = solarElevation(stamp, centre);
    return { stamp, position: i / count, elevation, phase: lightPhase(elevation) };
  });
  const events: { kind: "sunrise" | "sunset"; stamp: number; position: number }[] = [];
  for (let i = 1; i < samples.length; i++) {
    const a = samples[i - 1], b = samples[i];
    if ((a.elevation > -.833) === (b.elevation > -.833)) continue;
    let lo = a.stamp, hi = b.stamp;
    const rising = b.elevation > a.elevation;
    while (hi - lo > 1000) {
      const middle = (hi + lo) / 2;
      if ((solarElevation(middle, centre) > -.833) === rising) hi = middle;
      else lo = middle;
    }
    const stamp = (hi + lo) / 2;
    events.push({ kind: rising ? "sunrise" : "sunset", stamp, position: (stamp - bounds.start) / (bounds.end - bounds.start) });
  }
  const ticks = [0, 6, 12, 18].flatMap((hour) => resolveLocal(date, `${pad(hour)}:00`, zone)
    .map((instant) => ({ label: `${pad(hour)}:00`, position: (Date.parse(instant.utc) - bounds.start) / (bounds.end - bounds.start) })))
    .concat([{ label: "24:00", position: 1 }]);
  return { ...bounds, samples, events, ticks };
}
