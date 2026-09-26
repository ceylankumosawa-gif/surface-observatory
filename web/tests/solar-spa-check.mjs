// Run on Hetzner from /opt/lst-pilot/web; pvlib provides an independent reference.
import { execFileSync } from "node:child_process";
import assert from "node:assert/strict";
import { solarDay, solarElevation } from "../src/local-time.ts";

const sites = [
  ["London", "Europe/London", [-.088637, 51.488962], "2026-09-08"],
  ["Sioux Falls", "America/Chicago", [-96.73, 43.68], "2026-09-08"],
  ["Utqiagvik summer", "America/Anchorage", [-156.61, 71.25], "2026-06-21"],
  ["Utqiagvik winter", "America/Anchorage", [-156.61, 71.25], "2026-12-21"],
  ["Sodankyla summer", "Europe/Helsinki", [26.64, 67.36], "2026-06-21"],
  ["Sodankyla winter", "Europe/Helsinki", [26.64, 67.36], "2026-12-21"],
];
const cases = sites.map(([name, zone, point, date]) => {
  const day = solarDay(date, zone, point);
  const instants = [...day.samples.map((sample) => sample.stamp), ...day.events.map((event) => event.stamp)];
  return { name, point, events: day.events, instants, browser: instants.map((stamp) => solarElevation(stamp, point)) };
});
const reference = JSON.parse(execFileSync(process.env.LST_TEST_PYTHON || "python", ["-c", `
import json, sys
import pandas as pd
from pvlib.solarposition import spa_python
cases=json.load(sys.stdin)
print(json.dumps([spa_python(pd.to_datetime(case['instants'],unit='ms',utc=True),latitude=case['point'][1],longitude=case['point'][0]).elevation.tolist() for case in cases]))
`], { input: JSON.stringify(cases), encoding: "utf8" }));
const report = cases.map((sample, i) => {
  const errors = sample.browser.map((value, j) => Math.abs(value - reference[i][j]));
  const maximum = Math.max(...errors);
  assert.ok(maximum < .05, `${sample.name} browser/SPA elevation error ${maximum} degrees`);
  const eventErrors = sample.events.map((event, j) => Math.abs(reference[i][sample.instants.length - sample.events.length + j] + .833));
  assert.ok(eventErrors.every((error) => error < .05), `${sample.name} sunrise/sunset differs from SPA`);
  return { site: sample.name, checked_instants: errors.length, max_elevation_error_degrees: maximum, event_times_utc: sample.events.map((event) => ({ kind: event.kind, utc: new Date(event.stamp).toISOString() })), max_event_elevation_error_degrees: eventErrors.length ? Math.max(...eventErrors) : null };
});
console.log(JSON.stringify(report, null, 2));
