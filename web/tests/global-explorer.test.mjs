import test from "node:test";
import assert from "node:assert/strict";
import { dateBound, finiteNumber, globalFormat, validZone, localCandidates, selectionEstimate,
  requestProblem, nearestEarlierStep, globalFile, normalizeGlobalResult, progressDescription } from "../src/global-explorer.ts";
import { localParts } from "../src/local-time.ts";

const cap = { status: "experimental", resolutions_m: [100, 250, 500, 1000],
  date_range: { min: "2021-01-01", max: "2023-06-30", timezone: "UTC", time_step_minutes: 60 },
  limits: { max_area_km2_by_resolution: { 100: 6400, 250: 14400, 500: 25600, 1000: 25600 }, max_output_pixels: 800000, max_source_tiles: 16, max_pending: 3, job_timeout_seconds: 1800 },
  accuracy: { target_mae_c: 3, qualified: false }, spatial: {}, queue: { pending: 0 } };
const polygon = { type: "Polygon", coordinates: [[[-1.3, 51.7], [-1.2, 51.7], [-1.2, 51.8], [-1.3, 51.8], [-1.3, 51.7]]] };
test("global missing values never become zero temperatures", () => {
  for (const value of [null, undefined, "0", false, NaN, Infinity]) { assert.equal(finiteNumber(value), false); assert.equal(globalFormat(value), "—"); }
  assert.equal(globalFormat(0), "0.0"); assert.equal(globalFormat(-2.3), "-2.3");
  const normalized = normalizeGlobalResult({ summary: { predicted_lst_c: { mean: null, min: null, max: null } }, legends: { temperature: { min_c: 0, max_c: 5 } } });
  assert.equal(normalized.summary.mean_c, null); assert.equal(normalized.legends.prediction.min_c, 0);
});
test("arbitrary IANA zones retain gaps and repeated civil times", () => {
  assert.equal(validZone("Not/AZone"), false); assert.deepEqual(localCandidates("2023-01-01", "12:00", "Not/AZone"), []);
  assert.equal(localCandidates("2023-03-26", "01:30", "Europe/London").length, 0);
  assert.equal(localCandidates("2023-10-29", "01:30", "Europe/London").length, 2);
});
test("half-hour-zone alignment is explicit and preserves the chosen clock convention", () => {
  const value = localCandidates("2023-06-21", "12:00", "Asia/Kolkata")[0].utc;
  assert.equal(value, "2023-06-21T06:30:00.000Z");
  assert.match(requestProblem(cap, polygon, 100, value), /UTC steps/);
  const rounded = nearestEarlierStep(value, 60);
  assert.equal(rounded, "2023-06-21T06:00:00.000Z"); assert.equal(localParts(rounded, "Asia/Kolkata").time, "11:30");
  assert.equal(requestProblem(cap, polygon, 100, rounded), "");
});
test("server date bounds and resolutions control admission", () => {
  assert.equal(dateBound("2023-06-30", true), Date.parse("2023-06-30T23:59:59.999Z"));
  assert.equal(requestProblem(cap, polygon, 250, "2023-06-30T23:00:00Z"), "");
  assert.match(requestProblem(cap, polygon, 250, "2023-07-01T00:00:00Z"), /outside/);
  assert.match(requestProblem(cap, polygon, 50, "2023-06-21T12:00:00Z"), /cell size/);
  assert.match(requestProblem(cap, polygon, 100, undefined), /local time/);
});
test("bounding area is positive independent of drawing direction and pixels scale with resolution", () => {
  const forward = selectionEstimate(polygon, 100);
  const reverse = selectionEstimate({ ...polygon, coordinates: [[...polygon.coordinates[0]].reverse()] }, 100);
  assert.ok(forward.areaKm2 > 70 && forward.areaKm2 < 80); assert.deepEqual(forward, reverse);
  assert.ok(selectionEstimate(polygon, 1000).outputPixels < forward.outputPixels / 90);
  assert.deepEqual(forward.centre, [-1.25, 51.75]);
});
test("unsupported or degenerate selections do not yield plausible requests", () => {
  assert.equal(selectionEstimate({ type: "Polygon", coordinates: [[[179, 1], [-179, 1], [-179, 2], [179, 1]]] }, 100), null);
  assert.match(requestProblem(cap, null, 100, "2023-06-21T12:00:00Z"), /Draw an area/);
  assert.match(requestProblem({ ...cap, limits: { ...cap.limits, max_output_pixels: 10 } }, polygon, 100, "2023-06-21T12:00:00Z"), /too many/);
  assert.match(requestProblem({ ...cap, limits: { ...cap.limits, max_area_km2_by_resolution: { 100: 1 } } }, polygon, 100, "2023-06-21T12:00:00Z"), /area exceeds/);
  assert.match(requestProblem({ ...cap, queue: { pending: 3 } }, polygon, 100, "2023-06-21T12:00:00Z"), /queue is full/);
});
test("only same-origin download paths are used", () => {
  assert.equal(globalFile({ files: { prediction_tif: { url: "/api/global/jobs/abc/prediction.tif" } } }, "prediction_tif"), "/api/global/jobs/abc/prediction.tif");
  for (const bad of ["javascript:alert(1)", "//example.org/file", "https://example.org/file", null]) assert.equal(globalFile({ files: { prediction_tif: bad } }, "prediction_tif"), null);
});
test("progress uses authoritative messages without fabricating percentages", () => {
  assert.equal(progressDescription({ status: "running", message: "7 of 12 source tiles" }), "7 of 12 source tiles");
  assert.equal(progressDescription({ status: "queued" }), "Waiting for a worker");
});
