import test from "node:test";
import assert from "node:assert/strict";
import { PixelProbe, formatPixelValue, parsePixelResponse, insidePixel, withinBounds, tooltipPosition, PIXEL_MISSING } from "../src/raster-inspector.ts";

const wait = () => new Promise((resolve) => setTimeout(resolve, 5));
const point = (lon = .5, lat = .5) => ({ lon, lat, x: 100, y: 80 });
function response(p = point(), prediction = 0, offset = 0) {
  return { job_id: "job", query: { lon: p.lon, lat: p.lat }, status: "ok", units: "degC",
    pixel: { row: 0, column: offset, center: [offset + .5, .5], corners: [[offset, 0], [offset + 1, 0], [offset + 1, 1], [offset, 1], [offset, 0]] },
    values: { prediction, observed: null, residual: null } };
}

test("temperature formatting preserves real zero and negatives without coercing missing values", () => {
  assert.equal(formatPixelValue(0, "prediction"), "0.0 °C");
  assert.equal(formatPixelValue(-12.35, "prediction"), "−12.3 °C");
  assert.equal(formatPixelValue(5.25, "residual"), "+5.3 °C");
  assert.equal(formatPixelValue(-.01, "residual"), "+0.0 °C");
  for (const value of [null, undefined, "0", false, NaN, Infinity]) assert.equal(formatPixelValue(value, "observed"), null);
  assert.equal(PIXEL_MISSING.observed, "No satellite observation");
  assert.equal(PIXEL_MISSING.residual, "No comparable observation");
});

test("responses must identify the current job/query, units, finite values and real pixel geometry", () => {
  assert.equal(parsePixelResponse(response(), "job", point()).values.prediction, 0);
  for (const change of [{ job_id: "another" }, { units: "K" }, { query: { lon: 1, lat: 1 } },
    { values: { prediction: "0", observed: null, residual: null } },
    { pixel: { ...response().pixel, row: -1 } }]) {
    assert.throws(() => parsePixelResponse({ ...response(), ...change }, "job", point()));
  }
  const missing = { ...response(), status: "nodata", values: { prediction: null, observed: null, residual: null } };
  assert.equal(parsePixelResponse(missing, "job", point()).values.prediction, null);
});

test("cache membership uses cell interiors and delegates boundaries to the server", () => {
  const ring = response().pixel.corners;
  assert.equal(insidePixel(point(), ring), true);
  assert.equal(insidePixel(point(1, .5), ring), false);
  assert.equal(insidePixel(point(1.001, .5), ring), false);
  assert.equal(withinBounds(1.01, .5, [0, 0, 1, 1]), false);
  assert.equal(withinBounds(NaN, .5, [0, 0, 1, 1]), false);
});

test("debounce requests only the last location and clears any previous display immediately", async () => {
  const calls = [], views = [];
  const probe = new PixelProbe("job", (view) => views.push(view), async (_, p) => {
    calls.push(p); return response(p);
  }, 1);
  probe.move(point(.2)); probe.move(point(.3)); probe.move(point(.4));
  assert.equal(views.at(-1), null);
  await wait();
  assert.equal(calls.length, 1); assert.equal(calls[0].lon, .4);
  assert.equal(views.at(-1).data.values.prediction, 0);
  probe.dispose();
});

test("late responses cannot overwrite a new pointer even when a server ignores abort", async () => {
  const calls = [], views = [];
  const probe = new PixelProbe("job", (view) => views.push(view), (_, p, signal) => new Promise((resolve) => calls.push({ p, signal, resolve })), 1);
  probe.move(point(.2)); await wait();
  probe.move(point(1.5));
  assert.equal(views.at(-1), null); assert.equal(calls[0].signal.aborted, true);
  await wait(); calls[1].resolve(response(point(1.5), -8, 1)); await wait();
  calls[0].resolve(response(point(.2), 99)); await wait();
  assert.equal(views.at(-1).data.values.prediction, -8);
  assert.equal(views.at(-1).point.lon, 1.5);
  probe.dispose();
});

test("clear/dispose cancel pending work for layer/job changes, drawing, leave and map movement", async () => {
  for (const action of ["clear", "dispose"]) {
    let resolve, signal; const views = [];
    const probe = new PixelProbe("job", (view) => views.push(view), (_, __, s) => {
      signal = s; return new Promise((r) => { resolve = r; });
    }, 1);
    probe.move(point()); await wait(); probe[action]();
    const count = views.length; assert.equal(signal.aborted, true);
    resolve(response()); await wait();
    assert.equal(views.length, count); probe.dispose();
  }
});

test("same-cell movement reuses all layer values; cache is bounded and outside yields no display", async () => {
  let requests = 0; const views = [];
  const probe = new PixelProbe("job", (view) => views.push(view), async (_, p) => {
    requests++; return response(p, 0, Math.floor(p.lon));
  }, 1, 2);
  probe.move(point(.4)); await wait();
  probe.move(point(.6)); assert.equal(requests, 1); assert.equal(views.at(-1).point.lon, .6);
  probe.move(point(1.5)); await wait(); probe.move(point(2.5)); await wait();
  probe.move(point(.4)); await wait(); assert.equal(requests, 4);
  probe.dispose();
  const outside = new PixelProbe("job", (view) => views.push(view), async (_, p) => ({ ...response(p), status: "outside", pixel: null }), 1);
  outside.move(point()); await wait(); assert.equal(views.at(-1), null); outside.dispose();
});

test("tooltip stays within a narrow viewport instead of covering the offscreen edge", () => {
  const position = tooltipPosition({ ...point(), x: 318, y: 300 }, 320, 320);
  assert.ok(position.left >= 8 && position.left + 232 <= 312);
  assert.ok(position.top >= 8 && position.top + 116 <= 312);
});
