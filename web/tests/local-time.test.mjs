import test from "node:test";
import assert from "node:assert/strict";
import {
  PILOT_TIME_ZONES, resolveLocal, localParts, localDayBounds, solarElevation,
  solarDay, daylightGuide, selectionCentre, formatLocal, utcOffsetLabel,
} from "../src/local-time.ts";

test("local noon converts using the date's IANA rules, including half-hour zones", () => {
  assert.equal(resolveLocal("2026-09-08", "12:00", "America/Chicago")[0].utc, "2026-09-08T17:00:00.000Z");
  assert.equal(resolveLocal("2026-01-08", "12:00", "America/Chicago")[0].utc, "2026-01-08T18:00:00.000Z");
  assert.equal(resolveLocal("2026-09-08", "12:00", "Europe/London")[0].utc, "2026-09-08T11:00:00.000Z");
  assert.equal(resolveLocal("2026-09-08", "12:00", "Australia/Darwin")[0].utc, "2026-09-08T02:30:00.000Z");
  assert.equal(resolveLocal("2026-09-08", "00:15", "Asia/Shanghai")[0].utc, "2026-09-07T16:15:00.000Z");
});
test("spring-forward gaps are rejected and autumn folds return both explicit instants", () => {
  assert.deepEqual(resolveLocal("2026-03-29", "01:30", "Europe/London"), []);
  assert.deepEqual(resolveLocal("2026-10-25", "01:30", "Europe/London").map((p) => p.utc), ["2026-10-25T00:30:00.000Z", "2026-10-25T01:30:00.000Z"]);
  assert.deepEqual(resolveLocal("2026-03-08", "02:30", "America/Chicago"), []);
  assert.equal(resolveLocal("2026-11-01", "01:30", "America/Chicago").length, 2);
});
test("actual local day is 23, 24 or 25 hours and never forces 24", () => {
  assert.equal(localDayBounds("2026-03-29", "Europe/London").durationHours, 23);
  assert.equal(localDayBounds("2026-10-25", "Europe/London").durationHours, 25);
  assert.equal(localDayBounds("2026-06-21", "Australia/Darwin").durationHours, 24);
  const bounds = localDayBounds("2026-10-25", "Europe/London");
  assert.equal(localParts(bounds.start, "Europe/London").time, "00:00");
  assert.equal(localParts(bounds.end, "Europe/London").date, "2026-10-26");
});
test("invalid dates and skipped civil dates do not normalize into another request", () => {
  assert.deepEqual(resolveLocal("2026-02-29", "12:00", "Europe/London"), []);
  assert.deepEqual(resolveLocal("2026-02-30", "12:00", "Europe/London"), []);
  assert.equal(localDayBounds("2026-02-30", "Europe/London"), null);
  assert.equal(localDayBounds("2011-12-30", "Pacific/Apia"), null);
  assert.equal(resolveLocal("2024-02-29", "12:00", "Europe/London").length, 1);
});
test("all pilot zones are recognized and retain exact satellite UTC seconds", () => {
  assert.equal(Object.keys(PILOT_TIME_ZONES).length, 12);
  for (const zone of Object.values(PILOT_TIME_ZONES)) {
    assert.equal(resolveLocal("2026-09-08", "12:00", zone).length, 1);
  }
  const scene = "2023-09-07T10:59:42.123456Z";
  assert.equal(localParts(scene, "Europe/London").time, "11:59");
  assert.match(formatLocal(scene, "Europe/London", true), /11:59:42/);
  assert.equal(scene, "2023-09-07T10:59:42.123456Z");
  assert.equal(utcOffsetLabel(34200), "UTC+09:30");
});
test("London and Sioux Falls summer dates have ordered sunrise and sunset", () => {
  for (const [zone, point] of [["Europe/London", [-.088637, 51.488962]], ["America/Chicago", [-96.73, 43.68]]]) {
    const day = solarDay("2026-09-08", zone, point);
    assert.deepEqual(day.events.map((event) => event.kind), ["sunrise", "sunset"]);
    assert.ok(day.events[0].stamp < day.events[1].stamp);
    assert.ok(day.samples.some((sample) => sample.phase === "night"));
    assert.ok(day.samples.some((sample) => sample.phase === "twilight"));
    assert.ok(day.samples.some((sample) => sample.phase === "day"));
    for (const event of day.events) assert.ok(Math.abs(solarElevation(event.stamp, point) + .833) < .01);
  }
});
test("polar summer/winter never invent sunrise or sunset markers", () => {
  const summer = solarDay("2026-06-21", "America/Anchorage", [-156.61, 71.25]);
  const winter = solarDay("2026-12-21", "America/Anchorage", [-156.61, 71.25]);
  assert.equal(summer.events.length, 0);
  assert.ok(summer.samples.every((sample) => sample.phase === "day"));
  assert.equal(winter.events.length, 0);
  assert.ok(winter.samples.every((sample) => sample.elevation < -.833));
});
test("solar geometry matches published SPA example within 0.1 degrees", () => {
  // NREL SPA example: 17 Oct 2003, 12:30:30 MST, 39.742476N 105.1786W.
  // Apparent zenith 50.111622 degrees; refraction accounts for a small difference.
  const elevation = solarElevation("2003-10-17T19:30:30Z", [-105.1786, 39.742476]);
  assert.ok(Math.abs(elevation - (90 - 50.111622)) < .1);
});
test("daylight guide blocks night while leaving precise horizon decisions to the server", () => {
  const centre = [-.088637, 51.488962];
  const polygon = [[-.2, 51.4], [.2, 51.4], [.2, 51.6], [-.2, 51.6]];
  assert.equal(daylightGuide("2026-09-08T00:00:00Z", polygon, centre).definitelyNight, true);
  assert.equal(daylightGuide("2026-09-08T12:00:00Z", polygon, centre).nearHorizon, false);
  const c = selectionCentre(polygon, centre);
  assert.ok(Math.abs(c[0]) < 1e-8 && Math.abs(c[1] - 51.5) < 1e-8);
  assert.deepEqual(selectionCentre([], centre), centre);
});
