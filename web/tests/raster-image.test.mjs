import test from "node:test";
import assert from "node:assert/strict";
import { mountRasterImage } from "../src/raster-image.ts";

class FakeMap {
  sources = new Map(); layers = new Map(); events = new Map(); cacheReady = false;
  on(name, handler) { if (!this.events.has(name)) this.events.set(name, new Set()); this.events.get(name).add(handler); }
  off(name, handler) { this.events.get(name)?.delete(handler); }
  emit(name, sourceId, sourceDataType = "metadata") { for (const handler of this.events.get(name) || []) handler({ sourceId, sourceDataType }); }
  addSource(id) { this.sources.set(id, { loaded: this.cacheReady }); if (this.cacheReady) this.emit("sourcedata", id); }
  getSource(id) { return this.sources.get(id); }
  removeSource(id) { this.sources.delete(id); }
  isSourceLoaded(id) { return this.sources.get(id)?.loaded === true; }
  addLayer(layer, before) { assert.equal(before, "selection-line"); this.layers.set(layer.id, layer); }
  getLayer(id) { return this.layers.get(id); }
  removeLayer(id) { this.layers.delete(id); }
  setLayoutProperty(id, key, value) { this.layers.get(id).layout[key] = value; }
}
function options(sourceId, statuses) {
  return { sourceId, url: `${sourceId}.png`, coordinates: [[0, 1], [1, 1], [1, 0], [0, 0]], opacity: .85,
    onStatus: (status) => statuses.push(status) };
}

test("a replacement stays hidden until its own image has loaded; stale source events cannot release it", () => {
  const map = new FakeMap(), old = [], next = [];
  const disposeOld = mountRasterImage(map, options("old", old));
  map.sources.get("old").loaded = true; map.emit("sourcedata", "old");
  assert.equal(old.at(-1), "ready"); disposeOld();
  assert.equal(map.getLayer("result"), undefined);
  const disposeNext = mountRasterImage(map, options("new", next));
  assert.equal(map.getLayer("result").layout.visibility, "none");
  map.emit("sourcedata", "old"); map.emit("error", "old");
  assert.deepEqual(next, ["loading"]);
  map.emit("sourcedata", "new"); assert.deepEqual(next, ["loading"]);
  map.sources.get("new").loaded = true; map.emit("sourcedata", "new");
  assert.equal(next.at(-1), "ready"); assert.equal(map.getLayer("result").layout.visibility, "visible");
  disposeNext();
});

test("a failed replacement never reveals the previous image or enables inspection later", () => {
  const map = new FakeMap(), statuses = [];
  const dispose = mountRasterImage(map, options("new", statuses));
  map.emit("error", "new");
  assert.equal(statuses.at(-1), "error"); assert.equal(map.getLayer("result").layout.visibility, "none");
  map.sources.get("new").loaded = true; map.emit("sourcedata", "new");
  assert.equal(statuses.at(-1), "error"); dispose();
});

test("loaded status alone is insufficient because failed ImageSource loads also set it", () => {
  const map = new FakeMap(), statuses = [];
  const dispose = mountRasterImage(map, options("failed", statuses));
  map.sources.get("failed").loaded = true;
  map.emit("sourcedata", "failed", "content");
  assert.deepEqual(statuses, ["loading"]);
  assert.equal(map.getLayer("result").layout.visibility, "none");
  map.emit("error", "failed"); assert.equal(statuses.at(-1), "error"); dispose();
});

test("cached images can become ready immediately and cleanup removes all listeners/source", () => {
  const map = new FakeMap(), statuses = []; map.cacheReady = true;
  const dispose = mountRasterImage(map, options("cached", statuses));
  assert.deepEqual(statuses, ["loading", "ready"]); dispose();
  map.emit("sourcedata", "cached"); assert.equal(statuses.length, 2);
  assert.equal(map.sources.size, 0); assert.equal(map.layers.size, 0);
  assert.equal(map.events.get("sourcedata").size, 0); assert.equal(map.events.get("error").size, 0);
});
