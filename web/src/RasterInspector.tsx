import { useLayoutEffect, useState, type MutableRefObject } from "react";
import type { Map as GlobeMap, GeoJSONSource, MapMouseEvent, MapTouchEvent } from "maplibre-gl";
import {
  PixelProbe, PIXEL_LABELS, PIXEL_MISSING, finiteValue, formatPixelValue, withinBounds, tooltipPosition,
  type RasterLayer, type ProbeView, type ProbePoint,
} from "./raster-inspector";

export type RasterClick = (event: MapMouseEvent) => boolean;
type Props = {
  map: GlobeMap | null;
  jobId: string | null;
  layer: RasterLayer;
  bounds: number[] | undefined;
  enabled: boolean;
  clickRef: MutableRefObject<RasterClick>;
  apiBase?: "/api/jobs" | "/api/global/jobs";
};
const SOURCE = "inspected-raster-pixel";
const EMPTY = { type: "FeatureCollection" as const, features: [] };

export function RasterInspector({ map, jobId, layer, bounds, enabled, clickRef, apiBase = "/api/jobs" }: Props) {
  const [view, setView] = useState<(ProbeView & { key: string }) | null>(null);
  const boundsKey = bounds?.join(",") || "";
  const key = `${apiBase}:${jobId}:${layer}:${boundsKey}`;

  useLayoutEffect(() => {
    if (!map) return;
    map.addSource(SOURCE, { type: "geojson", data: EMPTY });
    map.addLayer({ id: `${SOURCE}-halo`, type: "line", source: SOURCE,
      paint: { "line-color": "#08131b", "line-width": 5, "line-opacity": 0.9 } });
    map.addLayer({ id: SOURCE, type: "line", source: SOURCE,
      paint: { "line-color": "#f0fff8", "line-width": 2 } });
    return () => {
      // The parent map can already have been removed during app unmount.
      try {
        if (map.getLayer(SOURCE)) map.removeLayer(SOURCE);
        if (map.getLayer(`${SOURCE}-halo`)) map.removeLayer(`${SOURCE}-halo`);
        if (map.getSource(SOURCE)) map.removeSource(SOURCE);
      } catch { /* Map lifecycle already ended. */ }
    };
  }, [map]);

  useLayoutEffect(() => {
    setView(null);
    if (!map) return;
    const outline = (value: ProbeView | null) => {
      const data = value?.data;
      const corners = data?.pixel?.corners;
      const valid = data && finiteValue(data.values[layer]) && corners;
      const ring = valid ? [...corners] : null;
      if (ring && (ring[0][0] !== ring.at(-1)![0] || ring[0][1] !== ring.at(-1)![1])) ring.push(ring[0]);
      (map.getSource(SOURCE) as GeoJSONSource | undefined)?.setData(ring
        ? { type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: [ring] } } : EMPTY);
    };
    outline(null);
    if (!enabled || !jobId || !boundsKey) return;
    const extent = boundsKey.split(",").map(Number);
    const probe = new PixelProbe(jobId, (next) => {
      setView(next ? { ...next, key } : null); outline(next);
    }, async (id, point, signal) => {
      const query = new URLSearchParams({ lon: String(point.lon), lat: String(point.lat) });
      const response = await fetch(`${apiBase}/${encodeURIComponent(id)}/pixel?${query}`, { signal });
      if (!response.ok) throw new Error("Pixel read failed");
      return response.json();
    });
    let touchStart: { x: number; y: number; time: number } | null = null;
    const eventPoint = (event: MapMouseEvent | MapTouchEvent): ProbePoint => ({
      lon: event.lngLat.lng, lat: event.lngLat.lat, x: event.point.x, y: event.point.y,
    });
    const inspect = (event: MapMouseEvent | MapTouchEvent, immediate = false) => {
      const target = event.originalEvent.target;
      if (map.isMoving() || !map.getLayer("result") || map.getLayoutProperty("result", "visibility") === "none"
        || (target instanceof Element && target.closest(".pilot-marker, .maplibregl-ctrl"))) {
        probe.clear(); return false;
      }
      const point = eventPoint(event);
      if (!withinBounds(point.lon, point.lat, extent)) { probe.clear(); return false; }
      probe.move(point, immediate); return true;
    };
    const hover = (event: MapMouseEvent) => { inspect(event); };
    const clear = () => { touchStart = null; probe.clear(); };
    const touchBegin = (event: MapTouchEvent) => {
      clear();
      if (event.originalEvent.touches.length === 1) touchStart = { ...event.point, time: Date.now() };
    };
    const touchMove = (event: MapTouchEvent) => {
      if (touchStart && Math.hypot(event.point.x - touchStart.x, event.point.y - touchStart.y) > 8) clear();
    };
    const touchEnd = (event: MapTouchEvent) => {
      const tap = touchStart; touchStart = null;
      if (tap && Date.now() - tap.time < 500 && event.originalEvent.touches.length === 0) inspect(event, true);
    };
    // The parent calls this before pilot-fill navigation. Marker buttons retain
    // their own handler; a click/tap on a visible raster inspects rather than flies.
    const click: RasterClick = (event) => inspect(event, true);
    clickRef.current = click;
    map.on("mousemove", hover); map.on("mouseout", clear); map.on("movestart", clear);
    map.on("touchstart", touchBegin); map.on("touchmove", touchMove); map.on("touchend", touchEnd);
    const canvas = map.getCanvas(); canvas.addEventListener("mouseleave", clear);
    const escape = (event: KeyboardEvent) => { if (event.key === "Escape") clear(); };
    window.addEventListener("keydown", escape);
    return () => {
      probe.dispose();
      if (clickRef.current === click) clickRef.current = () => false;
      map.off("mousemove", hover); map.off("mouseout", clear); map.off("movestart", clear);
      map.off("touchstart", touchBegin); map.off("touchmove", touchMove); map.off("touchend", touchEnd);
      canvas.removeEventListener("mouseleave", clear); window.removeEventListener("keydown", escape);
      try { outline(null); } catch { /* Parent map was removed. */ }
    };
  }, [map, jobId, layer, enabled, boundsKey, key, clickRef, apiBase]);

  if (!enabled || !map || !view || view.key !== key) return null;
  const position = tooltipPosition(view.point, map.getContainer().clientWidth, map.getContainer().clientHeight);
  const text = view.data ? formatPixelValue(view.data.values[layer], layer) : null;
  return <div className={`raster-inspector ${text ? "has-value" : "is-empty"}`} style={position} role="tooltip">
    <div className="raster-inspector-title"><span />{PIXEL_LABELS[layer]}</div>
    {view.state === "loading" ? <p className="raster-inspector-message">Reading pixel…</p>
      : view.state === "error" ? <p className="raster-inspector-message">Could not read this pixel</p>
      : text ? <strong className="raster-inspector-value">{text}</strong>
      : <p className="raster-inspector-message">{PIXEL_MISSING[layer]}</p>}
    {view.data?.pixel && <div className="raster-inspector-location">
      Row {view.data.pixel.row + 1} <span>·</span> Column {view.data.pixel.column + 1}
    </div>}
  </div>;
}
