import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import * as maplibregl from "maplibre-gl";
import {
  type Map as GlobeMap,
  type GeoJSONSource,
} from "maplibre-gl";
import {
  Globe2,
  ArrowUpRight,
  Crosshair,
  Pentagon,
  X,
  Layers3,
  Download,
  Info,
  ChevronRight,
  LoaderCircle,
  ArrowLeft,
  Check,
  RotateCcw,
  Thermometer,
  CalendarDays,
  FlaskConical,
  MapPin,
} from "lucide-react";
import "maplibre-gl/dist/maplibre-gl.css";
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
maplibregl.setWorkerUrl(workerUrl);
import "./style.css";
import { LocalTimePicker } from "./LocalTimePicker";
import { RasterInspector, type RasterClick } from "./RasterInspector";
import type { RasterLayer } from "./raster-inspector";
import { mountRasterImage, type RasterImageStatus } from "./raster-image";
import { PILOT_TIME_ZONES, resolveLocal, localParts, formatLocal, selectionCentre, daylightGuide } from "./local-time";
import { GlobalPanel } from "./GlobalPanel";
import { GlobalResearchEvidence } from "./GlobalResearchEvidence";
import { normalizeGlobalResult, type GlobalJob } from "./global-explorer";

type Point = [number, number];
type Polygon = { type: "Polygon"; coordinates: Point[][] };
type Scene = {
  scene_id: string;
  datetime_utc: string;
  cloud_cover: number;
  platform: string;
};
type Pilot = {
  type: "Feature";
  id: string;
  geometry: any;
  properties: {
    id: string;
    label: string;
    center: Point;
    climate: string;
    bbox: number[];
    extent_m: [number, number, number, number];
    scenes: Scene[];
    landscape: string;
    eligible_sample_count: number;
  };
};
type Catalog = {
  pilots: { type: "FeatureCollection"; features: Pilot[] };
  support: any;
  model: any;
  sources: any[];
  limitations: string[];
  ui_copy: any;
};
type Job = {
  id: string;
  status: string;
  stage?: string;
  progress?: any;
  result?: any;
  error?: string;
};
const EMPTY: any = { type: "FeatureCollection", features: [] };
const V1_HEADLINE_LABELS: Record<
  string,
  { label: string; description?: string }
> = {
  heldout_london: { label: "London · held-out place" },
  temporal_2024: { label: "2024 · held-out dates" },
};
function headlineLabel(id: string, labels: any) {
  const entry = labels?.[id] ||
    V1_HEADLINE_LABELS[id] || { label: id.replaceAll("_", " ") };
  return typeof entry === "string" ? { label: entry } : entry;
}
const fmt = (x: any, d = 1) =>
  Number.isFinite(Number(x))
    ? Number(x).toLocaleString("en-GB", {
        maximumFractionDigits: d,
        minimumFractionDigits: d,
      })
    : "—";
const time = (s: string) =>
  new Date(s).toLocaleString("en-GB", {
    timeZone: "UTC",
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }) + " UTC";
function area(points: Point[]) {
  if (points.length < 3) return 0;
  const lat = points.reduce((a, p) => a + p[1], 0) / points.length;
  const k = 111.32;
  return (
    (Math.abs(
      points.reduce((s, p, i) => {
        const q = points[(i + 1) % points.length];
        return s + p[0] * q[1] - q[0] * p[1];
      }, 0),
    ) *
      k *
      k *
      Math.cos((lat * Math.PI) / 180)) /
    2
  );
}
function feature(p: Polygon): any {
  return { type: "Feature", properties: {}, geometry: p };
}
async function api(path: string, init?: RequestInit) {
  const r = await fetch(path, init);
  const data = await r.json();
  if (!r.ok)
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : JSON.stringify(data.detail || "Request failed"),
    );
  return data;
}

function App() {
  const [scope, setScope] = useState<"pilot" | "global">("global");
  const isGlobal = scope === "global";
  const [catalog, setCatalog] = useState<Catalog | null>(null),
    [error, setError] = useState(""),
    [tab, setTab] = useState("explore");
  const [pilotId, setPilotId] = useState("greater_london"),
    [mode, setMode] = useState("observed"),
    [sceneId, setSceneId] = useState(""),
    [surfaceSceneId, setSurfaceSceneId] = useState(""),
    [date, setDate] = useState("2023-09-07"),
    [hour, setHour] = useState("12:00"),
    [timeOccurrence, setTimeOccurrence] = useState(""),
    [zoneOverride, setZoneOverride] = useState("");
  const [pilotPolygon, setPilotPolygon] = useState<Polygon | null>(null),
    [globalPolygon, setGlobalPolygon] = useState<Polygon | null>(null),
    [wholePilotSelected, setWholePilotSelected] = useState(false),
    [drawing, setDrawing] = useState(false),
    [vertices, setVertices] = useState<Point[]>([]),
    [pilotJob, setPilotJob] = useState<Job | null>(null),
    [globalJob, setGlobalJob] = useState<GlobalJob | null>(null),
    [submitting, setSubmitting] = useState(false);
  const [layer, setLayer] = useState<RasterLayer>("prediction"),
    [opacity, setOpacity] = useState(0.85),
    [override, setOverride] = useState(false),
    [air, setAir] = useState("25"),
    [mapReady, setMapReady] = useState(false),
    [viewZoom, setViewZoom] = useState(1.6),
    [viewCentre, setViewCentre] = useState<Point>([12, 23]),
    [mapError, setMapError] = useState("");
  const [rasterImage, setRasterImage] = useState<{ key: string; status: RasterImageStatus }>({ key: "", status: "loading" });
  const [overlayAttempt, setOverlayAttempt] = useState(0);
  const imageSequence = useRef(0);
  const fittedGlobalJob = useRef("");
  const mapRef = useRef<GlobeMap | null>(null),
    host = useRef<HTMLDivElement>(null),
    drawRef = useRef(false),
    busyRef = useRef(false),
    scopeRef = useRef(scope),
    verticesRef = useRef<Point[]>([]),
    catalogRef = useRef<Catalog | null>(null),
    selectPilotRef = useRef<(id: string) => void>(() => {}),
    rasterClickRef = useRef<RasterClick>(() => false),
    markerRefs = useRef(new Map<string, maplibregl.Marker>());
  const polygon = isGlobal ? globalPolygon : pilotPolygon;
  const setPolygon = isGlobal ? setGlobalPolygon : setPilotPolygon;
  const job = isGlobal ? globalJob : pilotJob;
  const setJob = isGlobal ? setGlobalJob : setPilotJob;
  const pilot = catalog?.pilots.features.find(
      (p) => p.properties.id === pilotId,
    ),
    scene = pilot?.properties.scenes.find((s) => s.scene_id === sceneId),
    result = isGlobal ? normalizeGlobalResult(job?.result) : job?.result;
  const busy =
    submitting || ["queued", "running", "pending"].includes(job?.status || "");
  const done = ["completed", "complete", "succeeded"].includes(
    job?.status || "",
  );
  const withdrawnNightResult =
    result?.withdrawn === true ||
    ["nighttime_coarse_baseline", "mixed_day_night"].includes(
      result?.prediction_method,
    );
  const points = polygon?.coordinates[0].slice(0, -1) || vertices;
  const timeZone = zoneOverride || PILOT_TIME_ZONES[pilotId] || "UTC";
  const timeCentre = selectionCentre(points, pilot?.properties.center || [0, 0]);
  const timeCandidates = useMemo(() => resolveLocal(date, hour, timeZone), [date, hour, timeZone]);
  const selectedTime = timeCandidates.length === 1 ? timeCandidates[0]
    : timeCandidates.find((candidate) => candidate.utc === timeOccurrence);
  const solarSupport = selectedTime ? daylightGuide(selectedTime.utc, points, timeCentre) : null;
  const pilotExtent = pilot?.properties.extent_m;
  const pilotAreaKm = pilotExtent
    ? ((pilotExtent[2] - pilotExtent[0]) * (pilotExtent[3] - pilotExtent[1])) /
      1e6
    : 0;
  const areaKm = wholePilotSelected ? pilotAreaKm : area(points);
  const maxAreaKm = Number(catalog?.support?.max_bbox_area_km2 ?? 100);
  const maxVertices = Number(catalog?.support?.max_vertices ?? 64);
  const maxVerticesRef = useRef(64);
  maxVerticesRef.current = maxVertices;
  const requiresAir = mode === "scenario" || override;
  const validAir =
    air.trim() !== "" &&
    Number.isFinite(Number(air)) &&
    Number(air) >= -90 &&
    Number(air) <= 65;
  const validScenarioTime =
    date >= "1900-01-01" &&
    date <= "2100-12-31" &&
    !!selectedTime;
  useEffect(() => {
    api("/api/catalog")
      .then(setCatalog)
      .catch((e) => setError(e.message));
  }, []);
  useEffect(() => {
    const id = new URL(location.href).searchParams.get("global_job");
    if (!id || !/^[a-zA-Z0-9_-]{1,100}$/.test(id)) return;
    const controller = new AbortController();
    api(`/api/global/jobs/${encodeURIComponent(id)}`, { signal: controller.signal })
      .then(saved => {
        if (controller.signal.aborted) return;
        setGlobalJob(saved);
        if (saved.request?.polygon?.type === "Polygon") setGlobalPolygon(saved.request.polygon);
      })
      .catch(error => { if (!controller.signal.aborted) setError(`Could not restore this global job: ${error.message}`); });
    return () => controller.abort();
  }, []);
  useEffect(() => {
    catalogRef.current = catalog;
  }, [catalog]);
  useEffect(() => {
    busyRef.current = busy;
  }, [busy]);
  useEffect(() => { scopeRef.current = scope; }, [scope]);
  useEffect(() => {
    drawRef.current = drawing;
    mapRef.current
      ?.getCanvas()
      .style.setProperty("cursor", drawing ? "crosshair" : "");
    if (drawing) mapRef.current?.doubleClickZoom.disable();
    else mapRef.current?.doubleClickZoom.enable();
  }, [drawing]);
  useEffect(() => {
    verticesRef.current = vertices;
  }, [vertices]);
  useEffect(() => {
    if (!pilot) return;
    const ss = pilot.properties.scenes;
    setSceneId(
      (pilotId === "greater_london"
        ? ss.find((s) => s.datetime_utc.startsWith("2023-09-07"))
        : null
      )?.scene_id ||
        ss.slice().sort((a, b) => a.cloud_cover - b.cloud_cover)[0]?.scene_id ||
        "",
    );
    setPilotPolygon(null);
    setWholePilotSelected(false);
    setSurfaceSceneId("");
    setZoneOverride("");
    setTimeOccurrence("");
    if (scopeRef.current === "pilot") setVertices([]);
    setPilotJob(null);
    setError("");
  }, [pilotId, catalog]);
  useEffect(() => {
    if (!host.current || mapRef.current) return;
    try {
      const map = new maplibregl.Map({
        container: host.current,
        center: [12, 23],
        zoom: 1.6,
        maxZoom: 17,
        attributionControl: { compact: false },
        style: {
          version: 8,
          projection: { type: "globe" },
          sources: {
            basemap: {
              type: "raster",
              tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
              tileSize: 256,
              maxzoom: 19,
              attribution:
                '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>',
            },
          },
          layers: [
            {
              id: "space",
              type: "background",
              paint: { "background-color": "#0b121b" },
            },
            {
              id: "basemap",
              type: "raster",
              source: "basemap",
              paint: {
                "raster-saturation": -0.78,
                "raster-brightness-min": 0.08,
                "raster-brightness-max": 0.72,
                "raster-fade-duration": 180,
              },
            },
          ],
        },
      });
      mapRef.current = map;
      map.addControl(
        new maplibregl.NavigationControl({
          showCompass: true,
          visualizePitch: true,
        }),
        "bottom-right",
      );
      map.on("load", () => {
        map.addSource("pilots", { type: "geojson", data: EMPTY });
        map.addLayer({
          id: "pilot-fill",
          type: "fill",
          source: "pilots",
          paint: { "fill-color": "#72d6cd", "fill-opacity": 0.12 },
        });
        map.addLayer({
          id: "pilot-outline",
          type: "line",
          source: "pilots",
          paint: { "line-color": "#72d6cd", "line-width": 1.5 },
        });
        map.addSource("selection", { type: "geojson", data: EMPTY });
        map.addLayer({
          id: "selection-fill",
          type: "fill",
          source: "selection",
          filter: ["==", "$type", "Polygon"],
          paint: { "fill-color": "#ffb56e", "fill-opacity": 0.12 },
        });
        map.addLayer({
          id: "selection-line",
          type: "line",
          source: "selection",
          paint: {
            "line-color": "#ffb56e",
            "line-width": 2,
            "line-dasharray": [3, 2],
          },
        });
        map.addSource("vertices", { type: "geojson", data: EMPTY });
        map.addLayer({
          id: "vertex-dots",
          type: "circle",
          source: "vertices",
          paint: {
            "circle-color": "#ffb56e",
            "circle-radius": 4,
            "circle-stroke-width": 2,
            "circle-stroke-color": "#0b121b",
          },
        });
        setMapReady(true);
      });
      map.on("click", (e) => {
        if (busyRef.current) return;
        if ((e.originalEvent.target as HTMLElement)?.closest(".pilot-marker"))
          return;
        if (drawRef.current) {
          if (verticesRef.current.length < maxVerticesRef.current - 1) {
            const longitude = scopeRef.current === "global" ? ((e.lngLat.lng + 180) % 360 + 360) % 360 - 180 : e.lngLat.lng;
            const p: Point = [longitude, e.lngLat.lat];
            const prev = verticesRef.current;
            if (
              prev.length &&
              Math.hypot(prev.at(-1)![0] - p[0], prev.at(-1)![1] - p[1]) <
                0.00001
            )
              return;
            verticesRef.current = [...prev, p];
            setVertices([...verticesRef.current]);
          }
          return;
        }
        if (rasterClickRef.current(e)) return;
        if (scopeRef.current === "global") return;
        if (!map.getLayer("pilot-fill")) return;
        const fs = map.queryRenderedFeatures(e.point, {
          layers: ["pilot-fill"],
        });
        if (fs[0]?.properties?.id)
          selectPilotRef.current(String(fs[0].properties.id));
      });
      map.on("moveend", () => { setViewZoom(map.getZoom()); const centre = map.getCenter(); setViewCentre([centre.lng, centre.lat]); });
      map.on("mousemove", (e) => {
        if (drawRef.current || scopeRef.current === "global" || !map.getLayer("pilot-fill")) return;
        const overPilot =
          map.queryRenderedFeatures(e.point, { layers: ["pilot-fill"] })
            .length > 0;
        map.getCanvas().style.cursor =
          !busyRef.current && overPilot ? "pointer" : "";
      });
      map.on("error", (e) => {
        if ((e.error?.message || "").includes("WebGL"))
          setMapError(
            "Your browser could not start the globe. Enable hardware acceleration or try another browser.",
          );
      });
      return () => {
        markerRefs.current.forEach((marker) => marker.remove());
        markerRefs.current.clear();
        map.remove();
        mapRef.current = null;
      };
    } catch (e: any) {
      setMapError(e.message);
    }
  }, []);
  useEffect(() => {
    const m = mapRef.current;
    if (!m || !mapReady || !catalog) return;
    (m.getSource("pilots") as GeoJSONSource).setData(catalog.pilots as any);
    catalog.pilots.features.forEach((p) => {
      const id = p.properties.id;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "pilot-marker";
      button.title = `Explore ${p.properties.label}`;
      button.setAttribute("aria-label", `Explore ${p.properties.label} pilot`);
      const dot = document.createElement("span");
      dot.className = "pilot-marker-dot";
      dot.setAttribute("aria-hidden", "true");
      const label = document.createElement("span");
      label.className = "pilot-marker-label";
      label.textContent = p.properties.label;
      button.append(dot, label);
      button.addEventListener("click", (event) => {
        event.stopPropagation();
        if (busyRef.current || drawRef.current) return;
        let chosenId = id;
        // Nearby globe markers can overlap. Choose the centre closest to the
        // actual pointer, not whichever overlapping DOM button happens to win.
        if (event.detail > 0) {
          const rect = m.getCanvas().getBoundingClientRect();
          const pointer = {
            x: event.clientX - rect.left,
            y: event.clientY - rect.top,
          };
          let nearest = Infinity;
          markerRefs.current.forEach((marker, candidateId) => {
            if (Number(getComputedStyle(marker.getElement()).opacity) === 0)
              return;
            const point = m.project(marker.getLngLat());
            const distance = Math.hypot(
              point.x - pointer.x,
              point.y - pointer.y,
            );
            if (distance < nearest && distance <= 32) {
              nearest = distance;
              chosenId = candidateId;
            }
          });
        }
        selectPilotRef.current(chosenId);
      });
      const marker = new maplibregl.Marker({
        element: button,
        anchor: "center",
        opacityWhenCovered: 0,
        subpixelPositioning: true,
      })
        .setLngLat(p.properties.center)
        .addTo(m);
      markerRefs.current.set(id, marker);
    });
    return () => {
      markerRefs.current.forEach((marker) => marker.remove());
      markerRefs.current.clear();
    };
  }, [catalog, mapReady]);
  useEffect(() => {
    markerRefs.current.forEach((marker, id) => {
      const button = marker.getElement() as HTMLButtonElement;
      button.style.display = isGlobal ? "none" : "";
      button.classList.toggle("selected", id === pilotId);
      button.disabled = busy || drawing;
      button.setAttribute("aria-pressed", String(id === pilotId));
    });
    const m = mapRef.current;
    if (m?.getLayer("pilot-outline")) {
      m.setLayoutProperty("pilot-outline", "visibility", isGlobal ? "none" : "visible");
      m.setLayoutProperty("pilot-fill", "visibility", isGlobal ? "none" : "visible");
      m.setPaintProperty("pilot-outline", "line-width", [
        "case",
        ["==", ["get", "id"], pilotId],
        2.5,
        1,
      ]);
      m.setPaintProperty("pilot-fill", "fill-opacity", [
        "case",
        ["==", ["get", "id"], pilotId],
        0.13,
        0.045,
      ]);
    }
  }, [pilotId, busy, drawing, mapReady, catalog, isGlobal]);
  useEffect(() => {
    const m = mapRef.current;
    if (!m || !mapReady) return;
    let data: any = EMPTY;
    if (polygon) data = feature(polygon);
    else if (vertices.length >= 3)
      data = feature({
        type: "Polygon",
        coordinates: [[...vertices, vertices[0]]],
      });
    else if (vertices.length > 1)
      data = {
        type: "Feature",
        properties: {},
        geometry: { type: "LineString", coordinates: vertices },
      };
    (m.getSource("selection") as GeoJSONSource).setData(data);
    (m.getSource("vertices") as GeoJSONSource).setData({
      type: "FeatureCollection",
      features: vertices.map((p) => ({
        type: "Feature",
        properties: {},
        geometry: { type: "Point", coordinates: p },
      })),
    });
  }, [polygon, vertices, mapReady]);
  useEffect(() => {
    if (!job || !["queued", "running", "pending"].includes(job.status)) return;
    const controller = new AbortController();
    const t = window.setTimeout(
      () =>
        api((isGlobal ? "/api/global/jobs/" : "/api/jobs/") + encodeURIComponent(job.id), { signal: controller.signal })
          .then((j) => {
            if (controller.signal.aborted) return;
            setJob(j);
            setError("");
          })
          .catch((e) => {
            if (controller.signal.aborted) return;
            setError(e.message + " · Reconnecting…");
            setJob({ ...job });
          }),
      2000,
    );
    return () => { clearTimeout(t); controller.abort(); };
  }, [job, isGlobal]);
  function fileUrl(key: string) {
    const value = result?.files?.[key];
    if (!value) return null;
    return typeof value === "string" ? value : value.url;
  }
  const overlayUrl = fileUrl(
    layer === "observed"
      ? "observed_overlay_png"
      : layer === "residual"
        ? "residual_overlay_png"
        : "overlay_png",
  );
  const overlayKey = done && !withdrawnNightResult && overlayUrl && result?.bounds
    ? `${scope}:${job?.id}:${layer}:${overlayUrl}:${result.bounds.join(",")}:${overlayAttempt}` : "";
  const rasterImageReady = !!overlayKey && rasterImage.key === overlayKey && rasterImage.status === "ready";
  useEffect(() => {
    if (!isGlobal || !mapReady || !done || !globalJob || !result?.bounds || fittedGlobalJob.current === globalJob.id) return;
    const [west, south, east, north] = result.bounds;
    fittedGlobalJob.current = globalJob.id;
    mapRef.current?.fitBounds([[west, south], [east, north]], { padding: 85, maxZoom: 15, duration: 900, pitch: 0, bearing: 0 });
  }, [isGlobal, mapReady, done, globalJob?.id]);
  useLayoutEffect(() => {
    const m = mapRef.current;
    if (!mapReady || !m) return;
    if (!overlayKey || !overlayUrl || !result?.bounds) return;
    const [w, s, e, n] = result.bounds;
    const coordinates: [Point, Point, Point, Point] = [
      [w, n],
      [e, n],
      [e, s],
      [w, s],
    ];
    return mountRasterImage(m, {
      sourceId: `result-image-${++imageSequence.current}`, url: overlayUrl, coordinates, opacity,
      onStatus: (status) => setRasterImage({ key: overlayKey, status }),
    });
  }, [mapReady, overlayKey]);
  useEffect(() => {
    if (mapRef.current?.getLayer("result"))
      mapRef.current.setPaintProperty("result", "raster-opacity", opacity);
  }, [opacity]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setDrawing(false);
        setVertices([]);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  function fly(p = pilot) {
    const map = mapRef.current;
    if (!p || !map) return;
    const [w, s, e, n] = p.properties.bbox;
    map.stop();
    map.fitBounds(
      [
        [w, s],
        [e, n],
      ],
      {
        padding: { top: 78, bottom: 76, left: 38, right: 48 },
        maxZoom: 11,
        pitch: 0,
        bearing: 0,
        duration: 1300,
      },
    );
  }
  function selectPilot(id: string) {
    if (busyRef.current) return;
    const selected = catalogRef.current?.pilots.features.find(
      (p) => p.properties.id === id,
    );
    if (!selected) return;
    setScope("pilot");
    setPilotId(id);
    setTab("explore");
    setDrawing(false);
    drawRef.current = false;
    setVertices([]);
    verticesRef.current = [];
    fly(selected);
  }
  selectPilotRef.current = selectPilot;
  function switchScope(next: "pilot" | "global") {
    setScope(next); setDrawing(false); drawRef.current = false;
    setVertices([]); verticesRef.current = []; setError(""); setLayer("prediction");
  }
  function globalBox(centre: Point = viewCentre) {
    const [rawX, y] = centre;
    const x = ((rawX + 180) % 360 + 360) % 360 - 180;
    const dx = 2.5 / (111.32 * Math.max(.05, Math.cos(y * Math.PI / 180))), dy = 2.5 / 111.32;
    const ring: Point[] = [[x - dx, y - dy], [x + dx, y - dy], [x + dx, y + dy], [x - dx, y + dy], [x - dx, y - dy]];
    setGlobalPolygon({ type: "Polygon", coordinates: [ring] }); setGlobalJob(null); setError("");
    setDrawing(false); drawRef.current = false; setVertices([]); verticesRef.current = [];
    mapRef.current?.fitBounds([ring[0], ring[2]], { padding: 75, maxZoom: 13, duration: 900, pitch: 0, bearing: 0 });
  }
  function example() {
    if (!pilot) return;
    const [x, y] = pilot.properties.center;
    const dx = 2.5 / (111.32 * Math.cos((y * Math.PI) / 180)),
      dy = 2.5 / 111.32;
    const ps: Point[] = [
      [x - dx, y - dy],
      [x + dx, y - dy],
      [x + dx, y + dy],
      [x - dx, y + dy],
      [x - dx, y - dy],
    ];
    setPolygon({ type: "Polygon", coordinates: [ps] });
    setWholePilotSelected(false);
    setVertices([]);
    setDrawing(false);
    setJob(null);
    setError("");
    const map = mapRef.current;
    map?.stop();
    map?.fitBounds([ps[0], ps[2]], {
      padding: 80,
      maxZoom: 13,
      duration: 1300,
      pitch: 0,
      bearing: 0,
    });
  }
  function wholePilot() {
    if (!pilot || busy) return;
    setPolygon(pilot.geometry as Polygon);
    setWholePilotSelected(true);
    setVertices([]);
    verticesRef.current = [];
    setDrawing(false);
    drawRef.current = false;
    setJob(null);
    setError("");
    fly();
  }
  function finish() {
    if (vertices.length < 3) return;
    setPolygon({ type: "Polygon", coordinates: [[...vertices, vertices[0]]] });
    setWholePilotSelected(false);
    setVertices([]);
    setDrawing(false);
  }
  async function generate() {
    if (mode === "scenario" && (!validScenarioTime || solarSupport?.definitelyNight)) return;
    setError("");
    setSubmitting(true);
    setLayer("prediction");
    try {
      const payload = {
        pilot_id: pilotId,
        polygon,
        mode,
        scene_id: mode === "observed" ? sceneId : surfaceSceneId || undefined,
        datetime_utc:
          mode !== "observed" ? selectedTime?.utc : undefined,
        air_override: requiresAir ? Number(air) : undefined,
      };
      setJob(
        await api("/api/jobs", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        }),
      );
    } catch (e: any) {
      setError(e.message);
    } finally {
      setSubmitting(false);
    }
  }
  const stat = result?.summary || {};
  const nightBaseline =
    result?.prediction_method === "nighttime_coarse_baseline";
  const mixedDayNight = result?.prediction_method === "mixed_day_night";
  const legend =
    layer === "residual"
      ? result?.legends?.residual || {}
      : layer === "observed"
        ? result?.legends?.observed || {}
        : result?.legends?.prediction || {};
  const min =
      layer === "residual"
        ? -(stat.residual_display_limit_c || 1)
        : (legend.min_c ?? stat.min_c),
    max =
      layer === "residual"
        ? stat.residual_display_limit_c || 1
        : (legend.max_c ?? stat.max_c);
  const groups: Record<string, any[]> = {};
  catalog?.model.features.forEach((f: any) => {
    (groups[f.group] ||= []).push(f);
  });
  return (
    <div className="app">
      <header>
        <a className="brand" href="/" aria-label="Surface observatory home">
          <Globe2 size={25} />
          <span>
            degenerate<span className="brand-dot">.</span>energy
            <small>SURFACE OBSERVATORY</small>
          </span>
        </a>
        <nav aria-label="Main">
          <button
            className={tab === "explore" ? "active" : ""}
            onClick={() => setTab("explore")}
          >
            <Globe2 size={16} />
            Explore
          </button>
          <button
            className={tab === "evidence" ? "active" : ""}
            onClick={() => setTab("evidence")}
          >
            <FlaskConical size={16} />
            Model & evidence
          </button>
        </nav>
        <div className="pilot-badge">
          <i />
          {isGlobal ? "Global experimental" : "Research pilot"}<span>{isGlobal ? "UNVALIDATED" : "100 m"}</span>
        </div>
      </header>
      <main className={tab === "evidence" ? "evidence-open" : ""}>
        <aside className="controls">
          <div className="intro">
            <div className="eyebrow">AIR → SURFACE</div>
            <h1>
              Read the heat
              <br />
              of a place.
            </h1>
            <p>
              Explore how weather and the land beneath it shape surface
              temperature.
            </p>
          </div>
          <div className="scope-switch" role="group" aria-label="Explorer scope">
            <button className={!isGlobal ? "selected" : ""} disabled={submitting} onClick={() => switchScope("pilot")}>Pilot studies</button>
            <button className={isGlobal ? "selected" : ""} disabled={submitting} onClick={() => switchScope("global")}><Globe2 size={14} />Global experimental</button>
          </div>
          {isGlobal ? <GlobalPanel polygon={globalPolygon} drawing={drawing} vertices={vertices.length}
            mapReady={mapReady} mapCentre={viewCentre} job={globalJob} error={error} submitting={submitting}
            onJob={setGlobalJob} onSubmitting={setSubmitting} onError={setError}
            onDraw={() => { setDrawing(!drawing); setGlobalPolygon(null); setVertices([]); verticesRef.current = []; setGlobalJob(null); setError(""); }}
            onFinish={finish} onClear={() => { setGlobalPolygon(null); setGlobalJob(null); setError(""); }}
            onBox={globalBox} onLocate={(centre) => { mapRef.current?.stop(); mapRef.current?.flyTo({ center: centre, zoom: 11, pitch: 0, bearing: 0, duration: 900 }); }} /> : <>
          <section>
            <div className="step-title">
              <span>01</span>
              <h2>Choose your area</h2>
              <span className="quiet">12 pilots</span>
            </div>
            <label className="sr-only" htmlFor="pilot">
              Pilot area
            </label>
            <div className="select-line">
              <MapPin size={16} />
              <select
                id="pilot"
                value={pilotId}
                onChange={(e) => selectPilot(e.target.value)}
                disabled={!catalog || busy}
              >
                {catalog?.pilots.features.map((p) => (
                  <option value={p.properties.id} key={p.properties.id}>
                    {p.properties.label} · {p.properties.climate}
                  </option>
                ))}
              </select>
              <button
                className="icon-button"
                title="Show the whole selected pilot"
                aria-label="Show the whole selected pilot"
                onClick={() => fly()}
              >
                <Crosshair size={17} />
              </button>
            </div>
            <p className="helper">
              {pilot?.properties.landscape || "Loading the pilot catalog…"}
            </p>
            <div className="button-row area-actions">
              <button
                className={drawing ? "secondary selected" : "secondary"}
                disabled={busy || !mapReady}
                onClick={() => {
                  setDrawing(!drawing);
                  setPolygon(null);
                  setWholePilotSelected(false);
                  setVertices([]);
                  setJob(null);
                  fly();
                }}
              >
                <Pentagon size={15} />
                {drawing ? "Drawing…" : "Draw an area"}
              </button>
              <button
                className="secondary"
                disabled={
                  busy || !mapReady || !pilot || pilotAreaKm > maxAreaKm
                }
                onClick={wholePilot}
              >
                <Layers3 size={15} />
                Use whole pilot
              </button>
            </div>
            <button
              className="quick-area"
              disabled={busy || !mapReady || !pilot}
              onClick={example}
            >
              Quick test: use a 5 km box <ArrowUpRight size={13} />
            </button>
            {drawing ? (
              <div className="draw-hint">
                <span>
                  Click corners on the map. {vertices.length} vertices.
                </span>
                <button disabled={vertices.length < 3} onClick={finish}>
                  Finish area <Check size={14} />
                </button>
              </div>
            ) : polygon ? (
              <div
                className={"area-line " + (areaKm > maxAreaKm ? "invalid" : "")}
              >
                <span>
                  <span className="selection-dot" /> {fmt(areaKm)} km² ·{" "}
                  {wholePilotSelected ? "whole pilot" : "selected"}
                </span>
                <button
                  title="Clear area"
                  disabled={busy}
                  aria-label="Clear selected area"
                  onClick={() => {
                    setPolygon(null);
                    setWholePilotSelected(false);
                    setJob(null);
                  }}
                >
                  <X size={14} />
                </button>
              </div>
            ) : (
              <p className="helper small">
                Draw inside a pilot or select its full extent. Up to{" "}
                {fmt(maxAreaKm, 0)} km² per request.
              </p>
            )}
          </section>
          <section>
            <div className="step-title">
              <span>02</span>
              <h2>Set a moment</h2>
              <span className="quiet">Local time</span>
            </div>
            <div
              className="mode-switch"
              role="group"
              aria-label="Prediction mode"
            >
              <button
                className={mode === "observed" ? "selected" : ""}
                onClick={() => {
                  setMode("observed");
                  setJob(null);
                }}
                disabled={busy}
              >
                Satellite date
              </button>
              <button
                className={mode === "scenario" ? "selected" : ""}
                onClick={() => {
                  setMode("scenario");
                  setJob(null);
                }}
                disabled={busy}
              >
                Daytime scenario
              </button>
            </div>
            {mode === "observed" ? (
              <>
                <label htmlFor="scene">
                  Available observations · 2021–2024
                </label>
                <select
                  id="scene"
                  value={sceneId}
                  disabled={busy || !pilot}
                  onChange={(e) => {
                    setSceneId(e.target.value);
                    setJob(null);
                  }}
                >
                  {pilot?.properties.scenes.map((s) => (
                    <option key={s.scene_id} value={s.scene_id}>
                      {formatLocal(s.datetime_utc, timeZone)}
                    </option>
                  ))}
                </select>
                <p className="helper">
                  {scene
                    ? `${scene.platform.replace("-", " ").toUpperCase()} · ${fmt(scene.cloud_cover)}% cloud across the source scene`
                    : "Loading scene dates…"}
                </p>
                {scene && <div className="canonical-time satellite-clock">
                  <div><span>Local</span><strong>{formatLocal(scene.datetime_utc, timeZone, true)}</strong></div>
                  <div><span>Exact UTC</span><strong>{scene.datetime_utc}</strong></div>
                  <small>{timeZone} · satellite acquisition time is fixed</small>
                </div>}
                <div className="info-note">
                  <CalendarDays size={15} />
                  <span>
                    Predict at an observed overpass, then compare with the
                    satellite raster. Clear pixels only.
                  </span>
                </div>
              </>
            ) : (
              <>
                {pilot ? <LocalTimePicker
                  date={date} hour={hour} zone={timeZone} centre={timeCentre}
                  candidates={timeCandidates} selected={selectedTime} disabled={busy} hasArea={!!polygon}
                  onDate={(value) => { setDate(value); setTimeOccurrence(""); setJob(null); }}
                  onHour={(value) => { setHour(value); setTimeOccurrence(""); setJob(null); }}
                  onInstant={(utc) => {
                    const local = localParts(utc, timeZone);
                    setDate(local.date); setHour(local.time); setTimeOccurrence(utc); setJob(null);
                  }}
                  zoneChoice={pilotId === "singapore_johor" ? {
                    value: timeZone,
                    onChange: (zone) => { setZoneOverride(zone); setTimeOccurrence(""); setJob(null); },
                  } : undefined}
                /> : <p className="helper">Loading the local clock…</p>}
                {solarSupport?.definitelyNight ? (
                  <div className="info-note amber" role="status"><Info size={15} /><span>
                    Nighttime prediction is unavailable. Move the time slider into daylight; this model has not learned nighttime surface behaviour.
                  </span></div>
                ) : solarSupport?.nearHorizon ? (
                  <p className="helper small scenario-support">This moment is close to sunrise or sunset. A later morning or earlier evening time is more likely to keep your whole selection sunlit.</p>
                ) : null}
                <label htmlFor="scenario-air">Air temperature · required</label>
                <div className="unit-input">
                  <input
                    id="scenario-air"
                    type="number"
                    min="-90"
                    max="65"
                    step="0.1"
                    value={air}
                    onChange={(e) => {
                      setAir(e.target.value);
                      setJob(null);
                    }}
                    disabled={busy}
                    required
                  />
                  <span>°C</span>
                </div>
                <div className="info-note amber">
                  <FlaskConical size={15} />
                  <span>
                    Experimental daytime estimates with your reported air
                    temperature. The whole selection must be sunlit at the
                    chosen local time. Surface snapshot and weather basis are
                    shown with the result; arbitrary-date accuracy has not
                    been validated.
                  </span>
                </div>
                <p className="helper small scenario-support">
                  Nighttime prediction is unavailable. It needs a model that
                  learns how different surfaces cool at night; the previous
                  coarse weather baseline has been withdrawn.
                </p>
                <details className="surface-choice">
                  <summary>
                    Surface snapshot <ChevronRight size={14} />
                  </summary>
                  <label htmlFor="surface-scene">
                    Fixed land properties for this scenario
                  </label>
                  <select
                    id="surface-scene"
                    value={surfaceSceneId}
                    disabled={busy || !pilot}
                    onChange={(e) => {
                      setSurfaceSceneId(e.target.value);
                      setJob(null);
                    }}
                  >
                    <option value="">Automatic · closest season</option>
                    {pilot?.properties.scenes.map((s) => (
                      <option key={s.scene_id} value={s.scene_id}>
                        {formatLocal(s.datetime_utc, timeZone)}
                      </option>
                    ))}
                  </select>
                  <p className="helper small">
                    Uses an actual 2021–2024 surface snapshot. It can be earlier
                    or later than the scenario date; this is a fixed-land
                    experiment, not a historical reconstruction.
                  </p>
                </details>
              </>
            )}
            {mode === "observed" && (
              <details className="air-input">
                <summary>
                  <Thermometer size={14} />
                  Use your own air temperature
                  <ChevronRight size={14} />
                </summary>
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={override}
                    onChange={(e) => {
                      setOverride(e.target.checked);
                      setJob(null);
                    }}
                    disabled={busy}
                  />
                  Override the local air reference
                </label>
                {override && (
                  <>
                    <div className="unit-input">
                      <input
                        aria-label="Air temperature Celsius"
                        type="number"
                        min="-90"
                        max="65"
                        step="0.1"
                        value={air}
                        onChange={(e) => {
                          setAir(e.target.value);
                          setJob(null);
                        }}
                        disabled={busy}
                      />
                      <span>°C</span>
                    </div>
                    <p className="helper">
                      Anchored at your area's centre. Nearby ERA5 variation is
                      retained. This is a scenario, not an observed temperature.
                    </p>
                  </>
                )}
              </details>
            )}
          </section>
          <section className="run-section">
            <button
              className="generate"
              onClick={generate}
              disabled={
                !polygon ||
                drawing ||
                busy ||
                !catalog ||
                areaKm > maxAreaKm ||
                areaKm <= 0 ||
                (!sceneId && mode === "observed") ||
                (mode === "scenario" && (!validScenarioTime || solarSupport?.definitelyNight)) ||
                (requiresAir && !validAir)
              }
            >
              {busy ? (
                <LoaderCircle size={18} className="spin" />
              ) : (
                <Layers3 size={18} />
              )}
              <span>
                {busy ? "Preparing raster…" : "Generate temperature raster"}
              </span>
              {!busy && <ArrowUpRight size={18} />}
            </button>
            <p className="helper small">
              100 m cells · GeoTIFF + PNG + source manifest
              <br />
              One request at a time. Whole pilots take longer than small
              selections.
            </p>
            {busy && (
              <div className="job-progress" role="status">
                <span className="live-dot" />
                {typeof job?.stage === "string"
                  ? (
                      {
                        validate: "Checking area and time",
                        optical: "Reading satellite surface features",
                        land: "Checking the land mask",
                        terrain: "Reading elevation and terrain",
                        climate: "Adding climate and sun position",
                        weather: "Joining historical weather",
                        stations: "Finding nearby station readings",
                        radiation: "Reading thermal radiation and snow",
                        predict: "Predicting surface temperatures",
                        export: "Writing rasters and source manifest",
                      } as Record<string, string>
                    )[job.stage] || job.stage
                  : job?.status === "queued"
                    ? "Queued for processing"
                    : "Reading source data and building your raster…"}
              </div>
            )}
            {(error || job?.error) && (
              <div className="error" role="alert">
                <Info size={16} />
                <span>{error || job?.error}</span>
              </div>
            )}
          </section>
          {done && result && (
            <section className="result-panel">
              <div className="step-title">
                <span className="success">
                  <Check size={13} />
                </span>
                <h2>{withdrawnNightResult ? "Archived result" : "Your raster"}</h2>
                <span className="quiet">°C</span>
              </div>
              {withdrawnNightResult && (
                <div className="withdrawn-notice" role="alert">
                  <Info size={18} />
                  <div>
                    <strong>Withdrawn · unsupported nighttime result</strong>
                    <p>
                      {result.withdrawal_reason ||
                        "This output repeated coarse weather-grid temperatures; it did not learn local surface behaviour at night."}{" "}
                      Its map is hidden. The original files remain available
                      only to inspect what was produced.
                    </p>
                  </div>
                </div>
              )}
              <div className="result-stats">
                <div>
                  <strong>{fmt(stat.mean_c)}°</strong>
                  <span>
                    {nightBaseline
                      ? "Mean coarse baseline"
                      : "Mean model surface temp."}
                  </span>
                </div>
                <div>
                  <strong>{fmt(stat.valid_pixels, 0)}</strong>
                  <span>
                    {nightBaseline
                      ? "Output grid cells"
                      : "Predicted 100 m cells"}
                  </span>
                </div>
              </div>
              <p className="helper">
                {result.datetime_utc && <>{formatLocal(result.datetime_utc, timeZone)} local<br /></>}
                {time(result.datetime_utc || scene?.datetime_utc || date)}
                <br />
                {fmt(stat.min_c)} to {fmt(stat.max_c)} °C ·{" "}
                {fmt(stat.masked_pixels, 0)} masked cells
                <br />
                {result.counts?.observed_pixels > 0
                  ? `${fmt(result.counts.observed_pixels, 0)} cells have a satellite reference`
                  : "No coincident satellite reference"}
              </p>
              {result.weather_context?.weather_basis ===
                "seasonal_reference" && (
                <div className="info-note amber">
                  <Info size={15} />
                  <span>
                    Reference weather:{" "}
                    {time(
                      result.weather_context.weather_reference_datetime_utc,
                    )}
                    . Only your supplied air temperature and solar geometry
                    belong to the requested time; this is not a weather
                    forecast.
                  </span>
                </div>
              )}
              <p className="helper small">
                {nightBaseline
                  ? "Archived coarse weather baseline. It is not a supported surface-temperature prediction."
                  : mixedDayNight
                    ? "Withdrawn mixed result containing unsupported nighttime cells."
                    : stat.interval_radius_c != null
                      ? `Empirical interval: ±${fmt(stat.interval_radius_c)} °C. Coverage is not guaranteed for this area.`
                      : "A validated uncertainty interval is not available for this result."}
              </p>
              <div className="download-row">
                {fileUrl("prediction_tif") && (
                  <a href={fileUrl("prediction_tif")!} download>
                    <Download size={14} />
                    {withdrawnNightResult ? "Archived GeoTIFF" : "GeoTIFF"}
                  </a>
                )}
                {fileUrl("overlay_png") && (
                  <a href={fileUrl("overlay_png")!} download>
                    {withdrawnNightResult ? "Archived PNG" : "PNG"}{" "}
                    <ArrowUpRight size={13} />
                  </a>
                )}
                {fileUrl("provenance_json") && (
                  <a
                    href={fileUrl("provenance_json")!}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Sources <ArrowUpRight size={13} />
                  </a>
                )}
              </div>
              <details>
                <summary>
                  Data used for this result <ChevronRight size={14} />
                </summary>
                <Provenance result={result} catalog={catalog} />
              </details>
              {result.warnings?.length > 0 && (
                <details>
                  <summary>
                    Limitations & quality notes <ChevronRight size={14} />
                  </summary>
                  <ul className="notes">
                    {result.warnings.map((w: any, i: number) => (
                      <li key={i}>
                        {typeof w === "string" ? w : JSON.stringify(w)}
                      </li>
                    ))}
                  </ul>
                </details>
              )}
            </section>
          )}
          </>}
          <div className="controls-footer">
            <span className="live-dot" />
            Computed on Hetzner<span>{isGlobal ? "GLOBAL / EXPERIMENTAL" : "v0.1 / PILOT"}</span>
          </div>
        </aside>
        <div className="map-workspace">
          <div className="map" ref={host} />
          <RasterInspector map={mapReady ? mapRef.current : null} jobId={job?.id || null}
            layer={layer} bounds={result?.bounds} clickRef={rasterClickRef}
            apiBase={isGlobal ? "/api/global/jobs" : "/api/jobs"}
            enabled={mapReady && rasterImageReady && opacity > 0 && !drawing && !busy && tab === "explore"} />
          {overlayKey && !rasterImageReady && <div className="raster-image-status" role="status">
            {rasterImage.key === overlayKey && rasterImage.status === "error"
              ? <>Temperature image could not load.<button onClick={() => setOverlayAttempt((n) => n + 1)}>Retry</button></>
              : <><LoaderCircle className="spin" size={13} /> Loading temperature image…</>}
          </div>}
          {mapError && <div className="map-error">{mapError}</div>}
          <div className="map-top">
            <div className="map-title">
              <span className="live-dot" />
              {withdrawnNightResult
                ? "WITHDRAWN RESULT"
                : done
                  ? "SURFACE TEMPERATURE"
                  : isGlobal ? "GLOBAL EXPERIMENTAL" : "PILOT ATLAS"}
              <span>
                {withdrawnNightResult
                  ? "Unsupported night estimate · map hidden"
                  : done
                    ? isGlobal ? `${result?.grid?.resolution_m ?? "—"} m grid · accuracy unvalidated` : "100 m grid"
                    : isGlobal ? "Draw an area · source coverage varies" : "12 areas · 13 climate classes"}
              </span>
            </div>
            <button
              className="map-button"
              onClick={() => {
                mapRef.current?.stop();
                mapRef.current?.flyTo({
                  center: [12, 23],
                  zoom: 1.6,
                  duration: 1700,
                  bearing: 0,
                  pitch: 0,
                });
              }}
            >
              <Globe2 size={16} />
              World view
            </button>
          </div>
          {!polygon && !drawing && !done && viewZoom < 5 && (
            <div className="map-callout">
              <span className="eyebrow">
                {isGlobal ? "BEYOND THE PILOTS. EXPLICIT LIMITS." : "A GLOBAL QUESTION. TWELVE TEST GROUNDS."}
              </span>
              <h2>Every place holds heat differently.</h2>
              <p>
                {isGlobal ? "Zoom to a place and draw an area. Choose a local day or night moment; missing data stays visible as gaps."
                  : "Select a pilot on the globe, then choose an area to see how the model responds."}
              </p>
              <button onClick={isGlobal ? () => globalBox([-1.25, 51.75]) : example}>
                {isGlobal ? "Try an Oxford area" : `Explore ${pilot?.properties.label || "London"}`}{" "}
                <ArrowUpRight size={16} />
              </button>
            </div>
          )}
          {drawing && (
            <div className="drawing-banner">
              <Pentagon size={17} />
              Click to add corners · {vertices.length} vertices
              <button onClick={finish} disabled={vertices.length < 3}>
                Finish area <Check size={15} />
              </button>
            </div>
          )}
          {done && result && !withdrawnNightResult && (
            <div className="layer-panel">
              <div className="eyebrow">
                <Layers3 size={14} />
                RASTER LAYERS
              </div>
              <div className="layer-tabs">
                {[
                  ["prediction", nightBaseline ? "Baseline" : "Model"],
                  ...(fileUrl("observed_overlay_png")
                    ? [["observed", "Satellite"]]
                    : []),
                  ...(fileUrl("residual_overlay_png")
                    ? [["residual", "Difference"]]
                    : []),
                ].map(([id, label]) => (
                  <button
                    key={id}
                    className={layer === id ? "selected" : ""}
                    onClick={() => setLayer(id as RasterLayer)}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <div className="legend-label">
                {layer === "residual"
                  ? "Model − satellite"
                  : layer === "observed"
                    ? "Observed land surface temperature"
                    : nightBaseline
                      ? "Nighttime coarse baseline"
                      : "Predicted land surface temperature"}{" "}
                <span>°C</span>
              </div>
              <div
                className={
                  "color-ramp " + (layer === "residual" ? "diverging" : "")
                }
              />
              <div className="legend-ticks">
                <span>{fmt(min)}</span>
                <span>
                  {layer === "residual"
                    ? "0"
                    : fmt((Number(min) + Number(max)) / 2)}
                </span>
                <span>{fmt(max)}</span>
              </div>
              <label className="opacity">
                Opacity
                <input
                  type="range"
                  min="0"
                  max="1"
                  step=".05"
                  value={opacity}
                  onChange={(e) => setOpacity(Number(e.target.value))}
                />
                <span>{Math.round(opacity * 100)}%</span>
              </label>
              <p className="helper small">
                <span className="raster-inspection-hint">Hover over the raster to inspect a pixel. Click or tap works too.</span>
                Transparent cells have no supported estimate.
                <br />
                {isGlobal ? "Global extrapolation · no coincident satellite reference or validated error interval"
                  : result.mode === "scenario" || result.mode === "experimental"
                  ? nightBaseline
                    ? "Coarse night estimate · no learned 100 m detail"
                    : "Temperature scenario · no coincident ground truth"
                  : `${fmt(stat.compared_pixels ?? result.counts?.observed_pixels, 0)} satellite comparisons · ${fmt(100 * (stat.compared_coverage_fraction || 0))}% of predicted cells`}
              </p>
            </div>
          )}
          <div className="map-footnote">
            {!isGlobal && <><span className="pilot-key" />Pilot extent</>}
            <span className="selection-dot" />
            Your selection<span>Surface temperature ≠ air temperature</span>
          </div>
        </div>
        {tab === "evidence" && (
          <div className="evidence">
            <div className="evidence-inner">
              <button className="back-button" onClick={() => setTab("explore")}>
                <ArrowLeft size={16} />
                Back to the globe
              </button>
              <div className="eyebrow">OPEN THE MODEL</div>
              <h1>
                A warmer surface.
                <br />
                An explainable difference.
              </h1>
              <p className="lead">
                The model learns how far the land surface sits above or below
                the air temperature, using the properties of the place and its
                recent weather.
              </p>
              <div className="formula">
                <span>Air temperature</span>
                <b>+</b>
                <span>Learned surface–air difference</span>
                <b>=</b>
                <strong>Surface temperature</strong>
              </div>
              <p>
                One gradient-boosted model shares what it learns across places,
                with Köppen climate class as an input. It uses 40 features. It
                does not identify a single “most similar place” or fit a
                separate model for every climate.
              </p>
              <img
                className="flow-diagram"
                src="/model-flow.png"
                alt="Model pipeline: satellite and weather observations produce surface, terrain and climate features; a model learns the surface–air difference and predicts LST with an empirical interval."
              />
              {isGlobal && <GlobalResearchEvidence />}
              <h2>{isGlobal ? "Earlier pilot evaluation" : "What the evaluation tells us"}</h2>
              {isGlobal && <div className="info-note amber"><Info size={18} /><span>The following metrics describe the pilot studies. They do not validate the separate global experimental model. Its regional MAE target of 3°C remains unqualified; inspect each global result’s source manifest and coverage.</span></div>}
              <p>
                {catalog?.model.evaluation_description ||
                  "Entire London observations were held out. Future dates were tested separately. These are errors on sampled, eligible clear-sky daytime pixels, and do not establish accuracy for every raster cell or weather condition."}
              </p>
              <div className="metric-grid">
                {Object.entries(catalog?.model.headline || {})
                  .filter(([, value]) =>
                    Number.isFinite(
                      (value as any)?.ml?.mae_c ?? (value as any)?.mae_c,
                    ),
                  )
                  .map(([id, metric]) => (
                    <Metric
                      key={id}
                      {...headlineLabel(id, catalog?.model.headline_labels)}
                      metric={metric}
                    />
                  ))}
              </div>
              <RefinementComparison
                refinement={catalog?.model.refinement}
                labels={catalog?.model.headline_labels}
              />
              <div className="evidence-note">
                <Info size={18} />
                <div>
                  <strong>
                    100 m is a grid spacing, not 100 m weather detail.
                  </strong>
                  <p>
                    Satellite surface features vary by cell. The weather
                    background comes from coarse ERA5 cells, adjusted with
                    nearby station readings where available.
                  </p>
                </div>
              </div>
              <details className="regional-evaluation">
                <summary>
                  Compare evaluation across pilot areas{" "}
                  <ChevronRight size={15} />
                </summary>
                <p>
                  {catalog?.model.regional_evaluation_description ||
                    "London uses eight held-out dates from 2021–2024. Other areas use held-out 2024 dates. MAE is the mean absolute error; fewer dates mean weaker evidence."}
                </p>
                <div className="table-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>Area</th>
                        <th>MAE °C</th>
                        <th>RMSE °C</th>
                        <th>Dates</th>
                        <th>Pixels</th>
                      </tr>
                    </thead>
                    <tbody>
                      {catalog?.pilots.features.map((p) => {
                        const r = catalog.model.per_region[p.properties.id];
                        return r ? (
                          <tr key={p.properties.id}>
                            <td>{p.properties.label}</td>
                            <td>{fmt(r.ml.mae_c, 2)}</td>
                            <td>{fmt(r.ml.rmse_c, 2)}</td>
                            <td>{r.utc_day_count}</td>
                            <td>{fmt(r.ml.n, 0)}</td>
                          </tr>
                        ) : null;
                      })}
                    </tbody>
                  </table>
                </div>
              </details>
              <h2>What goes into a prediction</h2>
              <div className="feature-groups">
                {Object.entries(groups).map(([group, features]) => (
                  <details key={group}>
                    <summary>
                      <span>{group.replaceAll("_", " ")}</span>
                      <span>
                        {features.length} inputs <ChevronRight size={14} />
                      </span>
                    </summary>
                    {features.map((f) => (
                      <div className="feature-item" key={f.name}>
                        <strong>
                          {f.label}
                          <span>{f.unit}</span>
                        </strong>
                        <p>{f.note}</p>
                      </div>
                    ))}
                  </details>
                ))}
              </div>
              <h2>Weather matters. So does what is missing.</h2>
              <p>
                Humidity, wind, cloud cover, rain, soil moisture, sunlight,
                downward thermal radiation, snow and recent weather history are
                included. Individual buildings, tree heights, cast shadows and
                urban waste heat are not.
              </p>
              <p>
                <strong>Albedo is reflectivity. Shade is geometry.</strong> A
                darker pixel can reflect less light, but it does not tell us
                when a nearby tree or building blocks the sun. That needs height
                data and a separate shadow calculation.
              </p>
              <h2>{isGlobal ? "Available in pilot mode" : "Available here, today"}</h2>
              <div className="support-grid">
                <div>
                  <strong>12 pilot areas</strong>
                  <p>
                    Greater London plus eleven sites spanning 13 sampled Köppen
                    classes.
                  </p>
                </div>
                <div>
                  <strong>2021–2024 observations</strong>
                  <p>
                    95 catalogued scene–area observations support satellite
                    comparisons. Scenarios can use other dates with a supplied
                    air temperature and disclosed reference conditions.
                  </p>
                </div>
                <div>
                  <strong>Experimental daytime scenarios</strong>
                  <p>
                    The surface model accepts a reported air temperature when
                    the selection is sunlit. Scenario and cloudy-sky accuracy
                    remain unvalidated. Nighttime prediction is unavailable
                    until we can train and test it on nighttime observations.
                  </p>
                </div>
                <div>
                  <strong>Up to {fmt(maxAreaKm, 0)} km²</strong>
                  <p>
                    Select a whole pilot, including Greater London, or draw a
                    smaller area. One raster runs at a time; larger areas take
                    longer.
                  </p>
                </div>
              </div>
              <h2>{isGlobal ? "Pilot data sources" : "Data sources"}</h2>
              <div className="source-list">
                {catalog?.sources.map((s) => (
                  <a key={s.id} href={s.url} target="_blank" rel="noreferrer">
                    <div>
                      <strong>{s.label}</strong>
                      <p>{s.role}</p>
                      <small>
                        {s.license} · {s.attribution}
                      </small>
                    </div>
                    <ArrowUpRight size={17} />
                  </a>
                ))}
              </div>
              <details className="all-limitations">
                <summary>
                  All model limitations <ChevronRight size={15} />
                </summary>
                <ul>
                  {catalog?.limitations.map((s, i) => (
                    <li key={i}>{s}</li>
                  ))}
                </ul>
              </details>
              <button
                className="generate evidence-cta"
                onClick={() => setTab("explore")}
              >
                Explore a place <ArrowUpRight size={18} />
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
function Metric({
  label,
  description,
  metric,
}: {
  label: string;
  description?: string;
  metric: any;
}) {
  metric = metric?.ml || metric;
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>
        {fmt(metric?.mae_c, 2)}
        <small>°C</small>
      </strong>
      <p>Mean absolute error</p>
      <footer>
        RMSE {fmt(metric?.rmse_c, 2)} °C ·{" "}
        {fmt(metric?.n ?? metric?.n_samples ?? metric?.rows, 0)} pixels
      </footer>
      {description && <p className="metric-description">{description}</p>}
    </div>
  );
}
function RefinementComparison({
  refinement,
  labels,
}: {
  refinement: any;
  labels: any;
}) {
  const baseline = refinement?.baseline;
  const candidate = refinement?.candidate;
  const rows = Object.keys(baseline?.metrics || {}).filter(
    (id) => candidate?.metrics?.[id],
  );
  if (!rows.length) return null;
  const metricText = (metric: any) => {
    const value = metric?.ml || metric;
    return `${fmt(value?.mae_c, 2)} / ${fmt(value?.rmse_c, 2)}`;
  };
  const pixelText = (metric: any) => {
    const value = metric?.ml || metric;
    const n = value?.n ?? value?.n_samples ?? value?.rows;
    return n === undefined || n === null ? "" : `${fmt(n, 0)} pixels`;
  };
  return (
    <section
      className="refinement-comparison"
      aria-label="Model refinement comparison"
    >
      <div className="refinement-heading">
        <h3>Model refinement</h3>
        {refinement.status && (
          <span>{String(refinement.status).replaceAll("_", " ")}</span>
        )}
      </div>
      <p>
        {refinement.description ||
          "Recorded test errors for the baseline and candidate. Lower MAE and RMSE indicate smaller errors on the stated test samples."}
      </p>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Test cohort</th>
              <th>
                {baseline.label || "Baseline"}
                <small>MAE / RMSE · °C</small>
              </th>
              <th>
                {candidate.label || "Candidate"}
                <small>MAE / RMSE · °C</small>
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((id) => (
              <tr key={id}>
                <td>
                  {headlineLabel(id, refinement.cohort_labels || labels).label}
                  {refinement.cohort_notes?.[id] && (
                    <small>{refinement.cohort_notes[id]}</small>
                  )}
                </td>
                <td>
                  {metricText(baseline.metrics[id])}
                  <small>{pixelText(baseline.metrics[id])}</small>
                </td>
                <td>
                  {metricText(candidate.metrics[id])}
                  <small>{pixelText(candidate.metrics[id])}</small>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
function Provenance({
  result,
  catalog,
}: {
  result: any;
  catalog: Catalog | null;
}) {
  const values = result.feature_summary || {};
  const nighttime = result.prediction_method === "nighttime_coarse_baseline";
  const extraFeatures: Record<string, any> = {
    background_air_temperature_c: {
      name: "background_air_temperature_c",
      label: "ERA5 background air",
      unit: "°C",
    },
    era5_skin_temperature_c: {
      name: "era5_skin_temperature_c",
      label: "ERA5 modelled skin temperature",
      unit: "°C",
    },
  };
  const recordedFeatures = (
    Array.isArray(result.features) ? result.features : Object.keys(values)
  )
    .filter((name: string) => values[name] !== undefined)
    .map(
      (name: string) =>
        catalog?.model.features.find((f: any) => f.name === name) ||
        extraFeatures[name] || {
          name,
          label: name.replaceAll("_", " "),
          unit: "",
        },
    );
  const weather = result.weather_context || {};
  const weatherBasis =
    weather.weather_basis || weather.basis || result.weather_basis;
  const weatherReference =
    weather.weather_reference_datetime_utc ||
    weather.reference_datetime_utc ||
    result.weather_reference_datetime_utc;
  const describe = (f: any) =>
    f?.counts
      ? Object.entries(f.counts)
          .map(([k, v]) => `${k} (${v} cells)`)
          .join(", ")
      : f && Number.isFinite(f.mean)
        ? `${fmt(f.mean, 2)} mean · ${fmt(f.min, 2)}–${fmt(f.max, 2)}`
        : "Not recorded in this older result";
  const airLabels: Record<string, string> = {
    observed_station_residual_plus_ERA5_spatial_background:
      "Station-adjusted ERA5",
    observed_legacy_ISD_station_residual_plus_ERA5_spatial_background:
      "Station-adjusted ERA5 (legacy ISD)",
    ERA5_only_no_timely_station: "ERA5 only · no timely station",
    user_air_at_polygon_centroid_plus_ERA5_spatial_background:
      "Your air reference + ERA5 spatial variation",
    reported_air_at_centroid_plus_ERA5_spatial_background:
      "Your air reference + ERA5 spatial variation",
  };
  const sourceDescription = (s: any) =>
    s.collection === "landsat-c2-l2"
      ? result.mode === "observed"
        ? "Six reflectance bands describe the surface. Quality-screened thermal data provide the comparison."
        : "A fixed satellite snapshot supplies surface reflectance. It may precede or follow the scenario date; it is not a coincident thermal observation."
      : s.dataset?.includes("WorldCover")
        ? "Static 2021 land cover excludes water and cells with less than 80% supported land."
        : s.dataset?.includes("GLO-30")
          ? "Surface elevation, slope, aspect and nearby relief from a static digital surface model."
          : s.dataset?.includes("Köppen")
            ? "Climate class from the 1991–2020, 1 km classification."
            : s.dataset === "ERA5 via Open-Meteo"
              ? nighttime
                ? "The coarse ERA5 air-temperature background supplies spatial differences around your reported temperature."
                : "Hourly temperature, humidity, wind, pressure, cloud, rain, soil moisture and sunlight, plus recent weather history."
              : s.dataset === "ERA5 modelled skin temperature"
                ? "Coarse ERA5 modelled surface temperature supplies the nighttime skin–air difference. It is not a satellite observation or ground measurement."
                : s.dataset === "ARCO ERA5"
                  ? nighttime
                    ? "ERA5 modelled skin temperature and background air provide the coarse nighttime skin–air difference."
                    : "Downward thermal radiation and snow water equivalent at the last completed hour."
                  : s.dataset?.includes("NOAA")
                    ? "Quality-screened station readings correct the local air-temperature background."
                    : s.dataset?.includes("Manual")
                      ? "User-supplied temperature is anchored at the area centre; ERA5 spatial differences are retained."
                      : s.note ||
                        s.description ||
                        s.role ||
                        "See the source manifest for exact retrieval details.";
  return (
    <div className="provenance">
      <dl>
        <dt>Requested time</dt>
        <dd>{time(result.datetime_utc)}</dd>
        <dt>
          {nighttime
            ? "Catalog snapshot · not a night input"
            : "Surface snapshot"}
        </dt>
        <dd>{time(result.scene_datetime_utc)}</dd>
        <dt>Time relative to request</dt>
        <dd>
          {Math.abs(Number(result.source_age_days)) < 0.001
            ? "Same time"
            : `${fmt(Math.abs(Number(result.source_age_days)))} days ${Number(result.source_age_days) >= 0 ? "earlier" : "later"}`}
        </dd>
        <dt>Mode</dt>
        <dd>
          {result.mode === "observed"
            ? "Observed overpass"
            : result.mode === "scenario"
              ? "Supplied-temperature scenario"
              : "Experimental daytime hour"}
        </dd>
        <dt>Prediction method</dt>
        <dd>
          {result.prediction_method === "nighttime_coarse_baseline"
            ? "Withdrawn nighttime baseline · archive only"
            : result.prediction_method === "mixed_day_night"
              ? "Withdrawn mixed day/night result · archive only"
              : "Daytime surface model"}
        </dd>
        {weatherBasis && (
          <>
            <dt>Weather basis</dt>
            <dd>
              {weatherBasis === "actual_reanalysis"
                ? "Reanalysis at the requested time"
                : weatherBasis === "seasonal_reference"
                  ? "Seasonal reference weather · approximation"
                  : String(weatherBasis).replaceAll("_", " ")}
            </dd>
          </>
        )}
        {weatherReference && (
          <>
            <dt>Weather reference time</dt>
            <dd>{time(weatherReference)}</dd>
          </>
        )}
        <dt>Air-temperature reference</dt>
        {Object.entries(result.air_sources || {}).map(([k, v]) => (
          <dd key={k}>
            {airLabels[k] || k} · {fmt(v, 0)} cells
          </dd>
        ))}
        {Object.entries(result.station_ids || {})
          .filter(([id]) => id && id !== "nan" && id !== "None" && id !== "")
          .map(([id, count]) => (
            <React.Fragment key={id}>
              <dt>Station used</dt>
              <dd>
                {result.sources
                  ?.flatMap((s: any) => (Array.isArray(s.audit) ? s.audit : []))
                  .find((s: any) => s.station_id === id)?.name || id}{" "}
                · {id} · {fmt(count, 0)} cells
              </dd>
            </React.Fragment>
          ))}
        {result.station_details?.map((s: any, i: number) => (
          <React.Fragment key={i}>
            <dt>Station</dt>
            <dd>
              {s.name || s.station_name || s.station_id || JSON.stringify(s)}
            </dd>
          </React.Fragment>
        ))}
      </dl>
      <div className="weather-snapshot">
        {(nighttime
          ? [
              ["air_temperature_c", "Air"],
              ["background_air_temperature_c", "ERA5 background air"],
              ["era5_skin_temperature_c", "ERA5 skin temperature"],
            ]
          : [
              ["air_temperature_c", "Air"],
              ["cloud_cover_fraction", "Cloud fraction"],
              ["wind_speed_m_s", "Wind"],
              ["shortwave_down_w_m2", "Sunlight"],
            ]
        ).map(
          ([id, label]) =>
            values[id] && (
              <div key={id}>
                <span>{label}</span>
                <strong>
                  {fmt(
                    values[id].mean * (id === "cloud_cover_fraction" ? 100 : 1),
                    1,
                  )}
                  <small>
                    {id === "cloud_cover_fraction"
                      ? "%"
                      : (
                          catalog?.model.features.find(
                            (f: any) => f.name === id,
                          ) || extraFeatures[id]
                        )?.unit}
                  </small>
                </strong>
              </div>
            ),
        )}
      </div>
      {result.sources?.map?.((s: any, i: number) => (
        <div className="source-used" key={i}>
          {typeof s === "string" ? (
            s
          ) : (
            <>
              <strong>
                {s.collection === "landsat-c2-l2"
                  ? "Landsat surface observation"
                  : s.label || s.name || s.dataset || s.id || s.source}
              </strong>
              <p>{sourceDescription(s)}</p>
              {s.collection === "landsat-c2-l2" && <small>{s.id}</small>}
            </>
          )}
        </div>
      ))}
      <details>
        <summary>
          {recordedFeatures.length} recorded input values{" "}
          <ChevronRight size={13} />
        </summary>
        <p className="helper">
          Ranges cover valid predicted cells. Spatially constant weather is
          expected on a small area.
        </p>
        {recordedFeatures.map((f: any) => (
          <div className="actual-feature" key={f.name}>
            <strong>
              {f.label}
              <span>{f.unit}</span>
            </strong>
            <p>{describe(values[f.name])}</p>
          </div>
        ))}
      </details>
      <details>
        <summary>
          Pixel quality counts <ChevronRight size={13} />
        </summary>
        <dl>
          {Object.entries(result.counts || {}).map(([k, v]) => (
            <React.Fragment key={k}>
              <dt>{k.replaceAll("_", " ")}</dt>
              <dd>{fmt(v, 0)}</dd>
            </React.Fragment>
          ))}
        </dl>
      </details>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
