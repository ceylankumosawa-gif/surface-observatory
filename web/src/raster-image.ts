import type { Map as GlobeMap } from "maplibre-gl";
import type { Coordinate } from "./raster-inspector";

export type RasterImageStatus = "loading" | "ready" | "error";

// Each replacement owns a unique source ID. An event from an old image can
// therefore never enable inspection of the new job/layer, even after cancellation.
export function mountRasterImage(map: GlobeMap, options: {
  sourceId: string;
  url: string;
  coordinates: [Coordinate, Coordinate, Coordinate, Coordinate];
  opacity: number;
  onStatus: (status: RasterImageStatus) => void;
}) {
  const { sourceId, onStatus } = options;
  let active = true, metadataReady = false, status: RasterImageStatus = "loading";
  let timeout: ReturnType<typeof setTimeout> | undefined;
  const ownsLayer = () => (map.getLayer("result") as { source?: string } | undefined)?.source === sourceId;
  const failed = () => {
    if (!active || status === "error") return;
    status = "error"; clearTimeout(timeout);
    if (ownsLayer()) map.setLayoutProperty("result", "visibility", "none");
    onStatus("error");
  };
  const ready = () => {
    if (!active || !metadataReady || status !== "loading" || !ownsLayer() || !map.getSource(sourceId)
      || !map.isSourceLoaded(sourceId)) return;
    clearTimeout(timeout); status = "ready";
    map.setLayoutProperty("result", "visibility", "visible");
    onStatus("ready");
  };
  const sourceData = (event: { sourceId?: string; sourceDataType?: string }) => {
    if (event.sourceId === sourceId && event.sourceDataType === "metadata") {
      metadataReady = true; ready();
    }
  };
  const sourceError = (event: unknown) => {
    if (event && typeof event === "object" && "sourceId" in event && event.sourceId === sourceId) failed();
  };
  onStatus("loading");
  map.on("sourcedata", sourceData); map.on("error", sourceError);
  timeout = setTimeout(failed, 20000);
  try {
    map.addSource(sourceId, { type: "image", url: options.url, coordinates: options.coordinates });
    map.addLayer({ id: "result", type: "raster", source: sourceId,
      layout: { visibility: "none" },
      paint: { "raster-opacity": options.opacity, "raster-resampling": "nearest", "raster-fade-duration": 0 },
    }, "selection-line");
    ready(); // Also handles an image already loaded from browser cache.
  } catch { failed(); }
  return () => {
    active = false; clearTimeout(timeout);
    map.off("sourcedata", sourceData); map.off("error", sourceError);
    try {
      if (ownsLayer()) map.removeLayer("result");
      if (map.getSource(sourceId)) map.removeSource(sourceId);
    } catch { /* Parent map has already been removed. */ }
  };
}
