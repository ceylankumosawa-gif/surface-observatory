export type RasterLayer = "prediction" | "observed" | "residual";
export type Coordinate = [number, number];
export type PixelResponse = {
  job_id: string;
  query: { lon: number; lat: number };
  status: "ok" | "nodata" | "outside";
  pixel: { row: number; column: number; center: Coordinate; corners: Coordinate[] } | null;
  values: Record<RasterLayer, number | null>;
  units: "degC";
};
export type ProbePoint = { lon: number; lat: number; x: number; y: number };
export type ProbeView = { point: ProbePoint; state: "loading" | "error" | "ready"; data?: PixelResponse };

export const PIXEL_LABELS: Record<RasterLayer, string> = {
  prediction: "Model temperature", observed: "Satellite temperature", residual: "Model − satellite",
};
export const PIXEL_MISSING: Record<RasterLayer, string> = {
  prediction: "No supported estimate", observed: "No satellite observation", residual: "No comparable observation",
};

export function finiteValue(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

export function formatPixelValue(value: unknown, layer: RasterLayer): string | null {
  if (!finiteValue(value)) return null;
  const rounded = Number(value.toFixed(1)); // Numeric values only; null never becomes zero.
  const sign = rounded < 0 ? "−" : layer === "residual" ? "+" : "";
  return `${sign}${Math.abs(rounded).toFixed(1)} °C`;
}

export function withinBounds(lon: number, lat: number, bounds: readonly number[]): boolean {
  return bounds.length === 4 && bounds.every(finiteValue) && finiteValue(lon) && finiteValue(lat)
    && lon >= bounds[0] && lon <= bounds[2] && lat >= bounds[1] && lat <= bounds[3];
}

function coordinate(value: unknown): value is Coordinate {
  return Array.isArray(value) && value.length === 2 && value.every(finiteValue)
    && Math.abs(value[0]) <= 180 && Math.abs(value[1]) <= 90;
}

export function parsePixelResponse(value: unknown, jobId: string, point: ProbePoint): PixelResponse {
  const data = value as PixelResponse | null;
  if (!data || data.job_id !== jobId || data.units !== "degC"
    || !["ok", "nodata", "outside"].includes(data.status)
    || !data.query || !finiteValue(data.query.lon) || !finiteValue(data.query.lat)
    || Math.abs(data.query.lon - point.lon) > 1e-9 || Math.abs(data.query.lat - point.lat) > 1e-9
    || !data.values || !["prediction", "observed", "residual"].every((key) => {
      const v = data.values[key as RasterLayer]; return v === null || finiteValue(v);
    })) throw new Error("Invalid pixel response");
  if (data.status === "outside") {
    if (data.pixel !== null) throw new Error("Outside response contains a pixel");
  } else if (!data.pixel || !Number.isInteger(data.pixel.row) || data.pixel.row < 0
    || !Number.isInteger(data.pixel.column) || data.pixel.column < 0
    || !coordinate(data.pixel.center) || !Array.isArray(data.pixel.corners)
    || data.pixel.corners.length < 4 || data.pixel.corners.length > 6
    || !data.pixel.corners.every(coordinate)) throw new Error("Invalid pixel geometry");
  return data;
}

// Cache only the interior of the returned native cell. Shared edges are queried
// again so the server, not a rounded coordinate or polygon tie-break, owns them.
export function insidePixel(point: ProbePoint, ring: Coordinate[]): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [ax, ay] = ring[j], [bx, by] = ring[i];
    const dx = bx - ax, dy = by - ay, length = Math.hypot(dx, dy);
    if (length === 0) continue;
    const along = ((point.lon - ax) * dx + (point.lat - ay) * dy) / (length * length);
    const distance = Math.abs(dx * (point.lat - ay) - dy * (point.lon - ax)) / length;
    if (along >= 0 && along <= 1 && distance < 1e-8) return false;
    if ((ay > point.lat) !== (by > point.lat)
      && point.lon < (bx - ax) * (point.lat - ay) / (by - ay) + ax) inside = !inside;
  }
  return inside;
}

export function tooltipPosition(point: ProbePoint, width: number, height: number) {
  const cardWidth = Math.min(232, Math.max(0, width - 16)), cardHeight = 116, gap = 16;
  return {
    left: Math.max(8, Math.min(width - cardWidth - 8,
      point.x + gap + cardWidth <= width - 8 ? point.x + gap : point.x - cardWidth - gap)),
    top: Math.max(8, Math.min(height - cardHeight - 8,
      point.y + gap + cardHeight <= height - 8 ? point.y + gap : point.y - cardHeight - gap)),
  };
}

type PixelRequest = (jobId: string, point: ProbePoint, signal: AbortSignal) => Promise<unknown>;
export class PixelProbe {
  private jobId: string;
  private publish: (view: ProbeView | null) => void;
  private request: PixelRequest;
  private delay: number;
  private cacheLimit: number;
  private sequence = 0;
  private timer: ReturnType<typeof setTimeout> | undefined;
  private timeout: ReturnType<typeof setTimeout> | undefined;
  private pending: AbortController | undefined;
  private disposed = false;
  private cache: PixelResponse[] = [];
  constructor(jobId: string, publish: (view: ProbeView | null) => void,
    request: PixelRequest, delay = 150, cacheLimit = 64) {
    this.jobId = jobId; this.publish = publish; this.request = request;
    this.delay = delay; this.cacheLimit = cacheLimit;
  }

  clear() {
    this.sequence++;
    clearTimeout(this.timer); clearTimeout(this.timeout);
    this.pending?.abort(); this.pending = undefined;
    if (!this.disposed) this.publish(null);
  }

  move(point: ProbePoint, immediate = false) {
    if (this.disposed) return;
    this.clear();
    const sequence = this.sequence;
    const hit = this.cache.findIndex((item) => item.pixel && insidePixel(point, item.pixel.corners));
    if (hit !== -1) {
      const [data] = this.cache.splice(hit, 1); this.cache.push(data);
      this.publish({ point, state: "ready", data });
      return;
    }
    this.timer = setTimeout(async () => {
      const controller = new AbortController(); this.pending = controller;
      this.publish({ point, state: "loading" });
      this.timeout = setTimeout(() => controller.abort(), 6000);
      try {
        const response = await this.request(this.jobId, point, controller.signal);
        if (this.disposed || sequence !== this.sequence) return;
        const data = parsePixelResponse(response, this.jobId, point);
        if (data.pixel) {
          this.cache.push(data);
          if (this.cache.length > this.cacheLimit) this.cache.shift();
        }
        this.publish(data.status === "outside" ? null : { point, state: "ready", data });
      } catch {
        if (!this.disposed && sequence === this.sequence) this.publish({ point, state: "error" });
      } finally {
        if (sequence === this.sequence) {
          clearTimeout(this.timeout); this.pending = undefined;
        }
      }
    }, immediate ? 0 : this.delay);
  }

  dispose() { this.disposed = true; this.clear(); this.cache = []; }
}
