export type SolarComparison = {
  test_mode: "geography" | "month" | "reference";
  geography_role: "original_development" | "new_development" | "original_reference" | "reserved_new_geography";
  paired: number; dates: number; baseline_balanced_mae: number | null; candidate_balanced_mae: number | null;
  baseline_mae: number | null; candidate_mae: number | null;
  baseline_bias: number | null; candidate_bias: number | null;
  baseline_above_5_fraction: number | null; candidate_above_5_fraction: number | null;
  baseline_above_7_fraction: number | null; candidate_above_7_fraction: number | null;
  delta_balanced_mae: number | null; outcome: string;
};
export type SolarRegion = SolarComparison & { region_id: string; phase: "day" | "night" };
export type SolarEvidence = {
  status: "complete"; generated_at_utc: string; baseline_label: "Expanded baseline";
  candidate_label: "Solar-season candidate"; description: string; target_mae_c: 3; qualified: false;
  comparisons: SolarComparison[]; regions: SolarRegion[]; download_url: string; cases_download_url: string;
  source: { audit_sha256: string; metrics_sha256: string; trial_sha256: string };
};
export const solarRoleLabel = (role: string) => ({ original_development: "Original development areas", new_development: "New development areas", original_reference: "Cabauw reference", reserved_new_geography: "Reserved · unopened" }[role] ?? role);
export const solarTestLabel = (mode: string) => ({ geography: "Whole area held out", month: "Whole month held out", reference: "Cabauw reference" }[mode] ?? mode);
export const solarChartRows = (rows: SolarComparison[]) => rows.filter(r => r.test_mode === "reference"
  ? r.geography_role === "original_reference"
  : ["original_development", "new_development"].includes(r.geography_role));

export function readSolarEvidence(input: unknown, origin: string): SolarEvidence {
  const d = input as SolarEvidence;
  const fail = () => { throw new Error("The solar-season evidence is incomplete or has an unsupported format."); };
  const integer = (n: unknown) => typeof n === "number" && Number.isInteger(n) && n >= 0;
  const finite = (n: unknown) => n === null || (typeof n === "number" && Number.isFinite(n));
  const download = (p: unknown) => typeof p === "string" && /^\/research\/[a-z0-9-]+\.csv$/.test(p)
    && new URL(p, origin).origin === origin;
  const score = (r: SolarComparison) => r && ["geography", "month", "reference"].includes(r.test_mode)
    && ["original_development", "new_development", "original_reference", "reserved_new_geography"].includes(r.geography_role)
    && [r.paired, r.dates].every(integer) && ["missing", "improved", "worsened", "unchanged"].includes(r.outcome)
    && [r.baseline_mae, r.candidate_mae, r.baseline_balanced_mae, r.candidate_balanced_mae].every(v => finite(v) && (v === null || v >= 0))
    && [r.baseline_bias, r.candidate_bias, r.delta_balanced_mae].every(finite)
    && [r.baseline_above_5_fraction, r.candidate_above_5_fraction, r.baseline_above_7_fraction, r.candidate_above_7_fraction]
      .every(v => finite(v) && (v === null || (v >= 0 && v <= 1)))
    && [r.baseline_mae, r.candidate_mae, r.baseline_balanced_mae, r.candidate_balanced_mae, r.baseline_bias,
      r.candidate_bias, r.delta_balanced_mae, r.baseline_above_5_fraction, r.candidate_above_5_fraction,
      r.baseline_above_7_fraction, r.candidate_above_7_fraction].every(v => r.paired === 0 ? v === null : v !== null)
    && (r.geography_role !== "reserved_new_geography" || r.paired === 0)
    && (r.paired === 0 ? r.outcome === "missing" : r.outcome !== "missing");
  if (!d || d.status !== "complete" || d.baseline_label !== "Expanded baseline" || d.candidate_label !== "Solar-season candidate"
    || d.target_mae_c !== 3 || d.qualified !== false || typeof d.description !== "string" || !d.description
    || typeof d.generated_at_utc !== "string" || !Number.isFinite(Date.parse(d.generated_at_utc))
    || !download(d.download_url) || !download(d.cases_download_url) || !d.source
    || ![d.source.audit_sha256, d.source.metrics_sha256, d.source.trial_sha256].every(h => typeof h === "string" && /^[a-f0-9]{64}$/.test(h))) fail();
  if (!Array.isArray(d.comparisons) || d.comparisons.length !== 12 || !d.comparisons.every(score)
    || new Set(d.comparisons.map(r => `${r.test_mode}|${r.geography_role}`)).size !== 12
    || solarChartRows(d.comparisons).length !== 5) fail();
  if (!Array.isArray(d.regions) || d.regions.length !== 96 || new Set(d.regions.map(r => `${r.region_id}|${r.phase}`)).size !== 96
    || !d.regions.every(r => score(r) && typeof r.region_id === "string" && r.region_id.length > 0
      && ["day", "night"].includes(r.phase) && (r.test_mode === "reference") === (r.geography_role === "original_reference")
      && r.test_mode !== "month")) fail();
  return d;
}
