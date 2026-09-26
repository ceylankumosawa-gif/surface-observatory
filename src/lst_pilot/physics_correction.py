"""Fixed weak-physics experiment on frozen F; no acquisition or deployment."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from . import option_b_train as old
from . import more_days_train as more
from . import multisensor_train as multi
from .physics_features import add_physics_features

VERSION = "weak-physics-comparison-v1"
PHYSICS = ("physics_absorbed_shortwave_w_m2", "physics_longwave_cooling_potential_w_m2",
           "physics_absorbed_shortwave_mean6h_w_m2")
RAW = ("albedo_proxy", "shortwave_down_w_m2", "era5_longwave_down_w_m2",
       "physics_raw_shortwave_mean6h_w_m2")
ARMS = {"W": "Weather, weak", "P": "Physics, weak", "WP": "Weather + physics, weak",
        "WR": "Weather + raw measurements, weak"}
SPEC = {"ridge_alpha": 20.0, "shrinkage": .25, "max_abs_adjustment_c": 1.0,
        "penalized_constant": True, "common_fitting_and_prediction_support": True,
        "fit_weights": "original frozen F weights restricted to common valid rows, then normalized",
        "weight_scale": "effective independent global UTC date count on retained normalized weights",
        "minimum_group_utc_dates": 6, "minimum_group_year_months": 3, "minimum_group_folds": 2,
        "no_tuning": True, "no_base_refit": True, "auto_promotion": False,
        "all_evaluation_rows_retained": True, "missing_or_unsupported_fallback": "exact F",
        "primary_contrasts": ["WP versus F", "WP versus WR"], "all_arms_reported": list(ARMS)}


def numerical_inputs(frame, offset, kind):
    core, climate, phase = multi.correction_inputs(frame, offset)
    if kind == "W":
        values, names = core, list(multi.NUMERIC)
    elif kind == "P":
        values, names = frame[list(PHYSICS)].to_numpy(float), list(PHYSICS)
    elif kind in ("WP", "WR"):
        extra = PHYSICS if kind == "WP" else RAW
        values = np.column_stack([core, frame[list(extra)].to_numpy(float)])
        names = [*multi.NUMERIC, *extra]
    else:
        raise ValueError("Unknown fixed correction arm.")
    return values, climate, phase, names


def valid_physics(frame):
    if "physics_complete" not in frame:
        raise ValueError("Explicit physical-input availability proof is required.")
    proof = old.strict_bool(frame.physics_complete, "physics_complete").to_numpy()
    return proof & np.isfinite(frame[[*PHYSICS, RAW[-1]]].to_numpy(float)).all(axis=1)


@dataclass
class WeakCorrection:
    kind: str
    mean: np.ndarray
    scale: np.ndarray
    ridge: object
    support: dict
    effective_dates: float
    names: list[str]

    @classmethod
    def fit(cls, frame, oof_offset, folds, weights, kind, support):
        if not np.array_equal(multi.month_folds(frame), folds):
            raise ValueError("Correction fold alignment changed.")
        if not valid_physics(frame).all():
            raise ValueError("Every correction must use the same physical-complete fitting rows.")
        x, climate, phase, names = numerical_inputs(frame, oof_offset, kind)
        if not np.isfinite(x).all():
            raise ValueError("Correction inputs must be finite.")
        w = np.asarray(weights, float)
        effective = multi.global_date_scale(frame, w)
        w = w / w.sum()
        mean = np.average(x, axis=0, weights=w)
        scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=w))
        scale[scale < 1e-8] = 1
        design = multi.ResidualCorrection.design(x, climate, phase, mean, scale)
        target = old.target_offset(frame) - np.asarray(oof_offset)
        ridge = Ridge(alpha=SPEC["ridge_alpha"], fit_intercept=False, solver="svd")
        ridge.fit(design, target, sample_weight=w*effective)
        return cls(kind, mean, scale, ridge, support, effective, names)

    def predict(self, frame, offset):
        valid = valid_physics(frame)
        output = np.zeros(len(frame), dtype=float)
        supported = np.zeros(len(frame), dtype=bool)
        if valid.any():
            selected = frame.loc[valid]
            x, climate, phase, names = numerical_inputs(selected, np.asarray(offset)[valid], self.kind)
            if names != self.names or not np.isfinite(x).all():
                raise ValueError("Prediction feature contract changed.")
            allowed = np.array([self.support.get(f"{c}|{p}", {}).get("supported", False)
                                for c, p in zip(climate, phase)], bool)
            supported[np.flatnonzero(valid)] = allowed
            if allowed.any():
                design = multi.ResidualCorrection.design(x[allowed], climate[allowed], phase[allowed], self.mean, self.scale)
                output[np.flatnonzero(valid)[allowed]] = np.clip(
                    SPEC["shrinkage"]*self.ridge.predict(design), -SPEC["max_abs_adjustment_c"], SPEC["max_abs_adjustment_c"])
        return output, supported


def verify_inputs(manifest):
    for key, digest in manifest["input_hashes"].items():
        path = manifest["paths"].get(key)
        if path and old.sha(path) != digest:
            raise ValueError(f"Frozen F input changed: {key}")


def verify_experiment_freeze(records, frozen, protocol, output, fit_freeze_sha, extra_audits=()):
    if old.sha(protocol) != records["protocol_sha256"]:
        raise ValueError("Experiment protocol changed after fitting.")
    for name, digest in records["source_hashes"].items():
        if old.sha(Path(__file__).with_name(name)) != digest:
            raise ValueError(f"Experiment source changed after fitting: {name}")
    if old.sha(output/"fit_freeze.json") != fit_freeze_sha:
        raise ValueError("Experiment fitting freeze changed.")
    for name, digest in frozen.items():
        if old.sha(output/name) != digest:
            raise ValueError(f"Correction artifact changed after fitting: {name}")
    for audit in (records["feature_audit"], *extra_audits):
        for cached in audit.get("cache_records", []):
            if "path" in cached and "sha256" in cached and old.sha(cached["path"]) != cached["sha256"]:
                raise ValueError("An exact weather-cache input changed after feature derivation.")


def reconstruct_fit(manifest, reference_run):
    p = manifest["paths"]
    areas = {a["id"]: a for a in json.loads(Path(p["areas_path"]).read_text())["areas"]}
    original, e_fit, evaluation, _ = multi.reconstruct_e(
        p["original_input"], p["e_additions"], p["original_run"], p["e_run"], areas, p["baseline"])
    known = pd.concat([original, e_fit], ignore_index=True).drop_duplicates("sample_id")
    additions = multi.validate_new(multi.load_new(p["new_fit"]), known, areas)
    fit = multi.join_fitting(e_fit, additions)
    if len(fit) != manifest["fit_rows"] or old.row_hash(fit) != manifest["fit_row_sha256"]:
        raise ValueError("Frozen F fitting identities changed.")
    records = pd.read_parquet(reference_run/"fitting_rows.parquet")
    oof = pd.read_parquet(reference_run/"oof_predictions.parquet")
    for table in (records, oof):
        if table.sample_id.duplicated().any() or not np.array_equal(table.sample_id, fit.sample_id):
            raise ValueError("Saved F fitting/OOF identity order differs.")
    weights = old.balanced_weights(fit)
    if not np.array_equal(weights, records.weight.to_numpy(float)) or more.weight_hash(fit) != manifest["fit_weight_sha256"]:
        raise ValueError("Saved F fitting weights differ.")
    folds = multi.month_folds(fit)
    if (not np.array_equal(folds, oof.fold.to_numpy()) or
            not np.isfinite(oof.oof_offset_c.to_numpy()).all() or
            not np.allclose(old.target_offset(fit)-oof.oof_offset_c.to_numpy(), oof.residual_target_c, rtol=0, atol=1e-12)):
        raise ValueError("OOF assignment or residual target differs.")
    return fit, weights, oof, evaluation, known, areas


def saved_values(frame, path, base):
    saved = pd.read_parquet(path)
    if saved.sample_id.duplicated().any() or set(saved.sample_id) != set(frame.sample_id):
        raise ValueError(f"Saved comparison cohort changed: {path.name}")
    saved = saved.set_index("sample_id").loc[frame.sample_id]
    if not np.allclose(saved.lst_c, frame.lst_c, rtol=0, atol=1e-12):
        raise ValueError("Saved observation values differ.")
    predicted = frame.air_temperature_c.to_numpy(float)+base.predict(frame[list(multi.BASE)])
    if not np.allclose(predicted, saved.F_lst_c, rtol=0, atol=1e-10):
        raise ValueError("Frozen F predictions no longer reproduce.")
    return {name: saved[f"{name}_lst_c"].to_numpy(float) for name in ("E", "F", "G", "H")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference-run", "legacy-input", "legacy-run", "cache-root", "protocol", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    run, out = args.reference_run.resolve(), args.output.resolve()
    if out.exists():
        raise FileExistsError("Use a new immutable research output directory.")
    manifest = multi.verify_run(run)
    verify_inputs(manifest)
    if old.sha(args.legacy_input) != manifest["reference_hashes"]["legacy_2024_input"]:
        raise ValueError("Legacy input hash differs before its labels are read.")
    legacy_record = json.loads((args.legacy_run/"results.json").read_text())
    if (legacy_record["multisensor_freeze_sha256"] != old.sha(run/"post_evaluation_freeze.json") or
            legacy_record["predictions_sha256"] != old.sha(args.legacy_run/"predictions.parquet")):
        raise ValueError("Legacy comparison is not bound to frozen F/G/H.")
    fit, weights, oof, old_evaluation, known, areas = reconstruct_fit(manifest, run)
    fit, fit_audit = add_physics_features(fit, cache_root=args.cache_root)
    common = valid_physics(fit)
    if not common.any():
        raise ValueError("No physical-complete fitting data.")
    selected = fit.loc[common].reset_index(drop=True)
    selected_oof = oof.oof_offset_c.to_numpy(float)[common]
    folds = oof.fold.to_numpy(int)[common]
    selected_weights = weights[common]/weights[common].sum()
    support = multi.group_support(selected, folds)
    if not any(x["supported"] for x in support.values()):
        raise ValueError("No supported climate/phase group.")
    out.mkdir(parents=True)
    source_files = ("physics_correction.py", "physics_features.py", "physics_metrics.py")
    source_hashes = {name: old.sha(Path(__file__).with_name(name)) for name in source_files}
    records = {"version": VERSION, "specification": SPEC, "arms": ARMS,
               "reference_run": str(run), "reference_freeze_sha256": old.sha(run/"post_evaluation_freeze.json"),
               "reference_F_sha256": old.sha(run/"F.joblib"), "protocol_sha256": old.sha(args.protocol),
               "source_hashes": source_hashes, "fit_row_sha256": old.row_hash(fit),
               "common_fit_row_sha256": old.row_hash(selected), "base_fit_rows": len(fit),
               "correction_fit_rows": len(selected), "excluded_fit_rows": int((~common).sum()),
               "common_weight_sha256": hashlib.sha256(selected_weights.tobytes()).hexdigest(),
               "feature_audit": fit_audit, "support": support,
               "legacy_input_sha256": old.sha(args.legacy_input),
               "legacy_predictions_sha256": old.sha(args.legacy_run/"predictions.parquet"),
               "research_only": True, "all_2023_2024_evidence_already_inspected": True, "reserved_2025_opened": False}
    old.save_json(out/"manifest.json", records)
    fit[["sample_id", "region_id", "datetime_utc", "phase", "label_product", *PHYSICS, RAW[-1]]].assign(
        physics_complete=common, original_F_weight=weights).to_parquet(out/"fitting_proxy_audit.parquet", index=False)
    corrections = {}
    with threadpool_limits(limits=4):
        for kind in ARMS:
            correction = WeakCorrection.fit(selected, selected_oof, folds, selected_weights, kind, support)
            corrections[kind] = correction
            joblib.dump(correction, out/f"{kind}.joblib", compress=3)
            old.save_json(out/f"{kind}_coefficients.json", {"numeric_features": correction.names,
                "mean": correction.mean.tolist(), "scale": correction.scale.tolist(),
                "effective_global_utc_dates": correction.effective_dates,
                "coefficient_names": ["penalized_constant", *correction.names,
                    *[f"climate_{c}" for c in multi.CLIMATES], *[f"phase_{p}" for p in multi.PHASES]],
                "coefficients": correction.ridge.coef_.tolist()})
    if common.all():
        previous = joblib.load(run/"G.joblib")["model"].correction
        weak = corrections["W"]
        for name, current, original in (("mean", weak.mean, previous.mean),
                ("scale", weak.scale, previous.scale),
                ("coefficients", weak.ridge.coef_, previous.ridge.coef_)):
            if not np.allclose(current, original, rtol=0, atol=1e-10):
                raise AssertionError(f"Weak weather arm should retain G's learned {name} when all rows are available.")
        old.save_json(out/"weak_weather_reference_check.json", {"underlying_G_coefficients_reproduced": True,
            "only_application_strength_and_cap_differ": True})
    frozen = {p.name: old.sha(p) for p in out.iterdir() if p.is_file()}
    old.save_json(out/"fit_freeze.json", {"artifacts": frozen, "frozen_before_new_score_calculations": True})
    fit_freeze_sha = old.sha(out/"fit_freeze.json")
    print(json.dumps({"stage": "fit_frozen", "fit_rows": len(selected), "seconds": time.monotonic()-started}), flush=True)

    verify_experiment_freeze(records, frozen, args.protocol, out, fit_freeze_sha)
    multi.verify_run(run)
    verify_inputs(manifest)
    p = manifest["paths"]
    new_frame, registry = multi.load_new_evaluation(p["new_evaluation"], p["freshness_audit"],
        manifest["input_hashes"]["freshness_audit"], known)
    new_evaluation = multi.validate_new(new_frame, known, areas, evaluation=True,
        freshness_sha=manifest["input_hashes"]["freshness_audit"], registry=registry)
    verify_experiment_freeze(records, frozen, args.protocol, out, fit_freeze_sha)
    if old.sha(args.legacy_input) != records["legacy_input_sha256"]:
        raise ValueError("Legacy input changed before evaluation loading.")
    legacy = old.prepare_input(old.load_paired_input(args.legacy_input, evaluation_2024=True), evaluation_2024=True)
    v1 = joblib.load(p["baseline"])
    needed = tuple(dict.fromkeys([*multi.BASE, *v1["features"]]))
    legacy = legacy.loc[old.complete_rows(legacy, needed) & legacy.phase.isin(multi.PHASES)].sort_values("sample_id").copy()
    legacy["split"] = "legacy_test_2024"
    legacy["temporal_partition"] = "legacy_test_2024"
    if old.row_hash(legacy) != legacy_record["row_sha256"] or more.weight_hash(legacy) != legacy_record["weight_sha256"]:
        raise ValueError("Legacy evaluation identity or weight hash changed.")
    base = joblib.load(run/"F.joblib")["model"]
    all_metrics, all_pairs, evaluation_audits = [], [], {}
    from .physics_metrics import evaluate_cohort
    for family, frame, path in (("old", old_evaluation, run/"old_evaluation_predictions.parquet"),
                                ("newly_collected_2023", new_evaluation, run/"new_2023_predictions.parquet"),
                                ("legacy_2024", legacy, args.legacy_run/"predictions.parquet")):
        frame = frame.reset_index(drop=True)
        with threadpool_limits(limits=4):
            values = saved_values(frame, path, base)
        frame, audit = add_physics_features(frame, cache_root=args.cache_root)
        evaluation_audits[family] = audit
        adjustments, supported = {}, {}
        offset = values["F"]-frame.air_temperature_c.to_numpy(float)
        for kind, correction in corrections.items():
            adjustments[kind], supported[kind] = correction.predict(frame, offset)
            values[kind] = values["F"]+adjustments[kind]
            if (not np.array_equal(values[kind][~supported[kind]], values["F"][~supported[kind]]) or
                    np.abs(adjustments[kind]).max(initial=0) > 1+1e-12):
                raise AssertionError("Weak correction support or bound was violated.")
        if not all(np.array_equal(supported["W"], supported[k]) for k in ARMS):
            raise AssertionError("Correction arms have different support.")
        fields = ["sample_id", "region_id", "datetime_utc", "phase", "split", "season", "air_group", "snow_group",
                  "acquisition_id", "label_product", "climate_class", "weight_surface_group", "utc_day", "lst_c", "air_temperature_c", *PHYSICS, RAW[-1]]
        export = frame[fields].copy()
        for name, v in values.items(): export[f"{name}_lst_c"] = v
        for name in ARMS:
            export[f"{name}_adjustment_c"] = adjustments[name]
            export[f"{name}_supported"] = supported[name]
        export.to_parquet(out/f"{family}_predictions.parquet", index=False)
        for split, indexes in frame.groupby("split", observed=True).indices.items():
            cohort = family if family == "legacy_2024" else f"{family}_{split}"
            result, pairs = evaluate_cohort(frame.iloc[indexes], {k:v[indexes] for k,v in values.items()},
                {k:v[indexes] for k,v in adjustments.items()}, {k:v[indexes] for k,v in supported.items()}, cohort)
            all_metrics.extend(result); all_pairs.extend(pairs)
        print(json.dumps({"stage": "scored", "family": family, "rows": len(frame)}), flush=True)
    pd.DataFrame(all_metrics).to_csv(out/"metrics.csv", index=False)
    pd.DataFrame(all_pairs).to_csv(out/"paired_deltas.csv", index=False)
    verify_experiment_freeze(records, frozen, args.protocol, out, fit_freeze_sha, tuple(evaluation_audits.values()))
    multi.verify_run(run)
    verify_inputs(manifest)
    old.save_json(out/"results.json", {"status": "all_fixed_arms_evaluated_no_promotion", "specification": SPEC,
        "fitting_rows": len(selected), "base_fitting_rows": len(fit), "evaluation_feature_audits": evaluation_audits,
        "metric_rows": len(all_metrics), "paired_rows": len(all_pairs), "elapsed_seconds": time.monotonic()-started,
        "reserved_2025_opened": False, "production_model_sha256": old.sha(p["baseline"]),
        "limitations": ["Repeated exploratory 2023/2024 evidence, not a new blind evaluation.",
            "No uncertainty interval refit or universal physical constraint.",
            "WP versus WR matches source information, weights and penalty; numeric feature counts differ.",
            "All-weather/global accuracy is not established by clear-source pilot observations."]})
    old.save_json(out/"post_evaluation_freeze.json", {"artifacts": {p.name: old.sha(p) for p in out.iterdir() if p.is_file()},
        "research_only": True, "auto_promotion": False, "reserved_2025_opened": False})
    print(json.dumps({"status": "complete", "output": str(out), "seconds": time.monotonic()-started}), flush=True)


if __name__ == "__main__":
    from .physics_correction import main as canonical_main
    canonical_main()
