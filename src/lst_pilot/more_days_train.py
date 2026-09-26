"""Fit one research model after adding dates; preserve every original test row.

No acquisition, feature selection, scenario substitutions or production changes.
The separate 2024 command verifies completed frozen fitting and evaluation first.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from pyproj import Transformer
from threadpoolctl import threadpool_limits

from . import option_b_train as original
from .option_b_cohort import spatial_flags

VERSION = "more-days-fit-only-v1"
SPLITS = ("development", "calibration", "heldout_region", "heldout_spatial")
BASE = original.BASE_FEATURES


def weight_hash(frame):
    return hashlib.sha256(original.balanced_weights(frame).tobytes()).hexdigest()


def load_additions(path):
    stamps = pd.to_datetime(pd.read_parquet(path, columns=["datetime_utc"]).datetime_utc,
                            utc=True, errors="raise")
    if stamps.isna().any() or not stamps.between(pd.Timestamp("2021-01-01", tz="UTC"),
            pd.Timestamp("2023-01-01", tz="UTC"), inclusive="left").all():
        raise ValueError("New fitting dates must be 2021--2022; thermal columns were not loaded.")
    return pd.read_parquet(path)


def load_reference(run_dir, input_path, baseline_path):
    """Bind original rows, preprocessing, model and weights to their saved hashes."""
    run = Path(run_dir)
    freeze, a_path = original.verify_frozen_run(run, baseline_path)
    if freeze["selected_candidate"] != "A":
        raise ValueError("This fixed ablation compares the frozen 40-feature A model.")
    manifest = json.loads((run / "frozen_manifest.json").read_text())
    if original.sha(input_path) != manifest["input_sha256"]:
        raise ValueError("Original paired input hash changed.")
    if original.sha(original.__file__) != manifest["source_sha256"]:
        raise ValueError("Original preprocessing/weighting implementation changed.")
    if original.sha(Path(original.__file__).with_name("model.py")) != manifest["model_dependency_sha256"]:
        raise ValueError("Original estimator implementation changed.")
    frame = original.prepare_input(original.load_paired_input(input_path))
    cohorts, evaluation, _ = original.candidate_cohorts(frame)
    fit = cohorts["A"].copy()
    recorded = manifest["candidates"]["A"]
    if original.row_hash(fit) != recorded["fit_sample_id_sha256"] or weight_hash(fit) != recorded["weight_sha256"]:
        raise ValueError("Reconstructed original fitting rows or weights differ from A.")
    result = json.loads((run / "results.json").read_text())
    for split in SPLITS:
        group = evaluation.loc[evaluation.split.eq(split)]
        expected = result["candidates"]["A"]["evaluation"][split]["model"]["overall"]
        if len(group) != expected["n"] or (len(group) and original.row_hash(group) != expected["sample_id_sha256"]):
            raise ValueError(f"Original evaluation row identities changed: {split}")
    a = joblib.load(a_path)
    if tuple(a["features"]) != BASE or a["config"] != asdict(original.CONFIG):
        raise ValueError("A features or estimator capacity do not match the fixed ablation.")
    return frame, fit, evaluation, a, joblib.load(baseline_path), manifest


def validate_additions(frame, old_frame, areas):
    """Require an already admitted fitting-only table, then check its boundaries."""
    required = {"research_admissibility_reason", "grid_row", "grid_col"}
    if not required.issubset(frame):
        raise ValueError(f"New fitting admission metadata missing: {sorted(required-set(frame))}")
    if not frame.research_admissibility_reason.eq("").fillna(False).all():
        raise ValueError("Every new row must have passed research admission.")
    data = original.prepare_input(frame)
    if data.empty:
        raise ValueError("No new fitting rows were supplied.")
    if not data.datetime_utc.dt.year.isin([2021, 2022]).all():
        raise ValueError("New fitting rows cannot contain 2023 or later observations.")
    if not data.cohort_origin.eq("expanded").all() or not data.label_product.eq("landsat_c2_l2").all():
        raise ValueError("Only newly paired expanded Landsat daytime rows may be added.")
    if not data.phase.eq("day").all():
        raise ValueError("Night observations and twilight are fixed; additions must be daytime.")
    if data.region_id.eq("cabauw").any() or not data.region_id.isin(areas).all():
        raise ValueError("Cabauw or an unknown pilot cannot enter new fitting rows.")
    if not original.complete_rows(data, BASE).all():
        raise ValueError("New fitting table contains incomplete base40 predictors.")
    for region, indexes in data.groupby("region_id").groups.items():
        area = areas[region]
        rows = data.loc[indexes, "grid_row"].to_numpy(float)
        cols = data.loc[indexes, "grid_col"].to_numpy(float)
        h, w = area["grid_shape"]
        if not (np.isfinite(rows).all() and np.isfinite(cols).all()
                and np.equal(rows, np.floor(rows)).all() and np.equal(cols, np.floor(cols)).all()
                and ((rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)).all()):
            raise ValueError("New row has invalid pilot grid coordinates.")
        left, _, _, top = area["extent_m"]
        x, y = Transformer.from_crs(4326, area["epsg"], always_xy=True).transform(
            data.loc[indexes, "longitude"].to_numpy(), data.loc[indexes, "latitude"].to_numpy())
        if not (np.hypot(x-(left+(cols+.5)*100), y-(top-(rows+.5)*100)) <= 1).all():
            raise ValueError("New row geographic coordinates disagree with its grid cell.")
    checked = spatial_flags(data, areas)
    if checked.spatial_holdout.any() or checked.in_holdout_buffer.any():
        raise ValueError("Reserved or buffer cells cannot enter the fitting-only additions.")
    if not checked.block_id.eq(data.block_id).all() or not data.split.eq("fit").all():
        raise ValueError("New row spatial/temporal classification disagrees with fitting-only admission.")
    if set(data.sample_id) & set(old_frame.sample_id) or set(data.acquisition_id) & set(old_frame.acquisition_id):
        raise ValueError("New sample/acquisition identities overlap the frozen original cohort.")
    known_dates = set(zip(old_frame.region_id, old_frame.utc_day))
    if any(key in known_dates for key in zip(data.region_id, data.utc_day)):
        raise ValueError("A new pilot-date overlaps an existing cohort date.")
    if data.groupby(["region_id", "utc_day"]).acquisition_id.nunique().gt(1).any():
        raise ValueError("New additions must contain one acquisition per pilot-date.")
    if data.duplicated(["region_id", "acquisition_id", "grid_row", "grid_col"]).any():
        raise ValueError("New table duplicates a physical acquisition/grid cell.")
    return data.sort_values("sample_id").copy()


def append_fitting(original_fit, additions, maximum=original.MAX_ROWS):
    capacity = maximum-len(original_fit)
    if capacity <= 0:
        raise ValueError("The fitting cap leaves no room for new dates.")
    chosen = original.cap_rows(additions, capacity)
    fit = pd.concat([original_fit, chosen], ignore_index=True).sort_values("sample_id")
    if fit.sample_id.duplicated().any() or not set(original_fit.sample_id).issubset(fit.sample_id):
        raise ValueError("Expanded fit did not preserve every original A fitting identity.")
    preserved = fit.set_index("sample_id").loc[original_fit.sample_id, original_fit.columns.drop("sample_id")]
    pd.testing.assert_frame_equal(preserved, original_fit.set_index("sample_id"), check_dtype=False)
    if not fit.split.eq("fit").all():
        raise ValueError("Non-fitting observations entered the estimator.")
    return fit, chosen


def score_matched(data, predictions):
    scores = {name: original.group_metrics(data, values) for name, values in predictions.items()}
    paired = {}
    if len(data):
        for name in ("A", "v1", "air_only"):
            paired[name] = {"same_sample_id_sha256": original.row_hash(data),
                "delta_mae_c": scores["E"]["overall"]["mae_c"]-scores[name]["overall"]["mae_c"],
                "delta_pixel_mae_c": scores["E"]["overall"]["unweighted_pixel_mae_c"]-scores[name]["overall"]["unweighted_pixel_mae_c"],
                "delta_centered_contrast_mae_c": scores["E"]["overall"]["centered_contrast_mae_c"]-scores[name]["overall"]["centered_contrast_mae_c"]}
    return {"metrics": scores, "paired_comparisons": paired, "support": original.support_counts(data),
            "row_sha256": original.row_hash(data), "weight_sha256": weight_hash(data)}


def predict_all(data, e, a, v1):
    air = data.air_temperature_c.to_numpy()
    predictions = {"air_only": air}
    for name, bundle in (("E", e), ("A", a), ("v1", v1)):
        predictions[name] = air+bundle["model"].predict(data[bundle["features"]]) if len(data) else np.array([])
    return predictions


def reuse_reference_predictions(data, predictions, path, *, legacy=False):
    """Verify numerical reproduction, then use the exact saved A/v1 values."""
    names = {"A": "candidate_lst_c" if legacy else "predicted_lst_c", "v1": "v1_lst_c"}
    columns = ["sample_id", *names.values()] + ([] if legacy else ["candidate"])
    recorded = pd.read_parquet(path, columns=columns)
    if not legacy: recorded = recorded.loc[recorded.candidate.eq("A")]
    if recorded.sample_id.duplicated().any() or set(recorded.sample_id) != set(data.sample_id):
        raise ValueError("Saved original predictions do not match the fixed evaluation identities.")
    recorded = recorded.set_index("sample_id").loc[data.sample_id]
    for name, column in names.items():
        values = recorded[column].to_numpy(float)
        if not np.allclose(values, predictions[name], rtol=0, atol=1e-10):
            raise ValueError(f"Frozen {name} predictions do not reproduce their saved row values.")
        predictions[name] = values
    return predictions


def run_experiment(original_input, additions_path, original_run, areas_path, baseline_path, protocol_path, output_dir,
                   *, original_2024_dir):
    started = time.monotonic()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("Use a new immutable more-days run directory.")
    old, old_fit, evaluation, a, v1, old_manifest = load_reference(original_run, original_input, baseline_path)
    old_2024_path = Path(original_2024_dir)/"results.json"
    old_2024 = json.loads(old_2024_path.read_text())  # Frozen report metadata only; no 2024 label arrays.
    if (old_2024["status"] != "legacy_2024_evaluated_no_refit_no_promotion"
            or old_2024["selected_candidate"] != "A"
            or old_2024["post_selection_freeze_sha256"] != original.sha(Path(original_run)/"post_selection_freeze.json")
            or old_2024["selected_model_sha256"] != original.sha(Path(original_run)/"A/model.joblib")
            or old_2024["baseline_sha256"] != original.sha(baseline_path)):
        raise ValueError("Existing 2024 report does not reference the unchanged original A/v1 models.")
    areas = {r["id"]: r for r in json.loads(Path(areas_path).read_text())["areas"]}
    additions = validate_additions(load_additions(additions_path), old, areas)
    fit, chosen = append_fitting(old_fit, additions)
    output.mkdir(parents=True)
    manifest = {"version": VERSION, "candidate": "E", "research_only": True, "auto_promotion": False,
        "selection": "One predeclared more-dates ablation; no candidate selection or reranking.",
        "features": list(BASE), "estimator": asdict(original.CONFIG), "thread_limit": 4,
        "protocol_sha256": original.sha(protocol_path), "source_sha256": original.sha(__file__),
        "dependencies_sha256": {name: original.sha(Path(__file__).with_name(name)) for name in
                                 ("option_b_train.py", "option_b_cohort.py", "model.py")},
        "original_input_sha256": original.sha(original_input), "new_input_sha256": original.sha(additions_path),
        "original_freeze_sha256": original.sha(Path(original_run)/"post_selection_freeze.json"),
        "original_A_sha256": original.sha(Path(original_run)/"A/model.joblib"),
        "original_predictions_sha256": original.sha(Path(original_run)/"evaluation_predictions.parquet"),
        "original_2024_results_sha256": original.sha(old_2024_path),
        "original_2024_input_sha256": old_2024["input_sha256"],
        "original_2024_predictions_sha256": original.sha(Path(original_2024_dir)/"predictions.parquet"),
        "v1_sha256": original.sha(baseline_path), "areas_sha256": original.sha(areas_path),
        "original_fit_rows": len(old_fit), "original_fit_row_sha256": original.row_hash(old_fit),
        "new_admitted_rows": len(additions), "new_chosen_rows": len(chosen), "new_rows_omitted_by_cap": len(additions)-len(chosen),
        "fit_rows": len(fit), "fit_row_sha256": original.row_hash(fit), "fit_weight_sha256": weight_hash(fit),
        "fit_weight_scale": "Original balanced algorithm, recomputed after adding dates, normalized to mean one for fitting.",
        "fit_support": original.support_counts(fit), "new_support": original.support_counts(chosen),
        "fixed_evaluation": {s: {"rows": len(g), "row_sha256": original.row_hash(g), "weight_sha256": weight_hash(g)}
                             for s in SPLITS for g in [evaluation.loc[evaluation.split.eq(s)]]},
        "original_A_fit_manifest": old_manifest["candidates"]["A"]}
    original.save_json(output/"manifest.json", manifest)
    fit[["sample_id", "region_id", "datetime_utc", "phase", "acquisition_id", "block_id"]].assign(
        added_date=fit.sample_id.isin(chosen.sample_id), weight=original.balanced_weights(fit)).to_parquet(output/"fitting_rows.parquet", index=False)
    with threadpool_limits(limits=4):
        estimator, _ = original.build_estimators(BASE, original.CONFIG)
        estimator.fit(fit[list(BASE)], original.target_offset(fit),
                      regressor__sample_weight=original.balanced_weights(fit)*len(fit))
        e = {"research_only": True, "candidate": "E", "features": list(BASE), "model": estimator,
             "config": asdict(original.CONFIG), "target": "lst_c - air_temperature_c", "auto_promotion": False,
             "training_climate_classes": sorted(fit.climate_class.unique().tolist())}
        joblib.dump(e, output/"model.joblib", compress=3)
        # Freeze the estimator before inspecting any new development/calibration/test predictions.
        original.save_json(output/"fit_freeze.json", {"candidate": "E", "model_sha256": original.sha(output/"model.joblib"),
            "manifest_sha256": original.sha(output/"manifest.json"), "fitting_rows_sha256": original.sha(output/"fitting_rows.parquet"),
            "model_frozen_before_evaluation": True, "refit_or_selection_allowed": False})
        predictions = reuse_reference_predictions(evaluation, predict_all(evaluation, e, a, v1),
                                                  Path(original_run)/"evaluation_predictions.parquet")
    cal = evaluation.split.eq("calibration").to_numpy()
    intervals = original.phase_calibration(evaluation.loc[cal], predictions["E"][cal])
    original.save_json(output/"calibration.json", intervals)
    results = {"status": "more_days_fitted_no_selection_no_promotion", "candidate": "E", "research_only": True,
        "auto_promotion": False, "model_sha256": original.sha(output/"model.joblib"), "evaluation": {},
        "phase_intervals": intervals, "warnings": ["All evaluation observations remain exactly the original rows.",
        "Previously inspected evaluations cannot establish blind confirmation or select another model.",
        "More days still represent sampled clear-sky satellite observations, not all-weather truth.",
        "Night observations are unchanged; new daytime training may still change shared-model nighttime predictions.",
        "2024 evaluation is a separate guarded command; 2025 remains unopened."]}
    original_result = json.loads((Path(original_run)/"results.json").read_text())
    for split in SPLITS:
        mask = evaluation.split.eq(split).to_numpy()
        data = evaluation.loc[mask]
        result = score_matched(data, {k:v[mask] for k,v in predictions.items()})
        if len(data):
            expected = original_result["candidates"]["A"]["evaluation"][split]["model"]["overall"]
            if not np.isclose(result["metrics"]["A"]["overall"]["mae_c"], expected["mae_c"], rtol=0, atol=1e-10):
                raise ValueError(f"Frozen A does not reproduce its original matched score: {split}")
        result["empirical_interval_coverage"] = original.interval_coverage(data, predictions["E"][mask], intervals)
        results["evaluation"][split] = result
    rows = evaluation[["sample_id", "region_id", "datetime_utc", "split", "phase", "air_group", "snow_group",
                       "acquisition_id", "block_id", "lst_c", "air_temperature_c"]].copy()
    for name, values in predictions.items(): rows[f"{name}_lst_c"] = values
    rows.to_parquet(output/"predictions.parquet", index=False)
    results["elapsed_seconds"] = time.monotonic()-started
    original.save_json(output/"results.json", results)
    original.save_json(output/"post_evaluation_freeze.json", {"candidate": "E", "original_freeze_sha256": manifest["original_freeze_sha256"],
        "fit_freeze_sha256": original.sha(output/"fit_freeze.json"), "results_sha256": original.sha(output/"results.json"),
        "calibration_sha256": original.sha(output/"calibration.json"), "refit_or_selection_allowed": False,
        "legacy_2024_opened": False, "blind_2025_opened": False})
    return results


def verify_experiment(run_dir, original_run, baseline_path):
    run = Path(run_dir)
    frozen = json.loads((run/"post_evaluation_freeze.json").read_text())
    if frozen["refit_or_selection_allowed"] or frozen["candidate"] != "E":
        raise ValueError("Invalid more-days evaluation freeze.")
    for name in ("fit_freeze", "results", "calibration"):
        if original.sha(run/f"{name}.json") != frozen[f"{name}_sha256"]:
            raise ValueError(f"More-days frozen artifact changed: {name}")
    fit = json.loads((run/"fit_freeze.json").read_text())
    for name,key in (("model.joblib","model_sha256"),("manifest.json","manifest_sha256"),("fitting_rows.parquet","fitting_rows_sha256")):
        if original.sha(run/name) != fit[key]: raise ValueError(f"More-days frozen artifact changed: {name}")
    manifest = json.loads((run/"manifest.json").read_text())
    if original.sha(__file__) != manifest["source_sha256"]:
        raise ValueError("More-days evaluation runner changed after fitting.")
    names = {"option_b_train.py", "option_b_cohort.py", "model.py"}
    if set(manifest["dependencies_sha256"]) != names:
        raise ValueError("More-days frozen dependency allowlist changed.")
    for name, expected in manifest["dependencies_sha256"].items():
        if original.sha(Path(__file__).with_name(name)) != expected:
            raise ValueError(f"More-days evaluation dependency changed after fitting: {name}")
    if not fit["model_frozen_before_evaluation"] or fit["refit_or_selection_allowed"]:
        raise ValueError("Estimator was not frozen before evaluation.")
    original.verify_frozen_run(original_run, baseline_path)
    if original.sha(Path(original_run)/"post_selection_freeze.json") != frozen["original_freeze_sha256"]:
        raise ValueError("Original comparison freeze changed.")
    return frozen


def evaluate_legacy_2024(input_path, run_dir, original_run, original_2024_dir, baseline_path, output_dir):
    frozen = verify_experiment(run_dir, original_run, baseline_path)
    output = Path(output_dir)
    if output.exists(): raise FileExistsError("Use a new immutable 2024 evaluation directory.")
    recorded = json.loads((Path(original_2024_dir)/"results.json").read_text())
    manifest = json.loads((Path(run_dir)/"manifest.json").read_text())
    if (original.sha(Path(original_2024_dir)/"results.json") != manifest["original_2024_results_sha256"]
            or original.sha(input_path) != manifest["original_2024_input_sha256"]
            or original.sha(Path(original_2024_dir)/"predictions.parquet") != manifest["original_2024_predictions_sha256"]):
        raise ValueError("2024 report or input changed after E fitting was frozen.")
    if recorded["status"] != "legacy_2024_evaluated_no_refit_no_promotion" or recorded["selected_candidate"] != "A":
        raise ValueError("The existing 2024 comparison must be the frozen A evaluation.")
    if original.sha(input_path) != recorded["input_sha256"] or recorded["post_selection_freeze_sha256"] != frozen["original_freeze_sha256"]:
        raise ValueError("2024 input or original freeze differs from the existing comparison.")
    data = original.prepare_input(original.load_paired_input(input_path, evaluation_2024=True), evaluation_2024=True)
    e = joblib.load(Path(run_dir)/"model.joblib")
    a = joblib.load(Path(original_run)/"A/model.joblib")
    v1 = joblib.load(baseline_path)
    if tuple(e["features"]) != BASE or e["candidate"] != "E": raise ValueError("Frozen E feature allowlist changed.")
    required = tuple(dict.fromkeys(list(BASE)+v1["features"]))
    common = data.loc[original.complete_rows(data, required) & data.phase.isin(["day", "night"])].sort_values("sample_id").copy()
    expected = recorded["metrics"]["candidate"]["overall"]
    if len(common) != expected["n"] or original.row_hash(common) != expected["sample_id_sha256"]:
        raise ValueError("2024 evaluation row identities changed.")
    common["split"] = "legacy_test_2024"; common["temporal_partition"] = "legacy_test_2024"
    with threadpool_limits(limits=4):
        predictions = reuse_reference_predictions(common, predict_all(common, e, a, v1),
                                                  Path(original_2024_dir)/"predictions.parquet", legacy=True)
    result = score_matched(common, predictions)
    if not np.isclose(result["metrics"]["A"]["overall"]["mae_c"], expected["mae_c"], rtol=0, atol=1e-10):
        raise ValueError("Frozen A does not reproduce its original 2024 comparison.")
    intervals = json.loads((Path(run_dir)/"calibration.json").read_text())
    result.update(status="more_days_2024_evaluated_no_refit_no_promotion", candidate="E", research_only=True, auto_promotion=False,
        input_sha256=original.sha(input_path), original_2024_results_sha256=original.sha(Path(original_2024_dir)/"results.json"),
        more_days_freeze_sha256=original.sha(Path(run_dir)/"post_evaluation_freeze.json"),
        empirical_interval_coverage=original.interval_coverage(common, predictions["E"], intervals),
        warnings=["Previously inspected 2024 regression comparison; not a blind test.", "No refitting, selection or promotion; 2025 remains unopened."])
    output.mkdir(parents=True)
    rows = common[["sample_id", "region_id", "datetime_utc", "phase", "air_group", "snow_group", "acquisition_id", "lst_c"]].copy()
    for name, values in predictions.items(): rows[f"{name}_lst_c"] = values
    rows.to_parquet(output/"predictions.parquet", index=False)
    original.save_json(output/"results.json", result)
    return result


def main():
    legacy = len(sys.argv)>1 and sys.argv[1]=="legacy-2024"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--baseline-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original-2024", type=Path, required=True)
    if legacy:
        parser.add_argument("--input", type=Path, required=True)
        parser.add_argument("--experiment", type=Path, required=True)
        args = parser.parse_args(sys.argv[2:])
        result = evaluate_legacy_2024(args.input,args.experiment,args.original_run,args.original_2024,args.baseline_model,args.output)
    else:
        parser.add_argument("--original-input", type=Path, required=True)
        parser.add_argument("--new-input", type=Path, required=True)
        parser.add_argument("--areas", type=Path, required=True)
        parser.add_argument("--protocol", type=Path, required=True)
        args = parser.parse_args()
        result = run_experiment(args.original_input,args.new_input,args.original_run,args.areas,args.baseline_model,args.protocol,args.output,
                                original_2024_dir=args.original_2024)
    print(json.dumps({"status":result["status"],"output":str(args.output)}))


if __name__ == "__main__": main()
