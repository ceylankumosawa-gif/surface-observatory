"""Preserved daytime London refinement: fixed blocks, dates and held-out tests.

Acquisition uses the serving renderer's optical-only features. Thermal data are
joined afterwards solely as labels. No user-selected error polygons are used.
Heavy acquisition/fitting is intended for the remote pilot server only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from pyproj import Transformer
import rasterio
from threadpoolctl import threadpool_limits

VERSION = "london-refinement-v2-fixed-blocks"
LONDON = "greater_london"
SEED = 2708
HOT_C = 35.0
COLD_C = 15.0


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime, Path)):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, default=json_default, allow_nan=False) + "\n")
    temporary.replace(path)


def scientific_signature():
    from . import raster, satellite, terrain, context, weather, assemble, radiation
    functions = [satellite.qa_valid, satellite.scale_reflectance, satellite.scale_temperature,
                 satellite.surface_descriptors, raster._warp, raster.aggregate_optical, raster.read_optical,
                 raster.terrain_arrays, terrain.terrain_descriptors, raster.worldcover_fractions,
                 context.add_context, weather.enrich_weather, weather.fetch_archive,
                 assemble.attach_stations, radiation.add_radiation]
    return {f.__module__ + "." + f.__name__: hashlib.sha256(inspect.getsource(f).encode()).hexdigest()
            for f in functions}


def regular_blocks():
    """Six equal projected blocks fixed without inspecting residuals or labels."""
    result = []
    for iy, y in enumerate((168400, 188400)):
        for ix, x in enumerate((512800, 532800, 552800)):
            result.append({"id": f"london_b{iy * 3 + ix}",
                           "bounds_m": [x - 2500, y - 2500, x + 2500, y + 2500],
                           "spatial_holdout": (iy, ix) == (1, 2)})
    return result


def block_polygon(block, epsg):
    left, bottom, right, top = block["bounds_m"]
    # Densify in the projected CRS before conversion; the intended grid is fixed.
    ring = [(x, bottom) for x in np.linspace(left, right, 21)]
    ring += [(right, y) for y in np.linspace(bottom, top, 21)[1:]]
    ring += [(x, top) for x in np.linspace(right, left, 21)[1:]]
    ring += [(left, y) for y in np.linspace(top, bottom, 21)[1:]]
    transform = Transformer.from_crs(epsg, 4326, always_xy=True)
    return {"type": "Polygon", "coordinates": [[list(transform.transform(x, y)) for x, y in ring]]}


def create_protocol(root, output):
    root, output = Path(root), Path(output)
    path = output / "protocol.json"
    if path.exists():
        return json.loads(path.read_text())
    manifest = json.loads((root / "runs/pilot_v1_satellite/satellite_manifest.json").read_text())
    items = {item["id"]: item for p in (root / "runs/pilot_v1_satellite/stac").glob("*.json")
             for item in json.loads(p.read_text())["items"]}
    scenes = [{"scene_id": entry["scene_id"],
               "datetime_utc": items[entry["scene_id"]]["properties"]["datetime"]}
              for entry in manifest["scenes"] if entry["region_id"] == LONDON and entry["samples"] > 0]
    region = next(r for r in manifest["regions"] if r["id"] == LONDON)
    protocol = {
        "version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_model_sha256": digest(root / "runs/pilot_v1_model/model.joblib"),
        "baseline_input_sha256": digest(root / "runs/pilot_v1_assembly/model_input.parquet"),
        "scientific_signature": scientific_signature(), "region": region,
        "blocks": regular_blocks(), "scenes": sorted(scenes, key=lambda x: x["datetime_utc"]),
        "training": "All original non-London 2021–2022 fitting rows, original London 2021–2022 and renderer-derived London 2021–2022 labels, excluding reserved block plus 1 km buffer.",
        "development": "2023 dates before July 1, excluding reserved London spatial block plus 1 km buffer. Configurations selected using these dates only.",
        "calibration": "2023 dates from July 1 onward, excluding reserved London spatial block plus 1 km buffer. Never used to select configurations.",
        "test": "Original 2024 dataset kept unchanged for paired baseline/candidate comparison; separate predetermined London 2024 dense blocks. No test rows used for fitting or model selection.",
        "spatial_test": "Reserved north-east 5 km block on 2021–2023 dates; 1 km buffer removed from all London fitting/development/calibration rows.",
        "independence": "Existing 2024 results were previously inspected during pilot reporting. This is a preserved retrospective test, not a new blind external validation.",
        "surface_sampling": "All valid renderer pixels in fixed windows, with independent WorldCover>=0.8, optical>=0.8, complete model features and separate finite thermal label.",
        "weighting": "Equal region/date total weight; London dates have twice the total weight of another region/date. Pixel number within a date cannot dominate weighting.",
        "candidate_configs": [{"name": "same_capacity", "max_iter": 150, "max_leaf_nodes": 15, "max_depth": 6, "min_samples_leaf": 30},
                              {"name": "more_capacity", "max_iter": 250, "max_leaf_nodes": 31, "max_depth": 8, "min_samples_leaf": 40}],
        "selection_score": "Mean of London-development MAE and other-region-development MAE; all settings frozen before acquiring supplementary pixels.",
        "tail_thresholds_c": {"cold_at_most": COLD_C, "hot_at_least": HOT_C},
        "acceptance": "Original London2024 MAE improves>=5%, its hot/cold-tail MAEs do not worsen, other-region2024 MAE increases<=5%; dense London2024 MAE improves>=5% and hot/cold tails do not worsen. No automatic activation.",
        "seed": SEED, "cpu_limit": 4, "maximum_training_rows": 100000,
        "user_selected_error_polygons_used": False,
        "nighttime_or_hourly_validation": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    save_json(path, protocol)
    return protocol


def _join_renderer_labels(directory, block, scene):
    frame = pd.read_parquet(directory / "features.parquet")
    frame.attrs = {}
    with rasterio.open(directory / "prediction.tif") as src:
        try:
            band = src.descriptions.index("observed_lst_c") + 1
        except ValueError:
            return pd.DataFrame()
        observed = src.read(band, masked=True).filled(np.nan).ravel()
    frame["lst_c"] = observed[frame["_raster_position"].to_numpy(dtype=int)]
    frame = frame.loc[np.isfinite(frame.lst_c)].copy()
    frame["scene_id"] = scene["scene_id"]
    frame["source_scene_id"] = scene["scene_id"]
    frame["label_source"] = "Landsat C2 L2 thermal; renderer-independent optical mask; ST uncertainty<=3 K"
    frame["refinement_block"] = block["id"]
    frame["refinement_source"] = VERSION
    return frame


def acquire(root, output, maximum_jobs=None):
    root, output = Path(root), Path(output)
    protocol = create_protocol(root, output)
    if protocol["scientific_signature"] != scientific_signature():
        raise ValueError("Scientific preprocessing changed since protocol creation; review before using a new protocol.")
    items = {item["id"]: item for p in (root / "runs/pilot_v1_satellite/stac").glob("*.json")
             for item in json.loads(p.read_text())["items"]}
    from .raster import render_raster
    log = output / "acquisition.json"
    records = json.loads(log.read_text()) if log.exists() else []
    done = {(r["scene_id"], r["block_id"]): r for r in records}
    count = 0
    for scene in protocol["scenes"]:
        for block in protocol["blocks"]:
            key = (scene["scene_id"], block["id"])
            if key in done:
                continue
            if maximum_jobs is not None and count >= maximum_jobs:
                return
            if protocol["scientific_signature"] != scientific_signature():
                raise ValueError("Scientific preprocessing changed during acquisition; paused before another window.")
            directory = output / "windows" / f"{scene['scene_id']}_{block['id']}"
            label_file = output / "labels" / f"{scene['scene_id']}_{block['id']}.parquet"
            started = time.monotonic()
            record = {"scene_id": scene["scene_id"], "block_id": block["id"], "datetime_utc": scene["datetime_utc"]}
            print("acquire", len(records) + 1, "/", len(protocol["scenes"]) * len(protocol["blocks"]), key, flush=True)
            try:
                if not (directory / "provenance.json").exists():
                    render_raster(protocol["region"], block_polygon(block, protocol["region"]["epsg"]),
                                  items[scene["scene_id"]], scene["datetime_utc"], directory,
                                  root / "cache", root / "runs/pilot_v1_model/model.joblib",
                                  mode="observed", air_override=None)
                frame = _join_renderer_labels(directory, block, scene)
                label_file.parent.mkdir(parents=True, exist_ok=True)
                if not frame.empty:
                    frame.to_parquet(label_file, index=False)
                record.update(status="complete", labelled_pixels=len(frame), label_file=str(label_file) if len(frame) else None)
            except Exception as exc:
                record.update(status="unavailable", labelled_pixels=0, error=f"{type(exc).__name__}: {str(exc)[:1000]}")
            record["elapsed_seconds"] = round(time.monotonic() - started, 2)
            records.append(record)
            save_json(log, records)
            count += 1
            print(json.dumps(record), flush=True)


def _spatial_reserved(frame, protocol, buffer_m=1000):
    block = next(b for b in protocol["blocks"] if b["spatial_holdout"])
    left, bottom, right, top = block["bounds_m"]
    return (frame.region_id.eq(LONDON) & frame.pixel_x.between(left - buffer_m, right + buffer_m)
            & frame.pixel_y.between(bottom - buffer_m, top + buffer_m))


def partitions(frame, protocol):
    """Whole UTC dates partition time; reserved London block never enters fit/dev/cal."""
    dt = pd.to_datetime(frame.datetime_utc, utc=True)
    reserved = _spatial_reserved(frame, protocol)
    year = dt.dt.year
    return {
        "train": year.le(2022) & ~reserved,
        "development": year.eq(2023) & dt.dt.month.lt(7) & ~reserved,
        "calibration": year.eq(2023) & dt.dt.month.ge(7) & ~reserved,
        "test_2024": year.eq(2024),
        "spatial_test": reserved & year.le(2023),
    }


def row_weights(frame):
    keys = pd.DataFrame({"region": frame.region_id, "date": pd.to_datetime(frame.datetime_utc, utc=True).dt.floor("D")})
    size = keys.groupby(["region", "date"])["date"].transform("size")
    weights = np.where(frame.region_id.eq(LONDON), 2., 1.) / size.to_numpy()
    return weights / weights.mean()


def error_stats(observed, predicted):
    observed, predicted = np.asarray(observed, float), np.asarray(predicted, float)
    good = np.isfinite(observed) & np.isfinite(predicted)
    err = predicted[good] - observed[good]
    if not len(err):
        return {"n": 0, "mae_c": None, "rmse_c": None, "bias_c": None, "p90_abs_c": None}
    return {"n": len(err), "mae_c": float(np.abs(err).mean()), "rmse_c": float(np.sqrt(np.mean(err ** 2))),
            "bias_c": float(err.mean()), "p90_abs_c": float(np.quantile(np.abs(err), .9))}


def detailed_metrics(frame, column):
    result = error_stats(frame.lst_c, frame[column])
    result.update(region_count=int(frame.region_id.nunique()), utc_days=int(pd.to_datetime(frame.datetime_utc, utc=True).dt.floor("D").nunique()))
    result["cold_tail"] = error_stats(frame.loc[frame.lst_c.le(COLD_C), "lst_c"], frame.loc[frame.lst_c.le(COLD_C), column])
    result["hot_tail"] = error_stats(frame.loc[frame.lst_c.ge(HOT_C), "lst_c"], frame.loc[frame.lst_c.ge(HOT_C), column])
    return result


def compare(frame, baseline, candidate, features):
    frame = frame.copy()
    frame["baseline_lst_c"] = frame.air_temperature_c + baseline["model"].predict(frame[features])
    frame["candidate_lst_c"] = frame.air_temperature_c + candidate.predict(frame[features])
    result = {"baseline": detailed_metrics(frame, "baseline_lst_c"), "candidate": detailed_metrics(frame, "candidate_lst_c")}
    result["per_region"] = {str(k): {"baseline": detailed_metrics(g, "baseline_lst_c"), "candidate": detailed_metrics(g, "candidate_lst_c")}
                            for k, g in frame.groupby("region_id")}
    frame["utc_day"] = pd.to_datetime(frame.datetime_utc, utc=True).dt.strftime("%Y-%m-%d")
    result["per_region_day"] = {str(r) + "/" + str(d): {"baseline": detailed_metrics(g, "baseline_lst_c"), "candidate": detailed_metrics(g, "candidate_lst_c")}
                                for (r, d), g in frame.groupby(["region_id", "utc_day"])}
    for name, values in {"ndvi": pd.cut(frame.ndvi, [-np.inf, .2, .4, .6, .8, np.inf]),
                         "ndbi": pd.cut(frame.ndbi, [-np.inf, -.2, 0, .2, np.inf])}.items():
        result["by_" + name] = {str(k): {"baseline": detailed_metrics(g, "baseline_lst_c"), "candidate": detailed_metrics(g, "candidate_lst_c")}
                                for k, g in frame.groupby(values, observed=True)}
    return result, frame


def _tail_no_worse(baseline, candidate):
    return all(baseline[t]["n"] == 0 or candidate[t]["mae_c"] <= baseline[t]["mae_c"] for t in ("cold_tail", "hot_tail"))


def fit(root, output):
    from .model import ModelConfig, build_estimators, target_offset, residual_interval_radius
    root, output = Path(root), Path(output)
    protocol = json.loads((output / "protocol.json").read_text())
    model_dir = output / "model"
    if (model_dir / "model.joblib").exists():
        raise FileExistsError("Candidate model already exists; never overwrite an evaluated model.")
    baseline_path = root / "runs/pilot_v1_model/model.joblib"
    input_path = root / "runs/pilot_v1_assembly/model_input.parquet"
    if digest(baseline_path) != protocol["baseline_model_sha256"] or digest(input_path) != protocol["baseline_input_sha256"]:
        raise ValueError("The fixed baseline model/input changed.")
    baseline = joblib.load(baseline_path)
    features = list(baseline["features"])
    original = pd.read_parquet(input_path)
    original.attrs = {}
    original["refinement_source"] = "original_v1"
    paths = sorted((output / "labels").glob("*.parquet"))
    if not paths:
        raise ValueError("No supplementary labels were acquired.")
    extra = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    extra.attrs = {}
    original.datetime_utc = pd.to_datetime(original.datetime_utc, utc=True)
    extra.datetime_utc = pd.to_datetime(extra.datetime_utc, utc=True)
    # Updated serving-consistent features supersede the sparse feature row only
    # within fitting/development/calibration data. Original test remains separate.
    combined = pd.concat([original, extra], ignore_index=True)
    combined = combined.drop_duplicates(["region_id", "datetime_utc", "pixel_id"], keep="last")
    split = partitions(combined, protocol)
    train = combined.loc[split["train"]].copy()
    dev = combined.loc[split["development"]].copy()
    cal = combined.loc[split["calibration"]].copy()
    if len(train) > protocol["maximum_training_rows"]:
        raise ValueError("Fixed row cap exceeded; do not silently change the sampling plan.")
    if min(len(train), len(dev), len(cal)) < 100 or train.datetime_utc.dt.floor("D").nunique() < 4:
        raise ValueError("Insufficient independent date partitions.")
    if any(len(part[part.region_id.eq(LONDON)]) < 50 for part in (train, dev, cal)):
        raise ValueError("Insufficient London rows in fit/development/calibration.")
    model_dir.mkdir(parents=True, exist_ok=True)
    train[features + ["lst_c", "region_id", "datetime_utc", "pixel_id"]].to_parquet(output / "fitting_rows.parquet", index=False)
    selection = []
    candidates = {}
    with threadpool_limits(limits=4):
        for choice in protocol["candidate_configs"]:
            name = choice["name"]
            params = {k: v for k, v in choice.items() if k != "name"}
            config = ModelConfig(features=tuple(features), heldout_regions=(), random_state=SEED,
                                 train_end="2022-12-31", calibration_end="2023-12-31", **params)
            candidate, _ = build_estimators(features, config)
            started = time.monotonic()
            candidate.fit(train[features], target_offset(train), regressor__sample_weight=row_weights(train))
            predicted = dev.air_temperature_c + candidate.predict(dev[features])
            london = dev.region_id.eq(LONDON)
            lm = error_stats(dev.loc[london, "lst_c"], predicted[london])
            other = error_stats(dev.loc[~london, "lst_c"], predicted[~london])
            score = .5 * lm["mae_c"] + .5 * other["mae_c"]
            selection.append({"name": name, "config": asdict(config), "development_london": lm, "development_other_regions": other,
                              "score": score, "fit_seconds": round(time.monotonic() - started, 3)})
            candidates[name] = candidate
        chosen = min(selection, key=lambda x: x["score"])
        candidate = candidates[chosen["name"]]
        # Freeze the chosen configuration/model before any 2024 prediction.
        save_json(output / "selection_frozen_before_test.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "candidates": selection, "chosen": chosen["name"], "test_used": False})
        radius = residual_interval_radius(cal.lst_c - (cal.air_temperature_c + candidate.predict(cal[features])), .9)
        config = ModelConfig(features=tuple(features), heldout_regions=(), random_state=SEED,
                             train_end="2022-12-31", calibration_end="2023-12-31",
                             **{k: v for k, v in next(c for c in protocol["candidate_configs"] if c["name"] == chosen["name"]).items() if k != "name"})
        _, linear = build_estimators(features, config)
        linear.fit(train[features], target_offset(train), regressor__sample_weight=row_weights(train))
        bundle = {"schema_version": 1, "model": candidate, "linear_baseline": linear, "features": features,
                  "target": "lst_c - air_temperature_c", "interval_radius_c": radius,
                  "config": asdict(config), "training_climate_classes": sorted(train.climate_class.unique()), "smoke_only": False,
                  "refinement_version": VERSION, "protocol_sha256": digest(output / "protocol.json"), "baseline_sha256": digest(baseline_path)}
        joblib.dump(bundle, model_dir / "model.joblib", compress=3)
        groups = {"original_2024": original.loc[original.datetime_utc.dt.year.eq(2024)].copy(),
                  "dense_london_2024": extra.loc[extra.datetime_utc.dt.year.eq(2024)].copy(),
                  "reserved_spatial_2021_2023": combined.loc[split["spatial_test"]].copy(),
                  "development_2023": dev, "calibration_2023": cal}
        evaluations = {}
        for name, data in groups.items():
            if data.empty:
                evaluations[name] = {"status": "unavailable"}
                continue
            evaluation, predictions = compare(data, baseline, candidate, features)
            evaluations[name] = evaluation
            predictions.to_parquet(output / (name + "_predictions.parquet"), index=False)
    orig_london = evaluations["original_2024"]["per_region"][LONDON]
    orig_other = original.loc[original.datetime_utc.dt.year.eq(2024) & ~original.region_id.eq(LONDON)]
    other, _ = compare(orig_other, baseline, candidate, features)
    dense = evaluations["dense_london_2024"]
    acceptance = {
        "original_london_mae_improves_5pct": orig_london["candidate"]["mae_c"] <= .95 * orig_london["baseline"]["mae_c"],
        "original_london_tails_not_worse": _tail_no_worse(orig_london["baseline"], orig_london["candidate"]),
        "other_regions_mae_within_5pct": other["candidate"]["mae_c"] <= 1.05 * other["baseline"]["mae_c"],
        "dense_london_mae_improves_5pct": dense.get("candidate", {}).get("mae_c", np.inf) <= .95 * dense.get("baseline", {}).get("mae_c", 0),
        "dense_london_tails_not_worse": "candidate" in dense and _tail_no_worse(dense["baseline"], dense["candidate"]),
    }
    report = {"version": VERSION, "status": "candidate_passes_predeclared_checks" if all(acceptance.values()) else "candidate_not_recommended",
              "production_activated": False, "selected_candidate": chosen["name"], "selection": selection,
              "counts": {"train": len(train), "development": len(dev), "calibration": len(cal), "supplementary": len(extra)},
              "training_region_counts": train.region_id.value_counts().to_dict(), "training_date_counts": train.groupby("region_id").datetime_utc.nunique().to_dict(),
              "interval": {"radius_c": radius, "nominal_coverage": .9, "guaranteed_coverage": False},
              "acceptance": acceptance, "evaluations": evaluations, "other_regions_2024": other,
              "model_sha256": digest(model_dir / "model.joblib"), "baseline_model_sha256": digest(baseline_path),
              "baseline_input_sha256_after": digest(input_path), "protocol_sha256": digest(output / "protocol.json"),
              "limitations": ["Only eight London dates, with two 2024 dates; spatially correlated pixels do not supply independent weather events.",
                              "Existing 2024 pilot metrics were already inspected; this is preserved retrospective assessment, not a fresh blind test.",
                              "The new London rows use serving-consistent masks, but original global rows retain their original preprocessing.",
                              "Daytime clear-sky labels only; no nighttime, cloudy-sky or arbitrary-hour validation.",
                              "London is now represented in training; new metrics measure local temporal/spatial transfer, not unseen-city transfer."]}
    save_json(output / "refinement_metrics.json", report)
    print(json.dumps({"status": report["status"], "counts": report["counts"], "chosen": chosen["name"], "acceptance": acceptance,
                      "London_original_2024": orig_london, "London_dense_2024": {k: dense[k] for k in ("baseline", "candidate") if k in dense},
                      "other_regions_2024": {k: other[k] for k in ("baseline", "candidate")}}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "acquire", "fit", "correction-plan", "correction-history", "correction-fit", "correction-test"])
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="runs/pilot_v2_refinement")
    parser.add_argument("--max-jobs", type=int)
    args = parser.parse_args()
    if args.command.startswith("correction-"):
        action = args.command.removeprefix("correction-")
        {"plan": correction_plan, "history": correction_history,
         "fit": correction_fit, "test": correction_test}[action](args.root, args.output)
    elif args.command == "plan":
        create_protocol(args.root, args.output)
    elif args.command == "acquire":
        with threadpool_limits(limits=4):
            acquire(args.root, args.output, args.max_jobs)
    else:
        fit(args.root, args.output)


CORRECTION_VERSION = "cfb-capped-ridge-v3-fresh2025"


def correction_terms(frame):
    """Only existing daytime optical/shortwave predictors; no target or scene ID."""
    ndvi = np.asarray(frame.ndvi, float)
    ndbi = np.asarray(frame.ndbi, float)
    albedo = np.asarray(frame.albedo_proxy, float)
    sun = np.maximum(np.asarray(frame.shortwave_down_w_m2, float), 0) / 1000
    return np.column_stack([ndvi, ndbi, albedo, sun, ndvi * sun, ndbi * sun, albedo * sun])


class CfbResidualModel:
    """Frozen global offset estimator plus a bounded Cfb daytime correction."""
    def __init__(self, baseline, ridge, center, scale, cap_c=2.):
        self.baseline, self.ridge = baseline, ridge
        self.center, self.scale = np.asarray(center), np.asarray(scale)
        self.cap_c = cap_c

    def correction(self, frame):
        result = np.zeros(len(frame), float)
        active = frame.climate_class.eq("Cfb").to_numpy() & frame.solar_elevation_deg.gt(0).to_numpy()
        if active.any():
            values = (correction_terms(frame.loc[active]) - self.center) / self.scale
            result[active] = np.clip(self.ridge.predict(values), -self.cap_c, self.cap_c)
        return result

    def predict(self, frame):
        return self.baseline.predict(frame) + self.correction(frame)


def correction_plan(root, output):
    """Freeze method first, then unsigned 2025 metadata; never read labels here."""
    from .satellite import search_scenes, scene_coverage_fraction
    root, output = Path(root), Path(output)
    path = output / "protocol.json"
    if path.exists():
        protocol = json.loads(path.read_text())
    else:
        manifest = json.loads((root / "runs/pilot_v1_satellite/satellite_manifest.json").read_text())
        regions = [r for r in manifest["regions"] if r["id"] in (LONDON, "cabauw")]
        historical = {i["id"]: i for p in (root / "runs/pilot_v1_satellite/stac").glob("*.json")
                      for i in json.loads(p.read_text())["items"]}
        scenes = [{"region_id": s["region_id"], "scene_id": s["scene_id"],
                   "datetime_utc": historical[s["scene_id"]]["properties"]["datetime"]}
                  for s in manifest["scenes"] if s["region_id"] in (LONDON, "cabauw") and s["samples"] > 0
                  and pd.Timestamp(historical[s["scene_id"]]["properties"]["datetime"]).year <= 2023]
        blocks = {}
        for region in regions:
            w, s, e, n = region["extent_m"]
            cx, cy = (w+e)/2, (s+n)/2
            blocks[region["id"]] = [{"id": f"{region['id']}_central_{label}",
                                    "bounds_m": [cx-2500, cy+dy-2500, cx+2500, cy+dy+2500]}
                                   for label, dy in (("south", -10000), ("north", 10000))]
        protocol = {"version": CORRECTION_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
                    "baseline_model_sha256": digest(root / "runs/pilot_v1_model/model.joblib"),
                    "baseline_input_sha256": digest(root / "runs/pilot_v1_assembly/model_input.parquet"),
                    "scientific_signature": scientific_signature(), "regions": regions, "blocks": blocks,
                    "historical_scenes": sorted(scenes, key=lambda s: (s["region_id"], s["datetime_utc"])),
                    "training": "Original Cfb 2021–2022 rows plus exact-rendered historical blocks; deduplicate source/time/pixel, retain exact-rendered rows.",
                    "development": "Both 2023 dates in London and Cabauw, original plus exact-rendered blocks. Equal date weight. This replaces prior calibration use explicitly; it is not a new blind test.",
                    "correction": {"terms": ["ndvi", "ndbi", "albedo_proxy", "shortwave_down/1000", "ndvi*shortwave_down/1000", "ndbi*shortwave_down/1000", "albedo_proxy*shortwave_down/1000"],
                                   "ridge_alphas": [0.1, 1., 10.], "cap_c": 2., "active_climate": "Cfb",
                                   "outside_climate": "Exactly zero correction", "weights": "Equal region/date totals normalized to sum one; weighted centering and scaling."},
                    "development_gate": "London mean of per-date MAEs improves>=3%; neither London date worsens>0.1°C; Cabauw per-date mean MAE increase<=2% and<=0.05°C; neither Cabauw date worsens>0.1°C; hot>=35 and cold<=15 pooled tail MAE increase<=0.1°C in each city when>=50 rows. Select lowest mean of London/Cabauw date-balanced MAEs among eligible shrinkages, else stop without reading2025 labels.",
                    "fresh_selection": "One scene per calendar quarter in 2025 per region: at most64 unsigned metadata candidates, Tier1 L2SP Landsat8/9, cloud<=40%, footprint covers>=95% region; rank cloud ascending, distance from quarter midpoint ascending, scene ID. Missing quarter aborts without replacement; no label-based scene substitution.",
                    "fresh_test": "Two fixed blocks per region/quarter, source timestamp exactly equals target. Require>=3 labelled dates in each city,>=50 London hot and>=50 London cold rows. London date-balanced MAE improves>=5%; no London date worsens>0.1°C; Cabauw mean MAE increase<=2% and<=0.05°C; no Cabauw date worsens>0.1°C; hot/cold MAE increase<=0.1°C in both cities when>=50rows. Otherwise rejected or insufficient evidence. No second retune.",
                    "interval": "Retain original width for bundle compatibility, explicitly not recalibrated or validated for correction. Accuracy acceptance is not automatic production authorization.",
                    "previously_seen_2024": "Preserved audit only, not selection; original v1/v2 inputs and models never overwritten.",
                    "limits": {"maximum_windows": 48, "planned_windows": 40, "maximum_training_rows": 100000, "cpu_threads": 4},
                    "spatial_limitation": "Fixed-location temporal validation; no claim of new-city or independent spatial validation.",
                    "no_user_error_polygons": True, "nighttime_validation": False}
        output.mkdir(parents=True, exist_ok=True)
        save_json(path, protocol)
    metadata_path = output / "fresh_scene_selection.json"
    if not metadata_path.exists():
        selected = []
        for region in protocol["regions"]:
            for quarter, (start, end) in enumerate((("2025-01-01", "2025-04-01"), ("2025-04-01", "2025-07-01"), ("2025-07-01", "2025-10-01"), ("2025-10-01", "2026-01-01")), 1):
                items = search_scenes(region, start+"/"+end, output / "fresh_stac", max_scenes=1, max_candidates=64, cloud_cover_max=40)
                eligible = [i for i in items if scene_coverage_fraction(i, region) >= .95
                            and pd.Timestamp(start, tz="UTC") <= pd.Timestamp(i["properties"]["datetime"]) < pd.Timestamp(end, tz="UTC")]
                if not eligible:
                    raise ValueError(f"No eligible metadata-only scene for {region['id']} quarter {quarter}; no label-based replacement.")
                midpoint = pd.Timestamp(start, tz="UTC") + (pd.Timestamp(end, tz="UTC")-pd.Timestamp(start, tz="UTC"))/2
                chosen = min(eligible, key=lambda i: (i["properties"].get("eo:cloud_cover", 100), abs((pd.Timestamp(i["properties"]["datetime"])-midpoint).total_seconds()), i["id"]))
                selected.append({"region_id": region["id"], "quarter": quarter, "scene_id": chosen["id"],
                                 "datetime_utc": chosen["properties"]["datetime"], "cloud_cover_pct": chosen["properties"]["eo:cloud_cover"],
                                 "metadata_candidates": len(items), "eligible_metadata_candidates": len(eligible)})
        save_json(metadata_path, {"created_utc": datetime.now(timezone.utc).isoformat(), "temperature_labels_read": False,
                                  "protocol_sha256": digest(path), "scenes": selected})
    print(json.dumps({"protocol": str(path), "fresh_scenes": json.loads(metadata_path.read_text())["scenes"]}), flush=True)
    return protocol


def research_observed_time(scene, requested_datetime, mode):
    """Separate research process only: exact actual 2021–2025 overpasses."""
    target, source = pd.Timestamp(requested_datetime), pd.Timestamp(scene["properties"]["datetime"])
    if mode != "observed" or target.tzinfo is None or source.tzinfo is None or pd.isna(target) or pd.isna(source):
        raise ValueError("Research acquisition requires observed mode and explicit actual timestamps.")
    target, source = target.tz_convert("UTC"), source.tz_convert("UTC")
    if target != source or not 2021 <= target.year <= 2025:
        raise ValueError("Research labels require exact actual 2021–2025 source timestamp.")
    return target, source, 0.


def _correction_acquire(root, output, fresh=False):
    from unittest.mock import patch
    from . import raster
    root, output = Path(root), Path(output)
    protocol = json.loads((output / "protocol.json").read_text())
    if protocol["scientific_signature"] != scientific_signature():
        raise ValueError("Preprocessing changed since the new protocol was frozen.")
    if digest(root / "runs/pilot_v1_model/model.joblib") != protocol["baseline_model_sha256"]:
        raise ValueError("Original baseline changed.")
    if fresh:
        selection = json.loads((output / "selection_frozen_before_test.json").read_text())
        if selection["status"] != "eligible_for_fresh_test":
            raise ValueError("No development-qualified candidate; do not read fresh labels.")
        if (selection["protocol_sha256"] != digest(output / "protocol.json")
                or selection["fresh_metadata_sha256"] != digest(output / "fresh_scene_selection.json")
                or selection["candidate_sha256"] != digest(output / "model/model.joblib")):
            raise ValueError("Frozen method, fresh-scene selection or candidate changed before fresh evaluation.")
        scenes = json.loads((output / "fresh_scene_selection.json").read_text())["scenes"]
    else:
        scenes = protocol["historical_scenes"]
    items = {i["id"]: i for directory in (root / "runs/pilot_v1_satellite/stac", output / "fresh_stac")
             for p in directory.glob("*.json") for i in json.loads(p.read_text())["items"]}
    path = output / "acquisition.json"
    records = json.loads(path.read_text()) if path.exists() else []
    done = {(r["scene_id"], r["block_id"]) for r in records}
    # patch is process-local and restored on every exit, including exceptions.
    with threadpool_limits(limits=4), patch.object(raster, "validate_time", research_observed_time):
        for scene in scenes:
            region = next(r for r in protocol["regions"] if r["id"] == scene["region_id"])
            for block in protocol["blocks"][region["id"]]:
                if (scene["scene_id"], block["id"]) in done:
                    continue
                if len(records) >= protocol["limits"]["maximum_windows"]:
                    raise ValueError("Acquisition cap reached.")
                directory = output / "windows" / (scene["scene_id"] + "_" + block["id"])
                record = {**scene, "block_id": block["id"], "fresh_2025": fresh}
                print("correction acquire", len(records)+1, scene["region_id"], scene["datetime_utc"], block["id"], flush=True)
                started = time.monotonic()
                try:
                    if not (directory / "provenance.json").exists():
                        raster.render_raster(region, block_polygon(block, region["epsg"]), items[scene["scene_id"]],
                                             scene["datetime_utc"], directory, root / "cache", root / "runs/pilot_v1_model/model.joblib", mode="observed")
                    frame = _join_renderer_labels(directory, block, scene)
                    frame["refinement_source"] = CORRECTION_VERSION
                    label = output / "labels" / (scene["scene_id"] + "_" + block["id"] + ".parquet")
                    label.parent.mkdir(parents=True, exist_ok=True)
                    frame.to_parquet(label, index=False)
                    record.update(status="complete", labelled_rows=len(frame))
                except Exception as exc:
                    from .satellite import _safe_error
                    record.update(status="unavailable", labelled_rows=0, error=_safe_error(exc))
                record["seconds"] = round(time.monotonic()-started, 3)
                records.append(record)
                save_json(path, records)


def correction_history(root, output):
    _correction_acquire(root, output, fresh=False)


def _date_score(frame, prediction):
    error = np.abs(np.asarray(prediction)-np.asarray(frame.lst_c))
    dates = pd.to_datetime(frame.datetime_utc, utc=True).dt.strftime("%Y-%m-%d")
    values = pd.DataFrame({"date": np.asarray(dates), "error": error}).groupby("date").error.mean()
    summary_frame = frame.assign(_correction_prediction=np.asarray(prediction))
    return {"date_balanced_mae_c": float(values.mean()), "per_date_mae_c": values.to_dict(),
            **detailed_metrics(summary_frame, "_correction_prediction")}


def _correction_comparison(frame, baseline, candidate, features):
    result = {}
    for region_id, group in frame.groupby("region_id"):
        b = group.air_temperature_c + baseline["model"].predict(group[features])
        c = group.air_temperature_c + candidate.predict(group[features])
        result[str(region_id)] = {"baseline": _date_score(group, b), "candidate": _date_score(group, c)}
    return result


def _correction_gates(comparison, improvement=.03):
    if not all(r in comparison for r in (LONDON, "cabauw")):
        return {"both_regions": False}
    london, control = comparison[LONDON], comparison["cabauw"]
    bm, cm = control["baseline"]["date_balanced_mae_c"], control["candidate"]["date_balanced_mae_c"]
    gates = {"london_mean_improvement": london["candidate"]["date_balanced_mae_c"] <= (1-improvement)*london["baseline"]["date_balanced_mae_c"],
             "cabauw_mean_stable": cm <= bm * 1.02 and cm <= bm + .05}
    for region_id, values in comparison.items():
        b, c = values["baseline"], values["candidate"]
        gates[region_id+"_no_bad_date"] = all(c["per_date_mae_c"][date] <= value+.1 for date, value in b["per_date_mae_c"].items())
        gates[region_id+"_tails_stable"] = all(c[tail]["mae_c"] <= b[tail]["mae_c"]+.1
                                              for tail in ("hot_tail", "cold_tail") if b[tail]["n"] >= 50)
    return gates


def correction_fit(root, output):
    from sklearn.linear_model import Ridge
    from .refine import CfbResidualModel as StableModel
    root, output = Path(root), Path(output)
    protocol = json.loads((output / "protocol.json").read_text())
    selection_path = output / "selection_frozen_before_test.json"
    if selection_path.exists():
        raise FileExistsError("Do not refit or overwrite a frozen correction selection.")
    if (digest(root / "runs/pilot_v1_model/model.joblib") != protocol["baseline_model_sha256"]
            or digest(root / "runs/pilot_v1_assembly/model_input.parquet") != protocol["baseline_input_sha256"]
            or scientific_signature() != protocol["scientific_signature"]):
        raise ValueError("Original baseline/input or exact preprocessing changed before correction fitting.")
    original = pd.read_parquet(root / "runs/pilot_v1_assembly/model_input.parquet")
    extra = pd.concat([pd.read_parquet(p) for p in sorted((output / "labels").glob("*.parquet"))], ignore_index=True)
    original.attrs, extra.attrs = {}, {}
    data = pd.concat([original, extra], ignore_index=True).drop_duplicates(["region_id", "datetime_utc", "pixel_id"], keep="last")
    data.datetime_utc = pd.to_datetime(data.datetime_utc, utc=True)
    data = data.loc[data.climate_class.eq("Cfb") & data.region_id.isin([LONDON, "cabauw"])].copy()
    train = data.loc[data.datetime_utc.dt.year.le(2022)].copy()
    dev = data.loc[data.datetime_utc.dt.year.eq(2023)].copy()
    if len(train) > protocol["limits"]["maximum_training_rows"] or min(len(train), len(dev)) < 100:
        raise ValueError("Training/development row bounds failed.")
    if any(g.datetime_utc.nunique() < 2 for _, g in dev.groupby("region_id")):
        raise ValueError("Both 2023 events are needed per city.")
    baseline = joblib.load(root / "runs/pilot_v1_model/model.joblib")
    features = baseline["features"]
    keys = train.region_id.astype(str)+"|"+train.datetime_utc.dt.strftime("%Y-%m-%d")
    weights = 1 / keys.map(keys.value_counts()).to_numpy(float)
    weights /= weights.sum()
    phi = correction_terms(train)
    center = np.average(phi, axis=0, weights=weights)
    scale = np.sqrt(np.average((phi-center)**2, axis=0, weights=weights))
    scale[scale < 1e-8] = 1.
    residual = train.lst_c - (train.air_temperature_c + baseline["model"].predict(train[features]))
    records, candidates = [], {}
    with threadpool_limits(limits=4):
        for alpha in protocol["correction"]["ridge_alphas"]:
            ridge = Ridge(alpha=alpha).fit((phi-center)/scale, residual, sample_weight=weights)
            model = StableModel(baseline["model"], ridge, center, scale, cap_c=2.)
            comp = _correction_comparison(dev, baseline, model, features)
            gates = _correction_gates(comp)
            score = np.mean([v["candidate"]["date_balanced_mae_c"] for v in comp.values()])
            records.append({"alpha": alpha, "development": comp, "gates": gates, "eligible": all(gates.values()), "score": float(score)})
            candidates[alpha] = model
        eligible = [r for r in records if r["eligible"]]
        chosen = min(eligible, key=lambda r: r["score"]) if eligible else None
        result = {"version": CORRECTION_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
                  "status": "eligible_for_fresh_test" if chosen else "rejected_on_development", "chosen_alpha": chosen["alpha"] if chosen else None,
                  "candidates": records, "fresh_2025_labels_read": False, "seen_2024_used_for_selection": False,
                  "protocol_sha256": digest(output / "protocol.json"), "fresh_metadata_sha256": digest(output / "fresh_scene_selection.json"),
                  "counts": {"train": len(train), "development": len(dev)}, "production_activated": False}
        if chosen:
            bundle = {**baseline, "model": candidates[chosen["alpha"]], "refinement_version": CORRECTION_VERSION,
                      "interval_recalibrated": False, "protocol_sha256": result["protocol_sha256"]}
            (output / "model").mkdir(exist_ok=True)
            joblib.dump(bundle, output / "model/model.joblib", compress=3)
            result["candidate_sha256"] = digest(output / "model/model.joblib")
        save_json(selection_path, result)
    print(json.dumps(result), flush=True)
    return result


def correction_test(root, output):
    root, output = Path(root), Path(output)
    if (output / "fresh_test_metrics.json").exists():
        raise FileExistsError("Fresh test already evaluated; do not rerun or tune.")
    _correction_acquire(root, output, fresh=True)
    frame = pd.concat([pd.read_parquet(p) for p in sorted((output / "labels").glob("*.parquet"))], ignore_index=True)
    frame.datetime_utc = pd.to_datetime(frame.datetime_utc, utc=True)
    frame = frame.loc[frame.datetime_utc.dt.year.eq(2025) & frame.climate_class.eq("Cfb")].copy()
    baseline = joblib.load(root / "runs/pilot_v1_model/model.joblib")
    candidate = joblib.load(output / "model/model.joblib")
    with threadpool_limits(limits=4):
        comp = _correction_comparison(frame, baseline, candidate["model"], baseline["features"])
    gates = _correction_gates(comp, improvement=.05)
    gates["three_dates_each"] = all(r in comp and len(comp[r]["baseline"]["per_date_mae_c"]) >= 3 for r in (LONDON, "cabauw"))
    gates["london_hot_and_cold_evidence"] = LONDON in comp and all(comp[LONDON]["baseline"][t]["n"] >= 50 for t in ("hot_tail", "cold_tail"))
    frame["baseline_predicted_lst_c"] = frame.air_temperature_c + baseline["model"].predict(frame[baseline["features"]])
    frame["candidate_predicted_lst_c"] = frame.air_temperature_c + candidate["model"].predict(frame[baseline["features"]])
    frame.attrs = {}
    frame.to_parquet(output / "fresh_2025_predictions.parquet", index=False)
    result = {"version": CORRECTION_VERSION, "status": "passes_fresh_accuracy_checks" if all(gates.values()) else "candidate_not_recommended",
              "evaluations": comp, "acceptance": gates, "n": len(frame), "production_activated": False,
              "protocol_sha256": digest(output / "protocol.json"), "selection_sha256": digest(output / "selection_frozen_before_test.json"),
              "baseline_model_sha256": digest(root / "runs/pilot_v1_model/model.joblib"),
              "limitations": ["Two Cfb regions only; other climate predictions unchanged by construction.",
                              "Four metadata-selected quarterly scenes per city, daytime clear-sky support only.",
                              "Fixed-location temporal test; not independent unseen-city validation.",
                              "Original interval width was not recalibrated for the correction."]}
    save_json(output / "fresh_test_metrics.json", result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    main()
