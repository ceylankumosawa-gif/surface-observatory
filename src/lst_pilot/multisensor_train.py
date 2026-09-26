"""Fixed F/G research calibration; no acquisition, search or production changes.

F refits the E estimator with new admitted fine labels. G adds a penalized,
shrunk correction learned from global-calendar-month OOF predictions only.
The separate legacy-2024 command checks every frozen dependency before labels.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from . import option_b_train as old
from . import more_days_train as more
from .option_b_cohort import spatial_flags
from . import multisensor_context as context

VERSION = "multisensor-fixed-fgh-v1"
BASE = old.BASE_FEATURES
PRODUCTS = ("ecostress_v2", "aster_ast08_v004")
DEPENDENCIES = ("option_b_train.py", "more_days_train.py", "option_b_cohort.py", "model.py", "multisensor_context.py", "coarse_join.py")
NUMERIC = ("predicted_offset_c", "air_temperature_c", "solar_elevation_deg",
           "cloud_cover_fraction", "wind_speed_m_s", "relative_humidity_pct", "log_snow")
CLIMATES = tuple("ABCDE")
PHASES = ("day", "night")
SPEC = {"folds": 3, "fold_formula": "((UTC_year-2021)*12+UTC_month-1)%3",
        "ridge_alpha": 20.0, "shrinkage": 0.5, "max_abs_correction_c": 3.0,
        "minimum_group_utc_dates": 6, "minimum_group_year_months": 3,
        "minimum_group_folds": 2, "numeric_features": list(NUMERIC),
        "categorical_features": ["broad_climate_group", "phase"],
        "penalized_intercept": True, "auto_promotion": False,
        "regularization_weight_scale": "effective independent global UTC-date count",
        "optional_native_context": "separate fixed H, never a pseudo-label"}


def load_new(path, *, evaluation=False):
    """Reject wrong years using only timestamps, before decoding label columns."""
    dates = pd.to_datetime(pd.read_parquet(path, columns=["datetime_utc"]).datetime_utc,
                           utc=True, errors="raise")
    years = [2023] if evaluation else [2021, 2022]
    if dates.isna().any() or not dates.dt.year.isin(years).all():
        raise ValueError(f"New {'evaluation' if evaluation else 'fitting'} years must be {years}; thermal columns were not loaded.")
    return pd.read_parquet(path)


def load_freshness_registry(path, expected_sha):
    if old.sha(path) != expected_sha:
        raise ValueError("Freshness registry changed before evaluation labels were read.")
    registry = json.loads(Path(path).read_text())
    if registry.get("version") != "multisensor-prior-date-registry-v1" or not isinstance(registry.get("dates"), list):
        raise ValueError("Invalid previously inspected/attempted date registry.")
    for record in registry["dates"]:
        day = record.get("utc_date")
        if (not isinstance(record.get("region_id"), str) or not record["region_id"] or
                not isinstance(day, str) or len(day) != 10 or pd.Timestamp(day).strftime("%Y-%m-%d") != day):
            raise ValueError("Invalid prior pilot-date in the freshness registry.")
    return registry


def validate_freshness(frame, registry, expected_sha, reference):
    required = {"region_id", "datetime_utc", "fresh_2023", "freshness_audit_sha256"}
    if not required.issubset(frame):
        raise ValueError("New evaluation requires explicit freshness audit fields.")
    dates = pd.to_datetime(frame.datetime_utc, utc=True, errors="raise")
    if dates.isna().any() or not dates.dt.year.eq(2023).all():
        raise ValueError("Fresh evaluation timestamps must be in 2023; thermal columns were not loaded.")
    flags = old.strict_bool(frame.fresh_2023, "fresh_2023")
    if not expected_sha or not frame.freshness_audit_sha256.eq(expected_sha).fillna(False).all():
        raise ValueError("Freshness audit hash does not match the frozen acquisition registry.")
    if registry is None:
        raise ValueError("Verified prior attempted-date registry contents are required.")
    metadata = pd.DataFrame({"region": frame.region_id.to_numpy(),
                             "day": dates.dt.strftime("%Y-%m-%d").to_numpy(), "fresh": flags.to_numpy()})
    if metadata.groupby(["region", "day"]).fresh.nunique().gt(1).any():
        raise ValueError("A pilot-date cannot be both fresh and repeated across sensors.")
    seen = {(r["region_id"], r["utc_date"]) for r in registry["dates"]}
    # Retain the admitted-row check as a second guard if a registry is incomplete.
    observed = set(zip(reference.region_id, pd.to_datetime(reference.datetime_utc, utc=True).dt.strftime("%Y-%m-%d")))
    for region, day, declared in metadata.itertuples(index=False, name=None):
        if declared and (region, day) in observed:
            raise ValueError("A previously observed pilot-date cannot be called fresh.")
        expected = (region, day) not in seen and (region, day) not in observed
        if declared != expected:
            raise ValueError("Freshness flag disagrees with prior inspected/attempted-date registry.")
    return flags


def load_new_evaluation(path, registry_path, expected_sha, reference):
    registry = load_freshness_registry(registry_path, expected_sha)
    metadata = pd.read_parquet(path, columns=["region_id", "datetime_utc", "fresh_2023", "freshness_audit_sha256"])
    validate_freshness(metadata, registry, expected_sha, reference)
    return pd.read_parquet(path), registry


def check_grid(data, areas):
    if not {"grid_row", "grid_col"}.issubset(data):
        raise ValueError("Grid row/column metadata is required.")
    if not data.region_id.isin(areas).all():
        raise ValueError("Unknown pilot.")
    for region, indexes in data.groupby("region_id").groups.items():
        area = areas[region]
        rr, cc = (data.loc[indexes, f].to_numpy(float) for f in ("grid_row", "grid_col"))
        h, w = area["grid_shape"]
        if not (np.isfinite(rr).all() and np.isfinite(cc).all() and
                np.equal(rr, np.floor(rr)).all() and np.equal(cc, np.floor(cc)).all() and
                ((rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)).all()):
            raise ValueError("Invalid pilot grid coordinates.")
        left, _, _, top = area["extent_m"]
        x, y = Transformer.from_crs(4326, area["epsg"], always_xy=True).transform(
            data.loc[indexes, "longitude"].to_numpy(), data.loc[indexes, "latitude"].to_numpy())
        if not (np.hypot(x-(left+(cc+.5)*100), y-(top-(rr+.5)*100)) <= 1).all():
            raise ValueError("Coordinates disagree with the 100 m grid cell.")
    checked = spatial_flags(data, areas)
    for field in ("block_id", "spatial_holdout", "in_holdout_buffer"):
        if not checked[field].eq(data[field]).all():
            raise ValueError(f"Recomputed whole-cell spatial classification differs: {field}")


def validate_new(frame, reference, areas, *, evaluation=False, freshness_sha=None, registry=None):
    required = {"research_admissibility_reason", "source_screen_pass", "label_source_sha256"}
    if not evaluation:
        required.add("native_fit_support_pass")
    if not required.issubset(frame):
        raise ValueError(f"Missing source admission fields: {sorted(required-set(frame))}")
    if frame.empty and evaluation:
        empty = reference.iloc[:0].copy()
        empty["fresh_2023"] = pd.Series(dtype=bool)
        empty["freshness_audit_sha256"] = pd.Series(dtype=str)
        return empty
    if not frame.research_admissibility_reason.eq("").fillna(False).all():
        raise ValueError("Every new row must have passed research admission.")
    if not old.strict_bool(frame.source_screen_pass, "source_screen_pass").all():
        raise ValueError("Every new row needs a passed source screen.")
    if not evaluation and not old.strict_bool(frame.native_fit_support_pass, "native_fit_support_pass").all():
        raise ValueError("Every new fitting label needs native contributor footprint support.")
    if not frame.label_source_sha256.astype("string").str.fullmatch(r"[0-9a-f]{64}").fillna(False).all():
        raise ValueError("A valid source SHA-256 is required.")
    data = old.prepare_input(frame)
    allowed_years = [2023] if evaluation else [2021, 2022]
    if not data.datetime_utc.dt.year.isin(allowed_years).all():
        raise ValueError("New rows are outside their declared fitting/evaluation years.")
    if not data.cohort_origin.eq("expanded").all() or not data.label_product.isin(PRODUCTS).all():
        raise ValueError("Only admitted new ECOSTRESS/ASTER fine labels are allowed.")
    if not old.complete_rows(data, BASE).all() or not data.phase.isin(PHASES).all():
        raise ValueError("New rows require complete base40 predictors and day/night support.")
    check_grid(data, areas)
    if data.in_holdout_buffer.any() or (not evaluation and not data.split.eq("fit").all()):
        raise ValueError("Cabauw, reserved cells or buffers cannot enter fitting; buffers are never evaluated.")
    if set(data.sample_id) & set(reference.sample_id):
        raise ValueError("New sample identities overlap a frozen earlier cohort.")
    keys = ["region_id", "acquisition_id", "grid_row", "grid_col"]
    if data.duplicated(keys).any():
        raise ValueError("Duplicate physical acquisition/grid cell.")
    if set(keys).issubset(reference):
        known = pd.MultiIndex.from_frame(reference[keys])
        if pd.MultiIndex.from_frame(data[keys]).isin(known).any():
            raise ValueError("New physical acquisition/grid cell overlaps earlier data.")
    if evaluation:
        data["fresh_2023"] = validate_freshness(data, registry, freshness_sha, reference)
    return data.sort_values("sample_id").reset_index(drop=True)


def join_fitting(e_fit, additions):
    if not e_fit.split.eq("fit").all() or not additions.split.eq("fit").all():
        raise ValueError("Only fitting rows may enter F/G.")
    result = pd.concat([e_fit, additions], ignore_index=True).sort_values("sample_id").reset_index(drop=True)
    if result.sample_id.duplicated().any() or len(result) > old.MAX_ROWS:
        raise ValueError("Duplicate fitting identity or fitting cap exceeded; no original rows may be dropped.")
    preserved = result.set_index("sample_id").loc[e_fit.sample_id, e_fit.columns.drop("sample_id")]
    pd.testing.assert_frame_equal(preserved, e_fit.set_index("sample_id"), check_dtype=False)
    return result


def month_folds(frame):
    stamps = pd.to_datetime(frame.datetime_utc, utc=True)
    if stamps.isna().any() or not stamps.dt.year.isin([2021, 2022]).all():
        raise ValueError("OOF calibration may use only 2021--2022.")
    if (not frame.split.eq("fit").all() or frame.region_id.eq("cabauw").any()
            or frame.spatial_holdout.any() or frame.in_holdout_buffer.any()):
        raise ValueError("OOF fitting includes a reserved geographic/temporal row.")
    return (((stamps.dt.year-2021)*12+stamps.dt.month-1) % 3).to_numpy(int)


def global_date_scale(frame, weights):
    """Effective date count, invariant to replicated pixels and date co-occurrence."""
    weights = np.asarray(weights, float)
    if weights.shape != (len(frame),) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Invalid residual fitting weights.")
    weights = weights/weights.sum()
    dates = pd.to_datetime(frame.datetime_utc, utc=True).dt.floor("D")
    masses = pd.DataFrame({"day": dates.to_numpy(), "weight": weights}).groupby("day").weight.sum()
    return float(1/np.square(masses).sum())


def correction_inputs(frame, offset):
    values = np.asarray(offset, float)
    if values.shape != (len(frame),):
        raise ValueError("Correction predictions are misaligned.")
    climate = frame.climate_class.astype(str).str[0].str.upper().to_numpy()
    if not np.isin(climate, CLIMATES).all():
        raise ValueError("Correction requires a recognized broad climate group A--E.")
    solar = frame.solar_elevation_deg.to_numpy(float)
    phase = np.where(solar >= 10, "day", np.where(solar <= -6, "night", "twilight"))
    snow = frame.era5_snow_water_equivalent_m.to_numpy(float)
    if (snow < 0).any():
        raise ValueError("Negative physical snow water equivalent.")
    x = np.column_stack([values, frame.air_temperature_c, solar, frame.cloud_cover_fraction,
                         frame.wind_speed_m_s, frame.relative_humidity_pct, np.log1p(snow/.001)])
    if not np.isfinite(x).all():
        raise ValueError("Correction features must be finite.")
    return x, climate, phase


def group_support(frame, folds):
    _, climate, phase = correction_inputs(frame, np.zeros(len(frame)))
    stamps = pd.to_datetime(frame.datetime_utc, utc=True)
    records = pd.DataFrame({"climate": climate, "phase": phase, "day": stamps.dt.floor("D"),
                            "month": stamps.dt.strftime("%Y-%m"), "fold": folds})
    output = {}
    for (c, p), rows in records.groupby(["climate", "phase"]):
        dates, months, nfolds = rows.day.nunique(), rows.month.nunique(), rows.fold.nunique()
        output[f"{c}|{p}"] = {"utc_dates": int(dates), "year_months": int(months), "folds": int(nfolds),
            "supported": bool(dates >= SPEC["minimum_group_utc_dates"] and
                              months >= SPEC["minimum_group_year_months"] and
                              nfolds >= SPEC["minimum_group_folds"])}
    return output


@dataclass
class ResidualCorrection:
    mean: np.ndarray
    scale: np.ndarray
    ridge: object
    support: dict
    effective_utc_dates: float

    @staticmethod
    def design(x, climate, phase, mean, scale):
        return np.column_stack([np.ones(len(x)), (x-mean)/scale,
                                *(climate == c for c in CLIMATES), *(phase == p for p in PHASES)]).astype(float)

    @classmethod
    def fit(cls, frame, oof_offset, folds):
        expected = month_folds(frame)
        if not np.array_equal(expected, folds):
            raise ValueError("Residual fitting fold identities changed.")
        x, climate, phase = correction_inputs(frame, oof_offset)
        weights = old.balanced_weights(frame)
        effective = global_date_scale(frame, weights)
        mean = np.average(x, axis=0, weights=weights)
        scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=weights))
        scale[scale < 1e-8] = 1.0
        support = group_support(frame, folds)
        ridge = None
        if any(g["supported"] for g in support.values()):
            ridge = Ridge(alpha=SPEC["ridge_alpha"], fit_intercept=False, solver="svd")
            target = old.target_offset(frame)-np.asarray(oof_offset)
            ridge.fit(cls.design(x, climate, phase, mean, scale), target,
                      sample_weight=weights*effective)
        return cls(mean, scale, ridge, support, effective)

    def predict(self, frame, offset):
        x, climate, phase = correction_inputs(frame, offset)
        supported = np.array([self.support.get(f"{c}|{p}", {}).get("supported", False)
                              for c, p in zip(climate, phase)], bool)
        correction = np.zeros(len(frame))
        if self.ridge is not None and supported.any():
            raw = self.ridge.predict(self.design(x, climate, phase, self.mean, self.scale))
            correction[supported] = np.clip(SPEC["shrinkage"]*raw[supported],
                -SPEC["max_abs_correction_c"], SPEC["max_abs_correction_c"])
        return correction, supported


@dataclass
class CalibratedEstimator:
    base: object
    correction: ResidualCorrection

    def predict(self, frame):
        offset = self.base.predict(frame[list(BASE)])
        adjustment, _ = self.correction.predict(frame, offset)
        return offset+adjustment


H_NUMERIC = ("coarse_lst_minus_current_air_c", "coarse_oldest_age_hours", "coarse_context_missing")
H_SPEC = {"additional_predictors": list(H_NUMERIC), "ridge_alpha": 20.0, "shrinkage": .5,
          "max_abs_total_correction_c": 3.0, "maximum_source_age_hours": 24,
          "context_support_minimum_dates": 6, "context_support_minimum_months": 3,
          "context_support_minimum_folds": 2, "fallback": "exactly G",
          "fit_rows": "identical to F/G, including missing-context rows",
          "legacy_2024_context": "none; H point predictions exactly equal G"}


def context_inputs(frame):
    eligible=old.strict_bool(frame.coarse_context_eligible,"coarse_context_eligible").to_numpy()
    values=np.column_stack([frame.coarse_lst_c.to_numpy(float)-frame.air_temperature_c.to_numpy(float),
                            frame.coarse_age_hours.to_numpy(float)])
    if (not np.isfinite(values[eligible]).all() or (values[eligible,1]<0).any() or (values[eligible,1]>24).any()):
        raise ValueError("Eligible context must have finite, causal values no older than 24 hours.")
    if np.isfinite(values[~eligible]).any():
        raise ValueError("Missing context cannot contain imputed coarse values.")
    return values,eligible


@dataclass
class ContextResidualCorrection:
    core_mean: np.ndarray
    core_scale: np.ndarray
    context_mean: np.ndarray
    context_scale: np.ndarray
    ridge: object
    support: dict
    effective_utc_dates: float

    def design(self,frame,offset):
        x,climate,phase=correction_inputs(frame,offset)
        values,eligible=context_inputs(frame)
        standardized=np.zeros_like(values)
        standardized[eligible]=(values[eligible]-self.context_mean)/self.context_scale
        return np.column_stack([ResidualCorrection.design(x,climate,phase,self.core_mean,self.core_scale),
                                standardized,(~eligible).astype(float)]),climate,phase,eligible

    @classmethod
    def fit(cls,frame,oof_offset,folds,core):
        if not np.array_equal(month_folds(frame),folds): raise ValueError("H OOF folds changed.")
        values,eligible=context_inputs(frame);weights=old.balanced_weights(frame)
        mean=np.zeros(2);scale=np.ones(2)
        if eligible.any():
            mean=np.average(values[eligible],axis=0,weights=weights[eligible])
            scale=np.sqrt(np.average((values[eligible]-mean)**2,axis=0,weights=weights[eligible]))
            scale[scale<1e-8]=1
        support=group_support(frame.loc[eligible].reset_index(drop=True),np.asarray(folds)[eligible])
        instance=cls(core.mean.copy(),core.scale.copy(),mean,scale,None,support,global_date_scale(frame,weights))
        if any(v["supported"] for v in support.values()):
            design,*_=instance.design(frame,oof_offset)
            instance.ridge=Ridge(alpha=20.,fit_intercept=False,solver="svd").fit(
                design,old.target_offset(frame)-np.asarray(oof_offset),sample_weight=weights*instance.effective_utc_dates)
        return instance

    def predict(self,frame,offset):
        design,climate,phase,eligible=self.design(frame,offset)
        supported=eligible & np.array([self.support.get(f"{c}|{p}",{}).get("supported",False)
                                      for c,p in zip(climate,phase)],bool)
        adjustment=np.zeros(len(frame))
        if self.ridge is not None and supported.any():
            adjustment[supported]=np.clip(.5*self.ridge.predict(design[supported]),-3.,3.)
        return adjustment,supported


@dataclass
class ContextCalibratedEstimator:
    base: object
    correction: ResidualCorrection
    context_correction: ContextResidualCorrection

    def predict(self,frame):
        offset=self.base.predict(frame[list(BASE)])
        adjustment,_=self.correction.predict(frame,offset)
        contextual,supported=self.context_correction.predict(frame,offset)
        adjustment[supported]=contextual[supported]
        return offset+adjustment


def fit_base(frame):
    estimator, _ = old.build_estimators(BASE, old.CONFIG)
    estimator.fit(frame[list(BASE)], old.target_offset(frame),
                  regressor__sample_weight=old.balanced_weights(frame)*len(frame))
    return estimator


def out_of_fold(frame, output=None):
    folds = month_folds(frame)
    predicted = np.full(len(frame), np.nan)
    assignments = np.zeros(len(frame), int)
    records = []
    months = pd.to_datetime(frame.datetime_utc, utc=True).dt.strftime("%Y-%m")
    for fold in range(3):
        held = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        if not len(held) or len(train) < 2:
            raise ValueError("Every fixed OOF fold requires fitting and held-month observations.")
        fit, test = frame.iloc[train], frame.iloc[held]
        if set(months.iloc[train]) & set(months.iloc[held]):
            raise AssertionError("Global UTC calendar-month leakage.")
        estimator = fit_base(fit)
        predicted[held] = estimator.predict(test[list(BASE)])
        assignments[held] += 1
        record = {"fold": fold, "fit_rows": len(train), "held_rows": len(held),
                  "fit_sample_id_sha256": old.row_hash(fit), "held_sample_id_sha256": old.row_hash(test),
                  "fit_months": sorted(months.iloc[train].unique()),
                  "held_months": sorted(months.iloc[held].unique())}
        if output is not None:
            path = Path(output)/f"oof_fold_{fold}.joblib"
            joblib.dump(estimator, path, compress=3)
            record["model_sha256"] = old.sha(path)
        records.append(record)
    if not np.equal(assignments, 1).all() or not np.isfinite(predicted).all():
        raise ValueError("OOF predictions do not cover each fitting identity exactly once.")
    return predicted, folds, records


def reconstruct_e(original_input, e_additions, original_run, e_run, areas, baseline):
    more.verify_experiment(e_run, original_run, baseline)
    manifest = json.loads((Path(e_run)/"manifest.json").read_text())
    if old.sha(original_input) != manifest["original_input_sha256"] or old.sha(e_additions) != manifest["new_input_sha256"]:
        raise ValueError("Frozen E fitting source changed before labels were read.")
    original, a_fit, evaluation, a, v1, _ = more.load_reference(original_run, original_input, baseline)
    additions = more.validate_additions(more.load_additions(e_additions), original, areas)
    fit, _ = more.append_fitting(a_fit, additions)
    if (len(fit) != manifest["fit_rows"] or old.row_hash(fit) != manifest["fit_row_sha256"]
            or more.weight_hash(fit) != manifest["fit_weight_sha256"]):
        raise ValueError("Reconstructed E fitting identities/weights changed.")
    records = pd.read_parquet(Path(e_run)/"fitting_rows.parquet").set_index("sample_id").loc[fit.sample_id]
    if not np.array_equal(records.weight.to_numpy(), old.balanced_weights(fit)):
        raise ValueError("Recorded E fitting weights changed.")
    e = joblib.load(Path(e_run)/"model.joblib")
    if tuple(e["features"]) != BASE or e["config"] != asdict(old.CONFIG):
        raise ValueError("Frozen E estimator specification changed.")
    return original, fit.reset_index(drop=True), evaluation, {"E": e, "A": a, "v1": v1}


def predictions(frame, bundles):
    air = frame.air_temperature_c.to_numpy(float)
    return {"air_only": air, **{name: air+bundle["model"].predict(frame[bundle["features"]])
                               if len(frame) else np.array([]) for name, bundle in bundles.items()}}


def reuse_old_predictions(frame, values, path):
    saved = pd.read_parquet(path, columns=["sample_id", "E_lst_c", "A_lst_c", "v1_lst_c"])
    if saved.sample_id.duplicated().any() or set(saved.sample_id) != set(frame.sample_id):
        raise ValueError("Frozen reference evaluation identities changed.")
    saved = saved.set_index("sample_id").loc[frame.sample_id]
    for name in ("E", "A", "v1"):
        exact = saved[f"{name}_lst_c"].to_numpy(float)
        if not np.allclose(values[name], exact, rtol=0, atol=1e-10):
            raise ValueError(f"Frozen {name} evaluation predictions no longer reproduce.")
        values[name] = exact
    return values


def score(frame, values, intervals=None, correction=None, h_correction=None):
    result = {"metrics": {name: old.group_metrics(frame, p) for name, p in values.items()},
              "support": old.support_counts(frame), "row_sha256": old.row_hash(frame),
              "weight_sha256": more.weight_hash(frame)}
    if intervals is not None:
        result["empirical_interval_coverage"] = {name: old.interval_coverage(frame, values[name], intervals[name])
                                                for name in intervals if name in values}
    if correction is not None and len(frame):
        adjustment, supported = correction.predict(frame, values["F"]-frame.air_temperature_c.to_numpy())
        if not np.allclose(values["G"]-values["F"], adjustment, rtol=0, atol=1e-10):
            raise ValueError("Reported correction does not reproduce the F/G prediction difference.")
        weights = old.balanced_weights(frame)
        result["correction_usage"] = {"supported_rows": int(supported.sum()),
            "unsupported_rows_unchanged_from_F": int((~supported).sum()),
            "weighted_supported_fraction": float(weights @ supported),
            "weighted_mean_abs_adjustment_c": float(weights @ np.abs(adjustment)),
            "max_abs_adjustment_c": float(np.max(np.abs(adjustment))),
            "weighted_fraction_at_adjustment_limit": float(weights @ (np.abs(adjustment) >= SPEC["max_abs_correction_c"]-1e-12))}
    if h_correction is not None and len(frame):
        adjustment,supported=h_correction.predict(frame,values["F"]-frame.air_temperature_c.to_numpy())
        eligible=old.strict_bool(frame.coarse_context_eligible,"coarse_context_eligible").to_numpy()
        if (not np.allclose(values["H"][supported]-values["F"][supported],adjustment[supported],rtol=0,atol=1e-10)
                or not np.array_equal(values["H"][~supported],values["G"][~supported])):
            raise ValueError("H did not reproduce the bounded correction or exact G fallback.")
        weights=old.balanced_weights(frame)
        result["context_correction_usage"]={"eligible_rows":int(eligible.sum()),"supported_rows":int(supported.sum()),
            "fallback_rows_equal_G":int((~supported).sum()),"weighted_eligible_fraction":float(weights@eligible),
            "weighted_supported_fraction":float(weights@supported),
            "weighted_mean_abs_total_adjustment_c":float(weights@np.abs(values["H"]-values["F"])),
            "weighted_mean_abs_change_from_G_c":float(weights@np.abs(values["H"]-values["G"])),
            "max_abs_total_adjustment_c":float(np.abs(values["H"]-values["F"]).max()),
            "supported_subset_metrics":{n:old.group_metrics(frame.loc[supported],v[supported]) for n,v in values.items()},
            "eligible_support":old.support_counts(frame.loc[eligible]),
            "by_product":frame.loc[eligible,"coarse_product"].value_counts().to_dict() if eligible.any() else {},
            "unique_native_cells":int(frame.loc[eligible,"coarse_native_id"].nunique()) if eligible.any() else 0,
            "unique_native_acquisitions":int(frame.loc[eligible,"coarse_acquisition_id"].nunique()) if eligible.any() else 0,
            "by_product_phase":[{"product":product,"target_phase":phase,"rows":len(group),
                "pilot_dates":len(group.assign(utc_day=pd.to_datetime(group.datetime_utc,utc=True).dt.floor("D"))[["region_id","utc_day"]].drop_duplicates()),
                "native_cells":int(group.coarse_native_id.nunique()),"native_acquisitions":int(group.coarse_acquisition_id.nunique())}
                for (product,phase),group in frame.loc[eligible].groupby(["coarse_product","phase"])] if eligible.any() else [],
            "oldest_age_hours":{k:float(v) for k,v in frame.loc[eligible,"coarse_age_hours"].describe().items()} if eligible.any() else {}}
    return result


def save_predictions(frame, values, path, correction=None, h_correction=None):
    fields = [f for f in ("sample_id", "region_id", "datetime_utc", "split", "phase", "label_product",
                         "air_group", "snow_group", "acquisition_id", "lst_c", "fresh_2023") if f in frame]
    result = frame[fields].copy()
    for name, p in values.items():
        result[f"{name}_lst_c"] = p
    if correction is not None:
        adjustment, supported = correction.predict(frame, values["F"]-frame.air_temperature_c.to_numpy())
        result["G_adjustment_c"] = adjustment
        result["G_correction_supported"] = supported
    if h_correction is not None:
        adjustment,supported=h_correction.predict(frame,values["F"]-frame.air_temperature_c.to_numpy())
        result["H_adjustment_c"]=values["H"]-values["F"]
        result["H_correction_supported"]=supported
        for field in ("coarse_context_eligible","coarse_age_hours","coarse_product","coarse_native_id","coarse_acquisition_id","coarse_source_sha256"):
            if field in frame: result[field]=frame[field].to_numpy()
    result.to_parquet(path, index=False)


def source_hashes():
    return {"source_sha256": old.sha(__file__),
            "dependencies_sha256": {name: old.sha(Path(__file__).with_name(name)) for name in DEPENDENCIES}}


def verify_pending_evaluation_inputs(manifest):
    for name in ("new_evaluation", "freshness_audit", "areas_path"):
        if old.sha(manifest["paths"][name]) != manifest["input_hashes"][name]:
            raise ValueError(f"Frozen pending evaluation input changed before labels were read: {name}")


def run_experiment(*, original_input, e_additions, original_run, e_run, e_2024,
                   new_fit, new_evaluation, freshness_audit, areas_path, baseline,
                   protocol, output, context_spec=None, h_protocol=None):
    started = time.monotonic()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new immutable multisensor experiment directory.")
    areas = {a["id"]: a for a in json.loads(Path(areas_path).read_text())["areas"]}
    original, e_fit, evaluation, bundles = reconstruct_e(original_input, e_additions, original_run, e_run, areas, baseline)
    known = pd.concat([original, e_fit], ignore_index=True).drop_duplicates("sample_id")
    additions = validate_new(load_new(new_fit), known, areas)
    fit = join_fitting(e_fit, additions)
    h_enabled=context_spec is not None
    if h_enabled != (h_protocol is not None): raise ValueError("H requires both frozen context specifications and its predeclared protocol.")
    contexts=json.loads(Path(context_spec).read_text()) if h_enabled else {}
    context_hashes=context.freeze_inputs(contexts) if h_enabled else {}
    context_audits={}
    if h_enabled:
        fit,context_audits["fit"]=context.attach(fit,contexts.get("fit"),areas,for_fitting=True)
    # Only reference hashes/metadata are read here; new 2023/2024 thermal arrays wait.
    previous = json.loads((Path(e_2024)/"results.json").read_text())
    if (previous["status"] != "more_days_2024_evaluated_no_refit_no_promotion" or
            previous["more_days_freeze_sha256"] != old.sha(Path(e_run)/"post_evaluation_freeze.json")):
        raise ValueError("Existing 2024 reference is not bound to frozen E.")
    source = source_hashes()
    manifest = {"version": VERSION, "research_only": True, "auto_promotion": False,
        "selection": "Fixed F and G, both reported; no ranking or refitting after evaluation.",
        "correction_specification": SPEC, "base_features": list(BASE), "estimator": asdict(old.CONFIG),
        **source, "thread_limit": 4, "protocol_sha256": old.sha(protocol),
        "paths": {k: str(Path(v).resolve()) for k, v in dict(original_input=original_input,
            e_additions=e_additions, original_run=original_run, e_run=e_run, e_2024=e_2024,
            new_fit=new_fit, new_evaluation=new_evaluation, freshness_audit=freshness_audit,
            areas_path=areas_path, baseline=baseline).items()},
        "reference_hashes": {"E": old.sha(Path(e_run)/"model.joblib"),
            "A": old.sha(Path(original_run)/"A/model.joblib"), "v1": old.sha(baseline),
            "e_freeze": old.sha(Path(e_run)/"post_evaluation_freeze.json"),
            "old_predictions": old.sha(Path(e_run)/"predictions.parquet"),
            "e_2024_results": old.sha(Path(e_2024)/"results.json"),
            "e_2024_predictions": old.sha(Path(e_2024)/"predictions.parquet"),
            "legacy_2024_input": previous["input_sha256"]},
        "input_hashes": {k: old.sha(v) for k, v in dict(original_input=original_input,
            e_additions=e_additions, new_fit=new_fit, new_evaluation=new_evaluation,
            freshness_audit=freshness_audit, areas_path=areas_path).items()},
        "e_fit_rows": len(e_fit), "e_fit_row_sha256": old.row_hash(e_fit),
        "new_fit_rows": len(additions), "fit_rows": len(fit), "fit_row_sha256": old.row_hash(fit),
        "fit_weight_sha256": more.weight_hash(fit), "support": old.support_counts(fit),
        "old_evaluation": {s: {"rows": len(g), "row_sha256": old.row_hash(g), "weight_sha256": more.weight_hash(g)}
                           for s in more.SPLITS for g in [evaluation.loc[evaluation.split.eq(s)]]}}
    manifest["h_enabled"]=h_enabled
    if h_enabled:
        manifest["selection"]="Fixed F, G and H, all reported; no ranking or refitting after evaluation."
        manifest["context_specification"]=H_SPEC
        manifest["context_join_specification"]=contexts
        manifest["context_inputs_sha256"]=context_hashes
        manifest["context_fit_audit"]=context_audits["fit"]
        manifest["h_protocol_sha256"]=old.sha(h_protocol)
        manifest["context_spec_sha256"]=old.sha(context_spec)
    # The fixed three-fold structure must be viable before creating run output.
    if set(month_folds(fit)) != {0, 1, 2}:
        raise ValueError("Fitting data do not populate all three predeclared OOF folds.")
    output.mkdir(parents=True)
    old.save_json(output/"manifest.json", manifest)
    fit[["sample_id", "region_id", "datetime_utc", "phase", "label_product", "acquisition_id", "block_id"]].assign(
        original_E=fit.sample_id.isin(e_fit.sample_id), weight=old.balanced_weights(fit)
    ).to_parquet(output/"fitting_rows.parquet", index=False)
    with threadpool_limits(limits=4):
        oof, folds, fold_records = out_of_fold(fit, output)
        correction = ResidualCorrection.fit(fit, oof, folds)
        base = fit_base(fit)
        h_correction=ContextResidualCorrection.fit(fit,oof,folds,correction) if h_enabled else None
        for name, model in (("F", base), ("G", CalibratedEstimator(base, correction))):
            bundle = {"candidate": name, "model": model, "features": list(BASE), "config": asdict(old.CONFIG),
                      "research_only": True, "auto_promotion": False, "target": "lst_c - air_temperature_c"}
            if name == "G": bundle["correction_specification"] = SPEC
            joblib.dump(bundle, output/f"{name}.joblib", compress=3)
            bundles[name] = bundle
        if h_enabled:
            bundle={"candidate":"H","model":ContextCalibratedEstimator(base,correction,h_correction),
                    "features":list(BASE)+context.MODEL_CONTEXT,"config":asdict(old.CONFIG),"research_only":True,
                    "auto_promotion":False,"target":"lst_c - air_temperature_c","context_specification":H_SPEC}
            joblib.dump(bundle,output/"H.joblib",compress=3);bundles["H"]=bundle
            old.save_json(output/"h_correction.json",{"specification":H_SPEC,"support":h_correction.support,
                "effective_global_utc_dates":h_correction.effective_utc_dates,"context_fit_audit":context_audits["fit"],
                "context_mean":h_correction.context_mean.tolist(),"context_scale":h_correction.context_scale.tolist(),
                "coefficients":h_correction.ridge.coef_.tolist() if h_correction.ridge is not None else None,
                "coefficient_names":["penalized_constant",*NUMERIC,*[f"climate_{c}" for c in CLIMATES],*[f"phase_{p}" for p in PHASES],*H_NUMERIC],
                "no_context_in_2024":True,"no_context_values_are_fine_labels":True})
        fit[["sample_id", "datetime_utc", "region_id", "label_product"]].assign(
            fold=folds, oof_offset_c=oof, residual_target_c=old.target_offset(fit)-oof
        ).to_parquet(output/"oof_predictions.parquet", index=False)
        old.save_json(output/"correction.json", {"specification": SPEC, "support": correction.support,
            "effective_global_utc_dates": correction.effective_utc_dates,
            "correction_enabled": correction.ridge is not None, "folds": fold_records,
            "numeric_mean": correction.mean.tolist(), "numeric_scale": correction.scale.tolist(),
            "coefficients": correction.ridge.coef_.tolist() if correction.ridge is not None else None,
            "coefficient_names": ["penalized_constant", *NUMERIC, *[f"climate_{c}" for c in CLIMATES],
                                  *[f"phase_{p}" for p in PHASES]],
            "in_sample_correction_error_is_not_independent_performance": True,
            "maximum_adjustment_is_not_an_error_bound": True})
        frozen_names = ("manifest.json", "F.joblib", "G.joblib", "fitting_rows.parquet", "oof_predictions.parquet",
                        "correction.json", "oof_fold_0.joblib", "oof_fold_1.joblib", "oof_fold_2.joblib")
        if h_enabled: frozen_names += ("H.joblib","h_correction.json")
        old.save_json(output/"fit_freeze.json", {"version": VERSION,
            "artifacts": {n: old.sha(output/n) for n in frozen_names},
            "models_frozen_before_new_evaluation": True, "refit_or_selection_allowed": False})
        if h_enabled:
            context.verify_inputs(context_hashes)
            evaluation,context_audits["old_evaluation"]=context.attach(evaluation,contexts.get("old_evaluation"),areas,for_fitting=False)
        values = reuse_old_predictions(evaluation, predictions(evaluation, bundles), Path(e_run)/"predictions.parquet")
        cal = evaluation.split.eq("calibration").to_numpy()
        intervals = {n: old.phase_calibration(evaluation.loc[cal], values[n][cal]) for n in (("F", "G", "H") if h_enabled else ("F","G"))}
        old.save_json(output/"calibration.json", intervals)
        result = {"status": "multisensor_fg_frozen_evaluated_no_selection_no_promotion", "research_only": True,
            "auto_promotion": False, "original_E_rows_preserved": len(e_fit), "new_fit_rows": len(additions),
            "old_evaluation": {}, "new_2023_evaluation": {}, "correction_support": correction.support,
            "warnings": ["Clear-source observations do not establish all-weather accuracy.",
                         "OOF residuals train G; only separate evaluation estimates its performance.",
                         "Correction is at most 3 degrees, not a bound on prediction error.",
                         "No sensor ID predictor or coarse-sensor pseudo-100m training label."]}
        for split in more.SPLITS:
            mask = evaluation.split.eq(split).to_numpy()
            result["old_evaluation"][split] = score(evaluation.loc[mask], {n: p[mask] for n, p in values.items()}, intervals, correction, h_correction)
        save_predictions(evaluation, values, output/"old_evaluation_predictions.parquet", correction, h_correction)
        verify_pending_evaluation_inputs(manifest)
        new_frame, registry = load_new_evaluation(new_evaluation, freshness_audit,
                                                 manifest["input_hashes"]["freshness_audit"], known)
        fresh = validate_new(new_frame, known, areas, evaluation=True,
                             freshness_sha=manifest["input_hashes"]["freshness_audit"], registry=registry)
        if h_enabled:
            context.verify_inputs(context_hashes)
            fresh,context_audits["new_evaluation"]=context.attach(fresh,contexts.get("new_evaluation"),areas,for_fitting=False)
            result["status"]="multisensor_fgh_frozen_evaluated_no_selection_no_promotion"
            result["context_audits"]=context_audits
            result["context_correction_support"]=h_correction.support
            result["warnings"].append("H uses causal native context only when supported; absent context falls back exactly to G. No 2024 context test.")
        new_values = predictions(fresh, bundles)
        for freshness, flag in (("fresh", True), ("repeated", False)):
            result["new_2023_evaluation"][freshness] = {}
            for split in more.SPLITS:
                mask = (fresh.fresh_2023.eq(flag) & fresh.split.eq(split)).to_numpy()
                result["new_2023_evaluation"][freshness][split] = score(
                    fresh.loc[mask], {n: p[mask] for n, p in new_values.items()}, intervals, correction, h_correction)
        save_predictions(fresh, new_values, output/"new_2023_predictions.parquet", correction, h_correction)
    result["elapsed_seconds"] = time.monotonic()-started
    old.save_json(output/"results.json", result)
    old.save_json(output/"post_evaluation_freeze.json", {"version": VERSION,
        "artifacts": {n: old.sha(output/n) for n in ("fit_freeze.json", "calibration.json", "results.json",
                     "old_evaluation_predictions.parquet", "new_2023_predictions.parquet")},
        "refit_or_selection_allowed": False, "legacy_2024_opened": False, "blind_2025_opened": False})
    return result


def verify_run(run):
    run = Path(run)
    freeze = json.loads((run/"post_evaluation_freeze.json").read_text())
    if freeze.get("version") != VERSION or freeze.get("refit_or_selection_allowed") is not False:
        raise ValueError("Invalid multisensor freeze.")
    expected_post = {"fit_freeze.json", "calibration.json", "results.json", "old_evaluation_predictions.parquet", "new_2023_predictions.parquet"}
    expected_fit = {"manifest.json", "F.joblib", "G.joblib", "fitting_rows.parquet", "oof_predictions.parquet", "correction.json",
                    "oof_fold_0.joblib", "oof_fold_1.joblib", "oof_fold_2.joblib"}
    if set(freeze["artifacts"]) != expected_post:
        raise ValueError("Frozen artifact allowlist changed.")
    for name, expected in freeze["artifacts"].items():
        if old.sha(run/name) != expected: raise ValueError(f"Frozen multisensor artifact changed: {name}")
    manifest = json.loads((run/"manifest.json").read_text())
    if manifest.get("h_enabled"): expected_fit |= {"H.joblib","h_correction.json"}
    fitted = json.loads((run/"fit_freeze.json").read_text())
    if (set(fitted["artifacts"]) != expected_fit or fitted.get("refit_or_selection_allowed") is not False
            or fitted.get("models_frozen_before_new_evaluation") is not True):
        raise ValueError("Invalid pre-evaluation model freeze.")
    for name, expected in fitted["artifacts"].items():
        if old.sha(run/name) != expected: raise ValueError(f"Frozen multisensor model artifact changed: {name}")
    manifest = json.loads((run/"manifest.json").read_text())
    if any(manifest[k] != v for k, v in source_hashes().items()):
        raise ValueError("Multisensor trainer/dependency source changed after fitting.")
    if manifest["correction_specification"] != SPEC:
        raise ValueError("Frozen correction specification changed.")
    if manifest.get("h_enabled") and manifest["context_specification"]!=H_SPEC: raise ValueError("H specification changed.")
    paths, hashes = manifest["paths"], manifest["reference_hashes"]
    more.verify_experiment(paths["e_run"], paths["original_run"], paths["baseline"])
    files = {"E": Path(paths["e_run"])/"model.joblib", "A": Path(paths["original_run"])/"A/model.joblib",
             "v1": Path(paths["baseline"]), "e_freeze": Path(paths["e_run"])/"post_evaluation_freeze.json",
             "old_predictions": Path(paths["e_run"])/"predictions.parquet",
             "e_2024_results": Path(paths["e_2024"])/"results.json",
             "e_2024_predictions": Path(paths["e_2024"])/"predictions.parquet"}
    for name, path in files.items():
        if old.sha(path) != hashes[name]: raise ValueError(f"Frozen reference changed: {name}")
    return manifest


def evaluate_2024(input_path, run, output):
    manifest = verify_run(run)  # Crucially precedes any 2024 thermal-column load.
    if old.sha(input_path) != manifest["reference_hashes"]["legacy_2024_input"]:
        raise ValueError("2024 reference input differs from the pre-fit frozen hash.")
    output = Path(output)
    if output.exists(): raise FileExistsError("Use a new immutable 2024 comparison directory.")
    paths = manifest["paths"]
    bundles = {n: joblib.load(p) for n, p in {"F": Path(run)/"F.joblib", "G": Path(run)/"G.joblib",
        "E": Path(paths["e_run"])/"model.joblib", "A": Path(paths["original_run"])/"A/model.joblib", "v1": paths["baseline"]}.items()}
    if manifest.get("h_enabled"): bundles["H"]=joblib.load(Path(run)/"H.joblib")
    data = old.prepare_input(old.load_paired_input(input_path, evaluation_2024=True), evaluation_2024=True)
    needed = tuple(dict.fromkeys(list(BASE)+bundles["v1"]["features"]))
    data = data.loc[old.complete_rows(data, needed) & data.phase.isin(PHASES)].sort_values("sample_id").copy()
    previous = json.loads((Path(paths["e_2024"])/"results.json").read_text())
    if old.row_hash(data) != previous["row_sha256"] or more.weight_hash(data) != previous["weight_sha256"]:
        raise ValueError("2024 reference rows or evaluation weights changed.")
    if manifest.get("h_enabled"): data=context.missing_context(data)
    data["split"] = "legacy_test_2024"; data["temporal_partition"] = "legacy_test_2024"
    with threadpool_limits(limits=4):
        values = reuse_old_predictions(data, predictions(data, bundles), Path(paths["e_2024"])/"predictions.parquet")
    intervals = json.loads((Path(run)/"calibration.json").read_text())
    correction = bundles["G"]["model"].correction
    h_correction=bundles["H"]["model"].context_correction if manifest.get("h_enabled") else None
    if h_correction is not None and not np.array_equal(values["H"],values["G"]): raise ValueError("2024 H must equal G fallback.")
    result = score(data, values, intervals, correction, h_correction)
    result.update(status="multisensor_2024_evaluated_no_refit_no_selection_no_promotion", research_only=True,
        auto_promotion=False, input_sha256=old.sha(input_path), multisensor_freeze_sha256=old.sha(Path(run)/"post_evaluation_freeze.json"),
        warnings=["Previously inspected 2024 reference, not a blind test.", "No changes to F/G after evaluating errors; 2025 remains unopened."])
    result["H_context_evaluation"]=False
    result["H_equals_G_by_design"]=bool(manifest.get("h_enabled"))
    output.mkdir(parents=True)
    save_predictions(data, values, output/"predictions.parquet", correction, h_correction)
    result["predictions_sha256"] = old.sha(output/"predictions.parquet")
    old.save_json(output/"results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    legacy = len(sys.argv) > 1 and sys.argv[1] == "legacy-2024"
    if legacy:
        for flag in ("input", "experiment", "output"): parser.add_argument("--"+flag, type=Path, required=True)
        args = parser.parse_args(sys.argv[2:])
        result = evaluate_2024(args.input, args.experiment, args.output)
    else:
        for flag in ("original-input", "e-additions", "original-run", "e-run", "e-2024", "new-fit", "new-evaluation",
                     "freshness-audit", "areas-path", "baseline", "protocol", "output"):
            parser.add_argument("--"+flag, type=Path, required=True)
        parser.add_argument("--context-spec",type=Path)
        parser.add_argument("--h-protocol",type=Path)
        args = parser.parse_args()
        result = run_experiment(**vars(args))
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    # Preserve importable estimator class identities in saved joblib bundles.
    from .multisensor_train import main as canonical_main
    canonical_main()
