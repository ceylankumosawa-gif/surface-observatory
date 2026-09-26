"""Strict, label-preserving regional accuracy acceptance. No training or IO.

Reference rows and required groups must be frozen independently of predictions.
The evaluator cannot establish worldwide coverage from a finite pilot panel.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib

import numpy as np
import pandas as pd


VERSION = "regional-accuracy-acceptance-v1"
KEYS = ("region_id", "phase", "resolution_m", "validation_kind")
KINDS = ("date_holdout", "region_holdout", "spatial_diagnostic")


@dataclass(frozen=True)
class Policy:
    target_mae_c: float = 3.0
    minimum_dates: int = 12
    minimum_dates_per_quarter: int = 3
    minimum_rows: int = 100

    def __post_init__(self):
        if not np.isfinite(self.target_mae_c) or self.target_mae_c <= 0:
            raise ValueError("The MAE target must be finite and positive.")
        for name in ("minimum_dates", "minimum_dates_per_quarter", "minimum_rows"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Coverage minima must be positive integers.")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _dates(frame):
    value = pd.to_datetime(frame.datetime_utc, utc=True, errors="raise")
    _require(not value.isna().any(), "Missing timestamps are not admissible.")
    return value


def _identity_digest(frame):
    data = frame[["sample_id", *KEYS, "datetime_utc", "lst_c"]].sort_values("sample_id")
    return hashlib.sha256(pd.util.hash_pandas_object(data, index=False).to_numpy(dtype="<u8").tobytes()).hexdigest()


def _stats(group):
    """Equal UTC dates within this group; every pixel equal within its date."""
    finite = np.isfinite(group.predicted_lst_c.to_numpy(float))
    valid = group.loc[finite].copy()
    if valid.empty:
        return {"date_balanced_mae_c": None, "pixel_mae_c": None,
                "date_balanced_bias_c": None, "pixel_bias_c": None,
                "date_balanced_gt5_fraction": None, "date_balanced_gt7_fraction": None,
                "pixel_gt5_fraction": None, "pixel_gt7_fraction": None}
    error = valid.predicted_lst_c.to_numpy(float) - valid.lst_c.to_numpy(float)
    dates = valid._utc_date
    daily = pd.DataFrame({"date": dates.to_numpy(), "absolute": np.abs(error), "bias": error,
                          "gt5": np.abs(error) > 5, "gt7": np.abs(error) > 7}).groupby("date").mean().mean()
    return {"date_balanced_mae_c": float(daily.absolute),
            "pixel_mae_c": float(np.abs(error).mean()),
            "date_balanced_bias_c": float(daily.bias), "pixel_bias_c": float(error.mean()),
            **{f"date_balanced_gt{n}_fraction": float(daily[f"gt{n}"]) for n in (5, 7)},
            **{f"pixel_gt{n}_fraction": float((np.abs(error) > n).mean()) for n in (5, 7)}}


def evaluate(reference, predictions, training_membership, requirements, *,
             model_id, evidence_status="repeated_diagnostic", policy=Policy()):
    """Return a JSON-ready receipt and group table, retaining every expected row.

    reference: sample_id, region_id, phase, resolution_m, validation_kind,
      datetime_utc, lst_c. Unique sample IDs, finite source-screened labels.
    predictions: sample_id, predicted_lst_c, model_fit_id. Missing rows/values
      remain visible coverage failures; extra IDs or duplicate IDs are errors.
    training_membership: model_fit_id, sample_id, region_id, datetime_utc for
      EVERY estimator/correction training row associated with that prediction.
    requirements: explicit nonempty list of all required KEYS combinations.
      All observed combinations must occur here, so hard groups cannot vanish.
    evidence_status: 'fresh_locked_confirmation' or 'repeated_diagnostic'.

    Month-disjoint date tests and whole-region-disjoint region tests are checked
    against actual fitting membership. Spatial diagnostics cannot qualify.
    A fresh status is a caller's audited provenance assertion, never inferred
    from low error. Global qualification remains false for this regional panel.
    """
    _require(evidence_status in ("fresh_locked_confirmation", "repeated_diagnostic"), "Unknown evidence status.")
    _require(isinstance(model_id, str) and bool(model_id), "A model identity is required.")
    required_columns = {"sample_id", *KEYS, "datetime_utc", "lst_c"}
    _require(required_columns.issubset(reference.columns), "Reference schema is incomplete.")
    _require({"sample_id", "predicted_lst_c", "model_fit_id"}.issubset(predictions.columns), "Prediction schema is incomplete.")
    _require({"sample_id", "region_id", "datetime_utc", "model_fit_id"}.issubset(training_membership.columns), "Training membership schema is incomplete.")
    for name, frame in (("reference", reference), ("predictions", predictions)):
        _require(not frame.sample_id.isna().any() and not frame.sample_id.duplicated().any(), f"Invalid or duplicate {name} sample IDs.")
    _require(not reference[list(KEYS)].isna().any().any(), "Reference grouping metadata is missing.")
    _require(reference.phase.isin(["day", "night"]).all(), "Unknown solar phase.")
    _require(reference.validation_kind.isin(KINDS).all(), "Unknown independence kind.")
    _require(np.isfinite(reference.lst_c.to_numpy(float)).all(), "Reference labels must be finite; never silently exclude them.")
    _require(not set(predictions.sample_id) - set(reference.sample_id), "Predictions contain unexpected sample IDs.")
    _require(not predictions.model_fit_id.isna().any(), "Every supplied prediction needs a model fit identity.")
    _require(not training_membership[["sample_id", "region_id", "model_fit_id"]].isna().any().any(), "Missing training identity.")
    requested = pd.DataFrame(requirements)
    _require(not requested.empty and set(KEYS).issubset(requested.columns), "Freeze a nonempty coverage registry.")
    _require(not requested[list(KEYS)].isna().any().any() and not requested.duplicated(list(KEYS)).any(), "Invalid coverage registry.")
    _require(requested.phase.isin(["day", "night"]).all() and requested.validation_kind.isin(KINDS).all(), "Invalid required group.")
    _require(np.isfinite(requested.resolution_m.to_numpy(float)).all() and requested.resolution_m.gt(0).all(), "Invalid required resolution.")
    expected = set(map(tuple, requested[list(KEYS)].to_numpy()))
    observed = set(map(tuple, reference[list(KEYS)].to_numpy()))
    _require(observed.issubset(expected), "Observed groups cannot be omitted from the required registry.")
    data = reference.copy()
    data["datetime_utc"] = _dates(data)
    data["_utc_date"] = data.datetime_utc.dt.strftime("%Y-%m-%d")
    data["_month"] = data.datetime_utc.dt.strftime("%Y-%m")
    data["_quarter"] = data.datetime_utc.dt.quarter
    data = data.merge(predictions[["sample_id", "predicted_lst_c", "model_fit_id"]], on="sample_id", how="left", validate="one_to_one")
    training = training_membership.copy()
    training["datetime_utc"] = _dates(training)
    training["_month"] = training.datetime_utc.dt.strftime("%Y-%m")
    _require(not training.duplicated(["model_fit_id", "sample_id"]).any(), "Duplicate fitting identities.")
    training_by_fit = {k: g for k, g in training.groupby("model_fit_id")}
    data["_leakage"] = False
    for fit_id, held in data.dropna(subset=["model_fit_id"]).groupby("model_fit_id"):
        _require(fit_id in training_by_fit, "Prediction refers to an unknown fitting population.")
        train = training_by_fit[fit_id]
        leaked = held.sample_id.isin(train.sample_id)
        leaked |= held.validation_kind.eq("date_holdout") & held._month.isin(train._month)
        leaked |= held.validation_kind.eq("region_holdout") & held.region_id.isin(train.region_id)
        data.loc[held.index, "_leakage"] = leaked
    rows = []
    groups = {key: frame for key, frame in data.groupby(list(KEYS), sort=False)}
    for key in sorted(expected):
        group = groups.get(key, data.iloc[:0])
        stats = _stats(group)
        n, dates = len(group), group._utc_date.nunique()
        finite = np.isfinite(group.predicted_lst_c.to_numpy(float))
        quarters = {str(q): int(group.loc[group._quarter.eq(q), "_utc_date"].nunique()) for q in range(1, 5)}
        reasons = []
        if n == 0: reasons.append("missing_group")
        if n < policy.minimum_rows: reasons.append("insufficient_reference_rows")
        if dates < policy.minimum_dates: reasons.append("insufficient_independent_dates")
        if any(value < policy.minimum_dates_per_quarter for value in quarters.values()): reasons.append("insufficient_seasonal_dates")
        if int(finite.sum()) != n: reasons.append("missing_predictions")
        if group._leakage.any(): reasons.append("training_overlap")
        if key[3] == "spatial_diagnostic": reasons.append("spatial_only_not_independent_date_or_region_test")
        if evidence_status != "fresh_locked_confirmation": reasons.append("previously_inspected_evidence")
        point_pass = n > 0 and finite.all() and stats["date_balanced_mae_c"] <= policy.target_mae_c and stats["pixel_mae_c"] <= policy.target_mae_c
        if n and not point_pass: reasons.append("mae_target_or_coverage_failed")
        quarter_stats = []
        for q, quarter in group.groupby("_quarter"):
            qs = _stats(quarter)
            quarter_stats.append({"quarter": int(q), "dates": int(quarter._utc_date.nunique()), **qs})
            if qs["date_balanced_mae_c"] is not None and (qs["date_balanced_mae_c"] > policy.target_mae_c or qs["pixel_mae_c"] > policy.target_mae_c):
                reasons.append(f"quarter_{q}_mae_target_failed")
        rows.append({**dict(zip(KEYS, key)), "reference_rows": n, "predicted_rows": int(finite.sum()),
                     "prediction_coverage": float(finite.mean()) if n else 0., "independent_dates": int(dates),
                     "quarter_date_counts": quarters, "quarter_metrics": quarter_stats,
                     "observed": bool(n), "point_target_met": bool(point_pass), "passed": not reasons,
                     "reasons": reasons, **stats})
    passed = sum(row["passed"] for row in rows)
    summary = {"version": VERSION, "model_id": model_id, "target_mae_c": policy.target_mae_c,
               "policy": asdict(policy), "evidence_status": evidence_status,
               "reference_rows": len(reference), "reference_sha256": _identity_digest(reference),
               "required_groups": len(rows), "observed_groups": len(observed), "passed_groups": passed,
               "missing_groups": len(expected - observed), "point_target_met_groups": sum(r["point_target_met"] for r in rows),
               "regional_panel_target_met": passed == len(rows), "global_target_met": False,
               "global_coverage_status": "A regional panel cannot establish all-region global coverage.",
               "status": "target_unmet" if passed < len(rows) else "regional_panel_only",
               "mean_definition": "equal UTC-date MAE within each region/phase/resolution, plus ordinary pixel MAE; both <= target",
               "groups": rows}
    return summary
