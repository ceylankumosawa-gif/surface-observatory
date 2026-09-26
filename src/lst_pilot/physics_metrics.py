"""Matched, descriptive evaluation for the frozen physics-correction experiment.

This module performs no fitting or file access. Point estimates reuse the prior
experiment's weighting and metric implementation. Bootstrap intervals describe
resampling observed global UTC dates, conditional on the fixed models, cohort
and segment weights; they do not establish generalization or independence of
weather on successive dates. Dates shared by different pilots resample together.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import option_b_train as old

BOOTSTRAP_SEED = 2708
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_MIN_DATES = 6
COMPARISONS = (("WP", "F"), ("WP", "WR"), ("P", "F"),
               ("W", "F"), ("WR", "F"), ("WP", "W"))
PAIRED_METRICS = (
    "mae_c", "bias_c", "fraction_abs_error_gt_3c",
    "fraction_abs_error_gt_5c", "fraction_abs_error_gt_7c",
    "centered_contrast_mae_c", "unweighted_pixel_mae_c",
    "unweighted_gt5_fraction", "unweighted_gt7_fraction",
)
BOOTSTRAP_SEGMENTS = frozenset(("overall", "region_phase", "region_phase_season"))
_REQUIRED = (
    "sample_id", "region_id", "datetime_utc", "utc_day", "phase",
    "acquisition_id", "weight_surface_group", "season", "air_group",
    "label_product", "lst_c",
)
_INTERPRETATION = (
    "Descriptive percentile interval from global UTC-date clusters, conditional "
    "on these fixed models, observed rows and segment weights; not proof of "
    "generalization. Successive dates may remain dependent."
)


def _arrays(values, n, name, *, boolean=False):
    output = {}
    for model, value in values.items():
        if not isinstance(model, str) or not model:
            raise ValueError(f"{name} model names must be nonempty strings.")
        arr = np.asarray(value)
        if arr.shape != (n,):
            raise ValueError(f"{name}[{model}] must align positionally with every input row.")
        if boolean:
            # Do not turn strings such as 'false', missing values or NaNs into True.
            if arr.dtype.kind != "b":
                raise ValueError(f"{name}[{model}] must contain explicit boolean values.")
        else:
            arr = np.asarray(value, dtype=float)
            if not np.isfinite(arr).all():
                raise ValueError(f"{name}[{model}] contains nonfinite values; rows cannot be dropped.")
        output[model] = arr
    return output


def _prepare(frame):
    missing = sorted(set(_REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError(f"Evaluation columns missing: {missing}")
    data = frame.copy().reset_index(drop=True)
    for column in set(_REQUIRED) - {"lst_c", "datetime_utc", "utc_day"}:
        if data[column].isna().any():
            raise ValueError(f"Missing {column} would silently remove a weighting/grouping row.")
    if data.sample_id.duplicated().any():
        raise ValueError("Evaluation sample_id must be unique.")
    time = pd.to_datetime(data.datetime_utc, utc=True, errors="raise")
    day = pd.to_datetime(data.utc_day, utc=True, errors="raise")
    if time.isna().any() or day.isna().any() or not day.eq(time.dt.floor("D")).all():
        raise ValueError("utc_day must equal the global UTC date of datetime_utc.")
    data["datetime_utc"], data["utc_day"] = time, day
    if not np.isfinite(data.lst_c.to_numpy(dtype=float)).all():
        raise ValueError("Observed LST must be finite; evaluation rows cannot be dropped.")
    # Centered contrast is linear in fixed within-acquisition deviations only
    # when each acquisition belongs to one resampled date cluster.
    if data.groupby(["region_id", "acquisition_id"], observed=True).utc_day.nunique().gt(1).any():
        raise ValueError("One acquisition spans multiple UTC dates; date-cluster contrast is ambiguous.")
    data["_observed_surface_group"] = np.select(
        [data.lst_c.le(0), data.lst_c.ge(35)], ["cold", "hot"], default="mild"
    )
    return data


def _segments(data):
    yield "overall", "overall", np.arange(len(data))
    for kind, columns in (
        ("region_phase", ["region_id", "phase"]),
        ("region_phase_season", ["region_id", "phase", "season"]),
        ("region_phase_air", ["region_id", "phase", "air_group"]),
        ("region_phase_surface", ["region_id", "phase", "_observed_surface_group"]),
        ("source", ["label_product"]),
    ):
        for key, indexes in data.groupby(columns, observed=True, sort=True).indices.items():
            label = "|".join(map(str, key if isinstance(key, tuple) else (key,)))
            yield kind, label, np.asarray(indexes, dtype=int)


def _contributions(data, prediction, weights):
    """Per-row numerators; denominators are weight sum or raw row count."""
    error = prediction - data.lst_c.to_numpy(dtype=float)
    absolute = np.abs(error)
    centered = data[["region_id", "acquisition_id"]].copy()
    centered["_error_weight"] = error * weights
    centered["_weight"] = weights
    grouped = centered.groupby(["region_id", "acquisition_id"], observed=True)
    mean = (grouped._error_weight.transform("sum") /
            grouped._weight.transform("sum")).to_numpy()
    return {
        "mae_c": absolute * weights,
        "bias_c": error * weights,
        **{f"fraction_abs_error_gt_{threshold}c":
           (absolute > threshold) * weights for threshold in (3, 5, 7)},
        "centered_contrast_mae_c": np.abs(error - mean) * weights,
        "unweighted_pixel_mae_c": absolute,
        "unweighted_gt5_fraction": (absolute > 5).astype(float),
        "unweighted_gt7_fraction": (absolute > 7).astype(float),
    }


def _bootstrap(data, predictions, weights):
    """Aggregate first, then jointly resample dates without rebalancing weights."""
    dates, code = np.unique(data.utc_day.to_numpy(), return_inverse=True)
    count = len(dates)
    # Multinomial cluster multiplicities are ordinary sampling of n dates with
    # replacement. One draw matrix is shared by every model and comparison.
    draws = np.random.default_rng(BOOTSTRAP_SEED).multinomial(
        count, np.full(count, 1.0 / count), size=BOOTSTRAP_DRAWS
    )
    date_weights = np.bincount(code, weights=weights, minlength=count)
    date_rows = np.bincount(code, minlength=count)
    denominator = draws @ date_weights
    raw_denominator = draws @ date_rows
    output = {}
    for model, predicted in predictions.items():
        contributions = _contributions(data, predicted, weights)
        names = list(PAIRED_METRICS)
        numerators = np.column_stack([
            np.bincount(code, weights=contributions[name], minlength=count) for name in names
        ])
        sampled = draws @ numerators
        output[model] = {
            name: sampled[:, i] / (raw_denominator if name.startswith("unweighted_") else denominator)
            for i, name in enumerate(names)
        }
    return output


def evaluate_cohort(frame, predictions, adjustments, supported, cohort):
    """Return metric rows and paired comparison rows for one immutable cohort.

    Every array is positional and must contain one finite value (or explicit
    boolean) per row. ``adjustments`` and ``supported`` have matching keys for
    the new correction arms. Their weighted summaries use the same segment
    weights as the errors; mean correction size includes exact-zero fallbacks.
    Unavailable comparison models are skipped; no samples are filtered.

    ``date_count`` retains the old definition of summed pilot-dates.
    ``utc_date_count`` and bootstrap clusters count distinct dates globally.
    All paired deltas are candidate minus reference, so negative MAE/tail/contrast
    deltas favor the candidate; the sign of a bias delta alone is not an improvement.
    """
    data = _prepare(frame)
    if not predictions:
        raise ValueError("At least one prediction model is required.")
    pred = _arrays(predictions, len(data), "predictions")
    change = _arrays(adjustments, len(data), "adjustments")
    support = _arrays(supported, len(data), "supported", boolean=True)
    if set(change) != set(support) or not set(change).issubset(pred):
        raise ValueError("Adjustment/support keys must match and name supplied prediction models.")
    metric_rows, paired_rows = [], []
    comparisons = [(a, b) for a, b in COMPARISONS if a in pred and b in pred]
    for kind, segment, indexes in _segments(data):
        subset = data.iloc[indexes]
        w = old.balanced_weights(subset)
        row_base = {"cohort": str(cohort), "segment_type": kind, "segment": segment}
        model_metrics = {}
        for model, values in pred.items():
            row = {**row_base, "model": model, **old.metrics(subset, values[indexes])}
            error = np.abs(values[indexes] - subset.lst_c.to_numpy(dtype=float))
            row.update({f"unweighted_gt{t}_fraction": float(np.mean(error > t)) if len(error) else None
                        for t in (5, 7)})
            if model in change:
                magnitude = np.abs(change[model][indexes])
                flags = support[model][indexes]
                row.update({
                    "adjusted_fraction": float(np.sum(w * (magnitude > 0))) if len(w) else None,
                    "mean_abs_adjustment_c": float(np.sum(w * magnitude)) if len(w) else None,
                    "supported_fraction": float(np.sum(w * flags)) if len(w) else None,
                    "unweighted_adjusted_fraction": float(np.mean(magnitude > 0)) if len(w) else None,
                    "unweighted_mean_abs_adjustment_c": float(np.mean(magnitude)) if len(w) else None,
                    "unweighted_supported_fraction": float(np.mean(flags)) if len(w) else None,
                })
            metric_rows.append(row)
            model_metrics[model] = row
        dates = int(subset.utc_day.nunique())
        eligible_type = kind in BOOTSTRAP_SEGMENTS
        can_bootstrap = eligible_type and dates >= BOOTSTRAP_MIN_DATES
        bootstrap_models = {name for pair in comparisons for name in pair}
        samples = (_bootstrap(subset, {m: pred[m][indexes] for m in sorted(bootstrap_models)}, w)
                   if can_bootstrap and comparisons else {})
        status = ("descriptive_available" if can_bootstrap else
                  "sparse_global_utc_dates" if eligible_type else "not_requested_for_segment")
        reason = (None if can_bootstrap else
                  f"At least {BOOTSTRAP_MIN_DATES} distinct global UTC dates are required; observed {dates}."
                  if eligible_type else "Intervals are predeclared only for overall, region/phase and region/phase/season.")
        for candidate, reference in comparisons:
            a, b = model_metrics[candidate], model_metrics[reference]
            row = {**row_base, "comparison": f"{candidate}_vs_{reference}",
                   "candidate": candidate, "reference": reference,
                   "status": a["status"], "n": len(subset),
                   "date_count": int(a["date_count"]), "utc_date_count": dates,
                   "acquisition_count": int(subset.acquisition_id.nunique()),
                   "sample_id_sha256": old.row_hash(subset),
                   "bootstrap_status": status, "bootstrap_reason": reason,
                   "bootstrap_draws": BOOTSTRAP_DRAWS if can_bootstrap else 0,
                   "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_cluster_count": dates,
                   "bootstrap_cluster": "global_utc_date", "bootstrap_confidence_level": 0.95,
                   "bootstrap_interpretation": _INTERPRETATION}
            for metric in PAIRED_METRICS:
                row[f"delta_{metric}"] = float(a[metric] - b[metric]) if len(subset) else None
                row[f"delta_{metric}_ci95"] = (
                    np.quantile(samples[candidate][metric] - samples[reference][metric], [0.025, 0.975]).tolist()
                    if can_bootstrap else None
                )
            paired_rows.append(row)
    return metric_rows, paired_rows
