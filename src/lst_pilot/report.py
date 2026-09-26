"""Small, reproducible figures and inference timing for a completed remote run."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def make_report(model_dir, data_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from .model import predict_frame

    root = Path(model_dir)
    metrics = json.loads((root / "metrics.json").read_text())
    predictions = pd.read_parquet(root / "heldout_predictions.parquet")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#f7f8fa", "axes.facecolor": "#f7f8fa"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.7), layout="constrained")
    for ax, split, title in zip(axes, ("test_region", "test_temporal"),
                               ("London · excluded from training", "Other regions · later dates")):
        group = predictions[predictions.split.eq(split)]
        if group.empty:
            ax.text(.5, .5, "No observations in this split", transform=ax.transAxes, ha="center")
            continue
        ax.scatter(group.lst_c, group.predicted_lst_c, s=9, alpha=.27, color="#167a80", rasterized=True)
        limits = [min(group.lst_c.min(), group.predicted_lst_c.min()) - 3,
                  max(group.lst_c.max(), group.predicted_lst_c.max()) + 3]
        ax.plot(limits, limits, color="#50576b", linestyle="--", linewidth=1)
        ax.set(xlim=limits, ylim=limits, xlabel="Observed satellite LST (°C)", ylabel="Predicted LST (°C)")
        rmse = np.sqrt(np.mean(group.error_c ** 2))
        mae = np.mean(np.abs(group.error_c))
        days = group.assign(day=pd.to_datetime(group.datetime_utc, utc=True).dt.date).groupby(["region_id", "day"]).ngroups
        ax.set_title(f"{title}\nRMSE {rmse:.2f} °C · MAE {mae:.2f} °C", loc="left", fontweight="bold", pad=14)
        ax.text(.04, .96, f"{len(group):,} samples / {days} region-days", transform=ax.transAxes, va="top", fontsize=9)
        ax.grid(alpha=.15)
    fig.suptitle("First LST transfer experiment", fontsize=19, fontweight="bold", x=.05, ha="left")
    fig.supxlabel("Clear-sky daytime labels only. Samples within a scene are correlated; this is an early pilot assessment.", fontsize=9)
    fig.savefig(root / "evaluation.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set(xlim=(0, 12), ylim=(0, 7))
    ax.axis("off")
    def box(x, y, w, h, title, detail, color="#e5eef0"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12,rounding_size=.12", fc=color, ec="none"))
        ax.text(x+.2, y+h-.3, title, fontsize=12, weight="bold", va="top", color="#172b40")
        ax.text(x+.2, y+h-.72, detail, fontsize=10, va="top", linespacing=1.6, color="#33495d")
    def arrow(start, end):
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#40556c", "lw": 1.6})
    ax.text(.1, 6.7, "How the implemented model works", fontsize=21, weight="bold", color="#172b40")
    ax.text(.1, 6.2, "Learn the surface–air difference from comparable conditions and surface properties.", fontsize=11)
    box(.2, 3.95, 3.15, 1.55, "1  Local air and weather", "Station temperature + ERA5\nHumidity, wind, rain and snow\nRadiation and recent history")
    box(.2, 1.85, 3.15, 1.55, "2  Describe each place", "100 m optical properties\nKöppen climate and sun position\nElevation, slope and aspect")
    box(4.1, 2.55, 3.15, 2.2, "3  Compact tree model", "Training target:\nobserved LST − local air\n\nSimilar inputs share\nlearned thermal responses.", "#dcece6")
    box(8.05, 2.55, 3.45, 2.2, "4  Add local air back", "Predicted LST in °C\n+ empirical error interval\n\nTest on whole unseen regions\nand later dates.", "#e8e4f3")
    arrow((3.45, 4.65), (4, 4.1))
    arrow((3.45, 2.55), (4, 3.25))
    arrow((7.4, 3.6), (7.9, 3.6))
    ax.text(.2, .85, "Evidence now", weight="bold", fontsize=11, color="#172b40")
    ax.text(2.2, .85, "Clear-sky daytime satellite observations; radiation reference audit is available.", fontsize=10)
    ax.text(.2, .4, "Still to validate", weight="bold", fontsize=11, color="#172b40")
    ax.text(2.2, .4, "Night, clouds, hourly maps, detailed shade and global transfer.", fontsize=10)
    fig.savefig(root / "model_flow.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    if (root / "inference_benchmark.json").exists():
        return  # Redrawing a figure must not replace the original measured benchmark.
    bundle = joblib.load(root / "model.joblib")
    data = pd.read_parquet(data_path)
    features = bundle["features"]
    batch = pd.concat([data[features]] * max(1, int(np.ceil(100000 / len(data)))), ignore_index=True).iloc[:100000]
    predict_frame(batch.iloc[:1000], bundle)  # warm kernels; benchmark excludes data retrieval
    start = time.perf_counter()
    result = predict_frame(batch, bundle)
    elapsed = time.perf_counter() - start
    timing = {"rows": len(batch), "seconds": elapsed, "rows_per_second": len(batch) / elapsed,
              "model_file_bytes": (root / "model.joblib").stat().st_size,
              "all_predictions_finite": bool(np.isfinite(result.predicted_lst_c).all()),
              "scope": "Warm in-memory full prediction wrapper, including intervals; repeated real input rows. Excludes weather, raster reads, static feature construction, compression, writing and network. Not a global service benchmark."}
    (root / "inference_benchmark.json").write_text(json.dumps(timing, indent=2) + "\n")
    print(json.dumps(timing, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    make_report(args.model_dir, args.input)


if __name__ == "__main__":
    main()
