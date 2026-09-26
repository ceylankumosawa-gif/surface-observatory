"""Create research figures from completed result JSON, never from thermal arrays."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

COLORS = {"Air": "#a6b1ba", "v1": "#566b80", "S0": "#d49c60", "A": "#457a91",
          "B": "#26776a", "C": "#9b78a3", "D": "#263e5b", "Selected": "#26776a"}
LABELS = {"greater_london": "Greater London", "sioux_falls": "Sioux Falls", "cabauw": "Cabauw"}
METRICS = (("mae_c", "Mean absolute error (°C)", 1),
           ("bias_c", "Mean bias: prediction − observation (°C)", 1),
           ("fraction_abs_error_gt_5c", "Errors larger than 5°C (%)", 100),
           ("fraction_abs_error_gt_7c", "Errors larger than 7°C (%)", 100))
SPLITS = {"development": "2023H1 development · used for feature selection",
          "calibration": "2023H2 calibration · used for empirical intervals",
          "heldout_spatial": "Reserved spatial blocks · not used for selection",
          "heldout_region": "Entire reserved region · not used for selection",
          "legacy_2024": "2024 legacy comparison · previously inspected test cohort"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _research_groups(result, split, group):
    groups = {}
    source = result["candidates"].get("A", next(iter(result["candidates"].values())))
    scores = source["evaluation"][split]
    for name, key in (("Air", "air_only"), ("v1", "v1")):
        if key in scores:
            values = scores[key].get("by_region_phase", {}).get(group)
            if values:
                groups[name] = values
    for name in ("S0", "A", "B", "C", "D"):
        if name in result["candidates"]:
            values = result["candidates"][name]["evaluation"][split]["model"].get("by_region_phase", {}).get(group)
            if values:
                groups[name] = values
    return groups


def _legacy_groups(result, group):
    groups = {}
    for name, key in (("Air", "air_only"), ("v1", "v1"), (result["selected_candidate"], "candidate")):
        values = result["metrics"][key].get("by_region_phase", {}).get(group)
        if values:
            groups[name] = values
    return groups


def comparison_figure(result, split, region, destination, *, legacy=False):
    fig, axes = plt.subplots(4, 2, figsize=(12, 12), layout="constrained")
    any_observations = False
    for col, phase in enumerate(("day", "night")):
        group = f"{region}|{phase}"
        scores = _legacy_groups(result, group) if legacy else _research_groups(result, split, group)
        hashes = {v.get("sample_id_sha256") for v in scores.values()}
        if len(hashes) > 1:
            plt.close(fig)
            raise ValueError(f"Cannot present unmatched candidates as paired: {split}/{group}")
        labels = list(scores)
        any_observations |= bool(labels)
        support = next(iter(scores.values())) if scores else {}
        title = (f"{phase.title()} · {support['date_count']} dates · {support['n']:,} cells"
                 if scores else f"{phase.title()} · no observations")
        for row, (metric, ylabel, scale) in enumerate(METRICS):
            ax = axes[row, col]
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#e8edf0", linewidth=.7)
            if row == 0:
                ax.set_title(title, fontsize=11, weight="bold")
            if not scores:
                ax.text(.5, .5, "Unsupported: no observations", transform=ax.transAxes,
                        ha="center", va="center", color="#6e7b85", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            values = [scores[k][metric] * scale for k in labels]
            bars = ax.bar(labels, values, color=[COLORS.get(k, COLORS["Selected"]) for k in labels], width=.68)
            ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=3, fontsize=8)
            if metric == "bias_c":
                bound = max(.5, max(abs(v) for v in values) * 1.35)
                ax.set_ylim(-bound, bound)
                ax.axhline(0, color="#495762", linewidth=.7)
            else:
                ax.set_ylim(0, max(1, max(values) * 1.3))
            ax.tick_params(axis="x", labelsize=9)
    fig.suptitle(f"{LABELS.get(region, region)}\n{SPLITS[split]}", fontsize=15, weight="bold")
    fig.supxlabel("Matched observations; date/acquisition/surface-balanced errors. Sparse dates and clear-sky selection limit interpretation.\n"
                  "Air = input air temperature alone · v1 = original model · S0 = legacy refit · A/B/C/D = feature variants\n"
                  "v1 and S0 have daytime-only training; their nighttime bars are diagnostic comparisons.",
                  fontsize=8.5)
    fig.savefig(destination, dpi=170, facecolor="white")
    plt.close(fig)
    return {"file": destination.name, "region_id": region, "split": split,
            "has_observations": any_observations, "sha256": digest(destination)}


def dataflow_figure(selection, destination):
    fig, ax = plt.subplots(figsize=(14, 7.4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.4)
    ax.axis("off")
    def box(x, y, w, h, title, body, color="#eaf2f0"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.10",
                                   edgecolor="#76938b", facecolor=color, linewidth=1))
        ax.text(x+w/2, y+h-.25, title, ha="center", va="top", fontsize=11, weight="bold", color="#233a40")
        ax.text(x+w/2, y+h-.65, body, ha="center", va="top", fontsize=9.5, linespacing=1.5, color="#344e56")
    def arrow(start, end):
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "-|>", "color": "#56736d", "lw": 1.3})
    box(.3, 5.15, 3.2, 1.5, "Inputs for each place and time", "Air temperature + archived weather\nIndependent surface properties\nClimate class + sun position")
    box(.3, 2.95, 3.2, 1.45, "Extra history tested in B and D", "Heating and cooling over 6–24 h\nOnly completed prior hours\nNo future or reference-year substitution")
    box(.3, .7, 3.2, 1.45, "Satellite observations", "Observed clear-sky LST\nCloud, quality and location checks\nTemperature is a label, not an input", "#eef1f7")
    box(4.65, 3.35, 3.75, 2.3, "One paired learning table", "Same pixel and acquisition time\nLST − air temperature is the target\n2021–2022 fitting rows\nReserved places and buffers excluded")
    box(9.65, 3.35, 3.75, 2.3, "One small climate-aware model", "Learn the surface–air difference\nCompare S0 / A / B / C / D fairly\n2023H1 selects features\nLST = air + predicted difference")
    chosen = (f"Candidate {selection['selected_candidate']} retained for comparison; development coverage is limited."
              if selection['status']=='unsupported_development' else f"Candidate {selection['selected_candidate']} chosen by the development comparison.")
    box(5.0, .55, 7.95, 1.55, "Separate evidence after selection", chosen+"\n"
        "2023H2 empirical intervals · Cabauw and reserved blocks · separate 2024 legacy check\n"
        "2025 remains blind. Experimental artifacts do not replace the live model.", "#faf2e7")
    arrow((3.65, 5.75), (4.5, 4.95))
    arrow((3.65, 3.6), (4.5, 4.1))
    arrow((3.65, 1.4), (4.65, 3.2))
    arrow((8.55, 4.5), (9.5, 4.5))
    arrow((11.5, 3.2), (11.5, 2.25))
    ax.text(7, 7.05, "How the exploratory model learns", ha="center", fontsize=18, weight="bold", color="#233a40")
    fig.savefig(destination, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"file": destination.name, "sha256": digest(destination)}


def render_figures(run_dir, output_dir, *, legacy_dir=None):
    run = Path(run_dir)
    path = run / "results.json"
    result = json.loads(path.read_text())  # Missing real results fail; no placeholder scores.
    if result.get("status") != "exploratory_fitted_no_promotion":
        raise ValueError("A completed exploratory fit is required before plotting results.")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("Use a new figures directory to preserve existing artifacts.")
    output.mkdir(parents=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    figures = [dataflow_figure(result["selection"], output / "dataflow.svg")]
    for split in ("development", "heldout_spatial"):
        for region in ("greater_london", "sioux_falls"):
            figures.append(comparison_figure(result, split, region, output / f"{split}_{region}.png"))
    figures.append(comparison_figure(result, "heldout_region", "cabauw", output / "heldout_region_cabauw.png"))
    sources = {str(path.resolve()): digest(path)}
    if legacy_dir is not None:
        lp = Path(legacy_dir) / "results.json"
        legacy = json.loads(lp.read_text())
        if legacy.get("status") != "legacy_2024_evaluated_no_refit_no_promotion":
            raise ValueError("A completed frozen-candidate 2024 evaluation is required.")
        if legacy["selected_candidate"] != result["selection"]["selected_candidate"]:
            raise ValueError("Legacy evaluation and research run selected different candidates.")
        sources[str(lp.resolve())] = digest(lp)
        figures.append(comparison_figure(legacy, "legacy_2024", "greater_london", output / "legacy_2024_greater_london.png", legacy=True))
        figures.append(comparison_figure(legacy, "legacy_2024", "sioux_falls", output / "legacy_2024_sioux_falls.png", legacy=True))
    manifest = {"source_results": sources, "figures": figures, "raw_thermal_arrays_read": False,
                "metrics_fabricated": False, "production_changed": False}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = ["# Exploratory fit figures", "", "These figures use completed result JSON only. Missing observation groups remain explicitly unsupported.", ""]
    lines += [f"![{f['file']}]({f['file']})\n" for f in figures]
    (output / "README.md").write_text("\n".join(lines))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--legacy-evaluation", type=Path)
    args = parser.parse_args()
    result = render_figures(args.run, args.output, legacy_dir=args.legacy_evaluation)
    print(json.dumps({"figures": len(result["figures"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
