"""Read completed F/G experiments; emit a new, immutable figures/report folder.

Frozen JSON supplies weighted metrics. Hash-bound saved prediction tables supply
ordinary pixel tails; this never opens satellite assets or fits/selects a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import multisensor_train as train
from . import option_b_train as metrics

MODELS = ("E", "F", "G")
LABELS = {"E": "E · previous data", "F": "F · additional fine labels", "G": "G · calibrated F", "H":"H · native context or G fallback"}
COLORS = {"E": "#647a91", "F": "#36838e", "G": "#23775b", "H":"#aa743b"}
SOURCES = {"landsat_c2_l2": "Landsat", "ecostress_v2": "ECOSTRESS", "aster_ast08_v004": "ASTER"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_predictions(path, expected_sha, years):
    if sha(path) != expected_sha:
        raise ValueError("Saved prediction table differs from its completed result/freeze.")
    stamps = pd.to_datetime(pd.read_parquet(path, columns=["datetime_utc"]).datetime_utc, utc=True)
    if stamps.isna().any() or not stamps.dt.year.isin(years).all():
        raise ValueError("Prediction years outside this comparison; label columns were not read.")
    data = pd.read_parquet(path)
    if data.sample_id.duplicated().any():
        raise ValueError("Saved comparison duplicates an observation identity.")
    data["datetime_utc"] = stamps
    data["utc_day"] = stamps.dt.floor("D")
    return data


def fitting_coverage(frame, manifest):
    frame = frame.copy()
    frame["original_E"] = metrics.strict_bool(frame.original_E, "original_E")
    frame["utc_day"] = pd.to_datetime(frame.datetime_utc, utc=True).dt.floor("D")
    if (len(frame) != manifest["fit_rows"] or int(frame.original_E.sum()) != manifest["e_fit_rows"] or
            metrics.row_hash(frame) != manifest["fit_row_sha256"] or
            metrics.row_hash(frame.loc[frame.original_E]) != manifest["e_fit_row_sha256"]):
        raise ValueError("Coverage table disagrees with the original-E preservation manifest.")
    result = {"by_source_phase": [], "by_pilot_source_phase": []}
    for name, keys in (("by_source_phase", ["label_product", "phase"]),
                       ("by_pilot_source_phase", ["region_id", "label_product", "phase"])):
        for key, group in frame.groupby(keys, sort=True):
            row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
            for model, subset in (("E", group.loc[group.original_E]), ("F_G", group)):
                row[model] = {"rows": len(subset), "pilot_dates": len(subset[["region_id", "utc_day"]].drop_duplicates()),
                              "utc_dates": subset.utc_day.nunique(), "acquisitions": subset.acquisition_id.nunique()}
            result[name].append(row)
    return result


def select_rows(frame, level, key):
    fields = {"by_region": ["region_id"], "by_phase": ["phase"], "by_air_group": ["air_group"],
              "by_region_phase": ["region_id", "phase"],
              "by_region_phase_air": ["region_id", "phase", "air_group"],
              "by_label_product": ["label_product"]}
    if level == "overall": return frame
    values = str(key).split("|")
    if level not in fields or len(values) != len(fields[level]):
        raise ValueError("Unknown comparison grouping.")
    keep = np.ones(len(frame), bool)
    for field, value in zip(fields[level], values): keep &= frame[field].eq(value).to_numpy()
    return frame.loc[keep]


def matched_group(container, frame, label, level="overall", key=None):
    selected = select_rows(frame, level, key)
    scores = {name: container["metrics"][name].get(level, {}) for name in (*MODELS,*(["H"] if "H" in container["metrics"] else []))}
    if level != "overall": scores = {name: score.get(key, {}) for name, score in scores.items()}
    if not len(selected):
        if any(s.get("n", 0) for s in scores.values()):
            raise ValueError("Metric group claims observations absent from saved predictions.")
        return {"label": label, "status": "unsupported_no_observations", "metrics": {}}
    row_sha = metrics.row_hash(selected)
    pilot_dates = len(selected[["region_id", "utc_day"]].drop_duplicates())
    output = {}
    observed = selected.lst_c.to_numpy(float)
    for name, score in scores.items():
        if score.get("n") != len(selected) or score.get("sample_id_sha256") != row_sha or score.get("date_count") != pilot_dates:
            raise ValueError("Models or saved metrics do not describe exactly the same observations.")
        error = selected[f"{name}_lst_c"].to_numpy(float)-observed
        if not np.isfinite(error).all() or not np.isclose(np.mean(np.abs(error)), score["unweighted_pixel_mae_c"], rtol=0, atol=1e-10):
            raise ValueError("Saved predictions do not reproduce the frozen ordinary pixel MAE.")
        output[name] = {**score, **{f"raw_fraction_abs_error_gt_{t}c": float(np.mean(np.abs(error) > t)) for t in (5, 7)}}
    correction = selected.G_lst_c.to_numpy()-selected.F_lst_c.to_numpy()
    if not np.allclose(correction, selected.G_adjustment_c, rtol=0, atol=1e-10):
        raise ValueError("Saved G adjustment differs from the F/G predictions.")
    if np.max(np.abs(correction)) > 3+1e-10:
        raise ValueError("Saved correction exceeds its declared adjustment limit.")
    supported = metrics.strict_bool(selected.G_correction_supported, "G_correction_supported").to_numpy()
    if np.any(np.abs(correction[~supported]) > 1e-10):
        raise ValueError("Unsupported correction rows must retain F.")
    h_audit={}
    if "H" in output:
        total=selected.H_lst_c.to_numpy()-selected.F_lst_c.to_numpy()
        h_supported=metrics.strict_bool(selected.H_correction_supported,"H_correction_supported").to_numpy()
        eligible=metrics.strict_bool(selected.coarse_context_eligible,"coarse_context_eligible").to_numpy()
        if (not np.allclose(total,selected.H_adjustment_c,rtol=0,atol=1e-10) or np.abs(total).max()>3+1e-10
                or (h_supported & ~eligible).any() or not np.array_equal(selected.H_lst_c.to_numpy()[~h_supported],selected.G_lst_c.to_numpy()[~h_supported])):
            raise ValueError("Saved H correction violates exact G fallback, support or its total adjustment limit.")
        h_audit={"eligible_rows":int(eligible.sum()),"supported_rows":int(h_supported.sum()),"fallback_rows_equal_G":int((~h_supported).sum()),
            "ordinary_mean_abs_change_from_G_c":float(np.mean(np.abs(selected.H_lst_c-selected.G_lst_c))),
            "max_abs_total_adjustment_c":float(np.abs(total).max()),
            "by_product":selected.loc[eligible,"coarse_product"].value_counts().to_dict() if eligible.any() else {},
            "context_covered_pilot_dates":len(selected.loc[eligible,["region_id","utc_day"]].drop_duplicates())}
    return {"label": label, "context_correction":h_audit, "status": "sparse_one_pilot_date" if pilot_dates < 2 else "observed_exploratory",
        "date_count_kind": "UTC dates" if selected.region_id.nunique() == 1 else "pilot-dates",
        "period_utc": [selected.datetime_utc.min().isoformat(), selected.datetime_utc.max().isoformat()],
        "metrics": output, "raw_correction": {"supported_rows": int(supported.sum()),
            "unsupported_rows": int((~supported).sum()), "ordinary_mean_abs_correction_c": float(np.mean(np.abs(correction))),
            "max_abs_correction_c": float(np.max(np.abs(correction))),
            "ordinary_fraction_at_3c_limit": float(np.mean(np.abs(correction) >= 3-1e-12))}}


def figure_footer(fig, handles, labels, note, *, ncol=2):
    """Reserve physical space for legend and note, independent of chart height."""
    height = fig.get_figheight()
    text_height = .16 * len(note.splitlines())
    legend_height = .20 * int(np.ceil(len(labels) / ncol))
    legend_center = .08 + text_height + .12 + legend_height / 2
    reserved = .08 + text_height + .12 + legend_height + .22
    fig.get_layout_engine().set(rect=(0, reserved / height, 1, 1 - reserved / height))
    fig.legend(handles, labels, loc="center", bbox_to_anchor=(.5, legend_center / height),
               ncol=ncol, frameon=False, fontsize=9)
    fig.text(.5, .08 / height, note, ha="center", va="bottom", fontsize=8.5)


def comparison_figure(groups, title, path, *, tails=False):
    groups = [g for g in groups if g["metrics"]]
    if not groups: return None
    if len(groups) > 5: raise ValueError("Paginate comparison charts at five groups for readable labels.")
    fields = (["fraction_abs_error_gt_5c", "raw_fraction_abs_error_gt_5c", "fraction_abs_error_gt_7c", "raw_fraction_abs_error_gt_7c"]
              if tails else ["mae_c", "unweighted_pixel_mae_c"])
    titles = (["Weighted errors >5°C (%)", "Ordinary pixel errors >5°C (%)", "Weighted errors >7°C (%)", "Ordinary pixel errors >7°C (%)"]
              if tails else ["Date-balanced MAE (°C)", "Ordinary pixel MAE (°C)"])
    fig, axes = plt.subplots(2 if tails else 1, 2, figsize=(12, (2 if tails else 1)*(1.08*len(groups)+1.5)+1.2),
                             squeeze=False, layout="constrained")
    models=[m for m in (*MODELS,"H") if m in groups[0]["metrics"]]
    pos = np.arange(len(groups))
    labels = []
    for group in groups:
        m = group["metrics"]["E"]
        noun = group["date_count_kind"] if m["date_count"] != 1 else group["date_count_kind"].removesuffix("s")
        sparse = " · sparse" if m["date_count"] < 2 else ""
        labels.append(f"{group['label']}\n{m['date_count']} {noun} · {m['n']:,} cells{sparse}")
    for index, (ax, field, xlabel) in enumerate(zip(axes.flat, fields, titles)):
        for i, model in enumerate(models):
            values = [group["metrics"][model][field]*(100 if tails else 1) for group in groups]
            bars = ax.barh(pos+(i-(len(models)-1)/2)*.18, values, height=.16, color=COLORS[model], label=LABELS[model])
            ax.bar_label(bars,labels=[f"{v:.2f}" for v in values],padding=3,fontsize=8)
        ax.set_yticks(pos, labels if index%2 == 0 else [""]*len(groups)); ax.invert_yaxis()
        ax.set_xlabel(xlabel); ax.set_axisbelow(True); ax.grid(axis="x",color="#e8edf0")
        ax.spines[["top","right","left"]].set_visible(False); ax.tick_params(axis="y",length=0,labelsize=9)
        ax.set_xlim(0,max(.5,ax.get_xlim()[1]*1.17))
    fig.suptitle(title,fontsize=15,weight="bold")
    figure_footer(fig, *axes[0,0].get_legend_handles_labels(),
                  "Same sampled cells within each group. Clear-source observations; sparse dates do not establish broad accuracy.\n"
                  "Weighted rates balance pilot/phase, date, acquisition and surface; ordinary rates give each pixel equal weight.\n"
                  "H uses no new 2024 context; its 2024 point predictions equal G by design.")
    fig.savefig(path,dpi=160,facecolor="white"); plt.close(fig)
    return {"file":path.name,"sha256":sha(path)}


def coverage_figure(coverage, path):
    fig, axes = plt.subplots(1,2,figsize=(11,5),layout="constrained")
    largest=max((r["F_G"]["pilot_dates"] for r in coverage["by_source_phase"]),default=0)
    for ax, phase in zip(axes,("day","night")):
        groups=[r for r in coverage["by_source_phase"] if r["phase"]==phase]
        position=np.arange(len(groups))
        for offset, model, label, color in ((-.18,"E","E · earlier fitting",COLORS["E"]),(.18,"F_G","F/G/H · expanded fitting",COLORS["F"])):
            bars=ax.barh(position+offset,[r[model]["pilot_dates"] for r in groups],height=.3,color=color,label=label)
            ax.bar_label(bars,padding=3,fontsize=9)
        ax.set_yticks(position,[SOURCES.get(r["label_product"],r["label_product"]) for r in groups])
        ax.invert_yaxis(); ax.set_title(phase.title()+" fitting dates",weight="bold")
        ax.set_xlabel("Pilot-dates, counted separately for each source");ax.set_xlim(0,max(1,largest*1.18))
        ax.set_axisbelow(True);ax.grid(axis="x",color="#e8edf0");ax.spines[["top","right","left"]].set_visible(False)
    fig.suptitle("Fine-resolution observations used for fitting",fontsize=15,weight="bold")
    figure_footer(fig, *axes[0].get_legend_handles_labels(),
                  "One pilot-date is one area/UTC-day pair; the same day can appear under several sensors.\n"
                  "All E rows are retained. F, G and enabled H use the same fitting observations. Coarse context is not a 100 m label.")
    fig.savefig(path,dpi=160,facecolor="white");plt.close(fig)
    return {"file":path.name,"sha256":sha(path)}


def correction_figure(result, correction, path):
    entries=[]
    for split, group in result["old_evaluation"].items():
        if group.get("correction_usage"):
            entries.append((split.replace("_"," ").title(),group["correction_usage"]))
    if not entries:return None
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout="constrained")
    position=np.arange(len(entries));labels=[x[0] for x in entries]
    for ax,field,scale,title in ((axes[0],"weighted_mean_abs_adjustment_c",1,"Weighted mean absolute adjustment (°C)"),
                                (axes[1],"weighted_supported_fraction",100,"Weighted rows with supported correction (%)")):
        values=[r[field]*scale for _,r in entries]
        bars=ax.barh(position,values,color=COLORS["G"],height=.5);ax.bar_label(bars,fmt="%.2f",padding=4,fontsize=9)
        ax.set_yticks(position,labels if ax is axes[0] else [""]*len(labels));ax.invert_yaxis();ax.set_xlabel(title)
        ax.set_axisbelow(True);ax.grid(axis="x",color="#e8edf0");ax.spines[["top","right","left"]].set_visible(False)
        ax.set_xlim(0,3.3 if ax is axes[0] else 112)
    fig.suptitle("What the residual correction actually changed",fontsize=15,weight="bold")
    fig.supxlabel(f"Regularization uses {correction['effective_global_utc_dates']:.1f} effective UTC dates, not the number of pixels.\n"
                  "Adjustment is capped at ±3°C; prediction errors may be larger. Unsupported climate/phase groups retain F.",fontsize=9)
    fig.savefig(path,dpi=160,facecolor="white");plt.close(fig)
    return {"file":path.name,"sha256":sha(path)}



def context_figure(result,path):
    entries=[(split.replace("_"," ").title(),r["context_correction_usage"])
             for split,r in result["old_evaluation"].items() if r.get("context_correction_usage")]
    if not entries: return None
    fig,axes=plt.subplots(1,2,figsize=(12,5.4),layout="constrained");position=np.arange(len(entries))
    for shift,field,label,color in ((-.15,"weighted_eligible_fraction","Eligible native context",COLORS["F"]),
                                    (.15,"weighted_supported_fraction","Context and fitting support",COLORS["H"])):
        bars=axes[0].barh(position+shift,[100*r[field] for _,r in entries],height=.25,label=label,color=color)
        axes[0].bar_label(bars,fmt="%.2f",padding=3,fontsize=8)
    bars=axes[1].barh(position,[r["weighted_mean_abs_change_from_G_c"] for _,r in entries],height=.45,color=COLORS["H"])
    axes[1].bar_label(bars,fmt="%.3f",padding=3,fontsize=8)
    for index,ax in enumerate(axes):
        ax.set_yticks(position,[label for label,_ in entries] if index==0 else [""]*len(entries));ax.invert_yaxis()
        ax.spines[["top","right","left"]].set_visible(False);ax.grid(axis="x",color="#e8edf0");ax.set_axisbelow(True)
    axes[0].set_xlabel("Date-balanced fraction of evaluated rows (%)");axes[0].set_xlim(0,112)
    axes[1].set_xlabel("Weighted mean absolute H − G change (°C)");axes[1].set_xlim(0,max(.05,axes[1].get_xlim()[1]*1.25))
    fig.suptitle("Native context coverage and its actual effect",fontsize=15,weight="bold")
    figure_footer(fig, *axes[0].get_legend_handles_labels(),
                  "All original evaluation rows are retained. Missing or unsupported context uses G exactly.\n"
                  "Repeated fine cells can share one native observation; coverage does not prove improved accuracy.")
    fig.savefig(path,dpi=160,facecolor="white");plt.close(fig)
    return {"file":path.name,"sha256":sha(path)}


def build_groups(result, old_frame, fresh_frame, legacy=None, legacy_frame=None):
    sections={}
    old=lambda split:old_frame.loc[old_frame.split.eq(split)]
    sections["fixed_2023"]=[matched_group(result["old_evaluation"][split],old(split),label)
        for split,label in (("development","Earlier 2023 H1 development"),("calibration","Earlier 2023 H2 calibration"))]
    sections["nighttime"]=[matched_group(result["old_evaluation"][split],old(split),f"{label} night · {half}","by_region_phase",f"{region}|night")
        for split,half in (("development","2023 H1"),("calibration","2023 H2"))
        for region,label in (("greater_london","London"),("sioux_falls","Sioux Falls"))]
    sections["reserved_places"]=[matched_group(result["old_evaluation"][split],old(split),label,"by_region_phase",key)
        for split,label,key in [("heldout_region","Cabauw day · withheld pilot","cabauw|day"),
            ("heldout_spatial","London day · reserved blocks","greater_london|day"),
            ("heldout_spatial","London night · reserved blocks","greater_london|night"),
            ("heldout_spatial","Sioux night · reserved blocks","sioux_falls|night")]]
    for split,half in (("development","2023 H1"),("calibration","2023 H2")):
        sections[f"thermal_regimes_{split}"]=[matched_group(result["old_evaluation"][split],old(split),
            f"{label} {phase} · {regime} air · {half}","by_region_phase_air",f"{region}|{phase}|{regime}")
            for region,label in (("greater_london","London"),("sioux_falls","Sioux Falls"))
            for phase in ("day","night") for regime in ("cold","hot")]
    for fresh_name,flag in (("fresh",True),("repeated",False)):
        for split in ("development","calibration","heldout_region","heldout_spatial"):
            frame=fresh_frame.loc[fresh_frame.fresh_2023.eq(flag) & fresh_frame.split.eq(split)]
            entry=result["new_2023_evaluation"][fresh_name][split]
            sources=sorted(entry["metrics"]["E"].get("by_label_product",{}))
            sections[f"new2023_{fresh_name}_{split}"]=[matched_group(entry,frame,
                f"{SOURCES.get(source,source)} · {fresh_name} 2023 · {split.replace('_',' ')}","by_label_product",source) for source in sources]
            if not sources:
                sections[f"new2023_{fresh_name}_{split}"]=[matched_group(entry,frame,
                    f"{fresh_name.title()} 2023 fine-source observations · {split.replace('_',' ')}")]
    # Region/phase checks are reported separately from the pooled fresh-source
    # result: a low pooled weighted error can conceal a local regression.
    for split,half in (("development","2023 H1"),("calibration","2023 H2")):
        frame=fresh_frame.loc[fresh_frame.fresh_2023 & fresh_frame.split.eq(split)]
        entry=result["new_2023_evaluation"]["fresh"][split]
        sections[f"fresh2023_{split}_regional"]=[matched_group(entry,frame,
            f"{label} {phase} · fresh {half}","by_region_phase",f"{region}|{phase}")
            for region,label in (("greater_london","London"),("sioux_falls","Sioux Falls")) for phase in ("day","night")]
        sections[f"fresh2023_{split}_thermal_regimes"]=[matched_group(entry,frame,
            f"{label} {phase} · {regime} air · fresh {half}","by_region_phase_air",f"{region}|{phase}|{regime}")
            for region,label in (("greater_london","London"),("sioux_falls","Sioux Falls"))
            for phase in ("day","night") for regime in ("cold","hot")]
    sections["earlier_daytime"]=[matched_group(result["old_evaluation"][split],old(split),
        f"{label} day · {half}","by_region_phase",f"{region}|day")
        for split,half in (("development","2023 H1"),("calibration","2023 H2"))
        for region,label in (("greater_london","London"),("sioux_falls","Sioux Falls"))]
    if "context_correction_support" in result:
        sections["context_supported_old"]=[]
        for split in ("development","calibration","heldout_region","heldout_spatial"):
            entry=result["old_evaluation"][split];usage=entry.get("context_correction_usage",{})
            if usage.get("supported_rows"):
                frame=old(split);selected=frame.loc[metrics.strict_bool(frame.H_correction_supported,"H_correction_supported")]
                sections["context_supported_old"].append(matched_group({"metrics":usage["supported_subset_metrics"]},selected,
                    "H-supported cells · "+split.replace("_"," ")))
        sections["context_supported_fresh2023"]=[]
        for split in ("development","calibration","heldout_region","heldout_spatial"):
            entry=result["new_2023_evaluation"]["fresh"][split];usage=entry.get("context_correction_usage",{})
            if usage.get("supported_rows"):
                frame=fresh_frame.loc[fresh_frame.fresh_2023 & fresh_frame.split.eq(split)]
                selected=frame.loc[metrics.strict_bool(frame.H_correction_supported,"H_correction_supported")]
                sections["context_supported_fresh2023"].append(matched_group({"metrics":usage["supported_subset_metrics"]},selected,
                    "Fresh H-supported cells · "+split.replace("_"," ")))
    if legacy is not None:
        sections["repeated_2024"]=[matched_group(legacy,legacy_frame,"All earlier 2024 observations"),
            matched_group(legacy,legacy_frame,"London · earlier 2024","by_region","greater_london"),
            matched_group(legacy,legacy_frame,"Sioux Falls · earlier 2024","by_region","sioux_falls")]
    return sections


def write_report(summary, destination):
    lines=["# Multisensor calibration comparisons","",
        "E is the frozen prior experiment. F adds admitted fine-resolution satellite labels using the same 40 predictors and model capacity. G adds the fixed, regularized out-of-fold residual correction. Optional H adds prior native satellite context with exact G fallback when absent or unsupported. These artifacts do not select or promote a model.","",
        "Balanced MAE and weighted tails use the frozen evaluation weights. Ordinary pixel errors give every sampled pixel equal weight; their ranking can differ. Bias is prediction minus observation. Aggregate dates are pilot-date pairs, not necessarily distinct global dates. Clear-source samples do not establish all-weather coverage. Cold air means ≤0°C; hot air means ≥30°C. A single-date check is explicitly sparse.","",
        "## Fitting coverage","","| Source | Phase | E pilot-dates | F/G/H pilot-dates | E cells | F/G/H cells |","|---|---|---:|---:|---:|---:|"]
    for row in summary["coverage"]["by_source_phase"]:
        e,f=row["E"],row["F_G"]
        lines.append(f"| {SOURCES.get(row['label_product'],row['label_product'])} | {row['phase']} | {e['pilot_dates']} | {f['pilot_dates']} | {e['rows']:,} | {f['rows']:,} |")
    lines += ["","Source date counts can overlap; F, G and enabled H share exactly the same fitting rows.",""]
    for section,groups in summary["groups"].items():
        if not groups:continue
        if not any(group["metrics"] for group in groups):
            lines += [f"## {section.replace('_',' ').title()}","","No supporting observations in these groups; no chart or accuracy estimate is generated.",""]
            continue
        lines += [f"## {section.replace('_',' ').title()}","",
            "| Group | Model | Dates / cells | Balanced MAE °C | Pixel MAE °C | Bias °C | Weighted >5°C | Pixel >5°C | Weighted >7°C | Pixel >7°C | Contrast MAE °C |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for group in groups:
            if not group["metrics"]:
                lines.append(f"| {group['label']} | No supporting observations | — | — | — | — | — | — | — | — | — |")
                continue
            for model,row in group["metrics"].items():
                lines.append(f"| {group['label']} | {model} | {row['date_count']} / {row['n']:,} | {row['mae_c']:.3f} | {row['unweighted_pixel_mae_c']:.3f} | {row['bias_c']:+.3f} | "
                    f"{row['fraction_abs_error_gt_5c']*100:.2f}% | {row['raw_fraction_abs_error_gt_5c']*100:.2f}% | {row['fraction_abs_error_gt_7c']*100:.2f}% | {row['raw_fraction_abs_error_gt_7c']*100:.2f}% | {row['centered_contrast_mae_c']:.3f} |")
        lines += [""]
    lines += ["## Correction support","",
        "The adjustment can be at most ±3°C. This is not an error guarantee or instrument calibration. G retains F where fitting support is insufficient.","",
        "| Broad climate / phase | Distinct UTC dates | Year-months | Folds | Correction supported |","|---|---:|---:|---:|---|"]
    for key,row in summary["correction"]["support"].items():
        lines.append(f"| {key.replace('|',' / ')} | {row['utc_dates']} | {row['year_months']} | {row['folds']} | {row['supported']} |")
    if summary.get("h_correction"):
        lines += ["","## H native context support","",
            "H replaces G only where recent native context and independent fitting-date support both pass. Missing or unsupported context keeps G exactly. No 2024 native context was collected, so H equals G there; this is not an independent test of H context.","",
            "| Broad climate / phase | Context-covered UTC dates | Year-months | Folds | H supported |","|---|---:|---:|---:|---|"]
        for key,row in summary["h_correction"]["support"].items():
            lines.append(f"| {key.replace('|',' / ')} | {row['utc_dates']} | {row['year_months']} | {row['folds']} | {row['supported']} |")
        lines += ["","| Matched evaluation group | Context cells | Supported H cells | Exact G fallback cells | Context pilot-dates | Mean absolute change from G °C |","|---|---:|---:|---:|---:|---:|"]
        for groups in summary["groups"].values():
            for group in groups:
                h=group.get("context_correction")
                if h: lines.append(f"| {group['label']} | {h['eligible_rows']} | {h['supported_rows']} | {h['fallback_rows_equal_G']} | {h['context_covered_pilot_dates']} | {h['ordinary_mean_abs_change_from_G_c']:.3f} |")
    lines += ["","## Figures",""]
    for item in summary["figures"]:lines += [f"![{item['file']}]({item['file']})",""]
    destination.write_text("\n".join(lines)+"\n")


def render(run, output, legacy_evaluation=None):
    run,output=Path(run),Path(output)
    if output.exists():raise FileExistsError("Use a new figure folder; frozen artifacts and earlier figures are preserved.")
    manifest=train.verify_run(run)
    result=json.loads((run/"results.json").read_text())
    if result.get("status") not in ("multisensor_fg_frozen_evaluated_no_selection_no_promotion","multisensor_fgh_frozen_evaluated_no_selection_no_promotion"):
        raise ValueError("Completed real F/G experiment outputs are required; placeholders are not rendered.")
    post=json.loads((run/"post_evaluation_freeze.json").read_text())
    fit=json.loads((run/"fit_freeze.json").read_text())
    old_frame=load_predictions(run/"old_evaluation_predictions.parquet",post["artifacts"]["old_evaluation_predictions.parquet"],[2021,2022,2023])
    fresh_frame=load_predictions(run/"new_2023_predictions.parquet",post["artifacts"]["new_2023_predictions.parquet"],[2023])
    fresh_frame["fresh_2023"]=metrics.strict_bool(fresh_frame.fresh_2023,"fresh_2023")
    counts=fitting_coverage(pd.read_parquet(run/"fitting_rows.parquet"),manifest)
    correction=json.loads((run/"correction.json").read_text())
    sources={str((run/name).resolve()):sha(run/name) for name in ("results.json","manifest.json","fit_freeze.json","post_evaluation_freeze.json",
        "fitting_rows.parquet","old_evaluation_predictions.parquet","new_2023_predictions.parquet","correction.json")}
    legacy=legacy_frame=None
    if legacy_evaluation is not None:
        legacy_dir=Path(legacy_evaluation)
        legacy=json.loads((legacy_dir/"results.json").read_text())
        if (legacy.get("status")!="multisensor_2024_evaluated_no_refit_no_selection_no_promotion" or
                legacy.get("multisensor_freeze_sha256")!=sha(run/"post_evaluation_freeze.json")):
            raise ValueError("2024 results are not bound to this completed frozen experiment.")
        legacy_frame=load_predictions(legacy_dir/"predictions.parquet",legacy["predictions_sha256"],[2024])
        for name in ("results.json","predictions.parquet"):sources[str((legacy_dir/name).resolve())]=sha(legacy_dir/name)
    groups=build_groups(result,old_frame,fresh_frame,legacy,legacy_frame)
    h_correction=json.loads((run/"h_correction.json").read_text()) if manifest.get("h_enabled") else None
    if h_correction: sources[str((run/"h_correction.json").resolve())]=sha(run/"h_correction.json")
    # Every identity/numeric integrity check above completes before any figures are created.
    output.mkdir(parents=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9})
    figures=[coverage_figure(counts,output/"fitting_coverage.png")]
    item=correction_figure(result,correction,output/"correction.png")
    if item:figures.append(item)
    if h_correction:
        item=context_figure(result,output/"native_context.png")
        if item: figures.append(item)
    for key,entries in groups.items():
        observed=[entry for entry in entries if entry["metrics"]]
        for start in range(0,len(observed),5):
            page=observed[start:start+5]
            suffix=f"_{start//5+1}" if len(observed)>5 else ""
            for tails in (False,True):
                file=output/f"{key}{suffix}_{'tails' if tails else 'mae'}.png"
                title_prefix = {
                    "fresh2023_development_regional": "Fresh 2023: first-half regional comparisons",
                    "fresh2023_calibration_regional": "Fresh 2023: second-half regional comparisons",
                    "fresh2023_development_thermal_regimes": "Fresh 2023: first-half hot and cold conditions",
                    "fresh2023_calibration_thermal_regimes": "Fresh 2023: second-half hot and cold conditions",
                }.get(key, key.replace("_", " ").title())
                title=title_prefix+(" · large errors" if tails else " · average errors")
                figures.append(comparison_figure(page,title,file,tails=tails))
    summary={"version":"multisensor-plots-v1","models":[*MODELS,*(["H"] if h_correction else [])],"source_artifacts":sources,
        "plot_source_sha256":sha(__file__),"coverage":counts,"groups":groups,"correction":correction,
        "h_correction":h_correction,"context_audits":result.get("context_audits"),"figures":figures,"production_changed":False,"models_fitted":False,"selection_performed":False,
        "satellite_assets_opened":False,"blind_2025_opened":False,"saved_evaluation_predictions_read":True,
        "weighted_metrics_recomputed":False,"ordinary_pixel_tails_computed_from_same_saved_rows":True}
    (output/"SUMMARY.json").write_text(json.dumps(metrics.json_ready(summary),indent=2,allow_nan=False)+"\n")
    write_report(summary,output/"SUMMARY.md")
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--legacy-evaluation",type=Path)
    args=parser.parse_args();result=render(args.run,args.output,args.legacy_evaluation)
    print(json.dumps({"figures":len(result["figures"]),"output":str(args.output)}))


if __name__=="__main__":main()
