"""Compact matched-comparison figures from completed research JSON only."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS=("E","A","v1","air_only")
LABELS={"E":"E: more dates","A":"A: prior experiment","v1":"v1: original model","air_only":"Air alone"}
COLORS={"E":"#217663","A":"#438397","v1":"#71849a","air_only":"#b9c0c7"}
FIELDS=("n","date_count","utc_date_count","acquisition_count","mae_c","unweighted_pixel_mae_c","bias_c",
        "fraction_abs_error_gt_3c","fraction_abs_error_gt_5c","fraction_abs_error_gt_7c","centered_contrast_mae_c",
        "sample_id_sha256","status")


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def coverage(manifest):
    before={(r['region_id'],r['phase']):r for r in manifest['original_A_fit_manifest']['support']}
    after={(r['region_id'],r['phase']):r for r in manifest['fit_support']}
    result=[]
    for region in sorted({key[0] for key in before}|{key[0] for key in after}):
        for phase in ('day','night'):
            a=before.get((region,phase),{});e=after.get((region,phase),{})
            row={'region_id':region,'phase':phase,'A_dates':a.get('dates',0),'E_dates':e.get('dates',0),
                 'A_rows':a.get('rows',0),'E_rows':e.get('rows',0)}
            if row['E_dates']<row['A_dates'] or row['E_rows']<row['A_rows']:
                raise ValueError('Coverage does not preserve the old fitting observations.')
            if phase=='night' and (row['A_dates'],row['A_rows'])!=(row['E_dates'],row['E_rows']):
                raise ValueError('A more-days coverage chart cannot claim unchanged night observations.')
            result.append(row)
    return result


def coverage_figure(rows,destination):
    fig,axes=plt.subplots(1,2,figsize=(12,6.3),layout='constrained')
    for ax,phase in zip(axes,('day','night')):
        selected=[r for r in rows if r['phase']==phase and r['E_dates']>0]
        centre=np.arange(len(selected));labels=[r['region_id'].replace('_',' ').title() for r in selected]
        for offset,key,label,color in [(-.18,'A_dates','A: prior experiment',COLORS['A']),(.18,'E_dates','E: more dates',COLORS['E'])]:
            bars=ax.barh(centre+offset,[r[key] for r in selected],height=.31,color=color,label=label)
            ax.bar_label(bars,padding=3,fontsize=9)
        ax.set_yticks(centre,labels);ax.invert_yaxis()
        ax.set_title('Day fitting dates' if phase=='day' else 'Night fitting dates\nNo new nighttime observations',weight='bold')
        ax.set_xlabel('Distinct UTC dates per pilot');ax.set_axisbelow(True);ax.grid(axis='x',color='#e8edf0')
        ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
        maximum=max(r['E_dates'] for r in rows);ax.set_xlim(0,max(1,maximum*1.14))
    axes[0].legend(loc='lower right',fontsize=9,frameon=False)
    fig.suptitle('What more Landsat dates added to fitting',fontsize=15,weight='bold')
    fig.supxlabel('Counts use the admitted fitting table, after station, quality, feature and spatial checks.\n'
                  'Original fitting observations are preserved; evaluation dates and rows remain fixed.\n'
                  'Fitting pilots absent from the night panel have no nighttime training observations.',fontsize=9)
    fig.savefig(destination,dpi=165,facecolor='white');plt.close(fig)
    return {'file':destination.name,'sha256':digest(destination)}


def group(scores, label, region_phase=None):
    metrics={name:(scores[name]["overall"] if region_phase is None else
                   scores[name].get("by_region_phase",{}).get(region_phase)) for name in MODELS}
    if any(v is None or v.get("n",0)==0 for v in metrics.values()):
        return {"label":label,"status":"unsupported_no_matched_observations","metrics":{}}
    hashes={v["sample_id_sha256"] for v in metrics.values()}
    if len(hashes)!=1:raise ValueError("Cannot compare models on different observation rows.")
    return {"label":label,"status":"observed_sparse" if metrics['E']['date_count']<2 else "observed",
            "date_count_kind":"pilot-dates" if region_phase is None else "dates",
            "metrics":{name:{k:v[k] for k in FIELDS if k in v} for name,v in metrics.items()}}


def figure(groups, title, destination):
    observed=[g for g in groups if g['metrics']]
    if not observed:return None
    fig, axes=plt.subplots(1,2,figsize=(12,1.2*len(observed)+2.1),layout='constrained')
    centre=np.arange(len(observed)); width=.18
    labels=[]
    for g in observed:
        count=g['metrics']['E']['date_count'];noun=g['date_count_kind']
        if count==1:noun=noun.removesuffix('s')
        labels.append(f"{g['label']}\n{count} {noun} · {g['metrics']['E']['n']:,} cells")
    for ax,key,scale,xlabel in zip(axes,['mae_c','fraction_abs_error_gt_5c'],[1,100],
                                  ['Balanced mean absolute error (°C)','Weighted errors larger than 5°C (%)']):
        for i,name in enumerate(MODELS):
            values=[g['metrics'][name][key]*scale for g in observed]
            bars=ax.barh(centre+(i-1.5)*width,values,height=width*.86,color=COLORS[name],label=LABELS[name])
            ax.bar_label(bars,labels=[f'{v:.2f}' for v in values],padding=3,fontsize=8)
        ax.set_yticks(centre,labels if ax is axes[0] else ['']*len(labels))
        ax.invert_yaxis();ax.set_xlabel(xlabel,fontsize=10);ax.set_axisbelow(True)
        ax.grid(axis='x',color='#e8edf0',linewidth=.7);ax.spines[['top','right','left']].set_visible(False)
        lo,hi=ax.get_xlim();ax.set_xlim(0,max(1,hi*1.14));ax.tick_params(axis='y',length=0,labelsize=9)
    fig.suptitle(title,fontsize=15,weight='bold')
    axes[0].legend(loc='upper left',bbox_to_anchor=(0,-.14),ncol=2,fontsize=9,frameon=False)
    fig.supxlabel('Same observations and fixed evaluation weights. Previously inspected, clear-sky samples; no blind confirmation.\n'
                  'Bias, >7°C errors, ordinary pixel MAE and spatial-contrast errors appear in SUMMARY.md.',fontsize=9)
    fig.savefig(destination,dpi=165,facecolor='white');plt.close(fig)
    return {'file':destination.name,'sha256':digest(destination)}


def render(run_dir,output_dir,legacy_dir=None):
    run=Path(run_dir);path=run/'results.json';result=json.loads(path.read_text())
    if result.get('status')!='more_days_fitted_no_selection_no_promotion':
        raise ValueError('Completed real more-days experiment results are required.')
    fit=json.loads((run/'fit_freeze.json').read_text());freeze=json.loads((run/'post_evaluation_freeze.json').read_text())
    if digest(path)!=freeze['results_sha256'] or digest(run/'fit_freeze.json')!=freeze['fit_freeze_sha256']:
        raise ValueError('Experiment result/freeze hashes differ.')
    if not fit['model_frozen_before_evaluation']:raise ValueError('Model freeze must precede evaluation.')
    if digest(run/'manifest.json')!=fit['manifest_sha256']:raise ValueError('Fitting coverage manifest hash differs.')
    counts=coverage(json.loads((run/'manifest.json').read_text()))
    sources={str(path.resolve()):digest(path),str((run/'manifest.json').resolve()):digest(run/'manifest.json')}
    groups={}
    groups['fixed_development']=[group(result['evaluation']['development']['metrics'],'All observed development groups'),
        group(result['evaluation']['development']['metrics'],'London day','greater_london|day'),
        group(result['evaluation']['development']['metrics'],'Sioux Falls day','sioux_falls|day')]
    groups['nighttime_checks']=[group(result['evaluation'][split]['metrics'],label,key) for split,label,key in [
        ('development','London night · development','greater_london|night'),
        ('calibration','London night · calibration','greater_london|night'),
        ('development','Sioux night · development','sioux_falls|night'),
        ('calibration','Sioux night · calibration','sioux_falls|night')]]
    groups['spatial_checks']=[group(result['evaluation']['heldout_spatial']['metrics'],'London day · reserved blocks','greater_london|day'),
        group(result['evaluation']['heldout_spatial']['metrics'],'London night · reserved blocks','greater_london|night'),
        group(result['evaluation']['heldout_region']['metrics'],'Cabauw · withheld region','cabauw|day')]
    if legacy_dir is not None:
        lp=Path(legacy_dir)/'results.json';legacy=json.loads(lp.read_text())
        if legacy.get('status')!='more_days_2024_evaluated_no_refit_no_promotion':
            raise ValueError('Completed frozen 2024 evaluation is required.')
        if legacy['more_days_freeze_sha256']!=digest(run/'post_evaluation_freeze.json'):
            raise ValueError('2024 evaluation references a different frozen experiment.')
        sources[str(lp.resolve())]=digest(lp)
        groups['legacy_2024']=[group(legacy['metrics'],'All 2024 observations'),
            group(legacy['metrics'],'London · 2024','greater_london|day'),
            group(legacy['metrics'],'Sioux Falls · 2024','sioux_falls|day')]
    output=Path(output_dir)
    if output.exists():raise FileExistsError('Use a new figures directory to preserve prior artifacts.')
    output.mkdir(parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    titles={'fixed_development':'More Landsat dates: the fixed development comparison',
            'nighttime_checks':'Nighttime checks: training nights stayed unchanged',
            'spatial_checks':'The same reserved places after adding training dates',
            'legacy_2024':'The same 2024 daytime observations'}
    figures=[coverage_figure(counts,output/'fitting_dates.png')]
    figures += [item for key,values in groups.items() if (item:=figure(values,titles[key],output/f'{key}.png'))]
    summary={'candidate':'E','production_changed':False,'source_results':sources,'groups':groups,'figures':figures,
             'fitting_coverage':counts,'raw_thermal_arrays_read':False,'metrics_fabricated':False,'blind_2025_opened':False}
    (output/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# More dates: matched model comparisons','',
        'E adds fitting dates while keeping the 40 predictors, model capacity and evaluation observations fixed. '
        'These are repeated exploratory comparisons. No model is selected or promoted by these charts.','',
        'Balanced errors give equal weight to observed pilot/phase groups, then dates, acquisitions and surface classes. '
        'Ordinary pixel MAE is included because the two weightings can favor different models. '
        'Bias is prediction minus observation. Counts describe sampled observations, not whole-city coverage. '
        'Aggregate date counts are pilot-dates (each area/date pair); individual-pilot counts are UTC dates.','']
    lines += ['## Fitting date coverage','','| Pilot | Phase | A dates | E dates | A cells | E cells |',
              '|---|---|---:|---:|---:|---:|']
    for row in counts:
        lines += [f"| {row['region_id']} | {row['phase']} | {row['A_dates']} | {row['E_dates']} | {row['A_rows']:,} | {row['E_rows']:,} |"]
    lines += ['','![Fitting date coverage](fitting_dates.png)','']
    for key,values in groups.items():
        lines += [f'## {titles[key]}','',
            '| Group | Model | Dates / cells | Balanced MAE °C | Pixel MAE °C | Bias °C | Weighted >5°C | Weighted >7°C | Contrast MAE °C |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for value in values:
            if not value['metrics']:
                lines += [f"| {value['label']} | Unsupported: no matched observations | — | — | — | — | — | — | — |"]
                continue
            for name,z in value['metrics'].items():
                lines += [f"| {value['label']} | {LABELS[name]} | {z['date_count']} / {z['n']:,} | {z['mae_c']:.2f} | "
                    f"{z['unweighted_pixel_mae_c']:.2f} | {z['bias_c']:+.2f} | {100*z['fraction_abs_error_gt_5c']:.2f}% | "
                    f"{100*z['fraction_abs_error_gt_7c']:.2f}% | {z['centered_contrast_mae_c']:.2f} |"]
        lines += ['',f'![{titles[key]}]({key}.png)',''] if any(v['metrics'] for v in values) else ['']
    (output/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--legacy-evaluation',type=Path)
    args=parser.parse_args();result=render(args.run,args.output,args.legacy_evaluation)
    print(json.dumps({'figures':len(result['figures']),'output':str(args.output)}))


if __name__=='__main__':main()
