#!/usr/bin/env bash
# Research-only fixed F/G/H launcher. This file does nothing until explicitly run.
# Arguments: new-fine-fit new-fine-evaluation freshness-registry context-spec output-dir
set -euo pipefail
if [[ $# -ne 5 ]]; then
  printf '%s\n' 'Usage: run_multisensor_fit.sh NEW_FIT NEW_EVALUATION FRESHNESS_REGISTRY CONTEXT_SPEC NEW_OUTPUT_DIR' >&2
  exit 2
fi
lst_root=/opt/lst-pilot
lst_output=$5
for lst_argument in "$@"; do
  [[ "$lst_argument" = /* ]] || { printf 'Use absolute paths: %s\n' "$lst_argument" >&2; exit 2; }
done
if [[ -e "$lst_output" || -e "${lst_output}.fit.log" || -e "${lst_output}.resources.json" ]]; then
  printf '%s\n' 'Choose a new output directory and log names; existing experiment artifacts are preserved.' >&2
  exit 2
fi
for lst_input in "$1" "$2" "$3" "$4"; do
  [[ -f "$lst_input" ]] || { printf 'Missing input: %s\n' "$lst_input" >&2; exit 2; }
done
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$lst_root"
"$lst_root/.venv/bin/python" "$lst_root/deploy/research_timer.py" "${lst_output}.resources.json" -- \
  "$lst_root/.venv/bin/python" -m lst_pilot.multisensor_train \
  --original-input "$lst_root/runs/option_b_retrain_20260909/cohort_final_v1/paired_input.parquet" \
  --e-additions "$lst_root/runs/more_days_20260909_v1/paired_new_fit_only.parquet" \
  --original-run "$lst_root/runs/option_b_retrain_20260909/model_experiment_v1" \
  --e-run "$lst_root/runs/more_days_20260909_v1/model_experiment" \
  --e-2024 "$lst_root/runs/more_days_20260909_v1/evaluation_2024" \
  --new-fit "$1" --new-evaluation "$2" --freshness-audit "$3" \
  --areas-path "$lst_root/pilot/areas_resolved.json" \
  --baseline "$lst_root/runs/pilot_v1_model/model.joblib" \
  --protocol "$lst_root/reports/multisensor/PROTOCOL_2026-09-10.md" \
  --context-spec "$4" \
  --h-protocol "$lst_root/reports/multisensor/COARSE_CONTEXT_H_ADDENDUM_2026-09-10.md" \
  --output "$lst_output" >"${lst_output}.fit.log" 2>&1
