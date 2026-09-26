"""Research orchestration only: two pilot batches, four unchanged workers each."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
from pathlib import Path
import time

import pandas as pd

from . import more_days_pair as pair


def run(experiment, root, legacy, cache):
    root, experiment, legacy, cache = map(lambda p: Path(p).resolve(), [root, experiment, legacy, cache])
    experiment.mkdir(parents=True, exist_ok=True)
    with (experiment/"pairing_coordinator.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = experiment/"acquisition/plan.json"
        plan = json.loads(plan_path.read_text())
        prior = Path(plan["old_cohort"])
        if pair.f._sha(prior) != plan["old_cohort_sha256"]:
            raise ValueError("Prior cohort identity hash changed.")
        old = pd.read_parquet(prior, columns=["region_id", "datetime_utc", "sample_id", "acquisition_id"])
        ids = [g["region"]["id"] for g in plan["groups"]]
        complete, pending = {}, {}
        start = time.monotonic()
        areas_path = root/"pilot/areas_resolved.json"
        def process(path):
            completion = json.loads(path.read_text())
            lock = experiment/"pairing"/completion["region_id"]/"orchestration.lock"
            lock.parent.mkdir(parents=True, exist_ok=True)
            with lock.open("a") as stream:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return pair.process_batch(completion, experiment, root, areas_path, old, legacy, cache)
        with ThreadPoolExecutor(max_workers=2) as pool:
            while len(complete) < len(ids):
                queued = set(pending.values())
                for rid in ids:
                    if len(pending) >= 2:
                        break
                    source = experiment/"acquisition"/rid/"completion.json"
                    if rid not in complete and rid not in queued and source.exists():
                        pending[pool.submit(process, source)] = rid
                if not pending:
                    if time.monotonic()-start > 8*3600:
                        raise TimeoutError("Waiting for acquisition completion exceeded eight hours; checkpoints retained.")
                    time.sleep(30)
                    continue
                done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                for future in done:
                    rid = pending.pop(future)
                    complete[rid] = future.result()
        paths = [Path(complete[rid]["fit_path"]) for rid in ids if complete[rid].get("fit_path")]
        data = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
        if len(data):
            areas = {a["id"]: a for a in json.loads(areas_path.read_text())["areas"]}
            pair.verify_new_identity(data, areas, old)
        output = experiment/"paired_new_fit_only.parquet"
        data.to_parquet(output, index=False)
        report = {"version": pair.VERSION, "source_sha256": pair.f._sha(pair.__file__),
                  "resource_orchestrator_sha256": pair.f._sha(__file__),
                  "resource_addendum_sha256": pair.f._sha(root/"reports/more_days/PAIRING_RESOURCE_ADDENDUM.md"),
                  "plan_sha256": pair.f._sha(plan_path), "original_cohort_identity_source_sha256": pair.f._sha(prior),
                  "legacy_admission_reference_path": str(legacy), "legacy_admission_reference_sha256": pair.f._sha(legacy),
                  "fit_path": str(output), "fit_sha256": pair.f._sha(output), "fit_rows": len(data),
                  "fit_pilot_dates": sum(r.get("fit_dates", 0) for r in complete.values()),
                  "per_pilot": [complete[rid] for rid in ids], "workers_global_max": 8, "concurrent_pilots": 2,
                  "workers_per_pilot": 4, "input_years": [2021, 2022], "old_rows_combined": False, "model_fitted": False,
                  "fit_policy": "Existing source admission; recomputed spatial flags; no reserved/buffer cells; complete frozen base40; exact audited actual station; solar elevation >=10 degrees; independent new pilot/date and sample/acquisition IDs."}
        pair.f._write_json(experiment/"pairing_manifest.json", report)
        print(json.dumps({"completed": True, "fit_path": str(output), "fit_sha256": pair.f._sha(output), "rows": len(data)}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True)
    p.add_argument("--root", default=".")
    p.add_argument("--legacy", required=True)
    p.add_argument("--cache", default="cache")
    a = p.parse_args()
    run(a.experiment, a.root, a.legacy, a.cache)
