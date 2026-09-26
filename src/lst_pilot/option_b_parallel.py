"""Bounded research launcher; four independent acquisitions, shared cache locks.

Numerical assembly stays in option_b_features. Lock wrappers are installed only
inside these worker processes; production adapters and serving are unchanged.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time

import pandas as pd

from . import option_b_features as features


@contextmanager
def cache_lock(root, namespace, identity):
    path = Path(root) / "option-b-locks" / namespace / (features._hash(identity) + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def install_cache_locks(root):
    """Serialize only identical cache keys, preserving independent I/O overlap."""
    original_tile = features._cached_tile
    original_search = features.search_optical
    original_static = features.raster._stac_tiles
    original_weather = features.weather.fetch_archive
    original_object = features.radiation._cached_object
    original_station = features.assemble.station_series

    def tile(kind, grid, cache, source_identity, reader):
        with cache_lock(root, "tiles", [str(cache), kind, grid.epsg, grid.bounds, grid.shape, source_identity]):
            return original_tile(kind, grid, cache, source_identity, reader)

    def search(area, target, cache, lookback_days=32, max_scenes=16):
        with cache_lock(root, "search", [str(cache), area, str(target), lookback_days]):
            return original_search(area, target, cache, lookback_days, max_scenes)

    def static(collection, bbox, cache, date=None):
        with cache_lock(root, "static-search", [str(cache), collection, list(bbox), date]):
            return original_static(collection, bbox, cache, date)

    def weather(latitude, longitude, start, end, cache_dir):
        with cache_lock(root, "weather", [str(cache_dir), float(latitude), float(longitude), str(start)[:10], str(end)[:10]]):
            return original_weather(latitude, longitude, start, end, cache_dir)

    def objects(fs, key, cache, limit, manifest):
        with cache_lock(root, "radiation", [str(cache), key]):
            return original_object(fs, key, cache, limit, manifest)

    def station(station_id, years, cache):
        # Overlapping year sets must lock by station, not by the requested set.
        with cache_lock(root, "station", [str(cache), station_id]):
            return original_station(station_id, years, cache)

    features._cached_tile = tile
    features.search_optical = search
    features.raster._stac_tiles = static
    features.weather.fetch_archive = weather
    features.radiation._cached_object = objects
    features.assemble.station_series = station


def _worker(spec):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    from threadpoolctl import threadpool_limits
    install_cache_locks(spec["cache"])
    with threadpool_limits(limits=1):
        _, manifest = features.build(spec["input"], spec["output"], spec["areas"], spec["cache"],
                                     preserve_existing_base=spec["preserve_existing_base"], max_optical_scenes=spec["max_optical_scenes"])
    return manifest


def run(input_path, output_dir, areas_path, cache, *, workers=4, preserve_existing_base=False,
        max_optical_scenes=16, resume_from=None):
    if not 1 <= workers <= 4:
        raise ValueError("Research launcher permits 1–4 workers.")
    started = time.monotonic()
    input_path, output, areas_path, cache = map(lambda p: Path(p).resolve(), (input_path, output_dir, areas_path, cache))
    area_map = {area["id"]: area for area in json.loads(areas_path.read_text())["areas"]}
    data = features.validate_samples(pd.read_parquet(input_path), area_map)
    groups = list(data.groupby(["region_id", "acquisition_id", "datetime_utc"], sort=True))
    workers = min(workers, len(groups))
    signature = {"input_sha256": features._sha(input_path), "areas_sha256": features._sha(areas_path),
                 "builder_sha256": features._sha(features.__file__), "launcher_sha256": features._sha(__file__),
                 "workers": workers, "preserve_existing_base": preserve_existing_base, "max_optical_scenes": max_optical_scenes}
    output.mkdir(parents=True, exist_ok=True)
    guard = output / "parallel_signature.json"
    if guard.exists() and json.loads(guard.read_text()) != signature:
        raise ValueError("Parallel output belongs to different source/code/options; select a fresh output.")
    features._write_json(guard, signature)
    if resume_from:
        resume_from = Path(resume_from)
        previous = json.loads((resume_from / "signature.json").read_text())["specification"]
        for name in ("input_sha256", "areas_sha256", "builder_sha256", "preserve_existing_base", "max_optical_scenes"):
            if previous[name] != signature[name]:
                raise ValueError(f"Prior checkpoint has incompatible {name}; no work copied.")
    assignments = [[] for _ in range(workers)]
    for number, item in enumerate(groups):
        assignments[number % workers].append(item)
    specs, reused = [], []
    for number, assignment in enumerate(assignments):
        shard = output / "shards" / str(number)
        shard.mkdir(parents=True, exist_ok=True)
        path = shard / "input.parquet"
        shard_data = pd.concat([part for _, part in assignment], ignore_index=True)
        if path.exists():
            pd.testing.assert_frame_equal(pd.read_parquet(path), shard_data, check_exact=True)
        else:
            shard_data.to_parquet(path, index=False)
        destination = shard / "features"
        if resume_from:
            for key, part in assignment:
                name = features._hash(tuple(map(str, key)))
                old = resume_from / "acquisitions" / (name + ".parquet")
                audit = old.with_suffix(".json")
                if not old.exists() or not audit.exists():
                    continue
                record = json.loads(audit.read_text())
                if features._sha(old) != record["output_sha256"]:
                    raise ValueError("Previous completed checkpoint hash mismatch.")
                retained = features.preserve_identity(part.reset_index(drop=True), pd.read_parquet(old))
                if preserve_existing_base:
                    pd.testing.assert_frame_equal(part[list(features.BASE_FEATURES)].reset_index(drop=True),
                                                  retained[list(features.BASE_FEATURES)], check_exact=True)
                target = destination / "acquisitions" / old.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copy2(old, target)
                    shutil.copy2(audit, target.with_suffix(".json"))
                reused.append({"source": str(old), "destination": str(target), "sha256": record["output_sha256"]})
        specs.append({"input": str(path), "output": str(destination), "areas": str(areas_path), "cache": str(cache),
                      "preserve_existing_base": preserve_existing_base, "max_optical_scenes": max_optical_scenes})
    features._write_json(output / "reused_checkpoints.json", reused)
    print(json.dumps({"workers": workers, "acquisitions": len(groups), "rows": len(data), "reused_checkpoints": len(reused)}), flush=True)
    manifests = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = [executor.submit(_worker, spec) for spec in specs]
        for future in as_completed(futures):
            manifests.append(future.result())
    result = features.preserve_identity(data, pd.concat([pd.read_parquet(manifest["output_path"]) for manifest in manifests], ignore_index=True))
    if preserve_existing_base:
        pd.testing.assert_frame_equal(data[list(features.BASE_FEATURES)], result[list(features.BASE_FEATURES)], check_exact=True)
    result_path = output / "features.parquet"
    result.to_parquet(result_path.with_suffix(".tmp.parquet"), index=False)
    result_path.with_suffix(".tmp.parquet").replace(result_path)
    manifest = {"input_path": str(input_path), "input_sha256": signature["input_sha256"], "output_path": str(result_path),
                "output_sha256": features._sha(result_path), "rows": len(result), "acquisitions": len(groups), "workers": workers,
                "complete_input_processed": True, "reused_checkpoints": reused, "shards": manifests,
                "elapsed_seconds": time.monotonic()-started, "research_only": True, "row_eligibility_assigned": False,
                "completeness": {name: int(result[name].sum()) for name in ("features_A_complete", "features_B_complete", "features_C_complete", "features_D_complete", "station_pair_available")}}
    features._write_json(output / "manifest.json", manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key not in ("shards", "reused_checkpoints")}, indent=2), flush=True)
    return result, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--areas", default="pilot/areas_resolved.json")
    parser.add_argument("--cache", default="cache")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--preserve-existing-base", action="store_true")
    parser.add_argument("--max-optical-scenes", type=int, default=16)
    parser.add_argument("--resume-from")
    args = parser.parse_args()
    features.satellite.configure_safe_logging()
    run(args.input, args.output_dir, args.areas, args.cache, workers=args.workers,
        preserve_existing_base=args.preserve_existing_base, max_optical_scenes=args.max_optical_scenes, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
