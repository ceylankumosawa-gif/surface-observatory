# Pairing resource adjustment — 9 September 2026

The initial pairing job used one pilot batch at a time, with four numerical workers and a combined 4 CPU / 8 GiB limit. London completed in 226 seconds; most elapsed time involved public weather and raster I/O. The existing 16 CPU / approximately 30 GiB VM had sufficient free resources to overlap two batches.

Before fitting, the coordinating task authorized two independent pilot batches at a time, each retaining the existing four-worker partitions, under a combined 8 CPU / 16 GiB limit. This uses the existing VM and adds no instance or service. Numerical threads remain one per worker. Shared cache locks serialize identical objects; the new coordinator additionally locks each pilot batch and the overall run.

Handover pauses only the outer coordinator while its current feature subprocess completes. After that normal feature boundary, the original service stops and the replacement reuses all completed acquisitions in their original directories. The existing launcher verifies input, source and checkpoint hashes. Completed pilot admission files are reused with their source and output hashes checked.

The adjustment changes scheduling and resource caps only. Feature equations, four-worker shard membership, surface acquisitions, quality screening, station rules, date selection, holdout geometry and model capacity remain unchanged. The reviewed Darwin-only legacy-ISD QC1 adapter is a separately audited existing pairing policy. No model is fitted or deployed by this stage.

The final `pairing_manifest.json` records eight maximum numerical workers, two concurrent pilots, the orchestration source hash and this addendum hash. Per-pilot completion records preserve every input count, admission loss, exact station-source audit and final fitting-only file hash.
