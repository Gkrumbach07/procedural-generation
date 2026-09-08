"""Refine stage driver (PLAN.md section 10.2): per-basin refinement of the
coarse result into the ``fine/`` per-face rasters.

Pipeline::

    create fine/ memmaps (rasterize.create_fine)
    base pass: plain upsample of every face (6 jobs)         -- fills every cell
    basin jobs, largest first (basin_job.job -> rasterize.write_result)
    quicklooks: the 3 largest basins (refine_basin<id>.png), then the planet

Workers (``refine.workers``, 0 = all cores) are a *spawned*
``concurrent.futures.ProcessPoolExecutor``: numba's OpenMP threading layer
deadlocks in a forked child once the parent has run a parallel kernel (the
upstream stages have).  Each worker loads the coarse fields once
(:func:`basin_job.pool_init`) and gets ``cpu_count / workers`` numba
threads.  With one worker everything runs in-process.  The executor (not
``multiprocessing.Pool``) is what makes a dead worker fatal: ``Pool``
silently replaces a worker killed by the OOM killer and the in-flight
task's result is then never produced, so ``imap_unordered`` blocks
forever; the executor raises ``BrokenProcessPool`` instead.

Determinism: a basin's arrays depend only on ``(seed, params, basin
record)`` (``params.rng("refine", basin_id)``; the kernel is thread-count
independent), every fine cell is written by exactly one basin job (its own
cells) after the base pass, so the raster does not depend on the job
scheduling.  ``OUTPUTS = ["fine"]`` is hashed by the manifest; nothing
time-dependent is written there.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from ..config import WorldParams
from ..io.world_store import WorldStore
from . import basin_job as bj
from . import rasterize as rz

OUTPUTS = ["fine"]

#: per-basin quicklooks for the k largest basins
N_BASIN_QUICKLOOKS = 3

#: peak resident bytes of a basin job per extended-window cell, reached
#: during the erosion iterations (kernel state ~205 B: 16-lane float32
#: samples, float64 fields and tracks; the window inputs it shares, the
#: thermal / route temporaries and the metre copies the rest); measured
#: on a 2208² window (R = 16 probe of a whole face, 1 iteration): 1.37 GB
#: above the loaded coarse inputs
BYTES_PER_WINDOW_CELL = 290


def load_basins(store: WorldStore) -> list[dict]:
    """Basin records, largest first (load balance: the big jobs start
    first; ties by id so the order is deterministic)."""
    basins = list(store.read_json("graph/basins.json")["basins"])
    basins.sort(key=lambda b: (-int(b.get("area_cells", 0)), int(b["id"])))
    return basins


def _run_jobs(jobs: list[dict], params: WorldParams, root: Path, workers: int, log, label: str, progress_every: float = 0.1) -> list[dict]:
    """Run ``jobs`` through :func:`basin_job.job` (in-process for one
    worker, else a spawned pool), logging progress.  Returns the stats in
    completion order."""
    n = len(jobs)
    out: list[dict] = []
    if n == 0:
        return out
    t0 = time.time()
    next_log = 0.0

    def report(st: dict, k: int):
        nonlocal next_log
        frac = k / n
        if k <= min(10, n) or frac >= next_log or k == n:
            next_log = frac + progress_every
            if st.get("kind") == "basin":
                d = st.get("deaths", {})
                log(
                    f"[refine] {label} {k}/{n}: basin {st['id']} (face {st['face']}, {st['area_cells']} cells, window {st['n_fine']}²) "
                    f"{st['iterations']} it, {st['particles']} particles, exits {d.get('exit', 0)} lakes {d.get('evap', 0)}, "
                    f"{st['written_cells']} cells written in {st['seconds']:.1f}s  [{time.time() - t0:.0f}s elapsed]"
                )
            else:
                log(f"[refine] {label} {k}/{n}: face {st['face']} {st['cells']} cells in {st['seconds']:.1f}s")

    if workers <= 1:
        for k, j in enumerate(jobs, 1):
            st = bj.job(j)
            out.append(st)
            report(st, k)
        return out
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    threads = max(1, (os.cpu_count() or 1) // workers)
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(workers, n), mp_context=ctx, initializer=bj.pool_init, initargs=(threads, str(root), params)) as ex:
        futs = [ex.submit(bj.job, j) for j in jobs]
        for k, fut in enumerate(as_completed(futs), 1):
            st = fut.result()  # a worker that died (OOM kill) raises BrokenProcessPool here
            out.append(st)
            report(st, k)
    return out


def run(store: WorldStore, params: WorldParams, log=print) -> dict:
    t0 = time.time()
    root = store.root
    rp = params.refine
    basins = load_basins(store)
    workers = params.workers()
    log(f"[refine] {len(basins)} basins, R={params.world.R}, N_fine={params.N_fine}, {rp.refine_iterations} iterations/basin, {workers} workers")
    if basins:
        # jobs run largest first, so the first `workers` windows are resident together
        ne = sorted((bj.basin_window(b, params.world.R, rp.halo_cells).NE for b in basins), reverse=True)[:workers]
        gb = sum(n * n for n in ne) * BYTES_PER_WINDOW_CELL / 1e9
        log(f"[refine] largest window {ne[0]}² fine cells; estimated peak job memory {gb:.1f} GB for {workers} workers (+ coarse inputs per worker); lower refine.workers if that exceeds the machine")
    rz.create_fine(root, params)

    # base pass: plain upsample of every face (fixed order before any basin writes)
    face_jobs = [{"kind": "face", "root": str(root), "params": params, "face": f} for f in range(6)]
    face_stats = _run_jobs(face_jobs, params, root, workers, log, "base")
    t_base = time.time() - t0

    jobs = []
    for k, b in enumerate(basins):
        qp = store.quicklook_path("refine", f"basin{int(b['id'])}") if k < N_BASIN_QUICKLOOKS else None
        jobs.append({"kind": "basin", "root": str(root), "params": params, "basin": b, "quicklook": None if qp is None else str(qp)})
    t1 = time.time()
    stats = _run_jobs(jobs, params, root, workers, log, "basin")
    t_basins = time.time() - t1

    stats.sort(key=lambda s: (-int(s.get("area_cells", 0)), int(s["id"])))
    deaths: dict[str, int] = {}
    for s in stats:
        for k, v in (s.get("deaths") or {}).items():
            deaths[k] = deaths.get(k, 0) + int(v)
    seconds_jobs = float(sum(s["seconds"] for s in stats))
    written = int(sum(s.get("written_cells", 0) for s in stats))
    land_fine = int(sum(int(b.get("area_cells", 0)) for b in basins)) * params.world.R ** 2
    info = {
        "n_basins": len(basins),
        "workers": workers,
        "iterations": int(rp.refine_iterations),
        "cells_written": written,
        "land_cells_fine": land_fine,
        "particles": int(sum(s.get("particles", 0) for s in stats)),
        "deaths": deaths,
        "pending_m_total": float(sum(s.get("pending_m", 0.0) for s in stats)),
        "lake_cells": int(sum(s.get("lake_cells", 0) for s in stats)),
        "lakes": int(sum(s.get("lakes", 0) for s in stats)),
        "lake_discarded_m": float(sum(s.get("lake_discarded_m", 0.0) for s in stats)),
        "drift_max_m": float(max([s.get("drift_max_m", 0.0) for s in stats] or [0.0])),
        "seconds_base": round(t_base, 2),
        "seconds_base_cpu": round(float(sum(s["seconds"] for s in face_stats)), 2),
        "seconds_basins_wall": round(t_basins, 2),
        "seconds_basins_cpu": round(seconds_jobs, 2),
        "seconds_total": round(time.time() - t0, 2),
        "largest": [
            {k: s[k] for k in ("id", "face", "area_cells", "n_fine", "active_cells", "iterations", "particles", "seconds", "seconds_erosion", "noise_max_m", "drift_max_m", "lake_cells", "lake_discarded_m")}
            for s in stats[:5]
        ],
        "quicklooks": [s["quicklook"] for s in stats if s.get("quicklook")],
    }
    if written != land_fine:
        log(f"[refine] WARNING: {written} basin cells written, {land_fine} fine land cells expected")
    log(
        f"[refine] base pass {t_base:.1f}s; {len(basins)} basins in {t_basins:.1f}s wall ({seconds_jobs:.1f}s cpu, {workers} workers); "
        f"{written} cells refined; deaths {deaths}; {info['lake_cells']} lake cells in {info['lakes']} lakes ({info['lake_discarded_m']:.0f} m·cells of lake deposits discarded); max coarse-scale drift removed {info['drift_max_m']:.0f} m"
    )
    return info


# --------------------------------------------------------------------------
# quicklook
# --------------------------------------------------------------------------
def _block_mean(a: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return np.asarray(a, dtype=np.float32)
    n = a.shape[0] // k
    return np.asarray(a[: n * k, : n * k], dtype=np.float32).reshape(n, k, n, k).mean(axis=(1, 3))


def _block_max(a: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return np.asarray(a, dtype=np.float32)
    n = a.shape[0] // k
    return np.asarray(a[: n * k, : n * k], dtype=np.float32).reshape(n, k, n, k).max(axis=(1, 3))


def quicklook(store: WorldStore, params: WorldParams, path, max_px: int = 1024) -> Path | None:
    """Whole-planet net: hillshaded fine surface (block-mean reduced to at
    most ``max_px`` per face), lakes (fine water surface > 0.05 m above
    the surface) and the fine discharge (block max, log-weighted blue)."""
    from ..viz import quicklook as ql

    root = store.root
    if not rz.fine_exists(root, params):
        return None
    N = params.N_fine
    k = 1
    while N // k > max_px:
        k *= 2
    n = N // k
    surf = np.empty((6, n, n), np.float32)
    lake = np.zeros((6, n, n), bool)
    q = np.empty((6, n, n), np.float32)
    for f in range(6):
        h = np.asarray(rz.open_fine(root, "height", f)) + np.asarray(rz.open_fine(root, "sediment", f))
        ws = np.asarray(rz.open_fine(root, "water_surface", f))
        surf[f] = _block_mean(h, k)
        lake[f] = _block_max(np.where((ws - h > 0.05) & (ws > 0.0), 1.0, 0.0).astype(np.float32), k) > 0.5
        q[f] = _block_max(np.asarray(rz.open_fine(root, "discharge", f)), k)
    cell = params.fine_cell_size_m * k
    img = ql.render_height(surf, 0.0, cell)
    img = ql.overlay(img, lake, (60, 120, 220), 0.9)
    lq = np.log1p(np.maximum(q, 0))
    if lq.max() > 0:
        thr = float(np.percentile(lq, 97))
        wgt = np.clip((lq - thr) / max(lq.max() - thr, 1e-9), 0, 1)
        img = ql.overlay(img, wgt > 0, (30, 80, 255), 0.95, weight=0.35 + 0.65 * wgt)
    return ql.save_image(path, img)


__all__ = ["OUTPUTS", "run", "quicklook", "load_basins", "N_BASIN_QUICKLOOKS"]
