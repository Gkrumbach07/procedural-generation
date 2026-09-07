"""Erosion stage driver (PLAN.md section 8.3): the global pass on the coarse
grid with checkpoints, resume and periodic quicklooks.

Inputs (coarse fields): ``bedrock``, ``uplift``, ``hardness`` (tectonics),
``precip``, ``evap`` (climate).  Outputs: ``height``, ``sediment``,
``discharge``, ``momentum`` (docs/DEVELOPING.md).  Sediment that found no
room below sea level by the end of the run (``ErosionState.pending``) is
not part of the outputs; its total is reported as ``pending_total_m`` in
the stage info (and ``land_fraction`` next to ``land_fraction_bedrock``:
the particle pass never turns sea into land or land into sea, so any
drift is tectonic uplift / subsidence).

Checkpoints: ``checkpoints/erosion_iterNNNN.npz`` (extended state arrays in
cell units) + ``checkpoints/erosion_iterNNNN.json`` (iteration, parameter
hash); ``run`` resumes from the newest checkpoint whose hash (parameters,
upstream output hashes, kernel version) matches unless ``erosion.resume``
is False.  ``bake.py --force`` clears the outputs but not ``checkpoints/``:
with an unchanged hash the final checkpoint is reused (that is what it is
for); a kernel change bumps ``KERNEL_VERSION`` and invalidates it.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from ..config import WorldParams
from ..field import FaceField
from ..io.world_store import WorldStore
from .maps import ErosionState, step
from .particle import KERNEL_VERSION

OUTPUTS = ["height", "sediment", "discharge", "momentum"]

_CKPT_RE = re.compile(r"erosion_iter(\d+)\.npz$")


def _ckpt_hash(params: WorldParams, store: WorldStore | None = None) -> str:
    """Parameters *and* upstream outputs a checkpoint depends on: the world
    and erosion groups, the kernel version (``particle.KERNEL_VERSION``: a
    code change never resumes stale results) and the manifest hashes of the
    tectonics and climate outputs (so a re-baked upstream stage — even with
    the same parameters, e.g. a stub replaced by the real stage —
    invalidates it)."""
    up = f":k{KERNEL_VERSION}"
    if store is not None:
        for s in ("tectonics", "climate"):
            info = store.stage_info(s) or {}
            up += ":" + str(info.get("hash", ""))
    return params.group_hash("world", "erosion") + ":" + params.group_hash("tectonics", "climate") + up


def save_checkpoint(store: WorldStore, state: ErosionState, params: WorldParams) -> Path:
    d = store.checkpoint_dir
    d.mkdir(parents=True, exist_ok=True)
    it = state.iteration
    p = d / f"erosion_iter{it:04d}.npz"
    tmp = p.with_suffix(".npz.tmp")
    arrays = dict(height=state.height, sediment=state.sediment, discharge=state.discharge, momentum=state.momentum, pending=state.pending)
    if state.route is not None:  # the routing surface is refreshed every flood_every iterations: part of the state
        arrays["route"] = state.route
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    tmp.replace(p)
    meta = {"iteration": it, "params_hash": _ckpt_hash(params, store), "height_unit_m": state.height_unit_m, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(p.with_suffix(".json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    return p


def find_checkpoint(store: WorldStore, params: WorldParams, max_iteration: int | None = None) -> tuple[Path, dict] | None:
    """Newest valid checkpoint (matching parameter hash, iteration <= max)."""
    d = store.checkpoint_dir
    if not d.exists():
        return None
    want = _ckpt_hash(params, store)
    best = None
    for p in d.glob("erosion_iter*.npz"):
        m = _CKPT_RE.search(p.name)
        if not m:
            continue
        jp = p.with_suffix(".json")
        if not jp.exists():
            continue
        try:
            meta = json.loads(jp.read_text())
        except json.JSONDecodeError:
            continue
        if meta.get("params_hash") != want:
            continue
        it = int(meta.get("iteration", -1))
        if max_iteration is not None and it > max_iteration:
            continue
        if best is None or it > best[1]["iteration"]:
            best = (p, meta)
    return best


def load_checkpoint(state: ErosionState, path: Path, meta: dict) -> None:
    with np.load(path) as z:
        state.height[...] = z["height"]
        state.sediment[...] = z["sediment"]
        state.discharge[...] = z["discharge"]
        state.momentum[...] = z["momentum"]
        state.pending[...] = z["pending"] if "pending" in z.files else 0.0
        state.route = np.ascontiguousarray(z["route"]) if "route" in z.files else None
    state.iteration = int(meta["iteration"])


def build_state(store: WorldStore, params: WorldParams) -> ErosionState:
    grid = params.coarse_grid()
    bed = store.load_field("bedrock", grid)
    hard = store.load_field("hardness", grid)
    upl = store.load_field("uplift", grid)
    pr = store.load_field("precip", grid)
    ev = store.load_field("evap", grid)
    return ErosionState.from_grid(grid, bed, hard, pr, ev, upl, params.erosion)


def write_outputs(store: WorldStore, state: ErosionState) -> None:
    for f in state.fields().values():
        store.save_field(f)


def run(store: WorldStore, params: WorldParams, log=print) -> dict:
    ep = params.erosion
    grid = params.coarse_grid()
    state = build_state(store, params)
    n_iter = int(ep.iterations)
    land0 = float(np.mean((state.height + state.sediment)[state.interior] >= 0))
    ck = find_checkpoint(store, params, max_iteration=n_iter) if ep.resume else None
    if ck is not None:
        load_checkpoint(state, *ck)
        log(f"[erosion] resumed from {ck[0].name} (iteration {state.iteration})")
    log(f"[erosion] {grid.describe()}; heights in units of {state.height_unit_m:.1f} m; {n_iter} iterations, {ep.particles_per_cell} particles/cell")
    times = []
    clamped = 0
    while state.iteration < n_iter:
        it = state.iteration
        t0 = time.time()
        st = step(state, params, it)
        dt = time.time() - t0
        times.append(dt)
        clamped += int(st.get("clamped", 0))
        d = st.get("deaths", {})
        log(
            f"[erosion] iter {it + 1}/{n_iter}: {st['particles']} particles, mean {st['steps_mean']:.0f} steps, "
            f"deaths ocean {d.get('ocean', 0)} pit {d.get('pit', 0)} age {d.get('age', 0)}, clamped {st.get('clamped', 0)}, "
            f"pending {st.get('pending_total', 0.0):.1f}, {st['seconds_particles']:.2f}s particles, {dt:.2f}s total"
        )
        done = state.iteration
        if ep.checkpoint_every > 0 and (done % ep.checkpoint_every == 0 or done == n_iter):
            p = save_checkpoint(store, state, params)
            log(f"[erosion] checkpoint -> {p.name}")
        if ep.quicklook_every > 0 and done % ep.quicklook_every == 0 and done != n_iter:
            try:
                qp = quicklook_state(state, store.quicklook_path("erosion", f"iter{done:04d}"))
                log(f"[erosion] quicklook -> {qp}")
            except Exception as e:  # never break a bake on a picture
                log(f"[erosion] quicklook failed: {e!r}")
    write_outputs(store, state)
    surf = (state.height + state.sediment)[state.interior]
    info = {
        "iterations": n_iter,
        "height_unit_m": state.height_unit_m,
        "seconds_per_iteration": float(np.mean(times)) if times else 0.0,
        "land_fraction": float(np.mean(surf >= 0)),
        "land_fraction_bedrock": land0,
        "max_discharge": float(state.discharge[state.interior].max()),
        "sediment_mean_m": float(state.sediment[state.interior].mean() * state.height_unit_m),
        "sediment_p99_m": float(np.percentile(state.sediment[state.interior], 99) * state.height_unit_m),
        "pending_total_m": float(state.pending[state.interior].sum() * state.height_unit_m),
        "clamped_entries": clamped,
    }
    return info


def quicklook_state(state: ErosionState, path) -> Path:
    from ..viz import quicklook as ql

    grid = state.grid
    h = FaceField(grid, (state.height + state.sediment).astype(np.float32) * state.height_unit_m, name="surface")
    q = FaceField(grid, state.discharge.astype(np.float32), name="discharge")
    return ql.quicklook_height(path, h, 0.0, grid.cell_size_m, discharge=q)


def quicklook(store: WorldStore, params: WorldParams, path) -> Path:
    """Hillshade of ``height + sediment`` with the discharge overlay."""
    from ..viz import quicklook as ql

    grid = params.coarse_grid()
    h = store.load_field("height", grid)
    h.data += store.load_field("sediment", grid).data
    q = store.load_field("discharge", grid)
    return ql.quicklook_height(path, h, 0.0, grid.cell_size_m, discharge=q)


__all__ = ["OUTPUTS", "run", "quicklook", "build_state", "save_checkpoint", "find_checkpoint", "load_checkpoint", "quicklook_state"]
