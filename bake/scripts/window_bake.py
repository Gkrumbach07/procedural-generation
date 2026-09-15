#!/usr/bin/env python3
"""One catchment of a baked world refined at R = 2 / 8 / 32 / 128 (4.9 km /
1.2 km / 305 m / 76 m on the earth preset; R must be a power of two), through
the refine stage's own basin job, measuring cost, convergence, river
connectivity, relief and lakes (docs/zoom-windows.md).

    # a level from the planet, then the next level chained from it
    python scripts/window_bake.py --world worlds/earth-v9 --outlet 5 239 913 --R 32 --iters 150 --save --out W --tag a_
    python scripts/window_bake.py --world worlds/earth-v9 --outlet 5 239 913 --R 128 --iters 40 \
        --parent W/a_R32.npz --parent-R 32 --save --out W --tag b_
    python scripts/window_view.py W/b_R128.npz --shot b.png

Defaults are the zoom-window setup measured best (``refine.zoom``: McDonald's
erosion settings with lakes in erosion, detail noise 3, the smooth drift
correction); ``--profile shipped`` runs the refine stage's own settings.
``--parent`` starts from a coarser saved window of the same catchment and
halo instead of the planet (upsampled surface and discharge, detail only
below the parent's cell, drift held to the parent at the parent's cell) --
soillib's multiscale procedure.  The window is the D8 catchment upstream of
``--outlet`` (face, i, j on the coarse grid), confined to one face, handed
to ``refine.basin_job._run_basin`` as a synthetic basin whose only exit is
the outlet cell.  Prototypes, monkeypatched in: ``--relief drainage``
(scripts/drainage_relief.py), ``--census N`` (scripts/pit_census.py),
``--breach`` (scripts/dam_breach.py).
"""
import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.environ.get("BAKE") or os.path.join(os.path.dirname(__file__), ".."))
import numba  # noqa: E402

from globe.config import WorldParams  # noqa: E402
from globe.erosion import particle as pk  # noqa: E402
from globe.io.world_store import WorldStore  # noqa: E402
from globe.refine import basin_job as bj  # noqa: E402
from globe.refine.upsample import basin_window  # noqa: E402

D8 = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
BID = 900_000


def catchment(world: Path, f: int, oi: int, oj: int) -> np.ndarray:
    fd = np.load(world / "coarse" / f"flow_dir.f{f}.npy").astype(np.int64)
    N = fd.shape[0]
    di = np.array([d[0] for d in D8]); dj = np.array([d[1] for d in D8])
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    ok = fd < 8
    k = np.minimum(fd, 7)
    ti, tj = ii + di[k], jj + dj[k]
    inside = ok & (ti >= 0) & (ti < N) & (tj >= 0) & (tj < N)
    down = np.where(inside, ti * N + tj, -1).ravel()
    up = np.zeros(N * N, bool)
    up[oi * N + oj] = True
    while True:
        nxt = (down >= 0) & up[np.maximum(down, 0)] & ~up
        if not nxt.any():
            return up.reshape(N, N)
        up |= nxt


def channel_metrics(q, active, cell_km, per_km2, outlet_cells):
    """Channels at upstream-area thresholds.  ``per_km2`` is the outlet's
    discharge divided by the catchment's area, so a threshold of A km^2 is
    q > A x per_km2 whatever the discharge units, precipitation or losses."""
    out = {}
    area_km2 = active.sum() * cell_km ** 2
    for A in (1000.0, 100.0, 10.0, 1.0, 0.1):
        if A < 4.0 * cell_km ** 2:   # a threshold below a few cells' own area marks every cell
            continue
        thr = A * per_km2
        ch = (q > thr) & active
        n = int(ch.sum())
        if n == 0:
            out[f"{A:g}km2"] = {"channel_cells": 0}
            continue
        lab, nlab = ndimage.label(ch, structure=np.ones((3, 3), bool))
        sizes = np.bincount(lab.ravel())[1:]
        out_lab = np.unique(lab[outlet_cells & ch])
        out_lab = out_lab[out_lab > 0]
        conn = float(sizes[out_lab - 1].sum() / n) if out_lab.size else 0.0
        dist = ndimage.distance_transform_edt(~ch)
        out[f"{A:g}km2"] = {
            "channel_cells": n,
            "drainage_density_km_per_km2": round(n * cell_km / area_km2, 4),
            "components": int(nlab),
            "share_connected_to_outlet": round(conn, 4),
            "mean_distance_to_channel_km": round(float(dist[active].mean() * cell_km), 3),
        }
    return out


def block_drift_stats(delta, act, R):
    """|mean(surface - plain upsample)| per coarse cell (fully active blocks):
    how far the refined surface left the parent at the parent's own scale."""
    n = (delta.shape[0] // R) * R
    d = np.where(act, delta, 0.0)[:n, :n].reshape(n // R, R, n // R, R)
    c = act[:n, :n].reshape(n // R, R, n // R, R).sum(axis=(1, 3))
    m = np.abs(d.sum(axis=(1, 3)) / np.maximum(c, 1))[c == R * R]
    return [round(float(x), 1) for x in (np.percentile(m, [50, 90]).tolist() + [m.max()])] if m.size else []


def relief_stats(surf, act, cell_m, L_m):
    """Local relief (max - min) in a square of side L_m around each active
    cell: the p50 / p90 over the catchment, metres."""
    k = max(3, int(round(L_m / cell_m)) | 1)
    mx = ndimage.maximum_filter(surf, size=k)
    mn = ndimage.minimum_filter(surf, size=k)
    r = (mx - mn)[act]
    return [round(float(x), 1) for x in np.percentile(r, [50, 90])] if r.size else []


def quicklook(res, path, cell_m, crop=800):
    from PIL import Image

    a = res.arrays
    surf = (a["height"] + a["sediment"]).astype(np.float64)
    q = a["discharge"].astype(np.float64)
    act = a["mask"] > 0
    lake = (a["water_surface"] - surf > 0.5) & act
    gy, gx = np.gradient(surf, cell_m)
    ex = 3.0
    nrm = np.stack([-gx * ex, -gy * ex, np.ones_like(surf)], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    L = np.array([-0.6, 0.6, 0.75]); L /= np.linalg.norm(L)
    shade = np.clip(nrm @ L / L[2], 0, 1.6)[..., None]
    lo, hi = np.percentile(surf[act], [2, 99.5]) if act.any() else (0, 1)
    t = np.clip((surf - lo) / max(hi - lo, 1), 0, 1)[..., None]
    base = (np.array([0.33, 0.52, 0.30]) * (1 - t) + np.array([0.80, 0.74, 0.60]) * t) * shade * 0.85
    base = np.where(act[..., None], base, base * 0.35)
    v = np.log1p(q)
    ql = v[act] if act.any() else v.ravel()
    vlo, vhi = np.percentile(ql, [85, 99.7])
    x = np.clip((v - vlo) / max(vhi - vlo, 1e-9), 0, 1)
    x = (x * x * (3 - 2 * x))[..., None]
    img = base * (1 - 0.9 * x) + np.array([0.16, 0.36, 0.74]) * 0.9 * x
    img = np.where(lake[..., None], np.array([0.22, 0.42, 0.72]), img)
    img8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(img8)
    full = im.resize((min(1400, im.width), min(1400, im.height)), Image.BILINEAR) if im.width > 1400 else im
    full.save(path.parent / (path.name + ".full.png"))
    c0 = max(0, im.width // 2 - crop // 2)
    im.crop((c0, c0, c0 + min(crop, im.width), c0 + min(crop, im.height))).save(path.parent / (path.name + ".crop.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    ap.add_argument("--outlet", type=int, nargs=3, required=True)
    ap.add_argument("--R", type=int, nargs="+", default=[2, 10, 40])
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--halo", type=int, default=2)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--detail-amp", type=float, default=None)
    ap.add_argument("--erosion", nargs="*", default=[], help="erosion overrides k=v (floats)")
    ap.add_argument("--relief", choices=["noise", "drainage"], default="noise")
    ap.add_argument("--no-block-drift", action="store_true", help="skip the job's per-coarse-cell drift correction (same as --drift none)")
    ap.add_argument("--drift", choices=["block", "smooth", "none"], default=None, help="drift correction: the job's per-coarse-cell one, refine.zoom.smooth_drift (default with --profile zoom), or none")
    ap.add_argument("--profile", choices=["zoom", "shipped"], default="zoom", help="zoom (default): refine.zoom.ZOOM_EROSION and ZOOM_REFINE; shipped: the world's own refine settings (--erosion / --detail-amp override either)")
    ap.add_argument("--breach", action="store_true", help="breach the dams each particle pass builds (scratch/window/dam_breach.py)")
    ap.add_argument("--parent", default=None, help="start from this window's saved arrays (a coarser --save run of the same catchment and halo) instead of the planet: multiscale chaining")
    ap.add_argument("--parent-R", type=int, default=None, help="the parent window's R")
    ap.add_argument("--chain-detail", type=float, default=0.5, help="with --parent: detail amplitude x min(parent slope x parent cell, parent 3x3 relief), octaves below the parent cell only")
    ap.add_argument("--census", type=int, default=0, help="pit census for the first N iterations (scratch/window/pit_census.py)")
    ap.add_argument("--relief-elev", type=float, default=0.1, help="drainage relief: valley depth per metre of elevation")
    ap.add_argument("--relief-rel", type=float, default=0.5, help="drainage relief: valley depth per metre of coarse 3x3 relief")
    ap.add_argument("--save", action="store_true", help="write the window arrays (float32 npz)")
    a = ap.parse_args()
    world = Path(a.world)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    store = WorldStore(world)
    base = WorldParams.from_dict(store.manifest["params"])
    f, oi, oj = a.outlet
    m = catchment(world, f, oi, oj)
    idx = np.argwhere(m)
    i0, j0 = idx.min(0); i1, j1 = idx.max(0)
    basin = {"id": BID, "face": f, "faces": [f], "outlet": [f, oi, oj], "exits": [[f, oi, oj]],
             "area_cells": int(m.sum()), "bbox": [int(i0), int(j0), int(i1) + 1, int(j1) + 1],
             "pieces": [{"face": f, "bbox": [int(i0), int(j0), int(i1) + 1, int(j1) + 1], "area_cells": int(m.sum())}]}
    print(f"catchment of {a.outlet}: {m.sum()} coarse cells, bbox {basin['bbox']}", flush=True)

    grid, fields, derived = bj.coarse_inputs(world, base)
    bid_field = fields["basin_id"]
    H = grid.H
    inter = bid_field.data[f, H:-H, H:-H]
    inter[m] = BID
    inter[(inter == BID) & ~m] = -2  # (never: BID is unused in the world)
    bid_field.exchange_halos()

    coarse_km = grid.cell_size_m / 1000.0
    for R in a.R:
        ro = {"halo_cells": int(a.halo), "refine_iterations": int(a.iters)}
        if a.profile == "zoom":
            from globe.refine.zoom import ZOOM_REFINE
            ro.update(ZOOM_REFINE)
        if a.detail_amp is not None:
            ro["detail_amp"] = float(a.detail_amp)
        eo = {}
        if a.profile == "zoom":
            from globe.refine.zoom import ZOOM_EROSION
            eo.update(ZOOM_EROSION)
        eo.update({k: float(v) for k, v in (x.split("=") for x in a.erosion)})
        if "max_steps" in eo:
            eo["max_steps"] = int(eo["max_steps"])
        p = base.with_overrides(world={"R": int(R)}, refine=ro, erosion=eo)
        win = basin_window(basin, R, a.halo, face=f, N=grid.N)
        conv = []
        prev = {}

        real_step = bj.step
        from globe.erosion import maps as emaps
        real_run_it, real_thermal = emaps.run_iteration, emaps.thermal_erosion
        census_rows = []
        snaps = {}
        cstate = {"prev": None}
        breach_stats = {"breached": 0, "not_breachable": 0, "moved_m": 0.0}
        if a.census or a.breach:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import pit_census
            import dam_breach

            def run_it(state, *args_, **kw_):
                Hk = state.H
                snaps["s0"] = (state.height[0, Hk:-Hk, Hk:-Hk] + state.sediment[0, Hk:-Hk, Hk:-Hk]) * state.height_unit_m
                r_ = real_run_it(state, *args_, **kw_)
                if a.breach:
                    mk = state.mask[0, Hk:-Hk, Hk:-Hk]
                    if "drain" not in cstate:
                        cstate["drain"] = bj.exit_cells(basin, win, (win.NE, win.NE))[Hk:-Hk, Hk:-Hk] | (mk == pk.MASK_OUTSIDE)
                        cstate["fa"] = (mk > 0) | cstate["drain"]
                    b_ = dam_breach.breach(state, snaps["s0"], cstate["drain"], cstate["fa"])
                    for k_, v_ in b_.items():
                        breach_stats[k_] += v_
                    state.exchange_halos() if state.spherical else None
                snaps["s1"] = (state.height[0, Hk:-Hk, Hk:-Hk] + state.sediment[0, Hk:-Hk, Hk:-Hk]) * state.height_unit_m
                return r_

            emaps.run_iteration = run_it
            if not a.census:
                a.census = 0

        def step(state, ep, it, **kw):
            st = real_step(state, ep, it, **kw)
            if a.census and it < a.census:
                Hk = state.H
                s2 = (state.height[0, Hk:-Hk, Hk:-Hk] + state.sediment[0, Hk:-Hk, Hk:-Hk]) * state.height_unit_m
                mk = state.mask[0, Hk:-Hk, Hk:-Hk]
                act_c = mk == pk.MASK_ACTIVE
                if "drain" not in cstate:
                    drain_e = bj.exit_cells(basin, win, (win.NE, win.NE))[Hk:-Hk, Hk:-Hk]
                    cstate["drain"] = drain_e | (mk == pk.MASK_OUTSIDE)
                    cstate["fa"] = (mk > 0) | cstate["drain"]
                if cstate.get("prev") is None:
                    cstate["prev"] = pit_census.depressions(snaps["s0"], cstate["drain"], cstate["fa"], act_c, 0.1)[0]
                cs_, pits = pit_census.census(snaps["s0"], snaps["s1"], s2, cstate["drain"], cstate["fa"], act_c, cstate["prev"])
                cstate["prev"] = pits
                cs_["it"] = it + 1
                census_rows.append(cs_)
                print("  census", json.dumps(cs_), flush=True)
            Hk = state.H
            s = (state.height[0, Hk:-Hk, Hk:-Hk] + state.sediment[0, Hk:-Hk, Hk:-Hk]) * state.height_unit_m
            q = state.discharge[0, Hk:-Hk, Hk:-Hk]
            act = state.mask[0, Hk:-Hk, Hk:-Hk] == pk.MASK_ACTIVE
            if "s" in prev:
                conv.append({"it": it + 1,
                             "mean_abs_dz_m": float(np.abs(s - prev["s"])[act].mean()),
                             "rel_dq": float(np.abs(q - prev["q"])[act].sum() / max(np.abs(q)[act].sum(), 1e-12)),
                             "steps_mean": float(st.get("steps_mean", 0.0)),
                             "seconds": float(st.get("seconds_total", 0.0))})
            prev["s"], prev["q"] = s.copy(), q.copy()
            return st

        bj.step = step
        real_build_mask, real_detail, real_drift = bj.build_mask, bj.detail_noise, bj.block_drift
        real_upsample = bj.upsample_window
        if a.parent:
            from scipy.ndimage import map_coordinates, maximum_filter, minimum_filter
            from globe.refine.upsample import ridged_fbm
            par = np.load(a.parent)
            fR = R // int(a.parent_R)
            par_cell = grid.cell_size_m / int(a.parent_R)
            psurf = (par["height"] + par["sediment"]).astype(np.float64)
            pgy, pgx = np.gradient(psurf, par_cell)
            pslope = np.hypot(pgx, pgy)
            prel = maximum_filter(psurf, size=3) - minimum_filter(psurf, size=3)

            def to_child(arr, H_, NE_, order=3):
                e = (np.arange(NE_) - H_ + 0.5) / fR - 0.5
                I, J = np.meshgrid(e, e, indexing="ij")
                return map_coordinates(np.asarray(arr, np.float64), [I, J], order=order, mode="nearest")

            def upsample(fields_, derived_, win_, grid_):
                up_ = real_upsample(fields_, derived_, win_, grid_)
                H_, NE_ = win_.H, win_.NE
                up_["height0"] = to_child(par["height"], H_, NE_).astype(up_["height0"].dtype)
                up_["sediment0"] = np.maximum(to_child(par["sediment"], H_, NE_, order=1), 0.0).astype(up_["sediment0"].dtype)
                up_["discharge"] = np.maximum(to_child(par["discharge"], H_, NE_, order=1), 0.0).astype(up_["discharge"].dtype)
                # the saved window has no momentum: start without, rather than the
                # planet's over the parent's discharge (the push divides one by the
                # other, and that mismatch drew straight parallel tracks)
                up_["momentum"] = np.zeros_like(up_["momentum"])
                up_["slope"] = to_child(pslope, H_, NE_, order=1).astype(np.float32)
                up_["relief"] = to_child(prel, H_, NE_, order=1).astype(np.float32)
                return up_

            def chain_detail(win_, slope, relief, hardness, detail_amp, cell_size_m, gen, surface=None, coast_taper_m=0.0):
                amp = a.chain_detail * np.minimum(np.maximum(slope, 0) * par_cell, np.maximum(relief, 0)) * (0.5 + 0.5 * np.clip(hardness, 0, 1))
                if surface is not None and coast_taper_m > 0:
                    t = np.clip(np.abs(surface) / coast_taper_m, 0, 1)
                    amp = amp * t * t * (3 - 2 * t)
                return (amp * ridged_fbm((win_.NE, win_.NE), 2.0 * fR, gen)).astype(np.float32)

            bj.upsample_window, bj.detail_noise = upsample, chain_detail
        drift_mode = a.drift or ("none" if a.no_block_drift else ("smooth" if a.profile == "zoom" else "block"))
        if drift_mode == "none":
            bj.block_drift = lambda delta, cells, R_, **kw: np.zeros_like(delta)
        elif drift_mode == "smooth":
            from globe.refine.zoom import smooth_drift
            if a.parent:   # hold the parent at the parent's cell, not the planet's
                bj.block_drift = lambda delta, cells, R_, **kw: smooth_drift(delta, cells, R // int(a.parent_R))
            else:
                bj.block_drift = lambda delta, cells, R_, **kw: smooth_drift(delta, cells, R_)
        dstats = {}
        if a.relief == "drainage":
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from drainage_relief import drainage_relief
            stash = {}

            def build_mask(bid_up, bid):
                stash["bid_up"] = np.asarray(bid_up)
                stash["mask"] = real_build_mask(bid_up, bid)
                return stash["mask"]

            def detail(win_, slope, relief, hardness, detail_amp, cell_size_m, gen, surface=None, coast_taper_m=0.0):
                mask = stash["mask"]
                act_ = mask == pk.MASK_ACTIVE
                amp = (a.relief_elev * np.maximum(surface, 0.0) + a.relief_rel * np.maximum(relief, 0.0)) * (0.5 + 0.5 * np.clip(hardness, 0, 1))
                if coast_taper_m > 0:
                    t = np.clip(np.abs(surface) / coast_taper_m, 0, 1)
                    amp = amp * t * t * (3 - 2 * t)
                # drain where the job drains: the exit cells (and ocean, i.e.
                # outside the basin, never through the frozen divide ring)
                drain = bj.exit_cells(basin, win_, mask.shape) | (stash.get("bid_up") is not None and (stash["bid_up"] < 0))
                Z = drainage_relief(surface, act_ | drain, drain, win_.R, cell_size_m / win_.R, gen, amp, stats=dstats)
                return np.where(act_, Z - surface, 0.0).astype(np.float32)

            bj.build_mask, bj.detail_noise = build_mask, detail
        numba.set_num_threads(min(a.threads, numba.config.NUMBA_NUM_THREADS))
        ru0 = resource.getrusage(resource.RUSAGE_SELF)
        t0 = time.time()
        try:
            res = bj._run_basin(world, basin, p, None, t0, BID, p.refine, grid, fields, derived, win,
                                win.H, win.n, win.NE, up_threads=numba.get_num_threads())
        finally:
            bj.step = real_step
            emaps.run_iteration = real_run_it
            bj.build_mask, bj.detail_noise, bj.block_drift = real_build_mask, real_detail, real_drift
            bj.upsample_window = real_upsample
        wall = time.time() - t0
        ru1 = resource.getrusage(resource.RUSAGE_SELF)
        cpu = (ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)
        s = res.stats
        cell_m = grid.cell_size_m / R
        arr = res.arrays
        act = arr["mask"] == pk.MASK_ACTIVE
        # outlet: the fine block of the exit cell, and the frozen ring next to it
        outlet = np.zeros(act.shape, bool)
        a0 = (oi - win.ci0) * R; b0 = (oj - win.cj0) * R
        outlet[max(a0 - R, 0):a0 + 2 * R, max(b0 - R, 0):b0 + 2 * R] = True
        q, q0 = arr["discharge"].astype(np.float64), arr["discharge0"].astype(np.float64)
        surf = arr["height"] + arr["sediment"]
        plain = arr["height0"] + arr["sediment0"]
        lake = (arr["water_surface"] - surf > 0.5) & (arr["mask"] > 0)
        llab, nl = ndimage.label(lake, structure=np.ones((3, 3), bool))
        lsizes = np.bincount(llab.ravel())[1:] * (cell_m / 1000.0) ** 2 if nl else np.array([])
        n_it = max(s["iterations"], 1)
        rec = {
            "R": R, "cell_m": cell_m, "window_fine": win.n, "kernel_cells": win.NE ** 2,
            "active_cells": s["active_cells"], "iterations": s["iterations"],
            "wall_s": round(wall, 1), "cpu_s": round(cpu, 1), "erosion_s": round(s["seconds_erosion"], 1),
            "upsample_s": round(s["seconds_upsample"], 1), "threads": numba.get_num_threads(),
            "cpu_us_per_active_cell_iter": round(cpu / (s["active_cells"] * n_it) * 1e6, 3),
            "wall_us_per_active_cell_iter": round(s["seconds_erosion"] / (s["active_cells"] * n_it) * 1e6, 3),
            "peak_rss_mb": s["peak_rss_mb"], "steps_mean": round(s["steps_mean"], 1), "deaths": s["deaths"],
            "particles": s["particles"], "clamped": s["clamped"],
            "outlet_discharge_ratio": round(float(q[outlet].max() / max(q0[outlet].max(), 1e-12)), 3),
            "detail_std_m": round(float((surf - plain)[act].std()), 2), "noise_max_m": round(s["noise_max_m"], 1),
            "drift_max_m": round(s["drift_max_m"], 2),
            "lakes": int(nl), "lake_area_km2": round(float(lsizes.sum()), 2),
            "lake_sizes_km2_p50_p90_max": [round(float(x), 3) for x in (np.percentile(lsizes, [50, 90]).tolist() + [lsizes.max()])] if nl else [],
            "coarse_drift_p50_p90_max_m": block_drift_stats(surf - plain, act, R),
            "relief_3km_p50_p90_m": relief_stats(surf, act, cell_m, 3000.0),
            "relief_3km_plain_p50_p90_m": relief_stats(plain, act, cell_m, 3000.0),
            "census": census_rows, "breach": breach_stats, "profile": a.profile, "drift_mode": drift_mode,
            "relief_mode": a.relief, "relief_elev": a.relief_elev, "relief_rel": a.relief_rel, "drainage_stats": dstats,
            "channels": channel_metrics(q, act, cell_m / 1000.0, float(q[outlet].max()) / (m.sum() * coarse_km ** 2), outlet),
            "convergence": conv[:: max(1, len(conv) // 30)] + conv[-1:],
        }
        name = f"{a.tag}R{R}"
        if a.save:
            np.savez_compressed(out / f"{name}.npz", **{k: arr[k] for k in ("height", "sediment", "discharge", "water_surface", "mask", "height0", "sediment0")})
        (out / f"{name}.json").write_text(json.dumps(rec, indent=1))
        quicklook(res, out / f"{name}", cell_m)
        print(f"R={R:>3} cell {cell_m:7.1f} m  window {win.n}²  active {s['active_cells']:,}  wall {wall:7.1f}s  cpu {cpu:8.1f}s  "
              f"cpu µs/cell-it {rec['cpu_us_per_active_cell_iter']:.2f}  wall µs {rec['wall_us_per_active_cell_iter']:.3f}  "
              f"steps {rec['steps_mean']:.0f}  outlet q ratio {rec['outlet_discharge_ratio']}  lakes {nl}  rss {s['peak_rss_mb'][-1]:.0f} MB", flush=True)
        del res


if __name__ == "__main__":
    main()
