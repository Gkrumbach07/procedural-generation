"""A planet level chained from a coarser planet level (docs/zoom-windows.md,
"The planet at 305 m"): ``PlanetLevel(R=32, parent=8)`` erodes every tile
from the 1.2 km level's result instead of from the planet's upsample, the
way a zoom's levels chain (:func:`bake.level_inputs`).

What a chained tile takes from its parent, over its window:

* the surface (bicubic), sediment, discharge and momentum (bilinear) of the
  parent's work raster -- where the parent wrote them; the planet's
  upsample where it did not (open sea);
* detail noise below the parent's cell at ``chain_detail`` x min(parent
  slope x parent cell, parent 3x3 relief), hashed as the first level's is;
* the hold at ``hold_scale`` *parent* cells;
* inflow where the parent's drainage crosses into the window: per face, a
  flood tree of the parent's surface over its whole work raster (the face
  and its guard beyond the cube edges), draining to the sea and the raster's
  border, with the planet's rain on it and the planet's cross-face inflow
  (:func:`planet.planet_inflow`) entering at the border
  (:func:`parent_flow`, ``<parent>/flow.f{k}.{recv,flux}.npy``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.ndimage import map_coordinates

from ..cubesphere import from_sphere_v, to_sphere_v
from ..hydro.priority_flood import priority_flood_flat
from ..refine.upsample import detail_amplitude
from ..refine.zoom import DETAIL_HEIGHT_SHARE
from . import bake as zb


def flow_path(pdir: Path, face: int, name: str) -> Path:
    return Path(pdir) / f"flow.f{face}.{name}.npy"


def _guarded_coarse(a: np.ndarray, face: int, N: int, g: int) -> np.ndarray:
    """``a (6, N, N)`` read at the coarse cells ``[-g, N + g)^2`` of ``face``,
    beyond its edges from whichever face owns the ground."""
    k = np.arange(-g, N + g)
    I, J = np.meshgrid(k, k, indexing="ij")
    p = to_sphere_v(np.full(I.shape, face), (I + 0.5) / N, (J + 0.5) / N)
    f2, u2, v2 = from_sphere_v(p)
    return a[f2, np.minimum((u2 * N).astype(np.int64), N - 1), np.minimum((v2 * N).astype(np.int64), N - 1)]


def parent_flow(root: Path, pdir: Path, parent_level, face: int, log=None) -> tuple[Path, Path]:
    """Write (once) the flood tree of the parent level's work raster of
    ``face``: ``recv`` (int32 flat index of each cell's receiver, -1 at a
    drain) and ``flux`` (float32 rain volume through each cell, the planet's
    precip per coarse cell spread over its fine cells).  Returns the paths."""
    from . import planet as zp

    rp, fp = flow_path(pdir, face, "recv"), flow_path(pdir, face, "flux")
    if rp.exists() and fp.exists():
        return rp, fp
    R = int(parent_level.R)
    g = int(parent_level.guard)
    fd, fa = zp.planet_flow(root)
    N = fd.shape[1]
    NF = (N + 2 * g) * R
    precip = np.stack([np.load(Path(root) / "coarse" / f"precip.f{k}.npy") for k in range(6)]).astype(np.float32)
    ocean_c = np.stack([np.load(Path(root) / "coarse" / f"flow_dir.f{k}.npy") == 255 for k in range(6)])
    height_c = np.stack([np.load(Path(root) / "coarse" / f"height.f{k}.npy") for k in range(6)]).astype(np.float32)
    sed_c = np.stack([np.load(Path(root) / "coarse" / f"sediment.f{k}.npy") for k in range(6)]).astype(np.float32)
    gp = _guarded_coarse(np.maximum(precip, 0.0), face, N, g)
    go = ndimage.binary_dilation(_guarded_coarse(ocean_c, face, N, g), structure=np.ones((3, 3), bool))
    gs = _guarded_coarse(height_c + sed_c, face, N, g)
    del precip, ocean_c, height_c, sed_c
    wk = {k: zp.open_work(pdir, face, k, NF, mode="r") for k in ("height", "sediment", "done")}
    surface = np.empty((NF, NF), np.float32)
    rows = 4 * R
    for i0 in range(0, NF, rows):
        i1 = min(NF, i0 + rows)
        plain = np.repeat(np.repeat(gs[i0 // R:(i1 + R - 1) // R], R, axis=0), R, axis=1)[: i1 - i0, :NF]
        done = np.asarray(wk["done"][i0:i1])
        surface[i0:i1] = np.where(done, np.asarray(wk["height"][i0:i1]) + np.asarray(wk["sediment"][i0:i1]), plain)
    del wk
    drain = (surface < 0.0) & np.repeat(np.repeat(go, R, axis=0), R, axis=1)
    drain[0, :] = drain[-1, :] = drain[:, 0] = drain[:, -1] = True
    weight = (np.repeat(np.repeat(gp, R, axis=0), R, axis=1) / float(R * R)).astype(np.float64)
    ring = np.pad(surface, 1, mode="edge")
    weight += zp.planet_inflow(fd, fa, face, -g, N + g, -g, N + g, R, ring)[1:-1, 1:-1]
    del ring
    fr = priority_flood_flat(surface, drain, None)
    del surface, drain
    acc = zb._accumulate(fr.pop_seq, fr.parent, weight.ravel())
    np.save(fp, acc.reshape(NF, NF).astype(np.float32))
    del acc, weight
    np.save(rp, fr.parent.astype(np.int32).reshape(NF, NF))
    if log is not None:
        log(f"parent flow of face {face}: {NF}^2 cells")
    return rp, fp


def chained_inputs(pdir: Path, parent_level, level, N: int, face: int, ta0: int, tb0: int, tr: dict, coarse_cell_m: float,
                   seed: int, coast_taper_m: float) -> dict:
    """A chained tile's surface, sediment, discharge, momentum, ocean, detail
    noise and inflow over its array (``tr``: the window plus a fine cell, as
    :func:`planet.planet_tile` cuts the upsample)."""
    from . import planet as zp

    Rc, Rp = int(level.R), int(parent_level.R)
    f = Rc // Rp
    gp = int(parent_level.guard)
    NFp = (N + 2 * gp) * Rp
    n = tr["height0"].shape[0]
    k = np.arange(n, dtype=np.float64)
    xi = (ta0 * Rc - 1 + k + 0.5) / Rc                      # coarse coordinates of the array's cell centres
    xj = (tb0 * Rc - 1 + k + 0.5) / Rc
    pi = (xi + gp) * Rp - 0.5                               # parent work-raster indices
    pj = (xj + gp) * Rp - 0.5
    r0, r1 = max(int(np.floor(pi[0])) - 3, 0), min(int(np.ceil(pi[-1])) + 4, NFp)
    c0, c1 = max(int(np.floor(pj[0])) - 3, 0), min(int(np.ceil(pj[-1])) + 4, NFp)
    wk = {nm: zp.open_work(pdir, face, nm, NFp, mode="r") for nm in zp.WORK_FIELDS}
    crop = {nm: np.asarray(wk[nm][r0:r1, c0:c1]) for nm in zp.WORK_FIELDS}
    I, J = np.meshgrid(pi - r0, pj - c0, indexing="ij")
    done = map_coordinates(crop["done"].astype(np.float32), [I, J], order=1, mode="nearest") > 0.999
    psurf = crop["height"].astype(np.float64) + crop["sediment"]
    plain_up = (tr["height0"] + tr["sediment0"]).astype(np.float64)
    plain = np.where(done, map_coordinates(psurf, [I, J], order=3, mode="nearest"), plain_up)
    sed0 = np.where(done, np.maximum(map_coordinates(crop["sediment"].astype(np.float64), [I, J], order=1, mode="nearest"), 0.0), tr["sediment0"])
    discharge = np.where(done, np.maximum(map_coordinates(crop["discharge"].astype(np.float64), [I, J], order=1, mode="nearest"), 0.0), tr["discharge"])
    momentum = np.stack([np.where(done, map_coordinates(crop["momentum"][..., c].astype(np.float64), [I, J], order=1, mode="nearest"), tr["momentum"][..., c])
                         for c in (0, 1)], -1)
    ocean = zb._ocean(plain, tr["basin_id"] < 0)
    pcell = coarse_cell_m / Rp
    gy, gx = np.gradient(psurf, pcell)
    slope = map_coordinates(np.hypot(gx, gy), [I, J], order=1, mode="nearest")
    relief = map_coordinates(zb._relief3(psurf), [I, J], order=1, mode="nearest")
    amp = detail_amplitude(slope, relief, tr["hardness"], float(level.chain_detail), pcell, plain, coast_taper_m, DETAIL_HEIGHT_SHARE)
    noise = np.where(ocean | ~done, 0.0, tr.get("quiet", 1.0) * amp * zp.hashed_ridged(seed, face, ta0 * Rc - 1, tb0 * Rc - 1, n, n, 2.0 * f))
    # inflow: parent cells outside the array's inner cells whose receiver is inside
    recv = np.asarray(np.load(flow_path(pdir, face, "recv"), mmap_mode="r")[r0:r1, c0:c1]).astype(np.int64)
    flux = np.asarray(np.load(flow_path(pdir, face, "flux"), mmap_mode="r")[r0:r1, c0:c1]).astype(np.float64)
    ki = ((np.arange(r0, r1) + 0.5) / Rp - gp) * Rc + 0.5 - ta0 * Rc        # child array coordinate of parent centres
    kj = ((np.arange(c0, c1) + 0.5) / Rp - gp) * Rc + 0.5 - tb0 * Rc
    in_i = (ki >= 1.0) & (ki <= n - 2.0)
    in_j = (kj >= 1.0) & (kj <= n - 2.0)
    ok = recv >= 0
    ri, rj = np.divmod(np.where(ok, recv, 0), NFp)
    ri_c, rj_c = ri - r0, rj - c0
    ok &= (ri_c >= 0) & (ri_c < r1 - r0) & (rj_c >= 0) & (rj_c < c1 - c0)
    di, dj = np.meshgrid(np.arange(r1 - r0), np.arange(c1 - c0), indexing="ij")
    inside_d = in_i[di] & in_j[dj]
    inside_r = np.zeros(ok.shape, bool)
    inside_r[ok] = in_i[ri_c[ok]] & in_j[rj_c[ok]]
    cross = ok & ~inside_d & inside_r
    ci = (ki[di[cross]] + ki[ri_c[cross]]) / 2.0
    cj = (kj[dj[cross]] + kj[rj_c[cross]]) / 2.0
    src = zb.spawn_at(ci, cj, flux[cross], plain, max(f // 2, 1))
    return {"plain": plain, "height0": plain - sed0, "sediment0": sed0, "discharge": discharge, "momentum": momentum, "ocean": ocean,
            "noise": noise, "src": src, "f": f}


__all__ = ["flow_path", "parent_flow", "chained_inputs"]
