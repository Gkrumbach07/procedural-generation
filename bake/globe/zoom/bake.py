"""Zoom bake: a square region around a spot refined level by level, each
level tiled and chained from the one above (docs/zoom-windows.md, "The zoom
stage").

A level is a square *product* of ``cells`` coarse cells around the spot at
refinement ``R``, inside a *work* array that adds ``guard`` coarse cells and
the one-coarse-cell halo of :class:`refine.upsample.Window` on every side.
All arrays of a level are that extended work array (``NE x NE``).

Per level:

1. inputs: the planet's climate, hardness, momentum and metric upsampled
   (:func:`refine.upsample.upsample_window`); the surface, sediment,
   discharge and lake depth from the level above -- bicubic / bilinear --
   or from the planet for the first level (soillib's multiscale procedure:
   a long run at the coarse level, then a short one per finer level);
2. detail noise below the parent's cell (the upsample has no relief there);
3. inflow at the work array's edge: where the parent's drainage crosses it
   (the planet's ``flow_dir`` / ``flow_acc``, or the parent level's own
   flood tree and flux) the rain volume of everything upstream enters as
   extra spawn weight;
4. tiles: the product is split into square cores of at most ``tile`` fine
   cells, each eroded in a window of its core plus ``margin`` on every side.
   Tiles run in passes whose windows do not overlap (the four index
   parities), each pass across worker processes; water crossing into a
   tile -- the flux of the work array's flood tree as the earlier passes
   left it -- spawns where it crosses (``ErosionState.inflow_volume``).  A
   tile writes its new cells in the core and the inner half of its margin,
   and cross-fades cells an earlier pass wrote back from that result;
5. each tile is held to its parent while it erodes (an uplift rate at
   ``hold_scale`` parent cells); after the tiles a last smooth drift
   correction, a local flood capped by the parent's lakes, and the flux on
   the final surface for the level below.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from numba import njit
from scipy import ndimage
from scipy.ndimage import map_coordinates

from ..config import WorldParams
from ..erosion import particle as pk
from ..erosion.maps import ErosionState, step
from ..hydro.priority_flood import priority_flood_flat
from ..io.world_store import WorldStore
from ..refine import basin_job as bj
from ..refine.upsample import Window, detail_noise, ridged_fbm, upsample_window
from ..refine.zoom import ZOOM_REFINE, smooth_drift, zoom_params

#: D8 offsets of the hydro stage's ``flow_dir`` codes (0..7; 8 and above = sink)
D8 = np.array([(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)], dtype=np.int64)
#: rng sub-key of zoom bakes: ``params.rng("refine", ZOOM_KEY, face, i, j, R)``
ZOOM_KEY = 7_700_001
#: a tile spawns at most this many times the particles its rain alone would
#: get when inflow adds to the spawn weight (beyond it the spawn volume grows)
MAX_INFLOW_PARTICLES = 3.0


@dataclass(frozen=True)
class ZoomLevel:
    R: int  # refinement of the planet grid (a power of two)
    cells: int  # product side, coarse cells
    iterations: int
    tile: int = 1024  # tile core side at most, fine cells
    margin: int = 64  # tile overlap on every side, fine cells
    chain_detail: float = 0.5  # detail noise below the parent's cell, x min(parent slope x parent cell, parent 3x3 relief)
    hold_every: int = 10  # hold each tile to its parent while it erodes, as an uplift rate reset every this many iterations (0 = only the correction after the tiles)
    hold_scale: float = 4.0  # the drift correction's Gaussian sigma, in parent cells: below it a level may reshape its parent


#: 1.2 km over 469 km, 305 m over 156 km, 76 m over 78 km on the earth preset
#: (~13 min on 20 threads)
DEFAULT_LEVELS = (
    ZoomLevel(8, 48, 200),
    ZoomLevel(32, 16, 400),
    ZoomLevel(128, 8, 150),
)


@dataclass(frozen=True)
class Geometry:
    face: int
    ci0: int  # product origin, coarse cells
    cj0: int
    cells: int
    guard: int
    R: int

    @property
    def win(self) -> Window:
        g = self.guard
        return Window(self.face, self.ci0 - g, self.ci0 + self.cells + g, self.cj0 - g, self.cj0 + self.cells + g, self.R)

    @property
    def NE(self) -> int:
        return (self.cells + 2 * self.guard + 2) * self.R

    @property
    def p0(self) -> int:
        """Work-array index of the product's first fine row / column."""
        return (self.guard + 1) * self.R

    @property
    def n(self) -> int:
        return self.cells * self.R

    @property
    def origin(self) -> tuple[int, int]:
        """Coarse coordinates of the work array's corner."""
        return self.ci0 - self.guard - 1, self.cj0 - self.guard - 1

    def product(self) -> tuple[slice, slice]:
        return slice(self.p0, self.p0 + self.n), slice(self.p0, self.p0 + self.n)


def place(face: int, ci: int, cj: int, level: ZoomLevel, N: int) -> Geometry:
    """The level's region centred on coarse cell ``(ci, cj)`` of ``face``,
    shifted onto the face if it would cross an edge (a zoom stays on one
    face: the planet's drainage it takes inflow from is read per face)."""
    guard = max(1, math.ceil((level.margin + 2) / level.R))
    lo = guard + 2
    hi = int(N) - level.cells - guard - 2
    if hi < lo:
        raise ValueError(f"a {level.cells}-cell zoom level does not fit on a {N}-cell face")
    ci0 = min(max(int(ci) - level.cells // 2, lo), hi)
    cj0 = min(max(int(cj) - level.cells // 2, lo), hi)
    return Geometry(int(face), ci0, cj0, int(level.cells), guard, int(level.R))


@dataclass
class LevelResult:
    geo: Geometry
    arrays: dict  # work arrays: height, sediment, discharge, momentum (NE, NE, 2), water_surface, flux, plain, ocean, done
    stats: dict = field(default_factory=dict)

    def surface(self) -> np.ndarray:
        return self.arrays["height"] + self.arrays["sediment"]


# --------------------------------------------------------------------------
# drainage
# --------------------------------------------------------------------------
@njit(cache=True)
def _accumulate(pop_seq, parent, w):
    acc = w.copy()
    for k in range(pop_seq.size - 1, -1, -1):
        c = pop_seq[k]
        p = parent[c]
        if p >= 0:
            acc[p] += acc[c]
    return acc


def drainage(surface: np.ndarray, ocean: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flood tree of ``surface`` draining to the ocean and the array border
    (every cell reached): the receiver of each cell (flat index, -1 on a
    drain) and ``weight`` accumulated down it."""
    drain = np.asarray(ocean, bool).copy()
    drain[0, :] = drain[-1, :] = drain[:, 0] = drain[:, -1] = True
    fr = priority_flood_flat(np.asarray(surface, np.float32), drain, None)
    acc = _accumulate(fr.pop_seq, fr.parent, np.asarray(weight, np.float64).ravel())
    return fr.parent, acc.reshape(surface.shape)


def spawn_at(ci: np.ndarray, cj: np.ndarray, vol: np.ndarray, surface: np.ndarray, radius: int) -> np.ndarray:
    """Spawn weight (``surface.shape``) of inflows entering at work-array
    coordinates ``(ci, cj)``: each at the lowest cell of ``surface`` within
    ``radius``, never on the array's one-cell border (a drain)."""
    NE0, NE1 = surface.shape
    out = np.zeros(surface.shape, dtype=np.float64)
    r = max(int(radius), 0)
    for a, b, v in zip(np.rint(ci).astype(np.int64), np.rint(cj).astype(np.int64), vol):
        i0, i1 = max(a - r, 1), min(a + r + 1, NE0 - 1)
        j0, j1 = max(b - r, 1), min(b + r + 1, NE1 - 1)
        if i0 >= i1 or j0 >= j1 or not (v > 0.0):
            continue
        blk = surface[i0:i1, j0:j1]
        k = int(np.argmin(blk))
        out[i0 + k // blk.shape[1], j0 + k % blk.shape[1]] += float(v)
    return out


def planet_inflow(root: Path, geo: Geometry, surface: np.ndarray) -> np.ndarray:
    """Spawn weight of the water the planet's drainage (hydro ``flow_dir`` /
    ``flow_acc``, precip volume) carries into the work array: every coarse
    cell outside it whose D8 receiver is inside sends its whole
    accumulation, entering at the lowest cell next to the crossing."""
    fd = np.load(Path(root) / "coarse" / f"flow_dir.f{geo.face}.npy").astype(np.int64)
    acc = np.load(Path(root) / "coarse" / f"flow_acc.f{geo.face}.npy").astype(np.float64)
    N = fd.shape[0]
    oi, oj = geo.origin
    nc = geo.NE // geo.R
    ii, jj = np.meshgrid(np.arange(max(oi - 1, 0), min(oi + nc + 1, N)), np.arange(max(oj - 1, 0), min(oj + nc + 1, N)), indexing="ij")

    def inside(a, b):
        return (a >= oi) & (a < oi + nc) & (b >= oj) & (b < oj + nc)

    code = fd[ii, jj]
    k = np.minimum(code, 7)
    ti, tj = ii + D8[k, 0], jj + D8[k, 1]
    cross = (code < 8) & ~inside(ii, jj) & inside(ti, tj)
    R = geo.R
    # enter at the crossing: midway between the donor's and the receiver's centres
    ci = ((ii[cross] + ti[cross]) / 2.0 + 0.5 - oi) * R - 0.5
    cj = ((jj[cross] + tj[cross]) / 2.0 + 0.5 - oj) * R - 0.5
    return spawn_at(ci, cj, acc[ii[cross], jj[cross]], surface, R // 2)


def level_inflow(parent: LevelResult, parent_recv: np.ndarray, geo: Geometry, surface: np.ndarray) -> np.ndarray:
    """:func:`planet_inflow` from a parent level: its cells outside ``geo``'s
    work array whose flood-tree receiver is inside send their flux."""
    pg = parent.geo
    NEp, NEc = pg.NE, geo.NE
    f = geo.R // pg.R
    # parent cell centres in child work-array coordinates
    e = np.arange(NEp) + 0.5
    ci_1 = ((pg.origin[0] + e / pg.R) - geo.origin[0]) * geo.R - 0.5
    cj_1 = ((pg.origin[1] + e / pg.R) - geo.origin[1]) * geo.R - 0.5
    in_i = (ci_1 >= 1.0) & (ci_1 <= NEc - 2.0)
    in_j = (cj_1 >= 1.0) & (cj_1 <= NEc - 2.0)
    inside = (in_i[:, None] & in_j[None, :]).ravel()
    recv = parent_recv
    ok = recv >= 0
    cross = ok & ~inside & inside[np.where(ok, recv, 0)]
    d = np.flatnonzero(cross)
    r = recv[d]
    di, dj = np.divmod(d, NEp)
    ri, rj = np.divmod(r, NEp)
    ci = (ci_1[di] + ci_1[ri]) / 2.0
    cj = (cj_1[dj] + cj_1[rj]) / 2.0
    return spawn_at(ci, cj, parent.arrays["flux"].ravel()[d], surface, max(f // 2, 1))


# --------------------------------------------------------------------------
# inputs of a level
# --------------------------------------------------------------------------
def _child_coords(geo: Geometry, pg: Geometry) -> tuple[np.ndarray, np.ndarray]:
    e = np.arange(geo.NE) + 0.5
    pi = ((geo.origin[0] + e / geo.R) - pg.origin[0]) * pg.R - 0.5
    pj = ((geo.origin[1] + e / geo.R) - pg.origin[1]) * pg.R - 0.5
    lo = 1.0
    if pi.min() < lo or pj.min() < lo or pi.max() > pg.NE - 1 - lo or pj.max() > pg.NE - 1 - lo:
        raise ValueError(f"zoom level R={geo.R} is not inside its parent R={pg.R}: place the spot further from the face edge")
    I, J = np.meshgrid(pi, pj, indexing="ij")
    return I, J


def _relief3(surface: np.ndarray) -> np.ndarray:
    return ndimage.maximum_filter(surface, size=3) - ndimage.minimum_filter(surface, size=3)


def _ocean(surface: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Cells below sea level connected to ``seed`` (the parent's ocean)."""
    below = surface < 0.0
    lab, n = ndimage.label(below, structure=np.ones((3, 3), bool))
    if n == 0:
        return np.zeros(surface.shape, bool)
    keep = np.zeros(n + 1, bool)
    keep[np.unique(lab[below & seed])] = True
    keep[0] = False
    return keep[lab]


def level_inputs(root: Path, params: WorldParams, geo: Geometry, level: ZoomLevel, parent: LevelResult | None, gen: np.random.Generator) -> dict:
    grid, fields, derived = bj.coarse_inputs(root, params)
    win = geo.win
    up = upsample_window(fields, derived, win, grid)
    assert up["height0"].shape == (geo.NE, geo.NE), (up["height0"].shape, geo)
    coast_taper = float(params.refine.coast_taper_m)
    if parent is None:
        plain = (up["height0"] + up["sediment0"]).astype(np.float64)
        sed0 = up["sediment0"].astype(np.float64)
        discharge = up["discharge"].astype(np.float64)
        depth = up["depth"].astype(np.float64)
        ocean = _ocean(plain, up["basin_id"] < 0)
        momentum = up["momentum"].astype(np.float64)
        noise = detail_noise(win, up["slope"], up["relief"], up["hardness"], float(ZOOM_REFINE["detail_amp"]), grid.cell_size_m, gen,
                             surface=plain, coast_taper_m=coast_taper).astype(np.float64)
        inflow = planet_inflow(root, geo, plain)
        f = geo.R
    else:
        pg = parent.geo
        pa = parent.arrays
        f = geo.R // pg.R
        I, J = _child_coords(geo, pg)
        psurf = parent.surface().astype(np.float64)
        plain = map_coordinates(psurf, [I, J], order=3, mode="nearest")
        sed0 = np.maximum(map_coordinates(pa["sediment"].astype(np.float64), [I, J], order=1, mode="nearest"), 0.0)
        discharge = np.maximum(map_coordinates(pa["discharge"].astype(np.float64), [I, J], order=1, mode="nearest"), 0.0)
        # momentum from the same level as the discharge: the push divides one
        # by the other, and the planet's momentum over a finer level's
        # discharge shoved every particle along the planet's flow direction
        # (straight parallel tracks across a whole 76 m window)
        momentum = np.stack([map_coordinates(pa["momentum"][..., k].astype(np.float64), [I, J], order=1, mode="nearest") for k in (0, 1)], -1)
        depth = np.maximum(map_coordinates(np.maximum(pa["water_surface"] - psurf, 0.0), [I, J], order=1, mode="nearest"), 0.0)
        seed = map_coordinates(pa["ocean"].astype(np.float32), [I, J], order=0, mode="nearest") > 0.5
        ocean = _ocean(plain, seed)
        parent_cell = grid.cell_size_m / pg.R
        gy, gx = np.gradient(psurf, parent_cell)
        slope = map_coordinates(np.hypot(gx, gy), [I, J], order=1, mode="nearest")
        relief = map_coordinates(_relief3(psurf), [I, J], order=1, mode="nearest")
        amp = float(level.chain_detail) * np.minimum(np.maximum(slope, 0.0) * parent_cell, np.maximum(relief, 0.0)) * (0.5 + 0.5 * np.clip(up["hardness"], 0.0, 1.0))
        if coast_taper > 0.0:
            t = np.clip(np.abs(plain) / coast_taper, 0.0, 1.0)
            amp = amp * t * t * (3.0 - 2.0 * t)
        noise = amp * ridged_fbm((geo.NE, geo.NE), 2.0 * f, gen)
        del I, J, gy, gx, slope, relief, amp
        recv, _ = drainage(psurf, pa["ocean"], np.zeros(psurf.shape))
        inflow = level_inflow(parent, recv, geo, plain)
        del recv
    height0 = plain - sed0
    noise = np.where(ocean, 0.0, noise)
    precip = np.where(ocean, 0.0, np.maximum(up["precip"], 0.0)).astype(np.float64)
    return {
        "plain": plain, "height": height0 + noise, "sediment": sed0, "discharge": discharge, "depth": depth, "ocean": ocean,
        "precip": precip, "inflow": inflow, "evap": up["evap"], "hardness": up["hardness"], "momentum": momentum,
        "metric": up["metric"], "metric_inv": up["metric_inv"], "f": f, "noise_max_m": float(np.abs(noise).max()),
    }


# --------------------------------------------------------------------------
# tiles
# --------------------------------------------------------------------------
def held_p90(delta: np.ndarray, cells: np.ndarray, block: int, prod: tuple[slice, slice]) -> float:
    """90th percentile of ``|mean(delta)|`` over the ``block``-cell squares of
    the product at least 3/4 ``cells`` -- how far a level stands from its
    parent a few parent cells out (metres)."""
    d, c = delta[prod], cells[prod]
    n = (d.shape[0] // block) * block
    if n == 0:
        return 0.0
    sm = np.where(c, d, 0.0)[:n, :n].reshape(n // block, block, n // block, block).sum(axis=(1, 3))
    cnt = c[:n, :n].reshape(n // block, block, n // block, block).sum(axis=(1, 3))
    full = cnt >= 0.75 * block * block
    return float(np.percentile(np.abs(sm[full] / cnt[full]), 90)) if full.any() else 0.0


def _lake_share(surface: np.ndarray, ocean: np.ndarray, prod: tuple[slice, slice], min_depth: float) -> float:
    drain = ocean.copy()
    drain[0, :] = drain[-1, :] = drain[:, 0] = drain[:, -1] = True
    fr = priority_flood_flat(surface.astype(np.float32), drain, None)
    lake = (fr.filled.reshape(surface.shape) - surface > min_depth) & ~ocean
    return float(lake[prod].mean())


def tile_starts(p0: int, n: int, tile: int) -> tuple[list[int], int]:
    """Core starts (work-array index) and core side covering ``[p0, p0 + n)``
    with cores of at most ``tile`` cells (the last core is shifted back to
    end at the product edge, overlapping its neighbour)."""
    nt = max(1, math.ceil(n / max(int(tile), 1)))
    c = math.ceil(n / nt)
    return [min(p0 + k * c, p0 + n - c) for k in range(nt)], c


def tile_windows(starts: list[int], c: int, m: int) -> list[tuple[int, int, int, int]]:
    """``(ti, tj, a, b)`` for every tile core ``[a, a + c) x [b, b + c)``."""
    return [(ti, tj, a, b) for ti, a in enumerate(starts) for tj, b in enumerate(starts)]


def tile_passes(windows: list[tuple[int, int, int, int]], c: int, m: int) -> list[list[tuple[int, int, int, int]]]:
    """Tiles grouped into passes whose windows (core + ``m`` on every side)
    do not overlap, so a pass can run in parallel with the same result as in
    order: the four parities of the tile indices (neighbours of one parity
    are a core apart), a tile that would overlap one of its parity in a pass
    of its own."""
    def apart(w1, w2):
        return (w1[2] + c + m <= w2[2] - m or w2[2] + c + m <= w1[2] - m or
                w1[3] + c + m <= w2[3] - m or w2[3] + c + m <= w1[3] - m)

    passes = [[w for w in windows if (w[0] % 2, w[1] % 2) == par] for par in ((0, 0), (0, 1), (1, 0), (1, 1))]
    out = []
    for ps in passes:
        # the last core is shifted back to end at the product edge: where that
        # brings two of a parity together, they run in turn
        group: list = []
        for w in ps:
            if all(apart(w, x) for x in group):
                group.append(w)
            else:
                out.append([w])
        if group:
            out.append(group)
    return out


def prepare_tile(level: ZoomLevel, geo: Geometry, inp: dict, cur: dict, flux_w: np.ndarray, recv: np.ndarray, a: int, b: int, c: int) -> dict | None:
    """The arrays one tile's erosion needs (copies, so a worker can take
    them), or None when the window has no land to erode.  The kernel array
    is the window plus a one-cell ring outside it (mask 0); the window's land
    is active except along the coast; water crossing into the active cells
    -- the flux of the work array's flood tree -- spawns where it crosses."""
    m = int(level.margin)
    NE = geo.NE
    wa0, wa1 = a - m, a + c + m
    wb0, wb1 = b - m, b + c + m
    assert wa0 >= 1 and wb0 >= 1 and wa1 <= NE - 1 and wb1 <= NE - 1, (a, b, c, m, NE)
    sl = (slice(wa0 - 1, wa1 + 1), slice(wb0 - 1, wb1 + 1))
    ocean = inp["ocean"][sl]
    inwin = np.zeros(ocean.shape, bool)
    inwin[1:-1, 1:-1] = True
    land = inwin & ~ocean
    frozen = land & ndimage.binary_dilation(ocean & inwin, structure=np.ones((3, 3), bool))
    active = land & ~frozen
    if not active.any():
        return None
    act_w = np.zeros((NE, NE), bool)
    act_w[sl] = active
    act_flat = act_w.ravel()
    ok = recv >= 0
    rc = np.where(ok, recv, 0)
    cross = ok & ~act_flat & act_flat[rc]
    src = np.zeros(NE * NE)
    np.add.at(src, recv[cross], flux_w.ravel()[cross])
    src = src.reshape(NE, NE)[sl]
    arr = {k: np.array(cur[k][sl]) for k in ("height", "sediment", "discharge", "momentum")}
    arr.update({k: np.array(inp[k][sl]) for k in ("hardness", "precip", "evap", "metric", "metric_inv", "plain")})
    return {"a": a, "b": b, "c": c, "sl": sl, "land": land, "active": active, "inwin": inwin, "ocean": ocean, "src": src, "arrays": arr, "f": int(inp["f"])}


def erode_tile(params: WorldParams, level: ZoomLevel, R: int, job: dict, key: tuple) -> dict:
    """Erode one prepared tile (runs in a worker): returns its height,
    sediment (metres), discharge and momentum over the kernel array, and
    stats.  ``key`` seeds the tile's own particle stream, so the result does
    not depend on which tiles run beside it."""
    arr = job["arrays"]
    active, land = job["active"], job["land"]
    mask = np.zeros(active.shape, np.uint8)
    mask[land] = pk.MASK_FROZEN
    mask[active] = pk.MASK_ACTIVE
    src = job["src"]
    rain = np.where(active, arr["precip"], 0.0)
    inflow = float(src[active].sum())
    weight = (rain + np.where(active, src, 0.0)).astype(np.float32)
    rain_total = float(rain.sum())
    ppc = float(params.refine.particles_per_cell) * (min(1.0 + inflow / rain_total, MAX_INFLOW_PARTICLES) if rain_total > 0 else 1.0)
    unit = params.fine_grid().cell_size_m
    state = ErosionState.window(
        arr["height"], arr["sediment"], arr["discharge"], arr["momentum"], arr["hardness"], weight,
        arr["evap"], np.zeros(active.shape), mask, arr["metric"], arr["metric_inv"], 1, unit,
    )
    state.inflow_volume = inflow
    ep = bj.basin_erosion_params(params, params.rng("refine", *key))
    t0 = time.time()
    deaths = {k: 0 for k in pk.DEATH_NAMES}
    particles = 0
    plain_t = arr["plain"]
    hold = int(level.hold_every)
    sigma = max(1, int(round(job["f"] * level.hold_scale)))
    off_prev = np.zeros(active.shape)
    for it in range(int(level.iterations)):
        st = step(state, ep, it, particles_per_cell=ppc, rng_stage="refine")
        particles += int(st.get("particles", 0))
        for k, v in (st.get("deaths") or {}).items():
            deaths[k] += int(v)
        if hold > 0 and (it + 1) % hold == 0:
            # hold the tile to its parent while it erodes, as an uplift rate:
            # the offset now, and how fast it grew over the last `hold`
            # iterations under the uplift already applied, set the rate that
            # brings it to zero by the next hold if the erosion keeps its pace
            # (docs/zoom-windows.md, "Pools from the drift correction")
            off = np.where(active, smooth_drift((state.height[0] + state.sediment[0]) * unit - plain_t, active, sigma), 0.0)
            state.uplift[0] -= np.where(active, (2.0 * off - off_prev) / (hold * unit), 0.0)
            off_prev = off
    tot = max(sum(deaths.values()), 1)
    stats = {"core": [job["a"], job["b"], job["c"]], "active_cells": int(active.sum()), "inflow": inflow, "rain": rain_total,
             "particles_per_cell": ppc, "particles": particles, "seconds": round(time.time() - t0, 1),
             "deaths_pct": {k: round(100.0 * v / tot, 1) for k, v in deaths.items() if v}}
    return {"height": state.height_m()[0], "sediment": state.sediment_m()[0], "discharge": state.discharge[0].copy(),
            "momentum": state.momentum[0].copy(), "stats": stats}


def write_tile(level: ZoomLevel, cur: dict, done: np.ndarray, job: dict, out: dict) -> int:
    """Write an eroded tile back into ``cur``: new cells of the core and the
    inner half of the margin as they are; cells an earlier tile wrote
    cross-faded, from the earlier result at the window edge (where this
    tile's own edge shows) to this tile's a margin in.  Freezing them instead
    cut every river that flows into an earlier tile at the strip (the kernel
    tracks no discharge on frozen cells).  Returns the blended cell count."""
    m = int(level.margin)
    sl, land, inwin, ocean = job["sl"], job["land"], job["inwin"], job["ocean"]
    side = land.shape[0] - 2
    e = np.arange(land.shape[0]) - 1
    de = np.minimum(e, side - 1 - e)                       # cells in from the window edge
    d = np.minimum(de[:, None], de[None, :])
    t = np.clip((d + 0.5) / max(m, 1), 0.0, 1.0)
    wgt = t * t * (3.0 - 2.0 * t)
    old = done[sl]
    new = land & ~old & (d >= m - m // 2)
    blend = land & old
    for name in ("height", "sediment", "discharge", "momentum"):
        arr = out[name]
        view = cur[name][sl]
        view[new] = arr[new]
        wb = wgt[blend] if arr.ndim == 2 else wgt[blend][:, None]
        view[blend] = wb * arr[blend] + (1.0 - wb) * view[blend]
    done[sl] |= new | (inwin & ocean)
    return int(blend.sum())


def _pool_init(threads: int) -> None:
    import numba

    numba.set_num_threads(max(1, min(int(threads), numba.config.NUMBA_NUM_THREADS)))


def _erode_job(args):
    return erode_tile(*args)


# --------------------------------------------------------------------------
# a level
# --------------------------------------------------------------------------
def pool_size(jobs: int, workers: int = 0) -> tuple[int, int]:
    """``(worker processes, numba threads each)`` for a pass of ``jobs``
    tiles: ``workers`` (0 = one per 4 cores, the most one kernel process
    uses well), never more than the jobs; the cores are split evenly."""
    import os

    cpus = os.cpu_count() or 1
    n = int(workers) if int(workers) > 0 else max(1, cpus // 4)
    n = max(1, min(n, int(jobs)))
    return n, max(1, cpus // n)


def run_level(root: Path, params: WorldParams, spot: tuple[int, int, int], level: ZoomLevel, parent: LevelResult | None, log=None,
              erosion: dict | None = None, workers: int = 0) -> LevelResult:
    root = Path(root)
    t0 = time.time()
    face, ci, cj = spot
    N = params.coarse_grid().N
    geo = place(face, ci, cj, level, N)
    lp = zoom_params(params, level.R, **(erosion or {}))
    gen = params.rng("refine", ZOOM_KEY, face, ci, cj, level.R)
    inp = level_inputs(root, lp, geo, level, parent, gen)
    t_in = time.time() - t0
    cur = {"height": inp["height"].copy(), "sediment": inp["sediment"].copy(), "discharge": inp["discharge"].copy(), "momentum": inp["momentum"].copy()}
    done = np.zeros((geo.NE, geo.NE), bool)
    starts, c = tile_starts(geo.p0, geo.n, level.tile)
    windows = tile_windows(starts, c, level.margin)
    passes = tile_passes(windows, c, level.margin)
    tiles = []
    weight = inp["precip"] + inp["inflow"]
    widest = max(len(ps) for ps in passes)
    n_workers, threads = pool_size(widest, workers)
    pool = None
    if n_workers > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        pool = ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn"), initializer=_pool_init, initargs=(threads,))
    try:
        for ps in passes:
            # the flood tree of the surface as it stands: the passes before
            # this one have moved the rivers it takes inflow from
            recv, flux_w = drainage(cur["height"] + cur["sediment"], inp["ocean"], weight)
            jobs = [(w, prepare_tile(level, geo, inp, cur, flux_w, recv, w[2], w[3], c)) for w in ps]
            jobs = [(w, jb) for w, jb in jobs if jb is not None]
            args = [(lp, level, geo.R, jb, (ZOOM_KEY, face, ci, cj, level.R, w[0], w[1])) for w, jb in jobs]
            outs = list(pool.map(_erode_job, args)) if pool is not None and len(args) > 1 else [_erode_job(x) for x in args]
            for (w, jb), out in zip(jobs, outs):
                st = out["stats"]
                st["blended_cells"] = write_tile(level, cur, done, jb, out)
                tiles.append(st)
                if log is not None:
                    log(f"  R={geo.R} tile {len(tiles)}/{len(windows)}: {st['active_cells']:,} cells, inflow {st['inflow']:.1f} "
                        f"of rain {st['rain']:.1f}, {st['seconds']:.0f}s, deaths {st['deaths_pct']}")
    finally:
        if pool is not None:
            pool.shutdown()
    t_tiles = time.time() - t0 - t_in
    # hold the level to its parent at the parent's cell
    plain = inp["plain"]
    height, sediment = cur["height"], cur["sediment"]
    cells = done & ~inp["ocean"]
    held_before = held_p90((height + sediment) - plain, cells, 4 * inp["f"], geo.product())
    lake_before = _lake_share(height + sediment, inp["ocean"], geo.product(), float(params.hydro.lake_min_depth))
    drift = smooth_drift((height + sediment) - plain, cells, max(1, int(round(inp["f"] * level.hold_scale))))
    fd = np.where(cells, drift, 0.0)
    lower = np.maximum(fd, 0.0)
    take = np.minimum(sediment, lower)
    sediment -= take
    height -= lower - take
    height -= np.minimum(fd, 0.0)
    surface = height + sediment
    held_after = held_p90(surface - plain, cells, 4 * inp["f"], geo.product())
    # water: a local flood, no higher than the parent's lakes where it has one
    drain = inp["ocean"].copy()
    drain[0, :] = drain[-1, :] = drain[:, 0] = drain[:, -1] = True
    ws = bj.local_flood(surface.astype(np.float32), drain, np.ones(drain.shape, bool), np.ones(drain.shape, np.uint8)).astype(np.float64)
    lake0 = inp["depth"] > 0.0
    ws = np.where(lake0, np.maximum(surface, np.minimum(ws, plain + inp["depth"])), ws)
    ws = np.where(inp["ocean"], np.maximum(0.0, surface), ws)
    _, flux = drainage(surface, inp["ocean"], weight)
    arrays = {
        "height": height.astype(np.float32), "sediment": sediment.astype(np.float32), "discharge": cur["discharge"].astype(np.float32),
        "momentum": cur["momentum"].astype(np.float32), "water_surface": ws.astype(np.float32), "flux": flux.astype(np.float32), "plain": plain.astype(np.float32),
        "ocean": inp["ocean"], "done": done,
    }
    prod = geo.product()
    lake = (ws - surface > float(params.hydro.lake_min_depth)) & ~inp["ocean"]
    stats = {
        "R": geo.R, "cell_m": params.coarse_grid().cell_size_m / geo.R, "geometry": asdict(geo), "level": asdict(level), "tiles": tiles,
        "seconds": round(time.time() - t0, 1), "seconds_inputs": round(t_in, 1), "seconds_tiles": round(t_tiles, 1),
        "passes": len(passes), "workers": n_workers,
        "noise_max_m": round(inp["noise_max_m"], 1), "drift_max_m": round(float(np.abs(fd).max()), 1),
        "held_p90_m": [round(held_before, 2), round(held_after, 2)], "lake_share_before_hold": round(lake_before, 4),
        "lake_share": round(float(lake[prod].mean()), 4), "change_std_m": round(float((surface - plain)[geo.product()][cells[geo.product()]].std()), 2) if cells.any() else 0.0,
        "inflow_total": float(inp["inflow"].sum()), "lake_cells_product": int(lake[prod].sum()),
        "rain_cell": float(inp["precip"][~inp["ocean"]].mean()) if (~inp["ocean"]).any() else 0.0,
        "relief_m": [float(surface[prod].min()), float(surface[prod].max())],
    }
    return LevelResult(geo, arrays, stats)


# --------------------------------------------------------------------------
# a zoom
# --------------------------------------------------------------------------
def spot_of_lonlat(params: WorldParams, lat_deg: float, lon_deg: float) -> tuple[int, int, int]:
    """(face, i, j) coarse cell under a latitude / longitude (pole on +Z, as
    ``Grid.latitude``)."""
    from ..cubesphere import from_sphere_v

    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    p = np.array([[math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)]])
    f, u, v = from_sphere_v(p)
    N = params.coarse_grid().N
    return int(f[0]), min(int(float(u[0]) * N), N - 1), min(int(float(v[0]) * N), N - 1)


def lonlat_of_spot(params: WorldParams, spot: tuple[int, int, int]) -> tuple[float, float]:
    """Latitude / longitude (degrees) of a coarse cell's centre."""
    from ..cubesphere import to_sphere_v

    N = params.coarse_grid().N
    face, i, j = spot
    p = to_sphere_v(np.array([face]), np.array([(i + 0.5) / N]), np.array([(j + 0.5) / N]))[0]
    return float(np.degrees(np.arcsin(np.clip(p[2], -1.0, 1.0)))), float(np.degrees(np.arctan2(p[1], p[0])))


def write_views(out: Path, levels, lat: float, lon: float, name: str, max_res: int = 1024) -> None:
    """``L{R}.html`` per level (its product square) and ``view.html`` (the
    finest), each linking back to the globe viewer at the spot."""
    from .view import build_html

    back = f"../../viewer/index.html#lat={lat:.3f}&lon={lon:.3f}&z=24"
    html = None
    for level in levels:
        res = load_level(out, level.R)
        a = res.arrays
        sl = res.geo.product()
        html, _ = build_html(a["height"][sl], a["sediment"][sl], a["discharge"][sl], a["water_surface"][sl], ~a["ocean"][sl],
                             res.stats["cell_m"], title=f"{name} · level R={level.R}", max_res=max_res, ocean=a["ocean"][sl],
                             rain_cell=res.stats.get("rain_cell") or None, back=back, crop=False)
        (Path(out) / f"L{level.R}.html").write_text(html)
    if html is not None:
        (Path(out) / "view.html").write_text(html)


def save_level(res: LevelResult, out: Path) -> Path:
    path = Path(out) / f"L{res.geo.R}.npz"
    np.savez_compressed(path, **{k: (v.astype(np.uint8) if v.dtype == bool else v) for k, v in res.arrays.items()})
    (Path(out) / f"L{res.geo.R}.json").write_text(json.dumps(res.stats, indent=1))
    return path


def load_level(out: Path, R: int) -> LevelResult:
    z = np.load(Path(out) / f"L{R}.npz")
    stats = json.loads((Path(out) / f"L{R}.json").read_text())
    arrays = {k: (z[k] > 0 if k in ("ocean", "done") else z[k]) for k in z.files}
    return LevelResult(Geometry(**stats["geometry"]), arrays, stats)


def run_zoom(root: str | Path, spot: tuple[int, int, int], levels=DEFAULT_LEVELS, out: str | Path | None = None, name: str | None = None,
             log=None, erosion: dict | None = None, resume: bool = True, workers: int = 0) -> Path:
    """Bake a zoom of the world at ``root`` around coarse cell ``spot`` =
    ``(face, i, j)`` into ``out`` (default ``<root>/zoom/<name>``): one
    ``L{R}.npz`` / ``L{R}.json`` per level, ``zoom.json``.  With ``resume``
    a level whose files exist (same spot and level settings) is loaded, not
    re-baked."""
    root = Path(root)
    store = WorldStore(root)
    params = WorldParams.from_dict(store.manifest["params"])
    face, ci, cj = (int(x) for x in spot)
    name = name or f"f{face}_{ci}_{cj}"
    out = Path(out) if out is not None else root / "zoom" / name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    parent = None
    records = []
    for level in levels:
        key = {"spot": [face, ci, cj], "level": asdict(level), "erosion": erosion or {}, "kernel": pk.KERNEL_VERSION}
        meta = out / f"L{level.R}.json"
        if resume and meta.exists() and (out / f"L{level.R}.npz").exists() and json.loads(meta.read_text()).get("key") == key:
            res = load_level(out, level.R)
            if log is not None:
                log(f"R={level.R}: resumed")
        else:
            if log is not None:
                log(f"R={level.R}: {level.cells} coarse cells, {level.iterations} iterations")
            res = run_level(root, params, (face, ci, cj), level, parent, log=log, erosion=erosion, workers=workers)
            res.stats["key"] = key
            save_level(res, out)
            if log is not None:
                log(f"R={level.R}: done in {res.stats['seconds']:.0f}s")
        records.append({"R": level.R, "cell_m": res.stats["cell_m"], "geometry": res.stats["geometry"], "seconds": res.stats["seconds"]})
        parent = res
    lat, lon = lonlat_of_spot(params, (face, ci, cj))
    write_views(out, levels, lat, lon, name)
    info = {"name": name, "world": root.name, "spot": [face, ci, cj], "lat": lat, "lon": lon, "levels": records,
            "seconds": round(time.time() - t0, 1), "kernel": pk.KERNEL_VERSION, "erosion": erosion or {}}
    (out / "zoom.json").write_text(json.dumps(info, indent=1))
    return out


__all__ = ["ZoomLevel", "DEFAULT_LEVELS", "Geometry", "LevelResult", "place", "drainage", "planet_inflow", "level_inflow",
           "level_inputs", "tile_starts", "tile_passes", "prepare_tile", "erode_tile", "write_tile", "pool_size", "run_level", "run_zoom", "spot_of_lonlat", "lonlat_of_spot", "save_level", "load_level"]
