"""Zoom bake: a square region around a spot refined level by level, each
level tiled and chained from the one above (docs/zoom-windows.md, "The zoom
stage").

A level is a square *product* of ``cells`` coarse cells around the spot at
refinement ``R``, inside a *work* array that adds ``guard`` coarse cells and
the one-coarse-cell halo of :class:`refine.upsample.Window` on every side.
All arrays of a level are that extended work array (``NE x NE``).  A level
given a ``size`` in fine cells is placed in fine cells instead (a game-scale
level, where a coarse cell is thousands of fine ones): its product is
``size`` fine cells centred on the same point, and its guard
:func:`fine_pad` fine cells (:class:`Geometry`).

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

import dataclasses
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from numba import njit
from scipy import ndimage
from scipy.ndimage import map_coordinates

from ..config import WorldParams
from ..erosion import particle as pk
from ..erosion import vegetation as veg
from ..erosion.maps import ErosionState, step
from ..hydro.priority_flood import priority_flood_flat
from ..io.world_store import WorldStore
from ..refine import basin_job as bj
from ..refine.upsample import FineWindow, Window, detail_noise, ridged_fbm, sample, upsample_window
from ..refine.zoom import ZOOM_REFINE, drain_noise, smooth_drift, zoom_params
from . import progress

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
    size: int = 0  # > 0: the product side in fine cells, placed in fine cells with a guard of fine_pad(margin) fine cells (`cells` unused); 0: `cells` coarse cells
    inflow_cap: float = 0.0  # > 0: a tile's inflow spawns at most this multiple of its rain (see erode_tile); 0: all of it
    snapshots: int = 24  # frames of the level's erosion kept for the viewer's time lapse (0 = none); only a level eroded as one tile has them


#: ZoomLevel fields added after zooms were baked, left out of a level's
#: resume key at their defaults (so existing zooms still resume)
_LEVEL_KEY_DEFAULTS = {"size": 0, "inflow_cap": 0.0, "snapshots": 24}


def level_key(level: ZoomLevel) -> dict:
    """``asdict(level)`` without the later fields at their defaults."""
    d = asdict(level)
    for k, v in _LEVEL_KEY_DEFAULTS.items():
        if d.get(k) == v:
            d.pop(k)
    return d


def fine_pad(margin: int) -> int:
    """Guard of a fine-cell level, fine cells from the work array's edge to
    its product: the tiles' margin and the kernel's ring (``margin + 2``,
    what :func:`prepare_tile` needs), then as much again, so the inflow from
    the parent -- spawned within a few cells of the array's border -- enters
    outside every tile's window and reaches the tiles down the flood tree
    (spawned on an active cell it would be lost)."""
    return 2 * (int(margin) + 2)


def parse_level(spec: str) -> ZoomLevel:
    """``R:cells:iterations[:tile[:margin]][:name=value ...]``
    (``scripts/zoom_bake.py --levels``).  ``cells`` is coarse cells (``8``);
    a fraction of them (``0.5``) or fine cells with an ``f`` (``1024f``)
    places the level in fine cells (``ZoomLevel.size``).  ``name=value``
    sets any other field (``inflow_cap=2``, ``chain_detail=0.5``)."""
    fields = {f.name: f.type for f in dataclasses.fields(ZoomLevel)}
    kw = {}
    parts = []
    for x in spec.split(":"):
        if "=" in x:
            k, _, v = x.partition("=")
            if k not in fields or k in ("R", "cells", "iterations", "size"):
                raise ValueError(f"zoom level {spec!r}: no field {k!r} to set by name")
            kw[k] = float(v) if fields[k] in ("float", float) else int(v)
        else:
            parts.append(x)
    if not 3 <= len(parts) <= 5:
        raise ValueError(f"zoom level {spec!r}: want R:cells:iterations[:tile[:margin]]")
    R = int(parts[0])
    rest = [int(x) for x in parts[2:]]
    c = parts[1].strip().lower()
    if c.endswith("f"):
        size = int(c[:-1])
    elif any(ch in c for ch in ".e"):
        size = float(c) * R
        if size != int(size):
            raise ValueError(f"zoom level {spec!r}: {c} coarse cells is not a whole number of fine cells at R={R}")
        size = int(size)
    else:
        return ZoomLevel(R, int(c), *rest, **kw)
    if size <= 0:
        raise ValueError(f"zoom level {spec!r}: size must be positive")
    return ZoomLevel(R, -(-size // R), *rest, size=size, **kw)


#: 1.2 km over 469 km, 305 m over 156 km, 76 m over 78 km on the earth preset
#: (~13 min on 20 threads)
DEFAULT_LEVELS = (
    ZoomLevel(8, 48, 200),
    ZoomLevel(32, 16, 400),
    ZoomLevel(128, 8, 150),
)

#: game-scale levels to chain below DEFAULT_LEVELS (``zoom_bake.py --game``),
#: placed in fine cells: 19 m over 19.5 km and 4.8 m over 9.8 km on the earth
#: preset, both centred where the levels above are.  Iterations from
#: convergence runs on earth-v9's peaks (5, 212, 902): at 19 m relief,
#: valley depth and slopes move < 5 % from 150 to 300 iterations (valley depth
#: p50 69 -> 73 m); at 5 m (a 1,024-cell core) the added detail is within
#: ~7 % from 50 to 100.  The 5 m level caps its inflow at twice its rain:
#: its window took 46x its rain in inflow, so 98 % of its particles spawned
#: at the crossings and its slopes stayed the 19 m bicubic (holes >0.5 m
#: 5.4 -> 1.6 per 10^4 with the cap, axis/diagonal power at 2-4 cells 2.5 ->
#: 0.9, 6-cell high-pass 6.4 -> 8.0 m against the plain's 5.8)
GAME_LEVELS = (
    ZoomLevel(512, 2, 150, size=1024),
    ZoomLevel(2048, 1, 60, tile=2048, size=2048, inflow_cap=2.0),
)


@dataclass(frozen=True)
class Geometry:
    """Where a level's arrays sit on the face.

    A coarse-cell level (``size`` 0, every level baked before game-scale
    levels): the product is coarse cells ``[ci0, ci0 + cells)`` (and the same
    along j) at ``R``; the work array adds ``guard`` coarse cells and a
    one-coarse-cell halo on every side.

    A fine-cell level (``size`` > 0): the product is face fine cells ``[fi0,
    fi0 + size) x [fj0, fj0 + size)``; the work array adds ``pad`` fine cells
    on every side.  ``ci0`` / ``cj0`` are then the coarse cells holding the
    product's first fine cell, ``cells`` its side in coarse cells rounded up,
    ``guard`` 0 (information only: nothing reads them for such a level).

    Either way, work-array index ``i`` is face fine cell ``fine_origin[0] +
    i``, the product is ``[p0, p0 + n)`` of the ``NE``-cell array, and
    ``product_origin`` = ``fine_origin + p0``."""

    face: int
    ci0: int  # product origin, coarse cells
    cj0: int
    cells: int
    guard: int
    R: int
    size: int = 0  # fine-cell level: product side, fine cells
    fi0: int = 0  # fine-cell level: product origin, face fine cells
    fj0: int = 0
    pad: int = 0  # fine-cell level: work-array cells outside the product on every side

    @property
    def fine(self) -> bool:
        return self.size > 0

    @property
    def win(self) -> Window | FineWindow:
        if self.fine:
            a0, b0 = self.fine_origin
            return FineWindow(self.face, a0, b0, self.NE, self.R)
        g = self.guard
        return Window(self.face, self.ci0 - g, self.ci0 + self.cells + g, self.cj0 - g, self.cj0 + self.cells + g, self.R)

    @property
    def NE(self) -> int:
        if self.fine:
            return self.size + 2 * self.pad
        return (self.cells + 2 * self.guard + 2) * self.R

    @property
    def p0(self) -> int:
        """Work-array index of the product's first fine row / column."""
        return self.pad if self.fine else (self.guard + 1) * self.R

    @property
    def n(self) -> int:
        return self.size if self.fine else self.cells * self.R

    @property
    def product_origin(self) -> tuple[int, int]:
        """Face fine cell of the product's first row / column."""
        if self.fine:
            return self.fi0, self.fj0
        return self.ci0 * self.R, self.cj0 * self.R

    @property
    def fine_origin(self) -> tuple[int, int]:
        """Face fine cell of the work array's corner (index 0, 0)."""
        a, b = self.product_origin
        return a - self.p0, b - self.p0

    @property
    def corner(self) -> tuple[float, float]:
        """Coarse coordinates of the work array's corner (``origin`` as
        floats, fractional for a fine-cell level)."""
        a, b = self.fine_origin
        return a / self.R, b / self.R

    @property
    def origin(self) -> tuple[int, int]:
        """Coarse coordinates of the work array's corner (a coarse-cell level)."""
        if self.fine:
            raise ValueError("a fine-cell level's work array does not start on a coarse cell: use fine_origin / corner")
        return self.ci0 - self.guard - 1, self.cj0 - self.guard - 1

    def product(self) -> tuple[slice, slice]:
        return slice(self.p0, self.p0 + self.n), slice(self.p0, self.p0 + self.n)

    def to_dict(self) -> dict:
        """The fields as saved (``L{R}.json``, ``zoom.json``): a coarse-cell
        level's without the fine-cell fields, as before they existed."""
        d = asdict(self)
        if not self.fine:
            for k in ("size", "fi0", "fj0", "pad"):
                d.pop(k)
        return d


def place(face: int, ci: int, cj: int, level: ZoomLevel, N: int) -> Geometry:
    """The level's region centred on coarse cell ``(ci, cj)`` of ``face``,
    shifted onto the face if it would cross an edge (a zoom stays on one
    face: the planet's drainage it takes inflow from is read per face).

    A fine-cell level (``level.size``) is centred on the corner ``(ci, cj)``
    of the spot's cell, where an even-``cells`` level is centred, and kept a
    coarse cell off the face's edges."""
    if int(level.size) > 0:
        R, s = int(level.R), int(level.size)
        pad = fine_pad(level.margin)
        lo = R + pad
        hi = (int(N) - 1) * R - s - pad
        if hi < lo:
            raise ValueError(f"a {s}-fine-cell zoom level does not fit on a {N}-cell face at R={R}")
        fi0 = min(max(int(ci) * R - s // 2, lo), hi)
        fj0 = min(max(int(cj) * R - s // 2, lo), hi)
        return Geometry(int(face), fi0 // R, fj0 // R, -(-s // R), 0, R, size=s, fi0=fi0, fj0=fj0, pad=pad)
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
    frames: list = field(default_factory=list)  # the time lapse of its erosion (ZoomLevel.snapshots), product only, block-averaged

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


def tile_water(surface: np.ndarray, ocean: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(accumulated weight, standing water depth)`` of a tile's surface: its
    flood tree draining to the ocean and the array border, and how deep the
    fill stands over each cell.  What the vegetation reads for streams and
    pools -- the particles' discharge counts a particle at every cell it
    crosses (hillslope cells came out ~40x their upstream area at 19 m) and
    the routing surface lifts every tiny hollow."""
    drain = np.asarray(ocean, bool).copy()
    drain[0, :] = drain[-1, :] = drain[:, 0] = drain[:, -1] = True
    fr = priority_flood_flat(np.asarray(surface, np.float32), drain, None)
    acc = _accumulate(fr.pop_seq, fr.parent, np.asarray(weight, np.float64).ravel()).reshape(surface.shape)
    depth = np.maximum(fr.filled.reshape(surface.shape) - surface, 0.0)
    return acc, np.where(drain, 0.0, depth)


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
    if geo.fine:
        raise ValueError("a zoom's first level takes the planet's inflow in coarse cells: make it a coarse-cell level (size 0)")
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
    ci_1 = ((pg.corner[0] + e / pg.R) - geo.corner[0]) * geo.R - 0.5
    cj_1 = ((pg.corner[1] + e / pg.R) - geo.corner[1]) * geo.R - 0.5
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
    # coarse coordinates (the corners are dyadic: exact in float64, and for a
    # coarse-cell level the same numbers as its integer origin)
    e = np.arange(geo.NE) + 0.5
    pi = ((geo.corner[0] + e / geo.R) - pg.corner[0]) * pg.R - 0.5
    pj = ((geo.corner[1] + e / geo.R) - pg.corner[1]) * pg.R - 0.5
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


#: the widest a time-lapse frame is stored at (cells a side); a level is block-averaged
#: down to it, so a 2048-cell level's frames are its 512-cell picture
SNAP_MAX = 512


def _snap_factor(side: int) -> int:
    f = 1
    while side // f > SNAP_MAX:
        f *= 2
    return f


def _block_mean(a: np.ndarray, f: int) -> np.ndarray:
    if f <= 1:
        return np.asarray(a, np.float32)
    n0 = (a.shape[0] // f) * f
    n1 = (a.shape[1] // f) * f
    return np.asarray(a[:n0, :n1], np.float32).reshape(n0 // f, f, n1 // f, f).mean(axis=(1, 3)).astype(np.float32)


#: sub-key of a tile's vegetation stream (after its particles' key)
VEG_KEY = 7919
#: iterations between a tile's flood fills for its vegetation
VEG_WATER_EVERY = 10
#: the coarse climate fields the vegetation reads, per world root
_CLIMATE: dict = {}


def climate_inputs(root: Path, params: WorldParams, win, grid) -> dict:
    """``forest`` (the climate's forest, :func:`veg.climate_forest`) and
    ``temp0`` (mean annual temperature at sea level, C: the coarse field's
    lapse undone at the bedrock it was computed on) over a level's work array; None
    where the world has no temperature or biome."""
    from ..derive.biomes import VEG_FACTOR

    key = str(Path(root).resolve())
    if key not in _CLIMATE:
        store = WorldStore(root)
        try:
            _CLIMATE.clear()
            _CLIMATE[key] = {n: store.load_field(n, grid) for n in ("temperature", "biome", "bedrock")}
        except (OSError, KeyError, ValueError):
            _CLIMATE[key] = None
    f = _CLIMATE[key]
    if f is None:
        return None
    # the stored temperature is the climate's, on the pre-erosion bedrock (derive/run.py)
    bed = sample(f["bedrock"], win, order=1)
    temp0 = sample(f["temperature"], win, order=1) + float(params.climate.lapse) * np.maximum(bed, 0.0) / 1000.0
    forest = veg.climate_forest(sample(f["biome"], win), VEG_FACTOR)
    return {"forest": forest.astype(np.float32), "temp0": temp0.astype(np.float32)}


def temperature_at(params: WorldParams, temp0: np.ndarray, surface_m: np.ndarray) -> np.ndarray:
    return temp0 - float(params.climate.lapse) * np.maximum(surface_m, 0.0) / 1000.0


def initial_cover(params: WorldParams, inp: dict, parent: LevelResult | None, geo: Geometry) -> np.ndarray:
    """A level's cover before it erodes: its parent's where the parent grew
    one, else half the climate's forest below the tree line (the first
    iterations take it off the channels, pools and cliffs)."""
    NE = geo.NE
    if parent is not None and "vegetation" in parent.arrays:
        I, J = _child_coords(geo, parent.geo)
        return np.clip(map_coordinates(parent.arrays["vegetation"].astype(np.float32), [I, J], order=1, mode="nearest"), 0.0, 1.0).astype(np.float32)
    if inp.get("forest") is None:
        return np.zeros((NE, NE), np.float32)
    p = veg.VegParams()
    t = temperature_at(params, inp["temp0"], inp["height"] + inp["sediment"])
    warm = np.clip((t - (p.tree_line_c - p.tree_line_fade_c)) / (2.0 * p.tree_line_fade_c), 0.0, 1.0)
    return np.where(inp["ocean"], 0.0, 0.5 * inp["forest"] * warm).astype(np.float32)


def level_inputs(root: Path, params: WorldParams, geo: Geometry, level: ZoomLevel, parent: LevelResult | None, gen: np.random.Generator) -> dict:
    grid, fields, derived = bj.coarse_inputs(root, params)
    win = geo.win
    up = upsample_window(fields, derived, win, grid)
    clim = climate_inputs(root, params, win, grid)
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
    noise = noise + drain_noise(height0 + noise + sed0, ocean)      # the noise's own basins, filled before anything erodes
    precip = np.where(ocean, 0.0, np.maximum(up["precip"], 0.0)).astype(np.float64)
    return {
        "plain": plain, "height": height0 + noise, "sediment": sed0, "discharge": discharge, "depth": depth, "ocean": ocean,
        "precip": precip, "inflow": inflow, "evap": up["evap"], "hardness": up["hardness"], "momentum": momentum,
        "metric": up["metric"], "metric_inv": up["metric_inv"], "f": f, "noise_max_m": float(np.abs(noise).max()),
        "forest": None if clim is None else clim["forest"], "temp0": None if clim is None else clim["temp0"],
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
    arr = {k: np.array(cur[k][sl]) for k in ("height", "sediment", "discharge", "momentum", "vegetation") if k in cur}
    arr.update({k: np.array(inp[k][sl]) for k in ("hardness", "precip", "evap", "metric", "metric_inv", "plain", "forest", "temp0") if inp.get(k) is not None})
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
    inflow = inflow_raw = float(src[active].sum())
    rain_total = float(rain.sum())
    cap = float(level.inflow_cap)
    if cap > 0.0 and rain_total > 0.0 and inflow > cap * rain_total:
        # a particle erodes the same whatever water it stands for (its change
        # scales with its volume over the spawn volume, 1 at birth), so where
        # the inflow is many times the tile's rain nearly every particle
        # spawns at a few crossings and the slopes go unworn: a 4.9 km 5 m
        # level took 43x its rain in inflow and 3 % of its particles fell on
        # its slopes.  The cap spawns the inflow as at most `cap` x the rain
        # (its crossings keep their shares; the flux the finish routes keeps
        # all of it)
        src = src * (cap * rain_total / inflow)
        inflow = cap * rain_total
    weight = (rain + np.where(active, src, 0.0)).astype(np.float32)
    ppc = float(params.refine.particles_per_cell) * (min(1.0 + inflow / rain_total, MAX_INFLOW_PARTICLES) if rain_total > 0 else 1.0)
    unit = params.fine_grid().cell_size_m
    state = ErosionState.window(
        arr["height"], arr["sediment"], arr["discharge"], arr["momentum"], arr["hardness"], weight,
        arr["evap"], np.zeros(active.shape), mask, arr["metric"], arr["metric_inv"], 1, unit,
    )
    state.inflow_volume = inflow
    ep = bj.basin_erosion_params(params, params.rng("refine", *key))
    vp = job.get("veg")
    grows = vp is not None and "vegetation" in arr and "forest" in arr
    if grows:
        cover = arr["vegetation"].astype(np.float32)
        state.roots = veg.roots(vp, cover)
        vrng = params.rng("refine", *key, VEG_KEY)
        cell_m = float(params.world.cell_size_m) / float(R)
        cell_km2 = (cell_m / 1000.0) ** 2
        rain_cell = max(rain_total / max(float(active.sum()), 1.0), 1e-30)
    t0 = time.time()
    deaths = {k: 0 for k in pk.DEATH_NAMES}
    particles = 0
    plain_t = arr["plain"]
    # the time lapse: the tile's surface and its streams every so many iterations, block-
    # averaged down (SNAP_MAX).  Only a level eroded as one tile keeps them -- tiles run in
    # passes, so two of them are never at the same iteration
    snaps = int(job.get("snapshots", 0))
    # spread over the whole run (the first iteration, then evenly to the last)
    snap_at = {0} | {int(round((k + 1) * int(level.iterations) / snaps)) - 1 for k in range(snaps)} if snaps > 0 else set()
    frames: list = []
    drain = job["ocean"] | ~job["inwin"]      # where a tile's water leaves it, for the flood tree
    prod = job.get("prod")
    hold = int(level.hold_every)
    sigma = max(1, int(round(job["f"] * level.hold_scale)))
    off_prev = np.zeros(active.shape)
    demand = float(ppc) * float(active.sum())
    with _Demand(demand):
        for it in range(int(level.iterations)):
            _share_threads(demand)
            st = step(state, ep, it, particles_per_cell=ppc, rng_stage="refine")
            snap = it in snap_at
            surf_m = (state.height[0] + state.sediment[0]) * unit if (grows or snap) else None
            flood = None                      # this iteration's flood tree, where one was taken
            if grows:
                # the trees after the water: the tile's streams (upstream area of its
                # rain and inflow) and pools, every few iterations
                if it % VEG_WATER_EVERY == 0:
                    acc, pool = tile_water(surf_m, drain, weight)
                    flood = acc
                    area = acc / rain_cell * cell_km2
                cap = veg.capacity(vp, arr["forest"], temperature_at(params, arr["temp0"], surf_m), surf_m, pool, area, cell_m, active)
                cover = veg.grow(vp, cover, cap, vrng)
                state.roots = veg.roots(vp, cover)
            progress.tile_tick(int(level.R), int(job["a"]), int(job["b"]), it, int(level.iterations))
            particles += int(st.get("particles", 0))
            for k, v in (st.get("deaths") or {}).items():
                deaths[k] += int(v)
            if snap:
                # the streams of a frame are the flood tree's, as the finished level's are --
                # the particles' own discharge is an average over the iteration's passes, which
                # swells and fades with them, so a time lapse of it pulsed instead of growing
                facc = flood if flood is not None else tile_water(surf_m, drain, weight)[0]
                fsl = prod if prod is not None else (slice(None), slice(None))
                fsurf = surf_m[fsl]
                fac = _snap_factor(fsurf.shape[0])
                frames.append({"it": it + 1, "surface": _block_mean(fsurf, fac),
                               "discharge": _block_mean(facc[fsl], fac), "factor": fac})
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
             **({"inflow_uncapped": inflow_raw} if inflow_raw != inflow else {}),
             "particles_per_cell": ppc, "particles": particles, "seconds": round(time.time() - t0, 1),
             "deaths_pct": {k: round(100.0 * v / tot, 1) for k, v in deaths.items() if v}}
    out = {"height": state.height_m()[0], "sediment": state.sediment_m()[0], "discharge": state.discharge[0].copy(),
           "momentum": state.momentum[0].copy(), "stats": stats}
    if frames:
        out["frames"] = frames
    if grows:
        out["vegetation"] = cover
        land = active
        stats["cover_mean"] = round(float(cover[land].mean()), 3) if land.any() else 0.0
    return out


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
    for name in ("height", "sediment", "discharge", "momentum", "vegetation"):
        if name not in out or name not in cur:
            continue
        arr = out[name]
        view = cur[name][sl]
        view[new] = arr[new]
        wb = wgt[blend] if arr.ndim == 2 else wgt[blend][:, None]
        view[blend] = wb * arr[blend] + (1.0 - wb) * view[blend]
    done[sl] |= new | (inwin & ocean)
    return int(blend.sum())


#: the pool's shared demand (``pool_demand``), set in each worker by ``_pool_init``
_DEMAND = None


def pool_demand():
    """Shared ``[particles per iteration of the tiles running now]`` for
    :func:`_pool_init`: the tiles of a pass differ by 50x in work, and with
    cores split evenly the biggest ran on 4 of 20 while the rest sat idle
    once the small ones were done."""
    import multiprocessing as mp

    return mp.get_context("spawn").Array("d", 1)


def _pool_init(threads: int, demand=None) -> None:
    import numba

    global _DEMAND
    _DEMAND = demand
    numba.set_num_threads(max(1, min(int(threads), numba.config.NUMBA_NUM_THREADS)))


def _share_threads(mine: float) -> None:
    """In a pool worker: take the share of the cores this tile's particles
    are of the running tiles' (at least one thread).  The kernel's results do
    not depend on the thread count, only its speed."""
    if _DEMAND is None:
        return
    import os

    import numba

    total = max(float(_DEMAND[0]), mine, 1.0)
    n = int(round((os.cpu_count() or 1) * mine / total))
    numba.set_num_threads(max(1, min(n, numba.config.NUMBA_NUM_THREADS)))


class _Demand:
    """Add a tile's demand to the pool's while it runs."""

    def __init__(self, amount: float):
        self.amount = float(amount)

    def __enter__(self):
        if _DEMAND is not None:
            with _DEMAND.get_lock():
                _DEMAND[0] += self.amount
        return self

    def __exit__(self, *exc):
        if _DEMAND is not None:
            with _DEMAND.get_lock():
                _DEMAND[0] = max(_DEMAND[0] - self.amount, 0.0)
        return False


def _erode_job(args):
    return erode_tile(*args)


# --------------------------------------------------------------------------
# a level
# --------------------------------------------------------------------------
def pool_size(jobs: int, workers: int = 0) -> tuple[int, int]:
    """``(worker processes, numba threads each to start with)`` for a pass
    of ``jobs`` tiles: ``workers`` (0 = one per core), never more than the
    jobs.  Once running, a worker's threads follow its tile's share of the
    particles (:func:`_share_threads`).  A process per tile beats threads:
    the change lists are applied serially (a 1152² land tile at R = 8: 17
    CPU-s of tracing an iteration, 3.4 s of applying), so a core does ~1.5x
    the work on its own tile as one of four on a shared one."""
    import os

    cpus = os.cpu_count() or 1
    n = int(workers) if int(workers) > 0 else cpus
    n = max(1, min(n, int(jobs)))
    return n, max(1, cpus // n)


def run_level(root: Path, params: WorldParams, spot: tuple[int, int, int], level: ZoomLevel, parent: LevelResult | None, log=None,
              erosion: dict | None = None, workers: int = 0) -> LevelResult:
    root = Path(root)
    t0 = time.time()
    face, ci, cj = spot
    N = params.coarse_grid().N
    geo = place(face, ci, cj, level, N)
    if geo.fine and parent is not None and geo.p0 - int(level.margin) <= level.R // parent.geo.R + 2:
        # the parent's inflow enters within about a parent cell of the border
        raise ValueError(f"R={level.R}: a guard of {geo.p0} fine cells leaves no room between the inflow and the tiles' margin {level.margin}")
    erosion = dict(erosion or {})
    vp = veg.VegParams(**erosion.pop("vegetation")) if isinstance(erosion.get("vegetation"), dict) else \
        (None if erosion.pop("vegetation", True) is False else veg.VegParams())
    lp = zoom_params(params, level.R, **erosion)
    gen = params.rng("refine", ZOOM_KEY, face, ci, cj, level.R)
    inp = level_inputs(root, lp, geo, level, parent, gen)
    t_in = time.time() - t0
    cur = {"height": inp["height"].copy(), "sediment": inp["sediment"].copy(), "discharge": inp["discharge"].copy(), "momentum": inp["momentum"].copy()}
    if vp is not None and inp.get("forest") is not None:
        cur["vegetation"] = initial_cover(params, inp, parent, geo)
    done = np.zeros((geo.NE, geo.NE), bool)
    starts, c = tile_starts(geo.p0, geo.n, level.tile)
    windows = tile_windows(starts, c, level.margin)
    passes = tile_passes(windows, c, level.margin)
    tiles = []
    level_frames: list = []
    weight = inp["precip"] + inp["inflow"]
    widest = max(len(ps) for ps in passes)
    n_workers, threads = pool_size(widest, workers)
    pool = None
    if n_workers > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        pool = ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn"), initializer=_pool_init, initargs=(threads, pool_demand()))
    try:
        for ps in passes:
            # the flood tree of the surface as it stands: the passes before
            # this one have moved the rivers it takes inflow from
            recv, flux_w = drainage(cur["height"] + cur["sediment"], inp["ocean"], weight)
            jobs = [(w, prepare_tile(level, geo, inp, cur, flux_w, recv, w[2], w[3], c)) for w in ps]
            jobs = [(w, jb) for w, jb in jobs if jb is not None]
            for _, jb in jobs:
                jb["veg"] = vp
                if len(windows) == 1 and int(level.snapshots) > 0:
                    # the product, in the tile's own array: only a level of one tile is a time lapse
                    off = jb["sl"][0].start
                    jb["snapshots"] = int(level.snapshots)
                    jb["prod"] = (slice(geo.p0 - off, geo.p0 + geo.n - off), slice(geo.p0 - off, geo.p0 + geo.n - off))
            args = [(lp, level, geo.R, jb, (ZOOM_KEY, face, ci, cj, level.R, w[0], w[1])) for w, jb in jobs]
            outs = list(pool.map(_erode_job, args)) if pool is not None and len(args) > 1 else [_erode_job(x) for x in args]
            for (w, jb), out in zip(jobs, outs):
                frames = out.get("frames") or []
                st = out["stats"]
                st["blended_cells"] = write_tile(level, cur, done, jb, out)
                tiles.append(st)
                level_frames = frames or level_frames
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
    if "vegetation" in cur:
        arrays["vegetation"] = np.where(inp["ocean"], 0.0, cur["vegetation"]).astype(np.float32)
    rain_cell = float(inp["precip"][~inp["ocean"]].mean()) if (~inp["ocean"]).any() else 0.0
    _classify(root, lp, geo, arrays, params.coarse_grid().cell_size_m / geo.R, rain_cell)
    prod = geo.product()
    lake = (ws - surface > float(params.hydro.lake_min_depth)) & ~inp["ocean"]
    stats = {
        "R": geo.R, "cell_m": params.coarse_grid().cell_size_m / geo.R, "geometry": geo.to_dict(), "level": level_key(level), "tiles": tiles,
        "seconds": round(time.time() - t0, 1), "seconds_inputs": round(t_in, 1), "seconds_tiles": round(t_tiles, 1),
        "passes": len(passes), "workers": n_workers,
        "noise_max_m": round(inp["noise_max_m"], 1), "drift_max_m": round(float(np.abs(fd).max()), 1),
        "held_p90_m": [round(held_before, 2), round(held_after, 2)], "lake_share_before_hold": round(lake_before, 4),
        "lake_share": round(float(lake[prod].mean()), 4), "change_std_m": round(float((surface - plain)[geo.product()][cells[geo.product()]].std()), 2) if cells.any() else 0.0,
        "inflow_total": float(inp["inflow"].sum()), "lake_cells_product": int(lake[prod].sum()),
        "rain_cell": float(inp["precip"][~inp["ocean"]].mean()) if (~inp["ocean"]).any() else 0.0,
        "relief_m": [float(surface[prod].min()), float(surface[prod].max())],
    }
    return LevelResult(geo, arrays, stats, level_frames)


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


#: how far in the globe viewer's 3-D view a zoom opens (`view.html`)
VIEW_ZOOM = 12000
#: its tilt from straight down, degrees
VIEW_TILT = 62


def view_link(lat: float, lon: float) -> str:
    """The globe viewer's 3-D view over a zoom: its satellite ground, trees and
    water, the zoom's own levels drawn where they cover."""
    return (f"../../viewer/index.html#v=tilt&lat={lat:.4f}&lon={lon:.4f}&z={VIEW_ZOOM}"
            f"&tl={VIEW_TILT}&hd=0&x3=1&layer=satellite")


def write_views(out: Path, levels, lat: float, lon: float, name: str, max_res: int = 1024) -> None:
    """``L{R}.html`` per level (its own square as a lit mesh, self-contained)
    and ``view.html``, which opens the globe viewer's 3-D view over the zoom --
    the same satellite ground, trees and water as everywhere else, drawn from
    the levels' own textures, rather than a second renderer of its own."""
    from .view import build_html

    back = view_link(lat, lon)      # the viewer over this window, not the planet
    for level in levels:
        res = load_level(out, level.R)
        a = res.arrays
        sl = res.geo.product()
        html, _ = build_html(a["height"][sl], a["sediment"][sl], a["discharge"][sl], a["water_surface"][sl], ~a["ocean"][sl],
                             res.stats["cell_m"], title=f"{name} · level R={level.R}", max_res=max_res, ocean=a["ocean"][sl],
                             rain_cell=res.stats.get("rain_cell") or None, back=back, crop=False)
        (Path(out) / f"L{level.R}.html").write_text(html)
    link = view_link(lat, lon)
    levels_html = " · ".join(f'<a href="L{level.R}.html">R={level.R}</a>' for level in levels)
    (Path(out) / "view.html").write_text(
        "<!doctype html><meta charset=utf-8><title>%s</title>"
        "<meta http-equiv=refresh content='0; url=%s'>"
        "<body style='background:#0b0e16;color:#dfe6f2;font:14px system-ui;margin:2rem'>"
        "<p>Opening <a href='%s'>%s in the globe viewer's 3-D view</a>.</p>"
        "<p>This zoom's levels as self-contained meshes: %s</p>" % (name, link, link, name, levels_html))


def planet_dir(root: str | Path, R: int) -> Path | None:
    """The finished planet level at ``R`` (``globe.zoom.planet``), if there is one."""
    d = Path(root) / "zoom" / f"planet_R{int(R)}"
    if not (d / "planet.json").exists():
        return None
    return d if int(json.loads((d / "planet.json").read_text())["level"]["R"]) == int(R) else None


def level_from_planet(root: Path, params: WorldParams, spot: tuple[int, int, int], level: ZoomLevel, pdir: Path) -> LevelResult:
    """A zoom's first level cut from the planet level at its ``R`` instead of
    eroded again: the planet's height, sediment, discharge and water surface
    over the level's work array (a zoom stays on one face, and its work array
    inside it), the momentum of the planet's work raster, and the plain
    upsample, ocean and flux the next level chains from.  The planet level
    ran fewer iterations than a zoom's own first level (80 against 200 at
    R = 8), so the relief it hands down is younger."""
    from . import planet as zp

    t0 = time.time()
    face, ci, cj = spot
    N = params.coarse_grid().N
    geo = place(face, ci, cj, level, N)
    lp = zoom_params(params, level.R)
    inp = level_inputs(root, lp, geo, level, None, params.rng("refine", ZOOM_KEY, face, ci, cj, level.R))
    info = json.loads((pdir / "planet.json").read_text())
    R, NE = geo.R, geo.NE
    a0, b0 = geo.fine_origin
    if a0 < 0 or b0 < 0 or a0 + NE > N * R or b0 + NE > N * R:
        raise ValueError(f"zoom level R={R} at {spot} leaves face {face}")
    sl = (slice(a0, a0 + NE), slice(b0, b0 + NE))
    arr = {k: np.array(np.load(zp.out_path(pdir, R, face, k), mmap_mode="r")[sl], np.float64) for k in zp.OUT_FIELDS}
    g = (int(info["level"]["margin"]) // R + 1) * R
    mom = np.load(zp.work_path(pdir, face, "momentum"), mmap_mode="r")
    momentum = np.array(mom[a0 + g:a0 + g + NE, b0 + g:b0 + g + NE], np.float64)
    surface = arr["height"] + arr["sediment"]
    ocean = inp["ocean"]
    _, flux = drainage(surface, ocean, inp["precip"] + inp["inflow"])
    prod = geo.product()
    ws = np.maximum(arr["water_surface"], surface)
    lake = (ws - surface > float(params.hydro.lake_min_depth)) & ~ocean
    arrays = {"height": arr["height"].astype(np.float32), "sediment": arr["sediment"].astype(np.float32), "discharge": arr["discharge"].astype(np.float32),
              "momentum": momentum.astype(np.float32), "water_surface": ws.astype(np.float32), "flux": flux.astype(np.float32),
              "plain": inp["plain"].astype(np.float32), "ocean": ocean, "done": np.ones((NE, NE), bool)}
    if inp.get("forest") is not None:
        arrays["vegetation"] = initial_cover(params, {**inp, "height": arr["height"], "sediment": arr["sediment"]}, None, geo)
    _classify(root, lp, geo, arrays, params.coarse_grid().cell_size_m / R, float(inp["precip"][~ocean].mean()) if (~ocean).any() else 0.0)
    stats = {"R": R, "cell_m": params.coarse_grid().cell_size_m / R, "geometry": geo.to_dict(), "level": level_key(level), "tiles": [],
             "source": "planet", "planet": pdir.name, "planet_iterations": int(info["level"]["iterations"]),
             "seconds": round(time.time() - t0, 1), "inflow_total": float(inp["inflow"].sum()),
             "lake_share": round(float(lake[prod].mean()), 4), "lake_cells_product": int(lake[prod].sum()),
             "rain_cell": float(inp["precip"][~ocean].mean()) if (~ocean).any() else 0.0,
             "relief_m": [float(surface[prod].min()), float(surface[prod].max())]}
    return LevelResult(geo, arrays, stats)


def _classify(root: Path, params: WorldParams, geo: Geometry, arrays: dict, cell_m: float, rain_cell: float) -> None:
    """A level's own biomes (globe.zoom.biomes) and the life of its lakes
    (globe.zoom.lakes) into ``arrays``, unless the world has no climate."""
    from .biomes import level_biomes
    from .lakes import lake_life

    clim = climate_inputs(root, params, geo.win, params.coarse_grid())
    if rain_cell <= 0.0 or clim is None:
        return
    arrays["biome"] = level_biomes(root, params, geo, arrays, cell_m, rain_cell)
    surf = np.asarray(arrays["height"], np.float32) + np.asarray(arrays["sediment"], np.float32)
    life = lake_life(surf, arrays["water_surface"], arrays["ocean"], arrays["sediment"], arrays["flux"],
                     temperature_at(params, clim["temp0"], surf), cell_m, rain_cell, float(params.hydro.lake_min_depth))
    arrays["lake_age"] = life["age"]
    arrays["lake_emergent"] = life["emergent"]
    arrays["lake_submerged"] = life["submerged"]


def save_level(res: LevelResult, out: Path) -> Path:
    path = Path(out) / f"L{res.geo.R}.npz"
    np.savez_compressed(path, **{k: (v.astype(np.uint8) if v.dtype == bool else v) for k, v in res.arrays.items()})
    (Path(out) / f"L{res.geo.R}.json").write_text(json.dumps(res.stats, indent=1))
    if res.frames:
        # the time lapse: the product's surface and streams every few iterations
        np.savez_compressed(Path(out) / f"L{res.geo.R}.frames.npz",
                            surface=np.stack([f["surface"] for f in res.frames]),
                            discharge=np.stack([f["discharge"] for f in res.frames]),
                            iterations=np.array([f["it"] for f in res.frames], np.int32),
                            factor=np.array([f["factor"] for f in res.frames], np.int32))
    return path


def frames_path(out: Path, R: int) -> Path:
    return Path(out) / f"L{int(R)}.frames.npz"


def load_level(out: Path, R: int) -> LevelResult:
    z = np.load(Path(out) / f"L{R}.npz")
    stats = json.loads((Path(out) / f"L{R}.json").read_text())
    arrays = {k: (z[k] > 0 if k in ("ocean", "done") else z[k]) for k in z.files}
    return LevelResult(Geometry(**stats["geometry"]), arrays, stats)


def run_zoom(root: str | Path, spot: tuple[int, int, int], levels=DEFAULT_LEVELS, out: str | Path | None = None, name: str | None = None,
             log=None, erosion: dict | None = None, resume: bool = True, workers: int = 0, planet: bool | None = None) -> Path:
    """Bake a zoom of the world at ``root`` around coarse cell ``spot`` =
    ``(face, i, j)`` into ``out`` (default ``<root>/zoom/<name>``): one
    ``L{R}.npz`` / ``L{R}.json`` per level, ``zoom.json``.  With ``resume``
    a level whose files exist (same spot and level settings) is loaded, not
    re-baked.  The first level is cut from the planet level at its ``R``
    (:func:`level_from_planet`) when ``planet`` is True, or None and a
    finished one exists; False always erodes it."""
    root = Path(root)
    store = WorldStore(root)
    params = WorldParams.from_dict(store.manifest["params"])
    face, ci, cj = (int(x) for x in spot)
    name = name or f"f{face}_{ci}_{cj}"
    out = Path(out) if out is not None else root / "zoom" / name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # progress for whoever watches (globe/zoom/progress.py): the plan, and the
    # directory the tiles' workers tick into
    N = params.coarse_grid().N
    plan = []
    for k, level in enumerate(levels):
        cut = k == 0 and planet is not False and planet_dir(root, level.R) is not None
        try:
            tiles = len(tile_starts(place(face, ci, cj, level, N).p0, place(face, ci, cj, level, N).n, level.tile)[0]) ** 2
        except (ValueError, AttributeError):
            tiles = 1
        plan.append({"R": int(level.R), "work": 0.0 if cut else progress.level_work(level, tiles), "tiles": tiles, "state": "pending"})
    env_before = os.environ.get(progress.ENV)
    os.environ[progress.ENV] = str(progress.write_plan(out, plan))
    try:
        return _run_zoom_levels(root, store, params, (face, ci, cj), levels, out, name, log, erosion, resume, workers, planet, t0)
    finally:
        if env_before is None:
            os.environ.pop(progress.ENV, None)
        else:
            os.environ[progress.ENV] = env_before


def _run_zoom_levels(root, store, params, spot, levels, out, name, log, erosion, resume, workers, planet, t0) -> Path:
    face, ci, cj = spot
    parent = None
    records = []
    for level in levels:
        progress.set_level(out, int(level.R), "running")
        pdir = planet_dir(root, level.R) if parent is None and planet is not False else None
        if planet is True and parent is None and pdir is None:
            raise FileNotFoundError(f"no finished planet level at R={level.R} under {root / 'zoom'}")
        key = {"spot": [face, ci, cj], "level": level_key(level), "erosion": erosion or {}, "kernel": pk.KERNEL_VERSION}
        if pdir is not None:
            key["planet"] = [pdir.name, (pdir / "planet.json").stat().st_mtime_ns]
        meta = out / f"L{level.R}.json"
        if resume and meta.exists() and (out / f"L{level.R}.npz").exists() and json.loads(meta.read_text()).get("key") == key:
            res = load_level(out, level.R)
            if log is not None:
                log(f"R={level.R}: resumed")
        else:
            if pdir is not None:
                if log is not None:
                    log(f"R={level.R}: {level.cells} coarse cells, from {pdir.name}")
                res = level_from_planet(root, params, (face, ci, cj), level, pdir)
            else:
                if log is not None:
                    log(f"R={level.R}: " + (f"{level.size} fine cells" if level.size else f"{level.cells} coarse cells") + f", {level.iterations} iterations")
                res = run_level(root, params, (face, ci, cj), level, parent, log=log, erosion=erosion, workers=workers)
            res.stats["key"] = key
            save_level(res, out)
            if log is not None:
                log(f"R={level.R}: done in {res.stats['seconds']:.0f}s")
        records.append({"R": level.R, "cell_m": res.stats["cell_m"], "geometry": res.stats["geometry"], "seconds": res.stats["seconds"]})
        progress.set_level(out, int(level.R), "done")
        parent = res
    lat, lon = lonlat_of_spot(params, (face, ci, cj))
    write_views(out, levels, lat, lon, name)
    info = {"name": name, "world": root.name, "spot": [face, ci, cj], "lat": lat, "lon": lon, "levels": records,
            "seconds": round(time.time() - t0, 1), "kernel": pk.KERNEL_VERSION, "erosion": erosion or {}}
    (out / "zoom.json").write_text(json.dumps(info, indent=1))
    return out


__all__ = ["ZoomLevel", "DEFAULT_LEVELS", "GAME_LEVELS", "Geometry", "LevelResult", "place", "fine_pad", "level_key", "parse_level", "drainage", "planet_inflow", "level_inflow",
           "level_inputs", "tile_starts", "tile_passes", "prepare_tile", "erode_tile", "write_tile", "pool_size", "run_level", "planet_dir", "level_from_planet", "run_zoom", "spot_of_lonlat", "lonlat_of_spot", "save_level", "load_level"]
