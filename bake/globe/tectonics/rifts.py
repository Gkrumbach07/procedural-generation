"""Rift grabens (``tectonics.rift_graben_m``): the narrow basin along a continental rift that
is still opening, drawn on the coarse grid at its own width.

Why this is not crust in the cloud.  A continental rift here is a plate boundary, and the
crust knows of it only once the halves have drawn a spacing apart: the gap then takes a
margin segment and the zone ``margin_width_km`` either side thins with it
(``collision.spawn_segments``), which the splat draws one spacing (160 km) wide.  So a rift
goes from no mark at all to a sag several hundred km across that floats below sea level --
an arm of the sea -- and never passes through the stage Earth's deep lakes stand in: a
fault-bounded trough 50-80 km wide, its floor a kilometre or more under shoulders that are
still dry land (Baikal, Tanganyika, Malawi).  Measured on earth-v25 (seed 1423) across its
youngest continental rift: the bedrock at the axis stood a median 118 m *above* the lower
shoulder over 25 land sections, the uplift field's subsidence there was ~550 km wide, and
the stage that followed left no lake cell within 60 km of it.  At 160 km between segments no
reconstruction of the cloud can show a 50 km trough, for the reason it cannot show a 20 km
cone (``volcanoes``), so the graben is the same kind of thing the cones and the arc ridge
are: a reconstruction detail finalise stamps at its own size, where the simulation says a
rift is.

Why it is all subsidence.  A narrow low handed to the erosion stage in its starting surface
does not survive it: the strike valleys ``ranges`` puts in the belts are laid level by
``erosion.maps.fill_basins`` and then lost altogether (earth-v25: what stood more than 200 m
below its surroundings at the 40 km scale in the bedrock kept 0.00 of that in the final
surface).  What a rift lake needs is a floor that goes on sinking faster than its rivers
fill it.  So finalise puts the graben in ``bedrock`` *and* the whole of it in ``uplift``
(negative): the erosion stage's replay start (``uplift_mode`` 'replay') is the ground
before the graben, with nothing there for the basin fill to level, and the trough opens
under the rivers over the stage.

Where.  Along the continental contacts of the plate pairs that are rifting at the last
step: the pairs still held by their rift's strength (``TectonicSim.rift_pairs``, the slow
phase), as far as they have opened, and the halves of a rift that broke through within
``rift_free_my`` (``TectonicSim.rift_free``), wherever the ground on the axis is still
land.  A rift that failed and healed (``intraplate.heal_failed_rifts``) is one plate again
and leaves no trace to read, so it gets none, and one held shut on its way to failing has
not opened and gets none either; Baikal's own kind of rift -- slow, 30 My old and 10-20 km
open -- is the case this leaves out wherever the model's version of it has healed.
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from scipy.spatial import cKDTree

from .segments import CONTINENTAL
from .volcanoes import _chord, _flat_centers

# Earth's numbers.  Lengths are km on the tectonic sphere (a toy body's grabens keep their
# footprint relative to the plates, like its cones); the depth is the parameter's.  A pair is
# the range a basin's own number is drawn from: a chain of identical basins reads as a dashed
# line, and no two of Earth's are alike.
#: the floor's width: Baikal is 50-80 km across, Tanganyika ~50, Malawi ~75, Albert and Turkana 30-40
FLOOR_KM = (40.0, 70.0)
#: floor to shoulder: the border faults' scarps, a cell or two on the Earth grid
WALL_KM = 15.0
#: the shoulders' rise as a share of the depth: rift flanks stand 300-1000 m over the plateau
#: beside them, and it is their tilt away from the trough that keeps its rivers short
SHOULDER = 0.15
#: e-folding width of the shoulder away from the scarp (the flexural flank uplift is 50-100 km wide)
SHOULDER_KM = 60.0
#: ...and how many of those the shoulder is drawn out to, where it is taken smoothly to zero
SHOULDER_REACH = 4.0
#: a basin's length (log-uniform): a rift is a chain of basins, not one trench -- Albert 160 km,
#: Turkana 290, Malawi 580, Baikal 640, Tanganyika 670 -- and land axis shorter than the least holds none
BASIN_KM = (150.0, 600.0)
#: the high ground between two basins of a chain (an accommodation zone): what makes each a
#: closed basin rather than one valley draining along the rift to the sea
SILL_KM = (40.0, 120.0)
#: a basin's floor comes up to the sill over this length at either end
END_KM = 50.0
#: a basin's share of the parameter's depth (Tanganyika's water is 1,470 m deep, Malawi's 700):
#: the parameter is the deepest a basin gets
DEPTH_SHARE = (0.7, 1.0)
#: a basin stands up to this far to one side of the rift's line (basins step en echelon, their
#: border faults on alternate sides)...
ECHELON_KM = 20.0
#: ...and bows up to this far off it in the middle: the lakes are arcs (Tanganyika, Malawi,
#: Baikal), and a chain of straight slots along one smooth line reads as drawn with a ruler
BOW_KM = 25.0
#: the opening at which a rift's grabens have their depth, less in proportion below it: the
#: floor goes down as the crust is pulled apart (Baikal's basement fell 8-9 km over 10-20 km of
#: extension), and a rift held shut by its strength -- one about to fail and heal -- has none
OPEN_KM = 20.0
#: ...and a rift with less than this share of the depth to its name is left out
SHARE_MIN = 0.05
#: the axis counts as land where the ground stands this far above sea level: lower ground
#: beside a rift is the shelf its stretching has already drowned
LAND_MIN_M = 200.0
#: the axis is sampled this often (half a cell on the Earth grid)
AXIS_STEP_KM = 5.0
#: the contacts' midpoints are smoothed along the rift over this many spacings: they sit a
#: spacing apart and step from side to side with the cloud, and a fault does not
AXIS_SMOOTH = 1.0
#: an axis sample belongs to the rift only within this many spacings of a contact
AXIS_NEAR = 1.0
#: two continental segments of the halves are in contact within this many spacings, with no
#: segment between them: a rift in its slow phase has opened up to rift_break_factor x
#: rift_weaken_km (300 km), past the census' 1.25
CONTACT_REACH = 2.0
#: sub-key of the stream the basins' own numbers are drawn from (params.rng("tectonics", ...)):
#: a stream of its own, so every other draw of the stage is bit for bit what it was
RNG_KEY = 23


def rifting_pairs(sim) -> list[tuple[int, int, float]]:
    """``(a, b, share)`` for the plate pairs rifting at the last step: the rifts still held
    by their strength, and the halves of one that broke through within ``rift_free_my``
    (the pair itself is not kept once it breaks: its halves are the two plates that carry
    the same cut step and the same breakthrough step).  ``share`` is the share of the depth
    the rift has earned, at most 1: how much of the uplift window it has been open for --
    the graben sinks at one rate, and a rift cut halfway through the window has had half
    the time -- times how far it has opened towards ``OPEN_KM``.  A rift below
    ``SHARE_MIN`` is not in the list."""
    tp, plates = sim.tp, sim.plates
    k = int(sim.step_index)
    window = max(k - int(sim.ref_step), 1)

    def alive(q):
        return 0 <= q < plates.P and bool(plates.alive[q])

    out = []
    for (a, b), st in sim.rift_pairs.items():
        if alive(a) and alive(b):
            share = min(1.0, max(k - int(st.get("k0", k)), 0) / window) * min(1.0, float(st.get("delta", 0.0)) / sim.km(OPEN_KM))
            if share >= SHARE_MIN:
                out.append((int(a), int(b), share))
    held = {frozenset(pr) for pr in sim.rift_pairs}
    by_step: dict[tuple[int, int], list[int]] = {}
    for q, k_free in sim.rift_free.items():
        if alive(q) and k - int(k_free) <= sim.steps_of(tp.rift_free_my):
            by_step.setdefault((int(k_free), int(sim.last_rift.get(q, -1))), []).append(int(q))
    for qs in by_step.values():
        for a, b in combinations(sorted(qs), 2):
            if frozenset((a, b)) not in held:
                out.append((a, b, 1.0))               # it has been through the whole slow phase, and opened rift_break_factor x rift_weaken_km
    return out


def rift_contacts(seg, a: int, b: int, spacing: float, tree: cKDTree | None = None) -> np.ndarray:
    """Midpoints (K, 3) of the continental contacts between plates ``a`` and ``b``: each
    continental segment of one with the nearest continental segment of the other, where
    they lie within ``CONTACT_REACH`` spacings and nothing stands between them (the cloud's
    nearest segment to the midpoint is one of the two) -- so sea floor the rift has already
    opened, or a third plate's crust, ends the contact."""
    cont = seg.kind == CONTINENTAL
    ia = np.flatnonzero(cont & (seg.plate_id == a))
    ib = np.flatnonzero(cont & (seg.plate_id == b))
    if ia.size == 0 or ib.size == 0:
        return np.zeros((0, 3))
    r = float(_chord(CONTACT_REACH * spacing))
    da, ja = cKDTree(seg.pos[ib]).query(seg.pos[ia], distance_upper_bound=r)
    db, jb = cKDTree(seg.pos[ia]).query(seg.pos[ib], distance_upper_bound=r)
    fa, fb = np.isfinite(da), np.isfinite(db)
    pr = np.concatenate([np.stack([ia[fa], ib[ja[fa]]], axis=1), np.stack([ia[jb[fb]], ib[fb]], axis=1)])
    if pr.shape[0] == 0:
        return np.zeros((0, 3))
    pr = np.unique(pr, axis=0)
    mid = seg.pos[pr[:, 0]] + seg.pos[pr[:, 1]]
    mid /= np.linalg.norm(mid, axis=1, keepdims=True)
    tree = cKDTree(seg.pos) if tree is None else tree
    _, near = tree.query(mid)
    return mid[(near == pr[:, 0]) | (near == pr[:, 1])]


def rift_axis(mid: np.ndarray, spacing: float, step: float) -> list[np.ndarray]:
    """The contacts' midpoints as smooth lines: a list of (n, 3) runs of unit vectors
    ``step`` radians apart.

    A rift is cut along a great circle (``intraplate._force_rift_one``), so the midpoints
    are a graph over the great circle that fits them best: their offset from it is smoothed
    along it (a Gaussian of ``AXIS_SMOOTH`` spacings) and the line is cut wherever no
    contact lies within ``AXIS_NEAR`` spacings of it -- a stretch the rift has already
    opened into sea floor, or a place the smoothed line has left its contacts."""
    if mid.shape[0] < 3:
        return []
    _, v = np.linalg.eigh(mid.T @ mid)
    n = v[:, 0]                                             # the pole of the great circle through the contacts
    c = mid.mean(axis=0)
    e1 = c - n * float(c @ n)
    if float(np.linalg.norm(e1)) < 1e-9:
        return []
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    th = np.arctan2(mid @ e2, mid @ e1)                     # along the rift, 0 at its middle
    ph = np.arcsin(np.clip(mid @ n, -1.0, 1.0))             # off the great circle
    g = np.arange(float(th.min()), float(th.max()) + 0.5 * step, step)
    k = np.exp(-0.5 * ((g[:, None] - th[None, :]) / (AXIS_SMOOTH * spacing)) ** 2)
    p = (k @ ph) / np.maximum(k.sum(axis=1), 1e-300)
    pts = np.cos(p)[:, None] * (np.cos(g)[:, None] * e1 + np.sin(g)[:, None] * e2) + np.sin(p)[:, None] * n
    near, _ = cKDTree(mid).query(pts)
    on = near <= float(_chord(AXIS_NEAR * spacing))
    return [pts[i0:i1] for i0, i1 in _runs(on) if i1 - i0 >= 2]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``[i0, i1)`` of every run of True in a 1-D mask."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.flatnonzero(m[1:] != m[:-1])
    return list(zip(d[0::2].tolist(), d[1::2].tolist()))


def basins(s: np.ndarray, land: np.ndarray, rng, lengths: tuple[float, float], sills: tuple[float, float],
           end: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """The basins of one axis run: ``(sample indices, depth share 0-1 at each)``.

    ``s`` is the arc length at the run's samples and ``land`` where the ground on the axis is
    land.  Every stretch of land is walked from one end: a basin of a length drawn
    log-uniformly from ``lengths`` (or what land is left, if that is less and still a
    basin's), then a sill drawn from ``sills``, while the land lasts.  Each basin's floor
    comes up to nothing over ``end`` at either end, so a chain is closed basins strung along
    the rift."""
    lo, hi = float(lengths[0]), float(lengths[1])
    out = []
    for i0, i1 in _runs(land):
        at, stop = float(s[i0]), float(s[i1 - 1])
        while stop - at >= lo:
            each = min(lo * (hi / lo) ** float(rng.random()), stop - at)
            idx = i0 + np.flatnonzero((s[i0:i1] >= at) & (s[i0:i1] <= at + each))
            if idx.size >= 2:
                u = np.clip(np.minimum(s[idx] - at, at + each - s[idx]) / max(min(end, 0.5 * each), 1e-12), 0.0, 1.0)
                out.append((idx, u * u * (3.0 - 2.0 * u)))
            at += each + float(rng.uniform(float(sills[0]), float(sills[1])))
    return out


def _aside(line: np.ndarray, off: np.ndarray) -> np.ndarray:
    """``line`` (n, 3 unit vectors) with each point moved ``off`` radians to the left of the
    line's own direction."""
    b = np.cross(line, np.gradient(line, axis=0))
    b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-300)
    out = line + np.asarray(off)[:, None] * b
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def profile(d: np.ndarray, half, wall: float, lam: float) -> np.ndarray:
    """The graben across its strike, per unit depth, at distance ``d`` from the axis: -1 on
    the floor (``half`` either side, a number or one per ``d``), up the scarp over ``wall``
    to the shoulder's ``SHOULDER``, and from there down to nothing with e-folding ``lam``."""
    floor = np.where(d <= half, 1.0, np.where(d < half + wall, 0.5 * (1.0 + np.cos(math.pi * (d - half) / wall)), 0.0))
    t = math.exp(-SHOULDER_REACH)
    rim = np.clip((np.exp(-np.maximum(d - half - wall, 0.0) / lam) - t) / (1.0 - t), 0.0, 1.0)
    return -floor + SHOULDER * (1.0 - floor) * rim


def graben_field(sim, grid, bed: np.ndarray, depth_m: float, vfac: float = 1.0, ctree=None) -> tuple[np.ndarray, dict]:
    """The grabens on ``grid`` (metres, (6, N, N)): negative on the floors, positive on the
    shoulders, zero away from every rift.

    ``bed`` is the bedrock it is stamped into (metres, sea level 0), read only for where the
    axis is land; ``depth_m`` is ``tectonics.rift_graben_m`` and ``vfac`` the planet's
    vertical scale over Earth's (as the cones': ``volcanoes.stamp``).  Each cell takes the
    profile of the basin whose axis is nearest, so two basins never stack.  A basin's length,
    floor width, share of the depth and step to one side of the line are its own, drawn from
    the ranges above (a stream of the stage's own, ``RNG_KEY``).  The floor and the scarp
    are never narrower than a cell: below that a cell-centre sample misses the trough on
    some rows and the basin is a string of pits.  Returns ``(field, info)``."""
    N = grid.N
    out = np.zeros(6 * N * N)
    info = {"rifts": 0, "rifts_on_land": 0, "basins": 0, "length_km": 0.0, "deepest_m": 0.0, "cells": 0}
    pairs = rifting_pairs(sim)
    info["rifts"] = len(pairs)
    if depth_m <= 0.0 or not pairs:
        return out.reshape(6, N, N), info
    seg = sim.seg
    km = 1.0 / float(sim.R_km)                              # radians per km on the tectonic sphere
    cell = 0.5 * math.pi / N
    wall = max(WALL_KM * km, cell)
    lam = max(SHOULDER_KM * km, cell)
    step = min(AXIS_STEP_KM * km, 0.5 * cell)
    centers = _flat_centers(grid)
    ctree = cKDTree(centers) if ctree is None else ctree
    flat_bed = np.asarray(bed).reshape(-1)
    tree = cKDTree(seg.pos)
    rng = sim.params.rng("tectonics", RNG_KEY)
    pts, val, halves = [], [], []                           # every basin's axis samples, and the depth and half-floor each carries
    for a, b, share in pairs:
        n_before = len(pts)
        for run in rift_axis(rift_contacts(seg, a, b, sim.spacing, tree), sim.spacing, step):
            _, at = ctree.query(run)
            land = flat_bed[at] > LAND_MIN_M * vfac
            hop = 2.0 * np.arcsin(np.minimum(0.5 * np.linalg.norm(np.diff(run, axis=0), axis=1), 1.0))
            s = np.concatenate([[0.0], np.cumsum(hop)])
            for idx, e in basins(s, land, rng, (BASIN_KM[0] * km, BASIN_KM[1] * km), (SILL_KM[0] * km, SILL_KM[1] * km), END_KM * km):
                deep = share * depth_m * vfac * float(rng.uniform(*DEPTH_SHARE))
                # the basin's own line: to one side of the rift's, and bowed
                t = (s[idx] - s[idx[0]]) / max(float(s[idx[-1]] - s[idx[0]]), 1e-300)
                off = ECHELON_KM * km * float(rng.uniform(-1.0, 1.0)) + BOW_KM * km * float(rng.uniform(-1.0, 1.0)) * np.sin(math.pi * t)
                pts.append(_aside(run[idx], off))
                val.append(deep * e)
                # ...and its floor narrows to the ends as it shallows: a lens, not a slot
                halves.append(np.maximum(0.5 * float(rng.uniform(*FLOOR_KM)) * km * np.sqrt(e), cell))
                info["length_km"] += float(s[idx[-1]] - s[idx[0]]) / km
        info["rifts_on_land"] += int(len(pts) > n_before)
    info["basins"] = len(pts)
    if not pts:
        return out.reshape(6, N, N), info
    p = np.concatenate(pts)
    v = np.concatenate(val)
    half = np.concatenate(halves)
    reach = float(half.max()) + wall + SHOULDER_REACH * lam
    # the cells within reach of any basin: balls about every few samples cover the line
    hop = max(int(0.25 * reach / step), 1)
    lists = ctree.query_ball_point(p[::hop], float(_chord(reach + hop * step)))
    cells = np.unique(np.concatenate([np.asarray(c, dtype=np.int64) for c in lists]))
    d, j = cKDTree(p).query(centers[cells])
    d = 2.0 * np.arcsin(np.minimum(0.5 * d, 1.0))
    out[cells] = v[j] * profile(d, half[j], wall, lam)
    info["deepest_m"] = float(-out.min())
    info["cells"] = int((out < 0.0).sum())
    return out.reshape(6, N, N), info


__all__ = ["rifting_pairs", "rift_contacts", "rift_axis", "basins", "profile", "graben_field"]
