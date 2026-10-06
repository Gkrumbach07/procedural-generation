"""The lakes of a level below the coarse grid: its parent's, and no others.

A river-cut landscape has no closed depressions -- a lake silts up or its
outlet cuts down within a geological moment -- so every lake on Earth has a
cause that is recent or still at work: ice, a sinking or blocked basin, a dry
climate with no way out.  The planet's stages are where those causes are:
tectonics' basins, the glacial pass, hydro's water balance.  A level below
the coarse grid (a zoom window, the planet at 1.2 km) adds none of its own:
the depressions its flood finds are its erosion's dams, the pits of its
detail noise and the pools of its hold.  So a level takes its lakes from its
parent and treats them as water from its first iteration to its last:

* before it erodes (:func:`standing`: :func:`level_lakes` on the plain, the
  parent's surface upsampled): the detail noise fades to nothing at a lake's level
  (:func:`lake_quiet`), as it does at the sea's; the fill that drains the
  noise's own pits (``refine.zoom.drain_noise``) leaves the parent's basins
  open; and the lake's cells are flagged, so the particles lay their load at
  its shore and cross it without touching the bed (``zoom.bake.erode_tile``);
* when it has eroded (:func:`level_lakes` on the surface it ends with): a
  depression holds water only where it lies under a lake of the parent's, at
  that lake's level.

Measured on earth-v24 at 1.2 km before any of this: the pit fill laid 97 % of
a 31 m-deep plateau lake's volume as ground before the first particle moved
(the upsample of a coarser level does *not* drain everywhere: its lakes are
closed depressions), the erosion then cut that flat 40 m rms, uncorrelated
with the noise (r = 0.1), and the finish flooded the pieces lying below the
lake's level: fingers and loops along the level's own channels and banks.
"""
from __future__ import annotations

import numpy as np
from numba import njit
from scipy import ndimage

from ..hydro.priority_flood import priority_flood_flat

#: a depression of a level is a lake of its parent's where at least this share of the water it
#: would hold lies over that lake's cells (earth-v24 at 1.2 km: the lakes that are run 0.38-0.95;
#: a 44,000 km2 basin touching one at a corner had none of its water under it)
LAKE_SHARE = 0.25

#: the detail noise is tapered at a lake's level out to this many parent cells from its water
LAKE_REACH = 2.0


@njit(cache=True)
def _lakes_kernel(ws, surf, ocean, over, level, own, share):
    n0, n1 = ws.shape
    parent = np.full(n0 * n1, -1, np.int32)
    # the level's depressions: wet cells joined where they stand at one flood level
    for i in range(n0):
        for j in range(n1):
            if ocean[i, j] or not (ws[i, j] > surf[i, j]):
                continue
            c = i * n1 + j
            parent[c] = c
            for k in range(4):
                ii = i - 1 if k < 3 else i
                jj = j - 1 + k if k < 3 else j - 1
                if ii < 0 or jj < 0 or jj >= n1:
                    continue
                q = ii * n1 + jj
                if parent[q] < 0 or ws[ii, jj] != ws[i, j]:
                    continue
                a = c
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                while parent[q] != q:
                    parent[q] = parent[parent[q]]
                    q = parent[q]
                if a != q:
                    if a < q:
                        parent[q] = a
                    else:
                        parent[a] = q
    # the parent's lake over each depression: the highest level of the lake cells it lies under
    top = np.full(n0 * n1, -np.inf, np.float32)
    for i in range(n0):
        for j in range(n1):
            if not over[i, j]:
                continue
            c = i * n1 + j
            if parent[c] < 0:
                continue
            while parent[c] != c:
                parent[c] = parent[parent[c]]
                c = parent[c]
            if level[i, j] > top[c]:
                top[c] = level[i, j]
    # a depression no lake of the parent's lies over may still have a cause of the level's own
    # (`own`: the hollows the ice cut, zoom/ice.py): it stands at its spill point
    for i in range(n0):
        for j in range(n1):
            if not own[i, j]:
                continue
            c = i * n1 + j
            if parent[c] < 0:
                continue
            while parent[c] != c:
                parent[c] = parent[parent[c]]
                c = parent[c]
            if top[c] == -np.inf:
                top[c] = np.inf
    # ...which it is the same water as only if a fair share of what it would hold lies under that
    # lake's cells: a basin that touches a lake of the parent's at one corner is not that lake
    wet = np.zeros(n0 * n1, np.int32)
    under = np.zeros(n0 * n1, np.int32)
    for i in range(n0):
        for j in range(n1):
            c = i * n1 + j
            if parent[c] < 0:
                continue
            while parent[c] != c:
                parent[c] = parent[parent[c]]
                c = parent[c]
            z = top[c]
            if z > -np.inf and surf[i, j] < (ws[i, j] if ws[i, j] < z else z):
                wet[c] += 1
                if over[i, j] or own[i, j]:
                    under[c] += 1
    out = np.empty((n0, n1), np.float32)
    for i in range(n0):
        for j in range(n1):
            if ocean[i, j]:
                out[i, j] = 0.0
                continue
            out[i, j] = surf[i, j]
            c = i * n1 + j
            if parent[c] < 0:
                continue
            while parent[c] != c:
                parent[c] = parent[parent[c]]
                c = parent[c]
            z = top[c]
            if z > -np.inf and under[c] >= share * wet[c]:
                lv = ws[i, j] if ws[i, j] < z else z
                if lv > surf[i, j]:
                    out[i, j] = lv
    return out


def lakes_under(ws: np.ndarray, surf: np.ndarray, ocean: np.ndarray, over: np.ndarray, level: np.ndarray, own: np.ndarray | None = None) -> np.ndarray:
    """A level's water surface with the lakes its parent has, and no others.

    ``ws`` is the level's own flood (every depression of its surface filled
    to its spill point), ``surf`` its ground, ``ocean`` its sea; ``over``
    marks the level's cells that lie over a lake cell of the parent's and
    ``level`` holds that lake's water level there.

    A depression of the level holds water only where it lies under a lake of
    the parent's -- at least :data:`LAKE_SHARE` of the water it would hold
    over that lake's cells -- and then at that lake's level (or its own spill
    point, where its outlet has cut lower): one level surface per lake, and a
    shore that is the level's own ground meeting it.  Everything else is dry;
    the sea is at 0.

    ``own`` marks cells with a cause for a lake that is the level's own (the
    hollows the ice cut, zoom/ice.py).  A depression no lake of the parent's
    lies over holds water at its spill point where at least that share of it
    lies on such cells; where a lake of the parent's does, its level rules.

    Before this a level's water was its flood capped at the *upsampled*
    parent water surface wherever that stood above the ground, which is not a
    level surface (73 % of earth-v24's lake area at 1.2 km lay in water bodies
    whose surface varied by more than a metre), then cut to the coarse lake
    cells and one beside them, which is where the square corners and stair
    edges came from."""
    own = np.zeros(ws.shape, np.bool_) if own is None else np.ascontiguousarray(own, np.bool_)
    return _lakes_kernel(np.ascontiguousarray(ws, np.float32), np.ascontiguousarray(surf, np.float32), np.ascontiguousarray(ocean, np.bool_),
                         np.ascontiguousarray(over, np.bool_), np.ascontiguousarray(level, np.float32), own, LAKE_SHARE)


def over_cells(coarse: np.ndarray, R: int, shape: tuple[int, int]) -> np.ndarray:
    """A coarse-cell array on the fine cells of the same block (its corner on a coarse cell)."""
    return np.repeat(np.repeat(coarse, R, axis=0), R, axis=1)[:shape[0], :shape[1]]


def lakes_of_the_planet(ws: np.ndarray, surf: np.ndarray, ocean: np.ndarray, coarse_lake: np.ndarray, coarse_level: np.ndarray, R: int) -> np.ndarray:
    """:func:`lakes_under` with the parent's lakes given on the coarse cells
    of the same block (``coarse_lake``, ``coarse_level``; the block's corner
    on a coarse cell)."""
    return lakes_under(ws, surf, ocean, over_cells(np.asarray(coarse_lake, bool), R, ws.shape), over_cells(np.asarray(coarse_level, np.float32), R, ws.shape))


def _rim_levels(surf: np.ndarray, over: np.ndarray, level: np.ndarray) -> np.ndarray:
    """``surf`` with the border cells a parent's lake covers raised to its
    level: the border cells over the lake's own cells, and the cells along
    the border from them for as far as the ground stays under that level."""
    n0, n1 = surf.shape
    ii = np.concatenate([np.zeros(n1 - 1, np.int64), np.arange(n0 - 1), np.full(n1 - 1, n0 - 1), np.arange(n0 - 1, 0, -1)])
    jj = np.concatenate([np.arange(n1 - 1), np.full(n0 - 1, n1 - 1), np.arange(n1 - 1, 0, -1), np.zeros(n0 - 1, np.int64)])
    s, o, z = surf[ii, jj].astype(np.float64), over[ii, jj], level[ii, jj].astype(np.float64)
    held = np.where(o, np.maximum(s, z), s)
    m = len(s)
    for order in (range(2 * m), range(2 * m - 1, -1, -1)):         # round the ring twice each way: a lake may sit on its seam
        carry = -np.inf
        for k in order:
            i = k % m
            if o[i]:
                carry = z[i]
            elif s[i] < carry:
                held[i] = max(held[i], carry)
            else:
                carry = -np.inf
    out = surf.copy()
    out[ii, jj] = held
    return out


def _flood(s: np.ndarray, ocean: np.ndarray, over: np.ndarray, level: np.ndarray) -> np.ndarray:
    """A block's flood, draining to the sea and to its border -- at a lake's
    level where the border lies over a lake of the parent's (:func:`_rim_levels`)."""
    edge = np.zeros(s.shape, bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    held = _rim_levels(np.where(ocean, np.float32(np.inf), s), over, level) if (edge & over).any() else s
    held = np.where(ocean, s, held).astype(np.float32)
    fr = priority_flood_flat(held, ocean | edge, None)
    return np.maximum(fr.filled.reshape(s.shape), held)


#: a cell under less than this much of the level's own flood is dry ground, metres (what "dry
#: before the ice cut it" means in :func:`level_lakes`)
DRY_M = 0.5


def level_lakes(surf: np.ndarray, ocean: np.ndarray, over: np.ndarray, level: np.ndarray, cut: np.ndarray | None = None, wet: np.ndarray | None = None) -> np.ndarray:
    """The water surface of a block of a level (:func:`lakes_under`), from
    its ground alone: the ground itself where it is dry, a lake's level on
    the lake, 0 on the sea.

    The flood drains to the sea and to the block's border, and a lake that
    runs off the block would drain through the border along its bed and be
    lost there.  The border cells the parent's lake covers are the lake's
    surface, so they drain at its level instead (:func:`_rim_levels`).

    ``cut`` (metres, 0 where none) is how far the ice lowered ``surf`` at
    this level (zoom/ice.py), ``wet`` where the climate lets its hollows
    hold water (all of them when None).  A cell is the ice's own lake where
    it is under water now and was dry ground (under :data:`DRY_M` of flood)
    on the surface before the cut, ``surf + cut``: the hollows the ice made,
    not the lows the level's rivers had left."""
    s = np.ascontiguousarray(surf, np.float32)
    ocean = np.ascontiguousarray(ocean, np.bool_)
    over = np.asarray(over, bool) & ~ocean
    iced = cut is not None and bool((np.asarray(cut) > 0.0).any())
    if not over.any() and not iced:
        return np.where(ocean, np.float32(0.0), s).astype(np.float32)
    level = np.ascontiguousarray(level, np.float32)
    ws = _flood(s, ocean, over, level)
    own = None
    if iced:
        before = (s + np.asarray(cut, np.float32)).astype(np.float32)
        own = (ws > s) & ~(_flood(before, ocean, over, level) > before + np.float32(DRY_M)) & ~ocean
        if wet is not None:
            own &= np.asarray(wet, bool)
    return lakes_under(ws, s, ocean, over, level, own)


def parent_lakes(fields: dict, derived: dict, win) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(over, level, floor)`` on a window's extended fine array, from the
    planet (``refine.upsample.coarse_derived``): the cells over its lake
    cells, those lakes' water levels, and the cells over the floors of its
    basins."""
    from ..refine.upsample import sample

    return sample(derived["lake"], win) > 0, sample(fields["water_surface"], win, order=0).astype(np.float32), sample(derived["floor"], win) > 0


def standing(plain: np.ndarray, ocean: np.ndarray, over: np.ndarray, level: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(water, still)`` of a level before it erodes: :func:`level_lakes` on
    its plain as float64 -- the plain itself wherever no lake stands -- and
    the cells a lake stands on."""
    w = level_lakes(plain, ocean, over, level)
    still = (w > np.asarray(plain, np.float32)) & ~np.asarray(ocean, bool)
    return np.where(still, w.astype(np.float64), plain), still


def lake_quiet(plain: np.ndarray, water: np.ndarray, still: np.ndarray, taper_m: float, reach_cells: float) -> np.ndarray:
    """The share (0..1) of its detail noise a level keeps beside its parent's
    lakes (``water``, ``still``: :func:`standing`): none under a lake (a lake
    floor is draped in what settles on it), and on the ground within
    ``reach_cells`` of one, fading to none at the lake's level and full
    ``taper_m`` above it.

    The sea has had this since ``refine.coast_taper_m``: the noise is ridged,
    its amplitude set by the coarse slope, and added to ground within metres
    of a water level it moves the shore -- a fringe of two-cell inlets and
    islands.  Out past ``reach_cells`` nothing changes, the fade being smooth
    in the distance to the water."""
    q = np.ones(plain.shape, np.float32)
    if not still.any():
        return q
    if float(taper_m) > 0.0 and float(reach_cells) > 0.0:
        d, (ii, jj) = ndimage.distance_transform_edt(~still, return_indices=True)
        w = np.clip(1.0 - d / float(reach_cells), 0.0, 1.0)
        w = w * w * (3.0 - 2.0 * w)
        t = np.clip((plain - water[ii, jj]) / float(taper_m), 0.0, 1.0)
        t = t * t * (3.0 - 2.0 * t)
        q = (1.0 - w * (1.0 - t)).astype(np.float32)
    q[still] = 0.0
    return q


__all__ = ["LAKE_SHARE", "LAKE_REACH", "lakes_under", "lakes_of_the_planet", "level_lakes", "parent_lakes", "standing", "lake_quiet", "over_cells"]
