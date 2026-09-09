"""Relief away from active plate boundaries.

Measured on an Earth-scale world before this existed: 88 % of land sat in
plate interiors with a median local relief of **6 m** over a 107 km window,
against 696 m at collision zones and 50-300 m for real interior plains.
Land was either dead flat or a boundary ridge, with nothing in between,
and the bedrock spectrum was beta 10.7 where real topography is ~2.

The cause is structural rather than a tuning problem. Bedrock height is
crust thickness splatted from the segment cloud, and a segment in a plate
interior never has anything happen to it, so relief exists only where
collision thickened the crust. With a fixed plate configuration -- the
shipped model ran 16 plates for all 1500 steps, never splitting or
merging -- an interior is never a boundary, so it is never uplifted.

Earth's interiors are not like that because they are *former* boundaries.
The plate configuration reorganises every few hundred million years, so
crust that sits mid-plate today was a collision zone before: the
Appalachians, the Urals, the Caledonides are all dead orogens stranded
inside plates. Interior relief on Earth is inherited, a palimpsest of
worn-down belts.

Three processes here put that back, in the order they matter:

* :func:`reorganise` -- re-partition the existing crust into a fresh set of
  plates with new Euler poles. Heights are kept, so every previous
  generation of belts survives while the boundaries move somewhere new.
  Run over several reorganisations, the map accumulates generations of
  orogens the way a real continent does.
* :func:`rift` -- split one plate in two along a plane through its centre
  of mass and push the halves apart. This is the other half of the
  supercontinent cycle, and it opens new boundaries *inside* old interiors
  (East Africa is doing it now).
* :func:`hotspots` -- fixed points in the mantle frame that thicken the
  crust drifting over them. Plumes are the one source of interior relief
  that owes nothing to plate boundaries at all (Hawaii, Yellowstone, the
  Deccan), and because the points are fixed while the plates move, they
  paint tracks rather than blobs.

All three are deterministic: every random draw comes from
``params.rng("tectonics", ...)`` keyed on the step index.
"""
from __future__ import annotations

import numpy as np

from .plates import Plates, cluster_plates, random_unit_vectors


def _rebuild(seg, plate_id: np.ndarray, n_plates: int, rng, speed: float) -> Plates:
    """A fresh :class:`Plates` for an existing crust, with new Euler poles."""
    seg.plate_id = np.ascontiguousarray(plate_id, dtype=np.int32)
    plates = Plates(int(n_plates))
    plates.update_stats(seg)
    plates.omega[:] = random_unit_vectors(rng, (plates.P,)) * float(speed)
    plates.omega[~plates.alive] = 0.0
    return plates


def reorganise(sim, n_plates: int, rng) -> dict:
    """Re-partition the crust into `n_plates` new plates, keeping heights.

    The crust carries its accumulated thickness through unchanged; only the
    boundaries move. That is the whole point: today's interior becomes
    tomorrow's margin, and the belts already built stay where they are.
    """
    seg = sim.seg
    pid = cluster_plates(seg.pos, int(n_plates), rng, size_jitter=float(sim.tp.plate_size_jitter))
    sim.plates = _rebuild(seg, pid, int(n_plates), rng, sim.tp.convection * sim.spacing)
    return {"event": "reorganise", "plates": int(sim.plates.n_alive())}


def rift(sim, rng) -> dict:
    """Split the largest plate along a plane through its centre of mass.

    The two halves get poles that separate them, so the new boundary opens
    rather than closing -- a rift, not another collision. Everything the
    plate had already accumulated is untouched on both sides.
    """
    seg, plates = sim.seg, sim.plates
    if plates.n_alive() < 1:
        return {"event": "rift", "split": -1}
    target = int(np.argmax(np.where(plates.alive, plates.count, -1)))
    sel = seg.plate_id == target
    if int(sel.sum()) < 8:
        return {"event": "rift", "split": -1}

    com = plates.com[target]
    if np.linalg.norm(com) < 1e-9:
        com = seg.pos[sel].mean(axis=0)
        com = com / max(np.linalg.norm(com), 1e-12)
    # a cut plane through the centre of mass, normal perpendicular to it so
    # the plane passes through the plate rather than slicing off a cap
    v = random_unit_vectors(rng, (1,))[0]
    n = v - com * float(v @ com)
    ln = float(np.linalg.norm(n))
    if ln < 1e-9:
        return {"event": "rift", "split": -1}
    n /= ln

    side = (seg.pos[sel] @ n) > 0.0
    if side.all() or not side.any():
        return {"event": "rift", "split": -1}

    P = plates.P
    pid = seg.plate_id.copy()
    idx = np.flatnonzero(sel)
    pid[idx[side]] = P  # the far half becomes a brand-new plate
    new = _rebuild(seg, pid, P + 1, rng, sim.tp.convection * sim.spacing)
    # override the two halves so they actually diverge across the new cut
    sep = float(sim.tp.convection) * sim.spacing * float(sim.tp.rift_speed_factor)
    new.omega[target] = -n * sep
    new.omega[P] = n * sep
    new.omega[~new.alive] = 0.0
    sim.plates = new
    return {"event": "rift", "split": target, "moved": int(side.sum()), "plates": int(new.n_alive())}


def seed_hotspots(count: int, rng) -> np.ndarray:
    """`count` fixed points in the mantle frame (unit vectors, (H, 3))."""
    if count <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    return random_unit_vectors(rng, (int(count),))


def apply_hotspots(sim, spots: np.ndarray, rate: float, radius: float) -> dict:
    """Thicken crust drifting over each hotspot.

    The spots do not move with the plates, so a plate crossing one comes out
    with a linear track of thickened crust behind it, the way the Hawaiian
    and Yellowstone chains record their plate's motion. Buoyancy does the
    rest: `height = thickness * (1 - density)`.
    """
    seg = sim.seg
    if spots.shape[0] == 0 or rate <= 0.0:
        return {"event": "hotspot", "cells": 0, "added": 0.0}
    cos_r = float(np.cos(radius))
    hit = np.zeros(seg.M, dtype=bool)
    add = np.zeros(seg.M, dtype=np.float64)
    for s in spots:
        d = seg.pos @ s
        m = d > cos_r
        if not m.any():
            continue
        # taper to nothing at the rim so a plume builds a swell, not a plateau
        w = (d[m] - cos_r) / max(1.0 - cos_r, 1e-12)
        add[m] += rate * w
        hit |= m
    if not hit.any():
        return {"event": "hotspot", "cells": 0, "added": 0.0}
    seg.thickness += add
    seg.mass += add * seg.density
    return {"event": "hotspot", "cells": int(hit.sum()), "added": float(add.sum())}


__all__ = ["reorganise", "rift", "seed_hotspots", "apply_hotspots"]
