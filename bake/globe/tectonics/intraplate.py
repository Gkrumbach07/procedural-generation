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

from ..stubs import fbm_at
from .plates import Plates, cluster_plates, random_unit_vectors, snap_cratons


def _rebuild(seg, plate_id: np.ndarray, n_plates: int, rng, speed: float, keep: np.ndarray | None = None) -> Plates:
    """A fresh :class:`Plates` for an existing crust.

    With `keep` given, those Euler poles carry over and only the plates
    beyond them get new random ones. :func:`reorganise` wants the opposite
    -- a whole new convection pattern -- and passes nothing.

    Rifting has to keep them. Sharing the re-randomising path meant one rift
    event re-drew the pole of *every* plate on the planet, so a single split
    every `rift_every` steps scrambled all the plate motions with it: not a
    rift but a global reorganisation wearing one.
    """
    seg.plate_id = np.ascontiguousarray(plate_id, dtype=np.int32)
    snap_cratons(seg)          # a boundary goes around a craton, not through it
    plates = Plates(int(n_plates))
    plates.update_stats(seg)
    if keep is None:
        plates.omega[:] = random_unit_vectors(rng, (plates.P,)) * float(speed)
    else:
        k = min(int(keep.shape[0]), plates.P)
        plates.omega[:k] = keep[:k]
        if plates.P > k:
            plates.omega[k:] = random_unit_vectors(rng, (plates.P - k,)) * float(speed)
    plates.omega[~plates.alive] = 0.0
    return plates


def _capped(omega: np.ndarray, cap: float) -> np.ndarray:
    """An angular velocity clamped to ``tectonics.max_speed``.

    Anything that hands a plate a new pole has to respect the same cap the
    force model does, because the plate *moves* before ``update_omega`` is
    reached: a step is rotate, collide, then forces.  An over-fast pole is
    therefore not merely corrected one step later -- it has already teleported
    the crust.
    """
    s = float(np.linalg.norm(omega))
    return omega * (cap / s) if cap > 0.0 and s > cap else omega


def reorganise(sim, n_plates: int, rng) -> dict:
    """Re-partition the crust into `n_plates` new plates, keeping heights.

    The crust carries its accumulated thickness through unchanged; only the
    boundaries move. That is the whole point: today's interior becomes
    tomorrow's margin, and the belts already built stay where they are.
    """
    seg = sim.seg
    pid = cluster_plates(seg.pos, int(n_plates), rng, size_jitter=float(sim.tp.plate_size_jitter))
    sim.plates = _rebuild(seg, pid, int(n_plates), rng, float(sim.tp.initial_speed) * sim.spacing)
    return {"event": "reorganise", "plates": int(sim.plates.n_alive())}


def _rift_one(sim, target: int, rng) -> dict:
    """Open a rift through plate `target`, threading between its cratons.

    A rift is not a straight cut. Two things shape where it goes:

    * **It is segmented.** A spreading centre is a zig-zag of ridge
      segments offset by transform faults, because the plate cannot pull
      apart along one smooth arc -- the Mid-Atlantic Ridge is a staircase,
      not a line. Spherical noise added to the cut plane reproduces that at
      the scale the model resolves.
    * **It avoids cratons.** Archean nuclei are thick, cold and strong;
      extension localises in the weaker mobile belts welded between them.
      Gondwana split *between* Amazonia, West Africa, Congo and Kalahari,
      leaving each craton intact on one side or the other, which is why the
      same nuclei are still recognisable on both sides of the Atlantic.
      Each craton is therefore assigned whole, by majority vote, to
      whichever side most of it fell on.

    A straight plane through the centre of mass -- what this did before --
    cuts cratons in half as readily as anything else, which is the one thing
    a real rift does not do.
    """
    seg, plates, tp = sim.seg, sim.plates, sim.tp
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

    # signed distance from the plane, made a staircase by spherical noise
    d = seg.pos[sel] @ n
    zig = float(tp.rift_zigzag)
    if zig > 0.0:
        d = d + zig * fbm_at(seg.pos[sel], rng, octaves=3, base_freq=float(tp.rift_zigzag_freq))
    side = d > 0.0
    if side.all() or not side.any():
        return {"event": "rift", "split": -1}

    # keep every craton whole: whichever side holds most of it takes all of it
    cr = seg.craton[sel]
    intact = 0
    for c in np.unique(cr[cr > 0]):
        m = cr == c
        maj = bool(side[m].mean() > 0.5)
        if side[m].mean() not in (0.0, 1.0):
            intact += 1
        side[m] = maj
    if side.all() or not side.any():
        return {"event": "rift", "split": -1}

    P = plates.P
    pid = seg.plate_id.copy()
    idx = np.flatnonzero(sel)
    pid[idx[side]] = P  # the far half becomes a brand-new plate
    new = _rebuild(seg, pid, P + 1, rng, float(tp.initial_speed) * sim.spacing, keep=plates.omega)
    # Open the cut.  The Euler pole of a spreading pair lies *on* the rift,
    # 90 degrees from its middle, so the halves turn about `com x n` in
    # opposite senses and separate along n.  Turning about n itself -- what
    # this did -- holds every segment at its own distance from the cut plane
    # and slides the two halves along it: a transform fault the length of a
    # continent, not a rift.  With the zig-zag cut the teeth then grind
    # through each other, and continent-on-continent collisions over the
    # fifty steps after a rift ran 6-10x the rate over the fifty before it
    # (64 -> 602 at step 400, 278 -> 1552 at step 1200).
    axis = np.cross(com, n)
    ln = float(np.linalg.norm(axis))
    if ln < 1e-9:
        return {"event": "rift", "split": -1}
    axis /= ln
    # The halves keep the plate's own motion and add the opening to it, at a
    # plate speed: `rift_speed_factor` multiplies the speed cap, not
    # `convection` (which is a force gain, 33x the cap -- see config.py).
    rate = float(sim.max_omega) if sim.max_omega > 0 else float(tp.initial_speed) * sim.spacing
    sep = float(tp.rift_speed_factor) * rate
    base = plates.omega[target].copy()
    new.omega[target] = _capped(base - 0.5 * sep * axis, sim.max_omega)
    new.omega[P] = _capped(base + 0.5 * sep * axis, sim.max_omega)
    new.omega[~new.alive] = 0.0
    sim.plates = new
    return {"event": "rift", "split": target, "moved": int(side.sum()),
            "cratons_spared": intact, "plates": int(new.n_alive())}


def rift(sim, rng, max_plates: int = 1) -> dict:
    """Rift one or two plates, chosen at random, weighted by area.

    Not the largest one every time. Targeting `argmax` made the event
    deterministic given the configuration, so the same plate -- usually the
    supercontinent -- got sliced again and again along a fresh plane, which
    reads as the whole map coming apart on a schedule rather than as
    individual rifts opening. Real rifting picks its moment and its place:
    the Atlantic opened while the Pacific plates carried on untouched.

    Area weighting keeps the physics that motivated `argmax` in the first
    place -- a large plate insulates the mantle beneath it and is the one
    most likely to fail, which is the supercontinent cycle -- while leaving
    it a *tendency* rather than a certainty. The count is 1 or 2 so an event
    is not always the same size either.
    """
    plates = sim.plates
    alive = np.flatnonzero(plates.alive & (plates.count >= 8))
    if alive.size == 0:
        return {"event": "rift", "split": -1}
    n = 1 + int(rng.integers(0, max(1, int(max_plates))))
    n = min(n, alive.size)
    w = plates.count[alive].astype(np.float64)
    w = w / w.sum() if w.sum() > 0 else None
    targets = rng.choice(alive, size=n, replace=False, p=w)
    out = []
    for t in np.atleast_1d(targets):
        # plate ids are stable across a split (the new half is appended), so
        # an earlier split in this event cannot invalidate a later target
        out.append(_rift_one(sim, int(t), rng))
    split = [r["split"] for r in out if r["split"] >= 0]
    return {"event": "rift", "split": split, "plates": int(sim.plates.n_alive()),
            "moved": sum(r.get("moved", 0) for r in out)}


def split_disconnected(sim, min_segments: int = 16, link_factor: float = 1.6, rng=None) -> dict:
    """A plate that has been cut in two is two plates.

    Subduction eats a plate from its edges, and where a trench cuts right
    across one it leaves the remainder in separate pieces.  Nothing here
    noticed: a plate is a rigid rotation about one pole, and a rigid rotation
    preserves distances, so the pieces kept their separation for the rest of
    the run and swept across the planet locked together.  Measured on the
    Earth preset at step 800, one plate was four pieces of 30 / 24 / 16 / 11 %
    of its area lying **46 to 83 degrees apart**, another two pieces 97
    degrees apart; over a run the ocean plates ended up interleaved in
    ribbons, because a piece of one plate is dragged through its neighbours
    by a pole it no longer has any physical connection to.

    So every step, the segment cloud of each plate is split into connected
    components (segments within ``link_factor`` spacings of each other are
    joined).  The largest component keeps the plate; any other component of
    at least ``min_segments`` becomes a new plate, starting with its parent's
    motion and free to diverge from it under the forces afterwards.  Smaller
    fragments are welded onto whichever neighbouring plate surrounds them --
    a sliver of crust is part of the plate it is embedded in, not a plate.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    from .collision import build_tree

    seg, plates = sim.seg, sim.plates
    if seg.M < 2:
        return {"event": "split", "split": 0}
    pairs = build_tree(seg).query_pairs(float(link_factor) * sim.spacing, output_type="ndarray")
    if pairs.shape[0] == 0:
        return {"event": "split", "split": 0}
    e = pairs[seg.plate_id[pairs[:, 0]] == seg.plate_id[pairs[:, 1]]]
    g = coo_matrix((np.ones(e.shape[0], np.int8), (e[:, 0], e[:, 1])), shape=(seg.M, seg.M))
    _, comp = connected_components(g, directed=False)
    sizes = np.bincount(comp)
    pid = seg.plate_id.copy()
    P = plates.P
    extra: list[np.ndarray] = []
    orphans = np.zeros(seg.M, dtype=bool)
    for p in np.unique(seg.plate_id):
        cs = np.unique(comp[seg.plate_id == p])
        if cs.size < 2:
            continue
        for c in cs[np.argsort(sizes[cs])[::-1]][1:]:      # every piece but the largest
            m = comp == c
            if sizes[c] >= int(min_segments):
                pid[m] = P + len(extra)
                extra.append(plates.omega[p].copy())
            else:
                orphans |= m
    if orphans.any():
        # weld a fragment onto the plate around it: the commonest plate among
        # the nearest segments that are not part of the fragment itself
        idx = np.flatnonzero(orphans)
        k = min(9, seg.M)
        _, nb = build_tree(seg).query(seg.pos[idx], k=k, workers=-1)
        nb = np.atleast_2d(nb).reshape(idx.size, k)
        near = np.where(orphans[nb], -1, pid[nb])
        for row, i in zip(near, idx):
            vals = row[row >= 0]
            if vals.size:
                pid[i] = np.bincount(vals).argmax()
    if not extra and not orphans.any():
        return {"event": "split", "split": 0}
    keep = np.vstack([plates.omega] + [np.asarray(o)[None, :] for o in extra]) if extra else plates.omega
    sim.plates = _rebuild(seg, pid, P + len(extra), rng, float(sim.tp.initial_speed) * sim.spacing, keep=keep)
    return {"event": "split", "split": len(extra), "welded": int(orphans.sum()),
            "plates": int(sim.plates.n_alive())}


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
