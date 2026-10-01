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


def _rebuild(seg, plate_id: np.ndarray, n_plates: int, rng, speed: float, keep: np.ndarray | None = None,
             snap: bool = True, keep_welds: bool = False) -> Plates:
    """A fresh :class:`Plates` for an existing crust.

    With `keep` given, those Euler poles carry over and only the plates
    beyond them get new random ones. :func:`reorganise` wants the opposite
    -- a whole new convection pattern -- and passes nothing.

    Rifting has to keep them. Sharing the re-randomising path meant one rift
    event re-drew the pole of *every* plate on the planet, so a single split
    every `rift_every` steps scrambled all the plate motions with it: not a
    rift but a global reorganisation wearing one.

    `snap` keeps every craton on one plate (:func:`snap_cratons`). That is a
    rule for drawing a *new* boundary -- a boundary goes around a craton, not
    through it -- and only :func:`reorganise` draws boundaries from nothing.
    A rift keeps its cratons whole with its own vote, and a split or a suture
    draws no boundary at all: the pieces are where the plates already put
    them. Running it on every rebuild (88-91 % of steps at Earth scale, since
    :func:`split_disconnected` rebuilds whenever anything moved) teleported
    any craton that collisions or a trench had left across two plates onto
    whichever held more of it, a piece of crust jumping to a plate it was not
    touching -- which the next split then promoted to a plate of its own:
    450-2100 plate births a run, half of them dead within 10 steps. The
    fixed-area model (``variable_extent`` off) keeps the old rule, bit for bit:
    a snap on every rebuild, welds and all (`keep_welds`, see snap_cratons).
    """
    seg.plate_id = np.ascontiguousarray(plate_id, dtype=np.int32)
    if snap:
        snap_cratons(seg, keep_welds)      # a boundary goes around a craton, not through it
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


#: the fewest segments a plate may be cut from (the count the fixed-area model still uses,
#: and the sampling floor under ``tectonics.rift_min_area`` with variable extent)
RIFT_MIN_SEGMENTS = 8


def min_ground(sim, area: float, segments: int) -> float:
    """The ground, in steradians, a plate-sized piece of crust needs: `area` of
    the sphere, but never less than `segments` design segments' worth.

    A size rather than a count, because a count is a size that shrinks with
    the resolution: 16 segments is 8e-4 of the sphere at 20000 and a quarter
    of that at 80000, and the 80k runs shattered into 3-10x the plates at
    equal steps. The count stays as a floor, because below a handful of
    points a plate is the sampling rather than the planet.
    """
    return max(float(area) * 4.0 * np.pi, float(segments) * float(sim.spacing) ** 2)


def plate_ground(sim) -> np.ndarray:
    """Summed extent per plate, steradians (P,)."""
    P = sim.plates.P
    return np.bincount(sim.seg.plate_id, weights=sim.seg.ext, minlength=P)[:P]


def reorganise(sim, n_plates: int, rng) -> dict:
    """Re-partition the crust into `n_plates` new plates, keeping heights.

    The crust carries its accumulated thickness through unchanged; only the
    boundaries move. That is the whole point: today's interior becomes
    tomorrow's margin, and the belts already built stay where they are.
    """
    seg = sim.seg
    pid = cluster_plates(seg.pos, int(n_plates), rng, size_jitter=float(sim.tp.plate_size_jitter))
    sim.plates = _rebuild(seg, pid, int(n_plates), rng, float(sim.tp.initial_speed) * sim.spacing,
                          keep_welds=bool(sim.tp.variable_extent))
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
    if (float(seg.ext[sel].sum()) < min_ground(sim, tp.rift_min_area, RIFT_MIN_SEGMENTS) if tp.variable_extent
            else int(sel.sum()) < RIFT_MIN_SEGMENTS):
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

    idx = np.flatnonzero(sel)
    stranded = 0
    if tp.variable_extent:
        side, stranded = _settle_cut(sim, idx, side)
        if side.all() or not side.any():
            return {"event": "rift", "split": -1}

    P = plates.P
    pid = seg.plate_id.copy()
    pid[idx[side]] = P  # the far half becomes a brand-new plate
    # the vote above is this rift's craton rule; a global snap here would move cratons on plates
    # the cut never touched (the fixed-area model keeps it, bit for bit)
    new = _rebuild(seg, pid, P + 1, rng, float(tp.initial_speed) * sim.spacing, keep=plates.omega,
                   snap=not tp.variable_extent)
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
    return {"event": "rift", "split": target, "new": int(P), "moved": int(side.sum()),
            "cratons_spared": intact, "stranded": stranded, "plates": int(new.n_alive())}


def _settle_cut(sim, idx: np.ndarray, side: np.ndarray, link_factor: float = 1.6, rounds: int = 4) -> tuple[np.ndarray, int]:
    """Give the pieces a rift cut strands back to the half that surrounds them.

    The zig-zag cut is a noisy plane, and noise makes islands: a tooth of one
    half pinched off inside the other. Every one of 24 trial cuts of the
    step-0 supercontinent left such pieces (up to 12 per half), and in 11 of
    them a piece of 16 segments or more -- which the next
    :func:`split_disconnected` promoted to a plate of its own, with the far
    half's pole: a sliver driven through the half it sits in.

    So the pieces are found the way the split finds them (segments of the
    plate within ``link_factor`` spacings, same side), and every piece of a
    half but its main body changes side if the other half borders it and
    either it is smaller than a plate may be (`plate_min_area`, so the split
    would only weld it to whatever surrounds it) or the other half owns at
    least as much of its boundary as the plates beyond the rifted one do.
    A large piece that lies mostly against other plates is a lobe of the
    plate on this side of the cut, not an island, and keeps its side. A few
    rounds catch an island inside an island. `idx` is the plate's segments,
    `side` their half; returns the new `side` and how many segments changed.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    from .collision import build_tree

    seg = sim.seg
    side = side.copy()
    n_loc = idx.size
    pairs = build_tree(seg).query_pairs(float(link_factor) * sim.spacing, output_type="ndarray")
    loc = np.full(seg.M, -1, dtype=np.int64)
    loc[idx] = np.arange(n_loc)
    a = np.concatenate([pairs[:, 0], pairs[:, 1]])
    b = np.concatenate([pairs[:, 1], pairs[:, 0]])
    a, b = loc[a], loc[b]
    on = a >= 0
    a, b = a[on], b[on]                    # links from the plate: to the plate (b >= 0) or off it
    inside = b >= 0
    ext = seg.ext[idx]
    small = min_ground(sim, sim.tp.plate_min_area, sim.tp.plate_split_min)
    changed = np.zeros(n_loc, dtype=bool)
    for _ in range(int(rounds)):
        same = inside & (side[a] == side[np.maximum(b, 0)])
        g = coo_matrix((np.ones(int(same.sum()), np.int8), (a[same], b[same])), shape=(n_loc, n_loc))
        n, comp = connected_components(g, directed=False)
        ground = np.bincount(comp, weights=ext, minlength=n)
        body = np.zeros(n, dtype=bool)
        for s in (False, True):
            cs = np.unique(comp[side == s])
            if cs.size:
                body[cs[np.argmax(ground[cs])]] = True
        other = np.bincount(comp[a[inside & ~same]], minlength=n)      # links into the other half
        beyond = np.bincount(comp[a[~inside]], minlength=n)            # links off the rifted plate
        flip = ~body & (other > 0) & ((ground < small) | (other >= beyond))
        if not flip.any():
            break
        f = flip[comp]
        side ^= f
        changed ^= f
    return side, int(changed.sum())


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
    plates, tp = sim.plates, sim.tp
    if str(tp.rift_mode) == "insulation":
        # one entry point for both modes, so whatever wraps `rift` sees every rift
        return insulation_rifts(sim, rng)
    if str(tp.rift_mode) == "force":
        return force_rifts(sim, rng)
    big = (plate_ground(sim) >= min_ground(sim, tp.rift_min_area, RIFT_MIN_SEGMENTS) if tp.variable_extent
           else plates.count >= RIFT_MIN_SEGMENTS)
    alive = np.flatnonzero(plates.alive & big)
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
            "pairs": [(r["split"], r["new"]) for r in out if r["split"] >= 0],
            "moved": sum(r.get("moved", 0) for r in out)}


def insulation_deficit(sim) -> tuple[np.ndarray, np.ndarray]:
    """Per segment, how far the mantle under it has been insulated towards
    `insulation_floor`, as a share of the full deficit (0 neutral mantle, 1
    fully insulated), and the plate's continental share of the sphere."""
    from ..field import FaceField
    from .segments import CONTINENTAL

    seg, tp = sim.seg, sim.tp
    H = sim.heat.grid.H
    neu = sim.heat_neutral[:, H:-H, H:-H]
    bg = sim.heat_bg[:, H:-H, H:-H]
    full = np.maximum(neu - float(tp.insulation_floor), 1e-6)
    f = FaceField.from_interior(sim.heat.grid, np.clip((neu - bg) / full, 0.0, 1.0), exchange=True)
    D = np.clip(f.sample_sphere(seg.pos).astype(np.float64), 0.0, 1.0)
    cont = seg.kind == CONTINENTAL
    P = sim.plates.P
    Ac = np.bincount(seg.plate_id[cont], weights=seg.ext[cont], minlength=P)[:P] / (4.0 * np.pi)
    return D, Ac


def insulation_rifts(sim, rng) -> dict:
    """Rift every continental plate whose insulation stress is past its strength.

    A supercontinent insulates the mantle under it; the mantle warms, wells
    up and puts the lithosphere above it in tension, while the circum-
    supercontinent trenches pull its margins outwards.  The stress grows with
    the deficit (how long the continent has sat still) and with the size of
    the continent, so a supercontinent that has sat for ~100 My breaks and a
    small, moving continent does not.  ``sigma = mean deficit under the
    plate's continental crust * sqrt(continental share)``; a plate rifts when
    sigma exceeds ``rift_stress`` (times a per-plate strength jitter of
    +-15 %), and not again for ``rift_refractory_my``.  Ocean plates do not
    rift this way."""
    from .segments import CONTINENTAL

    seg, plates, tp = sim.seg, sim.plates, sim.tp
    k = int(sim.step_index)
    D, Ac = insulation_deficit(sim)
    cont = seg.kind == CONTINENTAL
    P = plates.P
    w = seg.ext * cont
    Dp = np.bincount(seg.plate_id, weights=D * w, minlength=P)[:P] / np.maximum(np.bincount(seg.plate_id, weights=w, minlength=P)[:P], 1e-30)
    refr = float(tp.rift_refractory_my) / max(float(tp.myr_per_step), 1e-9)
    out = []
    for p in np.flatnonzero(plates.alive & (Ac >= float(tp.rift_min_cont))):
        p = int(p)
        if k - sim.last_rift.get(p, -10 ** 9) < refr:
            continue
        sigma = float(Dp[p] * np.sqrt(Ac[p]))
        jit = sim.rift_jitter.setdefault(p, 1.0 + 0.15 * (2.0 * float(rng.random()) - 1.0))
        if sigma < float(tp.rift_stress) * jit:
            continue
        r = _rift_insulated(sim, p, rng, D)
        if r["split"] >= 0:
            r["trigger"] = sigma
            out.append(r)
    pairs = [(r["split"], r["new"]) for r in out]
    return {"event": "rift", "split": [r["split"] for r in out], "pairs": pairs,
            "centre": out[0]["centre"] if out else None, "trigger": [r["trigger"] for r in out],
            "plates": int(sim.plates.n_alive())}


def _rift_insulated(sim, target: int, rng, D: np.ndarray, candidates: int = 32) -> dict:
    """Cut plate `target` along the weakest line through its upwelling.

    The cut is a great circle through the deficit-weighted centre of the
    plate's continent (where the insulation upwelling is), chosen among
    `candidates` orientations for the lowest cost: every crossing of the
    cut costs its length, four times that through a craton (cratons are
    strong), scaled by the crust's assembly age (`rework`, low on a young
    suture or belt: rifts reopen sutures, Buiter & Torsvik 2014).  Both
    halves must keep a quarter of the plate's continent, and the cut must
    stay within 80 degrees of its centre: the normal component of any
    relative rotation along a great circle goes as cos(theta - theta0), so
    a cut longer than ~180 degrees must converge somewhere (the shipped
    rift's far side).  Cratons go whole to the side holding most of them,
    and any piece of a half that is not connected to the rest of it joins
    the other half, so the cut leaves no stranded slivers.

    No kick: the halves keep the plate's motion.  What opens the rift is the
    forces -- the cut lies on the insulation low, the halves' outer margins
    on the girdle -- held back at first by the rift's own strength
    (`rift_strength`, weakening with opening; see forces.py).
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    from .segments import CONTINENTAL

    seg, plates, tp = sim.seg, sim.plates, sim.tp
    sel = np.flatnonzero(seg.plate_id == target)
    if sel.size < 16:
        return {"event": "rift", "split": -1}
    pos = seg.pos[sel]
    cont = seg.kind[sel] == CONTINENTAL
    ext = seg.ext[sel]
    wc = ext * cont * (0.05 + D[sel])
    c = (pos * wc[:, None]).sum(axis=0)
    if np.linalg.norm(c) < 1e-9:
        return {"event": "rift", "split": -1}
    c /= np.linalg.norm(c)
    sp = sim.spacing
    pairs = cKDTree(pos).query_pairs(1.3 * sp, output_type="ndarray")
    if pairs.shape[0] == 0:
        return {"event": "rift", "split": -1}
    a, b = pairs[:, 0], pairs[:, 1]
    cr = seg.craton[sel] > 0
    rw = seg.rework[sel]
    mrw = float(rw[cont].mean()) if cont.any() else 1.0
    wseg = np.where(cont, (np.where(cr, 4.0, 1.0)) * (0.5 + rw / max(mrw, 1e-9)), 0.3)
    wpair = 0.5 * (wseg[a] + wseg[b])
    mid = pos[a] + pos[b]
    mid /= np.linalg.norm(mid, axis=1, keepdims=True)
    ang_mid = np.degrees(np.arccos(np.clip(mid @ c, -1.0, 1.0)))
    zig = float(tp.rift_cut_zigzag)
    noise = zig * fbm_at(pos, rng, octaves=3, base_freq=float(tp.rift_zigzag_freq)) if zig > 0.0 else 0.0
    Ctot = float((ext * cont).sum())
    best = None
    for _ in range(int(candidates)):
        v = random_unit_vectors(rng, (1,))[0]
        n = v - c * float(v @ c)
        ln = float(np.linalg.norm(n))
        if ln < 1e-9:
            continue
        n /= ln
        side = (pos @ n + noise) > 0.0
        cross = side[a] != side[b]
        if not cross.any():
            continue
        ca = float((ext * cont * side).sum())
        if min(ca, Ctot - ca) < 0.25 * Ctot:
            continue
        if float(ang_mid[cross].max()) > 80.0:
            continue
        cost = float(wpair[cross].sum())
        if best is None or cost < best[0]:
            best = (cost, n, side.copy())
    if best is None:
        return {"event": "rift", "split": -1}
    _, n, side = best
    # cratons whole
    crs = seg.craton[sel]
    for q in np.unique(crs[crs > 0]):
        m = crs == q
        side[m] = bool(side[m].mean() > 0.5)
    # no stranded pieces: a half keeps only its largest connected piece
    for s_val in (True, False):
        m = side == s_val
        idx = np.flatnonzero(m)
        if idx.size < 2:
            continue
        loc = np.full(sel.size, -1)
        loc[idx] = np.arange(idx.size)
        e = (side[a] == s_val) & (side[b] == s_val)
        g = coo_matrix((np.ones(int(e.sum()), np.int8), (loc[a[e]], loc[b[e]])), shape=(idx.size, idx.size))
        _, comp = connected_components(g, directed=False)
        sizes = np.bincount(comp)
        if sizes.size > 1:
            side[idx[comp != int(np.argmax(sizes))]] = not s_val
    if side.all() or not side.any():
        return {"event": "rift", "split": -1}
    P = plates.P
    pid = seg.plate_id.copy()
    pid[sel[side]] = P
    new = _rebuild(seg, pid, P + 1, rng, float(tp.initial_speed) * sim.spacing, keep=plates.omega,
                   snap=not tp.variable_extent)
    new.omega[P] = plates.omega[target]
    new.omega[~new.alive] = 0.0
    sim.plates = new
    k = int(sim.step_index)
    sim.rift_pairs[(int(target), int(P))] = {"k0": k, "delta": 0.0}
    sim.last_rift[int(target)] = k
    sim.last_rift[int(P)] = k
    sim.rift_jitter.pop(int(target), None)
    return {"event": "rift", "split": int(target), "new": int(P), "moved": int(side.sum()), "centre": c.tolist(),
            "plates": int(new.n_alive())}


def _strength(sim, idx: np.ndarray) -> np.ndarray:
    """Lithospheric strength per segment, relative to a mobile belt (1): cratons
    ``rift_craton_strength``, a suture or belt assembled within ``rift_suture_my``
    ``rift_suture_strength`` (rifts reopen sutures; Buiter & Torsvik 2014), sea floor
    ``rift_ocean_strength``."""
    from .segments import CONTINENTAL

    tp, seg = sim.tp, sim.seg
    cont = seg.kind[idx] == CONTINENTAL
    young = seg.rework[idx] < sim.steps_of(tp.rift_suture_my)
    s = np.where(seg.craton[idx] > 0, float(tp.rift_craton_strength),
                 np.where(young, float(tp.rift_suture_strength), 1.0))
    return np.where(cont, s, float(tp.rift_ocean_strength))


def _wq(x: np.ndarray, w: np.ndarray, q: float) -> float:
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / max(float(w.sum()), 1e-30)
    return float(x[o][min(int(np.searchsorted(cw, q)), x.size - 1)])


def _keep_connected(side: np.ndarray, a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Each half keeps only its largest connected piece; the rest join the other half."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    side = side.copy()
    for s_val in (True, False):
        idx = np.flatnonzero(side == s_val)
        if idx.size < 2:
            continue
        loc = np.full(n, -1)
        loc[idx] = np.arange(idx.size)
        e = (side[a] == s_val) & (side[b] == s_val)
        g = coo_matrix((np.ones(int(e.sum()), np.int8), (loc[a[e]], loc[b[e]])), shape=(idx.size, idx.size))
        nc, comp = connected_components(g, directed=False)
        if nc > 1:
            sizes = np.bincount(comp)
            side[idx[comp != int(np.argmax(sizes))]] = not s_val
    return side


def force_rifts(sim, rng) -> dict:
    """Rift the continental plates whose lithosphere fails under the forces (synth-dyn).

    The trigger is the force balance itself, released along a candidate cut
    (dyn-events' test, in dyn-minimal's balance): a plate carrying at least
    ``rift_min_cont`` of continent, insulated underneath (mean deficit >=
    ``rift_deficit_min``: it has sat still long enough for the mantle under it
    to well up) and not rifted within ``rift_refractory_my``, is cut along
    ``rift_candidates`` great circles -- half through its insulation upwelling,
    half through points drawn by deficit / strength -- and each cut is released
    in the balance (:func:`forces.release`, every other plate held at its
    terminal velocity).  A cut qualifies if

    * it lies within ``rift_max_angle`` degrees of its centre (the normal
      component of a relative rotation along a great circle goes as
      cos(theta - theta0), so a cut longer than ~180 degrees must converge
      somewhere -- the shipped rift's far side);
    * at most ``rift_max_conv`` of it (by length) would converge, and its
      10th-percentile opening is at least ``rift_end_open`` of the mean, so the
      halves' relative pole is not at one end of the cut (dyn-minimal's
      first rifts hinged at the pivot and the halves stayed one landmass for
      ~500 My);
    * both halves keep ``rift_half_min`` of the plate's continent and at least
      ``rift_min_half`` of the sphere, and each half is one connected piece.

    Its tension is the force per unit cut length that would hold the halves
    together -- the free opening rate times the halves' reduced drag, over the
    cut length -- divided by the cut's mean strength (cratons strong, young
    sutures weak).  The best cut rifts if its tension exceeds
    ``rift_tension`` (x a +-15 % per-plate jitter).  No kick: both halves keep
    the plate's motion; the rift starts held by its own strength G0,
    calibrated so it opens at ``rift_slow_cmyr``, and necks with opening
    (forces.assemble), so it is slow first and fast after (Brune et al.
    2016)."""
    from .segments import CONTINENTAL

    seg, plates, tp = sim.seg, sim.plates, sim.tp
    bal = getattr(sim, "balance", None)
    if bal is None or bal.rc is None or bal.D.shape[0] != seg.M:
        return {"event": "rift", "split": [], "pairs": []}
    k = int(sim.step_index)
    D, Ac = insulation_deficit(sim)
    cont = seg.kind == CONTINENTAL
    P = plates.P
    w = seg.ext * cont
    Dp = np.bincount(seg.plate_id, weights=D * w, minlength=P)[:P] / np.maximum(np.bincount(seg.plate_id, weights=w, minlength=P)[:P], 1e-30)
    refr = sim.steps_of(tp.rift_refractory_my)
    busy = {a for pr in sim.rift_pairs for a in pr}
    out = []
    diag = {}
    for q in np.flatnonzero(plates.alive & (Ac >= float(tp.rift_min_cont))):
        q = int(q)
        if q in busy or k - sim.last_rift.get(q, -10 ** 9) < refr:
            continue
        # the time gate: the insulation under the plate's continent, scaled gently by the
        # continent's size (a small continent's dome leaks heat sideways), against the plate's
        # strength (+-15 % jitter).  The released-cut tension alone is flat in time -- the
        # girdle's pull is there from the start -- so it cannot say *when*; it says where
        sigma = float(Dp[q]) * (float(Ac[q]) / float(tp.rift_size_ref)) ** float(tp.rift_size_exponent)
        jit = sim.rift_jitter.setdefault(q, 1.0 + 0.15 * (2.0 * float(rng.random()) - 1.0))
        diag[q] = round(sigma, 3)
        if sigma < float(tp.rift_deficit) * jit:
            continue
        r = _force_rift_one(sim, q, rng, D)
        diag[q] = (round(sigma, 3), r.get("best_score"))
        if r.get("split", -1) >= 0:
            out.append(r)
            busy.update((r["a"], r["b"]))
    sim.rift_diag = diag
    pairs = [(r["a"], r["b"]) for r in out]
    return {"event": "rift", "split": [r["a"] for r in out], "pairs": pairs,
            "centre": out[0]["centre"] if out else None, "detail": out, "plates": int(sim.plates.n_alive())}


def _force_rift_one(sim, q: int, rng, D: np.ndarray) -> dict:
    from scipy.spatial import cKDTree

    from . import forces
    from .segments import CONTINENTAL

    seg, plates, tp = sim.seg, sim.plates, sim.tp
    bal = sim.balance
    sp = sim.spacing
    sel = np.flatnonzero(seg.plate_id == q)
    if sel.size < 40:
        return {"split": -1}
    pos = seg.pos[sel]
    cont = seg.kind[sel] == CONTINENTAL
    ext = seg.ext[sel]
    Ctot = float((ext * cont).sum())
    pr = cKDTree(pos).query_pairs(1.25 * sp, output_type="ndarray")
    if pr.shape[0] == 0:
        return {"split": -1}
    a, b = pr[:, 0], pr[:, 1]
    strength = _strength(sim, sel)
    Dl = D[sel]
    wc = ext * cont * (0.05 + Dl)
    c = (pos * wc[:, None]).sum(axis=0)
    if np.linalg.norm(c) < 1e-9:
        return {"split": -1}
    c /= np.linalg.norm(c)
    cl = np.flatnonzero(cont)
    pw = (0.05 + Dl[cl]) / strength[cl]
    pw /= pw.sum()
    zig = float(tp.rift_cut_zigzag)
    noise = zig * fbm_at(pos, rng, octaves=3, base_freq=float(tp.rift_zigzag_freq)) if zig > 0.0 else 0.0
    half_min = max(float(tp.rift_half_min) * Ctot, float(tp.rift_min_half) * 4.0 * np.pi)
    max_ang = float(tp.rift_max_angle)
    drag = sim.drag_per_len()
    damp = float(tp.damping)
    m_o = float(tp.oceanic_thickness) * float(tp.oceanic_density)
    cms = sim.R_km * 0.1 / float(tp.myr_per_step)
    best = None
    best_any = None
    crs = seg.craton[sel]
    ucr = np.unique(crs[crs > 0])
    for n_try in range(int(tp.rift_candidates)):
        p0 = c if n_try % 2 == 0 else pos[cl[rng.choice(cl.size, p=pw)]]
        v = random_unit_vectors(rng, (1,))[0]
        nn = v - p0 * float(v @ p0)
        ln = float(np.linalg.norm(nn))
        if ln < 1e-9:
            continue
        nn /= ln
        side = (pos @ nn + noise) > 0.0
        for cq in ucr:
            m = crs == cq
            side[m] = bool(side[m].mean() > 0.5)
        side = _keep_connected(side, a, b, sel.size)
        cb_ = float((ext * cont * side).sum())
        if min(cb_, Ctot - cb_) < half_min:
            continue
        cross = side[a] != side[b]
        if int(cross.sum()) < 4:
            continue
        ia = np.where(side[a[cross]], b[cross], a[cross])          # A side (False)
        ib = np.where(side[a[cross]], a[cross], b[cross])          # B side (True)
        mid = pos[ia] + pos[ib]
        mid /= np.linalg.norm(mid, axis=1, keepdims=True)
        cc = mid.mean(axis=0)
        cc /= max(np.linalg.norm(cc), 1e-12)
        ang = np.degrees(np.arccos(np.clip(mid @ cc, -1.0, 1.0)))
        if float(ang.max()) > max_ang:
            continue
        wA, wB = forces.release(bal, seg, sel, side, sel[ia], sel[ib], drag)
        nt = nn[None, :] - mid * (mid @ nn)[:, None]
        nt /= np.maximum(np.linalg.norm(nt, axis=1, keepdims=True), 1e-12)
        opening = np.sum(np.cross(wB - wA, mid) * nt, axis=1)          # rad/step, > 0 opening
        lw = 0.5 * (np.sqrt(ext[ia]) + np.sqrt(ext[ib]))
        mean = float(np.average(opening, weights=lw))
        conv = float(lw[opening < 0].sum() / lw.sum())
        q10 = _wq(opening, lw, 0.1)
        L = float(lw.sum()) / 1.5                                       # pairs at 1.25 sp overcount the line ~1.5x
        dA = float(np.trace(bal.D[sel[~side]].sum(axis=0))) / 3.0
        dB = float(np.trace(bal.D[sel[side]].sum(axis=0))) / 3.0
        mu = dA * dB / max(dA + dB, 1e-30) / (damp * m_o)
        s_cut = float(np.average(0.5 * (strength[ia] + strength[ib]), weights=lw))
        score = mean * cms * mu / max(L, 1e-6) / max(s_cut, 1e-9)
        ok = (mean * cms >= float(tp.rift_min_open_cmyr) and conv <= float(tp.rift_max_conv)
              and q10 >= float(tp.rift_end_open) * mean)
        rec = dict(score=score, mean=mean, conv=conv, q10=q10, side=side, ia=ia, ib=ib, wA=wA, wB=wB, mid=mid, nt=nt,
                   lw=lw, centre=cc, ext_deg=2.0 * float(ang.max()), s_cut=s_cut, mu=mu, L=L, ok=ok)
        if best_any is None or score > best_any["score"]:
            best_any = rec
        if ok and (best is None or score > best["score"]):
            best = rec
    def _d(r):
        if r is None:
            return None
        return (round(r["score"], 2), round(r["mean"] * cms, 2), round(r["mu"], 2), round(r["L"], 2), round(r["s_cut"], 2),
                round(r["conv"], 3), round(r["q10"] / max(r["mean"], 1e-30), 2), round(float(Dl[cont].mean()), 2))
    if best is None:
        return {"split": -1, "best_score": ("x",) + (_d(best_any) or ())}
    if best["score"] < float(tp.rift_tension):
        return {"split": -1, "best_score": _d(best)}
    side, ia, ib, mid, nt, lw = best["side"], best["ia"], best["ib"], best["mid"], best["nt"], best["lw"]
    # the rift's strength: the coupling that holds the opening at rift_slow_cmyr.  A trial
    # solve at G = 1 (the coupling forces.assemble gives a rift pair: G times the smaller
    # half's basal drag, spread over the contacts) says how the opening falls with G
    basal = sim.basal_seg()[sel] * float(tp.damping)
    Idm = min(float(basal[~side].sum()), float(basal[side].sum()))
    kv1 = Idm / max(float(lw.sum()), 1e-12) * lw
    wA1, wB1 = forces.release(bal, seg, sel, side, sel[ia], sel[ib], drag, couple=kv1)
    r0 = best["mean"]
    r1 = float(np.average(np.sum(np.cross(wB1 - wA1, mid) * nt, axis=1), weights=lw))
    slow = sim.cmyr(tp.rift_slow_cmyr)
    if r1 > 1e-12 and r0 > r1 and r0 > slow:
        G0 = (r0 / slow - 1.0) / (r0 / r1 - 1.0)
    else:
        G0 = 0.0
    G0 = float(min(max(G0, 0.0), float(tp.rift_strength_max)))
    P = plates.P
    pid = seg.plate_id.copy()
    pid[sel[side]] = P
    new = _rebuild(seg, pid, P + 1, rng, float(tp.initial_speed) * sim.spacing, keep=plates.omega, snap=False)
    new.omega[P] = plates.omega[q]
    new.omega[~new.alive] = 0.0
    sim.plates = new
    k = int(sim.step_index)
    sim.rift_pairs[(int(q), int(P))] = {"k0": k, "delta": 0.0, "G0": G0}
    sim.last_rift[int(q)] = k
    sim.last_rift[int(P)] = k
    sim.rift_jitter.pop(int(q), None)
    A4 = 4.0 * np.pi
    return {"split": int(q), "a": int(q), "b": int(P), "k0": k, "centre": best["centre"].tolist(),
            "score": round(best["score"], 3), "free_cmyr": round(r0 * sim.R_km * 0.1 / float(tp.myr_per_step), 3),
            "G0": round(G0, 3), "conv": round(best["conv"], 4), "q10_ratio": round(best["q10"] / max(r0, 1e-30), 3),
            "extent_deg": round(best["ext_deg"], 1), "strength": round(best["s_cut"], 2),
            "area_a": float(seg.ext[sel[~side]].sum() / A4), "area_b": float(seg.ext[sel[side]].sum() / A4),
            "cont_a": float((seg.ext * (seg.kind == CONTINENTAL))[sel[~side]].sum() / A4),
            "cont_b": float((seg.ext * (seg.kind == CONTINENTAL))[sel[side]].sum() / A4),
            "best_score": round(best["score"], 3)}


def micro_merge(sim, rng) -> dict:
    """Small plates live and die (dyn-events' merge_microplates, on dyn-minimal's slab field).

    A plate below ``micro_area`` of the sphere that is not the down-going plate of a
    live slab (slab >= a quarter of saturation along its trenches) and is not one half
    of an open rift is *passive*; once it has been passive for ``micro_life_my`` it is
    captured by the neighbour with the longest shared boundary that it is not
    converging on (mean approach <= ``micro_conv_cmyr``).  A remnant below
    ``plate_min_area`` -- what a trench leaves of a nearly consumed plate -- is
    captured at once by the neighbour it shares most boundary with and is not diving
    under (on its own it is a few segments with a slab's pull and little drag, and
    spun at the speed cap; the Monterey and Arguello remnants of the Farallon plate
    were captured by the Pacific).  Small plates on Earth live 10-20 My (Morra 2013)."""
    seg, plates, tp = sim.seg, sim.plates, sim.tp
    cen = getattr(sim, "census_last", None)
    if cen is None or cen["i"].size == 0 or int(max(cen["i"].max(), cen["j"].max())) >= seg.M:
        return {"event": "micro_merge", "merges": []}
    k = int(sim.step_index)
    P = plates.P
    pid = seg.plate_id.astype(np.int64)
    area = np.bincount(pid, weights=seg.ext, minlength=P)[:P] / (4.0 * np.pi)
    slabbed = np.zeros(P, bool)
    si = getattr(sim, "slab_info", None)
    if si is not None and si["seg"].size and int(si["seg"].max()) < seg.M:
        live = si["seg"][si["g"] >= 0.25]
        slabbed[np.unique(pid[live])] = True
    rifting = np.zeros(P, bool)
    for a_, b_ in sim.rift_pairs:
        if a_ < P:
            rifting[a_] = True
        if b_ < P:
            rifting[b_] = True
    small = plates.alive & (area > 0) & (area < float(tp.micro_area)) & ~slabbed & ~rifting
    for q in list(sim.micro_passive):
        if q >= P or not small[q]:
            sim.micro_passive.pop(q, None)
    for q in np.flatnonzero(small):
        sim.micro_passive.setdefault(int(q), k)
    life = sim.steps_of(tp.micro_life_my)
    due = [q for q, s0 in sim.micro_passive.items() if k - s0 >= life]
    remnant = [int(q) for q in np.flatnonzero(plates.alive & (area > 0) & (area < float(tp.plate_min_area))) if int(q) not in due]
    todo = [(q, False) for q in due] + [(q, True) for q in remnant]
    if not todo:
        return {"event": "micro_merge", "merges": []}
    i, j = cen["i"], cen["j"]
    pi_, pj_ = pid[i], pid[j]
    v = np.cross(plates.omega[pid], seg.pos)
    appr = np.sum((v[i] - v[j]) * cen["dd"], axis=1)             # > 0: i converging on j
    w = cen["w"]
    # who dives under whom along each contact (the census's velocity-independent rule)
    thr = sim.cmyr(tp.micro_conv_cmyr)
    new_pid = pid.copy()
    gone: set = set()
    merges = []
    for q, is_rem in todo:
        if q in gone:
            continue
        mi = (pi_ == q) & (pj_ != q)
        mj = (pj_ == q) & (pi_ != q)
        if not (mi.any() or mj.any()):
            continue
        other = np.concatenate([pj_[mi], pi_[mj]])
        ap = np.concatenate([appr[mi], appr[mj]])                   # q converging on the other: > 0
        ww = np.concatenate([w[mi], w[mj]])
        qdown = np.concatenate([cen["down_i"][mi], cen["down_j"][mj]])
        best, best_w = None, 0.0
        for o in np.unique(other):
            o = int(o)
            if o in gone or not plates.alive[o]:
                continue
            m = other == o
            L = float(ww[m].sum())
            dives = bool(qdown[m].mean() > 0.5) and slabbed[q]
            if not is_rem:
                if float(np.average(ap[m], weights=ww[m])) > thr or dives:
                    continue
            elif dives and np.unique(other).size > 1:
                continue
            if L > best_w:
                best, best_w = o, L
        if best is None:
            continue
        new_pid[new_pid == q] = best
        gone.add(q)
        merges.append({"plate": int(q), "into": int(best), "area": float(area[q]), "remnant": bool(is_rem)})
    if merges:
        sim.plates = _rebuild(seg, new_pid, P, rng, float(tp.initial_speed) * sim.spacing, keep=plates.omega, snap=False)
        for m in merges:
            q = m["plate"]
            sim.micro_passive.pop(q, None)
            sim.last_rift.pop(q, None)
            for pr in [pr for pr in sim.rift_pairs if q in pr]:
                sim.rift_pairs.pop(pr)
    return {"event": "micro_merge", "merges": merges, "plates": int(sim.plates.n_alive())}


def _margin_test(sim, p: int, co: np.ndarray, oc: np.ndarray) -> dict | None:
    """Would the old floor of plate ``p`` go down under its own continent if it broke
    away?  The plate is released along its continental margin in the force balance
    (forces.release, every other plate at its terminal velocity), the floor carrying
    the pull and the bending resistance of a seed slab of ``margin_collapse_slab_km``
    at the margin -- the old, dense floor's own negative buoyancy (spontaneous
    initiation, Stern 2004) -- and the margin is tested for convergence: the forced
    part of initiation (Gurnis et al. 2004).  None when the plate has no margin."""
    from scipy.spatial import cKDTree

    from . import forces

    seg, tp = sim.seg, sim.tp
    bal = getattr(sim, "balance", None)
    if bal is None or bal.rc is None or bal.D.shape[0] != seg.M:
        return None
    sel = np.concatenate([co, oc])
    side = np.zeros(sel.size, bool)
    side[co.size:] = True                                              # B: the floor
    pr = cKDTree(seg.pos[sel]).query_pairs(1.25 * sim.spacing, output_type="ndarray")
    cross = side[pr[:, 0]] != side[pr[:, 1]]
    if int(cross.sum()) < 4:
        return None
    pr = pr[cross]
    ia = np.where(side[pr[:, 0]], pr[:, 1], pr[:, 0])                 # continent
    ib = np.where(side[pr[:, 0]], pr[:, 0], pr[:, 1])                 # floor
    ga, gb = sel[ia], sel[ib]
    pos = seg.pos
    d = pos[ga] - pos[gb]                                              # floor -> continent
    rm = pos[ga] + pos[gb]
    rm /= np.linalg.norm(rm, axis=1, keepdims=True)
    d -= np.sum(d * rm, axis=1, keepdims=True) * rm
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    ell = np.sqrt(seg.ext)
    cnt = np.bincount(gb, minlength=seg.M)
    ln = ell[gb] / np.maximum(cnt[gb], 1)
    g = min(sim.km(tp.margin_collapse_slab_km) / max(sim.km(tp.slab_sat_km), 1e-12), 1.0)
    a_age = np.sqrt(np.clip(seg.age[gb] / max(sim.steps_of(tp.slab_age_my), 1e-9), 0.0, 1.0))
    a_age = float(tp.slab_age_floor) + (1.0 - float(tp.slab_age_floor)) * a_age
    f = (float(tp.slab_force) * g * a_age * ln)[:, None] * d
    tq = sim.gain * np.cross(pos[gb], f)
    u = sim.cmyr(tp.slab_speed_cmyr)
    trench = sim.gain * float(tp.slab_force) / u if u > 0 else 0.0
    ac = np.cross(pos[gb], d)
    blk = (trench * g * ln)[:, None, None] * ac[:, :, None] * ac[:, None, :]
    wC, wO = forces.release(bal, seg, sel, side, ga, gb, sim.drag_per_len(), extra_t=(gb, tq), extra_D=(gb, blk))
    vO = np.cross(wO, pos[gb])
    vC = np.cross(wC, pos[ga])
    appr = np.sum((vO - vC) * d, axis=1)                               # > 0: the floor closes on the margin
    cms = sim.R_km * 0.1 / float(tp.myr_per_step)
    return {"approach_cmyr": float(np.average(appr, weights=ln)) * cms, "conv_share": float(ln[appr > 0].sum() / ln.sum())}


def margin_collapse(sim, rng) -> dict:
    """Old passive margins fail: the closing half of the Wilson cycle.

    Ocean floor that a rift left attached to its continent (an Atlantic) rides
    the continent's plate and, with nothing to subduct it, only ages -- in this
    model indefinitely, since trenches start nowhere else: the floor's mean age
    climbed 39 -> 179 My over 4000 steps and every continent ended ringed by
    passive margins.  On Earth such floor is the densest lithosphere there is,
    and margins this old (the oldest Atlantic floor is ~180 My, anything much
    older is gone) are where subduction starts or invades (the Lesser Antilles,
    Scotia, Gibraltar; Stern 2004).

    So every `margin_collapse_every` steps, a plate that is mostly continent
    (>= 30 % of its area) and carries ocean floor (>= `margin_collapse_min` of
    the sphere) whose floor along its own continental margin has a median age
    past `margin_collapse_my` (x a per-plate jitter of +-15 %) loses that ocean
    floor to a plate of its own, and a slab of `margin_collapse_slab_km` is put
    under the margin -- about what underthrusting needs before it sustains
    itself (Gurnis 2004).  No kick: the new plate starts with its old motion and
    goes down only if the forces (slab pull, the heat field) take it there.
    """
    from scipy.spatial import cKDTree

    from . import forces
    from .segments import CONTINENTAL

    seg, plates, tp = sim.seg, sim.plates, sim.tp
    myr = float(tp.myr_per_step)
    P = plates.P
    pid = seg.plate_id
    cont = seg.kind == CONTINENTAL
    A = np.bincount(pid, weights=seg.ext, minlength=P)[:P]
    C = np.bincount(pid[cont], weights=seg.ext[cont], minlength=P)[:P]
    out = []
    for p in np.flatnonzero(plates.alive & (C >= 0.3 * np.maximum(A, 1e-30))):
        p = int(p)
        oc = np.flatnonzero((pid == p) & ~cont)
        if oc.size == 0 or float(seg.ext[oc].sum()) < float(tp.margin_collapse_min) * 4.0 * np.pi:
            continue
        co = np.flatnonzero((pid == p) & cont)
        d, _ = cKDTree(seg.pos[co]).query(seg.pos[oc], k=1)
        margin = oc[d < 1.5 * sim.spacing]
        if margin.size < 8:
            continue
        jit = sim.collapse_jitter.setdefault(p, 1.0 + 0.15 * (2.0 * float(rng.random()) - 1.0))
        if float(np.median(seg.age[margin])) * myr < float(tp.margin_collapse_my) * jit:
            continue
        test = None
        if tp.margin_collapse_test:
            test = _margin_test(sim, p, co, oc)
            if test is None or test["approach_cmyr"] < float(tp.margin_collapse_min_cmyr) or test["conv_share"] < 0.5:
                continue
        Pn = plates.P
        new_pid = seg.plate_id.copy()
        new_pid[oc] = Pn
        new = _rebuild(seg, new_pid, Pn + 1, rng, float(tp.initial_speed) * sim.spacing, keep=plates.omega,
                       snap=not tp.variable_extent)
        new.omega[Pn] = plates.omega[p]
        new.omega[~new.alive] = 0.0
        sim.plates = plates = new
        sim.collapse_jitter.pop(p, None)
        if tp.slab_force > 0.0 and float(tp.margin_collapse_slab_km) > 0.0:
            s0 = float(tp.margin_collapse_slab_km) / sim.R_km
            flat = sim.slab.interior.reshape(-1).copy()
            forces.deposit_blobs(flat, sim.heat_tree, seg.pos[margin], np.full(margin.size, 0.4 * s0), 2.0 * sim.spacing)
            sim.slab.interior[...] = np.maximum(sim.slab.interior, np.minimum(flat.reshape(sim.slab.interior.shape), s0))
        out.append({"plate": p, "new": int(Pn), "moved": int(oc.size), "margin_age_my": float(np.median(seg.age[margin]) * myr),
                    "test": test})
        P = plates.P
        pid = seg.plate_id
    return {"event": "collapse", "collapses": out, "plates": int(sim.plates.n_alive())}


def split_disconnected(sim, min_segments: int = 16, link_factor: float = 1.6, rng=None, tree=None,
                       min_area: float = 0.0) -> dict:
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

    ``tree`` is a KD-tree of the cloud as it is now, when the caller has one:
    the step passes the tree it built after the collisions, since nothing
    between there and here moves a segment.  Building it again here (twice:
    the pairs and the orphan weld each built their own) was two of the four
    tree builds a step made, ~3 ms each on the Earth preset.

    With ``tectonics.variable_extent`` on, "at least" is a size: the piece's
    summed extent against ``min_area`` of the sphere, floored at
    ``min_segments`` design segments' worth (:func:`min_ground`).
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    from .collision import build_tree, kd_workers

    seg, plates = sim.seg, sim.plates
    if seg.M < 2:
        return {"event": "split", "split": 0}
    if tree is None or tree.n != seg.M:
        tree = build_tree(seg)
    pairs = tree.query_pairs(float(link_factor) * sim.spacing, output_type="ndarray")
    if pairs.shape[0] == 0:
        return {"event": "split", "split": 0}
    e = pairs[seg.plate_id[pairs[:, 0]] == seg.plate_id[pairs[:, 1]]]
    g = coo_matrix((np.ones(e.shape[0], np.int8), (e[:, 0], e[:, 1])), shape=(seg.M, seg.M))
    _, comp = connected_components(g, directed=False)
    sizes = np.bincount(comp)
    if sim.tp.variable_extent:
        big = np.bincount(comp, weights=seg.ext) >= min_ground(sim, min_area, min_segments)
    else:
        big = sizes >= int(min_segments)
    pid = seg.plate_id.copy()
    P = plates.P
    extra: list[np.ndarray] = []
    # Edges join same-plate segments only, so every component lies inside one plate.  Each
    # plate's components are found from that table rather than by masking the whole cloud
    # once per plate and once more per piece -- the same pieces in the same order (plates
    # ascending, each plate's components ascending, then the same argsort by size), so the
    # same new plate ids; it was most of this function's 5 ms a step on the Earth preset
    plate_of = np.empty(sizes.size, dtype=np.int64)
    plate_of[comp] = seg.plate_id
    n_comp = np.bincount(plate_of)
    by_plate = np.argsort(plate_of, kind="stable")
    first = np.concatenate(([0], np.cumsum(n_comp)[:-1]))
    new_of = np.full(sizes.size, -1, dtype=np.int64)       # component -> the plate it becomes
    orphan_c = np.zeros(sizes.size, dtype=bool)
    for p in np.flatnonzero(n_comp >= 2):
        cs = by_plate[first[p]:first[p] + n_comp[p]]
        for c in cs[np.argsort(sizes[cs])[::-1]][1:]:      # every piece but the largest
            if big[c]:
                new_of[c] = P + len(extra)
                extra.append(plates.omega[p].copy())
            else:
                orphan_c[c] = True
    to = new_of[comp]
    pid[to >= 0] = to[to >= 0]
    orphans = orphan_c[comp]
    # a segment still welded from a continental collision keeps the plate it welded onto:
    # it lies inside the plate it came from, so the rule below would hand it straight back
    # and the same pair would collide again next step (globe/tectonics/collision.py)
    if hasattr(seg, "weld"):
        orphans &= seg.weld <= 0
    if orphans.any() and sim.tp.variable_extent:
        _weld_whole(seg, pid, comp, orphans, pairs)
    elif orphans.any():
        # weld a fragment onto the plate around it: the commonest plate among
        # the nearest segments that are not part of the fragment itself
        idx = np.flatnonzero(orphans)
        k = min(9, seg.M)
        _, nb = tree.query(seg.pos[idx], k=k, workers=kd_workers(idx.size))
        nb = np.atleast_2d(nb).reshape(idx.size, k)
        near = np.where(orphans[nb], -1, pid[nb])
        for row, i in zip(near, idx):
            vals = row[row >= 0]
            if vals.size:
                pid[i] = np.bincount(vals).argmax()
    if not extra and not orphans.any():
        return {"event": "split", "split": 0}
    keep = np.vstack([plates.omega] + [np.asarray(o)[None, :] for o in extra]) if extra else plates.omega
    sim.plates = _rebuild(seg, pid, P + len(extra), rng, float(sim.tp.initial_speed) * sim.spacing, keep=keep,
                          snap=not sim.tp.variable_extent)
    return {"event": "split", "split": len(extra), "welded": int(orphans.sum()),
            "plates": int(sim.plates.n_alive())}


def _tally(frag: np.ndarray, plate: np.ndarray, n_comp: int) -> np.ndarray:
    """Per component (n_comp,), the plate that appears most often among the
    (frag, plate) votes, ties to the lower plate id; -1 where nothing voted."""
    win = np.full(int(n_comp), -1, dtype=np.int64)
    if frag.size == 0:
        return win
    K = int(plate.max()) + 1
    key, cnt = np.unique(frag.astype(np.int64) * K + plate, return_counts=True)
    f, p = key // K, key % K
    o = np.lexsort((p, -cnt, f))                     # per fragment: most votes, then lowest id
    f, p = f[o], p[o]
    first = np.unique(f, return_index=True)[1]
    win[f[first]] = p[first]
    return win


def _weld_whole(seg, pid: np.ndarray, comp: np.ndarray, orphans: np.ndarray, pairs: np.ndarray) -> None:
    """Weld every sub-threshold fragment, whole and in one pass, onto the
    plate it is embedded in: the plate owning most of its boundary -- the
    links (``pairs``, the split's own neighbour pairs) from its segments to
    segments that are not being welded themselves. In place on `pid`.

    The per-segment rule (the commonest plate among each segment's 9 nearest
    neighbours outside the fragment) only reached the fragment's rim, since
    an interior segment has no outside neighbour among its nine. A fragment
    was peeled one rim a step -- 150 segments kept 95 / 59 / 26 / 7 / 0 on
    their old plate over five calls -- and meanwhile its interior rode a pole
    it was no longer attached to, tearing it apart.

    A fragment whose only links lead into other fragments waits for them to
    be welded first; one with no links at all falls back to its segments' 9
    nearest neighbours, tallied over the whole fragment.
    """
    n_comp = int(comp.max()) + 1
    todo = orphans.copy()
    a = np.concatenate([pairs[:, 0], pairs[:, 1]])
    b = np.concatenate([pairs[:, 1], pairs[:, 0]])
    m = orphans[a] & (comp[a] != comp[b])
    a, b = a[m], b[m]
    while todo.any():
        live = todo[a] & ~todo[b]
        if not live.any():
            break
        win = _tally(comp[a[live]], pid[b[live]].astype(np.int64), n_comp)
        sel = todo & (win[comp] >= 0)
        pid[sel] = win[comp[sel]]
        todo &= ~sel
    if todo.any():
        from .collision import build_tree

        idx = np.flatnonzero(todo)
        k = min(9, seg.M)
        _, nb = build_tree(seg).query(seg.pos[idx], k=k, workers=-1)
        nb = np.atleast_2d(nb).reshape(idx.size, k)
        ok = ~todo[nb] & (comp[nb] != comp[idx][:, None])
        fr = np.broadcast_to(comp[idx][:, None], nb.shape)[ok]
        win = _tally(fr, pid[nb][ok].astype(np.int64), n_comp)
        sel = todo & (win[comp] >= 0)
        pid[sel] = win[comp[sel]]


def suture(sim, a: int, b: int, rng) -> dict:
    """Weld plates ``a`` and ``b`` into one: the smaller (by area) joins the
    larger and the merged plate turns about the inertia-weighted mean of
    the two poles, so the momentum of the collision carries on.  The
    emptied plate id stays dead (``alive`` is False once it has no
    segments), as after any other loss of a plate."""
    seg, plates = sim.seg, sim.plates
    if plates.area[b] > plates.area[a]:
        a, b = b, a
    pid = seg.plate_id.copy()
    pid[pid == b] = a
    om = plates.omega.copy()
    Ia, Ib = float(plates.inertia[a]), float(plates.inertia[b])
    om[a] = (Ia * om[a] + Ib * om[b]) / max(Ia + Ib, 1e-12)
    om[b] = 0.0
    sim.plates = _rebuild(seg, pid, plates.P, rng, float(sim.tp.initial_speed) * sim.spacing, keep=om,
                          snap=not sim.tp.variable_extent)
    return {"event": "suture", "kept": int(a), "joined": int(b), "plates": int(sim.plates.n_alive())}


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
