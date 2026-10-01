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
