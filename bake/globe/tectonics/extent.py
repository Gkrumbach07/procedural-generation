"""Make extent visible: continental ground the cloud does not show becomes segments.

With ``tectonics.variable_extent`` every segment carries its own ground (``Segments.ext``),
but the map is drawn from the point cloud: the splat reads extents only against each other
(a power-cell shift of under one spacing at a coast) and the viewer's crust-kind layer is
the nearest segment's label.  So the rendered continental area follows the *count* of
continental segments.  Shortening removes points (the extent_min merge) while everything
that gives ground back -- collapse, the cap's lateral flow, stretching -- only widens the
points that are left.  Measured on the shipped Earth preset the continental count fell
15000 -> 4300 over 8000 steps while the extent share stayed at 0.34-0.39: rendered / extent
0.55-0.65.

:func:`shed_extent` closes that gap per plate, both ways, and only *shows* ground -- it
never makes or thins any:

* while a plate's continents own more ground than their cells (``B_p`` = the sum of ext
  less the rolling Voronoi area ``Segments.area``) by ``band`` of a cell, the segment with
  the most unseen ground sheds a child onto its plate's margin sea floor.  The child takes
  the sea-floor point's place and covers exactly its cell, with the parent's own column,
  density, age and assembly age, so the parent gives up just the ground the child covers:
  crust, volume and continental ground are unchanged, only the point that shows them moved.
  (proto/mass-visible's child was thinned to belt thickness; that made 5-8 sr of ground per
  8000 steps out of nothing but the split's own churn -- the continents' largest ground
  source -- and its thin margins set sea level, measured by the critic.)  A plate with no
  sea floor of its own at its margin takes a neighbour's: the trench is pushed back.
* while they own less by ``band`` of a cell (margin erosion, shortened C-C losers), the
  coastal segment with the least ground for its cell hands its crust and ground to its
  nearest continental neighbour of the same plate and becomes sea floor over its cell.

The thinning that makes the ground to show lives in one rated process
(:func:`~globe.tectonics.orogeny.collapse_orogens`, ``orogen_collapse_my``), not here.

Churn: the balance reads the rolling area, not the step's label map (whose cells flicker as
plates slide past each other: on the label map the split and retreat counts were 57k each
over 8000 steps, 21 % of retreats undoing a child at most 10 steps old), a plate needs
``band`` cells of imbalance either way (hysteresis), and a point the balance flipped is left
alone for ``residence`` steps (``Segments.shown``).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .segments import CONTINENTAL, OCEANIC, Segments


def margin_slots(seg: Segments, tree: cKDTree | None = None, k: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where a continent can take ground: oceanic segments among the ``k - 1`` nearest
    neighbours of a continental segment.  Returns ``(slot, owner, own)``: the oceanic segment,
    the plate of the continental segment it borders, and whether the slot is that plate's own
    sea floor (a passive margin, where sediment progrades the shelf) or another plate's (an
    active margin, where the margin advancing is the trench being pushed back).  One slot can
    border several plates and is listed once per plate."""
    c = np.flatnonzero(seg.kind == CONTINENTAL)
    if c.size == 0 or seg.M < k:
        z = np.zeros(0, np.int64)
        return z, z, np.zeros(0, bool)
    tree = cKDTree(seg.pos) if tree is None else tree
    _, nb = tree.query(seg.pos[c], k=k)
    nb = np.atleast_2d(nb)[:, 1:]
    owner = np.broadcast_to(seg.plate_id[c][:, None], nb.shape)
    oc = seg.kind[nb] == OCEANIC
    P1 = int(seg.plate_id.max()) + 1
    key = np.unique(nb[oc].astype(np.int64) * P1 + owner[oc].astype(np.int64))
    slot, own_p = key // P1, key % P1
    return slot, own_p, seg.plate_id[slot] == own_p


def _merge_into(seg: Segments, m: np.ndarray, n: np.ndarray) -> None:
    """Merge segments ``m`` into ``n`` (index arrays, disjoint): ground, crust and mass add up,
    the column is the extent-weighted mean (the extent_min merge's rule), the assembly age is
    mixed by crust.  The caller drops or flips ``m``."""
    em, en = seg.ext[m], seg.ext[n]
    e = em + en
    vm, vn = em * seg.thickness[m], en * seg.thickness[n]
    mm, mn = em * seg.mass[m], en * seg.mass[n]
    seg.rework[n] = (mm * seg.rework[m] + mn * seg.rework[n]) / np.maximum(mm + mn, 1e-30)
    seg.age[n] = (mm * seg.age[m] + mn * seg.age[n]) / np.maximum(mm + mn, 1e-30)
    seg.thickness[n] = (vm + vn) / np.maximum(e, 1e-30)
    seg.mass[n] = (mm + mn) / np.maximum(e, 1e-30)
    seg.density[n] = seg.mass[n] / np.maximum(seg.thickness[n], 1e-12)
    seg.ext[n] = e


def shed_extent(seg: Segments, area: np.ndarray, spacing: float, ext_max: float, rng: np.random.Generator,
                step: int = 0, max_flips: int = 300, band: float = 1.0, split_at: float = 0.75,
                residence: int = 0, active: bool = True, merge: bool = True,
                ocean_th: float = 0.2, ocean_rho: float = 0.88, min_area: float = 0.25) -> dict:
    """One pass of the balance (see the module docstring), per plate.

    ``area`` (M,) is each segment's cell in steradians -- the rolling Voronoi area
    ``Segments.area`` -- floored at ``min_area`` design cells.  ``ext_max`` (design extents)
    splits a segment whatever its plate's balance.  A segment is a parent only while its own
    unseen ground (ext less its cell) holds ``split_at`` of the child's cell, and it never
    gives up so much that it is left under ``min_area`` design cells.  Points flipped
    within ``residence`` steps of ``step`` (``Segments.shown``) are neither retreated nor
    split onto, and a split parent is neither merged nor merged into in the same pass (a
    capped parent splitting while its plate retreated it drove one seed to a negative
    extent at step 95).  ``max_flips`` bounds the splits and the retreats of one pass.

    Counters: ``split``, ``fail`` (a parent with no slot), ``active`` (onto a neighbour's sea
    floor), ``merged`` (coast retreats), ``overrun_mass`` / ``overrun_ground`` (the sea floor
    the children covered), ``ground_shown`` (ground the children moved into view), ``dist``
    (summed parent-child distance, radians), ``flip_mass`` (the sea floor a retreated coast
    became), ``undo`` / ``undo_young`` (retreats of a split child, of one under four
    residence times old)."""
    out = {"split": 0, "fail": 0, "active": 0, "merged": 0, "overrun_mass": 0.0, "overrun_ground": 0.0,
           "ground_shown": 0.0, "dist": 0.0, "flip_mass": 0.0, "undo": 0, "undo_young": 0}
    d2 = float(spacing) ** 2
    M = seg.M
    c = seg.kind == CONTINENTAL
    if not c.any():
        return out
    A = np.maximum(np.asarray(area, np.float64)[:M], float(min_area) * d2)
    pid = seg.plate_id.astype(np.int64)
    P = int(pid.max()) + 1
    unseen = seg.ext - A
    B = np.bincount(pid[c], weights=unseen[c], minlength=P)
    settled = seg.shown <= int(step) - int(residence)        # not flipped by the balance lately
    capped = c & (seg.ext > float(ext_max) * d2)
    used = np.zeros(M, bool)
    gone = np.zeros(M, bool)
    tree = None

    # ---- splits: plates whose continents own more ground than the map shows
    grow = (B >= float(band) * float(min_area) * d2) | (np.bincount(pid[capped], minlength=P) > 0)
    children = []
    a_lo = float(min_area) * d2                         # the smallest cell a child can take
    if grow.any():
        cand = np.flatnonzero(c & grow[pid] & ((unseen >= float(split_at) * a_lo) | capped))
        if cand.size:
            tree = cKDTree(seg.pos)
            slot, owner, own = margin_slots(seg, tree)
            # the slots each plate borders, its own sea floor first
            by_plate = {}
            if slot.size:
                order = np.lexsort((~own, owner))
                bounds = np.flatnonzero(np.diff(owner[order])) + 1
                for grp in np.split(order, bounds):
                    by_plate[int(owner[grp[0]])] = (slot[grp], own[grp])
            # the cheapest child each plate can make: a parent that cannot pay for it is no
            # candidate (most continental segments hold a little unseen ground)
            cheap = np.full(P, np.inf)
            if slot.size:
                np.minimum.at(cheap, owner, A[slot])
            cand = cand[capped[cand] | ((unseen[cand] >= float(split_at) * cheap[pid[cand]])
                                        & (B[pid[cand]] >= float(band) * cheap[pid[cand]]))]
            # capped parents first, then by unseen ground
            cand = cand[np.lexsort((-unseen[cand], ~capped[cand]))]
        for par in cand:
            if len(children) >= int(max_flips):
                break
            p = int(pid[par])
            if not capped[par] and (B[p] < float(band) * a_lo or unseen[par] < float(split_at) * a_lo):
                continue
            sl, ow = by_plate.get(p, (np.zeros(0, np.int64), np.zeros(0, bool)))
            mine = ~used[sl] & settled[sl]
            pick = mine & ow
            is_active = False
            if not pick.any():
                if not active:
                    out["fail"] += 1
                    continue
                pick = mine & ~ow
                is_active = True
                if not pick.any():
                    out["fail"] += 1
                    continue
            cs = sl[pick]
            d = np.linalg.norm(seg.pos[cs] - seg.pos[par], axis=1)
            d0 = float(d.min())
            # among the nearest stretch of margin, not always the one nearest slot: a single
            # slot fed every step grows a peninsula out of one point of the coast
            near = cs[d <= 1.5 * d0 + spacing]
            q = int(near[int(rng.integers(near.size))]) if near.size > 1 else int(cs[int(np.argmin(d))])
            e_child = float(A[q])
            # visibility only: the child stands on ground the parent owned and the map did not
            # show, at the parent's column -- no ground made, none thinned
            if not capped[par]:
                if B[p] < float(band) * e_child or unseen[par] < float(split_at) * e_child:
                    continue
            if float(seg.ext[par]) - e_child < a_lo:
                continue
            used[q] = True
            used[par] = True                            # a parent is neither merged nor merged into
            B[p] -= e_child
            unseen[par] -= e_child
            children.append((int(par), q, is_active, e_child, float(np.linalg.norm(seg.pos[q] - seg.pos[par]))))
    # ---- coast retreats: plates whose continents own less ground than the map shows
    merges = []
    if merge and (B <= -float(band) * float(min_area) * d2).any():
        tree = cKDTree(seg.pos) if tree is None else tree
        short = B <= -float(band) * float(min_area) * d2
        ci = np.flatnonzero(c & settled & ~used & short[pid])
        if ci.size:
            _, nb = tree.query(seg.pos[ci], k=min(7, M))
            nb = np.atleast_2d(nb)[:, 1:]
            coastal = (seg.kind[nb] != CONTINENTAL).any(axis=1)
            ci, nb = ci[coastal], nb[coastal]
            ratio = seg.ext[ci] / A[ci]
            order = np.argsort(ratio, kind="stable")
            for r in order:
                if len(merges) >= int(max_flips):
                    break
                m = int(ci[r])
                p = int(pid[m])
                if used[m] or gone[m]:
                    continue
                ret = float(A[m])                           # its whole cell goes to the sea floor
                if B[p] > -float(band) * ret:
                    continue
                # the nearest continental neighbour of its own plate that is staying
                tgt = -1
                for q in nb[r]:
                    if seg.kind[q] == CONTINENTAL and pid[q] == p and not gone[q] and not used[q]:
                        tgt = int(q)
                        break
                if tgt < 0:
                    continue
                gone[m] = True
                used[tgt] = True                            # one merge into a segment per pass
                B[p] += ret
                ocean_nb = nb[r][seg.kind[nb[r]] != CONTINENTAL]
                merges.append((m, tgt, float(seg.age[ocean_nb].mean()) if ocean_nb.size else 0.0))
    if not children and not merges:
        return out
    if merges:
        mm = np.array([x[0] for x in merges], np.int64)
        tt = np.array([x[1] for x in merges], np.int64)
        # churn: how many of the retreats take back a split child, and how many a young one
        was = seg.shown[mm]
        out["undo"] = int((was > Segments.NEVER).sum())
        out["undo_young"] = int((was > int(step) - 4 * max(int(residence), 1)).sum())
        _merge_into(seg, mm, tt)
        out["merged"] = int(mm.size)
        # the point stays, as sea floor continuing the ocean beside it, over its cell: a
        # removed point left a hole the void fill refilled with continental crust (churn)
        seg.kind[mm] = OCEANIC
        seg.thickness[mm] = float(ocean_th)
        seg.density[mm] = float(ocean_rho)
        seg.mass[mm] = float(ocean_th) * float(ocean_rho)
        seg.ext[mm] = A[mm]
        seg.age[mm] = np.array([x[2] for x in merges])
        seg.rework[mm] = seg.age[mm]
        seg.craton[mm] = 0
        seg.weld[mm] = 0
        seg.h_ref[mm] = seg.thickness[mm] * (1.0 - seg.density[mm])
        seg.shown[mm] = int(step)
        out["flip_mass"] = float((seg.ext[mm] * seg.mass[mm]).sum())
        gone[mm] = False
    keep = ~gone
    child = None
    if children:
        par = np.array([x[0] for x in children], np.int64)
        slt = np.array([x[1] for x in children], np.int64)
        act = np.array([x[2] for x in children], bool)
        e_child = np.array([x[3] for x in children])
        seg.ext[par] -= e_child
        out["split"] = int(par.size)
        out["active"] = int(act.sum())
        out["ground_shown"] = float(e_child.sum())
        out["dist"] = float(sum(x[4] for x in children))
        out["overrun_mass"] = float((seg.ext[slt] * seg.mass[slt]).sum())
        out["overrun_ground"] = float(seg.ext[slt].sum())
        th, rho = seg.thickness[par], seg.density[par]
        # the parent's crust, moved: its column, its density, as old as it was, assembled when it was
        child = Segments(seg.pos[slt].copy(), th.copy(), rho.copy(), seg.age[par].copy(), seg.plate_id[par].copy(), e_child,
                         h_ref=th * (1.0 - rho), mass=seg.mass[par].copy(), kind=CONTINENTAL, craton=0, weld=0,
                         ext=e_child, rework=seg.rework[par].copy(), shown=int(step))
        keep[slt] = False
    seg.compress(keep)
    if child is not None:
        seg.append(child)
    return out


__all__ = ["margin_slots", "shed_extent"]
