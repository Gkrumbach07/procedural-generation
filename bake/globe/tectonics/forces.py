"""Boundary forces on top of the heat-gradient drive (dyn-minimal prototype).

The heat field stays the driver: a plate is pushed up the heat gradient, and
by Stokes that torque is the line integral of heat around the plate's
boundary.  What the heat model cannot express is anything *one-sided* or
anything *resisting*, so this module adds exactly those, all of them read off
a velocity-independent census of the plate boundaries taken every step:

* **slab pull** -- a force per unit trench length on the plate whose ocean
  floor goes down, along the line to the overriding plate.  Its size is set
  by the slab hanging there (the `slab` field: slab length in the mantle
  frame, fed by subduction, saturating at ``slab_sat_km`` and detaching with
  ``slab_detach_my``) and by the slab's age (sqrt, thicker lithosphere), not
  by how fast the trench is consuming this step.  The shipped ``slab_pull``
  counted this step's consumed segments, which is a velocity-proportional
  force -- negative drag -- and rode the speed cap.  In heat units: a slab
  pull of ``slab_force`` per unit trench length is the same torque as a heat
  step of ``slab_force`` across that trench (Stokes).
* **boundary drag** -- plate boundaries resist motion (transform friction,
  slab bending): a drag on the plate's boundary length, so a small plate is
  no longer driven by its perimeter against an area-sized drag.  Expressed
  as a length: the boundary of a plate drags like a strip ``boundary_drag_km``
  wide of oceanic lithosphere on its base.
* **collisional resistance** -- continent-continent contacts that are
  converging couple the two plates' normal motion, as a drag on the
  *relative* velocity (``collision_drag`` x the boundary drag per unit
  length), solved implicitly so it slows convergence and never locks it.
* **rift strength** -- a newly cut rift couples its two halves (normal
  motion, both signs) with a strength that weakens as the rift extends
  (strain weakening, e-folding over ``rift_weaken_km`` of opening): the slow
  phase of a rift, then the fast one (Brune et al. 2016).

The plate velocity update keeps the shipped structure: with every term off,
``omega* = gain * tau / (I * damping)`` and ``omega += damping * (omega* -
omega)`` is exactly ``omega = (1 - damping) * omega + gain * tau / I``.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from .segments import CONTINENTAL, OCEANIC


def deposit_blobs(flat: np.ndarray, cell_tree, pts: np.ndarray, peaks: np.ndarray, radius: float) -> None:
    """``flat += peak_i * exp(-d^2 / (2 (radius/2)^2))`` for cells within
    ``radius`` of each point -- `CellTree.add_blobs` with a peak per point."""
    if pts.shape[0] == 0:
        return
    sigma2 = (0.5 * radius) ** 2
    lists = cell_tree.tree.query_ball_point(pts, radius, workers=1)
    lens = np.fromiter((len(l) for l in lists), dtype=np.int64, count=len(lists))
    if lens.sum() == 0:
        return
    cells = np.concatenate([np.asarray(l, dtype=np.int64) for l in lists])
    src = np.repeat(np.arange(pts.shape[0]), lens)
    d2 = np.sum((cell_tree.centers[cells] - pts[src]) ** 2, axis=1)
    np.add.at(flat, cells, np.asarray(peaks)[src] * np.exp(-d2 / (2.0 * sigma2)))


def census(seg, plates, spacing: float, radius_factor: float = 1.25, tree: cKDTree | None = None) -> dict:
    """Every cross-plate pair within ``radius_factor`` spacings: who goes
    down (crust type first, then the plate whose crust is older along that
    boundary -- the same rules `collide` applies), the unit direction from
    ``i`` to ``j``, the approach rate (current omegas; > 0 converging) and a
    per-pair share of boundary length, so that summing ``w`` over a plate's
    pairs gives its boundary length (radians)."""
    from .collision import plate_pair_polarity

    tree = cKDTree(seg.pos) if tree is None else tree
    pr = tree.query_pairs(radius_factor * spacing, output_type="ndarray")
    pid = seg.plate_id.astype(np.int64)
    if pr.shape[0]:
        pr = pr[pid[pr[:, 0]] != pid[pr[:, 1]]]
    i, j = (pr[:, 0], pr[:, 1]) if pr.shape[0] else (np.zeros(0, np.int64), np.zeros(0, np.int64))
    pos = seg.pos
    d = pos[j] - pos[i]
    dn = np.maximum(np.linalg.norm(d, axis=1), 1e-12)
    dd = d / dn[:, None]
    rm = pos[i] + pos[j]
    rm /= np.maximum(np.linalg.norm(rm, axis=1, keepdims=True), 1e-12)
    dd -= np.sum(dd * rm, axis=1, keepdims=True) * rm          # tangent at the contact
    dd /= np.maximum(np.linalg.norm(dd, axis=1, keepdims=True), 1e-12)
    om = plates.omega
    vi = np.cross(om[pid[i]], pos[i])
    vj = np.cross(om[pid[j]], pos[j])
    appr = np.sum((vi - vj) * dd, axis=1)
    ki, kj = seg.kind[i], seg.kind[j]
    Pm = int(pid.max()) + 1 if pid.size else 1
    pol = plate_pair_polarity(seg.plate_id, seg.age, seg.kind, np.ascontiguousarray(pr), Pm) if pr.shape[0] else None
    down_i = np.zeros(i.size, bool)
    down_j = np.zeros(i.size, bool)
    mixed = ki != kj
    down_i[mixed & (ki == OCEANIC)] = True
    down_j[mixed & (kj == OCEANIC)] = True
    oo = (ki == OCEANIC) & (kj == OCEANIC)
    if pol is not None:
        pij = pol[pid[i], pid[j]] == 1
        down_i[oo & pij] = True
        down_j[oo & ~pij] = True
    M = seg.M
    cnt = np.bincount(i, minlength=M) + np.bincount(j, minlength=M)
    ell = np.sqrt(seg.ext) if hasattr(seg, "ext") else np.full(M, spacing)
    li = ell[i] / np.maximum(cnt[i], 1)
    lj = ell[j] / np.maximum(cnt[j], 1)
    return dict(i=i, j=j, pi=pid[i], pj=pid[j], dd=dd, rm=rm, appr=appr, ki=ki, kj=kj,
                down_i=down_i, down_j=down_j, li=li, lj=lj, w=0.5 * (li + lj), cnt=cnt)


def slab_torques(seg, cen: dict, P: int, slab_at: np.ndarray, force: float, sat: float, age_ref: float,
                 age_floor: float) -> tuple[np.ndarray, dict]:
    """Slab pull per plate (P, 3), heat-torque units.  On every down-going
    *oceanic* boundary segment where a slab hangs (``slab_at`` > 0, slab
    length in radians at the segment), a force ``force * min(slab/sat, 1) *
    w(age) * length`` towards the overriding plate.  Velocity enters nowhere:
    a stalled trench with a slab still pulls; a contact with no slab (a ridge,
    a new passive margin) does not."""
    tau = np.zeros((P, 3))
    info = dict(n=0, L=0.0, seg=[], plate=[], g=[], len=[], dirn=[])
    if cen["i"].size == 0 or force <= 0.0:
        return tau, info
    pos = seg.pos
    a_age = np.sqrt(np.clip(seg.age / max(age_ref, 1e-9), 0.0, 1.0))
    a_age = age_floor + (1.0 - age_floor) * a_age
    parts = []
    for side, other_sign in (("i", 1.0), ("j", -1.0)):
        m = cen["down_" + side] & ((cen["ki"] if side == "i" else cen["kj"]) == OCEANIC)
        if not m.any():
            continue
        s = cen[side][m]
        g = np.clip(slab_at[s] / max(sat, 1e-12), 0.0, 1.0)
        ln = (cen["li"] if side == "i" else cen["lj"])[m]
        mag = force * g * a_age[s] * ln
        f = (other_sign * mag)[:, None] * cen["dd"][m]          # i -> j is towards the overrider when i goes down
        tq = np.cross(pos[s], f)
        pp = cen["pi"][m] if side == "i" else cen["pj"][m]
        for k in range(3):
            tau[:, k] += np.bincount(pp, weights=tq[:, k], minlength=P)[:P]
        info["n"] += int((g > 0).sum())
        info["L"] += float(ln[g > 0].sum())
        keep = g > 0
        info["seg"].append(s[keep]); info["plate"].append(pp[keep]); info["g"].append(g[keep])
        info["len"].append(ln[keep]); info["dirn"].append(other_sign * cen["dd"][m][keep])
    for kk in ("seg", "plate", "g", "len"):
        info[kk] = np.concatenate(info[kk]) if info[kk] else np.zeros(0)
    info["dirn"] = np.concatenate(info["dirn"]) if info["dirn"] else np.zeros((0, 3))
    return tau, info


def solve_omega(plates, tau_gain: np.ndarray, damping: float, seg, cen: dict | None, drag_per_len: float,
                cc_per_len: float, rift_pairs: dict | None, rift_per_len: float, rift_weaken: float,
                slab: dict | None = None, trench_per_len: float = 0.0, basal: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """omega* from (I d + R_p) omega*_p + sum_c K_c a a^T (omega*_p - omega*_q) = gain tau_p,
    on the live plates only.  ``tau_gain`` is ``gain * tau`` (P, 3)."""
    P = plates.P
    alive = np.flatnonzero(plates.alive)
    n = alive.size
    loc = np.full(P, -1, np.int64)
    loc[alive] = np.arange(n)
    A = np.zeros((3 * n, 3 * n))
    I3 = np.eye(3)
    # basal drag: the shipped I = sum(area * column mass) unless a keel-based drag is given
    Id = np.maximum((plates.inertia if basal is None else basal)[alive], 1e-12) * damping
    for a in range(n):
        A[3 * a:3 * a + 3, 3 * a:3 * a + 3] = Id[a] * I3
    info = dict(cc_pairs=0, cc_len=0.0, rift_len=0.0, drag_share=0.0)
    if slab is not None and trench_per_len > 0.0 and slab["seg"].size:
        # slab resistance (bending, the interface): a drag on the normal motion of the slab's
        # own plate at its trench, as long as the pull, so a slab-attached plate saturates at
        # slab_force * gain / trench_per_len whatever its size (Forsyth & Uyeda 1975)
        r = seg.pos[slab["seg"].astype(np.int64)]
        ac = np.cross(r, slab["dirn"])
        kv = trench_per_len * slab["g"] * slab["len"]
        blk = kv[:, None, None] * ac[:, :, None] * ac[:, None, :]
        pp = loc[slab["plate"].astype(np.int64)]
        ok = pp >= 0
        Bt = np.zeros((n, 3, 3))
        np.add.at(Bt, pp[ok], blk[ok])
        for a in range(n):
            A[3 * a:3 * a + 3, 3 * a:3 * a + 3] += Bt[a]
        tr = np.trace(Bt, axis1=1, axis2=2)
        info["trench_share"] = float(np.median((tr / (tr + Id))[tr > 0])) if (tr > 0).any() else 0.0
    if cen is not None and cen["i"].size:
        pos = seg.pos
        if drag_per_len > 0.0:
            # boundary drag: sum over a plate's boundary segments of len * (I - r r^T)
            Dm = np.zeros((P, 3, 3))
            for side in ("i", "j"):
                s = cen[side]
                ln = cen["l" + side]
                pp = cen["p" + side]
                r = pos[s]
                proj = I3[None, :, :] - r[:, :, None] * r[:, None, :]
                for x in range(3):
                    for y in range(3):
                        Dm[:, x, y] += np.bincount(pp, weights=ln * proj[:, x, y], minlength=P)[:P]
            Dm *= drag_per_len
            for a, q in enumerate(alive):
                A[3 * a:3 * a + 3, 3 * a:3 * a + 3] += Dm[q]
            tr = np.trace(Dm[alive], axis1=1, axis2=2) / 2.0
            info["drag_share"] = float(np.median(tr / (tr + Id))) if n else 0.0

        def couple(mask, kvec):
            if not mask.any():
                return 0.0
            ac = np.cross(cen["rm"][mask], cen["dd"][mask])
            blk = (kvec[:, None, None]) * ac[:, :, None] * ac[:, None, :]
            pa, pb = loc[cen["pi"][mask]], loc[cen["pj"][mask]]
            ok = (pa >= 0) & (pb >= 0)
            key = pa[ok] * n + pb[ok]
            blk = blk[ok]
            uk, inv = np.unique(key, return_inverse=True)
            B = np.zeros((uk.size, 3, 3))
            np.add.at(B, inv, blk)
            for kk, Bk in zip(uk, B):
                qa, qb = int(kk // n), int(kk % n)
                A[3 * qa:3 * qa + 3, 3 * qa:3 * qa + 3] += Bk
                A[3 * qb:3 * qb + 3, 3 * qb:3 * qb + 3] += Bk
                A[3 * qa:3 * qa + 3, 3 * qb:3 * qb + 3] -= Bk
                A[3 * qb:3 * qb + 3, 3 * qa:3 * qa + 3] -= Bk
            return float(np.sum(kvec[ok]))

        if cc_per_len > 0.0:
            cc = (cen["ki"] == CONTINENTAL) & (cen["kj"] == CONTINENTAL) & (cen["appr"] > 0.0)
            if rift_pairs:
                # a rift's own two halves are coupled by its strength below, not by collision
                rk = np.array([a * 1000003 + b for a, b in rift_pairs] + [b * 1000003 + a for a, b in rift_pairs], np.int64)
                cc &= ~np.isin(cen["pi"] * 1000003 + cen["pj"], rk)
            info["cc_pairs"] = int(cc.sum())
            info["cc_len"] = couple(cc, cc_per_len * cen["w"][cc]) / max(cc_per_len, 1e-30)
        if rift_pairs and rift_per_len > 0.0:
            # rift_per_len is the rift's strength G0: the coupling, summed over the rift's
            # contacts, is G0 times the smaller half's own (basal) drag, so it holds back the
            # opening by about 1 / (1 + G) whatever the size of the plates; it weakens with
            # opening, exp(-delta / rift_weaken)
            key = cen["pi"] * 1000003 + cen["pj"]
            kv = np.zeros(cen["i"].size)
            m = np.zeros(cen["i"].size, bool)
            for (a, b), st in rift_pairs.items():
                mm = (key == a * 1000003 + b) | (key == b * 1000003 + a)
                if not mm.any() or loc[a] < 0 or loc[b] < 0:
                    continue
                Lc = float(cen["w"][mm].sum())
                # necking: the strength holds while the rift stretches its first ~rift_weaken,
                # then goes (a Gaussian in the opening, not an exponential: Brune et al. 2016's
                # slow phase ends in a few My, not over the whole extension)
                g = rift_per_len * math.exp(-(st["delta"] / max(rift_weaken, 1e-12)) ** 2)
                kv[mm] = g * min(Id[loc[a]], Id[loc[b]]) / max(Lc, 1e-12) * cen["w"][mm]
                m |= mm
            info["rift_len"] = float(cen["w"][m].sum())
            couple(m, kv[m])
    rhs = np.asarray(tau_gain)[alive].reshape(-1)
    # scipy's LAPACK, not numpy's: numpy's OpenBLAS spins up its whole thread pool for a
    # 3P x 3P system and took 0.27-0.45 s a solve on a loaded machine (scipy: ~2 ms)
    import scipy.linalg as sla
    try:
        sol = sla.solve(A, rhs, check_finite=False)
    except (sla.LinAlgError, ValueError):
        sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
    wstar = np.zeros((P, 3))
    wstar[alive] = sol.reshape(n, 3)
    return wstar, info


def rift_opening(cen: dict, rift_pairs: dict) -> dict:
    """Mean opening rate (radians/step, > 0 opening) over each rift pair's contacts."""
    out = {}
    if not rift_pairs or cen is None or cen["i"].size == 0:
        return out
    key = cen["pi"] * 1000003 + cen["pj"]
    for (a, b) in rift_pairs:
        m = (key == a * 1000003 + b) | (key == b * 1000003 + a)
        out[(a, b)] = float(np.average(-cen["appr"][m], weights=cen["w"][m])) if m.any() else None
    return out


__all__ = ["census", "slab_torques", "solve_omega", "deposit_blobs", "rift_opening"]
