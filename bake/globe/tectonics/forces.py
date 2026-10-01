"""Plate driving and resisting forces as an overdamped balance (synth-dyn).

The heat field stays the mantle's flow potential: a plate is pushed up the
heat gradient, and by Stokes that torque is the line integral of heat around
the plate's boundary.  What the heat model cannot express is anything
*one-sided* or anything *resisting*, so this module adds exactly those, all
read off a velocity-independent census of the plate boundaries taken every
step (who goes down is decided by crust type and by the age of each side's
crust along that boundary, never by the velocities):

* **slab pull** -- a force per unit trench length on the plate whose ocean
  floor goes down, along the line to the overriding plate.  Its size is set
  by the slab hanging there (the ``slab`` field: slab length in the mantle
  frame, fed by subduction, saturating at ``slab_sat_km`` and detaching with
  ``slab_detach_my``) and by the slab's age (sqrt, thicker lithosphere), not
  by how fast the trench is consuming this step.  The shipped ``slab_pull``
  counted this step's consumed segments, which is a velocity-proportional
  force -- negative drag -- and rode the speed cap.  ``slab_onset_km`` > 0
  gives the pull a smoothstep onset between 0.5x and 1.5x that length of
  slab: a contact needs ~100-150 km of underthrusting before its slab pulls
  (Gurnis et al. 2004).
* **slab resistance** (bending, the interface) -- a drag on the normal motion
  of the slab's own plate at its trench, as long as the pull, so slab-attached
  plates saturate at ``slab_speed_cmyr`` whatever their size (Forsyth & Uyeda
  1975).
* **boundary drag** -- plate boundaries resist motion (transform friction):
  a drag on the plate's boundary length, so a small plate is not driven by
  its perimeter against an area-sized drag.
* **basal drag** -- oceanic lithosphere drags like its column, continents
  ``basal_drag_continental`` x that (keels), cratons 1.5x more.
* **collisional resistance** -- continent-continent contacts that are
  converging couple the two plates' normal motion, as a drag on the
  *relative* velocity, solved implicitly so it slows convergence and never
  locks it.
* **rift strength** -- a newly cut rift couples its two halves (normal
  motion, both signs) with a strength that weakens as the rift extends
  (necking): the slow phase of a rift, then the fast one (Brune et al. 2016).

Everything is assembled **per segment** first (a 3x3 drag block and a torque
per segment, and segment-pair coupling blocks), then summed per plate and
solved.  The per-segment form is what makes a *trial* cheap: releasing a cut
through one plate (a candidate rift, a failing passive margin) is two sums
over that plate's segments and a 6x6 solve, with every other plate held at
its terminal velocity (:func:`release`).

With every boundary term off the solve is exactly the shipped
``omega = (1 - damping) omega + gain tau / I`` (tests/test_dyn_minimal.py).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from .segments import CONTINENTAL, OCEANIC

KEY = 1000003


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


def all_pairs(seg, spacing: float, radius_factor: float = 1.25, tree: cKDTree | None = None) -> np.ndarray:
    """Every pair of segments within ``radius_factor`` spacings (unfiltered)."""
    tree = cKDTree(seg.pos) if tree is None else tree
    return tree.query_pairs(radius_factor * spacing, output_type="ndarray")


def census(seg, plates, spacing: float, radius_factor: float = 1.25, tree: cKDTree | None = None,
           pairs: np.ndarray | None = None, labels: np.ndarray | None = None) -> dict:
    """Every cross-plate pair within ``radius_factor`` spacings: who goes
    down (crust type first, then the plate whose crust is older along that
    boundary -- the same rules `collide` applies), the unit direction from
    ``i`` to ``j``, the approach rate (current omegas; > 0 converging) and a
    per-pair share of boundary length, so that summing ``w`` over a plate's
    pairs gives its boundary length (radians).  ``pairs`` (all pairs, any
    plate) and ``labels`` (plate ids) may be given to reuse a query or to
    take the census of a hypothetical partition."""
    from .collision import plate_pair_polarity

    pr = all_pairs(seg, spacing, radius_factor, tree) if pairs is None else pairs
    pid = (seg.plate_id if labels is None else labels).astype(np.int64)
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
    pi_, pj_ = pid[i], pid[j]
    oi = om[np.minimum(pi_, om.shape[0] - 1)] if om.shape[0] else np.zeros((i.size, 3))
    oj = om[np.minimum(pj_, om.shape[0] - 1)] if om.shape[0] else np.zeros((i.size, 3))
    vi = np.cross(oi, pos[i])
    vj = np.cross(oj, pos[j])
    appr = np.sum((vi - vj) * dd, axis=1)
    ki, kj = seg.kind[i], seg.kind[j]
    Pm = int(pid.max()) + 1 if pid.size else 1
    pol = plate_pair_polarity(pid.astype(np.int32), seg.age, seg.kind, np.ascontiguousarray(pr), Pm) if pr.shape[0] else None
    down_i = np.zeros(i.size, bool)
    down_j = np.zeros(i.size, bool)
    mixed = ki != kj
    down_i[mixed & (ki == OCEANIC)] = True
    down_j[mixed & (kj == OCEANIC)] = True
    oo = (ki == OCEANIC) & (kj == OCEANIC)
    if pol is not None:
        pij = pol[pi_, pj_] == 1
        down_i[oo & pij] = True
        down_j[oo & ~pij] = True
    M = seg.M
    cnt = np.bincount(i, minlength=M) + np.bincount(j, minlength=M)
    ell = np.sqrt(seg.ext) if hasattr(seg, "ext") else np.full(M, spacing)
    li = ell[i] / np.maximum(cnt[i], 1)
    lj = ell[j] / np.maximum(cnt[j], 1)
    return dict(i=i, j=j, pi=pi_, pj=pj_, dd=dd, rm=rm, appr=appr, ki=ki, kj=kj,
                down_i=down_i, down_j=down_j, li=li, lj=lj, w=0.5 * (li + lj), cnt=cnt)


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def slab_contrib(seg, cen: dict, slab_at: np.ndarray, force: float, sat: float, age_ref: float,
                 age_floor: float, onset: float = 0.0) -> dict:
    """Slab pull per down-going *oceanic* contact (heat-torque units, before
    the gain): ``force * g * w(age) * length`` towards the overriding plate,
    with ``g = min(slab / sat, 1)`` (times a smoothstep onset over
    ``[0.5, 1.5] * onset`` when ``onset`` > 0).  Velocity enters nowhere: a
    stalled trench with a slab still pulls; a contact with no slab (a ridge,
    a new passive margin) does not.  Returns per-contact arrays: the
    segment, its plate, g, length, the pull direction and the torque."""
    out = dict(seg=np.zeros(0, np.int64), plate=np.zeros(0, np.int64), g=np.zeros(0), len=np.zeros(0),
               dirn=np.zeros((0, 3)), tq=np.zeros((0, 3)), n=0, L=0.0)
    if cen["i"].size == 0 or force <= 0.0:
        return out
    pos = seg.pos
    a_age = np.sqrt(np.clip(seg.age / max(age_ref, 1e-9), 0.0, 1.0))
    a_age = age_floor + (1.0 - age_floor) * a_age
    parts = []
    for side, sign in (("i", 1.0), ("j", -1.0)):
        m = cen["down_" + side] & ((cen["ki"] if side == "i" else cen["kj"]) == OCEANIC)
        if not m.any():
            continue
        s = cen[side][m]
        S = slab_at[s]
        g = np.clip(S / max(sat, 1e-12), 0.0, 1.0)
        if onset > 0.0:
            g = g * _smoothstep((S - 0.5 * onset) / max(onset, 1e-12))
        keep = g > 0
        if not keep.any():
            continue
        s = s[keep]
        g = g[keep]
        ln = (cen["li"] if side == "i" else cen["lj"])[m][keep]
        dirn = sign * cen["dd"][m][keep]                 # towards the overrider
        pp = (cen["pi"] if side == "i" else cen["pj"])[m][keep]
        mag = force * g * a_age[s] * ln
        tq = np.cross(pos[s], mag[:, None] * dirn)
        parts.append((s, pp, g, ln, dirn, tq))
    if not parts:
        return out
    out["seg"] = np.concatenate([p[0] for p in parts]).astype(np.int64)
    out["plate"] = np.concatenate([p[1] for p in parts]).astype(np.int64)
    out["g"] = np.concatenate([p[2] for p in parts])
    out["len"] = np.concatenate([p[3] for p in parts])
    out["dirn"] = np.concatenate([p[4] for p in parts])
    out["tq"] = np.concatenate([p[5] for p in parts])
    out["n"] = int(out["seg"].size)
    out["L"] = float(out["len"].sum())
    return out


def slab_torques(seg, cen: dict, P: int, slab_at: np.ndarray, force: float, sat: float, age_ref: float,
                 age_floor: float, onset: float = 0.0) -> tuple[np.ndarray, dict]:
    """Slab pull per plate (P, 3), heat-torque units, and the per-contact info."""
    info = slab_contrib(seg, cen, slab_at, force, sat, age_ref, age_floor, onset)
    tau = np.zeros((P, 3))
    if info["seg"].size:
        for k in range(3):
            tau[:, k] += np.bincount(info["plate"], weights=info["tq"][:, k], minlength=P)[:P]
    return tau, info


def _outer(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    return k[:, None, None] * a[:, :, None] * a[:, None, :]


class Balance:
    """The force balance of one step, per segment.

    ``D`` (M, 3, 3): each segment's share of its plate's drag (basal,
    boundary, slab resistance, and the diagonal of every coupling it is an
    end of).  ``t`` (M, 3): its share of the driving torque, times the gain.
    ``ci, cj, cB``: segment-pair coupling blocks between plates (collisional
    resistance, rift strength); a plate pair's off-diagonal block is minus
    their sum.  ``wstar`` is the solution, ``rc`` (M, 3) the coupling
    blocks applied to the partner plate's terminal velocity (what a trial
    holds fixed)."""

    def __init__(self, M: int):
        self.D = np.zeros((M, 3, 3))
        self.t = np.zeros((M, 3))
        self.ci = np.zeros(0, np.int64)
        self.cj = np.zeros(0, np.int64)
        self.cB = np.zeros((0, 3, 3))
        self.wstar = None
        self.rc = None
        self.extra = None
        self.info: dict = {}

    def add_D(self, s: np.ndarray, blocks: np.ndarray) -> None:
        if s.size:
            np.add.at(self.D, s, blocks)

    def add_coupling(self, ci: np.ndarray, cj: np.ndarray, B: np.ndarray) -> None:
        if ci.size == 0:
            return
        np.add.at(self.D, ci, B)
        np.add.at(self.D, cj, B)
        self.ci = np.concatenate([self.ci, ci.astype(np.int64)])
        self.cj = np.concatenate([self.cj, cj.astype(np.int64)])
        self.cB = np.concatenate([self.cB, B])


def assemble(seg, plates, cen: dict, *, gain: float, damping: float, grad3: np.ndarray | None,
             basal_seg: np.ndarray, drag_per_len: float, cc_per_len: float, slab: dict | None,
             trench_per_len: float, rift_pairs: dict | None, rift_scale: float, rift_weaken: float,
             rift_power: float = 2.0, rift_strength: float = 4.0) -> Balance:
    """Per-segment drag blocks, torques and coupling blocks (see `Balance`).

    ``basal_seg`` (M,) is each segment's basal drag before ``damping``;
    ``grad3`` the heat gradient at the segments (None: no heat torque)."""
    M = seg.M
    pos = seg.pos
    bal = Balance(M)
    I3 = np.eye(3)
    bal.D += (basal_seg * damping)[:, None, None] * I3[None, :, :]
    if grad3 is not None:
        bal.t += gain * np.cross(pos, grad3 * seg.area[:, None])
    if slab is not None and slab["seg"].size:
        np.add.at(bal.t, slab["seg"], gain * slab["tq"])
        if trench_per_len > 0.0:
            # slab resistance (bending, the interface): a drag on the normal motion of the slab's
            # own plate at its trench, as long as the pull, so a slab-attached plate saturates at
            # slab_force * gain / trench_per_len whatever its size (Forsyth & Uyeda 1975)
            ac = np.cross(pos[slab["seg"]], slab["dirn"])
            bal.add_D(slab["seg"], _outer(ac, trench_per_len * slab["g"] * slab["len"]))
    pid = seg.plate_id.astype(np.int64)
    Id_plate = np.bincount(pid, weights=basal_seg * damping, minlength=plates.P)[:plates.P]
    if cen is not None and cen["i"].size:
        if drag_per_len > 0.0:
            # boundary drag: len * (I - r r^T) on each side of every boundary contact
            for side in ("i", "j"):
                s = cen[side]
                r = pos[s]
                proj = I3[None, :, :] - r[:, :, None] * r[:, None, :]
                bal.add_D(s, (drag_per_len * cen["l" + side])[:, None, None] * proj)
        if cc_per_len > 0.0:
            cc = (cen["ki"] == CONTINENTAL) & (cen["kj"] == CONTINENTAL) & (cen["appr"] > 0.0)
            if rift_pairs:
                # a rift's own two halves are coupled by its strength below, not by collision
                rk = np.array([a * KEY + b for a, b in rift_pairs] + [b * KEY + a for a, b in rift_pairs], np.int64)
                cc &= ~np.isin(cen["pi"] * KEY + cen["pj"], rk)
            if cc.any():
                ac = np.cross(cen["rm"][cc], cen["dd"][cc])
                bal.add_coupling(cen["i"][cc], cen["j"][cc], _outer(ac, cc_per_len * cen["w"][cc]))
            bal.info["cc_pairs"] = int(cc.sum())
            bal.info["cc_len"] = float(cen["w"][cc].sum())
        if rift_pairs:
            # the rift's strength G (G0 per rift, calibrated when it was cut, or rift_strength):
            # the coupling, summed over the rift's contacts, is G times the smaller half's basal
            # drag, so it holds the opening back to ~1 / (1 + G) whatever the size of the
            # plates; it necks with opening, exp(-(delta / rift_weaken)^power)
            key = cen["pi"] * KEY + cen["pj"]
            mm_all = np.zeros(cen["i"].size, bool)
            kv = np.zeros(cen["i"].size)
            for (a, b), st in rift_pairs.items():
                if a >= plates.P or b >= plates.P or not (plates.alive[a] and plates.alive[b]):
                    continue
                mm = (key == a * KEY + b) | (key == b * KEY + a)
                if not mm.any():
                    continue
                Lc = float(cen["w"][mm].sum())
                G0 = float(st.get("G0", rift_strength)) * rift_scale
                g = G0 * math.exp(-(st["delta"] / max(rift_weaken, 1e-12)) ** float(rift_power))
                kv[mm] = g * min(Id_plate[a], Id_plate[b]) / max(Lc, 1e-12) * cen["w"][mm]
                mm_all |= mm
            if mm_all.any():
                ac = np.cross(cen["rm"][mm_all], cen["dd"][mm_all])
                bal.add_coupling(cen["i"][mm_all], cen["j"][mm_all], _outer(ac, kv[mm_all]))
            bal.info["rift_len"] = float(cen["w"][mm_all].sum())
    bal.info["Id_plate"] = Id_plate
    return bal


def solve(bal: Balance, plates, pid: np.ndarray, extra_tau: np.ndarray | None = None) -> np.ndarray:
    """omega* (P, 3) of the live plates from the per-segment balance."""
    P = plates.P
    alive = np.flatnonzero(plates.alive)
    n = alive.size
    loc = np.full(P, -1, np.int64)
    loc[alive] = np.arange(n)
    pid = pid.astype(np.int64)
    Dp = np.zeros((P, 3, 3))
    tp_ = np.zeros((P, 3))
    for x in range(3):
        tp_[:, x] = np.bincount(pid, weights=bal.t[:, x], minlength=P)[:P]
        for y in range(3):
            Dp[:, x, y] = np.bincount(pid, weights=bal.D[:, x, y], minlength=P)[:P]
    if extra_tau is not None:
        k = min(extra_tau.shape[0], P)
        tp_[:k] += extra_tau[:k]
    A = np.zeros((3 * n, 3 * n))
    for a, q in enumerate(alive):
        A[3 * a:3 * a + 3, 3 * a:3 * a + 3] = Dp[q]
    if bal.ci.size:
        pa, pb = loc[pid[bal.ci]], loc[pid[bal.cj]]
        ok = (pa >= 0) & (pb >= 0) & (pa != pb)
        key = pa[ok] * n + pb[ok]
        uk, inv = np.unique(key, return_inverse=True)
        B = np.zeros((uk.size, 3, 3))
        np.add.at(B, inv, bal.cB[ok])
        for kk, Bk in zip(uk, B):
            qa, qb = int(kk // n), int(kk % n)
            A[3 * qa:3 * qa + 3, 3 * qb:3 * qb + 3] -= Bk
            A[3 * qb:3 * qb + 3, 3 * qa:3 * qa + 3] -= Bk
    rhs = tp_[alive].reshape(-1)
    # scipy's LAPACK, not numpy's: numpy's OpenBLAS spins up its whole thread pool for a
    # 3P x 3P system and took 0.27-0.45 s a solve on a loaded machine (scipy: ~2 ms)
    import scipy.linalg as sla
    try:
        sol = sla.solve(A, rhs, check_finite=False)
    except (sla.LinAlgError, ValueError):
        sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
    wstar = np.zeros((P, 3))
    wstar[alive] = sol.reshape(n, 3)
    bal.wstar = wstar
    # what each segment's couplings hold it to: B @ omega* of the plate at the other end
    rc = np.zeros((bal.D.shape[0], 3))
    if bal.ci.size:
        wi = wstar[pid[bal.ci]]
        wj = wstar[pid[bal.cj]]
        np.add.at(rc, bal.ci, np.einsum("nab,nb->na", bal.cB, wj))
        np.add.at(rc, bal.cj, np.einsum("nab,nb->na", bal.cB, wi))
    bal.rc = rc
    return wstar


def release(bal: Balance, seg, sel: np.ndarray, side: np.ndarray, cut_a: np.ndarray, cut_b: np.ndarray,
            drag_per_len: float, couple: np.ndarray | None = None,
            extra_t: tuple | None = None, extra_D: tuple | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Terminal velocities of the two parts of one plate, released along a cut.

    ``sel`` are the plate's segments, ``side`` (bool per ``sel``) puts each in
    part B (True) or A.  Every other plate is held at its terminal velocity
    (its couplings to this plate enter as ``rc``).  ``cut_a`` / ``cut_b`` are
    global indices of the contact pairs across the cut (A side, B side): each
    becomes boundary for both parts (boundary drag), and with ``couple``
    (per cut pair, a stiffness) the parts are also coupled along the cut
    normal-to-strike, as a rift's strength couples them.  ``extra_t`` /
    ``extra_D`` are (segments, torques) / (segments, 3x3 blocks) to add (a
    seed slab's pull and resistance).  Returns (omega_A, omega_B)."""
    I3 = np.eye(3)
    pos = seg.pos
    isB = np.zeros(seg.M, bool)
    isB[sel[side]] = True
    a_idx, b_idx = sel[~side], sel[side]
    DA = bal.D[a_idx].sum(axis=0)
    DB = bal.D[b_idx].sum(axis=0)
    rA = (bal.t[a_idx] + bal.rc[a_idx]).sum(axis=0)
    rB = (bal.t[b_idx] + bal.rc[b_idx]).sum(axis=0)
    if extra_t is not None and extra_t[0].size:
        s, tq = extra_t
        b_ = isB[s]
        rA += tq[~b_].sum(axis=0)
        rB += tq[b_].sum(axis=0)
    if extra_D is not None and extra_D[0].size:
        s, blk = extra_D
        b_ = isB[s]
        DA += blk[~b_].sum(axis=0)
        DB += blk[b_].sum(axis=0)
    K = np.zeros((3, 3))
    if cut_a.size:
        ell = np.sqrt(seg.ext)
        cnt = np.bincount(np.concatenate([cut_a, cut_b]), minlength=seg.M)
        la = ell[cut_a] / np.maximum(cnt[cut_a], 1)
        lb = ell[cut_b] / np.maximum(cnt[cut_b], 1)
        if drag_per_len > 0.0:
            ra, rb = pos[cut_a], pos[cut_b]
            DA += np.einsum("n,nab->ab", drag_per_len * la, I3[None] - ra[:, :, None] * ra[:, None, :])
            DB += np.einsum("n,nab->ab", drag_per_len * lb, I3[None] - rb[:, :, None] * rb[:, None, :])
        if couple is not None:
            rm = pos[cut_a] + pos[cut_b]
            rm /= np.maximum(np.linalg.norm(rm, axis=1, keepdims=True), 1e-12)
            d = pos[cut_b] - pos[cut_a]
            d -= np.sum(d * rm, axis=1, keepdims=True) * rm
            d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
            ac = np.cross(rm, d)
            K = np.einsum("n,na,nb->ab", couple, ac, ac)
    if not K.any():
        try:
            wA = np.linalg.solve(DA, rA)
            wB = np.linalg.solve(DB, rB)
        except np.linalg.LinAlgError:
            wA = np.linalg.lstsq(DA, rA, rcond=None)[0]
            wB = np.linalg.lstsq(DB, rB, rcond=None)[0]
        return wA, wB
    A6 = np.zeros((6, 6))
    A6[:3, :3] = DA + K
    A6[3:, 3:] = DB + K
    A6[:3, 3:] = -K
    A6[3:, :3] = -K
    sol = np.linalg.solve(A6, np.concatenate([rA, rB]))
    return sol[:3], sol[3:]


def rift_opening(cen: dict, rift_pairs: dict) -> dict:
    """Mean opening rate (radians/step, > 0 opening) over each rift pair's contacts."""
    out = {}
    if not rift_pairs or cen is None or cen["i"].size == 0:
        return out
    key = cen["pi"] * KEY + cen["pj"]
    for (a, b) in rift_pairs:
        m = (key == a * KEY + b) | (key == b * KEY + a)
        out[(a, b)] = float(np.average(-cen["appr"][m], weights=cen["w"][m])) if m.any() else None
    return out


__all__ = ["census", "all_pairs", "slab_contrib", "slab_torques", "assemble", "solve", "release", "Balance",
           "deposit_blobs", "rift_opening"]
