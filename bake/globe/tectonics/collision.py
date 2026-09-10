"""Label map, gap filling, subduction, crystallisation and the grid
post-processing (cascade, Gaussian) of PLAN.md section 6.2 / 6.4.

All functions are deterministic: KD-tree queries are exact per point (so
``workers=-1`` is safe), pair lists are sorted before they are applied,
and every sequential update runs in index order inside numba.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree

from ..cubesphere import Grid, from_sphere_v
from ..field import FaceField
from .segments import CONTINENTAL, OCEANIC, Segments, _hash_insert, _voxel_coord, _voxel_resolution, greedy_accept


# --------------------------------------------------------------------------
# label map
# --------------------------------------------------------------------------
_CENTERS_CACHE: dict = {}


def interior_centers_flat(grid: Grid) -> np.ndarray:
    """Contiguous (6*N*N, 3) copy of ``grid.interior_centers`` (cached per
    grid: the reshape of the strided halo view costs ~30 ms at N = 512)."""
    key = (id(grid), grid.N, grid.H)
    c = _CENTERS_CACHE.get(key)
    if c is None:
        c = np.ascontiguousarray(grid.interior_centers.reshape(-1, 3))
        _CENTERS_CACHE[key] = c
    return c


def build_tree(seg: Segments) -> cKDTree:
    """KD-tree on the current segment positions (one per step; reuse it for
    every query of that step)."""
    return cKDTree(seg.pos)


def label_map(tree: cKDTree, grid: Grid):
    """Nearest segment of every interior cell (exact KD-tree query).
    Returns ``(idx, dist)``: ``idx`` (6, N, N) int32 segment index, ``dist``
    (6, N, N) float64 chord distance.  Every cell gets a label (there are
    no holes by construction)."""
    c = interior_centers_flat(grid)
    dist, idx = tree.query(c, k=1, workers=-1)
    N = grid.N
    return idx.reshape(6, N, N).astype(np.int32), dist.reshape(6, N, N)


@njit(cache=True)
def _build_hash(pts, G):
    head = np.full(G * G * G, -1, dtype=np.int32)
    nxt = np.empty(pts.shape[0], dtype=np.int32)
    for i in range(pts.shape[0]):
        _hash_insert(head, nxt, pts, i, G)
    return head, nxt


@njit(cache=True, parallel=True)
def _nearest_kernel(q, pts, head, nxt, G, cap2):
    """Nearest hashed point of every query (exact whenever the distance² is
    < cap2; otherwise idx = -1).  Ties resolve to the lowest index, so the
    parallel loop is deterministic."""
    n = q.shape[0]
    idx = np.full(n, -1, dtype=np.int32)
    d2o = np.full(n, cap2, dtype=np.float64)
    for a in prange(n):
        qx, qy, qz = q[a, 0], q[a, 1], q[a, 2]
        ix = _voxel_coord(qx, G)
        iy = _voxel_coord(qy, G)
        iz = _voxel_coord(qz, G)
        best = cap2
        bi = -1
        for dx in range(-1, 2):
            jx = ix + dx
            if jx < 0 or jx >= G:
                continue
            for dy in range(-1, 2):
                jy = iy + dy
                if jy < 0 or jy >= G:
                    continue
                for dz in range(-1, 2):
                    jz = iz + dz
                    if jz < 0 or jz >= G:
                        continue
                    k = head[(jx * G + jy) * G + jz]
                    while k >= 0:
                        ex = pts[k, 0] - qx
                        ey = pts[k, 1] - qy
                        ez = pts[k, 2] - qz
                        d2 = ex * ex + ey * ey + ez * ez
                        if d2 < best or (d2 == best and k < bi):
                            best = d2
                            bi = k
                        k = nxt[k]
        idx[a] = bi
        d2o[a] = best
    return idx, d2o


def label_map_fast(seg: Segments, grid: Grid, cap_radius: float, tree: cKDTree | None = None):
    """Same result as :func:`label_map` (nearest segment per interior cell,
    lowest index on exact ties) but ~3x faster: a numba voxel hash resolves
    every cell whose nearest segment is closer than ``cap_radius``; the few
    remaining cells (gaps) are resolved with the KD-tree (built here if
    ``tree`` is None).  ``cap_radius`` must be >= the gap radius so gap
    detection stays exact."""
    c = interior_centers_flat(grid)
    G = _voxel_resolution(cap_radius)
    head, nxt = _build_hash(seg.pos, G)
    idx, d2 = _nearest_kernel(c, seg.pos, head, nxt, G, float(cap_radius) ** 2)
    dist = np.sqrt(d2)
    miss = idx < 0
    if miss.any():
        tree = build_tree(seg) if tree is None else tree
        dm, im = tree.query(c[miss], k=1, workers=-1)
        idx[miss] = im
        dist[miss] = dm
    N = grid.N
    return idx.reshape(6, N, N), dist.reshape(6, N, N)


def cell_area_steradians(grid: Grid) -> np.ndarray:
    """Interior cell areas in steradians (6, N, N) float64."""
    return grid.interior_cell_area.astype(np.float64) / grid.R_planet**2


def accumulate_area(seg: Segments, idx: np.ndarray, area_sr: np.ndarray, blend: float) -> None:
    """``area = (1 - blend) * area + blend * measured`` with the measured
    steradians from the label map (PLAN 6.2.1, blend 0.01 == '0.99 rolling')."""
    measured = np.bincount(idx.ravel(), weights=area_sr.ravel(), minlength=seg.M)[: seg.M]
    if blend >= 1.0:
        seg.area = measured
    else:
        seg.area = (1.0 - blend) * seg.area + blend * measured


def splat(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Per-segment values (M,) -> (6, N, N) through the label map."""
    return np.asarray(values)[idx]


class SmoothSplat:
    """Gaussian-weighted blend of the ``knn`` nearest segments for every
    interior cell: ``v(cell) = Σ w_k v_k / Σ w_k``, ``w_k = exp(-d_k² /
    2σ²)``.  Replaces the step function of the nearest-segment splat by a
    surface that is smooth at the segment-spacing scale while keeping
    features one spacing wide (belts).  Query once, splat many fields."""

    def __init__(self, tree: cKDTree, grid: Grid, sigma: float, knn: int = 12):
        c = interior_centers_flat(grid)
        kk = min(int(knn), tree.n)
        d, nb = tree.query(c, k=kk, workers=-1)
        d = np.atleast_2d(d).reshape(c.shape[0], kk)
        self.nb = np.atleast_2d(nb).reshape(c.shape[0], kk)
        w = np.exp(-(d * d) / (2.0 * float(sigma) ** 2))
        w[:, 0] = np.maximum(w[:, 0], 1e-300)  # the nearest always counts
        self.w = w / w.sum(axis=1, keepdims=True)
        self.shape = (6, grid.N, grid.N)

    def __call__(self, values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=np.float64)[self.nb]
        return (v * self.w).sum(axis=1).reshape(self.shape)


# --------------------------------------------------------------------------
# crystallisation (PLAN 6.2.6)
# --------------------------------------------------------------------------
def deposit_density(T: np.ndarray, k_D: float) -> np.ndarray:
    """``D = k_D (1 - T) / (1 - k_D (1 - T))`` in [0, 1] for T in [0, 1]."""
    x = k_D * (1.0 - T)
    return x / np.maximum(1.0 - x, 1e-6)


def crystallise(seg: Segments, T: np.ndarray, growth: float, density_base: float, k_D: float, dissolution_factor: float, max_thickness: float = 0.0, min_thickness: float = 1e-3) -> None:
    """PLAN 6.2.6, one step.  Growth ``G = k_G (1 - T)(1 - T - d_b)``:
    positive -> deposit thickness ``G`` at density ``D(T)`` (faded by
    ``exp(-thickness / max_thickness)`` if > 0, so cratons thicken
    logarithmically with age instead of linearly forever); negative -> dissolve
    ``dissolution_factor * |G|`` at the segment's own density (never more
    than half its thickness).  Mass changes only here (and in
    :func:`spawn_segments`); ``age += 1`` (steps).

    **Continental crust only.**  Oceanic crust ages but does not grow: real
    ocean floor is the same ~7 km thick when it subducts as when it was
    erupted, because it is a chilled melt layer rather than an accreting
    pile.  Letting it crystallise was what made the height distribution one
    continuum -- every segment integrating the heat it drifted through, with
    nothing to hold the seafloor down at its birth thickness."""
    T = np.clip(np.asarray(T, dtype=np.float64), 0.0, 1.0)
    G = growth * (1.0 - T) * (1.0 - T - density_base)
    D = deposit_density(T, k_D)
    grow = G > 0
    if max_thickness > 0:
        fade = np.exp(-seg.thickness / max_thickness)  # asymptotic: old crust keeps (slowly) getting thicker
        G = np.where(grow, G * fade, G)
    dth = np.where(grow, G, dissolution_factor * G)
    dth = np.maximum(dth, -0.5 * seg.thickness)
    dth = np.maximum(dth, min_thickness - seg.thickness)
    dth = np.where(seg.kind == CONTINENTAL, dth, 0.0)
    dm = np.where(grow, dth * D, dth * seg.density)
    seg.mass += dm
    seg.thickness += dth
    seg.density = np.clip(seg.mass / np.maximum(seg.thickness, 1e-12), 0.0, 1.0)
    seg.mass = seg.thickness * seg.density
    seg.age += 1.0


# --------------------------------------------------------------------------
# gaps -> new crust (PLAN 6.2.4)
# --------------------------------------------------------------------------
def spawn_segments(seg: Segments, idx: np.ndarray, dist: np.ndarray, grid: Grid, gap_radius: float, r_min: float, rng: np.random.Generator, heat: FaceField, new_thickness: float, oceanic_density: float, jitter_cells: float = 0.5, omega: np.ndarray | None = None) -> tuple[Segments, np.ndarray]:
    """Cells farther than ``gap_radius`` from every segment are divergent
    boundaries — provided the nearest segment is moving *away* from the
    cell (``omega`` (P, 3) rad/step given; holes left by subduction at a
    convergent boundary are closed by the incoming plate instead of being
    filled with new crust).  Their (jittered) centres are candidate
    positions, walked in a random order and accepted greedily with minimum
    spacing ``r_min`` (also against the existing segments).  New segments
    are thin (``new_thickness``), have age 0, the plate of the nearest
    existing segment, and are always **oceanic** at ``oceanic_density``.

    New crust at a divergent boundary is mid-ocean-ridge basalt: dense,
    compositionally uniform, and the same everywhere.  It used to be given
    ``deposit_density(T)``, which reads the local heat -- so crust born at a
    ridge, where the heat is by construction highest, came out *light* and
    therefore buoyant.  That is backwards, and it meant the model had no way
    to make ocean floor at all.

    Returns ``(new_segments, gap_mask)``; the caller appends the segments
    and cools the heat field under ``gap_mask``."""
    gap = dist > gap_radius
    if omega is not None and gap.any():
        cells = np.nonzero(gap.ravel())[0]
        near = idx.ravel()[cells]
        ps = seg.pos[near]
        v = np.cross(omega[seg.plate_id[near]], ps)
        away = ps - interior_centers_flat(grid)[cells]
        div = np.sum(v * away, axis=1) > 0.0
        gap.ravel()[cells[~div]] = False
    n = int(gap.sum())
    if n == 0:
        return Segments(np.zeros((0, 3)), 0.0, 0.0, 0.0, 0, 0.0), gap
    cells = np.nonzero(gap.ravel())[0]
    cands = interior_centers_flat(grid)[cells]
    order = rng.permutation(n)
    cells = cells[order]
    cands = cands[order]
    if jitter_cells > 0:
        cell_rad = math.pi / (2.0 * grid.N)
        cands = cands + rng.normal(size=cands.shape) * (jitter_cells * cell_rad)
        cands /= np.linalg.norm(cands, axis=1, keepdims=True)
    acc = greedy_accept(cands, seg.pos, r_min)
    pos = cands[acc]
    plate = seg.plate_id[idx.ravel()[cells[acc]]]
    mean_area = float(seg.area.mean()) if seg.M else 0.0
    new = Segments(pos, new_thickness, oceanic_density, 0.0, plate, mean_area, kind=OCEANIC)
    return new, gap


# --------------------------------------------------------------------------
# collisions (PLAN 6.2.5)
# --------------------------------------------------------------------------
def differentiate(seg: Segments, survivors: np.ndarray, rate: float, floor: float) -> float:
    """Make collided crust lighter -- the process that builds continents.

    Earth's surface is famously bimodal: a peak at the continental shelf and
    another on the abyssal plain, with little between, because it carries two
    kinds of crust. Oceanic crust is basaltic, dense and thin; continental
    crust is granitic, light and thick, and floats about 4 km higher.

    Collision alone cannot produce that split. Merging two segments averages
    their mass and thickness, so density only ever moves toward the mean and
    the elevation histogram stays a single narrow spike (measured: land mean
    324 m and ocean mean -189 m, against Earth's 840 m and -3700 m).

    What separates the two populations on Earth is *differentiation*. Crust
    thickened at an arc partially melts; the light granitic fraction rises
    and stays, while the dense residue delaminates and is lost to the mantle.
    Crust that has been through a collision therefore comes out lighter than
    it went in, and repeated orogeny ratchets it toward continental.

    So this pulls each survivor's density a fraction `rate` toward `floor`,
    keeping thickness and dropping mass. The lost mass is returned so the
    caller can book it: it has left the crust for the mantle, which is
    physical rather than a leak.

    **Measured: this alone does not produce the bimodality.** Sweeping
    `differentiation` 0 -> 0.04 -> 0.10 moved the land/ocean mean gap
    1723 -> 1171 -> 1472 m, against Earth's 4540 m, with ocean mean depth
    stuck near -300 m rather than -3700 m. Making continents lighter also
    lifts the sea-level quantile they are measured against, and
    `relief_m` then rescales the whole field, so the ratio barely moves.
    Raising `deposit_density` (denser new oceanic crust) is likewise only
    marginal: 0.5 -> 0.9 took ocean-depth-over-land-relief from 0.03 to
    0.06, where Earth is 0.42.

    The deeper obstacle is that our ocean floor is not a basin. It spans
    about 400 m (-600 to -190) hugging sea level, where Earth's spans 3000
    (-5500 to -2500) and is *separated* from the shelf by a steep, narrow
    slope. Earth is bimodal because it carries two discrete crust
    populations with a sharp margin between them; a smooth splat of a
    continuously-varying thickness gives one continuum, and no amount of
    density tuning turns a continuum into two peaks. Kept because the
    mechanism is real and may matter alongside a genuine crust-type split,
    but it is not the lever on its own.
    """
    if rate <= 0.0 or survivors.size == 0:
        return 0.0
    su = np.unique(survivors)
    su = su[(su >= 0) & (su < seg.M)]
    if su.size == 0:
        return 0.0
    d0 = seg.density[su]
    d1 = np.maximum(d0 - rate * (d0 - float(floor)), float(floor))
    lost = float(((d0 - d1) * seg.thickness[su]).sum())
    seg.density[su] = d1
    seg.mass[su] = seg.thickness[su] * d1
    return lost


# numba closes over module globals as compile-time constants; the int8 typing
# has to match ``Segments.kind`` for the comparisons inside the kernel
OCEANIC_K = np.int8(OCEANIC)
CONTINENTAL_K = np.int8(CONTINENTAL)


@njit(cache=True)
def _apply_collisions(pairs, plate_id, omega_dt, pos, mass, thickness, density, age, kind, craton, alive, overlap2, accretion, arc_birth, birth_draw):
    n = pairs.shape[0]
    losers = np.empty(n, dtype=np.int64)
    survivors = np.empty(n, dtype=np.int64)
    k = 0
    for e in range(n):
        i = pairs[e, 0]
        j = pairs[e, 1]
        if not alive[i] or not alive[j]:
            continue
        pi = plate_id[i]
        pj = plate_id[j]
        if pi == pj:
            continue
        # approaching?  (v_i - v_j) . (p_j - p_i) > 0 with v = (omega dt) x p
        vix = omega_dt[pi, 1] * pos[i, 2] - omega_dt[pi, 2] * pos[i, 1]
        viy = omega_dt[pi, 2] * pos[i, 0] - omega_dt[pi, 0] * pos[i, 2]
        viz = omega_dt[pi, 0] * pos[i, 1] - omega_dt[pi, 1] * pos[i, 0]
        vjx = omega_dt[pj, 1] * pos[j, 2] - omega_dt[pj, 2] * pos[j, 1]
        vjy = omega_dt[pj, 2] * pos[j, 0] - omega_dt[pj, 0] * pos[j, 2]
        vjz = omega_dt[pj, 0] * pos[j, 1] - omega_dt[pj, 1] * pos[j, 0]
        dx = pos[j, 0] - pos[i, 0]
        dy = pos[j, 1] - pos[i, 1]
        dz = pos[j, 2] - pos[i, 2]
        if (vix - vjx) * dx + (viy - vjy) * dy + (viz - vjz) * dz <= 0.0:
            # receding / sliding past: only collide once they overlap deeply
            if dx * dx + dy * dy + dz * dz > overlap2:
                continue
        # Who goes down.  Crust type first: a continent cannot be subducted
        # under ocean floor at any density, which is the irreversibility that
        # separates the two populations.  Within a type, the denser (older,
        # colder) slab sinks, as before.
        if kind[i] != kind[j]:
            if kind[i] == OCEANIC_K:
                lo, su = i, j
            else:
                lo, su = j, i
        elif kind[i] == CONTINENTAL_K and craton[i] != craton[j]:
            # A craton against a mobile belt: the belt deforms. Cratons are
            # cold, thick and depleted, so they behave as rigid indenters --
            # which is why the same Archean nuclei have survived every cycle
            # while the crust welded between them has been reworked
            # repeatedly. Without this a craton is just another continental
            # segment and gets consumed at the same rate as its surroundings.
            if craton[i] == 0:
                lo, su = i, j
            else:
                lo, su = j, i
        elif density[i] > density[j]:
            lo, su = i, j
        elif density[j] > density[i]:
            lo, su = j, i
        else:
            lo, su = (j, i) if j > i else (i, j)

        # How much of the slab stays at the surface.  Ocean floor going down a
        # trench mostly leaves the system: its sediment and a melt fraction are
        # welded onto the overriding plate as an arc, the rest returns to the
        # mantle.  Handing over 100 %, as this did, made the crust a monotone
        # accumulator -- 1500 steps of it drove thickness to 8.3 against an
        # initial 0.4, and every collision zone toward the same saturated
        # height.  Continent-on-continent keeps everything: nothing subducts,
        # the crust doubles, and that is what a Tibet is.
        f = 1.0 if kind[lo] == CONTINENTAL_K else accretion
        mass[su] += f * mass[lo]
        thickness[su] += f * thickness[lo]
        density[su] = mass[su] / thickness[su]
        # the survivor's age is the older of the two only when it keeps the
        # whole slab; an arc is new crust welded to old, not old crust
        if f >= 1.0 and age[lo] > age[su]:
            age[su] = age[lo]
        # Island arcs: repeated ocean-on-ocean subduction is how continental
        # crust is *born* (the Japans, the Aleutians, the Andean margin before
        # it was a margin).  Without a birth channel the continental area can
        # only shrink from whatever the initial condition seeded, and the
        # supercontinent cycle runs down.
        if kind[su] == OCEANIC_K and kind[lo] == OCEANIC_K and birth_draw[e] < arc_birth:
            kind[su] = CONTINENTAL_K            # island arc -> new continental crust, not craton
        alive[lo] = False
        losers[k] = lo
        survivors[k] = su
        k += 1
    return losers[:k], survivors[:k]


def collide(seg: Segments, tree: cKDTree, radius: float, omega_dt: np.ndarray, alive: np.ndarray, overlap_fraction: float = 0.5, accretion: float = 1.0, arc_birth: float = 0.0, rng: np.random.Generator | None = None):
    """Subduction: for every pair of segments of different plates within
    chord ``radius`` (KD-tree pair query, applied in sorted order) that are
    *approaching* — or closer than ``overlap_fraction * radius`` whatever
    their relative motion, so plates sliding past each other cannot
    interleave — one subducts: the oceanic member of a mixed pair whatever
    its density, else the denser one.  A fraction ``accretion`` of its mass
    and thickness goes to the survivor and the rest is lost to the mantle
    (all of it, and the older age, when the loser is continental — that
    crust cannot sink, so a continent-continent collision doubles the crust
    instead of destroying half of it).  With probability ``arc_birth`` an
    ocean-on-ocean survivor becomes continental.  The loser is flagged dead
    in ``alive`` (in place).  The loser's own arrays keep the
    transferred amounts until ``seg.compress(alive)``; the total mass of
    live segments is conserved.  Returns ``(losers, survivors)`` index
    arrays (into the current arrays; a survivor may appear several times)."""
    pairs = tree.query_pairs(radius, output_type="ndarray")
    if pairs.shape[0] == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    pairs = np.sort(pairs, axis=1)
    pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    draw = rng.random(pairs.shape[0]) if (rng is not None and arc_birth > 0.0) else np.zeros(pairs.shape[0])
    return _apply_collisions(np.ascontiguousarray(pairs), seg.plate_id, np.ascontiguousarray(omega_dt), seg.pos, seg.mass, seg.thickness, seg.density, seg.age, seg.kind, seg.craton, alive, (float(overlap_fraction) * float(radius)) ** 2, float(accretion), float(arc_birth), np.ascontiguousarray(draw))


@njit(cache=True)
def _spread_kernel(losers, survivors, nbrs, pos, plate_id, kind, mass, thickness, density, alive, inv2s2, accretion):
    K = nbrs.shape[1]
    w = np.empty(K, dtype=np.float64)
    for e in range(losers.shape[0]):
        su = survivors[e]
        lo = losers[e]
        if not alive[su]:
            continue  # the survivor was subducted later in this step; its mass already moved on
        # only what the survivor actually received: an oceanic slab hands over
        # `accretion` of itself and the rest goes to the mantle, so spreading
        # the whole slab would create mass that was never accreted
        f = 1.0 if kind[lo] == CONTINENTAL_K else accretion
        m = f * mass[lo]
        th = f * thickness[lo]
        tot = 0.0
        for q in range(K):
            n = nbrs[e, q]
            w[q] = 0.0
            if n < 0 or not alive[n] or plate_id[n] != plate_id[su] or kind[n] != kind[su]:
                continue
            dx = pos[n, 0] - pos[su, 0]
            dy = pos[n, 1] - pos[su, 1]
            dz = pos[n, 2] - pos[su, 2]
            w[q] = np.exp(-(dx * dx + dy * dy + dz * dz) * inv2s2)
            tot += w[q]
        if tot <= 0.0:
            continue
        # the survivor already holds (m, th); hand the neighbours their share
        for q in range(K):
            if w[q] <= 0.0:
                continue
            n = nbrs[e, q]
            if n == su:
                continue
            f = w[q] / tot
            mass[su] -= f * m
            thickness[su] -= f * th
            mass[n] += f * m
            thickness[n] += f * th
            density[n] = mass[n] / thickness[n]
        density[su] = mass[su] / thickness[su]


def spread_collisions(seg: Segments, tree: cKDTree, losers: np.ndarray, survivors: np.ndarray, alive: np.ndarray, sigma: float, knn: int = 12, accretion: float = 1.0) -> None:
    """Belt formation: the mass and thickness a survivor just received from
    a subducted segment are shared, with Gaussian weights ``exp(-d²/2σ²)``,
    among the survivor and its ``knn`` nearest *live, same-plate, same-kind* segments
    (the survivor itself is among them with weight 1), so repeated
    collisions along a boundary build a belt ~2σ wide instead of isolated
    peaks.  Mass conserving.  Must run before ``seg.compress`` (the dead
    losers' arrays still hold the transferred amounts)."""
    if losers.size == 0:
        return
    kk = min(int(knn), tree.n)
    _, nb = tree.query(seg.pos[survivors], k=kk, workers=-1)
    nb = np.atleast_2d(nb).reshape(survivors.size, kk).astype(np.int64)
    _spread_kernel(np.ascontiguousarray(losers), np.ascontiguousarray(survivors), np.ascontiguousarray(nb), seg.pos, seg.plate_id, seg.kind, seg.mass, seg.thickness, seg.density, alive, 1.0 / (2.0 * float(sigma) ** 2), float(accretion))


@njit(cache=True)
def _segment_cascade(order, nbrs, kind, thickness, mass, density, alive, rate, thr):
    K = nbrs.shape[1]
    for e in range(order.shape[0]):
        s = order[e]
        if not alive[s]:
            continue
        for q in range(K):
            n = nbrs[e, q]
            if n == s or n < 0 or not alive[n] or kind[n] != kind[s]:
                continue
            ds = density[s]
            hs = thickness[s] * (1.0 - ds)
            hn = thickness[n] * (1.0 - density[n])
            delta = hs - hn
            if delta <= thr or ds >= 0.999:
                continue
            dh = rate * (delta - thr) * 0.5 / K
            dth = dh / (1.0 - ds)
            if dth > 0.5 * thickness[s]:
                dth = 0.5 * thickness[s]  # oceanic crust is thin; unclamped this went negative
            thickness[s] -= dth
            mass[s] -= dth * ds
            thickness[n] += dth
            mass[n] += dth * ds
            density[s] = mass[s] / thickness[s]
            density[n] = mass[n] / thickness[n]


def segment_cascade(seg: Segments, tree: cKDTree, survivors: np.ndarray, alive: np.ndarray, rate: float, threshold: float, knn: int = 8) -> None:
    """PLAN 6.2.5 / 6.4 on segments: for each survivor (in order) move
    height ``rate * (Δh - threshold) / 2`` (split over its ``knn`` nearest
    live neighbours) to lower neighbours as thickness at the survivor's
    density — mass conserving."""
    if survivors.size == 0:
        return
    order = np.unique(survivors)
    _, nb = tree.query(seg.pos[order], k=min(knn + 1, tree.n), workers=-1)
    nb = np.atleast_2d(nb).astype(np.int64)
    _segment_cascade(order, np.ascontiguousarray(nb), seg.kind, seg.thickness, seg.mass, seg.density, alive, float(rate), float(threshold))


@njit(cache=True)
def _relax_kernel(nbrs, pos, kind, thickness, mass, density, rate, thr_per_rad):
    M, K = nbrs.shape
    for s in range(M):
        for q in range(K):
            n = nbrs[s, q]
            if n == s or n < 0 or kind[n] != kind[s]:
                continue
            ds = density[s]
            if ds >= 0.999:
                continue
            hs = thickness[s] * (1.0 - ds)
            hn = thickness[n] * (1.0 - density[n])
            dx = pos[n, 0] - pos[s, 0]
            dy = pos[n, 1] - pos[s, 1]
            dz = pos[n, 2] - pos[s, 2]
            thr = thr_per_rad * np.sqrt(dx * dx + dy * dy + dz * dz)
            delta = hs - hn
            if delta <= thr:
                continue
            dh = rate * (delta - thr) * 0.5 / K
            dth = dh / (1.0 - ds)
            if dth > 0.5 * thickness[s]:
                dth = 0.5 * thickness[s]
            thickness[s] -= dth
            mass[s] -= dth * ds
            thickness[n] += dth
            mass[n] += dth * ds
            density[s] = mass[s] / thickness[s]
            density[n] = mass[n] / thickness[n]


def relax_segments(seg: Segments, tree: cKDTree, rate: float, threshold_per_spacing: float, spacing: float, knn: int = 8) -> None:
    """PLAN 6.4 cascade applied to the segment cloud every step: for every
    segment (index order) and each of its ``knn`` nearest *same-kind*
    neighbours whose bedrock height is lower by more than
    ``threshold_per_spacing`` × their
    distance (in spacings), move ``rate * (Δh - thr) / 2 / knn`` of height
    downhill as thickness at the giver's density (mass conserving).  Turns
    stacked collision peaks into belts with foothills; the threshold is
    the maximum stable slope in bedrock units per spacing.

    Restricted to same-kind pairs because the continent-ocean contact is a
    ~4 km step in bedrock height, far above any plausible threshold, so an
    unrestricted cascade drains every coastal continental segment into the
    seafloor beside it -- exactly the leak that closes the gap the crust
    types exist to open."""
    if seg.M < 2:
        return
    kk = min(int(knn) + 1, tree.n)
    _, nb = tree.query(seg.pos, k=kk, workers=-1)
    nb = np.atleast_2d(nb).reshape(seg.M, kk).astype(np.int64)
    _relax_kernel(np.ascontiguousarray(nb), seg.pos, seg.kind, seg.thickness, seg.mass, seg.density, float(rate), float(threshold_per_spacing) / float(spacing))


def delaminate(seg: Segments, limit: float, rate: float) -> float:
    """Shed the root of over-thickened crust; returns the mass lost.

    Continent-on-continent collision stacks crust with nothing to stop it:
    every orogeny doubles the survivor, so measured over 1500 steps a
    handful of segments reached thickness 8.3 against a continental median
    of 2.2.  That tail is not harmless -- the vertical scale in
    :func:`~globe.tectonics.run.finalise` is pinned to a high percentile of
    land, so a few runaway spikes squash the whole map beneath them (the
    land/ocean gap came out at 400 m, worse than before crust types
    existed).

    Earth does not do this: crustal thickness saturates near 70 km, about
    twice normal, even under Tibet after 50 My of the largest collision
    going.  The reason is that a thick root is pushed into the eclogite
    field, becomes denser than the mantle it sits in, and founders.  So
    thickness above ``limit`` decays toward it at ``rate`` per step, and the
    mass goes to the mantle rather than to a neighbour -- delamination is a
    loss, not a transfer.
    """
    if limit <= 0.0 or rate <= 0.0:
        return 0.0
    over = seg.thickness > limit
    if not over.any():
        return 0.0
    excess = (seg.thickness[over] - limit) * float(rate)
    lost = float((excess * seg.density[over]).sum())
    seg.thickness[over] -= excess
    seg.mass[over] = seg.thickness[over] * seg.density[over]
    return lost


# --------------------------------------------------------------------------
# heat sources
# --------------------------------------------------------------------------
class CellTree:
    """Static KD-tree of the interior cell centres of a grid, for
    seamless 'all cells within r of a point' queries (heat blobs, collision
    zones)."""

    def __init__(self, grid: Grid):
        self.grid = grid
        self.centers = np.ascontiguousarray(grid.interior_centers.reshape(-1, 3))
        self.tree = cKDTree(self.centers)

    def add_blobs(self, flat_interior: np.ndarray, points: np.ndarray, peak: float, radius: float) -> None:
        """``flat_interior += peak * exp(-d² / (2 (radius/2)²))`` for every
        cell within ``radius`` of every point (accumulating, order-free)."""
        if points.shape[0] == 0:
            return
        sigma2 = (0.5 * radius) ** 2
        lists = self.tree.query_ball_point(points, radius, workers=-1)
        cells = np.concatenate([np.asarray(l, dtype=np.int64) for l in lists]) if len(lists) else np.zeros(0, np.int64)
        if cells.size == 0:
            return
        src = np.repeat(np.arange(points.shape[0]), [len(l) for l in lists])
        d2 = np.sum((self.centers[cells] - points[src]) ** 2, axis=1)
        np.add.at(flat_interior, cells, peak * np.exp(-d2 / (2.0 * sigma2)))

    def within(self, points: np.ndarray, radius: float) -> np.ndarray:
        """Bool mask (6, N, N) of cells within ``radius`` of any point."""
        N = self.grid.N
        mask = np.zeros(6 * N * N, dtype=bool)
        if points.shape[0]:
            lists = self.tree.query_ball_point(points, radius, workers=-1)
            for l in lists:
                mask[np.asarray(l, dtype=np.int64)] = True
        return mask.reshape(6, N, N)


# --------------------------------------------------------------------------
# grid post-processing (PLAN 6.3 / 6.4)
# --------------------------------------------------------------------------
@njit(cache=True, parallel=True)
def _cascade_pass(d, out, rate, thr):
    F, NE, _ = d.shape
    for f in prange(F):
        for i in range(1, NE - 1):
            for j in range(1, NE - 1):
                h = d[f, i, j]
                acc = 0.0
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        if di == 0 and dj == 0:
                            continue
                        t = thr * math.sqrt(float(di * di + dj * dj))
                        delta = h - d[f, i + di, j + dj]
                        if delta > t:
                            acc -= rate * (delta - t) * 0.5 / 8.0
                        elif -delta > t:
                            acc += rate * (-delta - t) * 0.5 / 8.0
                out[f, i, j] = h + acc


def grid_cascade(field: FaceField, passes: int, rate: float, threshold: float) -> FaceField:
    """PLAN 6.4 height cascade on a grid: between 8-neighbours whose
    heights differ by more than ``threshold`` (× neighbour distance) move
    ``rate * (Δh - threshold) / 2`` (split over the 8 neighbours) from high
    to low; symmetric, hence conservative.  Halos exchanged between passes."""
    d = field.data.astype(np.float64)
    out = d.copy()
    f = FaceField(field.grid, d, name=field.name)
    for _ in range(int(passes)):
        f.exchange_halos()
        _cascade_pass(f.data, out, float(rate), float(threshold))
        f.data, out = out, f.data
    f.exchange_halos()
    return f


@njit(cache=True, parallel=True)
def _blur_i(d, out, w, r):
    F, NE, _ = d.shape
    for f in prange(F):
        for i in range(r, NE - r):
            for j in range(NE):
                acc = 0.0
                for k in range(-r, r + 1):
                    acc += w[k + r] * d[f, i + k, j]
                out[f, i, j] = acc


@njit(cache=True, parallel=True)
def _blur_j(d, out, w, r):
    F, NE, _ = d.shape
    for f in prange(F):
        for i in range(NE):
            for j in range(r, NE - r):
                acc = 0.0
                for k in range(-r, r + 1):
                    acc += w[k + r] * d[f, i, j + k]
                out[f, i, j] = acc


def gaussian_smooth(field: FaceField, sigma_cells: float) -> FaceField:
    """Separable Gaussian blur with halo exchange (seamless).  The kernel
    radius is limited to ``H - 1`` cells per pass, so larger sigmas are
    reached with several passes (``n = ceil(sigma²)``, each ``sigma/√n``)."""
    d = field.data.astype(np.float64)
    f = FaceField(field.grid, d, name=field.name)
    if sigma_cells <= 1e-3:
        f.exchange_halos()
        return f
    H = field.grid.H
    n = max(1, int(math.ceil(sigma_cells**2)))
    s = sigma_cells / math.sqrt(n)
    r = int(max(1, min(H - 1, math.ceil(3.0 * s))))
    k = np.arange(-r, r + 1, dtype=np.float64)
    w = np.exp(-0.5 * (k / s) ** 2)
    w /= w.sum()
    out = np.empty_like(d)
    for _ in range(n):
        f.exchange_halos()
        _blur_i(f.data, out, w, r)
        f.data, out = out, f.data
        f.exchange_halos()
        _blur_j(f.data, out, w, r)
        f.data, out = out, f.data
    f.exchange_halos()
    return f


def boundary_distance(tree: cKDTree, seg: Segments, grid: Grid, idx: np.ndarray, k: int = 12) -> np.ndarray:
    """Chord distance from every interior cell to the nearest segment of a
    plate *different* from the cell's own (6, N, N); cells with no foreign
    segment among the ``k`` nearest get the distance of the k-th."""
    c = interior_centers_flat(grid)
    kk = min(k, tree.n)
    dist, nb = tree.query(c, k=kk, workers=-1)
    dist = np.atleast_2d(dist).reshape(c.shape[0], kk)
    nb = np.atleast_2d(nb).reshape(c.shape[0], kk)
    own = seg.plate_id[idx.ravel()]
    foreign = seg.plate_id[nb] != own[:, None]
    first = np.where(foreign.any(axis=1), np.argmax(foreign, axis=1), kk - 1)
    return dist[np.arange(c.shape[0]), first].reshape(6, grid.N, grid.N)


def resample_to(field: FaceField, grid: Grid, nearest: bool = False) -> np.ndarray:
    """Sample a (halo-exchanged) field of one resolution on the interior
    cell centres of another grid -> (6, N, N[, C]) array."""
    p = grid.interior_centers
    if nearest:
        face, u, v = from_sphere_v(p)
        return field.sample_nearest(face, u, v)
    return field.sample_sphere(p)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Area-weighted quantile (``q`` in [0, 1]) of ``values``."""
    v = np.asarray(values, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    order = np.argsort(v, kind="stable")
    cw = np.cumsum(w[order])
    return float(v[order][min(int(np.searchsorted(cw, q * cw[-1])), v.size - 1)])


__all__ = [
    "build_tree", "label_map", "label_map_fast", "cell_area_steradians", "accumulate_area", "splat",
    "SmoothSplat", "deposit_density", "crystallise", "spawn_segments", "collide", "spread_collisions", "segment_cascade", "relax_segments",
    "CellTree", "grid_cascade", "gaussian_smooth", "boundary_distance", "resample_to", "weighted_quantile",
]
