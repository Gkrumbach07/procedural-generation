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
from .segments import Segments, greedy_accept


# --------------------------------------------------------------------------
# label map
# --------------------------------------------------------------------------
def build_tree(seg: Segments) -> cKDTree:
    """KD-tree on the current segment positions (one per step; reuse it for
    every query of that step)."""
    return cKDTree(seg.pos)


def label_map(tree: cKDTree, grid: Grid):
    """Nearest segment of every interior cell.  Returns ``(idx, dist)``:
    ``idx`` (6, N, N) int32 segment index, ``dist`` (6, N, N) float64 chord
    distance.  Every cell gets a label (there are no holes by construction)."""
    c = grid.interior_centers.reshape(-1, 3)
    dist, idx = tree.query(c, k=1, workers=-1)
    N = grid.N
    return idx.reshape(6, N, N).astype(np.int32), dist.reshape(6, N, N)


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


# --------------------------------------------------------------------------
# crystallisation (PLAN 6.2.6)
# --------------------------------------------------------------------------
def deposit_density(T: np.ndarray, k_D: float) -> np.ndarray:
    """``D = k_D (1 - T) / (1 - k_D (1 - T))`` in [0, 1] for T in [0, 1]."""
    x = k_D * (1.0 - T)
    return x / np.maximum(1.0 - x, 1e-6)


def crystallise(seg: Segments, T: np.ndarray, growth: float, density_base: float, k_D: float, dissolution_factor: float, dt: float, min_thickness: float = 1e-3) -> None:
    """Growth ``G = k_G (1 - T)(1 - T - d_b)``: positive -> deposit thickness
    ``G`` at density ``D(T)``; negative -> dissolve ``dissolution_factor *
    |G|`` at the segment's own density (never more than half its thickness).
    Mass changes only here (and in :func:`spawn_segments`); ``age += dt``."""
    T = np.clip(np.asarray(T, dtype=np.float64), 0.0, 1.0)
    G = growth * (1.0 - T) * (1.0 - T - density_base)
    D = deposit_density(T, k_D)
    grow = G > 0
    dth = np.where(grow, G, dissolution_factor * G)
    dth = np.maximum(dth, -0.5 * seg.thickness)
    dth = np.maximum(dth, min_thickness - seg.thickness)
    dm = np.where(grow, dth * D, dth * seg.density)
    seg.mass += dm
    seg.thickness += dth
    seg.density = np.clip(seg.mass / np.maximum(seg.thickness, 1e-12), 0.0, 1.0)
    seg.mass = seg.thickness * seg.density
    seg.age += dt


# --------------------------------------------------------------------------
# gaps -> new crust (PLAN 6.2.4)
# --------------------------------------------------------------------------
def spawn_segments(seg: Segments, idx: np.ndarray, dist: np.ndarray, grid: Grid, gap_radius: float, r_min: float, rng: np.random.Generator, heat: FaceField, new_thickness: float, k_D: float, jitter_cells: float = 0.5) -> tuple[Segments, np.ndarray]:
    """Cells farther than ``gap_radius`` from every segment are divergent
    boundaries.  Their (jittered) centres are candidate positions, walked
    in a random order and accepted greedily with minimum spacing ``r_min``
    (also against the existing segments).  New segments are thin
    (``new_thickness``), have age 0, the deposit density of the local heat
    and the plate of the nearest existing segment.

    Returns ``(new_segments, gap_mask)``; the caller appends the segments
    and cools the heat field under ``gap_mask``."""
    gap = dist > gap_radius
    n = int(gap.sum())
    if n == 0:
        return Segments(np.zeros((0, 3)), 0.0, 0.0, 0.0, 0, 0.0), gap
    cells = np.nonzero(gap.ravel())[0]
    cands = grid.interior_centers.reshape(-1, 3)[cells]
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
    T = np.clip(heat.sample_sphere(pos).astype(np.float64), 0.0, 1.0)
    dens = np.clip(deposit_density(T, k_D), 0.05, 1.0)
    mean_area = float(seg.area.mean()) if seg.M else 0.0
    new = Segments(pos, new_thickness, dens, 0.0, plate, mean_area)
    return new, gap


# --------------------------------------------------------------------------
# collisions (PLAN 6.2.5)
# --------------------------------------------------------------------------
@njit(cache=True)
def _apply_collisions(pairs, plate_id, omega_dt, pos, mass, thickness, density, age, alive):
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
            continue
        if density[i] > density[j]:
            lo, su = i, j
        elif density[j] > density[i]:
            lo, su = j, i
        else:
            lo, su = (j, i) if j > i else (i, j)
        mass[su] += mass[lo]
        thickness[su] += thickness[lo]
        density[su] = mass[su] / thickness[su]
        if age[lo] > age[su]:
            age[su] = age[lo]
        alive[lo] = False
        losers[k] = lo
        survivors[k] = su
        k += 1
    return losers[:k], survivors[:k]


def collide(seg: Segments, tree: cKDTree, radius: float, omega_dt: np.ndarray, alive: np.ndarray):
    """Subduction: for every pair of *approaching* segments of different
    plates within chord ``radius`` (KD-tree pair query, applied in sorted
    order) the denser one subducts: its mass and thickness go to the
    survivor (density = combined mass / thickness, age = max), it is
    flagged dead in ``alive`` (in place).  Total mass is conserved.
    Returns ``(losers, survivors)`` index arrays (into the current arrays;
    a survivor may appear several times)."""
    pairs = tree.query_pairs(radius, output_type="ndarray")
    if pairs.shape[0] == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    pairs = np.sort(pairs, axis=1)
    pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    return _apply_collisions(np.ascontiguousarray(pairs), seg.plate_id, np.ascontiguousarray(omega_dt), seg.pos, seg.mass, seg.thickness, seg.density, seg.age, alive)


@njit(cache=True)
def _segment_cascade(order, nbrs, thickness, mass, density, alive, rate, thr):
    K = nbrs.shape[1]
    for e in range(order.shape[0]):
        s = order[e]
        if not alive[s]:
            continue
        for q in range(K):
            n = nbrs[e, q]
            if n == s or n < 0 or not alive[n]:
                continue
            ds = density[s]
            hs = thickness[s] * (1.0 - ds)
            hn = thickness[n] * (1.0 - density[n])
            delta = hs - hn
            if delta <= thr or ds >= 0.999:
                continue
            dh = rate * (delta - thr) * 0.5 / K
            dth = dh / (1.0 - ds)
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
    _segment_cascade(order, np.ascontiguousarray(nb), seg.thickness, seg.mass, seg.density, alive, float(rate), float(threshold))


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
    c = grid.interior_centers.reshape(-1, 3)
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
    "build_tree", "label_map", "cell_area_steradians", "accumulate_area", "splat",
    "deposit_density", "crystallise", "spawn_segments", "collide", "segment_cascade",
    "CellTree", "grid_cascade", "gaussian_smooth", "boundary_distance", "resample_to", "weighted_quantile",
]
