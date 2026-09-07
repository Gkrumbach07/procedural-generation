"""Crust segments: a point cloud on the unit sphere (PLAN.md section 6.1).

A :class:`Segments` object holds parallel arrays, one entry per segment::

    pos        (M, 3) float64  unit vectors
    mass       (M,)   float64  crust mass per unit area (= thickness * density)
    thickness  (M,)   float64  dimensionless crust thickness
    density    (M,)   float64  dimensionless, in (0, 1]; bedrock = thickness * (1 - density)
    age        (M,)   float64  simulation time since the segment formed
    plate_id   (M,)   int32    owning plate
    area       (M,)   float64  steradians claimed on the label map (rolling blend)
    h_ref      (M,)   float64  bedrock height at the uplift reference step

Geometry helpers work on the unit sphere with *chord* distances (``|p - q|``,
equal to the great-circle angle to second order); all radii in this
package are chord lengths in radians-equivalent units.

Sampling uses a voxel hash on ``[-1, 1]^3`` so Mitchell's best-candidate
sampler (:func:`best_candidate_sphere`) and the greedy Poisson acceptance
used for gap filling (:func:`greedy_accept`) are O(M).  Both are pure
functions of their inputs (candidates are drawn by the caller from
``params.rng``), so output is byte-identical across runs.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

SPHERE_AREA = 4.0 * math.pi
_MAX_VOXELS_PER_AXIS = 192


def mean_spacing(n_segments: int) -> float:
    """Mean segment spacing (chord ≈ arc) for ``n_segments`` uniformly
    covering the unit sphere: ``sqrt(4π / M)``."""
    return math.sqrt(SPHERE_AREA / max(int(n_segments), 1))


def random_unit_vectors(rng: np.random.Generator, shape) -> np.ndarray:
    """Uniformly distributed unit vectors of the given leading shape."""
    p = rng.normal(size=tuple(shape) + (3,))
    p /= np.linalg.norm(p, axis=-1, keepdims=True)
    return p


def _voxel_resolution(radius: float) -> int:
    """Voxels per axis such that the voxel edge is >= ``radius`` (so a
    27-voxel neighbourhood contains every point within ``radius``)."""
    return int(max(1, min(_MAX_VOXELS_PER_AXIS, math.floor(2.0 / max(radius, 1e-9)))))


# --------------------------------------------------------------------------
# voxel hash (numba)
# --------------------------------------------------------------------------
@njit(cache=True, inline="always")
def _voxel_coord(x, G):
    k = int((x + 1.0) * 0.5 * G)
    if k < 0:
        k = 0
    if k > G - 1:
        k = G - 1
    return k


@njit(cache=True, inline="always")
def _hash_insert(head, nxt, pts, i, G):
    ix = _voxel_coord(pts[i, 0], G)
    iy = _voxel_coord(pts[i, 1], G)
    iz = _voxel_coord(pts[i, 2], G)
    v = (ix * G + iy) * G + iz
    nxt[i] = head[v]
    head[v] = i


@njit(cache=True, inline="always")
def _nearest_dist2(head, nxt, pts, qx, qy, qz, G, cap2):
    """Squared chord distance from (qx, qy, qz) to the nearest hashed point,
    capped at ``cap2`` (exact whenever the true distance² < cap2, provided
    the voxel edge is >= sqrt(cap2))."""
    ix = _voxel_coord(qx, G)
    iy = _voxel_coord(qy, G)
    iz = _voxel_coord(qz, G)
    best = cap2
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
                    if d2 < best:
                        best = d2
                    k = nxt[k]
    return best


@njit(cache=True)
def _best_candidate(cands, G, cap2):
    M, K = cands.shape[0], cands.shape[1]
    out = np.empty((M, 3), dtype=np.float64)
    head = np.full(G * G * G, -1, dtype=np.int32)
    nxt = np.empty(M, dtype=np.int32)
    for i in range(M):
        best_j = 0
        best_d = -1.0
        for j in range(K):
            d2 = _nearest_dist2(head, nxt, out, cands[i, j, 0], cands[i, j, 1], cands[i, j, 2], G, cap2)
            if d2 > best_d:
                best_d = d2
                best_j = j
        out[i, 0] = cands[i, best_j, 0]
        out[i, 1] = cands[i, best_j, 1]
        out[i, 2] = cands[i, best_j, 2]
        _hash_insert(head, nxt, out, i, G)
    return out


@njit(cache=True)
def _greedy_accept(cands, existing, r2, G):
    n = cands.shape[0]
    m = existing.shape[0]
    pts = np.empty((m + n, 3), dtype=np.float64)
    head = np.full(G * G * G, -1, dtype=np.int32)
    nxt = np.empty(m + n, dtype=np.int32)
    for i in range(m):
        pts[i, 0] = existing[i, 0]
        pts[i, 1] = existing[i, 1]
        pts[i, 2] = existing[i, 2]
        _hash_insert(head, nxt, pts, i, G)
    accept = np.zeros(n, dtype=np.bool_)
    count = m
    for i in range(n):
        d2 = _nearest_dist2(head, nxt, pts, cands[i, 0], cands[i, 1], cands[i, 2], G, r2 + 1.0)
        if d2 >= r2:
            accept[i] = True
            pts[count, 0] = cands[i, 0]
            pts[count, 1] = cands[i, 1]
            pts[count, 2] = cands[i, 2]
            _hash_insert(head, nxt, pts, count, G)
            count += 1
    return accept


# --------------------------------------------------------------------------
# public sampling API
# --------------------------------------------------------------------------
def best_candidate_sphere(n_points: int, rng: np.random.Generator, candidates: int = 8, cap_radius: float | None = None) -> np.ndarray:
    """Mitchell's best-candidate sampling on the unit sphere: for each new
    point draw ``candidates`` uniform samples and keep the one farthest
    from the points placed so far (distances capped at ``cap_radius``,
    default 2x the mean spacing).  Returns (n_points, 3) float64.  All
    randomness comes from ``rng``; the placement itself is deterministic."""
    n_points = int(n_points)
    if n_points <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    if cap_radius is None:
        cap_radius = 2.0 * mean_spacing(n_points)
    cands = random_unit_vectors(rng, (n_points, int(candidates)))
    G = _voxel_resolution(cap_radius)
    return _best_candidate(np.ascontiguousarray(cands), G, float(cap_radius) ** 2)


def greedy_accept(cands: np.ndarray, existing: np.ndarray, r_min: float) -> np.ndarray:
    """Poisson-style acceptance: walk ``cands`` (n, 3) in order and accept a
    candidate iff no ``existing`` point and no previously accepted candidate
    lies within chord distance ``r_min``.  Returns a bool mask (n,)."""
    cands = np.ascontiguousarray(np.asarray(cands, dtype=np.float64).reshape(-1, 3))
    existing = np.ascontiguousarray(np.asarray(existing, dtype=np.float64).reshape(-1, 3))
    if cands.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    G = _voxel_resolution(r_min)
    return _greedy_accept(cands, existing, float(r_min) ** 2, G)


# --------------------------------------------------------------------------
# container
# --------------------------------------------------------------------------
class Segments:
    """Structure-of-arrays segment store (see module docstring).  All
    mutating methods keep the parallel arrays aligned."""

    FIELDS = ("pos", "mass", "thickness", "density", "age", "plate_id", "area", "h_ref")

    def __init__(self, pos, thickness, density, age, plate_id, area, h_ref=None, mass=None):
        self.pos = np.ascontiguousarray(pos, dtype=np.float64).reshape(-1, 3)
        M = self.pos.shape[0]
        self.thickness = np.array(np.broadcast_to(np.asarray(thickness, dtype=np.float64), (M,)), dtype=np.float64)
        self.density = np.array(np.broadcast_to(np.asarray(density, dtype=np.float64), (M,)), dtype=np.float64)
        self.mass = (self.thickness * self.density) if mass is None else np.array(np.broadcast_to(np.asarray(mass, dtype=np.float64), (M,)), dtype=np.float64)
        self.age = np.array(np.broadcast_to(np.asarray(age, dtype=np.float64), (M,)), dtype=np.float64)
        self.plate_id = np.array(np.broadcast_to(np.asarray(plate_id, dtype=np.int32), (M,)), dtype=np.int32)
        self.area = np.array(np.broadcast_to(np.asarray(area, dtype=np.float64), (M,)), dtype=np.float64)
        self.h_ref = self.height() if h_ref is None else np.array(np.broadcast_to(np.asarray(h_ref, dtype=np.float64), (M,)), dtype=np.float64)

    @property
    def M(self) -> int:
        return self.pos.shape[0]

    def __len__(self) -> int:
        return self.M

    def height(self) -> np.ndarray:
        """Buoyant bedrock height per segment: ``thickness * (1 - density)``."""
        return self.thickness * (1.0 - self.density)

    def total_mass(self) -> float:
        return float(self.mass.sum())

    def compress(self, keep: np.ndarray) -> None:
        """Drop segments where ``keep`` is False (in place)."""
        keep = np.asarray(keep, dtype=bool)
        if keep.all():
            return
        for name in self.FIELDS:
            setattr(self, name, np.ascontiguousarray(getattr(self, name)[keep]))

    def append(self, other: "Segments") -> None:
        if other.M == 0:
            return
        for name in self.FIELDS:
            setattr(self, name, np.ascontiguousarray(np.concatenate([getattr(self, name), getattr(other, name)])))

    def copy(self) -> "Segments":
        return Segments(self.pos.copy(), self.thickness.copy(), self.density.copy(), self.age.copy(), self.plate_id.copy(), self.area.copy(), self.h_ref.copy(), self.mass.copy())

    def renormalise(self) -> None:
        self.pos /= np.linalg.norm(self.pos, axis=1, keepdims=True)


__all__ = ["Segments", "best_candidate_sphere", "greedy_accept", "mean_spacing", "random_unit_vectors", "SPHERE_AREA"]
