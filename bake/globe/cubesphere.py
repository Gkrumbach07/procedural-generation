"""Cube-sphere geometry: face bases, equi-angular (EAC) mapping, Jacobians,
halo maps and face transitions.

This module is the *entire* seam abstraction.  Nothing else in the code base
is allowed to know how faces are oriented relative to each other; everything
goes through :func:`to_sphere`, :func:`from_sphere`, :func:`jacobian`,
:func:`transfer_velocity` and the tables built by :class:`Grid`.

All scalar functions are ``numba.njit`` so they can be called from kernels.
Vectorised NumPy versions are provided for bulk use (``*_v`` suffix) and use
the same arithmetic.

Face bases (OpenGL cubemap convention).  Faces 0..5 = +X, -X, +Y, -Y, +Z, -Z.
``BASES[f] = (right, up, normal)``.

Equi-angular mapping::

    s = tan((u - 0.5) * pi/2),  t = tan((v - 0.5) * pi/2)
    p = normalize(normal + s * right + t * up)

Geometry units: the unit sphere.  ``Grid`` converts to metres using
``R_planet = N * cell_size_m * 4 / (2*pi)`` so one cell along a face's
centre line is exactly ``cell_size_m`` long.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numba import njit

# --------------------------------------------------------------------------
# Face bases
# --------------------------------------------------------------------------
BASES = np.array(
    [
        # right          up             normal
        [[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]],  # 0 +X
        [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]],  # 1 -X
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],  # 2 +Y
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]],  # 3 -Y
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],  # 4 +Z
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],  # 5 -Z
    ],
    dtype=np.float64,
)
FACE_NAMES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
HALF_PI = math.pi / 2.0
TWO_OVER_PI = 2.0 / math.pi


def planet_radius(N: int, cell_size_m: float) -> float:
    """Radius such that a face edge (a quarter great circle) is N cells long."""
    return N * cell_size_m * 4.0 / (2.0 * math.pi)


# --------------------------------------------------------------------------
# Scalar (numba) mapping functions
# --------------------------------------------------------------------------
@njit(cache=True, inline="always")
def _face_axes(face):
    r = BASES[face, 0]
    u = BASES[face, 1]
    n = BASES[face, 2]
    return r, u, n


@njit(cache=True)
def to_sphere(face, u, v):
    """Face-local (u, v) -> unit vector.  Valid slightly outside [0, 1)."""
    s = math.tan((u - 0.5) * HALF_PI)
    t = math.tan((v - 0.5) * HALF_PI)
    r, up, n = _face_axes(face)
    x = n[0] + s * r[0] + t * up[0]
    y = n[1] + s * r[1] + t * up[1]
    z = n[2] + s * r[2] + t * up[2]
    inv = 1.0 / math.sqrt(x * x + y * y + z * z)
    return x * inv, y * inv, z * inv


@njit(cache=True)
def face_of(x, y, z):
    """Face index (argmax |component|, with sign).  Ties resolve X > Y > Z."""
    ax = abs(x)
    ay = abs(y)
    az = abs(z)
    if ax >= ay and ax >= az:
        return 0 if x >= 0.0 else 1
    if ay >= az:
        return 2 if y >= 0.0 else 3
    return 4 if z >= 0.0 else 5


@njit(cache=True)
def from_sphere(x, y, z):
    """Unit vector -> (face, u, v) with u, v in the *closed* interval [0, 1].

    A point lying exactly on a cube edge/corner is assigned to the
    higher-priority face by :func:`face_of` (X > Y > Z) and then has u or v
    exactly 1.0.  Callers computing a cell index must use
    ``min(floor(u*N), N-1)`` (or clamp the extended index as
    ``FaceField.sample_*`` and ``Grid.halo`` already do)."""
    face = face_of(x, y, z)
    r, up, n = _face_axes(face)
    d = x * n[0] + y * n[1] + z * n[2]
    s = (x * r[0] + y * r[1] + z * r[2]) / d
    t = (x * up[0] + y * up[1] + z * up[2]) / d
    u = math.atan(s) * TWO_OVER_PI + 0.5
    v = math.atan(t) * TWO_OVER_PI + 0.5
    return face, u, v


@njit(cache=True)
def project_to_face(face, x, y, z):
    """Project a unit vector onto a *given* face's parametrisation (may fall
    outside [0, 1) — used for extrapolated windows).  Returns (u, v).
    A point exactly on the face's edge gives u or v exactly 0.0 or 1.0;
    clamp ``floor(u*N)`` to ``N-1`` when indexing."""
    r, up, n = _face_axes(face)
    d = x * n[0] + y * n[1] + z * n[2]
    s = (x * r[0] + y * r[1] + z * r[2]) / d
    t = (x * up[0] + y * up[1] + z * up[2]) / d
    u = math.atan(s) * TWO_OVER_PI + 0.5
    v = math.atan(t) * TWO_OVER_PI + 0.5
    return u, v


@njit(cache=True)
def jacobian(face, u, v):
    """d p / d u and d p / d v on the unit sphere (each a 3-vector).

    Returns (jux, juy, juz, jvx, jvy, jvz).  Both vectors are tangent to the
    sphere at p and span the tangent plane; they are *not* orthogonal away
    from the face axes (up to 120 degrees apart at cube corners).
    """
    a = (u - 0.5) * HALF_PI
    b = (v - 0.5) * HALF_PI
    s = math.tan(a)
    t = math.tan(b)
    dsdu = HALF_PI * (1.0 + s * s)
    dtdv = HALF_PI * (1.0 + t * t)
    r, up, n = _face_axes(face)
    qx = n[0] + s * r[0] + t * up[0]
    qy = n[1] + s * r[1] + t * up[1]
    qz = n[2] + s * r[2] + t * up[2]
    ql = math.sqrt(qx * qx + qy * qy + qz * qz)
    px = qx / ql
    py = qy / ql
    pz = qz / ql
    # dq/du = r * dsdu ; dp/du = (dq/du - p (p . dq/du)) / |q|
    dux = r[0] * dsdu
    duy = r[1] * dsdu
    duz = r[2] * dsdu
    pd = px * dux + py * duy + pz * duz
    jux = (dux - px * pd) / ql
    juy = (duy - py * pd) / ql
    juz = (duz - pz * pd) / ql
    dvx = up[0] * dtdv
    dvy = up[1] * dtdv
    dvz = up[2] * dtdv
    pd = px * dvx + py * dvy + pz * dvz
    jvx = (dvx - px * pd) / ql
    jvy = (dvy - py * pd) / ql
    jvz = (dvz - pz * pd) / ql
    return jux, juy, juz, jvx, jvy, jvz


@njit(cache=True)
def metric_uv(face, u, v):
    """Metric tensor of the (u, v) parametrisation on the unit sphere:
    (g_uu, g_uv, g_vv)."""
    jux, juy, juz, jvx, jvy, jvz = jacobian(face, u, v)
    guu = jux * jux + juy * juy + juz * juz
    guv = jux * jvx + juy * jvy + juz * jvz
    gvv = jvx * jvx + jvy * jvy + jvz * jvz
    return guu, guv, gvv


@njit(cache=True)
def transfer_vector(face_src, u_src, v_src, a, b, face_dst, u_dst, v_dst):
    """Re-express a tangent vector given in (u, v)-parameter components on
    ``face_src`` at (u_src, v_src) into parameter components on ``face_dst``
    at (u_dst, v_dst).  Both points must be the same point on the sphere
    (or within the halo overlap).  Exact: solves the 2x2 normal equations.
    """
    jux, juy, juz, jvx, jvy, jvz = jacobian(face_src, u_src, v_src)
    wx = a * jux + b * jvx
    wy = a * juy + b * jvy
    wz = a * juz + b * jvz
    kux, kuy, kuz, kvx, kvy, kvz = jacobian(face_dst, u_dst, v_dst)
    guu = kux * kux + kuy * kuy + kuz * kuz
    guv = kux * kvx + kuy * kvy + kuz * kvz
    gvv = kvx * kvx + kvy * kvy + kvz * kvz
    ru = kux * wx + kuy * wy + kuz * wz
    rv = kvx * wx + kvy * wy + kvz * wz
    det = guu * gvv - guv * guv
    a2 = (gvv * ru - guv * rv) / det
    b2 = (guu * rv - guv * ru) / det
    return a2, b2


@njit(cache=True)
def transfer_velocity(face, u, v, a, b):
    """Particle face transition (PLAN 2.6), for (u, v) that may have left
    [0, 1).  ``(a, b)`` are parameter-space velocity components (du/dt,
    dv/dt); multiply by N to get cells.  Returns (face', u', v', a', b').
    If (u, v) is inside the face nothing changes."""
    if u >= 0.0 and u < 1.0 and v >= 0.0 and v < 1.0:
        return face, u, v, a, b
    x, y, z = to_sphere(face, u, v)
    f2, u2, v2 = from_sphere(x, y, z)
    if f2 == face:
        # Numerical edge case: clamp into the face.
        u2 = min(max(u2, 0.0), 0.999999999)
        v2 = min(max(v2, 0.0), 0.999999999)
        return face, u2, v2, a, b
    a2, b2 = transfer_vector(face, u, v, a, b, f2, u2, v2)
    return f2, u2, v2, a2, b2


@njit(cache=True)
def great_circle_distance(x0, y0, z0, x1, y1, z1):
    d = x0 * x1 + y0 * y1 + z0 * z1
    if d > 1.0:
        d = 1.0
    if d < -1.0:
        d = -1.0
    return math.acos(d)


# --------------------------------------------------------------------------
# Vectorised NumPy versions (same arithmetic)
# --------------------------------------------------------------------------
def to_sphere_v(face, u, v):
    """Vectorised to_sphere.  ``face`` int array, ``u``/``v`` float arrays.
    Returns (..., 3) float64 unit vectors."""
    face = np.asarray(face, dtype=np.int64)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    s = np.tan((u - 0.5) * HALF_PI)
    t = np.tan((v - 0.5) * HALF_PI)
    r = BASES[face, 0]
    up = BASES[face, 1]
    n = BASES[face, 2]
    q = n + s[..., None] * r + t[..., None] * up
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return q


def face_of_v(p):
    p = np.asarray(p, dtype=np.float64)
    ax, ay, az = np.abs(p[..., 0]), np.abs(p[..., 1]), np.abs(p[..., 2])
    face = np.where(
        (ax >= ay) & (ax >= az),
        np.where(p[..., 0] >= 0, 0, 1),
        np.where(ay >= az, np.where(p[..., 1] >= 0, 2, 3), np.where(p[..., 2] >= 0, 4, 5)),
    )
    return face.astype(np.int64)


def from_sphere_v(p):
    """Vectorised from_sphere: (..., 3) -> (face, u, v) arrays.  As for the
    scalar version u, v lie in the closed interval [0, 1]: a point exactly
    on a cube edge/corner goes to the higher-priority face (X > Y > Z) with
    u or v exactly 1.0, so index computations must use
    ``min(floor(u*N), N-1)``."""
    p = np.asarray(p, dtype=np.float64)
    face = face_of_v(p)
    return face, *project_to_face_v(face, p)


def project_to_face_v(face, p):
    p = np.asarray(p, dtype=np.float64)
    face = np.asarray(face, dtype=np.int64)
    r = BASES[face, 0]
    up = BASES[face, 1]
    n = BASES[face, 2]
    d = np.sum(p * n, axis=-1)
    s = np.sum(p * r, axis=-1) / d
    t = np.sum(p * up, axis=-1) / d
    u = np.arctan(s) * TWO_OVER_PI + 0.5
    v = np.arctan(t) * TWO_OVER_PI + 0.5
    return u, v


def jacobian_v(face, u, v):
    """Vectorised jacobian: returns (Ju, Jv) each (..., 3)."""
    face = np.asarray(face, dtype=np.int64)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    a = (u - 0.5) * HALF_PI
    b = (v - 0.5) * HALF_PI
    s = np.tan(a)
    t = np.tan(b)
    dsdu = HALF_PI * (1.0 + s * s)
    dtdv = HALF_PI * (1.0 + t * t)
    r = BASES[face, 0]
    up = BASES[face, 1]
    n = BASES[face, 2]
    q = n + s[..., None] * r + t[..., None] * up
    ql = np.linalg.norm(q, axis=-1, keepdims=True)
    p = q / ql
    du = r * dsdu[..., None]
    dv = up * dtdv[..., None]
    ju = (du - p * np.sum(p * du, axis=-1, keepdims=True)) / ql
    jv = (dv - p * np.sum(p * dv, axis=-1, keepdims=True)) / ql
    return ju, jv


def transfer_vector_v(face_src, u_src, v_src, ab, face_dst, u_dst, v_dst):
    """Vectorised transfer_vector; ``ab`` is (..., 2).  Returns (..., 2)."""
    ju, jv = jacobian_v(face_src, u_src, v_src)
    w = ab[..., :1] * ju + ab[..., 1:] * jv
    ku, kv = jacobian_v(face_dst, u_dst, v_dst)
    guu = np.sum(ku * ku, -1)
    guv = np.sum(ku * kv, -1)
    gvv = np.sum(kv * kv, -1)
    ru = np.sum(ku * w, -1)
    rv = np.sum(kv * w, -1)
    det = guu * gvv - guv * guv
    a2 = (gvv * ru - guv * rv) / det
    b2 = (guu * rv - guv * ru) / det
    return np.stack([a2, b2], axis=-1)


# --------------------------------------------------------------------------
# Grid: per-resolution geometry tables
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class HaloMap:
    """Gather tables for filling halo cells.

    Every halo cell is a linear combination of ``S`` source cells that are
    always *interior* cells (possibly of up to three faces):

    ``dst`` (K,) flat indices (into a (6, NE, NE) array) of the halo cells;
    ``src`` (K, S) flat indices of the source cells;
    ``w`` (K, S) weights (rows sum to 1; unused slots have weight 0);
    ``rot`` (K, S, 2, 2) matrices re-expressing a contravariant
    cell-component vector of each *source* cell into the *destination*
    face's components;
    ``nearest`` (K,) flat index of the single best source (for integer
    fields).

    Edge blocks (halo cells beyond one edge) use 4×4 cubic Lagrange
    interpolation on the neighbouring face (``w``; O(h⁴), so second-order
    stencils see no seam) with a bilinear alternative (``w_lin``; no
    overshoot).  Corner blocks (beyond two edges, where three faces meet)
    use a local least-squares quadratic fit over the nearest interior
    cells for ``w`` (exact for quadratic functions; weights may be
    negative) and convex inverse-distance-squared weights over the 4
    nearest interior cells for ``w_lin`` (non-negative, O(h²); monotone).

    The weight and rotation tables are float64 so the stated accuracy holds
    for float64 fields (``FaceField.exchange_halos`` casts them to the
    field dtype, so float32 fields pay nothing extra).  Table memory at
    N = 1024, H = 4 is ~76 MB (w + w_lin + rot).
    """

    dst: np.ndarray
    src: np.ndarray
    w: np.ndarray
    w_lin: np.ndarray
    rot: np.ndarray
    nearest: np.ndarray


def lagrange4_weights(t: np.ndarray) -> np.ndarray:
    """Cubic Lagrange weights for nodes at -1, 0, 1, 2 evaluated at ``t``
    (t measured from node 0).  Shape (..., 4).  Valid as extrapolation for
    t slightly outside [0, 1)."""
    t = np.asarray(t, dtype=np.float64)
    return np.stack(
        [
            -t * (t - 1) * (t - 2) / 6.0,
            (t + 1) * (t - 1) * (t - 2) / 2.0,
            -(t + 1) * t * (t - 2) / 2.0,
            (t + 1) * t * (t - 1) / 6.0,
        ],
        axis=-1,
    )


def _linear_weights_on_4(t: np.ndarray) -> np.ndarray:
    """Bilinear (2-node) weights expressed on the 4-node stencil -1, 0, 1, 2:
    uses the two nodes bracketing ``t`` (t may lie in [-1, 2))."""
    t = np.asarray(t, dtype=np.float64)
    w = np.zeros(t.shape + (4,), dtype=np.float64)
    lo = np.clip(np.floor(t).astype(np.int64), -1, 1)  # left node
    frac = np.clip(t - lo, 0.0, 1.0)
    idx = lo + 1  # index of left node in the 4-node array
    np.put_along_axis(w, idx[..., None], (1 - frac)[..., None], axis=-1)
    np.put_along_axis(w, (idx + 1)[..., None], frac[..., None], axis=-1)
    return w


class Grid:
    """Geometry cache for a cube-sphere resolution.

    Parameters
    ----------
    N : cells per face edge.
    H : halo width (cells).
    cell_size_m : nominal cell size (metres) along a face centre line.
    R_planet : planet radius in metres; derived from N and cell_size_m if
        omitted (a face edge is then exactly N cells of ``cell_size_m``).

    Attributes (all lazily computed and cached; extended arrays include the
    halo, i.e. shape (6, NE, NE, ...), NE = N + 2H):
    ``centers``  (6, NE, NE, 3) float64 unit vectors of cell centres.
    ``metric``   (6, NE, NE, 3) float32 dimensionless metric (g_ii, g_ij,
        g_jj) in cell units (≈ identity at face centre).  Physical squared
        length of a cell-displacement (di, dj) is
        ``cell_size_m**2 * (g_ii di² + 2 g_ij di dj + g_jj dj²)``.
    ``metric_inv`` (6, NE, NE, 3) float32 inverse metric.
    ``cell_area`` (6, NE, NE) float32 in m².
    ``halo``     HaloMap.
    ``owner``    (6, NE, NE) int64 flat index of the interior cell that owns
        each extended cell (identity for interior cells); used for D8 and
        any "which real cell is my neighbour" query.
    """

    HALO_SOURCES = 16

    def __init__(self, N: int, H: int = 4, cell_size_m: float = 50.0, R_planet: float | None = None):
        if N < 2 * H:
            raise ValueError("N must be at least 2H")
        self.N = int(N)
        self.H = int(H)
        self.NE = self.N + 2 * self.H
        self.cell_size_m = float(cell_size_m)
        self.R_planet = float(R_planet) if R_planet is not None else planet_radius(N, cell_size_m)

    # -- coordinates -------------------------------------------------------
    def cell_uv(self, ei, ej):
        """Extended cell index -> (u, v) at the cell centre (may be outside
        [0, 1) in the halo)."""
        return (np.asarray(ei) - self.H + 0.5) / self.N, (np.asarray(ej) - self.H + 0.5) / self.N

    def uv_cell(self, u, v):
        """(u, v) -> fractional extended index (fi, fj) such that integer
        values sit on cell centres."""
        return np.asarray(u) * self.N - 0.5 + self.H, np.asarray(v) * self.N - 0.5 + self.H

    def flat_index(self, face, ei, ej):
        """(face, extended i, extended j) -> flat index into (6, NE, NE)."""
        return (np.asarray(face) * self.NE + np.asarray(ei)) * self.NE + np.asarray(ej)

    def unflat_index(self, flat):
        flat = np.asarray(flat)
        f = flat // (self.NE * self.NE)
        rem = flat - f * self.NE * self.NE
        return f, rem // self.NE, rem % self.NE

    @cached_property
    def _uv_grid(self):
        e = np.arange(self.NE)
        u = (e - self.H + 0.5) / self.N
        U, V = np.meshgrid(u, u, indexing="ij")
        return U, V

    @cached_property
    def centers(self) -> np.ndarray:
        U, V = self._uv_grid
        out = np.empty((6, self.NE, self.NE, 3), dtype=np.float64)
        for f in range(6):
            out[f] = to_sphere_v(np.full(U.shape, f), U, V)
        return out

    @cached_property
    def interior_centers(self) -> np.ndarray:
        H = self.H
        return self.centers[:, H : H + self.N, H : H + self.N]

    @cached_property
    def _metric_uv(self):
        U, V = self._uv_grid
        g = np.empty((6, self.NE, self.NE, 3), dtype=np.float64)
        for f in range(6):
            ju, jv = jacobian_v(np.full(U.shape, f), U, V)
            g[f, ..., 0] = np.sum(ju * ju, -1)
            g[f, ..., 1] = np.sum(ju * jv, -1)
            g[f, ..., 2] = np.sum(jv * jv, -1)
        return g

    @cached_property
    def metric(self) -> np.ndarray:
        # cell units: J_cell = J_uv / N ; metres: R * J_cell ; dimensionless: / cell_size
        k = (self.R_planet / (self.N * self.cell_size_m)) ** 2
        return (self._metric_uv * k).astype(np.float32)

    @cached_property
    def metric_inv(self) -> np.ndarray:
        g = self.metric.astype(np.float64)
        det = g[..., 0] * g[..., 2] - g[..., 1] ** 2
        inv = np.empty_like(g)
        inv[..., 0] = g[..., 2] / det
        inv[..., 1] = -g[..., 1] / det
        inv[..., 2] = g[..., 0] / det
        return inv.astype(np.float32)

    @cached_property
    def sqrt_det_metric(self) -> np.ndarray:
        g = self.metric.astype(np.float64)
        return np.sqrt(g[..., 0] * g[..., 2] - g[..., 1] ** 2).astype(np.float32)

    @cached_property
    def cell_area(self) -> np.ndarray:
        """Cell area in m² (extended array; halo values are extrapolated)."""
        return (self.sqrt_det_metric.astype(np.float64) * self.cell_size_m**2).astype(np.float32)

    @cached_property
    def interior_cell_area(self) -> np.ndarray:
        H = self.H
        return self.cell_area[:, H : H + self.N, H : H + self.N]

    # -- halo tables -------------------------------------------------------
    @cached_property
    def halo_mask(self) -> np.ndarray:
        """(NE, NE) bool: True for halo cells (same pattern on every face)."""
        m = np.ones((self.NE, self.NE), dtype=bool)
        m[self.H : self.H + self.N, self.H : self.H + self.N] = False
        return m

    @cached_property
    def corner_mask(self) -> np.ndarray:
        """(NE, NE) bool: the four H×H corner blocks of the halo."""
        H, NE = self.H, self.NE
        m = np.zeros((NE, NE), dtype=bool)
        m[:H, :H] = m[:H, NE - H :] = m[NE - H :, :H] = m[NE - H :, NE - H :] = True
        return m

    @cached_property
    def edge_halo_mask(self) -> np.ndarray:
        """(NE, NE) bool: halo cells that are not in corner blocks."""
        return self.halo_mask & ~self.corner_mask

    @cached_property
    def halo(self) -> HaloMap:
        N, H, NE, S = self.N, self.H, self.NE, self.HALO_SOURCES
        # ---- edge blocks: cubic (and bilinear) from the neighbouring face --
        ei, ej = np.nonzero(self.edge_halo_mask)
        u, v = self.cell_uv(ei, ej)
        K1 = ei.size
        dst_e = np.empty(6 * K1, dtype=np.int64)
        src_e = np.zeros((6 * K1, S), dtype=np.int64)
        w_e = np.zeros((6 * K1, S), dtype=np.float64)
        wl_e = np.zeros((6 * K1, S), dtype=np.float64)
        rot_e = np.zeros((6 * K1, S, 2, 2), dtype=np.float64)
        near_e = np.empty(6 * K1, dtype=np.int64)
        offs = np.array([-1, 0, 1, 2])
        for f in range(6):
            faces = np.full(K1, f)
            p = to_sphere_v(faces, u, v)
            sf, su, sv = from_sphere_v(p)
            fi, fj = self.uv_cell(su, sv)
            # cubic stencil i0-1 .. i0+2 must stay inside the interior
            i0 = np.clip(np.floor(fi).astype(np.int64), H + 1, H + N - 3)
            j0 = np.clip(np.floor(fj).astype(np.int64), H + 1, H + N - 3)
            ti = fi - i0
            tj = fj - j0
            wi = lagrange4_weights(ti)  # (K1, 4)
            wj = lagrange4_weights(tj)
            ii = i0[:, None] + offs[None, :]  # (K1, 4)
            jj = j0[:, None] + offs[None, :]
            srcs = self.flat_index(sf[:, None, None], ii[:, :, None], jj[:, None, :]).reshape(K1, 16)
            wc = (wi[:, :, None] * wj[:, None, :]).reshape(K1, 16)
            # bilinear on the two stencil nodes bracketing the point
            wli = _linear_weights_on_4(ti)
            wlj = _linear_weights_on_4(tj)
            wl = (wli[:, :, None] * wlj[:, None, :]).reshape(K1, 16)
            sl = slice(f * K1, (f + 1) * K1)
            dst_e[sl] = self.flat_index(f, ei, ej)
            src_e[sl] = srcs
            w_e[sl] = wc
            wl_e[sl] = wl
            near_e[sl] = srcs[np.arange(K1), np.argmax(wl, axis=1)]
            e1 = transfer_vector_v(sf, su, sv, np.tile([[1.0, 0.0]], (K1, 1)), faces, u, v)
            e2 = transfer_vector_v(sf, su, sv, np.tile([[0.0, 1.0]], (K1, 1)), faces, u, v)
            R = np.empty((K1, 2, 2), dtype=np.float64)
            R[:, :, 0] = e1
            R[:, :, 1] = e2
            rot_e[sl] = R[:, None, :, :]
        # ---- corner blocks: local plane fit over nearest interior cells --
        ci, cj = np.nonzero(self.corner_mask)
        K2 = ci.size
        cu, cv = self.cell_uv(ci, cj)
        # candidate interior cells: within (H + 3) cells of any face corner
        M = H + 3
        cand_f, cand_i, cand_j = [], [], []
        for f in range(6):
            for (a0, b0) in ((H, H), (H, H + N - M), (H + N - M, H), (H + N - M, H + N - M)):
                ii, jj = np.meshgrid(np.arange(a0, a0 + M), np.arange(b0, b0 + M), indexing="ij")
                cand_f.append(np.full(ii.size, f))
                cand_i.append(ii.ravel())
                cand_j.append(jj.ravel())
        cand_f = np.concatenate(cand_f)
        cand_i = np.concatenate(cand_i)
        cand_j = np.concatenate(cand_j)
        cand_flat = self.flat_index(cand_f, cand_i, cand_j)
        cand_pos = self.centers[cand_f, cand_i, cand_j]
        cand_u, cand_v = self.cell_uv(cand_i, cand_j)
        from scipy.spatial import cKDTree

        tree = cKDTree(cand_pos)
        dst_c = np.empty(6 * K2, dtype=np.int64)
        src_c = np.empty((6 * K2, S), dtype=np.int64)
        w_c = np.empty((6 * K2, S), dtype=np.float64)
        wl_c = np.zeros((6 * K2, S), dtype=np.float64)
        rot_c = np.empty((6 * K2, S, 2, 2), dtype=np.float64)
        near_c = np.empty(6 * K2, dtype=np.int64)
        for f in range(6):
            faces = np.full(K2, f)
            p = to_sphere_v(faces, cu, cv)  # (K2, 3)
            _, idx = tree.query(p, k=S)  # (K2, S)
            sl = slice(f * K2, (f + 1) * K2)
            dst_c[sl] = self.flat_index(f, ci, cj)
            src_c[sl] = cand_flat[idx]
            near_c[sl] = cand_flat[idx[:, 0]]
            # local tangent frame at p
            ax = np.where(np.abs(p[:, :1]) < 0.9, np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]]))
            e1 = ax - p * np.sum(ax * p, axis=1, keepdims=True)
            e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
            e2 = np.cross(p, e1)
            q = cand_pos[idx]  # (K2, S, 3), nearest first
            # monotone alternative (w_lin): normalised inverse-distance-squared
            # weights over the 4 nearest sources (chord distance; never 0
            # because halo centres differ from interior centres)
            dist = np.linalg.norm(q[:, :4] - p[:, None, :], axis=2)  # (K2, 4)
            wl = 1.0 / (dist * dist)
            wl /= wl.sum(axis=1, keepdims=True)
            wl_c[sl, :4] = wl
            # gnomonic projection of neighbours onto the tangent plane at p
            dq = q / np.sum(q * p[:, None, :], axis=2, keepdims=True) - p[:, None, :]
            x = np.sum(dq * e1[:, None, :], axis=2)
            y = np.sum(dq * e2[:, None, :], axis=2)
            # quadratic least-squares fit f ≈ c0 + c1 x + c2 y + c3 x² + c4 xy + c5 y²
            A = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=2)  # (K2, S, 6)
            # weights = first row of pinv(A): evaluates the fitted surface at p
            pinv = np.linalg.pinv(A)  # (K2, 6, S)
            w_c[sl] = pinv[:, 0, :]
            # rotation from each source cell's face components to face f at p
            sfc = cand_f[idx]
            suc = cand_u[idx]
            svc = cand_v[idx]
            ff = np.repeat(faces[:, None], S, axis=1)
            uu = np.repeat(cu[:, None], S, axis=1)
            vv = np.repeat(cv[:, None], S, axis=1)
            r1 = transfer_vector_v(sfc, suc, svc, np.broadcast_to([1.0, 0.0], (K2, S, 2)), ff, uu, vv)
            r2 = transfer_vector_v(sfc, suc, svc, np.broadcast_to([0.0, 1.0], (K2, S, 2)), ff, uu, vv)
            rot_c[sl, :, :, 0] = r1
            rot_c[sl, :, :, 1] = r2
        return HaloMap(
            dst=np.concatenate([dst_e, dst_c]),
            src=np.concatenate([src_e, src_c]),
            w=np.concatenate([w_e, w_c]),
            w_lin=np.concatenate([wl_e, wl_c]),
            rot=np.concatenate([rot_e, rot_c]),
            nearest=np.concatenate([near_e, near_c]),
        )

    @cached_property
    def owner(self) -> np.ndarray:
        """(6, NE, NE) int64 flat index (into a (6, NE, NE) array) of the
        interior cell owning each extended cell.  Interior cells map to
        themselves; halo cells map to the nearest interior cell (of the
        neighbouring face)."""
        NE = self.NE
        own = np.arange(6 * NE * NE, dtype=np.int64)
        hm = self.halo
        own[hm.dst] = hm.nearest
        return own.reshape(6, NE, NE)

    @cached_property
    def owner_face_ij(self) -> np.ndarray:
        """(6, NE, NE, 3) int32: owner as (face, i, j) *interior* indices
        (0-based, no halo offset)."""
        NE = self.NE
        flat = self.owner
        f = flat // (NE * NE)
        rem = flat - f * NE * NE
        i = rem // NE - self.H
        j = rem % NE - self.H
        return np.stack([f, i, j], axis=-1).astype(np.int32)

    # -- convenience --------------------------------------------------------
    def latitude(self) -> np.ndarray:
        """Latitude in radians (extended array), +Z is the pole."""
        return np.arcsin(np.clip(self.centers[..., 2], -1.0, 1.0))

    def describe(self) -> str:
        return (
            f"Grid(N={self.N}, H={self.H}, cell={self.cell_size_m} m, "
            f"R_planet={self.R_planet/1000:.2f} km, cells={6*self.N*self.N:,})"
        )


_GRID_CACHE: dict[tuple, Grid] = {}


def get_grid(N: int, H: int = 4, cell_size_m: float = 50.0, R_planet: float | None = None) -> Grid:
    """Process-wide cache of Grid objects (tables are expensive to build)."""
    key = (int(N), int(H), float(cell_size_m), None if R_planet is None else float(R_planet))
    g = _GRID_CACHE.get(key)
    if g is None:
        g = Grid(N, H, cell_size_m, R_planet)
        _GRID_CACHE[key] = g
    return g
