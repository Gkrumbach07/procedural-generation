"""FaceField: six face arrays with halos, halo exchange, stencils, sampling.

Layout: ``data`` has shape ``(6, NE, NE)`` for scalars or ``(6, NE, NE, C)``
for C-component fields, ``NE = N + 2H``.  Interior cell ``(i, j)`` of face
``f`` is ``data[f, i + H, j + H]``; ``field.interior`` is a view of it.

Vector fields (``is_vector=True``, C == 2) hold contravariant cell
components (see package docstring).  Their halo exchange applies the exact
per-cell rotation from :class:`~globe.cubesphere.HaloMap`.

Float fields exchange halos by cubic interpolation (``exchange_halos(linear=True)``
for bilinear, monotone); integer and bool fields use the nearest source cell.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numba import njit, prange

from .cubesphere import Grid, from_sphere_v, get_grid, jacobian_v


class FaceField:
    __slots__ = ("grid", "data", "is_vector", "name")

    def __init__(self, grid: Grid, data: np.ndarray, is_vector: bool = False, name: str = ""):
        if data.shape[:3] != (6, grid.NE, grid.NE):
            raise ValueError(f"data shape {data.shape} does not match grid NE={grid.NE}")
        if is_vector and (data.ndim != 4 or data.shape[3] != 2):
            raise ValueError("vector fields must have shape (6, NE, NE, 2)")
        self.grid = grid
        self.data = data
        self.is_vector = bool(is_vector)
        self.name = name

    # -- constructors ------------------------------------------------------
    @classmethod
    def zeros(cls, grid: Grid, ncomp: int = 1, dtype=np.float32, is_vector: bool = False, name: str = "") -> "FaceField":
        shape = (6, grid.NE, grid.NE) if ncomp == 1 else (6, grid.NE, grid.NE, ncomp)
        return cls(grid, np.zeros(shape, dtype=dtype), is_vector=is_vector, name=name)

    @classmethod
    def full(cls, grid: Grid, value, ncomp: int = 1, dtype=np.float32, name: str = "") -> "FaceField":
        f = cls.zeros(grid, ncomp, dtype, name=name)
        f.data[...] = value
        return f

    @classmethod
    def from_interior(cls, grid: Grid, arr: np.ndarray, is_vector: bool = False, name: str = "", exchange: bool = True) -> "FaceField":
        """Build from a (6, N, N[, C]) interior array."""
        arr = np.asarray(arr)
        if arr.shape[:3] != (6, grid.N, grid.N):
            raise ValueError(f"interior shape {arr.shape} does not match grid N={grid.N}")
        f = cls.zeros(grid, 1 if arr.ndim == 3 else arr.shape[3], arr.dtype, is_vector=is_vector, name=name)
        f.interior[...] = arr
        if exchange:
            f.exchange_halos()
        return f

    @classmethod
    def from_function(cls, grid: Grid, fn, dtype=np.float32, name: str = "") -> "FaceField":
        """Evaluate ``fn(points3)`` (points3: (..., 3) unit vectors) on all
        extended cell centres — no exchange needed, halos are analytic."""
        vals = np.asarray(fn(grid.centers), dtype=dtype)
        return cls(grid, np.ascontiguousarray(vals), name=name)

    def copy(self, name: str | None = None) -> "FaceField":
        return FaceField(self.grid, self.data.copy(), self.is_vector, self.name if name is None else name)

    def astype(self, dtype) -> "FaceField":
        return FaceField(self.grid, self.data.astype(dtype), self.is_vector, self.name)

    def zeros_like(self, dtype=None, name: str = "") -> "FaceField":
        return FaceField(self.grid, np.zeros_like(self.data, dtype=dtype), self.is_vector, name)

    # -- views ---------------------------------------------------------------
    @property
    def N(self) -> int:
        return self.grid.N

    @property
    def H(self) -> int:
        return self.grid.H

    @property
    def ncomp(self) -> int:
        return 1 if self.data.ndim == 3 else self.data.shape[3]

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def interior(self) -> np.ndarray:
        H, N = self.grid.H, self.grid.N
        return self.data[:, H : H + N, H : H + N]

    def face(self, f: int) -> np.ndarray:
        """Interior view of one face, shape (N, N[, C])."""
        return self.interior[f]

    # -- halo exchange -------------------------------------------------------
    def exchange_halos(self, linear: bool = False) -> "FaceField":
        """Fill halo cells from neighbouring faces.

        Default: cubic interpolation (accurate enough for second-order
        stencils; may overshoot slightly at sharp features).  ``linear=True``
        uses bilinear weights (monotone, no overshoot) — use it for masks,
        probabilities and anything that must stay within bounds.  Integer
        and bool fields always use the nearest cell."""
        hm = self.grid.halo
        flat = self.data.reshape(6 * self.grid.NE * self.grid.NE, -1)
        if np.issubdtype(self.data.dtype, np.integer) or self.data.dtype == np.bool_:
            flat[hm.dst] = flat[hm.nearest]
        else:
            w = (hm.w_lin if linear else hm.w).astype(flat.dtype)
            if self.is_vector:
                # rotate every source into the destination frame, then blend
                g = np.einsum("ks,ksab,ksb->ka", w, hm.rot.astype(flat.dtype), flat[hm.src])
            else:
                g = np.einsum("ks,ks...->k...", w, flat[hm.src])
            flat[hm.dst] = g
        return self

    # -- stencils ------------------------------------------------------------
    def grad_cov(self) -> tuple[np.ndarray, np.ndarray]:
        """Covariant grid gradient (df/di, df/dj) per cell, central
        differences over the extended array (valid for indices 1..NE-2).
        Scalar fields only."""
        d = self.data.astype(np.float64) if self.data.dtype != np.float64 else self.data
        gi = np.zeros_like(d)
        gj = np.zeros_like(d)
        gi[:, 1:-1, :] = 0.5 * (d[:, 2:, :] - d[:, :-2, :])
        gj[:, :, 1:-1] = 0.5 * (d[:, :, 2:] - d[:, :, :-2])
        return gi, gj

    def gradient(self) -> "FaceField":
        """Physical gradient ∇f as a contravariant vector field:
        ``d = g^{-1} (df/di, df/dj) / cell_size_m²`` so that
        ``∇f = d_i E_i + d_j E_j`` (E_i = metres per cell along i).

        * steepest-ascent direction in cell components is ``d``;
        * ``d.vec_norm()`` is |∇f| in field-units per metre;
        * ``d.vec_dot(disp)`` for a displacement ``disp`` in cells is the
          change of f along it (same as :meth:`directional_derivative`).
        Valid for extended indices 1..NE-2 (exchange halos first)."""
        gi, gj = self.grad_cov()
        ginv = self.grid.metric_inv.astype(np.float64)
        cs2 = self.grid.cell_size_m ** 2
        out = np.empty(self.data.shape[:3] + (2,), dtype=np.float32)
        out[..., 0] = (ginv[..., 0] * gi + ginv[..., 1] * gj) / cs2
        out[..., 1] = (ginv[..., 1] * gi + ginv[..., 2] * gj) / cs2
        return FaceField(self.grid, out, is_vector=True, name=f"grad({self.name})")

    def directional_derivative(self, vec: "FaceField") -> "FaceField":
        """Change of this scalar per unit step along ``vec`` (contravariant
        cell components): ``df/di * a + df/dj * b``."""
        gi, gj = self.grad_cov()
        v = vec.data
        out = (gi * v[..., 0] + gj * v[..., 1]).astype(np.float32)
        return FaceField(self.grid, out, name=f"d{self.name}/d{vec.name}")

    def vec_norm(self) -> "FaceField":
        """Physical length in metres of a contravariant cell-component vector
        field (per cell-step)."""
        g = self.grid.metric.astype(np.float64)
        a, b = self.data[..., 0].astype(np.float64), self.data[..., 1].astype(np.float64)
        n2 = g[..., 0] * a * a + 2 * g[..., 1] * a * b + g[..., 2] * b * b
        return FaceField(self.grid, (np.sqrt(np.maximum(n2, 0.0)) * self.grid.cell_size_m).astype(np.float32), name=f"|{self.name}|")

    def vec_dot(self, other: "FaceField") -> np.ndarray:
        """Physical dot product (m²) of two contravariant vector fields."""
        g = self.grid.metric.astype(np.float64)
        a, b = self.data[..., 0].astype(np.float64), self.data[..., 1].astype(np.float64)
        c, d = other.data[..., 0].astype(np.float64), other.data[..., 1].astype(np.float64)
        return (g[..., 0] * a * c + g[..., 1] * (a * d + b * c) + g[..., 2] * b * d) * self.grid.cell_size_m**2

    def laplacian(self) -> "FaceField":
        """Laplace-Beltrami operator (per m²), flux-form discretisation of
        (1/√g) ∂_a(√g g^{ab} ∂_b f).  Valid for extended indices 1..NE-2
        (halo must be exchanged first)."""
        d = self.data.astype(np.float64)
        ginv = self.grid.metric_inv.astype(np.float64)
        sq = self.grid.sqrt_det_metric.astype(np.float64)
        out = _laplacian_kernel(d, ginv, sq)
        out /= self.grid.cell_size_m**2
        return FaceField(self.grid, out.astype(np.float32), name=f"lap({self.name})")

    # -- sampling ------------------------------------------------------------
    def sample_bilinear(self, face, u, v) -> np.ndarray:
        """Bilinear sample at face-local (u, v) (arrays).  (u, v) may lie in
        the halo; indices are clamped to the extended array."""
        face = np.asarray(face, dtype=np.int64)
        fi, fj = self.grid.uv_cell(u, v)
        NE = self.grid.NE
        i0 = np.clip(np.floor(fi).astype(np.int64), 0, NE - 2)
        j0 = np.clip(np.floor(fj).astype(np.int64), 0, NE - 2)
        wi = np.clip(fi - i0, 0.0, 1.0)
        wj = np.clip(fj - j0, 0.0, 1.0)
        d = self.data
        if d.ndim == 4:
            wi = wi[..., None]
            wj = wj[..., None]
        return (
            d[face, i0, j0] * (1 - wi) * (1 - wj)
            + d[face, i0 + 1, j0] * wi * (1 - wj)
            + d[face, i0, j0 + 1] * (1 - wi) * wj
            + d[face, i0 + 1, j0 + 1] * wi * wj
        )

    def sample_sphere(self, p) -> np.ndarray:
        """Bilinear sample at unit vectors ``p`` (..., 3).  For vector fields
        the result is in the components of the face that owns ``p``."""
        face, u, v = from_sphere_v(p)
        return self.sample_bilinear(face, u, v)

    def sample_cubic(self, face, u, v) -> np.ndarray:
        """4x4 cubic Lagrange sample at face-local (u, v) (arrays).  The
        stencil is clamped to the extended array, so points up to ~H-2
        cells outside the face are still interpolated (not extrapolated)."""
        from .cubesphere import lagrange4_weights

        face = np.asarray(face, dtype=np.int64)
        fi, fj = self.grid.uv_cell(u, v)
        NE = self.grid.NE
        i0 = np.clip(np.floor(fi).astype(np.int64), 1, NE - 3)
        j0 = np.clip(np.floor(fj).astype(np.int64), 1, NE - 3)
        wi = lagrange4_weights(fi - i0)  # (..., 4)
        wj = lagrange4_weights(fj - j0)
        d = self.data
        out = None
        for a in range(4):
            row = None
            for b in range(4):
                val = d[face, i0 + a - 1, j0 + b - 1]
                w = wj[..., b]
                if d.ndim == 4:
                    w = w[..., None]
                row = val * w if row is None else row + val * w
            wa = wi[..., a]
            if d.ndim == 4:
                wa = wa[..., None]
            out = row * wa if out is None else out + row * wa
        return out

    def sample_window(self, face: int, i0: int, i1: int, j0: int, j1: int, R: int = 1, order: int = 1) -> np.ndarray:
        """Sample the field on a window of the face's *coarse* cell range
        ``[i0, i1) × [j0, j1)`` (interior indices, no halo offset; may extend
        beyond the face, even beyond the halo) at refinement factor ``R``:
        fine cell centres at ``u = (i + (k + 0.5) / R) / N``.  Returns an
        array of shape ``((i1-i0)*R, (j1-j0)*R[, C])`` indexed ``[fi, fj]``.

        Points outside the face are looked up on the face that owns them
        (via ``from_sphere``); vector fields are rotated into ``face``'s
        components.  ``order`` 1 = bilinear, 3 = cubic (use 3 for height)."""
        from .cubesphere import from_sphere_v, to_sphere_v, transfer_vector_v

        N = self.grid.N
        ui = (np.arange(i0 * R, i1 * R) + 0.5) / (N * R)
        vj = (np.arange(j0 * R, j1 * R) + 0.5) / (N * R)
        U, V = np.meshgrid(ui, vj, indexing="ij")
        inside = (U >= 0) & (U < 1) & (V >= 0) & (V < 1)
        faces = np.full(U.shape, face, dtype=np.int64)
        su, sv = U.copy(), V.copy()
        if not inside.all():
            p = to_sphere_v(faces[~inside], U[~inside], V[~inside])
            f2, u2, v2 = from_sphere_v(p)
            faces[~inside] = f2
            su[~inside] = u2
            sv[~inside] = v2
        if np.issubdtype(self.data.dtype, np.integer) or self.data.dtype == np.bool_:
            return self.sample_nearest(faces, su, sv)
        vals = self.sample_cubic(faces, su, sv) if order == 3 else self.sample_bilinear(faces, su, sv)
        if self.is_vector:
            other = faces != face
            if other.any():
                vals = np.array(vals, dtype=np.float64)
                vals[other] = transfer_vector_v(faces[other], su[other], sv[other], vals[other], np.full(int(other.sum()), face), U[other], V[other])
        return vals.astype(self.data.dtype)

    def sample_nearest(self, face, u, v) -> np.ndarray:
        face = np.asarray(face, dtype=np.int64)
        fi, fj = self.grid.uv_cell(u, v)
        NE = self.grid.NE
        i0 = np.clip(np.rint(fi).astype(np.int64), 0, NE - 1)
        j0 = np.clip(np.rint(fj).astype(np.int64), 0, NE - 1)
        return self.data[face, i0, j0]

    # -- statistics ----------------------------------------------------------
    def stats(self) -> dict:
        a = self.interior
        return {
            "name": self.name,
            "dtype": str(a.dtype),
            "shape": list(a.shape),
            "min": float(np.nanmin(a)),
            "max": float(np.nanmax(a)),
            "mean": float(np.nanmean(a)),
            "std": float(np.nanstd(a)),
        }

    # -- io -----------------------------------------------------------------
    def save(self, directory: str | Path, name: str | None = None) -> list[Path]:
        """Write interior of each face as ``<dir>/<name>.f{k}.npy``."""
        name = name or self.name
        if not name:
            raise ValueError("field needs a name to be saved")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for k in range(6):
            p = directory / f"{name}.f{k}.npy"
            np.save(p, np.ascontiguousarray(self.interior[k]))
            paths.append(p)
        return paths

    @classmethod
    def load(cls, directory: str | Path, name: str, grid: Grid, is_vector: bool | None = None) -> "FaceField":
        """Load ``<dir>/<name>.f{0..5}.npy`` and exchange halos.  Faces are
        copied into a fresh halo array; the on-disk .npy files are
        memory-mappable directly with ``np.load(mmap_mode="r")`` when only a
        single interior face is needed."""
        directory = Path(directory)
        faces = [np.load(directory / f"{name}.f{k}.npy") for k in range(6)]
        arr = np.stack(faces, axis=0)
        if is_vector is None:
            is_vector = arr.ndim == 4 and arr.shape[3] == 2 and name in VECTOR_FIELD_NAMES
        return cls.from_interior(grid, arr, is_vector=is_vector, name=name)

    @staticmethod
    def exists(directory: str | Path, name: str) -> bool:
        directory = Path(directory)
        return all((directory / f"{name}.f{k}.npy").exists() for k in range(6))

    def __repr__(self) -> str:
        return f"FaceField({self.name!r}, N={self.N}, H={self.H}, ncomp={self.ncomp}, dtype={self.dtype}, vector={self.is_vector})"


#: Names of stored fields that are tangent vectors (contravariant cell comps).
VECTOR_FIELD_NAMES = frozenset({"wind", "momentum", "plate_vel", "momentum_track"})


def rotation_field(grid: Grid, omega, name: str = "", dtype=np.float32) -> FaceField:
    """``v = omega x p`` (rigid rotation about the axis ``omega``, |omega| in
    radians per step) as contravariant cell components on every extended
    cell — a smooth, seamless tangent field (PLAN section 6 plate motion).
    Halos are analytic, no exchange needed."""
    U, V = grid._uv_grid
    om = np.asarray(omega, dtype=np.float64)
    out = np.empty((6, grid.NE, grid.NE, 2), dtype=np.float64)
    for f in range(6):
        ju, jv = jacobian_v(np.full(U.shape, f), U, V)
        w = np.cross(om[None, None, :], grid.centers[f])
        guu, guv, gvv = (ju * ju).sum(-1), (ju * jv).sum(-1), (jv * jv).sum(-1)
        wu, wv = (w * ju).sum(-1), (w * jv).sum(-1)
        det = guu * gvv - guv * guv
        out[f, ..., 0] = (gvv * wu - guv * wv) / det * grid.N  # per-u -> per-cell
        out[f, ..., 1] = (guu * wv - guv * wu) / det * grid.N
    return FaceField(grid, out.astype(dtype), is_vector=True, name=name)


@njit(cache=True, parallel=True)
def _laplacian_kernel(d, ginv, sq):
    F, NE, _ = d.shape
    out = np.zeros_like(d)
    for f in prange(F):
        for i in range(1, NE - 1):
            for j in range(1, NE - 1):
                # d f / d j at (i, j), (i+1, j), (i-1, j) ; d f / d i at (i, j±1)
                fj_c = 0.5 * (d[f, i, j + 1] - d[f, i, j - 1])
                fj_p = 0.5 * (d[f, i + 1, j + 1] - d[f, i + 1, j - 1])
                fj_m = 0.5 * (d[f, i - 1, j + 1] - d[f, i - 1, j - 1])
                fi_c = 0.5 * (d[f, i + 1, j] - d[f, i - 1, j])
                fi_p = 0.5 * (d[f, i + 1, j + 1] - d[f, i - 1, j + 1])
                fi_m = 0.5 * (d[f, i + 1, j - 1] - d[f, i - 1, j - 1])
                # fluxes through i-faces
                gii = 0.5 * (ginv[f, i, j, 0] + ginv[f, i + 1, j, 0])
                gij = 0.5 * (ginv[f, i, j, 1] + ginv[f, i + 1, j, 1])
                s = 0.5 * (sq[f, i, j] + sq[f, i + 1, j])
                Fi_p = s * (gii * (d[f, i + 1, j] - d[f, i, j]) + gij * 0.5 * (fj_c + fj_p))
                gii = 0.5 * (ginv[f, i, j, 0] + ginv[f, i - 1, j, 0])
                gij = 0.5 * (ginv[f, i, j, 1] + ginv[f, i - 1, j, 1])
                s = 0.5 * (sq[f, i, j] + sq[f, i - 1, j])
                Fi_m = s * (gii * (d[f, i, j] - d[f, i - 1, j]) + gij * 0.5 * (fj_c + fj_m))
                # fluxes through j-faces
                gjj = 0.5 * (ginv[f, i, j, 2] + ginv[f, i, j + 1, 2])
                gij = 0.5 * (ginv[f, i, j, 1] + ginv[f, i, j + 1, 1])
                s = 0.5 * (sq[f, i, j] + sq[f, i, j + 1])
                Fj_p = s * (gjj * (d[f, i, j + 1] - d[f, i, j]) + gij * 0.5 * (fi_c + fi_p))
                gjj = 0.5 * (ginv[f, i, j, 2] + ginv[f, i, j - 1, 2])
                gij = 0.5 * (ginv[f, i, j, 1] + ginv[f, i, j - 1, 1])
                s = 0.5 * (sq[f, i, j] + sq[f, i, j - 1])
                Fj_m = s * (gjj * (d[f, i, j] - d[f, i, j - 1]) + gij * 0.5 * (fi_c + fi_m))
                out[f, i, j] = (Fi_p - Fi_m + Fj_p - Fj_m) / sq[f, i, j]
    return out


# --------------------------------------------------------------------------
# Kernel-side helpers (njit) for use inside other jitted code
# --------------------------------------------------------------------------
@njit(cache=True, inline="always")
def bilinear_ext(face_arr, fi, fj):
    """Bilinear sample of one extended face array (NE, NE) at fractional
    extended index (fi, fj) (cell centres at integers).  Clamped."""
    NE = face_arr.shape[0]
    i0 = int(np.floor(fi))
    j0 = int(np.floor(fj))
    if i0 < 0:
        i0 = 0
    if i0 > NE - 2:
        i0 = NE - 2
    if j0 < 0:
        j0 = 0
    if j0 > NE - 2:
        j0 = NE - 2
    wi = fi - i0
    wj = fj - j0
    if wi < 0.0:
        wi = 0.0
    if wi > 1.0:
        wi = 1.0
    if wj < 0.0:
        wj = 0.0
    if wj > 1.0:
        wj = 1.0
    return (
        face_arr[i0, j0] * (1 - wi) * (1 - wj)
        + face_arr[i0 + 1, j0] * wi * (1 - wj)
        + face_arr[i0, j0 + 1] * (1 - wi) * wj
        + face_arr[i0 + 1, j0 + 1] * wi * wj
    )


def zeros(grid: Grid, ncomp: int = 1, dtype=np.float32, is_vector: bool = False, name: str = "") -> FaceField:
    return FaceField.zeros(grid, ncomp, dtype, is_vector, name)


__all__ = ["FaceField", "VECTOR_FIELD_NAMES", "bilinear_ext", "rotation_field", "zeros", "get_grid"]
