"""Fine-grid helpers for the derive stage: memmapped ``fine/`` I/O, banded
upsampling of coarse fields and a metric-aware slope on one fine face.

Fine arrays are ``fine/<name>.f{k}.npy``, ``N_fine x N_fine``, ``[i, j]``,
no halo (docs/DEVELOPING.md).  At ``N_fine = 4096`` a float32 face is
64 MB, so everything here works one face at a time and reads through
``np.load(mmap_mode="r")`` / writes through ``open_memmap``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..field import FaceField

FINE_DIR = "fine"


def fine_path(store, name: str, face: int) -> Path:
    return store.root / FINE_DIR / f"{name}.f{face}.npy"


def has_fine(store, name: str) -> bool:
    return all(fine_path(store, name, k).exists() for k in range(6))


def load_face(store, name: str, face: int, mmap: bool = True) -> np.ndarray:
    """One fine face; a read-only memmap by default (slice it, do not
    write).  Missing fields raise ``FileNotFoundError``."""
    return np.load(fine_path(store, name, face), mmap_mode="r" if mmap else None)


def write_face(store, name: str, face: int, arr: np.ndarray) -> Path:
    """Write one fine face through ``open_memmap`` (never a full copy of
    the world in memory)."""
    p = fine_path(store, name, face)
    p.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(arr)
    mm = np.lib.format.open_memmap(p, mode="w+", dtype=arr.dtype, shape=arr.shape)
    mm[...] = arr
    mm.flush()
    del mm
    return p


def upsample_nearest(a: np.ndarray, R: int) -> np.ndarray:
    """(N, N[, C]) -> (N R, N R[, C]) by cell replication (what the stub
    refine does)."""
    if R == 1:
        return np.asarray(a)
    return np.repeat(np.repeat(a, R, axis=0), R, axis=1)


def upsample_face(field: FaceField, face: int, R: int, order: int = 1, band: int = 64) -> np.ndarray:
    """Bilinear (``order=1``) or cubic (``order=3``) upsample of one face of
    a coarse ``FaceField`` to the fine grid, in bands of ``band`` coarse
    rows so the working set stays small.  Uses ``FaceField.sample_window``
    (halo-aware, so face edges interpolate against the neighbouring face).
    Returns ``(N R, N R)`` float32."""
    N = field.N
    out = np.empty((N * R, N * R), dtype=np.float32)
    for i0 in range(0, N, band):
        i1 = min(N, i0 + band)
        out[i0 * R : i1 * R] = field.sample_window(face, i0, i1, 0, N, R, order=order)
    return out


def coarse_face_array(grid, ext: np.ndarray, face: int) -> np.ndarray:
    """Interior ``(N, N[, C])`` view of face ``face`` of an extended
    ``(6, NE, NE[, C])`` grid table (``metric``, ``cell_area`` ...)."""
    H, N = grid.H, grid.N
    return ext[face, H : H + N, H : H + N]


def outside_cells(Nf: int, face: int, ii: np.ndarray, jj: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fine cells ``(ii, jj)`` of ``face`` (integer indices, usually
    *outside* ``[0, Nf)``) -> the ``(face2, i2, j2)`` fine cell that owns
    their centre, through ``cubesphere.to_sphere_v`` / ``from_sphere_v``
    (the same mapping ``FaceField.sample_window`` and the coarse halo use;
    consistent with ``grid.owner_face_ij`` at ``Nf = N R``).  Cells inside
    the face map to themselves."""
    from ..cubesphere import from_sphere_v, to_sphere_v

    ii = np.asarray(ii, dtype=np.int64)
    jj = np.asarray(jj, dtype=np.int64)
    u = (ii + 0.5) / Nf
    v = (jj + 0.5) / Nf
    f2, u2, v2 = from_sphere_v(to_sphere_v(np.full(ii.shape, face, dtype=np.int64), u, v))
    i2 = np.clip(np.floor(u2 * Nf).astype(np.int64), 0, Nf - 1)
    j2 = np.clip(np.floor(v2 * Nf).astype(np.int64), 0, Nf - 1)
    return f2.astype(np.int64), i2, j2


#: sides of a face in the order used by ``edge_neighbors`` (and
#: ``lakes.coarse_lake_labels``): 0 = i < 0, 1 = i >= Nf, 2 = j < 0, 3 = j >= Nf.
SIDES = ("-i", "+i", "-j", "+j")


def edge_neighbors(Nf: int, face: int, depth: int = 1) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """For each of the 4 sides of a fine face the ``(face2, i2, j2)`` owner
    cells of the ``depth`` rows of cells just outside it: side 0 / 1 give
    arrays ``(depth, Nf)`` with row ``k`` = fine row ``i = -depth + k`` /
    ``i = Nf + k``; side 2 / 3 give ``(Nf, depth)`` with column ``k`` =
    ``j = -depth + k`` / ``j = Nf + k``.  With ``depth = 1`` every owner
    lies on an edge row / column of its face."""
    r = np.arange(Nf, dtype=np.int64)
    d = np.arange(depth, dtype=np.int64)
    out = []
    for side in range(4):
        if side == 0:
            ii, jj = np.meshgrid(-depth + d, r, indexing="ij")
        elif side == 1:
            ii, jj = np.meshgrid(Nf + d, r, indexing="ij")
        elif side == 2:
            ii, jj = np.meshgrid(r, -depth + d, indexing="ij")
        else:
            ii, jj = np.meshgrid(r, Nf + d, indexing="ij")
        out.append(outside_cells(Nf, face, ii, jj))
    return out


def face_pads(getter, Nf: int, face: int, depth: int) -> list[np.ndarray]:
    """The 4 side pads of a fine face (``edge_neighbors`` layout) read from
    the neighbouring faces: ``getter(face2, i2, j2)`` returns the values of
    those cells (e.g. a memmapped fine surface indexed with the arrays)."""
    return [np.asarray(getter(f2, i2, j2), dtype=np.float32) for f2, i2, j2 in edge_neighbors(Nf, face, depth)]


def slope_magnitude(surface: np.ndarray, metric_inv_coarse: np.ndarray, R: int, cell_size_m: float, stencil: int = 1, pads=None) -> np.ndarray:
    """|grad surface| (rise/run, dimensionless) on one fine face.

    ``surface`` is ``(Nf, Nf)`` metres; ``metric_inv_coarse`` the coarse
    face's interior inverse metric ``(N, N, 3)`` (dimensionless, cell
    units; the fine grid has the same dimensionless metric at the same
    ``(u, v)``, so it is replicated ``R x``); ``cell_size_m`` is the *fine*
    cell size.  Central differences over ``+-stencil`` cells.  ``pads``
    (from :func:`face_pads` with ``depth = stencil``) supplies the
    neighbouring faces' cells beyond the four sides so the stencil is the
    same everywhere (no seam); without pads the outermost ``stencil`` cells
    fall back to a clamped, one-sided difference."""
    h = np.asarray(surface, dtype=np.float32)
    Nf = h.shape[0]
    s = max(1, int(stencil))
    if pads is not None:
        top, bottom, left, right = (np.asarray(p, dtype=np.float32) for p in pads)
        if top.shape != (s, Nf) or bottom.shape != (s, Nf) or left.shape != (Nf, s) or right.shape != (Nf, s):
            raise ValueError("pads must have the edge_neighbors layout with depth == stencil")
        hi = np.concatenate([top, h, bottom], axis=0)
        hj = np.concatenate([left, h, right], axis=1)
        gi = (hi[2 * s :, :] - hi[: Nf, :]) / np.float32(2 * s)
        gj = (hj[:, 2 * s :] - hj[:, :Nf]) / np.float32(2 * s)
    else:
        idx = np.arange(Nf)
        ip = np.minimum(idx + s, Nf - 1)
        im = np.maximum(idx - s, 0)
        span = (ip - im).astype(np.float32)
        gi = (h[ip, :] - h[im, :]) / span[:, None]
        gj = (h[:, ip] - h[:, im]) / span[None, :]
    g = np.asarray(metric_inv_coarse, dtype=np.float32)
    if R > 1:
        g = upsample_nearest(g, R)
    s2 = g[..., 0] * gi * gi + 2.0 * g[..., 1] * gi * gj + g[..., 2] * gj * gj
    return (np.sqrt(np.maximum(s2, 0.0)) / np.float32(cell_size_m)).astype(np.float32)


def face_bands(N: int, band: int):
    """Yield ``(i0, i1)`` row bands covering ``range(N)``."""
    for i0 in range(0, N, band):
        yield i0, min(N, i0 + band)


__all__ = [
    "FINE_DIR", "fine_path", "has_fine", "load_face", "write_face", "upsample_nearest", "upsample_face",
    "coarse_face_array", "outside_cells", "SIDES", "edge_neighbors", "face_pads", "slope_magnitude", "face_bands",
]
