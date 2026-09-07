"""Basin windows, coarse -> fine upsampling and detail noise (PLAN 10.2
steps 1-2).

A refinement window is a square range of *coarse* cells on one face —
the basin's bbox grown by ``refine.halo_cells`` and padded to a square
(the erosion kernel works on ``(NE, NE)`` arrays) — sampled at ``R x``
through :meth:`~globe.field.FaceField.sample_window`, which follows
points beyond the face edge onto the owning face (vector fields are
rotated into the window face's basis).  The kernel halo of the window is
``H = R`` fine cells (one coarse cell) and is sampled like everything
else, so the extended array covers the coarse range grown by one cell.

Fields: ``height`` and ``sediment`` are upsampled bicubically and
re-split so that ``height + sediment`` is exactly the bicubic surface
(sediment clipped at 0); everything else bilinearly; ``basin_id``
nearest.  ``precip`` is a *volume per cell*, so a fine cell gets ``1/R²``
of the coarse volume.  The lake depth ``water_surface - surface`` is
upsampled instead of the water surface itself (a bilinear water surface
against a bicubic terrain would show spurious 'lakes' on curved slopes).

Detail noise: ridged FBM value noise on the fine lattice, wavelengths
from 4 coarse cells down to ~2 fine cells, seeded by
``params.rng("refine", basin_id)``, amplitude ``detail_amp * min(slope *
cell_size_m, relief_3x3) * (0.5 + 0.5 * hardness)`` (metres; the local
height change over one coarse cell, never more than the coarse 3x3
relief), added to ``height`` only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..cubesphere import Grid, jacobian_v
from ..field import FaceField

#: coarse fields a basin job reads
COARSE_INPUTS = ("height", "sediment", "hardness", "precip", "evap", "discharge", "momentum", "water_surface", "basin_id")


@dataclass(frozen=True)
class Window:
    """Square coarse-cell range ``[ci0, ci1) x [cj0, cj1)`` on ``face``
    (may extend beyond the face) refined ``R x``; the fine window is
    ``n x n`` with origin fine cell ``(ci0 * R, cj0 * R)`` in face-fine
    coordinates; the kernel array is ``NE = n + 2H``, ``H = R``."""

    face: int
    ci0: int
    ci1: int
    cj0: int
    cj1: int
    R: int

    @property
    def nc(self) -> int:
        return self.ci1 - self.ci0

    @property
    def n(self) -> int:
        return self.nc * self.R

    @property
    def H(self) -> int:
        return self.R

    @property
    def NE(self) -> int:
        return self.n + 2 * self.H

    @property
    def fi0(self) -> int:
        """Face-fine index of the window's first fine row / column."""
        return self.ci0 * self.R

    @property
    def fj0(self) -> int:
        return self.cj0 * self.R

    def ext_range(self) -> tuple[int, int, int, int]:
        """Coarse range of the extended (halo-included) array."""
        return self.ci0 - 1, self.ci1 + 1, self.cj0 - 1, self.cj1 + 1

    def coarse_to_ext(self, i: int, j: int) -> tuple[int, int]:
        """Fine extended index of the first fine cell of coarse cell (i, j)."""
        return (i - self.ci0) * self.R + self.H, (j - self.cj0) * self.R + self.H


def basin_window(basin: dict, R: int, halo_cells: int) -> Window:
    """Window of a ``basins.json`` record: bbox (exclusive) grown by
    ``halo_cells`` and padded symmetrically to a square."""
    i0, j0, i1, j1 = (int(x) for x in basin["bbox"])
    ci0, ci1, cj0, cj1 = i0 - halo_cells, i1 + halo_cells, j0 - halo_cells, j1 + halo_cells
    wi, wj = ci1 - ci0, cj1 - cj0
    n = max(wi, wj)
    if wi < n:
        pad = n - wi
        ci0 -= pad // 2
        ci1 += pad - pad // 2
    if wj < n:
        pad = n - wj
        cj0 -= pad // 2
        cj1 += pad - pad // 2
    return Window(int(basin["face"]), ci0, ci1, cj0, cj1, int(R))


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------
def sample(field: FaceField, win: Window, order: int = 1) -> np.ndarray:
    """Field on the window's extended fine array ``(NE, NE[, C])``."""
    a0, a1, b0, b1 = win.ext_range()
    out = field.sample_window(win.face, a0, a1, b0, b1, win.R, order=order)
    assert out.shape[:2] == (win.NE, win.NE), (out.shape, win)
    return np.ascontiguousarray(out)


def window_metric(win: Window, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Dimensionless metric ``(g_ii, g_ij, g_jj)`` and its inverse at the
    extended fine cell centres (fine cell units; the same arithmetic as
    ``Grid.metric`` evaluated on the window face's extended
    parametrisation, so it is valid beyond the face edge)."""
    R, N = win.R, grid.N
    Nf = N * R
    a0, a1, b0, b1 = win.ext_range()
    u = (np.arange(a0 * R, a1 * R) + 0.5) / Nf
    v = (np.arange(b0 * R, b1 * R) + 0.5) / Nf
    U, V = np.meshgrid(u, v, indexing="ij")
    ju, jv = jacobian_v(np.full(U.shape, win.face), U, V)
    g = np.empty(U.shape + (3,), dtype=np.float64)
    g[..., 0] = np.sum(ju * ju, -1)
    g[..., 1] = np.sum(ju * jv, -1)
    g[..., 2] = np.sum(jv * jv, -1)
    cell_fine = grid.cell_size_m / R
    k = (grid.R_planet / (Nf * cell_fine)) ** 2
    metric = (g * k).astype(np.float32)
    gd = metric.astype(np.float64)
    det = gd[..., 0] * gd[..., 2] - gd[..., 1] ** 2
    inv = np.empty_like(gd)
    inv[..., 0] = gd[..., 2] / det
    inv[..., 1] = -gd[..., 1] / det
    inv[..., 2] = gd[..., 0] / det
    return metric, inv.astype(np.float32)


# --------------------------------------------------------------------------
# coarse derived fields (per process, once)
# --------------------------------------------------------------------------
def coarse_derived(fields: dict[str, FaceField]) -> dict[str, FaceField]:
    """``surface``, ``slope`` (rise/run), ``relief`` (3x3 max - min of the
    surface, metres) and ``depth`` (lake depth, 0 on ocean) on the coarse
    grid, valid on the extended arrays (indices 1..NE-2)."""
    h, s = fields["height"], fields["sediment"]
    grid = h.grid
    surf = FaceField(grid, (h.data.astype(np.float32) + s.data.astype(np.float32)), name="surface")
    slope = surf.gradient().vec_norm()  # |grad| in m per m
    slope.name = "slope"
    d = surf.data
    rel = np.zeros_like(d)
    mx = d[:, 1:-1, 1:-1].copy()
    mn = d[:, 1:-1, 1:-1].copy()
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            blk = d[:, 1 + di : d.shape[1] - 1 + di, 1 + dj : d.shape[2] - 1 + dj]
            np.maximum(mx, blk, out=mx)
            np.minimum(mn, blk, out=mn)
    rel[:, 1:-1, 1:-1] = mx - mn
    ws = fields["water_surface"].data
    depth = np.where(ws > 0.0, np.maximum(ws - d, 0.0), 0.0).astype(np.float32)
    return {
        "surface": surf,
        "slope": slope,
        "relief": FaceField(grid, rel.astype(np.float32), name="relief"),
        "depth": FaceField(grid, depth, name="depth"),
    }


# --------------------------------------------------------------------------
# detail noise
# --------------------------------------------------------------------------
def value_noise_2d(shape: tuple[int, int], wavelength: float, rng: np.random.Generator) -> np.ndarray:
    """Smoothstep-interpolated value noise in [-1, 1] on an integer lattice
    with the given wavelength (cells)."""
    n0, n1 = shape
    x = np.arange(n0) / wavelength
    y = np.arange(n1) / wavelength
    m0 = int(math.floor(x[-1])) + 2
    m1 = int(math.floor(y[-1])) + 2
    lat = rng.random((m0 + 1, m1 + 1))
    i0 = np.floor(x).astype(np.int64)
    j0 = np.floor(y).astype(np.int64)
    tx = x - i0
    ty = y - j0
    tx = tx * tx * (3 - 2 * tx)
    ty = ty * ty * (3 - 2 * ty)
    a = lat[i0[:, None], j0[None, :]]
    b = lat[i0[:, None] + 1, j0[None, :]]
    c = lat[i0[:, None], j0[None, :] + 1]
    d = lat[i0[:, None] + 1, j0[None, :] + 1]
    v = a * (1 - tx)[:, None] * (1 - ty)[None, :] + b * tx[:, None] * (1 - ty)[None, :] + c * (1 - tx)[:, None] * ty[None, :] + d * tx[:, None] * ty[None, :]
    return v * 2.0 - 1.0


def ridged_fbm(shape: tuple[int, int], base_wavelength: float, rng: np.random.Generator, min_wavelength: float = 2.0, gain: float = 0.5) -> np.ndarray:
    """Ridged FBM (Musgrave-style ``(1 - |n|)²`` octaves, lacunarity 2)
    normalised to zero mean and unit maximum magnitude over the array."""
    out = np.zeros(shape, dtype=np.float64)
    w = 1.0
    lam = float(base_wavelength)
    total = 0.0
    while lam >= min_wavelength:
        n = value_noise_2d(shape, lam, rng)
        r = 1.0 - np.abs(n)
        out += w * r * r
        total += w
        w *= gain
        lam *= 0.5
    if total > 0:
        out /= total
    out -= out.mean()
    m = float(np.abs(out).max())
    if m > 0:
        out /= m
    return out.astype(np.float32)


def detail_noise(win: Window, slope: np.ndarray, relief: np.ndarray, hardness: np.ndarray, detail_amp: float, cell_size_m: float, rng: np.random.Generator) -> np.ndarray:
    """Detail noise in metres on the extended window (``NE x NE``):
    ``detail_amp * min(slope * cell_size_m, relief) * (0.5 + 0.5 hardness)
    * ridged_fbm``.  ``slope`` (rise/run), ``relief`` (m) and ``hardness``
    are the bilinearly upsampled coarse fields on the same array."""
    amp = float(detail_amp) * np.minimum(np.maximum(slope, 0.0) * float(cell_size_m), np.maximum(relief, 0.0)) * (0.5 + 0.5 * np.clip(hardness, 0.0, 1.0))
    if detail_amp <= 0.0:
        return np.zeros((win.NE, win.NE), dtype=np.float32)
    noise = ridged_fbm((win.NE, win.NE), 4.0 * win.R, rng)
    return (amp * noise).astype(np.float32)


# --------------------------------------------------------------------------
# everything a basin job needs on its window
# --------------------------------------------------------------------------
def upsample_window(fields: dict[str, FaceField], derived: dict[str, FaceField], win: Window, grid: Grid) -> dict[str, np.ndarray]:
    """Upsampled inputs on the extended window (``NE x NE[, C]``, float32
    unless noted):

    ``height0``, ``sediment0`` (bicubic, re-split so the sum is the
    bicubic surface), ``hardness``, ``precip`` (per fine cell), ``evap``,
    ``discharge``, ``momentum`` (``(NE, NE, 2)``, rotated), ``depth``
    (lake depth), ``slope``, ``relief``, ``basin_id`` (int32, nearest),
    ``metric`` / ``metric_inv`` (``(NE, NE, 3)``).
    """
    R = win.R
    surf = sample(derived["surface"], win, order=3)
    sed = np.maximum(sample(fields["sediment"], win, order=3), 0.0).astype(np.float32)
    height = (surf - sed).astype(np.float32)
    out = {
        "height0": height,
        "sediment0": sed,
        "hardness": np.clip(sample(fields["hardness"], win, 1), 0.0, 1.0).astype(np.float32),
        "precip": (np.maximum(sample(fields["precip"], win, 1), 0.0) / float(R * R)).astype(np.float32),
        "evap": np.maximum(sample(fields["evap"], win, 1), 0.0).astype(np.float32),
        "discharge": np.maximum(sample(fields["discharge"], win, 1), 0.0).astype(np.float32),
        "momentum": sample(fields["momentum"], win, 1).astype(np.float32),
        "depth": np.maximum(sample(derived["depth"], win, 1), 0.0).astype(np.float32),
        "slope": np.maximum(sample(derived["slope"], win, 1), 0.0).astype(np.float32),
        "relief": np.maximum(sample(derived["relief"], win, 1), 0.0).astype(np.float32),
        "basin_id": sample(fields["basin_id"], win).astype(np.int32),
    }
    out["metric"], out["metric_inv"] = window_metric(win, grid)
    return out


def upsample_face(fields: dict[str, FaceField], derived: dict[str, FaceField], face: int, R: int, i0: int, i1: int) -> dict[str, np.ndarray]:
    """Plain upsample of the fine rows ``[i0*R, i1*R)`` of a whole face
    (the base pass of the rasteriser): ``height``, ``sediment``,
    ``hardness``, ``water_surface`` (surface + lake depth on land, 0 on
    ocean), ``discharge``, ``basin_id``; arrays ``((i1-i0)*R, N*R)``."""
    N = fields["height"].N
    f = int(face)
    surf = derived["surface"].sample_window(f, i0, i1, 0, N, R, order=3)
    sed = np.maximum(fields["sediment"].sample_window(f, i0, i1, 0, N, R, order=3), 0.0).astype(np.float32)
    height = (surf - sed).astype(np.float32)
    bid = fields["basin_id"].sample_window(f, i0, i1, 0, N, R).astype(np.int32)
    depth = np.maximum(derived["depth"].sample_window(f, i0, i1, 0, N, R, order=1), 0.0)
    ws = np.where(bid < 0, np.float32(0.0), (surf + depth).astype(np.float32)).astype(np.float32)
    return {
        "height": height,
        "sediment": sed,
        "hardness": np.clip(fields["hardness"].sample_window(f, i0, i1, 0, N, R, order=1), 0.0, 1.0).astype(np.float32),
        "water_surface": ws,
        "discharge": np.maximum(fields["discharge"].sample_window(f, i0, i1, 0, N, R, order=1), 0.0).astype(np.float32),
        "basin_id": bid,
    }


__all__ = [
    "COARSE_INPUTS", "Window", "basin_window", "sample", "window_metric", "coarse_derived",
    "value_noise_2d", "ridged_fbm", "detail_noise", "upsample_window", "upsample_face",
]
