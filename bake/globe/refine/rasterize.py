"""Rasterise basin results into the per-face fine memmaps (PLAN 10.2 step 5).

``fine/<name>.f{k}.npy`` (``N_fine x N_fine``, ``[i, j]``, no halo;
docs/DEVELOPING.md) for ``height, sediment, water_surface, discharge,
hardness`` (f32) and ``basin_id`` (i32) are created once by the driver
(:func:`create_fine`: zero-filled, ``basin_id`` -1) with
``np.lib.format.open_memmap`` so a face is never fully resident.

Two writers, both usable from a worker process (every process maps the
same files; writes go to disjoint cells):

* :func:`write_base_face` — the plain upsample of a whole face in row
  strips (bicubic surface split into height/sediment, bilinear hardness /
  discharge, nearest basin id, water surface = surface + lake depth on
  land and 0 on the ocean).  It fills every cell, so tiles are complete
  even where a basin job never writes (ocean, and cells outside every
  basin window); the driver runs it *before* the basin jobs so the write
  order is fixed.
* :func:`write_result` — a :class:`~globe.refine.basin_job.BasinResult`
  into the cells of its window that lie on the face **and** whose fine
  basin id (nearest-upsampled coarse id, i.e. block replication) is the
  job's basin.  Overlapping windows therefore never clobber each other and
  every land cell is written by exactly one job, so the raster does not
  depend on the job order.  A feather of ``refine.feather_cells`` fine
  cells blends the refined surface, water surface and discharge back to
  the plain upsample towards the divide: weight ``w = smoothstep((d - 1)
  / feather_cells)`` with ``d`` the chessboard distance to the nearest
  cell outside the basin (or the window border), so the frozen divide ring
  (``d = 1``) is exactly the plain upsample on both sides of the divide and
  the refined detail ramps in with zero slope at both ends over
  ``feather_cells`` cells (>= 2R, a coarse cell on each side, so the ramp
  is below the hillshade's resolution; a linear 2-cell ramp left a visible
  crease at every divide).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from ..config import WorldParams
from ..field import FaceField
from .upsample import Window, upsample_face

#: fine fields written by the stage (name -> dtype); ``fill`` per field
FINE_FIELDS: dict[str, np.dtype] = {
    "height": np.dtype(np.float32),
    "sediment": np.dtype(np.float32),
    "water_surface": np.dtype(np.float32),
    "discharge": np.dtype(np.float32),
    "hardness": np.dtype(np.float32),
    "basin_id": np.dtype(np.int32),
}
FINE_FILL = {"basin_id": -1}

#: fields feathered towards the plain upsample at divides (with their
#: plain-upsample reference in the job result)
FEATHERED = (("height", "height0"), ("sediment", "sediment0"), ("water_surface", "water_surface0"), ("discharge", "discharge0"))


def fine_dir(root: str | Path) -> Path:
    return Path(root) / "fine"


def fine_path(root: str | Path, name: str, face: int) -> Path:
    return fine_dir(root) / f"{name}.f{int(face)}.npy"


def create_fine(root: str | Path, params: WorldParams) -> Path:
    """Create (overwrite) the six memmaps of every fine field, zero-filled
    (``basin_id`` -1).  Returns the ``fine/`` directory."""
    d = fine_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    N = params.N_fine
    for name, dtype in FINE_FIELDS.items():
        fill = FINE_FILL.get(name, 0)
        for f in range(6):
            mm = np.lib.format.open_memmap(fine_path(root, name, f), mode="w+", dtype=dtype, shape=(N, N))
            if fill != 0:
                mm[...] = fill
            mm.flush()
            del mm
    return d


def open_fine(root: str | Path, name: str, face: int, mode: str = "r") -> np.ndarray:
    """Memmap of one fine face (``mode`` "r" or "r+")."""
    return np.load(fine_path(root, name, face), mmap_mode=mode)


def fine_exists(root: str | Path, params: WorldParams | None = None) -> bool:
    for name in FINE_FIELDS:
        for f in range(6):
            p = fine_path(root, name, f)
            if not p.exists():
                return False
            if params is not None:
                a = np.load(p, mmap_mode="r")
                if a.shape != (params.N_fine, params.N_fine):
                    return False
    return True


# --------------------------------------------------------------------------
# base pass
# --------------------------------------------------------------------------
def write_base_face(root: str | Path, params: WorldParams, face: int, fields: dict[str, FaceField], derived: dict[str, FaceField], strip: int = 64) -> int:
    """Plain upsample of a whole face into the memmaps, ``strip`` coarse
    rows at a time.  Returns the number of cells written (``N_fine²``)."""
    N, R = params.N_c, params.world.R
    mm = {name: open_fine(root, name, face, "r+") for name in FINE_FIELDS}
    for i0 in range(0, N, strip):
        i1 = min(N, i0 + strip)
        up = upsample_face(fields, derived, face, R, i0, i1)
        sl = slice(i0 * R, i1 * R)
        for name in FINE_FIELDS:
            mm[name][sl, :] = up[name]
    for m in mm.values():
        m.flush()
    return N * N * R * R


# --------------------------------------------------------------------------
# basin results
# --------------------------------------------------------------------------
def feather_weight(own: np.ndarray, feather_cells: int) -> np.ndarray:
    """Blend weight of the refined result on the basin's own cells:
    ``smoothstep(clip((d - 1) / feather_cells, 0, 1))``, ``d`` = chessboard
    distance to the nearest cell outside ``own`` or outside the array (so
    the divide ring has weight 0 and cells ``feather_cells`` further in
    weight 1, with zero slope at both ends).  Float32, 0 outside ``own``."""
    own = np.asarray(own, dtype=bool)
    padded = np.pad(own, 1, constant_values=False)
    d = ndimage.distance_transform_cdt(padded, metric="chessboard")[1:-1, 1:-1].astype(np.float32)
    if feather_cells <= 0:
        w = (d > 1).astype(np.float32)
    else:
        t = np.clip((d - 1.0) / float(feather_cells), 0.0, 1.0)
        w = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
    w[~own] = 0.0
    return w


def window_face_slices(win: Window, N_fine: int) -> tuple[slice, slice, slice, slice] | None:
    """Intersection of the window's fine cells with the face: ``(face_i,
    face_j, local_i, local_j)`` slices, or None if disjoint."""
    n = win.n
    fi0, fj0 = win.fi0, win.fj0
    a0, a1 = max(fi0, 0), min(fi0 + n, N_fine)
    b0, b1 = max(fj0, 0), min(fj0 + n, N_fine)
    if a0 >= a1 or b0 >= b1:
        return None
    return slice(a0, a1), slice(b0, b1), slice(a0 - fi0, a1 - fi0), slice(b0 - fj0, b1 - fj0)


def blend_result(arrays: dict[str, np.ndarray], bid: int, feather_cells: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Feathered output arrays of a job over its window and the ``own``
    mask (cells whose fine basin id is ``bid``).  Pure (no I/O) so tests
    can check the blend."""
    own = arrays["basin_id"] == int(bid)
    w = feather_weight(own, feather_cells)
    out = {}
    for name, ref in FEATHERED:
        a = arrays[name].astype(np.float32)
        b = arrays[ref].astype(np.float32)
        out[name] = (w * a + (1.0 - w) * b).astype(np.float32)
    out["hardness"] = arrays["hardness"].astype(np.float32)
    out["basin_id"] = arrays["basin_id"].astype(np.int32)
    return out, own


def write_result(root: str | Path, params: WorldParams, res) -> int:
    """Write a :class:`~globe.refine.basin_job.BasinResult` into the
    memmaps (see module docstring).  Returns the number of cells written."""
    win = res.win
    N_fine = params.N_fine
    sl = window_face_slices(win, N_fine)
    if sl is None:
        return 0
    fi, fj, li, lj = sl
    out, own = blend_result(res.arrays, res.id, int(params.refine.feather_cells))
    own_l = own[li, lj]
    n = int(own_l.sum())
    if n == 0:
        return 0
    for name in FINE_FIELDS:
        mm = open_fine(root, name, win.face, "r+")
        view = mm[fi, fj]
        view[own_l] = out[name][li, lj][own_l]
        mm.flush()
        del mm
    return n


__all__ = ["FINE_FIELDS", "FINE_FILL", "FEATHERED", "fine_dir", "fine_path", "create_fine", "open_fine", "fine_exists", "write_base_face", "feather_weight", "window_face_slices", "blend_result", "write_result"]
