"""Rasterise basin results into the per-face fine memmaps (PLAN 10.2 step 5).

``fine/<name>.f{k}.npy`` (``N_fine x N_fine``, ``[i, j]``, no halo;
docs/DEVELOPING.md) for ``height, sediment, water_surface, discharge,
hardness`` (f32) and ``basin_id`` (i32) are created once by the driver
(:func:`create_fine`: zero-filled, ``basin_id`` -1) with
``np.lib.format.open_memmap`` so a face is never fully resident.

Three writers; the first two are usable from a worker process (every
process maps the same files; writes go to disjoint cells):

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
  crease at every divide).  ``own`` is taken over the whole window, off-face
  strip included, so where the basin continues across a cube edge the seam
  is not a divide and the weight there is that of the basin interior; where
  the basin ends at the edge (a real divide, the coast) the cells across it
  are not ``own`` and the seam row is the plain upsample as before.
* :func:`write_seams` — the seam blend, run by the driver once after every
  basin job (docs/cross-face-basins.md).  A piece of a basin on several
  faces also returns :func:`seam_records`: its own refined detail
  (``surface - plain``, ``sediment - plain sediment``) on its face's cells
  within ``feather_cells`` of each cube edge the basin crosses, and its
  refined off-face strip resampled bilinearly onto the neighbouring face's
  fine lattice (cell centre -> ``to_sphere`` -> ``project_to_face`` of the
  window face).  :func:`blend_seams` combines, per target cell, the pieces
  of the cell's own basin with complementary weights —
  :func:`seam_side_weight` ``smoothstep((dist + F) / 2F)`` for the piece
  whose face it is and ``1 -`` that for the piece across the edge (times a
  ramp to that piece's window border), ``dist`` the cell centre's distance
  from the edge in cells — normalised to sum 1, so the seam row carries
  both pieces' detail at 1/2 each and the combined detail is continuous
  through the edge.  The result is ``plain + w * detail`` with ``w`` the
  owning piece's divide feather, so divides stay pinned.  ``basin_id``,
  ``hardness`` and ``discharge`` stay the owning piece's (the discharge
  field is a river position, not a quantity to average, and keeps the
  face-restricted feather: the seam row is the coarse-initialised
  discharge on both faces, as before); the water surface
  keeps the owning piece's level on its lake cells (``max(level,
  surface)``, still under refine's lake-balance cap) and is the blended
  surface on dry cells.  The records are sorted by ``(face, basin, kind,
  source face)`` and summed in that order, so the raster does not depend
  on which piece finished first.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from ..config import WorldParams
from ..cubesphere import project_to_face_v, to_sphere_v
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


def blend_result(arrays: dict[str, np.ndarray], bid: int, feather_cells: int, on_face: np.ndarray | None = None) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Feathered output arrays of a job over its window and the ``own``
    mask (cells whose fine basin id is ``bid``, on the face or beyond its
    edge).  The feather weight of the surface and water surface is computed
    on ``own`` over the whole window, so a cube edge the basin continues
    across is not a divide (the pieces on the two faces are reconciled
    there by :func:`blend_seams`) and an edge it ends at is.  The
    *discharge* is not feathered at divides or the coast: a river is a
    position, and blending a refined channel into the upsampled coarse
    discharge drew a sharp channel beside a blurred copy of itself and ended
    rivers short of the sea while the coarse blob reached it
    (docs/viewer-rivers.md).  With ``on_face`` (bool, the window cells on the
    job's face) it is feathered towards the cube edge only: a river position
    cannot be averaged between two pieces either, so at the seam both faces
    fall back to the coarse-initialised discharge, which is what carries a
    river across the edge.  Pure (no I/O) so tests can check the blend."""
    own = arrays["basin_id"] == int(bid)
    w = feather_weight(own, feather_cells)
    wq = own.astype(np.float32) if on_face is None else feather_weight(on_face, feather_cells) * own
    out = {}
    for name, ref in FEATHERED:
        a = arrays[name].astype(np.float32)
        b = arrays[ref].astype(np.float32)
        ww = wq if name == "discharge" else w
        out[name] = (ww * a + (1.0 - ww) * b).astype(np.float32)
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
    on_face = np.zeros(res.arrays["basin_id"].shape, dtype=bool)
    on_face[li, lj] = True
    out, own = blend_result(res.arrays, res.id, int(params.refine.feather_cells), on_face)
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


# --------------------------------------------------------------------------
# the coast
# --------------------------------------------------------------------------
#: the coast pass keeps a fine sea cell at least this far below sea level and
#: coastal land this far above it (m)
COAST_MARGIN_M = 1.0


def coast_fraction(grid, ocean_c: np.ndarray, passes: int = 2) -> FaceField:
    """The coarse ocean mask after ``passes`` 3x3 binomial passes across
    face edges, every cell centre held on its own side of 0.5 (by 0.1): its
    0.5 contour, interpolated, is a coastline that rounds the coarse cells'
    corners instead of tracing them (the viewer's ``smooth_mask``)."""
    m = np.asarray(ocean_c, bool)
    ff = FaceField.from_interior(grid, m.astype(np.float32), name="ocean", exchange=True)
    H, N = grid.H, grid.N
    w = (1.0, 2.0, 1.0)
    for _ in range(passes):
        d = ff.data
        out = np.zeros((6, N, N), np.float32)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                out += (w[di + 1] * w[dj + 1] / 16.0) * d[:, H + di:H + di + N, H + dj:H + dj + N]
        ff.data[:, H:H + N, H:H + N] = out
        ff.exchange_halos()
    inter = ff.data[:, H:H + N, H:H + N]
    inter[:] = np.where(m, np.maximum(inter, 0.6), np.minimum(inter, 0.4))
    ff.exchange_halos()
    return ff


def coast_zone(grid, ocean_c: np.ndarray) -> np.ndarray:
    """Coarse ``(6, N, N)`` cells the coast pass may change: ocean, or
    8-adjacent to it (across face edges)."""
    m = np.asarray(ocean_c, bool)
    ff = FaceField.from_interior(grid, m.astype(np.uint8), name="near", exchange=True)
    H, N = grid.H, grid.N
    out = np.empty_like(m)
    for f in range(6):
        out[f] = ndimage.binary_dilation(ff.data[f].astype(bool), structure=np.ones((3, 3), bool))[H:H + N, H:H + N]
    return out


def write_coast(root: str | Path, params: WorldParams, ocean_c: np.ndarray, strip: int = 64) -> dict:
    """Make the fine surface agree with a smooth coastline, after every other
    writer.  The upsample overshoots above sea level where a shallow shelf
    (erosion fills it to a few metres below the water) meets high ground --
    45k fine sea cells up to 245 m on earth-v9, drawn as specks of land
    offshore -- and the basin rasters' coastline is the coarse cells'
    outline.  In the coarse cells that are ocean or touch it: a cell inside
    the 0.5 contour of :func:`coast_fraction` (bilinear) is sea and stands at
    least ``COAST_MARGIN_M`` below 0 (sediment removed first), with no water
    surface above 0; any other cell there stands at least that far above 0,
    and a former sea cell among them takes the largest discharge of the
    refined land within two cells (not the sea's upsampled coarse value).
    Returns counts."""
    grid = params.coarse_grid()
    N, R = grid.N, int(params.world.R)
    ocean_c = np.asarray(ocean_c, bool)
    frac = coast_fraction(grid, ocean_c)
    near_c = coast_zone(grid, ocean_c)
    M = np.float32(COAST_MARGIN_M)
    lowered = raised = 0
    for f in range(6):
        mm = {name: open_fine(root, name, f, "r+") for name in ("height", "sediment", "water_surface", "discharge", "basin_id")}
        for i0 in range(0, N, strip):
            i1 = min(N, i0 + strip)
            near = np.repeat(np.repeat(near_c[f, i0:i1], R, axis=0), R, axis=1)
            if not near.any():
                continue
            sl = slice(i0 * R, i1 * R)
            sea = (frac.sample_window(f, i0, i1, 0, N, R, order=1) > 0.5) & near
            h = np.array(mm["height"][sl], dtype=np.float32)
            sd = np.array(mm["sediment"][sl], dtype=np.float32)
            ws = np.array(mm["water_surface"][sl], dtype=np.float32)
            surf = h + sd
            drop = np.where(sea & (surf > -M), surf + M, np.float32(0.0)).astype(np.float32)
            take = np.minimum(np.maximum(sd, 0.0), drop)
            sd -= take
            h -= drop - take
            ws[sea] = np.minimum(ws[sea], 0.0)
            low = near & ~sea & (surf < M)
            sd[low] += M - surf[low]
            land = near & ~sea                     # a sea cell the contour gives to land keeps no sea surface under it
            ws[land] = np.maximum(ws[land], h[land] + sd[land])
            # sea cells the contour gives to land have the upsampled coarse
            # discharge of the sea (a blue strip along the coast): they take
            # the refined land's next to them instead, so a river reaches the
            # new shore and nothing else shows
            q = np.array(mm["discharge"][sl], dtype=np.float32)
            was_sea = land & (np.asarray(mm["basin_id"][sl]) < 0)
            if was_sea.any():
                src = np.where(np.asarray(mm["basin_id"][sl]) >= 0, q, np.float32(0.0))
                q[was_sea] = ndimage.maximum_filter(src, size=5)[was_sea]
            lowered += int((drop > 0).sum())
            raised += int(low.sum())
            mm["height"][sl] = h
            mm["sediment"][sl] = sd
            mm["water_surface"][sl] = ws
            mm["discharge"][sl] = q
        for m in mm.values():
            m.flush()
        del mm
    return {"coast_lowered_cells": lowered, "coast_raised_cells": raised}


# --------------------------------------------------------------------------
# seam blend across cube-face edges
# --------------------------------------------------------------------------
#: record kinds (the owning piece sorts first within a basin)
SEAM_NATIVE, SEAM_CROSS = 0, 1
#: a native water surface this far above its surface (m) is a lake level to keep
SEAM_LAKE_EPS_M = 1e-3


def seam_side_weight(dist, feather_cells: int) -> np.ndarray:
    """Weight of a piece's own refined detail at ``dist`` cells (cell centre
    distance from a cube edge into the piece's face; negative beyond the
    edge): ``smoothstep(clip((dist + F) / 2F, 0, 1))``.  The piece across
    the edge sees the same point at ``-dist`` and gets ``1 -`` this (the
    smoothstep is point-symmetric about 1/2), so the two sum to 1, both
    are 1/2 on the edge and the ramp has zero slope ``F`` cells out on
    either side.  ``F <= 0``: 1 on the own face, 0 beyond (float64)."""
    d = np.asarray(dist, dtype=np.float64)
    F = float(feather_cells)
    if F <= 0.0:
        return (d > 0.0).astype(np.float64)
    return _smoothstep((d + F) / (2.0 * F))


def _smoothstep(t) -> np.ndarray:
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def side_cells(side: int, n: int, depth: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(i, j, k)`` of the cells of an ``n x n`` face within ``depth`` rows
    of ``side`` (``globe.refine.lod`` numbering: 0 = +i, 1 = -i, 2 = +j,
    3 = -j), ``k`` the row (0 = touching the edge)."""
    depth = max(0, min(int(depth), n))
    k = np.repeat(np.arange(depth, dtype=np.int64), n)
    a = np.tile(np.arange(n, dtype=np.int64), depth)
    if side == 0:
        return n - 1 - k, a, k
    if side == 1:
        return k, a, k
    if side == 2:
        return a, n - 1 - k, k
    return a, k, k


def _bilinear(a: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear sample (float64) of ``a`` at fractional indices ``x, y`` in
    ``[0, shape - 1]``."""
    x0 = np.clip(np.floor(x).astype(np.int64), 0, a.shape[0] - 2)
    y0 = np.clip(np.floor(y).astype(np.int64), 0, a.shape[1] - 2)
    tx, ty = x - x0, y - y0
    a = np.asarray(a, dtype=np.float64)
    return ((1.0 - tx) * (1.0 - ty) * a[x0, y0] + tx * (1.0 - ty) * a[x0 + 1, y0]
            + (1.0 - tx) * ty * a[x0, y0 + 1] + tx * ty * a[x0 + 1, y0 + 1])


def strip_window_coords(win: Window, face: int, i: np.ndarray, j: np.ndarray, N_fine: int) -> tuple[np.ndarray, np.ndarray]:
    """Fractional window indices ``(x, y)`` (into the window's ``n x n``
    arrays, cell centres at integers) of the fine cells ``(i, j)`` of
    ``face`` -- a face other than the window's: each cell centre goes to
    the sphere and is projected onto the window's face plane.  The one
    place the seam blend's resampling geometry lives, so a test can check
    it against the neighbouring face's own upsample."""
    p = to_sphere_v(np.full(np.shape(i), int(face)), (np.asarray(i) + 0.5) / N_fine, (np.asarray(j) + 0.5) / N_fine)
    u, v = project_to_face_v(np.full(np.shape(i), int(win.face)), p)
    return u * N_fine - 0.5 - win.fi0, v * N_fine - 0.5 - win.fj0


def seam_records(res, params: WorldParams, piece_faces, basin_id_coarse: np.ndarray) -> list[dict]:
    """The seam-blend inputs of one basin piece (module docstring); empty
    unless the basin has a piece on a face across one of this face's
    edges.  ``piece_faces``: the faces the basin has cells on;
    ``basin_id_coarse``: the coarse ``basin_id`` interior ``(6, N, N)``
    (picks this basin's cells on the neighbouring faces).  Records are dicts
    of plain arrays (picklable; at most ``feather_cells`` x ``N_fine`` cells
    per crossed edge):

    * ``kind`` :data:`SEAM_NATIVE` — the basin's cells on the piece's face
      within ``feather_cells`` of an edge to a face the basin has a piece
      on: ``idx`` (flat fine index), ``dsurf``, ``dsed`` (refined minus
      plain), ``beta`` (:func:`seam_side_weight`; near a cube corner the
      smaller of the two edges'), ``w`` (the divide feather), ``surf0`` /
      ``sed0`` (plain surface / sediment) and ``surf`` / ``ws`` (the surface
      and water surface :func:`write_result` wrote there);
    * ``kind`` :data:`SEAM_CROSS` — per crossed edge, the neighbouring
      face's cells of the basin within ``feather_cells`` of the edge whose
      centres project into the window: ``idx``, ``dsurf``, ``dsed``
      (bilinear from the window) and ``beta`` = ``1 - seam_side_weight``
      times a smoothstep ramp over ``feather_cells`` to the window border
      (a window ending along the edge fades out instead of switching off)."""
    from .lod import edge_links

    win = res.win
    f, bid, R = int(win.face), int(res.id), int(win.R)
    N_fine = int(params.N_fine)
    F = int(params.refine.feather_cells)
    others = {int(g) for g in piece_faces} - {f}
    if F <= 0 or not others:
        return []
    links = edge_links(N_fine)
    sides = [s for s in range(4) if links[f * 4 + s].nb_face in others]
    if not sides:
        return []
    a = res.arrays
    n = win.n
    out, own = blend_result(a, bid, F)
    w = feather_weight(own, F)
    surf0 = a["height0"].astype(np.float64) + a["sediment0"]
    dsurf = a["height"].astype(np.float64) + a["sediment"] - surf0
    dsed = a["sediment"].astype(np.float64) - a["sediment0"]
    recs: list[dict] = []

    # the piece's own face
    idx_l, beta_l = [], []
    for s in sides:
        i, j, k = side_cells(s, N_fine, F)
        li, lj = i - win.fi0, j - win.fj0
        ok = (li >= 0) & (li < n) & (lj >= 0) & (lj < n)
        ok[ok] = own[li[ok], lj[ok]]
        idx_l.append(i[ok] * N_fine + j[ok])
        beta_l.append(seam_side_weight(k[ok] + 0.5, F))
    idx = np.concatenate(idx_l)
    if idx.size:
        beta = np.concatenate(beta_l)
        order = np.lexsort((beta, idx))  # a cell in two strips (cube corner) keeps the smaller weight
        idx, first = np.unique(idx[order], return_index=True)
        beta = beta[order][first]
        li, lj = idx // N_fine - win.fi0, idx % N_fine - win.fj0
        recs.append({
            "kind": SEAM_NATIVE, "face": f, "src": f, "bid": bid, "idx": idx,
            "dsurf": dsurf[li, lj].astype(np.float32), "dsed": dsed[li, lj].astype(np.float32),
            "beta": beta.astype(np.float32), "w": w[li, lj],
            "surf0": surf0[li, lj].astype(np.float32), "sed0": a["sediment0"][li, lj].astype(np.float32),
            "surf": (out["height"][li, lj] + out["sediment"][li, lj]).astype(np.float32),
            "ws": out["water_surface"][li, lj],
        })

    # the refined off-face strip, resampled onto each neighbouring face's lattice
    for s in sides:
        ln = links[f * 4 + s]
        g = int(ln.nb_face)
        i, j, k = side_cells(ln.nb_side, N_fine, F)
        ok = basin_id_coarse[g, i // R, j // R] == bid
        i, j, k = i[ok], j[ok], k[ok]
        if i.size == 0:
            continue
        x, y = strip_window_coords(win, g, i, j, N_fine)
        border = np.minimum(np.minimum(x + 0.5, n - 0.5 - x), np.minimum(y + 0.5, n - 0.5 - y))
        beta = (1.0 - seam_side_weight(k + 0.5, F)) * _smoothstep(border / F)
        ok = (x >= 0.0) & (x <= n - 1) & (y >= 0.0) & (y <= n - 1) & (beta > 0.0)
        if not ok.any():
            continue
        x, y = x[ok], y[ok]
        recs.append({
            "kind": SEAM_CROSS, "face": g, "src": f, "bid": bid, "idx": i[ok] * N_fine + j[ok],
            "dsurf": _bilinear(dsurf, x, y).astype(np.float32), "dsed": _bilinear(dsed, x, y).astype(np.float32),
            "beta": beta[ok].astype(np.float32),
        })
    return recs


def blend_seams(records: list[dict], N_fine: int) -> dict[int, dict[str, np.ndarray]]:
    """Combine the :func:`seam_records` of all pieces (module docstring).
    On each target face the cells that have the owning piece's record, at
    least one cross record and a nonzero divide feather ``w`` get ``surface
    = plain + w * sum(beta dsurf) / sum(beta)``, ``sediment = max(plain
    sediment + w * sum(beta dsed) / sum(beta), 0)``, ``height = surface -
    sediment`` and the water surface of the module docstring.  Returns
    ``{face: {"idx", "height", "sediment", "water_surface"}}`` (float32,
    ``idx`` ascending).  Pure; the result does not depend on the order of
    ``records`` (they are sorted before anything is summed)."""
    recs = sorted(records, key=lambda r: (int(r["face"]), int(r["bid"]), int(r["kind"]), int(r["src"])))
    result: dict[int, dict[str, np.ndarray]] = {}
    for g in sorted({int(r["face"]) for r in recs}):
        rg = [r for r in recs if int(r["face"]) == g]
        nat = [r for r in rg if r["kind"] == SEAM_NATIVE]
        if not nat or len(nat) == len(rg):
            continue
        idx = np.concatenate([r["idx"] for r in rg]).astype(np.int64)
        cells, inv = np.unique(idx, return_inverse=True)
        m = cells.size
        beta = np.concatenate([r["beta"] for r in rg]).astype(np.float64)
        cross = np.concatenate([np.full(r["idx"].size, r["kind"] == SEAM_CROSS) for r in rg])
        den = np.bincount(inv, weights=beta, minlength=m)
        den_x = np.bincount(inv, weights=np.where(cross, beta, 0.0), minlength=m)
        num_s = np.bincount(inv, weights=beta * np.concatenate([r["dsurf"] for r in rg]), minlength=m)
        num_d = np.bincount(inv, weights=beta * np.concatenate([r["dsed"] for r in rg]), minlength=m)
        pos = np.searchsorted(cells, np.concatenate([r["idx"] for r in nat]).astype(np.int64))
        nv = {}
        for key in ("w", "surf0", "sed0", "surf", "ws"):
            nv[key] = np.zeros(m, dtype=np.float64)
            nv[key][pos] = np.concatenate([r[key] for r in nat])
        has_nat = np.zeros(m, dtype=bool)
        has_nat[pos] = True
        sel = has_nat & (den_x > 0.0) & (nv["w"] > 0.0)
        if not sel.any():
            continue
        w = nv["w"][sel]
        surf = (nv["surf0"][sel] + w * (num_s[sel] / den[sel])).astype(np.float32)
        sed = np.maximum(nv["sed0"][sel] + w * (num_d[sel] / den[sel]), 0.0).astype(np.float32)
        height = (surf - sed).astype(np.float32)
        surf = (height + sed).astype(np.float32)  # the surface as a reader of height + sediment sees it
        lake = (nv["ws"][sel] - nv["surf"][sel]) > SEAM_LAKE_EPS_M
        ws = np.where(lake, np.maximum(nv["ws"][sel].astype(np.float32), surf), surf).astype(np.float32)
        result[g] = {"idx": cells[sel], "height": height, "sediment": sed, "water_surface": ws}
    return result


def write_seams(root: str | Path, params: WorldParams, records: list[dict]) -> int:
    """Write :func:`blend_seams` into the memmaps (the driver, after every
    basin job has written).  Returns the number of cells rewritten."""
    N_fine = int(params.N_fine)
    total = 0
    for g, v in blend_seams(records, N_fine).items():
        i, j = np.divmod(v["idx"], N_fine)
        for name in ("height", "sediment", "water_surface"):
            mm = open_fine(root, name, g, "r+")
            mm[i, j] = v[name]
            mm.flush()
            del mm
        total += int(v["idx"].size)
    return total


__all__ = ["FINE_FIELDS", "FINE_FILL", "FEATHERED", "fine_dir", "fine_path", "create_fine", "open_fine", "fine_exists", "write_base_face", "feather_weight", "window_face_slices", "blend_result", "write_result",
           "SEAM_NATIVE", "SEAM_CROSS", "seam_side_weight", "side_cells", "strip_window_coords", "seam_records", "blend_seams", "write_seams"]
