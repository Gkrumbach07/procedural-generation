"""A planet level's finish in bounded memory (docs/zoom-windows.md, "The
planet at 305 m"): the same outputs as the in-memory finish of the 1.2 km
level -- height, sediment and discharge from the work raster, seams blended,
refine's coast rule, a water surface capped by the planet's lakes -- for a
face of any size.  At R = 32 a face is 32768^2 cells, 4.3 GB a float32
field, and a priority flood of it would need ~50 GB:

* the outputs are memmaps written a strip of coarse rows at a time;
* the seams read the neighbour face's work raster through its memmap;
* the water surface is a flood of the whole face while it is at most
  ``FLOOD_WHOLE`` cells a side (so the 1.2 km level is byte-identical to
  the in-memory finish), else of ``FLOOD_BLOCK`` blocks overlapping by
  ``FLOOD_OVERLAP``, each draining to its border as well: a depression
  wider than the overlap that crosses a block edge is drained there, which
  the cap by the planet's own lakes (the big ones) covers.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy import ndimage

from ..config import WorldParams
from ..hydro.priority_flood import priority_flood_flat
from ..io.world_store import WorldStore
from ..refine.zoom import zoom_params

FLOOD_WHOLE = 8192
FLOOD_BLOCK = 8192
FLOOD_OVERLAP = 512


def strip_rows(R: int) -> int:
    """Coarse rows per strip: ~512 fine rows (an upsample of a strip holds a
    dozen fields of it)."""
    return max(1, 512 // int(R))


def _flood_ws(surf: np.ndarray, ocean: np.ndarray) -> np.ndarray:
    drain = ocean.copy()
    drain[0, :] = drain[-1, :] = drain[:, 0] = drain[:, -1] = True
    fr = priority_flood_flat(surf, drain, None)
    return np.maximum(fr.filled.reshape(surf.shape), surf)


def finish_face(root: Path, out: Path, level, face: int) -> dict:
    from ..refine import rasterize as rz
    from ..refine.upsample import upsample_face
    from . import planet as zp

    t0 = time.time()
    root, out = Path(root), Path(out)
    params = WorldParams.from_dict(WorldStore(root).manifest["params"])
    lp = zoom_params(params, level.R)
    grid, fields, derived = zp.planet_inputs(root, lp, out)[:3]
    N, R = grid.N, level.R
    n = N * R
    g = level.guard * R
    NF = n + 2 * g
    strip = strip_rows(R)
    wk = {k: zp.open_work(out, face, k, NF, mode="r") for k in zp.WORK_FIELDS}
    arrays = {k: np.lib.format.open_memmap(zp.out_path(out, R, face, k), mode="w+", dtype=np.float32, shape=(n, n)) for k in zp.OUT_FIELDS}
    plain_ws_path = out / f"L{R}.f{face}.plain_ws.tmp.npy"
    plain_ws = np.lib.format.open_memmap(plain_ws_path, mode="w+", dtype=np.float32, shape=(n, n))
    for i0 in range(0, N, strip):
        i1 = min(N, i0 + strip)
        up = upsample_face(fields, derived, face, R, i0, i1)
        rs = slice(i0 * R, i1 * R)
        wr = slice(g + i0 * R, g + i1 * R)
        done = np.asarray(wk["done"][wr, g:g + n])
        for k in ("height", "sediment", "discharge"):
            arrays[k][rs] = np.where(done, wk[k][wr, g:g + n], up[k])
        plain_ws[rs] = up["water_surface"]
        del up, done
    seams = zp.blend_face_seams(out, level, N, face, arrays)
    ocean_all = np.stack([np.load(root / "coarse" / f"flow_dir.f{k}.npy") == 255 for k in range(6)])
    frac = rz.coast_fraction(grid, ocean_all)
    zone = rz.coast_zone(grid, ocean_all)
    M = np.float32(rz.COAST_MARGIN_M)
    coast_lowered = coast_raised = 0
    for i0 in range(0, N, strip):
        i1 = min(N, i0 + strip)
        near = np.repeat(np.repeat(zone[face, i0:i1], R, axis=0), R, axis=1)
        if not near.any():
            continue
        rs = slice(i0 * R, i1 * R)
        sea = (frac.sample_window(face, i0, i1, 0, N, R, order=1) > 0.5) & near
        h, sd = np.array(arrays["height"][rs]), np.array(arrays["sediment"][rs])
        sf = h + sd
        drop = np.where(sea & (sf > -M), sf + M, np.float32(0.0)).astype(np.float32)
        take = np.minimum(np.maximum(sd, 0.0), drop)
        sd -= take
        h -= drop - take
        low = near & ~sea & (sf < M)
        sd[low] += M - sf[low]
        arrays["height"][rs] = h
        arrays["sediment"][rs] = sd
        coast_lowered += int((drop > 0).sum())
        coast_raised += int(low.sum())
    sea_near_c = ndimage.binary_dilation(ocean_all[face], structure=np.ones((3, 3), bool))

    def rows(a0, a1, b0, b1):
        surf = np.asarray(arrays["height"][a0:a1, b0:b1]) + np.asarray(arrays["sediment"][a0:a1, b0:b1])
        ci0, cj0 = a0 // R, b0 // R
        near = np.repeat(np.repeat(sea_near_c[ci0:(a1 + R - 1) // R, cj0:(b1 + R - 1) // R], R, 0), R, 1)
        near = near[a0 - ci0 * R:a1 - ci0 * R, b0 - cj0 * R:b1 - cj0 * R]
        return surf, (surf < 0.0) & near

    ws_out = arrays["water_surface"]
    lake_cells = land_cells = 0
    if n <= FLOOD_WHOLE:
        surf, ocean = rows(0, n, 0, n)
        ws = _flood_ws(surf, ocean)
        pws = np.asarray(plain_ws)
        lake_c = pws > surf + 1e-3
        ws = np.where(lake_c, np.maximum(surf, np.minimum(ws, pws)), ws)
        ws = np.where(ocean, 0.0, ws)
        ws_out[:] = ws.astype(np.float32)
        lake_cells = int(((ws - surf > float(params.hydro.lake_min_depth)) & ~ocean).sum())
        land_cells = int((~ocean).sum())
        del surf, ocean, ws, pws, lake_c
    else:
        B, O = FLOOD_BLOCK, FLOOD_OVERLAP
        for a0 in range(0, n, B):
            for b0 in range(0, n, B):
                a1, b1 = min(n, a0 + B), min(n, b0 + B)
                e0, e1 = max(0, a0 - O), min(n, a1 + O)
                f0, f1 = max(0, b0 - O), min(n, b1 + O)
                surf, ocean = rows(e0, e1, f0, f1)
                ws = _flood_ws(surf, ocean)
                core = (slice(a0 - e0, a1 - e0), slice(b0 - f0, b1 - f0))
                s, oc, w = surf[core], ocean[core], ws[core]
                pws = np.asarray(plain_ws[a0:a1, b0:b1])
                w = np.where(pws > s + 1e-3, np.maximum(s, np.minimum(w, pws)), w)
                w = np.where(oc, 0.0, w)
                ws_out[a0:a1, b0:b1] = w.astype(np.float32)
                lake_cells += int(((w - s > float(params.hydro.lake_min_depth)) & ~oc).sum())
                land_cells += int((~oc).sum())
                del surf, ocean, ws
    for a in arrays.values():
        a.flush()
    del arrays, plain_ws
    plain_ws_path.unlink(missing_ok=True)
    return {"face": face, "seam_cells": seams, "coast_lowered_cells": coast_lowered, "coast_raised_cells": coast_raised,
            "land_cells": land_cells, "lake_cells": lake_cells, "seconds": round(time.time() - t0, 1)}


def quicklook_rows(out: Path, R: int, face: int, name: str, k: int, m: int, how: str = "mean") -> np.ndarray:
    """``name`` of a face reduced by ``k x k`` blocks, a block row at a time."""
    from . import planet as zp

    a = np.load(zp.out_path(out, R, face, name), mmap_mode="r")
    res = np.empty((m // k, m // k), np.float32)
    for r in range(m // k):
        blk = np.asarray(a[r * k:(r + 1) * k, :m], np.float32).reshape(k, m // k, k)
        res[r] = blk.max(axis=(0, 2)) if how == "max" else blk.mean(axis=(0, 2))
    return res


__all__ = ["FLOOD_WHOLE", "FLOOD_BLOCK", "FLOOD_OVERLAP", "strip_rows", "finish_face", "quicklook_rows"]
