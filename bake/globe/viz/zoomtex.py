"""A zoom level's texture for the globe viewer, which draws the level inline
at its own resolution.

One image per level of a listed zoom, ``viewer/zoomtex/{name}_L{R}.js``
(``GLOBE_VIEWER.zoomTex(name, R, "<base64 lossless RGBA WebP>")``:
``file://`` pages cannot fetch), with its scales in a sidecar
``{name}_L{R}.json`` that :func:`globe.zoom.index.scan` reads as the
level's ``tex`` record, so listing the zooms never opens an npz.

The image is the whole work array, ``NE`` pixels wide and ``2 NE`` tall;
pixel ``(x, y)`` is array cell ``(i, j) = (x, y)`` (row ``i`` along the
face's u axis), its fine face cell ``(ci0 R - p0 + i, cj0 R - p0 + j)``.
The top half is the ground as the detail tiles carry it
(:func:`detail.tile_image`): ``R, G`` the 16-bit code of the surface
(height + sediment) on ``(h0, h1)``, ``B`` the signed lake depth byte over
``lake_range``, ``A`` 255 - the smoothed ocean mask byte.  The bottom half is
the water: ``R`` the log byte of the level's flood-tree ``flux`` widened into
channels (:func:`detail.widen_rivers`), on a scale from the land of the
level's product (byte 1 at its median, 255 at its maximum; a river from
``river_min_byte`` to full ``river_span_byte`` above).
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np

LAKE_RANGE = 200.0
#: rivers from this percentile of the product's land flux, full at the second
RIVER_MIN_PCT, RIVER_FULL_PCT = 97.0, 99.9


def paths(viewer: Path, name: str, R: int) -> tuple[Path, Path]:
    """``(js, sidecar)`` of a level's texture under ``viewer/``."""
    d = Path(viewer) / "zoomtex"
    return d / f"{name}_L{int(R)}.js", d / f"{name}_L{int(R)}.json"


def _fresh(path: Path, src: Path) -> bool:
    try:
        return path.stat().st_mtime_ns >= src.stat().st_mtime_ns
    except OSError:
        return False


def record(viewer: Path, name: str, R: int, npz: Path) -> dict | None:
    """The level's ``tex`` record from its sidecar, or None when there is no
    texture or it is older than the level's npz (cheap: two stats and a small
    read)."""
    js, side = paths(viewer, name, R)
    if not (_fresh(js, npz) and _fresh(side, npz)):
        return None
    try:
        return json.loads(side.read_text())
    except (OSError, ValueError):
        return None


def river_scale(surf: np.ndarray, ocean: np.ndarray, flux: np.ndarray) -> dict:
    """The river byte scale from the land (``surf > 0``, not ocean, flux > 0)
    of the arrays given (the level's product): ``lo`` the median, ``hi`` the
    maximum, rivers from ``river_min`` to full at ``river_full``
    (:func:`detail.river_scale`'s guards)."""
    q = np.asarray(flux, np.float64)
    q = q[(np.asarray(surf) > 0.0) & ~np.asarray(ocean, bool) & (q > 0.0)]
    if not q.size:
        q = np.array([1.0])
    lo = max(float(np.percentile(q, 50)), 1e-9)
    hi = max(float(q.max()), lo * 10.0)
    rmin = max(float(np.percentile(q, RIVER_MIN_PCT)), lo * 1.001)
    return {"lo": lo, "hi": hi, "river_min": rmin, "river_full": max(float(np.percentile(q, RIVER_FULL_PCT)), rmin * 1.001)}


def level_image(a: dict, geo: dict, lake_range: float = LAKE_RANGE) -> tuple[np.ndarray, dict]:
    """The ``(2 NE, NE, 4)`` RGBA image of a level's work arrays ``a`` (as
    :func:`bake.load_level` or the npz holds them) and its scales ``{h0, h1,
    lake_range, river_min_byte, river_span_byte}``."""
    from . import detail as dt

    R, g = int(geo["R"]), int(geo["guard"])
    NE, p0, n = (int(geo["cells"]) + 2 * g + 2) * R, (g + 1) * R, int(geo["cells"]) * R
    surf = np.asarray(a["height"], np.float32) + np.asarray(a["sediment"], np.float32)
    if surf.shape != (NE, NE):
        raise ValueError(f"level arrays {surf.shape} do not match the geometry's {NE}^2")
    ocean = np.asarray(a["ocean"]) > 0
    ws = np.maximum(np.asarray(a["water_surface"], np.float32), surf)
    water = np.zeros((NE, NE), np.uint8)
    water[(ws - surf > 0.5) & ~ocean] = dt.WATER_LAKE
    water[ocean] = dt.WATER_OCEAN

    # the ground: as detail.tile_image
    h0, h1 = dt.height_grid(float(surf.min()), float(surf.max()))
    h = dt.encode_height_on(surf, h0, h1)
    ld = dt.face_lake_depth(surf, ws, water, lake_range)
    b = np.clip(np.round(255.0 * (ld + lake_range) / (2.0 * lake_range)), 0, 255).astype(np.uint8)
    om = dt.face_smooth_mask(water == dt.WATER_OCEAN)
    alpha = (255 - np.clip(np.round(255.0 * om), 0, 255)).astype(np.uint8)
    ground = np.stack([(h >> 8).astype(np.uint8), (h & 255).astype(np.uint8), b, alpha], axis=-1)
    del h, ld, om, b, alpha

    # the water: the flood-tree flux, widened, on the product's land scale
    flux = np.asarray(a["flux"], np.float32)
    c = slice(p0, p0 + n)
    scale = river_scale(surf[c, c], ocean[c, c], flux[c, c])
    q_lo, q_hi = scale["lo"], scale["hi"]
    q = np.asarray(dt.widen_rivers(flux, scale), np.float64)
    t = np.log(np.maximum(q, q_lo) / q_lo) / math.log(max(q_hi / q_lo, 1.0000001))
    qb = np.where(q > q_lo, np.clip(np.round(255.0 * t), 1, 255), 0).astype(np.uint8)
    del q, t
    zero = np.zeros_like(qb)
    wimg = np.stack([qb, zero, zero, np.full_like(qb, 255)], axis=-1)
    img = np.ascontiguousarray(np.concatenate([ground, wimg], axis=1).transpose(1, 0, 2))   # cell (i, j): ground at (x, y) = (i, j), water at (i, NE + j)

    byte = lambda v: int(round(255.0 * math.log(max(v, q_lo) / q_lo) / math.log(q_hi / q_lo)))
    scales = {"h0": h0, "h1": h1, "lake_range": float(lake_range),
              "river_min_byte": int(np.clip(byte(scale["river_min"]), 1, 254)),
              "river_span_byte": max(byte(scale["river_full"]) - byte(scale["river_min"]), 8)}
    return img, scales


def _replace(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_level(viewer: Path, name: str, zdir: Path, R: int, N: int, cell_m: float | None = None, geometry: dict | None = None,
                force: bool = False, log=None) -> dict | None:
    """Write level ``R`` of the zoom at ``zdir`` (listed as ``name``) as its
    texture and sidecar, unless both are newer than its npz (``force``
    rewrites).  Returns the ``tex`` record, None when the level has no npz."""
    from . import detail as dt

    npz = Path(zdir) / f"L{int(R)}.npz"
    if not npz.exists():
        return None
    if not force:
        rec = record(viewer, name, R, npz)
        if rec is not None:
            return rec
    t0 = time.time()
    stats_path = Path(zdir) / f"L{int(R)}.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    geo = stats.get("geometry") or geometry
    if geo is None:
        raise ValueError(f"no geometry for {npz}")
    cell_m = stats.get("cell_m", cell_m)
    keys = ("height", "sediment", "water_surface", "flux", "ocean")
    with np.load(npz) as z:
        a = {k: z[k] for k in keys}
    img, scales = level_image(a, geo)
    del a
    t1 = time.time()
    b64 = dt.webp_rgba_b64(img)
    js, side = paths(viewer, name, R)
    js.parent.mkdir(parents=True, exist_ok=True)
    g, Rg = int(geo["guard"]), int(geo["R"])
    rec = {"file": f"zoomtex/{js.name}", "face": int(geo["face"]), "N": int(N), "R": Rg, "ci0": int(geo["ci0"]), "cj0": int(geo["cj0"]),
           "cells": int(geo["cells"]), "guard": g, "NE": int(img.shape[1]), "p0": (g + 1) * Rg, "n": int(geo["cells"]) * Rg,
           **scales, "cell_m": None if cell_m is None else float(cell_m)}
    _replace(js, 'GLOBE_VIEWER.zoomTex("%s", %d, "%s");\n' % (name, Rg, b64))
    _replace(side, json.dumps(rec, indent=1))
    if log is not None:
        log(f"[zoomtex] {name} L{Rg}: {img.shape[1]}x{img.shape[0]}, {js.stat().st_size / 1e6:.2f} MB "
            f"(arrays {t1 - t0:.1f}s, webp {time.time() - t1:.1f}s)")
    return rec


__all__ = ["LAKE_RANGE", "RIVER_MIN_PCT", "RIVER_FULL_PCT", "paths", "record", "river_scale", "level_image", "write_level"]
