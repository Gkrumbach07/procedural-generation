"""A zoom level's texture for the globe viewer, which draws the level inline
at its own resolution.

One image per level of a listed zoom, ``viewer/zoomtex/{name}_L{R}.js``
(``GLOBE_VIEWER.zoomTex(name, R, "<base64 lossless RGBA WebP>")``:
``file://`` pages cannot fetch), with its scales in a sidecar
``{name}_L{R}.json`` that :func:`globe.zoom.index.scan` reads as the
level's ``tex`` record, so listing the zooms never opens an npz.

The image is the level's product (its eroded core) and :data:`CROP` cells
of the work array around it (fewer where the array has fewer), ``NE``
pixels wide and ``3 NE`` tall; pixel ``(x, y)`` is work-array cell ``(x0 +
x, x0 + y)`` (row along the face's u axis), its fine face cell ``(ci0 R -
p0 + x, cj0 R - p0 + y)``, the core pixels ``[p0, p0 + n)``.  ``ci0`` /
``cj0`` are the product's origin in coarse cells -- an integer for a level
placed in coarse cells, a dyadic fraction (``fi0 / R``, exact) for one
placed in fine cells -- and ``fi0`` / ``fj0`` the same in fine cells.
Until the crop the image was the whole work array (``x0`` 0, ``NE`` and
``p0`` the array's): at 5 m that is ~10,000 cells a side for a 2,048-cell
core.  WebGL textures stop at ``MAX_TEXTURE_SIZE`` (8192 here): a core over
:data:`MAX_TEX` / 3 is not written.

The top half is the ground as the detail tiles carry it
(:func:`detail.tile_image`): ``R, G`` the 16-bit code of the surface
(height + sediment) on ``(h0, h1)``, ``B`` the signed lake depth byte over
``lake_range``, ``A`` 255 - the smoothed ocean mask byte.  The bottom half is
the water: ``R`` the log byte of the level's flood-tree ``flux`` widened into
channels (:func:`detail.widen_rivers`), on a scale from the land of the
level's product (byte 1 at its median, 255 at its maximum; a river from
``river_min_byte`` to full ``river_span_byte`` above); ``G`` the flux itself on
that scale (``q_lo``, ``q_hi``: the flow layer); ``B`` the sediment log byte
over ``sed_lo`` .. ``sed_hi`` metres (version 3); ``A`` the canopy cover the
level grew (:mod:`globe.erosion.vegetation`), 0..255, where the record says
``veg`` (version 4; 255 on a level without one).  A third band below (version 5):
``R`` the level's own biome code (:mod:`globe.zoom.biomes`; classified from the
saved level when the bake did not), where the record says ``biome``.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np

LAKE_RANGE = 200.0
#: cells of the work array kept around the product (the viewer blends a level
#: in over its core's last 16 cells and reads a cell's neighbours)
CROP = 32
#: the tallest texture the viewer takes (3 NE <= MAX_TEX)
MAX_TEX = 8192
#: the record's format: 2 = cropped (``x0``, ``fi0``, ``array_NE``), 3 = the water half
#: also carries the flow (G) and sediment (B), 4 = and the canopy cover (A), 5 = a third
#: band of the level's biome codes; a
#: sidecar of another version is stale, so the next ``index.write`` crops it
TEX_VERSION = 5
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
    texture, it is older than the level's npz or of another
    :data:`TEX_VERSION` (cheap: two stats and a small read)."""
    js, side = paths(viewer, name, R)
    if not (_fresh(js, npz) and _fresh(side, npz)):
        return None
    try:
        rec = json.loads(side.read_text())
    except (OSError, ValueError):
        return None
    return rec if rec.get("version") == TEX_VERSION else None


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


def crop_window(NE: int, p0: int, n: int, crop: int = CROP, max_tex: int = MAX_TEX) -> tuple[int, int]:
    """``(x0, side)``: the square of a work array (``NE`` cells, product
    ``[p0, p0 + n)``) a level's texture covers -- the product and up to
    ``crop`` cells around it, as many as the array has on its narrower side
    and as ``3 side <= max_tex`` allows.  ValueError when the product alone
    is too tall."""
    if 3 * n > max_tex:
        raise ValueError(f"a {n}-cell core makes a texture taller than {max_tex}")
    m = max(0, min(int(crop), p0, NE - p0 - n, (max_tex // 3 - n) // 2))
    return p0 - m, n + 2 * m


def level_image(a: dict, geo: dict, lake_range: float = LAKE_RANGE, crop: int = CROP) -> tuple[np.ndarray, dict]:
    """The ``(3 NE, NE, 4)`` RGBA image of a level's work arrays ``a`` (as
    :func:`bake.load_level` or the npz holds them) over :func:`crop_window`,
    and its scales ``{h0, h1, lake_range, river_min_byte, river_span_byte}``
    and window ``{x0, NE, p0, n}`` (``NE`` the image's side, ``p0`` its core's
    first pixel, ``x0`` the work-array cell of its pixel 0).  Every byte is
    what the whole array would give there (the filters see a few cells past
    the crop), except the height codes, whose range is the crop's own."""
    from ..zoom.bake import Geometry
    from . import detail as dt

    g = Geometry(**geo)
    NEw, p0w, n = g.NE, g.p0, g.n
    shape = np.shape(a["height"])
    if tuple(shape) != (NEw, NEw):
        raise ValueError(f"level arrays {shape} do not match the geometry's {NEw}^2")
    x0, NE = crop_window(NEw, p0w, n, crop)
    p0 = p0w - x0
    ctx = dt.RIVER_RADIUS + 3                      # widest stencil below: the river discs
    e0, e1 = max(x0 - ctx, 0), min(x0 + NE + ctx, NEw)
    k = slice(x0 - e0, x0 - e0 + NE)               # the crop within the context slice
    ex = slice(e0, e1)
    surf = np.asarray(a["height"][ex, ex], np.float32) + np.asarray(a["sediment"][ex, ex], np.float32)
    ocean = np.asarray(a["ocean"][ex, ex]) > 0
    ws = np.maximum(np.asarray(a["water_surface"][ex, ex], np.float32), surf)
    water = np.zeros(surf.shape, np.uint8)
    water[(ws - surf > 0.5) & ~ocean] = dt.WATER_LAKE
    water[ocean] = dt.WATER_OCEAN

    # the ground: as detail.tile_image -- a lake's water, not its bed (a lake is a level
    # surface; the bed under it is the lake depth away)
    lake_here = water == dt.WATER_LAKE
    draw = np.where(lake_here, np.maximum(ws, surf), surf)
    sk = draw[k, k]
    h0, h1 = dt.height_grid(float(sk.min()), float(sk.max()))
    h = dt.encode_height_on(sk, h0, h1)
    ld = dt.face_lake_depth(surf, ws, water, lake_range)[k, k]
    b = np.clip(np.round(255.0 * (ld + lake_range) / (2.0 * lake_range)), 0, 255).astype(np.uint8)
    om = dt.face_smooth_mask(water == dt.WATER_OCEAN)[k, k]
    alpha = (255 - np.clip(np.round(255.0 * om), 0, 255)).astype(np.uint8)
    ground = np.stack([(h >> 8).astype(np.uint8), (h & 255).astype(np.uint8), b, alpha], axis=-1)
    del h, ld, om, b, alpha, ws, water

    # the water: the flood-tree flux, widened, on the product's land scale
    flux = np.asarray(a["flux"][ex, ex], np.float32)
    c = slice(p0w - e0, p0w - e0 + n)
    scale = river_scale(surf[c, c], ocean[c, c], flux[c, c])
    q_lo, q_hi = scale["lo"], scale["hi"]
    qb = dt.log_byte(dt.widen_rivers(flux, scale)[k, k], q_lo, q_hi)
    fb = dt.log_byte(flux[k, k], q_lo, q_hi)                                        # G: the flux itself (the flow layer)
    sb = dt.log_byte(np.asarray(a["sediment"][ex, ex], np.float32)[k, k], dt.SED_LO_M, dt.SED_HI_M)   # B: sediment
    has_veg = "vegetation" in a
    vb = (np.clip(np.round(255.0 * np.asarray(a["vegetation"][ex, ex], np.float32)[k, k]), 0, 255).astype(np.uint8)
          if has_veg else np.full_like(qb, 255))                                    # A: canopy cover
    wimg = np.stack([qb, fb, sb, vb], axis=-1)
    has_bio = "biome" in a
    bio = np.asarray(a["biome"][ex, ex], np.uint8)[k, k] if has_bio else np.zeros_like(qb)
    z8 = np.zeros_like(bio)
    bimg = np.stack([bio, z8, z8, np.full_like(bio, 255)], axis=-1)
    # cell (x0 + x, x0 + y): ground at (x, y), water at (x, NE + y), biome at (x, 2 NE + y)
    img = np.ascontiguousarray(np.concatenate([ground, wimg, bimg], axis=1).transpose(1, 0, 2))

    byte = lambda v: int(round(255.0 * math.log(max(v, q_lo) / q_lo) / math.log(q_hi / q_lo)))
    scales = {"h0": h0, "h1": h1, "lake_range": float(lake_range),
              "river_min_byte": int(np.clip(byte(scale["river_min"]), 1, 254)),
              "river_span_byte": max(byte(scale["river_full"]) - byte(scale["river_min"]), 8),
              "q_lo": float(q_lo), "q_hi": float(q_hi), "sed_lo": dt.SED_LO_M, "sed_hi": dt.SED_HI_M,
              "x0": int(x0), "NE": int(NE), "p0": int(p0), "n": int(n), "veg": bool(has_veg), "biome": bool(has_bio)}
    return img, scales


def _coarse(fine: int, R: int):
    """Fine cells as coarse cells: an int when whole, else the exact float."""
    return fine // R if fine % R == 0 else fine / R


def frame_paths(viewer: Path, name: str, R: int, k: int) -> Path:
    """The file of time-lapse frame ``k`` of a level (:func:`write_frames`)."""
    return Path(viewer) / "zoomtex" / f"{name}_L{int(R)}.t{int(k):02d}.js"


def frames_record(viewer: Path, name: str, R: int) -> Path:
    return Path(viewer) / "zoomtex" / f"{name}_L{int(R)}.frames.json"


def frames_image(surface: np.ndarray, discharge: np.ndarray, h0: float, h1: float, q_lo: float, q_hi: float) -> np.ndarray:
    """One time-lapse frame as a ``(2 S, S, 4)`` RGBA image, laid out as a
    level's texture so the viewer draws it the same way: the ground on top
    (the 16-bit surface code, no lake, opaque), its streams below (the
    discharge's log byte in R and G)."""
    from . import detail as dt

    h = dt.encode_height_on(np.asarray(surface, np.float32), h0, h1)
    S = h.shape[0]
    flat = np.full((S, S), 127, np.uint8)      # just under the shore byte: a frame is all land
    ground = np.stack([(h >> 8).astype(np.uint8), (h & 255).astype(np.uint8), flat, np.full((S, S), 255, np.uint8)], axis=-1)
    qb = dt.log_byte(np.asarray(discharge, np.float32), q_lo, q_hi)
    water = np.stack([qb, qb, np.zeros_like(qb), np.full_like(qb, 255)], axis=-1)
    return np.ascontiguousarray(np.concatenate([ground, water], axis=1).transpose(1, 0, 2))


def write_frames(viewer: Path, name: str, zdir: Path, R: int, N: int, cell_m: float | None = None,
                 geometry: dict | None = None, force: bool = False, log=None) -> dict | None:
    """Write a level's time-lapse (``globe.zoom.bake``: its product's surface
    and streams every few iterations, block-averaged) as one texture per frame
    and a sidecar the viewer plays them from.  All the frames share one height
    and one river scale, so nothing jumps between them.  None when the level
    kept no frames."""
    from ..zoom.bake import Geometry, frames_path

    npz = frames_path(Path(zdir), int(R))
    if not npz.exists():
        return None
    side = frames_record(viewer, name, R)
    if not force and _fresh(side, npz):
        try:
            rec = json.loads(side.read_text())
            if rec.get("version") == TEX_VERSION:
                return rec
        except (OSError, ValueError):
            pass
    t0 = time.time()
    stats_path = Path(zdir) / f"L{int(R)}.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    geo = stats.get("geometry") or geometry
    if geo is None:
        raise ValueError(f"no geometry for {npz}")
    gm = Geometry(**geo)
    with np.load(npz) as z:
        surf = np.asarray(z["surface"], np.float32)
        disch = np.asarray(z["discharge"], np.float32)
        its = [int(v) for v in z["iterations"]]
        factor = int(z["factor"][0])
    from . import detail as dt

    h0, h1 = dt.height_grid(float(surf.min()), float(surf.max()))
    last = disch[-1]
    scale = river_scale(surf[-1], np.zeros(surf[-1].shape, bool), last)
    q_lo, q_hi = scale["lo"], scale["hi"]
    byte = lambda v: int(round(255.0 * math.log(max(v, q_lo) / q_lo) / math.log(q_hi / q_lo)))
    fi0, fj0 = gm.product_origin
    files = []
    Path(viewer, "zoomtex").mkdir(parents=True, exist_ok=True)
    for k in range(surf.shape[0]):
        img = frames_image(surf[k], disch[k], h0, h1, q_lo, q_hi)
        js = frame_paths(viewer, name, R, k)
        _replace(js, 'GLOBE_VIEWER.zoomTex("%s", %d, "%s", %d);\n' % (name, gm.R, dt.webp_rgba_b64(img), k))
        files.append(f"zoomtex/{js.name}")
    S = int(surf.shape[1])
    rec = {"version": TEX_VERSION, "count": len(files), "files": files, "iterations": its, "factor": factor,
           "face": gm.face, "res": (int(N) * gm.R) / factor, "oi": fi0 / factor, "oj": fj0 / factor,
           "p0": 0, "n": S, "NE": S, "h0": h0, "h1": h1,
           "river_min_byte": int(np.clip(byte(scale["river_min"]), 1, 254)),
           "river_span_byte": max(byte(scale["river_full"]) - byte(scale["river_min"]), 8),
           "cell_m": None if cell_m is None else float(cell_m) * factor}
    _replace(side, json.dumps(rec, indent=1))
    if log is not None:
        log(f"[zoomtex] {name} L{gm.R}: {len(files)} time-lapse frames, {S}x{S} ({time.time() - t0:.1f}s)")
    return rec


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
    from ..zoom.bake import Geometry

    keys = ("height", "sediment", "water_surface", "flux", "ocean", "vegetation", "biome")
    with np.load(npz) as z:
        a = {k: z[k] for k in keys if k in z.files}
    if "biome" not in a:
        # a level baked before the bake classified its biomes
        try:
            from ..zoom.biomes import level_biomes_of

            a["biome"] = level_biomes_of(Path(zdir), int(R))
        except (OSError, ValueError, KeyError) as e:
            if log is not None:
                log(f"[zoomtex] {name} L{R}: no biomes ({e})")
    img, scales = level_image(a, geo)
    del a
    t1 = time.time()
    b64 = dt.webp_rgba_b64(img)
    js, side = paths(viewer, name, R)
    js.parent.mkdir(parents=True, exist_ok=True)
    gm = Geometry(**geo)
    Rg = gm.R
    fi0, fj0 = gm.product_origin
    assert int(img.shape[1]) == scales["NE"] and scales["n"] == gm.n
    # ci0 R - p0 is the face fine cell of pixel 0 (the viewer's offset), p0 the core's first pixel
    rec = {"file": f"zoomtex/{js.name}", "version": TEX_VERSION, "face": gm.face, "N": int(N), "R": Rg, "ci0": _coarse(fi0, Rg), "cj0": _coarse(fj0, Rg),
           "cells": gm.cells, "guard": gm.guard, "fi0": int(fi0), "fj0": int(fj0), "array_NE": gm.NE,
           **scales, "cell_m": None if cell_m is None else float(cell_m)}
    _replace(js, 'GLOBE_VIEWER.zoomTex("%s", %d, "%s");\n' % (name, Rg, b64))
    _replace(side, json.dumps(rec, indent=1))
    if log is not None:
        log(f"[zoomtex] {name} L{Rg}: {img.shape[1]}x{img.shape[0]}, {js.stat().st_size / 1e6:.2f} MB "
            f"(arrays {t1 - t0:.1f}s, webp {time.time() - t1:.1f}s)")
    return rec


__all__ = ["LAKE_RANGE", "CROP", "MAX_TEX", "TEX_VERSION", "RIVER_MIN_PCT", "RIVER_FULL_PCT", "paths", "record", "river_scale", "crop_window",
           "level_image", "write_level", "frame_paths", "frames_record", "frames_image", "write_frames"]
