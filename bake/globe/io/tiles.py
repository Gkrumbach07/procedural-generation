"""Tile format (PLAN.md section 3).

``tiles/L{lod}/f{face}/{x}_{y}/`` holds:

* ``height.png``  16-bit, normalised by ``meta["height_min"/"height_max"]``
* ``water.png``   16-bit water *surface* height; value 0 = no water,
  1..65535 map linearly to ``meta["water_min"/"water_max"]``
* ``layers.png``  RGBA8: R = sediment depth (m, clamped to
  ``LAYER_SEDIMENT_MAX``), G = hardness [0,1], B = biome id, A = vegetation
* ``flow.png``    RGB8: R = log-scaled discharge, G = basin-local id, B = river mask
  (PLAN section 3 says RG8; section 11 requires the river-mask channel)
* ``meta.json``   writer keys below plus any extra writer-supplied keys

Sample positions.  Tiles are ``(T + 1) × (T + 1)`` *vertex* samples.
Sample ``(k, l)`` of tile ``(lod, face, x, y)`` sits at face-local
``u = (x*T + k) * 2**lod / N_fine``, ``v = (y*T + l) * 2**lod / N_fine``,
``k, l = 0..T``, i.e. on fine-cell corners, so the ``k = T`` column is the
next tile's ``k = 0`` column and, on the last tile of a face, lies exactly
on the cube edge (``u = 1``) where the neighbouring face's tile places its
own edge column (cubesphere maps the edge exactly; cell-centre samples
would be offset by up to 0.5 cells along the edge).  Values are bilinear
interpolations of the fine cell-centre fields (2x2 average), evaluated
with cross-face halo data at face edges.  Shared columns agree to
interpolation and 16-bit quantisation tolerance; skirts (PLAN 12.3) still
hide LOD/quantisation cracks.

Ocean is implicit: ``water.png`` is 0 wherever ``water_surface <= 0``
(docs/DEVELOPING.md: ocean ``water_surface = 0``, ocean = surface < 0);
Godot renders sea level 0 where height < 0 and uses ``water.png`` only for
lakes.

Arrays are indexed ``[i, j]``; image row = j (v), column = i (u).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .png16 import denormalize_u16, normalize_u16, read_png16, read_png8, write_png8, write_png16

LAYER_SEDIMENT_MAX = 25.5  # m -> 0.1 m per 8-bit step
FLOW_DISCHARGE_LOG_MAX = 12.0


@dataclass
class Tile:
    lod: int
    face: int
    x: int
    y: int
    height: np.ndarray  # (T+1, T+1) float32, indexed [i, j]
    water: np.ndarray  # (T+1, T+1) float32 water surface (nan/0 -> none)
    layers: np.ndarray  # (T+1, T+1, 4) uint8
    flow: np.ndarray  # (T+1, T+1, 3) uint8
    meta: dict


def tile_dir(root: str | Path, lod: int, face: int, x: int, y: int) -> Path:
    return Path(root) / "tiles" / f"L{lod}" / f"f{face}" / f"{x}_{y}"


def encode_discharge(q: np.ndarray) -> np.ndarray:
    return np.clip(np.log1p(np.maximum(q, 0.0)) / FLOW_DISCHARGE_LOG_MAX * 255.0, 0, 255).astype(np.uint8)


def decode_discharge(b: np.ndarray) -> np.ndarray:
    return np.expm1(b.astype(np.float32) / 255.0 * FLOW_DISCHARGE_LOG_MAX)


def write_tile(root: str | Path, tile: Tile) -> Path:
    d = tile_dir(root, tile.lod, tile.face, tile.x, tile.y)
    d.mkdir(parents=True, exist_ok=True)
    # arrays are [i, j]; images are [row=j, col=i]
    h_u16, hmin, hmax = normalize_u16(tile.height.T)
    write_png16(d / "height.png", h_u16)
    water = np.nan_to_num(tile.water.T, nan=0.0)
    has_water = bool(np.any(water > 0))
    if has_water:
        wmin = float(np.min(water[water > 0]))
        w_u16, wmin, wmax = normalize_u16(np.where(water > 0, water, wmin), wmin, None, reserve_zero=True)
        w_u16[water <= 0] = 0
    else:
        wmin, wmax = 0.0, 1.0
        w_u16 = np.zeros_like(h_u16)
    write_png16(d / "water.png", w_u16)
    write_png8(d / "layers.png", np.transpose(tile.layers, (1, 0, 2)))
    write_png8(d / "flow.png", np.transpose(tile.flow, (1, 0, 2)))
    meta = dict(tile.meta)
    meta.update(
        {
            "lod": tile.lod,
            "face": tile.face,
            "x": tile.x,
            "y": tile.y,
            "size": int(tile.height.shape[0]),
            "height_min": hmin,
            "height_max": hmax,
            "water_min": wmin,
            "water_max": wmax,
            "has_water": has_water,
            "sediment_max": LAYER_SEDIMENT_MAX,
            "discharge_log_max": FLOW_DISCHARGE_LOG_MAX,
        }
    )
    with open(d / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=1, default=_json_default)
    return d


def read_tile(root: str | Path, lod: int, face: int, x: int, y: int) -> Tile:
    d = tile_dir(root, lod, face, x, y)
    with open(d / "meta.json") as fh:
        meta = json.load(fh)
    h = denormalize_u16(read_png16(d / "height.png"), meta["height_min"], meta["height_max"]).T
    w16 = read_png16(d / "water.png")
    w = denormalize_u16(w16, meta["water_min"], meta["water_max"], reserve_zero=True).T
    w[w16.T == 0] = 0.0
    layers = np.transpose(read_png8(d / "layers.png"), (1, 0, 2))
    flow = np.transpose(read_png8(d / "flow.png"), (1, 0, 2))
    return Tile(lod, face, x, y, h.astype(np.float32), w.astype(np.float32), layers, flow, meta)


def tile_exists(root: str | Path, lod: int, face: int, x: int, y: int) -> bool:
    return (tile_dir(root, lod, face, x, y) / "meta.json").exists()


def tiles_per_face(N_fine: int, T: int, lod: int) -> int:
    return max(1, (N_fine // T) >> lod)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))
