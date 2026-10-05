"""Static HTML viewer for a baked world: globe and flat map, pan/zoom, and a
timeline through tectonics -> erosion -> final that can be scrubbed or played.

    export_viewer(world_dir)            # -> <world>/viewer/index.html

Open ``index.html`` straight from disk; no server is needed.  Browsers
refuse ``fetch`` on ``file://``, so every data file is a ``<script>`` that
hands its payload to ``window.GLOBE_VIEWER``: ``data/meta.js`` describes the
frames, and ``data/fNNNN.js`` carries one frame's textures as base64 WebP.
``single=True`` also writes ``viewer/standalone.html`` with everything
inlined -- one file to copy or attach.

Textures are cube-face atlases: 3 columns x 2 rows of ``(res + 2)²`` tiles,
face ``f`` at column ``f % 3``, row ``f // 3``, with a one-cell border taken
from the neighbouring face so bilinear filtering is seamless.  Pixel
``(x, y) = (1 + i, 1 + j)`` of a tile is cell ``(i, j)``.  Texture 0 is
``R, G`` = 16-bit height over the frame's ``[h0, h1]`` metres and ``B`` = an
overlay byte (plate id in tectonics frames, log erosion discharge in erosion
frames and on the final frame).  Rivers are drawn from that discharge: the
particles' time-averaged stream map, a bundle of paths that fades out at its
edges, which the shader interpolates and eases the way McDonald renders his
stream map -- not hydro's D8 flow accumulation, which is one cell wide by
construction and can only step in 45-degree increments (the staircases and
dotted rivers of earth-v9).  The final frame adds climate/biome,
plate/sediment/crust and water/river-order/basin textures, a shoreline
texture (D8 flow, signed lake depth, ocean mask) from which the shader
places coasts and lake shores between cells instead of painting whole
cells, and a
``satellite`` layer that colours the ground continuously from the climate
channels.  The
shader (``viewer.html``) maps every pixel to a direction on the sphere and
then to ``(face, u, v)`` exactly as :mod:`globe.cubesphere` does, so no
reprojection happens anywhere.  Latitude has its pole on +Z, as in
:meth:`globe.cubesphere.Grid.latitude` and the climate stage.

Frames come from ``<world>/frames/`` (captured during the bake, see
:mod:`globe.viz.frames`); a world baked before frame capture existed still
gets a viewer with the final state, and ``scripts/capture_frames.py`` can
backfill its timeline.

Extra ``formats`` (comma separated): ``equirect`` writes lat/lon PNGs of the
final state (16-bit height, shaded relief, one per layer) to
``viewer/exports/``; ``anim`` writes an animated WebP of the whole timeline
as a flat map.
"""
from __future__ import annotations

import base64
import io
import json
import math
import shutil
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..cubesphere import from_sphere_v, to_sphere_v
from ..zoom import planet_frames as pfr
from . import frames as vf

TEMPLATE = Path(__file__).with_name("viewer.html")
PLACEHOLDER = "<!--GLOBE_VIEWER_DATA-->"
PAD = 1

# colour stops shared with the shader in viewer.html (keep in sync)
LAND = np.array([[.30, .50, .28], [.58, .64, .38], [.78, .70, .48], [.58, .45, .34], [.96, .96, .97]])
SEA = np.array([[.62, .80, .86], [.36, .62, .80], [.18, .42, .68], [.09, .24, .50], [.03, .09, .26]])
CMAPS = {
    "thermal": np.array([[.19, .07, .55], [.16, .44, .90], [.94, .94, .90], [.95, .55, .18], [.62, .05, .10]]),
    "rain": np.array([[.93, .87, .72], [.72, .80, .45], [.30, .65, .40], [.15, .45, .70], [.10, .15, .50]]),
    "viridis": np.array([[.27, 0, .33], [.23, .32, .55], [.13, .57, .55], [.37, .79, .38], [.99, .91, .14]]),
}
LIGHT = np.array([-0.6, 0.6, 0.75]) / np.linalg.norm([-0.6, 0.6, 0.75])


# --------------------------------------------------------------------------
# cube-face atlas encoding
# --------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _pad_index(res: int, pad: int):
    """For every cell of a padded face (``pad`` cells beyond each edge), the
    ``(face, i, j)`` of the interior cell containing its centre."""
    k = np.arange(-pad, res + pad)
    I, J = np.meshgrid(k, k, indexing="ij")
    shape = (6,) + I.shape
    faces = np.broadcast_to(np.arange(6)[:, None, None], shape)
    p = to_sphere_v(faces, np.broadcast_to((I + 0.5) / res, shape), np.broadcast_to((J + 0.5) / res, shape))
    f2, u2, v2 = from_sphere_v(p)
    i2 = np.minimum((u2 * res).astype(np.int64), res - 1)
    j2 = np.minimum((v2 * res).astype(np.int64), res - 1)
    return f2, i2, j2


def pad_faces(a: np.ndarray, pad: int = PAD) -> np.ndarray:
    """``(6, r, r)`` -> ``(6, r + 2·pad, r + 2·pad)`` with the border cells
    read from the neighbouring faces (nearest cell)."""
    f, i, j = _pad_index(a.shape[1], pad)
    return a[f, i, j]


def atlas(channels: list[np.ndarray]) -> np.ndarray:
    """Padded ``(6, T, T)`` uint8 channels -> ``(2T, 3T, C)`` atlas image."""
    T = channels[0].shape[1]
    img = np.zeros((2 * T, 3 * T, len(channels)), np.uint8)
    for f in range(6):
        r, c = divmod(f, 3)
        for k, ch in enumerate(channels):
            img[r * T:(r + 1) * T, c * T:(c + 1) * T, k] = ch[f].T
    return img


def encode_height(h: np.ndarray):
    """Padded metres -> (hi byte, lo byte, h0, h1).

    When the range spans sea level, 0 m is placed exactly on a code and
    every cell keeps its sign through the rounding.  On an Earth frame one
    code is ~0.2 m, and a quarter of the land cells on the coast stand under
    1 m (erosion fills the shelf to `dep_floor_m` below the water and plains
    to just above it): plain rounding put some of them below 0, the shader
    drew the whole cell as sea, and the coast came out as a staircase along
    cell centres."""
    h = np.asarray(h, np.float64)
    h0, h1 = float(np.floor(h.min())), float(np.ceil(h.max()))
    if h1 <= h0:
        h1 = h0 + 1.0
    if h0 < 0.0 < h1:
        # one code of slack, so moving h0 down onto the grid through 0
        # (by less than a code) cannot push the top value out of range
        step = (h1 - h0) / 65534.0
        z = math.ceil(-h0 / step)
        h0 = -z * step
        h1 = h0 + 65535.0 * step
        q = np.round((h - h0) / step)
        q = np.where(h < 0.0, np.minimum(q, z - 1), np.maximum(q, z))
    else:
        q = np.round((h - h0) / (h1 - h0) * 65535.0)
    q = np.clip(q, 0, 65535).astype(np.uint16)
    return (q >> 8).astype(np.uint8), (q & 255).astype(np.uint8), h0, h1


def log_byte(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """0 at or below ``lo``; ``255·ln(x/lo)/ln(hi/lo)`` above (clipped)."""
    x = np.asarray(x, np.float64)
    t = np.log(np.maximum(x, lo) / lo) / math.log(hi / lo)
    b = np.clip(np.round(255.0 * t), 1, 255).astype(np.uint8)
    b[~(x > lo)] = 0
    return b


def lin_byte(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(np.round(255.0 * (np.asarray(x, np.float64) - lo) / (hi - lo)), 0, 255).astype(np.uint8)


def webp_b64(img: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img), "RGB").save(buf, "WEBP", lossless=True, quality=70, method=4)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@lru_cache(maxsize=8)
def _area(res: int) -> np.ndarray:
    """Relative cell areas of an equi-angular face (same for all 6)."""
    s = np.tan(((np.arange(res) + 0.5) / res - 0.5) * (np.pi / 2))
    S, T = np.meshgrid(s, s, indexing="ij")
    w = (1 + S * S) * (1 + T * T) / (1 + S * S + T * T) ** 1.5
    return np.broadcast_to(w, (6, res, res))


def relief_exaggeration(h: np.ndarray, cell_m: float) -> float:
    """Default hillshade exaggeration: a 90th-percentile land slope shades
    like a 0.35 gradient.  An Earth-scale world (10 km cells, slopes ~1 %)
    needs tens; a 50 m-cell test world needs none."""
    g = np.hypot(np.diff(h, axis=1)[:, :, :-1], np.diff(h, axis=2)[:, :-1, :]) / max(cell_m, 1e-9)
    p90 = _pctl(g[h[:, :-1, :-1] >= 0], 90, 0.01)
    return float(np.clip(round(0.35 / max(p90, 1e-6)), 1, 50))


def frame_stats(h: np.ndarray) -> dict:
    w = _area(h.shape[1])
    land = h >= 0.0
    wl = float(w[land].sum())
    return {
        "land_pct": round(100.0 * wl / float(w.sum()), 3),
        "max_m": round(float(h.max()), 1),
        "median_land_m": round(float(np.median(h[land])), 1) if land.any() else None,
        "above2k_pct": round(100.0 * float(w[h >= 2000.0].sum()) / wl, 3) if wl > 0 else 0.0,
    }


# --------------------------------------------------------------------------
# collecting the frames
# --------------------------------------------------------------------------
def _load_faces(root: Path, name: str, sub: str = "coarse"):
    paths = [root / sub / f"{name}.f{k}.npy" for k in range(6)]
    if not all(p.exists() for p in paths):
        return None
    return np.stack([np.load(p) for p in paths])


def order_raster(root: Path, N: int) -> np.ndarray | None:
    """Strahler order per cell (uint8, 0 = no channel), painted from the
    reaches in ``graph/drainage.json`` -- the same graph the lakes' inflow
    and outflow nodes live in, so what is drawn as a river is what hydro
    routed, not the erosion stage's particle discharge."""
    path = root / "graph" / "drainage.json"
    if not path.exists():
        return None
    g = json.loads(path.read_text())
    out = np.zeros((6, N, N), np.uint8)
    for e in g.get("edges", []):
        cells = np.asarray(e.get("cells", ()), np.int64)
        if cells.size == 0:
            continue
        o = int(e.get("order", 1))
        f, i, j = cells[:, 0], cells[:, 1], cells[:, 2]
        cur = out[f, i, j]
        out[f, i, j] = np.maximum(cur, min(o, 255))
    return out


#: ``water`` channel codes.  Nothing else in the viewer knows what is water:
#: the timeline frames are pre-hydro, so there the shader still reads the sea
#: off the height, and only the final frame carries this.
WATER_LAND, WATER_LAKE, WATER_OCEAN = 0, 1, 2


def water_code(surface: np.ndarray, flow_dir: np.ndarray | None, water_surface: np.ndarray | None,
               lake_min_depth: float = 0.5) -> np.ndarray:
    """Per cell: 0 land, 1 lake, 2 ocean (uint8).

    Colouring elevation by ``height < 0`` gets both kinds of water wrong.  A
    lake is *above* sea level almost everywhere -- it is a filled depression,
    so its water stands at the spill point -- and therefore rendered as
    ordinary terrain, which is the "lakes don't render as lakes" report; and
    a closed basin below sea level is rendered as ocean, which is the
    "landlocked sea" one.  Hydro has already decided both: ``flow_dir ==
    OCEAN`` is the sea (:func:`globe.hydro.run.open_ocean`) and a water
    surface standing above the ground is a lake.
    """
    surf = np.asarray(surface, np.float32)
    ocean = (np.asarray(flow_dir) == 255) if flow_dir is not None else (surf < 0.0)
    code = np.full(surf.shape, WATER_LAND, np.uint8)
    code[ocean] = WATER_OCEAN
    if water_surface is not None:
        deep = (np.asarray(water_surface, np.float32) - surf) > np.float32(lake_min_depth)
        code[deep & ~ocean] = WATER_LAKE
    return code


#: signed lake depth encoding: metres of lake level above the ground, clipped
#: to +-LAKE_DEPTH_RANGE_M, so byte 127.5 is the shoreline
LAKE_DEPTH_RANGE_M = 200.0


def _dilate_max(a: np.ndarray, fill: float) -> np.ndarray:
    """3x3 maximum of a ``(6, r, r)`` field, reading across face edges."""
    f, i, j = _pad_index(a.shape[1], 1)
    p = a[f, i, j]
    out = np.full_like(a, fill)
    r = a.shape[1]
    for di in (0, 1, 2):
        for dj in (0, 1, 2):
            np.maximum(out, p[:, di:di + r, dj:dj + r], out=out)
    return out


def smooth_mask(mask: np.ndarray, passes: int = 2) -> np.ndarray:
    """``(6, r, r)`` boolean -> float in [0, 1]: 3x3 binomial passes across
    face edges.  Its 0.5 contour, interpolated, is a coastline that rounds
    the cell corners instead of tracing them.  Every cell centre is then
    held on its own side of 0.5 (by 0.1), so the blur rounds a one-cell
    island or strait instead of erasing it."""
    m = np.asarray(mask, bool)
    a = m.astype(np.float32)
    r = a.shape[1]
    w = (1.0, 2.0, 1.0)
    for _ in range(passes):
        f, i, j = _pad_index(r, 1)
        p = a[f, i, j]
        out = np.zeros_like(a)
        for di in (0, 1, 2):
            for dj in (0, 1, 2):
                out += (w[di] * w[dj] / 16.0) * p[:, di:di + r, dj:dj + r]
        a = out
    return np.where(m, np.maximum(a, 0.6), np.minimum(a, 0.4)).astype(np.float32)


def lake_depth(surface: np.ndarray, water_surface: np.ndarray | None, code: np.ndarray) -> np.ndarray | None:
    """Signed metres of lake level above the ground, per cell, for placing a
    shore *between* cells.

    A lake cell carries its own depth (``water_surface - surface``, > 0).  A
    cell next to a lake carries the highest neighbouring lake level less its
    own ground (< 0 where the ground stands above the water).  Interpolated
    bilinearly, that crosses 0 where the terrain meets the lake level, so the
    shoreline follows the ground instead of the cell grid; everything
    farther from a lake is ``-LAKE_DEPTH_RANGE_M``.  Unsigned depth would not
    do: between a lake cell and a dry one it never reaches 0, so the shore
    would sit on the dry cell's centre whatever the slope.
    """
    if water_surface is None:
        return None
    surf = np.asarray(surface, np.float64)
    lake = code == WATER_LAKE
    far = -LAKE_DEPTH_RANGE_M
    level = np.where(lake, np.asarray(water_surface, np.float64), -np.inf)
    near = _dilate_max(level, -np.inf)
    d = np.where(lake, level - surf, np.where(np.isfinite(near), near - surf, far))
    d[code == WATER_OCEAN] = far
    return np.clip(d, -LAKE_DEPTH_RANGE_M, LAKE_DEPTH_RANGE_M).astype(np.float32)


def _fine_final(root: Path, manifest: dict, surf_c: np.ndarray, flow_dir_c: np.ndarray | None, log=print) -> dict | None:
    """The final state on the refined grid (``fine/``), or None when refine
    has not run.  Sea and lakes follow derive's rules, so the viewer draws the
    water derive's rivers were clipped against: the sea is fine ground below
    0 in a coarse cell that is ocean or touches one
    (:func:`globe.derive.run.fine_ocean`), and a lake is a piece of fine
    water deeper than ``hydro.lake_min_depth`` that derive keeps
    (:func:`globe.derive.lakes.kept_lake_mask`)."""
    from ..derive import lakes as lakes_mod

    h = _load_faces(root, "height", "fine")
    if h is None:
        return None
    sed = _load_faces(root, "sediment", "fine")
    surf = (h + sed) if sed is not None else h
    N, Nf = surf_c.shape[1], surf.shape[1]
    if Nf % N:
        return None
    R = Nf // N
    params = manifest.get("params", {}) or {}
    hp = params.get("hydro", {}) or {}
    depth = max(float(hp.get("lake_min_depth", 0.5)), float(hp.get("marsh_depth", 0.0)))   # derive's line between lake and marsh
    min_cells = max(1, int(round(float((params.get("derive", {}) or {}).get("lake_min_cells", 1.0)) * R * R)))
    ocean_c = (np.asarray(flow_dir_c) == 255) if flow_dir_c is not None else (surf_c < 0.0)
    sea_near = _dilate_max(ocean_c.astype(np.float32), 0.0) > 0.0
    ws = _load_faces(root, "water_surface", "fine")
    ws_c = _load_faces(root, "water_surface")
    agree = float((params.get("derive", {}) or {}).get("lake_agree_cells", 0.0))    # a world derived before the rule keeps its lakes
    lake_c = None if ws_c is None else lakes_mod.lake_mask(surf_c, ws_c, depth, ocean=ocean_c)    # derive's coarse lakes
    water = np.zeros(surf.shape, np.uint8)
    for f in range(6):
        ocean = (surf[f] < 0.0) & np.repeat(np.repeat(sea_near[f], R, axis=0), R, axis=1)
        water[f][ocean] = WATER_OCEAN
        if ws is not None:
            lake = lakes_mod.lake_mask(surf[f], ws[f], depth, ocean=ocean)
            if lake_c is not None:
                lake = lakes_mod.agree_with_coarse(lake, lake_c[f], R, agree)      # as derive draws them
            water[f][lakes_mod.kept_lake_mask(lake, min_cells)] = WATER_LAKE
    return {"surf": surf, "sed": sed, "ws": ws, "water": water,
            "discharge": _load_faces(root, "discharge", "fine"), "biome": _load_faces(root, "biome", "fine"),
            "basin": _load_faces(root, "basin_id", "fine"),
            # the geologic map classified on the refined grid (derive/geology.classify_at): contacts as lines
            "rock": _load_faces(root, "rock", "fine"), "basement": _load_faces(root, "basement", "fine")}


def _planet_final(root: Path, manifest: dict, surf_c: np.ndarray, flow_dir_c: np.ndarray | None, planet: str | Path, res: int = 2048, log=print) -> dict | None:
    """The final state from a planet zoom level (``globe.zoom.planet``: e.g.
    ``zoom/planet_R8``, 1.2 km on the earth preset), block-reduced face by
    face to ``res`` cells per face -- a face of it is 8192^2, and the atlas
    of six at full resolution is past what a browser texture holds.  Heights
    and water surfaces are block means, discharge the block maximum (a river
    narrower than a block still shows) -- of the level's accumulated flow
    where it has one (:func:`detail.river_field`).  Sea and lakes follow
    derive's rules as :func:`_fine_final`'s do.  None when the level is not
    there."""
    from ..derive import lakes as lakes_mod
    from . import detail as dt

    pdir = Path(root) / planet if not Path(planet).is_absolute() else Path(planet)
    info_path = pdir / "planet.json"
    if not info_path.exists():
        return None
    R_lvl = int(json.loads(info_path.read_text())["level"]["R"])
    N = surf_c.shape[1]
    n = N * R_lvl
    res = int(min(res, n))
    k = n // res
    R = res // N
    params = manifest.get("params", {}) or {}
    depth = float((params.get("hydro", {}) or {}).get("lake_min_depth", 0.5))
    min_cells = max(1, int(round(float((params.get("derive", {}) or {}).get("lake_min_cells", 1.0)) * R * R)))
    ocean_c = (np.asarray(flow_dir_c) == 255) if flow_dir_c is not None else (surf_c < 0.0)
    sea_near = _dilate_max(ocean_c.astype(np.float32), 0.0) > 0.0
    river = dt.river_field(pdir, R_lvl)
    scale = dt.river_scale(pdir, R_lvl) if river == "flow" else None

    def red(face, name, how):
        a = np.load(pdir / f"L{R_lvl}.f{face}.{name}.npy", mmap_mode="r")
        out = np.empty((res, res), np.float32)
        for i in range(res):                                  # a block row at a time: never a whole face resident
            blk = np.asarray(a[i * k:(i + 1) * k, :res * k], np.float32).reshape(k, res, k)
            out[i] = blk.max(axis=(0, 2)) if how == "max" else blk.mean(axis=(0, 2))
        return out

    def red_rivers(face):
        """The flow widened at full resolution (:func:`detail.widen_rivers`),
        then block maxima: strips of block rows with the widening's reach
        beyond them."""
        a = np.load(pdir / f"L{R_lvl}.f{face}.flow.npy", mmap_mode="r")
        out = np.empty((res, res), np.float32)
        m, S = dt.RIVER_RADIUS, 64
        for i0 in range(0, res, S):
            i1 = min(res, i0 + S)
            r0, r1 = max(i0 * k - m, 0), min(i1 * k + m, n)
            w = dt.widen_rivers(np.asarray(a[r0:r1, :res * k], np.float32), scale)[i0 * k - r0:i1 * k - r0]
            out[i0:i1] = w.reshape(i1 - i0, k, res, k).max(axis=(1, 3))
        return out

    surf = np.empty((6, res, res), np.float32)
    sed = np.empty_like(surf)
    ws = np.empty_like(surf)
    q = np.empty_like(surf)
    water = np.zeros(surf.shape, np.uint8)
    for f in range(6):
        sed[f] = red(f, "sediment", "mean")
        surf[f] = red(f, "height", "mean") + sed[f]
        ws[f] = np.maximum(red(f, "water_surface", "mean"), surf[f])
        q[f] = red_rivers(f) if scale is not None else red(f, river, "max")
        ocean = (surf[f] < 0.0) & np.repeat(np.repeat(sea_near[f], R, axis=0), R, axis=1)
        water[f][ocean] = WATER_OCEAN
        lake = lakes_mod.kept_lake_mask(lakes_mod.lake_mask(surf[f], ws[f], depth, ocean=ocean), min_cells)
        water[f][lake] = WATER_LAKE
    log(f"[viewer] final frame from {pdir.name} (R={R_lvl}, {k}x{k} blocks -> {res}² per face; rivers from {river})")
    return {"surf": surf, "sed": sed, "ws": ws, "water": water, "discharge": q, "biome": None, "basin": None, "river_scale": scale}


def _fmt_cell(m: float) -> str:
    """A cell size as the viewer writes it: metres under a kilometre."""
    return f"{m / 1000.0:.1f} km" if m >= 1000.0 else f"{m:.0f} m"


class _Frame:
    """One timeline entry before encoding: metres + optional channels."""

    def __init__(self, stage, key, label, height, **ch):
        self.stage, self.key, self.label = stage, key, label
        self.height = np.asarray(height, np.float32)
        self.ch = {k: v for k, v in ch.items() if v is not None}
        self.river_scale = None           # the rivers' byte scale when the source sets it (a planet level's flow)

    @property
    def res(self) -> int:
        return int(self.height.shape[1])


def _thin(items: list, n: int) -> list:
    """``n`` evenly spaced items, always keeping the first and the last."""
    if len(items) <= n:
        return items
    return [items[i] for i in sorted({round(k * (len(items) - 1) / (n - 1)) for k in range(n)})]


def collect_frames(root: Path, final_res: int | None, log=print, frame_res: int | None = None,
                   max_frames: int | None = None, refined: bool = False, planet: str | None = None) -> tuple[list[_Frame], dict]:
    manifest = json.loads((root / "manifest.json").read_text())
    stages = manifest.get("stages", {})
    frames: list[_Frame] = []

    tect = vf.list_frames(root, "tectonics")
    ero = vf.list_frames(root, "erosion")
    # the last tectonics frame maps compact plate ids back to raw ones, so
    # read it before thinning can drop it
    alive_last = tect[-1][2].get("alive") if tect else None
    if max_frames:
        total = max(len(tect) + len(ero), 1)
        tect = _thin(tect, max(2, round(max_frames * len(tect) / total)))
        ero = _thin(ero, max(2, max_frames - len(tect)))
    lo = (lambda a, how="mean": vf.downsample(a, frame_res, how)) if frame_res else (lambda a, how="mean": a)
    scale = (stages.get("tectonics", {}).get("info", {}) or {}).get("scale_m_per_unit")
    if tect and scale:
        for key, p, meta in tect:
            with np.load(p) as z:
                # the crust's age and kind where the run captured them (frames before this
                # existed have neither, and the frame is what it was)
                crust = {"crust": lo(z["crust_kind"], "nearest"), "crust_age": lo(z["crust_age"].astype(np.float32))} \
                    if "crust_age" in z.files and "crust_kind" in z.files else {}
                frames.append(_Frame("tectonics", key, f"tectonics · step {key} / {meta.get('of', '?')}",
                                     lo(z["height"].astype(np.float32) * float(scale)), plate=lo(z["plate"], "nearest"), **crust))
    for key, p, meta in ero:
        with np.load(p) as z:
            frames.append(_Frame("erosion", key, f"erosion · iteration {key} / {meta.get('of', '?')}",
                                 lo(z["height"]), discharge=lo(z["discharge"], "max")))
    # then the planet level's own erosion, if it kept a time lapse: the same planet at the
    # level's cell (1.2 km on the earth preset) rather than the coarse grid's 9.8 km
    if planet:
        pdir = Path(root) / planet if not Path(planet).is_absolute() else Path(planet)
        info = json.loads((pdir / "planet.json").read_text()) if (pdir / "planet.json").exists() else {}
        R_lvl = int((info.get("level") or {}).get("R") or 0)
        lapse = pfr.load(pdir, R_lvl) if R_lvl else None
        if lapse is not None:
            cell = float(info.get("cell_m") or 0.0)
            at = _fmt_cell(cell) if cell else f"R = {R_lvl}"
            its = [int(v) for v in lapse["iterations"]]
            for k, it in enumerate(its):
                frames.append(_Frame("planet", it, f"{at} · iteration {it} / {its[-1]}",
                                     lapse["surface"][k].astype(np.float32),
                                     discharge=lapse["discharge"][k].astype(np.float32)))
            log(f"[viewer] planet time lapse: {len(its)} frames of {frames[-1].res}^2 a face at {at}")

    # the final state, from the coarse fields -- or the refined grid with
    # ``refined`` (render.viewer_refined) -- at (up to) full resolution
    h = _load_faces(root, "height")
    if h is not None:
        sed = _load_faces(root, "sediment")
        surf = h + (sed if sed is not None else 0.0)
        label = "final · " + ", ".join(s for s in ("erosion", "hydro", "derive") if stages.get(s, {}).get("done"))
    else:
        surf = _load_faces(root, "bedrock")
        sed = None
        label = "final · tectonics"
    if surf is None:
        raise FileNotFoundError(f"{root}: no height or bedrock on the coarse grid -- has tectonics run?")
    N = surf.shape[1]
    fine = None
    if planet and h is not None:
        fine = _planet_final(root, manifest, surf, _load_faces(root, "flow_dir"), planet, int(final_res or 2048), log)
    elif refined and h is not None:
        fine = _fine_final(root, manifest, surf, _load_faces(root, "flow_dir"), log)
    Nsrc = fine["surf"].shape[1] if fine is not None else N
    R = Nsrc // N
    r = min(Nsrc, int(final_res or Nsrc))
    # coarse-only channels are repeated onto the refined grid first
    up = (lambda a: None if a is None else np.repeat(np.repeat(a, R, axis=1), R, axis=2)) if R > 1 else (lambda a: a)
    ds = lambda a, how="mean": None if a is None else vf.downsample(a, r, how)
    plate = _load_faces(root, "plate_id")
    if plate is not None:
        plate = plate.astype(np.int64)
        if alive_last is not None and plate.max() < len(alive_last):
            plate = np.asarray(alive_last, np.int64)[np.maximum(plate, 0)]  # compact ids -> raw, as in the frames
        plate = plate + 1
    crust = _load_faces(root, "crust_kind", "diagnostics")
    crust_age = _load_faces(root, "crust_age", "diagnostics")
    if fine is not None:
        surf, sed, ws, water = fine["surf"], fine["sed"], fine["ws"], fine["water"]
        discharge, biome, basin = fine["discharge"], fine["biome"], fine["basin"]
        if biome is None and h is not None:
            # a planet level carries no classes: derive's, on the refined grid
            # where its resolution divides the frame's, else the coarse grid's
            fb = _load_faces(root, "biome", "fine")
            if fb is not None and Nsrc % fb.shape[1] == 0:
                k = Nsrc // fb.shape[1]
                biome = np.repeat(np.repeat(fb, k, axis=1), k, axis=2) if k > 1 else fb
            else:
                biome = up(_load_faces(root, "biome"))
    else:
        ws = _load_faces(root, "water_surface")
        water = water_code(surf, _load_faces(root, "flow_dir"), ws,
                           float((manifest.get("params", {}).get("hydro", {}) or {}).get("lake_min_depth", 0.5)))
        discharge, biome, basin = _load_faces(root, "discharge"), _load_faces(root, "biome"), _load_faces(root, "basin_id")
    ldepth = lake_depth(surf, ws, water)
    fine_rock = fine.get("rock") if fine is not None else None
    fine_base = fine.get("basement") if fine is not None else None
    if fine_rock is not None and fine_rock.shape != surf.shape:
        fine_rock = fine_base = None
    frames.append(_Frame(
        "final", Nsrc, label, ds(surf),
        water=ds(water, "nearest"),
        lake_depth=ds(ldepth, "nearest"),
        ocean=ds(smooth_mask(water == WATER_OCEAN)),
        discharge=ds(discharge, "max"),
        flow=ds(up(_load_faces(root, "flow_acc")), "max"),
        order=ds(up(order_raster(root, N)), "max"),
        basin=ds(None if basin is None else (basin.astype(np.int64) + 1).clip(0), "nearest"),
        temperature=ds(up(_load_faces(root, "temperature"))),
        precip=ds(up(_load_faces(root, "precip"))),
        biome=ds(biome, "nearest"),
        plate=ds(up(plate), "nearest"),
        sediment=ds(sed),
        crust=ds(up(crust), "nearest") if crust is not None else None,
        crust_age=ds(up(crust_age), "nearest") if crust_age is not None and crust is not None else None,
        # the geologic map (derive/geology.py) and the crust under it, for the cross-sections
        rock=ds(fine_rock if fine_rock is not None else up(_load_faces(root, "rock")), "nearest"),
        basement=ds(fine_base if fine_base is not None else up(_load_faces(root, "basement")), "nearest"),
        crust_thickness=ds(up(_load_faces(root, "crust_thickness", "diagnostics"))),
    ))
    frames[-1].river_scale = fine.get("river_scale") if fine is not None else None
    src = (f"planet level {planet}" if planet else f"refined grid, R={R}") if fine is not None else "coarse grid"
    log(f"[viewer] {len(tect) if scale else 0} tectonics + {sum(f.stage == 'erosion' for f in frames)} erosion frames + final ({r}² per face, {src})")
    return frames, manifest


# --------------------------------------------------------------------------
# channel encoding
# --------------------------------------------------------------------------
def _pctl(a, q, default):
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if a.size else default


def channel_specs(final: _Frame, river_threshold: float | None = None) -> dict:
    """Byte encodings, fixed across the timeline so frames compare.
    ``river_threshold`` is hydro's ``river_threshold_volume``: a cell whose
    flow accumulation exceeds it is a channel in the drainage graph, so the
    final frame's rivers are drawn at exactly that line."""
    specs = {}
    land = final.height >= 0
    if "biome" in final.ch:
        from ..derive.biomes import NAMES
        specs["satellite"] = {"label": "Satellite", "kind": "category", "cmap": "satellite",
                              "names": [n.replace("_", " ") for n in NAMES]}
    if "flow" in final.ch:
        q = final.ch["flow"].astype(np.float64)
        ql = q[land & (q > 0)]
        lo = max(_pctl(ql, 50, 1.0), 1e-6)
        hi = max(float(q.max()), lo * 10)
        rmin = float(river_threshold) if river_threshold else _pctl(ql, 97, lo * 4)
        specs["flow"] = {"label": "Flow accumulation", "kind": "log", "lo": lo, "hi": hi, "unit": "", "cmap": "viridis",
                         "river_min": max(rmin, lo * 1.001)}
    if "order" in final.ch:
        specs["order"] = {"label": "River order", "kind": "category", "cmap": "order",
                          "names": ["—"] + [str(k) for k in range(1, 256)]}
    if "basin" in final.ch:
        specs["basin"] = {"label": "Basins", "kind": "category", "cmap": "plates", "offset": 1}
    if "discharge" in final.ch:
        q = final.ch["discharge"].astype(np.float64)
        ql = q[land & (q > 0)]
        lo = max(_pctl(ql, 50, 1.0), 1e-6)
        hi = max(float(q.max()), lo * 10)
        # rivers fade in from the 85th percentile of land discharge and are
        # full strength by the 99.5th: an eased ramp, not a threshold, so a
        # channel has soft banks and the faint paths beside it show
        specs["discharge"] = {"label": "Discharge", "kind": "log", "lo": lo, "hi": hi, "unit": "", "cmap": "viridis",
                              "river_min": _pctl(ql, 85, lo * 2), "river_full": _pctl(ql, 99.5, lo * 40)}
        if getattr(final, "river_scale", None):
            # a planet level's accumulated flow, widened: the scale of its
            # source's cells, so the globe and the detail tiles draw one map
            specs["discharge"] = {"label": "Rivers (accumulated flow)", "kind": "log", "unit": "", "cmap": "viridis", **final.river_scale}
    if "temperature" in final.ch:
        specs["temperature"] = {"label": "Temperature", "kind": "linear", "lo": -45.0, "hi": 35.0, "unit": "°C", "cmap": "thermal"}
    if "precip" in final.ch:
        p = final.ch["precip"]
        specs["precip"] = {"label": "Precipitation", "kind": "log", "lo": 0.02, "hi": max(_pctl(p[land], 99.5, 5.0), 0.1),
                           "unit": "× land mean", "cmap": "rain"}
    if "biome" in final.ch:
        from ..derive.biomes import NAMES
        specs["biome"] = {"label": "Biome", "kind": "category", "cmap": "biome", "names": [n.replace("_", " ") for n in NAMES]}
    if "plate" in final.ch:
        specs["plate"] = {"label": "Plates", "kind": "category", "cmap": "plates", "offset": 1}
    if "sediment" in final.ch:
        s = final.ch["sediment"]
        specs["sediment"] = {"label": "Sediment", "kind": "log", "lo": 1.0, "hi": max(_pctl(s, 99.9, 100.0), 10.0), "unit": "m", "cmap": "viridis"}
    if "crust" in final.ch:
        # the kind is one bit and the age fills the rest of the same byte: a whole texture for
        # the age would be another 25 MB of the export, and a crust byte held one bit of news
        specs["crust"] = {"label": "Crust", "kind": "category", "cmap": "plates",
                          "names": ["oceanic", "continental"], "bits": [7, 1]}
    if "crust_age" in final.ch and "crust" in final.ch:
        # a scale for each kind of crust.  The sea floor is recycled -- half of it is younger
        # than a fifth of the run -- while the continents are mostly as old as the run itself,
        # so one scale for both puts every ocean in the first colours of the ramp and no ridge
        # shows.  Each kind gets the whole ramp over its own range, tails clipped so a rare
        # old sliver does not eat it (the byte says where along its kind's ramp a cell is)
        # Log, because the sea floor's ages are not spread evenly over their range: half of it
        # is younger than a fifth of the oldest, so on a linear scale a ridge and the floor a
        # thousand steps away are the same colour and the stripes never show
        age, cont = final.ch["crust_age"], np.asarray(final.ch["crust"]) > 0
        o, c = age[~cont], age[cont]
        o_lo = max(_pctl(o, 1, 1.0), 1.0) if o.size else 1.0
        o_hi = max(_pctl(o, 99, 10.0), o_lo * 2) if o.size else 10.0
        c_lo = max(_pctl(c, 2, 1.0), 1.0) if c.size else 1.0
        c_hi = max(_pctl(c, 100, 10.0), c_lo * 1.2) if c.size else 10.0
        specs["crust_age"] = {"label": "Crust age", "kind": "log", "lo": o_lo, "hi": o_hi, "unit": "steps",
                              "cont_lo": c_lo, "cont_hi": c_hi, "cmap": "age", "bits": [0, 127],
                              "lo_label": "new", "hi_label": f"{o_hi:.0f} / {c_hi:.0f} steps"}
    if "rock" in final.ch:
        from ..derive import geology
        specs["rock"] = {"label": "Geology", "kind": "category", "cmap": "rock", "names": list(geology.NAMES), "families": list(geology.FAMILY)}
        if "basement" in final.ch:
            specs["basement"] = {"label": "Basement rock", "kind": "category", "cmap": "rock", "names": list(geology.NAMES),
                                 "families": list(geology.FAMILY)}
    if "crust_thickness" in final.ch:
        specs["crust_thickness"] = {"label": "Crust thickness", "kind": "linear", "lo": 0.0, "hi": CRUST_THICKNESS_KM, "unit": "km", "cmap": "viridis"}
    if "water" in final.ch:
        specs["water"] = {"label": "Water", "kind": "category", "cmap": "plates", "names": ["land", "lake", "ocean"]}
    # read by the shader to place shores between cells, not offered as layers
    if "lake_depth" in final.ch:
        specs["lake_depth"] = {"label": "Lake level above ground", "kind": "linear", "lo": -LAKE_DEPTH_RANGE_M,
                               "hi": LAKE_DEPTH_RANGE_M, "unit": "m", "cmap": "viridis", "hidden": True}
    if "ocean" in final.ch:
        specs["ocean"] = {"label": "Ocean", "kind": "linear", "lo": 0.0, "hi": 1.0, "unit": "", "cmap": "viridis", "hidden": True}
    return specs


def _byte(name: str, a: np.ndarray, specs: dict, ch: dict | None = None) -> np.ndarray:
    s = specs[name]
    if name == "crust" and ch is not None and "crust_age" in ch and "crust_age" in specs:
        # kind in the top bit, age in the seven below it (see channel_specs)
        sa = specs["crust_age"]
        mask = int(sa["bits"][1])
        cont = np.asarray(a, np.int64) > 0
        lo = np.where(cont, sa.get("cont_lo", sa["lo"]), sa["lo"])
        hi = np.where(cont, sa.get("cont_hi", sa["hi"]), sa["hi"])
        v = np.maximum(np.asarray(ch["crust_age"], np.float32), lo)
        t = np.log(v / lo) / np.log(np.maximum(hi / lo, 1.0 + 1e-6))
        return ((cont.astype(np.uint8) << 7) | np.rint(np.clip(t, 0.0, 1.0) * mask).astype(np.uint8)).astype(np.uint8)
    if name in ("plate", "basin"):
        b = np.asarray(a, np.int64)
        return np.where(b > 0, 1 + (b - 1) % 255, 0).astype(np.uint8)
    if s["kind"] == "category":
        return np.clip(np.asarray(a, np.int64), 0, 255).astype(np.uint8)
    if s["kind"] == "log":
        return log_byte(a, s["lo"], s["hi"])
    return lin_byte(a, s["lo"], s["hi"])


# textures beyond texture 0 on the final frame: (R, G, B) channel names
# (short tuples are padded with zero channels)
FINAL_TEXTURES = (("temperature", "precip", "biome"), ("plate", "sediment", "crust"), ("water", "order", "basin"),
                  ("flow", "lake_depth", "ocean"), ("rock", "basement", "crust_thickness"))
#: top of the crust-thickness byte (km): Earth's thickest crust is ~75 km, under Tibet
CRUST_THICKNESS_KM = 80.0


def encode_frame(fr: _Frame, specs: dict) -> tuple[list[np.ndarray], dict]:
    """-> ([atlas images], frame meta)."""
    hi, lo, h0, h1 = encode_height(pad_faces(fr.height))
    zero = np.zeros_like(hi)
    over = "plate" if fr.stage == "tectonics" else "discharge"
    b = pad_faces(_byte(over, fr.ch[over], specs)) if over in fr.ch and over in specs else zero
    images = [atlas([hi, lo, b])]
    layers = {over: [0, 2]} if over in fr.ch and over in specs else {}
    river_min_byte = river_full_byte = None
    if over in layers and "river_min" in specs[over]:
        s = specs[over]
        river_min_byte = int(log_byte(np.array([s["river_min"]]), s["lo"], s["hi"])[0])
        river_full_byte = int(log_byte(np.array([s.get("river_full", s["river_min"])]), s["lo"], s["hi"])[0])
    if fr.stage == "final":
        for names in FINAL_TEXTURES:
            present = [n for n in names if n in fr.ch and n in specs]
            if not present:
                continue
            chans = [pad_faces(_byte(n, fr.ch[n], specs, fr.ch)) if n in present else zero for n in names]
            chans += [zero] * (3 - len(chans))
            k = len(images)
            images.append(atlas(chans))
            for c, n in enumerate(names):
                if n in present:
                    layers[n] = [k, c]
    elif "crust" in fr.ch and "crust" in specs:
        # a timeline frame carries the crust in a texture of its own: the sea floor's age is
        # the one thing about a tectonic step that its height does not show
        images.append(atlas([pad_faces(_byte("crust", fr.ch["crust"], specs, fr.ch)), zero, zero]))
        layers["crust"] = [len(images) - 1, 0]
        if "crust_age" in specs:
            layers["crust_age"] = layers["crust"]
    if fr.stage == "final":
        if "biome" in layers and "satellite" in specs:
            layers["satellite"] = layers["biome"]   # same texture, coloured for terrain rather than for classes
        if "crust" in layers and "crust_age" in specs:
            layers["crust_age"] = layers["crust"]   # the same byte, read from its other bits
    meta = {"stage": fr.stage, "key": int(fr.key), "label": fr.label, "res": fr.res, "pad": PAD,
            "h0": h0, "h1": h1, "layers": layers, "stats": frame_stats(fr.height)}
    if river_min_byte is not None:
        meta["river_min_byte"] = river_min_byte
        meta["river_span_byte"] = max(river_full_byte - river_min_byte, 1)
    return images, meta


# --------------------------------------------------------------------------
# river lines (docs/viewer-rivers.md, "River lines")
# --------------------------------------------------------------------------
#: traced river lines: rivers carry at least this percentile of land discharge
#: (the texture fades in from the 85th, but traced that far down a plain's
#: sheet flow becomes parallel hatching), with this many cells upstream
TRACED_RIVER_PERCENTILE = 97.0
TRACED_MIN_LENGTH = 8
#: tracing the source at its own resolution (a planet level: 8192² a face at
#: R = 8) finds far more, so the line is drawn higher up and short tributaries
#: are dropped, to keep the payload a few MB
TRACED_FULL_PERCENTILE = 99.0
TRACED_FULL_MIN_LENGTH = 12
TRACED_FULL_TOLERANCE = 0.4


def _traced_full(src, s: dict, log) -> tuple[dict, np.ndarray] | None:
    """Lines traced from the frame's own source at its resolution (``src``:
    :mod:`globe.viz.detail`'s), face by face."""
    from . import river_lines as rl

    step = max(1, src.res // 1024)
    qs = []
    for f in range(6):
        d = src.rows(f, 0, src.res)
        q, surf = d.get("discharge"), d["surf"]
        if q is None:
            return None
        qs.append(np.asarray(q[::step, ::step])[np.asarray(surf[::step, ::step]) > 0])
        del d, q, surf
    qs = np.concatenate(qs)
    qs = qs[qs > 0]
    if not qs.size:
        return None
    q_min = float(np.percentile(qs, TRACED_FULL_PERCENTILE))

    def load(f):
        d = src.rows(f, 0, src.res)
        water = np.where(d["water"] == 2, 2, np.where(d["water"] == 1, 1, 0)).astype(np.uint8)
        return {"discharge": d["discharge"], "water": water, "surface": d["surf"]}

    out = rl.trace_faces(load, src.res, q_min, min_length=TRACED_FULL_MIN_LENGTH, tolerance=TRACED_FULL_TOLERANCE)
    log(f"[viewer] traced river lines at {src.res}² a face: {({k: v for k, v in out.items() if not isinstance(v, np.ndarray)})}")
    return out, out["discharge"].astype(np.float64)


def river_lines(root: Path, final: _Frame, specs: dict, source: str = "graph", log=print, full=None) -> dict | None:
    """The final frame's rivers as polylines for the viewer to draw at their
    own width: ``source`` ``"graph"`` reads derive's ``graph/rivers.json``
    (the drainage graph's reaches traced through the refined channel and
    Catmull-Rom smoothed, a width in metres from their discharge), ``"traced"``
    follows the final frame's discharge down its surface's flood tree
    (:mod:`globe.viz.river_lines`).
    Returns ``{"xyz": (n, 3) int16 unit vector x 32767, "width": (n,) uint16
    metres, "byte": (n,) uint8 fade byte of ``channel``, "lengths": (m,)
    uint32 vertices per line, "channel": spec name}``, or None."""
    from . import river_lines as rl

    if source == "graph":
        path = Path(root) / "graph" / "rivers.json"
        if not path.exists() or "flow" not in specs:
            return None
        rivers = json.loads(path.read_text()).get("rivers", [])
        s = specs["flow"]
        pts, widths, bytes_, lengths = [], [], [], []
        for r in rivers:
            p = np.asarray(r["points"], np.float64)
            if p.shape[0] < 2:
                continue
            pts.append(to_sphere_v(p[:, 0].astype(np.int64), p[:, 1], p[:, 2]))
            widths.append(p[:, 3])
            q = r.get("discharge")
            b = int(log_byte(np.array([q if q is not None else s["river_min"]]), s["lo"], s["hi"])[0])
            bytes_.append(np.full(p.shape[0], b, np.uint8))
            lengths.append(p.shape[0])
        channel = "flow"
    elif source == "traced":
        if not {"discharge", "water"} <= final.ch.keys() or "discharge" not in specs:
            return None
        s = specs["discharge"]
        res = _traced_full(full, s, log) if full is not None else None
        if res is not None:
            out, q = res
        else:
            land_q = final.ch["discharge"][(final.height >= 0) & (final.ch["discharge"] > 0)]
            q_min = _pctl(land_q, TRACED_RIVER_PERCENTILE, s["river_min"])
            out = rl.trace(final.ch["discharge"], final.ch["water"], final.height, q_min, min_length=TRACED_MIN_LENGTH)
            q = out["discharge"].astype(np.float64)
        if not out["lengths"].size:
            return None
        qmax = max(float(q.max()), float(final.ch["discharge"].max()), 1e-9)
        pts = [out["xyz"]] if out["lengths"].size else []
        widths = [np.maximum(rl.RIVER_WIDTH_MAX_M * (q / qmax) ** rl.RIVER_WIDTH_EXPONENT, 30.0)]
        bytes_ = [log_byte(q, s["lo"], s["hi"])]
        lengths = list(out["lengths"])
        channel = "discharge"
        if res is None:
            log(f"[viewer] traced river lines on the frame ({final.res}² a face): {({k: v for k, v in out.items() if not isinstance(v, np.ndarray)})}")
    else:
        return None
    if not pts:
        return None
    xyz = np.concatenate(pts)
    return {"xyz": np.round(xyz * 32767.0).astype(np.int16), "width": np.clip(np.round(np.concatenate(widths)), 1, 65535).astype(np.uint16),
            "byte": np.concatenate(bytes_).astype(np.uint8), "lengths": np.asarray(lengths, np.uint32), "channel": channel, "source": source}


def river_lines_script(lines: dict, specs: dict) -> tuple[str, dict]:
    """``GLOBE_VIEWER.rivers({...})`` and the meta entry describing it."""
    b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).astype(a.dtype.newbyteorder("<")).tobytes()).decode("ascii")
    s = specs[lines["channel"]]
    min_byte = int(log_byte(np.array([s["river_min"]]), s["lo"], s["hi"])[0])
    # graded over the rivers drawn, so a trunk is wider and stays longer as
    # the view zooms out than a stream just over the line
    top = float(np.percentile(lines["byte"], 99)) if lines["byte"].size else min_byte + 12
    span = max(int(round(top)) - min_byte, 12)
    info = {"source": lines["source"], "lines": int(lines["lengths"].size), "vertices": int(lines["width"].size),
            "min_byte": min_byte, "span_byte": span}
    payload = {"xyz": b64(lines["xyz"]), "width": b64(lines["width"]), "byte": b64(lines["byte"]), "lengths": b64(lines["lengths"])}
    return "GLOBE_VIEWER.rivers(%s);\n" % json.dumps(payload, separators=(",", ":")), info


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------
def detail_source(root: Path, manifest: dict, refined: bool, planet: str | None):
    """The final frame's full-resolution source for detail tiles
    (:mod:`globe.viz.detail`), or None."""
    from . import detail as dt

    h = _load_faces(root, "height")
    if h is None:
        return None
    sed = _load_faces(root, "sediment")
    surf_c = h + (sed if sed is not None else 0.0)
    flow_dir_c = _load_faces(root, "flow_dir")
    if planet:
        pdir = Path(root) / planet if not Path(planet).is_absolute() else Path(planet)
        if not (pdir / "planet.json").exists():
            return None
        R = int(json.loads((pdir / "planet.json").read_text())["level"]["R"])
        params = manifest.get("params", {}) or {}
        depth = float((params.get("hydro", {}) or {}).get("lake_min_depth", 0.5))
        min_cells = max(1, int(round(float((params.get("derive", {}) or {}).get("lake_min_cells", 1.0)) * R * R)))
        ocean_c = (np.asarray(flow_dir_c) == 255) if flow_dir_c is not None else (surf_c < 0.0)
        sea_near = _dilate_max(ocean_c.astype(np.float32), 0.0) > 0.0
        return dt.PlanetSource(pdir, R, surf_c.shape[1], sea_near, depth, min_cells)
    if refined:
        fine = _fine_final(root, manifest, surf_c, flow_dir_c)
        return None if fine is None else dt.RefinedSource(fine["surf"], fine["ws"], fine["water"], fine["discharge"])
    return None


def export_viewer(world_dir, out=None, *, formats: str = "", final_res: int | None = None,
                  frame_res: int | None = None, max_frames: int | None = None,
                  single: bool = False, refined: bool | None = None, planet: str | None = None,
                  river_source: str = "none", detail: bool = False, log=print) -> Path:
    """``frame_res`` / ``max_frames`` downsample and thin the captured
    timeline -- a light export for a slow link or a phone.  ``river_source``
    (:func:`river_lines`): ``"auto"`` draws derive's graph rivers, except on a
    planet level's frame (they were traced on the refined grid, not on it)
    or where there are none, where the lines are traced; ``"none"`` leaves
    the rivers to the discharge texture."""
    t0 = time.time()
    root = Path(world_dir)
    out = Path(out) if out else root / "viewer"
    m0 = manifest_of(root)
    refined = bool(_render_param(m0, "viewer_refined", False)) if refined is None else bool(refined)
    # the refined frame is asked for to see the refined grid: full resolution unless told otherwise
    fres = final_res if final_res else (0 if refined else _render_param(m0, "viewer_final_res", 0))
    frames, manifest = collect_frames(root, fres, log,
                                      frame_res=frame_res, max_frames=max_frames, refined=refined, planet=planet)
    final = frames[-1]
    hydro_info = (manifest.get("stages", {}).get("hydro", {}) or {}).get("info", {}) or {}
    specs = channel_specs(final, hydro_info.get("river_threshold_volume"))
    land_final = final.height[final.height >= 0]
    land_top = max(_pctl(land_final, 99.5, 1000.0), 200.0)
    ocean_bottom = min(_pctl(final.height[final.height < 0], 1.0, -4000.0), -200.0)
    default_exag = relief_exaggeration(final.height, float(manifest.get("cell_size_m", 1.0)) * int(manifest.get("N_c", final.res)) / final.res)

    shutil.rmtree(out / "data", ignore_errors=True)
    (out / "data").mkdir(parents=True, exist_ok=True)
    metas, scripts, sizes = [], [], 0
    for i, fr in enumerate(frames):
        images, meta = encode_frame(fr, specs)
        payload = "GLOBE_VIEWER.frame(%d, [%s]);\n" % (i, ",".join('"%s"' % webp_b64(im) for im in images))
        sizes += len(payload)
        meta["file"] = f"data/f{i:04d}.js"
        if len(images) > 1:
            meta["tex"] = [{"res": fr.res, "pad": PAD} for _ in images]
        (out / meta["file"]).write_text(payload)
        if single:
            scripts.append(payload)
        metas.append(meta)

    rivers_js, rivers_info = None, None
    lines = None
    if river_source == "auto":
        lines = (None if planet else river_lines(root, final, specs, "graph", log)) or river_lines(root, final, specs, "traced", log, full=detail_source(root, manifest, refined, planet))
    elif river_source and river_source != "none":
        lines = river_lines(root, final, specs, river_source, log, full=detail_source(root, manifest, refined, planet) if river_source == "traced" else None)
    if lines is not None:
        rivers_js, rivers_info = river_lines_script(lines, specs)
        rivers_info["file"] = "data/rivers.js"
        (out / "data" / "rivers.js").write_text(rivers_js)
        sizes += len(rivers_js)
        log(f"[viewer] river lines ({river_source}): {rivers_info['lines']:,} lines, {rivers_info['vertices']:,} vertices, {len(rivers_js) / 1e6:.1f} MB")

    detail_info = None
    if detail:
        from . import detail as dt

        src = detail_source(root, manifest, refined, planet)
        if src is None:
            # asked for and not delivered: say so.  `--planet` is a path *under the world*
            # (`zoom/planet_R8`), and a bare level name finds nothing -- which used to
            # export a viewer with no detail layer at all and not one word about it
            hint = (f"no planet.json under {root / planet}" if planet
                    else "no --planet level and refine's fine output is not there")
            log(f"[viewer] WARNING: --detail asked for but there is no source for it ({hint}); "
                f"the viewer has no close-up layer")
        else:
            td = time.time()
            detail_info = dt.export_tiles(out, src, final.res, LAKE_DEPTH_RANGE_M, log)
            if detail_info:
                log(f"[viewer] detail tiles: {', '.join(f'L{lv['L']} {lv['res']}² {lv['tiles']:,}' for lv in detail_info['levels'])} in {time.time() - td:.0f}s")

    meta = {
        "world": root.resolve().name,
        "N_c": manifest.get("N_c"), "cell_size_m": manifest.get("cell_size_m"), "seed": manifest.get("seed"),
        "R_planet_m": float(manifest.get("R_planet") or manifest.get("N_c", 1024) * manifest.get("cell_size_m", 9773.0) * 2 / math.pi),
        "sea_level_m": 0.0, "land_top_m": land_top, "ocean_bottom_m": ocean_bottom, "default_exag": default_exag,
        "channels": specs,
        "biome_palette": _biome_palette() if "biome" in specs else [],
        "satellite_palette": SATELLITE_PALETTE if "satellite" in specs else [],
        "rock_palette": _rock_palette() if "rock" in specs else [],
        "biome_vegetation": _biome_vegetation() if "satellite" in specs else [],
        "biome_tree_kinds": _biome_tree_kinds() if "satellite" in specs else [],
        "stage_seconds": {s: round(v.get("seconds", 0.0), 1) for s, v in manifest.get("stages", {}).items()},
        "exported": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "final_index": len(frames) - 1,
        "frames": metas,
        "river_lines": rivers_info,
        "detail": detail_info,
    }
    meta_js = "GLOBE_VIEWER.setMeta(%s);\n" % json.dumps(meta, separators=(",", ":"))
    (out / "data" / "meta.js").write_text(meta_js)
    html = TEMPLATE.read_text()
    extra = '\n<script src="data/rivers.js"></script>' if rivers_js else ""
    (out / "index.html").write_text(html.replace(PLACEHOLDER, '<script src="data/meta.js"></script>' + extra + '\n<script src="zooms.js"></script>\n<script src="scout.js"></script>'))
    if not (out / "scout.js").exists():
        (out / "scout.js").write_text("GLOBE_VIEWER.setScout([]);\n")
    from ..zoom import index as zoom_index
    zooms = zoom_index.scan(root)
    (out / "zooms.js").write_text(zoom_index.script(zooms))
    if single:
        inline_meta = json.loads(json.dumps(meta))
        for m in inline_meta["frames"]:
            m["file"] = None
        if inline_meta.get("river_lines"):
            inline_meta["river_lines"]["file"] = None
        blocks = ["<script>GLOBE_VIEWER.setMeta(%s);</script>" % json.dumps(inline_meta, separators=(",", ":"))]
        if rivers_js:
            blocks.append("<script>%s</script>" % rivers_js)
        blocks += ["<script>%s</script>" % s for s in scripts]
        blocks.append("<script>%s</script>" % zoom_index.script(zooms))
        (out / "standalone.html").write_text(html.replace(PLACEHOLDER, "\n".join(blocks)))
    log(f"[viewer] {len(frames)} frames, {sizes / 1e6:.1f} MB -> {out / 'index.html'} in {time.time() - t0:.1f}s")

    fmts = {f.strip() for f in (formats or "").split(",") if f.strip()}
    if fmts:
        export_extras(frames, specs, meta, out / "exports", fmts, log)
    return out / "index.html"


def manifest_of(root: Path) -> dict:
    return json.loads((Path(root) / "manifest.json").read_text())


def _render_param(manifest: dict, key: str, default):
    return (manifest.get("params", {}).get("render", {}) or {}).get(key, default)


def _biome_vegetation() -> list:
    """derive.biomes' vegetation factor per class: how much of the ground the
    satellite layer covers with forest, and how dense its trees stand."""
    from ..derive.biomes import VEG_FACTOR

    return [round(float(v), 3) for v in VEG_FACTOR]


def _biome_tree_kinds() -> list:
    """derive.biomes' tree kind per class (TREE_*): the shape and colour of the
    satellite layer's tree symbols."""
    from ..derive.biomes import TREE_KIND

    return [int(v) for v in TREE_KIND]


def _rock_palette() -> list:
    from ..derive import geology
    return [list(c) for c in geology.PALETTE]


def _biome_palette() -> list:
    from ..derive.biomes import PALETTE
    return PALETTE.astype(int).tolist()


#: Ground colour per biome code (derive.biomes order) for the satellite
#: layer: what the class looks like from orbit rather than a legend colour.
#: Water classes are never used -- the water mask decides those.
SATELLITE_PALETTE = [
    [20, 50, 110],     # ocean (unused)
    [236, 240, 244],   # ice
    [146, 138, 112],   # tundra
    [46, 74, 46],      # boreal forest
    [142, 150, 84],    # temperate grassland
    [58, 96, 48],      # temperate forest
    [36, 82, 46],      # temperate rainforest
    [214, 190, 142],   # desert
    [150, 140, 92],    # shrubland
    [176, 160, 88],    # savanna
    [76, 116, 52],     # tropical seasonal forest
    [30, 86, 36],      # tropical rainforest
    [136, 126, 112],   # alpine
    [110, 100, 90],    # cliff
    [72, 110, 58],     # riparian
    [88, 116, 88],     # wetland
    [60, 110, 190],    # lake (unused)
]


# --------------------------------------------------------------------------
# extra formats: equirectangular PNGs, animated flat map
# --------------------------------------------------------------------------
def _ramp(t: np.ndarray, stops: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0) * (len(stops) - 1)
    k = np.minimum(t.astype(np.int64), len(stops) - 2)
    f = (t - k)[..., None]
    return stops[k] * (1 - f) + stops[k + 1] * f


LAKE_RGB = np.array([0.27, 0.51, 0.90])  # derive.biomes PALETTE[LAKE], as a fraction


def elevation_rgb(h: np.ndarray, land_top: float, ocean_bottom: float, water: np.ndarray | None = None) -> np.ndarray:
    """Hypsometric colours.  ``water`` (see :func:`water_code`) says which
    cells are sea and which are lake; without it the sea is ``h < 0``, which
    misses every lake and drowns every closed basin."""
    sea = _ramp(np.power(np.clip(-h / -ocean_bottom, 0, 1), 0.6), SEA)
    land = _ramp(np.power(np.clip(h / land_top, 0, 1), 0.75), LAND)
    if water is None:
        return np.where((h < 0)[..., None], sea, land)
    w = np.asarray(water)
    out = np.where((w == WATER_OCEAN)[..., None], sea, land)
    return np.where((w == WATER_LAKE)[..., None], LAKE_RGB, out)


@lru_cache(maxsize=4)
def _equirect_map(width: int, res: int, pad: int):
    """``(face, x, y)`` atlas-tile coordinates of every lat/lon pixel."""
    H, W = width // 2, width
    lat = (0.5 - (np.arange(H) + 0.5) / H) * np.pi
    lon = ((np.arange(W) + 0.5) / W - 0.5) * 2 * np.pi
    cl = np.cos(lat)[:, None]
    p = np.stack([cl * np.cos(lon)[None, :], cl * np.sin(lon)[None, :], np.broadcast_to(np.sin(lat)[:, None], (H, W))], -1)
    f, u, v = from_sphere_v(p)
    return f, u * res - 0.5 + pad, v * res - 0.5 + pad


def equirect(padded: np.ndarray, width: int, pad: int = PAD, nearest: bool = False) -> np.ndarray:
    """Sample a padded ``(6, T, T)`` face array on a ``width x width/2``
    lat/lon grid (pole on +Z, longitude 0 at +X, north at the top)."""
    res = padded.shape[1] - 2 * pad
    f, x, y = _equirect_map(width, res, pad)
    if nearest:
        return padded[f, np.clip(np.round(x).astype(np.int64), 0, res + 2 * pad - 1), np.clip(np.round(y).astype(np.int64), 0, res + 2 * pad - 1)]
    x0 = np.clip(np.floor(x).astype(np.int64), 0, res + 2 * pad - 2)
    y0 = np.clip(np.floor(y).astype(np.int64), 0, res + 2 * pad - 2)
    fx, fy = x - x0, y - y0
    a = padded.astype(np.float64)
    return ((a[f, x0, y0] * (1 - fx) + a[f, x0 + 1, y0] * fx) * (1 - fy)
            + (a[f, x0, y0 + 1] * (1 - fx) + a[f, x0 + 1, y0 + 1] * fx) * fy)


def relief(h: np.ndarray, R: float, land_top: float, ocean_bottom: float, exag: float = 12.0,
           river: np.ndarray | None = None, water: np.ndarray | None = None) -> np.ndarray:
    """Shaded relief of an equirect height map, lit as in the shader."""
    H, W = h.shape
    lat = (0.5 - (np.arange(H) + 0.5) / H) * np.pi
    dphi, dlam = np.pi / H, 2 * np.pi / W
    gx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) / (2 * R * dlam * np.maximum(np.cos(lat), 1e-3)[:, None])
    hp = np.pad(h, ((1, 1), (0, 0)), mode="edge")
    gy = (hp[:-2] - hp[2:]) / (2 * R * dphi)
    wet = (h < 0) if water is None else (np.asarray(water) > WATER_LAND)
    ex = np.where(wet, 0.35 * exag, exag)
    n = np.stack([-gx * ex, -gy * ex, np.ones_like(h)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    shade = np.clip(n @ LIGHT / LIGHT[2], 0, 2)
    col = elevation_rgb(h, land_top, ocean_bottom, water) * (0.15 + 0.85 * shade)[..., None]
    if river is not None:
        m = river & ~wet
        col[m] = col[m] * 0.35 + np.array([0.16, 0.40, 0.90]) * 0.65
    return (np.clip(col, 0, 1) ** 0.95 * 255).astype(np.uint8)


def export_extras(frames: list[_Frame], specs: dict, meta: dict, out: Path, fmts: set, log=print) -> None:
    from PIL import Image

    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    R, top, bot = meta["R_planet_m"], meta["land_top_m"], meta["ocean_bottom_m"]
    final = frames[-1]
    if "equirect" in fmts:
        W = min(8192, 4 * final.res)
        h = equirect(pad_faces(final.height), W)
        h0, h1 = float(np.floor(h.min())), float(np.ceil(h.max()))
        q = np.round((h - h0) / (h1 - h0) * 65535).astype(np.uint16)
        Image.fromarray(q).save(out / "equirect_height16.png")
        (out / "equirect_height16.json").write_text(json.dumps({"h0_m": h0, "h1_m": h1, "encoding": "h0 + v/65535*(h1-h0)",
                                                                 "projection": "equirectangular, pole +Z, lon 0 = +X, north up"}))
        river = _river_mask(final, specs, W)
        wat = equirect(pad_faces(final.ch["water"]), W, nearest=True) if "water" in final.ch else None
        Image.fromarray(relief(h, R, top, bot, river=river, water=wat)).save(out / "equirect_relief.png")
        for name, s in specs.items():
            if name in ("discharge", "flow", "satellite"):
                continue
            b = equirect(pad_faces(_byte(name, final.ch[name], specs)), W, nearest=True)
            if s["kind"] == "category":
                rgb = (np.array(meta["biome_palette"], np.uint8)[np.minimum(b, len(meta["biome_palette"]) - 1)]
                       if name == "biome" else _hash_rgb(b))
            else:
                rgb = (_ramp(b / 255.0, CMAPS.get(s["cmap"], CMAPS["viridis"])) * 255).astype(np.uint8)
            Image.fromarray(rgb).save(out / f"equirect_{name}.png")
        log(f"[viewer] equirect {W}x{W // 2} PNGs -> {out}")
    if "anim" in fmts:
        W = 1024
        imgs = []
        for fr in frames:
            h = equirect(pad_faces(fr.height), W)
            river = _river_mask(fr, specs, W)
            wat = equirect(pad_faces(fr.ch["water"]), W, nearest=True) if "water" in fr.ch else None
            imgs.append(Image.fromarray(relief(h, R, top, bot, river=river, water=wat)))
        dur = [100] * (len(imgs) - 1) + [2000]
        imgs[0].save(out / "timeline.webp", save_all=True, append_images=imgs[1:], duration=dur, loop=0, quality=80, method=4)
        log(f"[viewer] timeline.webp ({len(imgs)} frames) -> {out}")
    log(f"[viewer] extras in {time.time() - t0:.1f}s")


def _river_mask(fr: _Frame, specs: dict, W: int):
    """Equirect boolean river mask from the frame's overlay channel: hydro's
    flow accumulation on the final frame, erosion discharge before it."""
    for name in ("flow", "discharge"):
        if name in fr.ch and name in specs and "river_min" in specs[name]:
            return equirect(pad_faces(fr.ch[name].astype(np.float32)), W, nearest=True) > specs[name]["river_min"]
    return None


def _hash_rgb(b: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(7)
    pal = (rng.uniform(0.25, 0.9, (256, 3)) * 255).astype(np.uint8)
    pal[0] = 30
    return pal[b]
