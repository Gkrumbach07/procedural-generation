"""Detail tiles for the globe viewer's final frame (docs/viewer-rivers.md,
"Detail tiles").

The final frame is one atlas per texture, so its resolution is what a
browser texture holds: the refined grid at 2048^2 a face, the 1.2 km planet
level reduced from 8192^2 to 2048^2.  Detail tiles carry the terrain beyond
it, loaded only where the view is:

* levels ``L = 1, 2, ...`` at ``base_res x 2^L`` cells a face, up to the
  source's own resolution (block means of it below that);
* each level cut into ``TILE``-cell tiles per face (a row of tiles read at
  a time, so a 32768^2 face is never in memory), every tile one lossless
  RGBA WebP ``TILE + 2`` pixels wide and twice that tall (a cell of its
  neighbours around it, clamped at the face edge, so bilinear sampling needs
  no neighbour tile).  Its top half is the ground -- ``R, G`` = 16-bit height
  over the detail range (sea level on a code, every cell's sign kept, as
  :func:`viewer.encode_height`), ``B`` = the signed lake depth byte, ``A`` =
  255 - the smoothed ocean mask byte (land opaque, so a browser that
  premultiplies can only touch the sea) -- and its bottom half the water:
  ``R`` = the log byte of the water the rivers are drawn from (a planet
  level's accumulated flow, :func:`river_field`), so the stream map keeps
  the source's resolution at every zoom instead of the atlas' blur;
* written as ``tiles/L{L}/{face}_{ti}_{tj}.js`` (``GLOBE_VIEWER.tile(...)``:
  ``file://`` pages cannot fetch); a tile with no land and no lake in it is
  not written, and ``meta.detail`` carries a bitset of the ones that are.

Pixel ``(x, y) = (1 + i - ti TILE, 1 + j - tj TILE)`` of a tile is cell
``(i, j)`` of its face, as in the atlases.
"""
from __future__ import annotations

import base64
import io
import math
from pathlib import Path

import numpy as np
from scipy import ndimage

TILE = 256
WATER_LAND, WATER_LAKE, WATER_OCEAN = 0, 1, 2

#: rivers from a planet level's accumulated flow: a channel from this
#: percentile of land flow (98.5: ~800 km^2 of catchment at 1.2 km on
#: earth-v9 -- the 88th the erosion's discharge used draws 12 % of the land as
#: a hairline web), at full strength by the second, and widened up to
#: ``RIVER_RADIUS`` cells as it nears it (:func:`widen_rivers`)
RIVER_MIN_PCT, RIVER_FULL_PCT, RIVER_RADIUS = 98.5, 99.97, 3


# --------------------------------------------------------------------------
# sources: per face, at the source's resolution
# --------------------------------------------------------------------------
class RefinedSource:
    """The refined grid (``fine/``), from the arrays the final frame was built
    from (surface, water surface, water code at full resolution)."""

    def __init__(self, surf: np.ndarray, ws: np.ndarray | None, water: np.ndarray, discharge: np.ndarray | None = None):
        self.surf, self.ws, self.water, self.q = surf, ws, water, discharge
        self.res = int(surf.shape[1])

    def rows(self, f: int, r0: int, r1: int) -> dict:
        ws = self.ws[f, r0:r1] if self.ws is not None else self.surf[f, r0:r1]
        d = {"surf": np.asarray(self.surf[f, r0:r1], np.float32), "ws": np.asarray(ws, np.float32), "water": np.asarray(self.water[f, r0:r1])}
        if self.q is not None:
            d["discharge"] = np.asarray(self.q[f, r0:r1], np.float32)
        return d


class PlanetSource:
    """A planet zoom level's faces (``L{R}.f{k}.*.npy``), water by derive's
    rules as :func:`viewer._planet_final` draws them.  Its rivers are the
    level's ``flow`` (rain accumulated down the final surface,
    :func:`planet_finish.flow_face`) where the level has one, else the
    erosion's ``discharge``."""

    def __init__(self, pdir: Path, R: int, N: int, sea_near_c: np.ndarray, lake_min_depth: float, lake_min_cells: int):
        self.pdir, self.R, self.N = Path(pdir), int(R), int(N)
        self.sea_near_c, self.depth, self.min_cells = sea_near_c, float(lake_min_depth), int(lake_min_cells)
        self.res = self.N * self.R
        self.river = river_field(self.pdir, self.R)
        self.scale = river_scale(self.pdir, self.R) if self.river == "flow" else None

    def rows(self, f: int, r0: int, r1: int) -> dict:
        """Rows ``[r0, r1)`` of face ``f``.  A lake piece smaller than
        ``lake_min_cells`` is dropped as :func:`viewer._planet_final` drops
        it, judged over the rows plus a margin as wide as such a piece."""
        from ..derive import lakes as lakes_mod

        m = max(int(np.ceil(np.sqrt(self.min_cells))) + 2, RIVER_RADIUS)
        e0, e1 = max(r0 - m, 0), min(r1 + m, self.res)
        load = lambda name: np.load(self.pdir / f"L{self.R}.f{f}.{name}.npy", mmap_mode="r")[e0:e1]
        surf = np.asarray(load("height"), np.float32) + np.asarray(load("sediment"), np.float32)
        ws = np.maximum(np.asarray(load("water_surface"), np.float32), surf)
        water = np.zeros(surf.shape, np.uint8)
        c0 = e0 // self.R
        near = np.repeat(np.repeat(self.sea_near_c[f, c0:(e1 + self.R - 1) // self.R], self.R, axis=0), self.R, axis=1)[e0 - c0 * self.R:e1 - c0 * self.R]
        ocean = (surf < 0.0) & near
        water[ocean] = WATER_OCEAN
        lake = lakes_mod.kept_lake_mask(lakes_mod.lake_mask(surf, ws, self.depth, ocean=ocean), self.min_cells)
        water[lake] = WATER_LAKE
        c = slice(r0 - e0, r1 - e0)
        q = np.asarray(load(self.river), np.float32)
        if self.scale is not None:
            q = widen_rivers(q, self.scale)
        return {"surf": surf[c], "ws": ws[c], "water": water[c], "discharge": q[c]}


def river_field(pdir: Path, R: int) -> str:
    """``"flow"`` when every face of the planet level at ``pdir`` has its
    accumulated flow, else ``"discharge"``."""
    return "flow" if all((Path(pdir) / f"L{int(R)}.f{f}.flow.npy").exists() for f in range(6)) else "discharge"


def river_scale(pdir: Path, R: int, stride: int = 7) -> dict:
    """The byte scale of a planet level's flow over all six faces (every
    ``stride``-th cell of land): ``lo`` its median (byte 1), ``hi`` its
    maximum (byte 255), rivers from ``river_min`` to full at
    ``river_full``."""
    qs = []
    for f in range(6):
        q = np.asarray(np.load(Path(pdir) / f"L{int(R)}.f{f}.flow.npy", mmap_mode="r")[::stride, ::stride], np.float64)
        z = np.asarray(np.load(Path(pdir) / f"L{int(R)}.f{f}.height.npy", mmap_mode="r")[::stride, ::stride], np.float64)
        qs.append(q[(z > 0.0) & (q > 0.0)])
    q = np.concatenate(qs) if qs else np.array([1.0])
    lo = max(float(np.percentile(q, 50)), 1e-9)
    hi = max(float(q.max()), lo * 10.0)
    rmin = max(float(np.percentile(q, RIVER_MIN_PCT)), lo * 1.001)
    return {"lo": lo, "hi": hi, "river_min": rmin, "river_full": max(float(np.percentile(q, RIVER_FULL_PCT)), rmin * 1.001)}


def widen_rivers(q: np.ndarray, scale: dict, radius: int = RIVER_RADIUS) -> np.ndarray:
    """``q`` with each channel spread over a disc that grows with its
    strength ``s`` (0 at ``river_min``, 1 at ``river_full``, log): radius
    ``r`` where ``s >= (r - 0.5) / radius``, so a creek stays a cell wide and
    a trunk is ``2 radius + 1``, each ring a little weaker than the one inside
    it (soft banks).  Never lowers a cell; float32."""
    q = np.asarray(q, np.float32)
    qmin, qfull = float(scale["river_min"]), float(scale["river_full"])
    L = np.log(np.maximum(q, qmin) / qmin, dtype=np.float64)
    s = np.clip(L / math.log(qfull / qmin), 0.0, 1.0)
    out = q.copy()
    for r in range(1, int(radius) + 1):
        src = s >= (r - 0.5) / radius
        if not src.any():
            break
        ring = np.where(src, qmin * np.exp(L * 0.85 ** r), 0.0).astype(np.float32)   # the ring's strength a step down
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        out = np.maximum(out, ndimage.grey_dilation(ring, footprint=np.hypot(yy, xx) <= r + 0.01, mode="nearest"))
    return out


# --------------------------------------------------------------------------
# a face at a level
# --------------------------------------------------------------------------
def reduce_face(d: dict, k: int) -> dict:
    """Block ``k x k`` means (surface, water surface) of rows of a face; a
    block is ocean or lake where most of it is."""
    if k == 1:
        return d
    a_, b_ = d["surf"].shape[0] // k, d["surf"].shape[1] // k
    blk = lambda a: a[:a_ * k, :b_ * k].reshape(a_, k, b_, k)
    surf = blk(d["surf"]).mean(axis=(1, 3), dtype=np.float64).astype(np.float32)
    ws = blk(d["ws"]).mean(axis=(1, 3), dtype=np.float64).astype(np.float32)
    frac = lambda code: blk(d["water"] == code).mean(axis=(1, 3))
    out_q = {}
    if "discharge" in d:
        out_q["discharge"] = blk(d["discharge"]).max(axis=(1, 3)).astype(np.float32)   # a river narrower than a block still shows
    water = np.zeros((a_, b_), np.uint8)
    water[frac(WATER_LAKE) > 0.5] = WATER_LAKE
    water[frac(WATER_OCEAN) > 0.5] = WATER_OCEAN
    return {"surf": surf, "ws": np.maximum(ws, surf), "water": water, **out_q}


def face_lake_depth(surf: np.ndarray, ws: np.ndarray, water: np.ndarray, range_m: float) -> np.ndarray:
    """:func:`viewer.lake_depth` on one face (neighbours clamped at its edges)."""
    lake = water == WATER_LAKE
    level = np.where(lake, ws.astype(np.float64), -np.inf)
    near = ndimage.maximum_filter(level, size=3, mode="nearest")
    s = surf.astype(np.float64)
    d = np.where(lake, level - s, np.where(np.isfinite(near), near - s, -range_m))
    d[water == WATER_OCEAN] = -range_m
    return np.clip(d, -range_m, range_m).astype(np.float32)


def face_smooth_mask(mask: np.ndarray, passes: int = 2) -> np.ndarray:
    """:func:`viewer.smooth_mask` on one face (edges clamped)."""
    m = np.asarray(mask, bool)
    a = m.astype(np.float32)
    k = np.outer([1.0, 2.0, 1.0], [1.0, 2.0, 1.0]).astype(np.float32) / 16.0
    for _ in range(passes):
        a = ndimage.convolve(a, k, mode="nearest")
    return np.where(m, np.maximum(a, 0.6), np.minimum(a, 0.4)).astype(np.float32)


def height_grid(h_min: float, h_max: float) -> tuple[float, float]:
    """``(h0, h1)`` of a 16-bit height code over ``[h_min, h_max]`` with sea
    level on a code (:func:`viewer.encode_height`'s grid)."""
    h0, h1 = float(np.floor(h_min)), float(np.ceil(h_max))
    if h1 <= h0:
        h1 = h0 + 1.0
    if h0 < 0.0 < h1:
        step = (h1 - h0) / 65534.0
        z = math.ceil(-h0 / step)
        h0 = -z * step
        h1 = h0 + 65535.0 * step
    return h0, h1


def encode_height_on(h: np.ndarray, h0: float, h1: float) -> np.ndarray:
    """uint16 codes of metres ``h`` on the grid ``(h0, h1)``, a cell below 0
    never on or above sea level's code and one at or above never below it."""
    step = (h1 - h0) / 65535.0
    q = np.round((np.asarray(h, np.float64) - h0) / step)
    if h0 < 0.0 < h1:
        z = round(-h0 / step)
        q = np.where(h < 0.0, np.minimum(q, z - 1), np.maximum(q, z))
    return np.clip(q, 0, 65535).astype(np.uint16)


def tile_image(surf, ld, ocean_m, disch, ti: int, tj: int, h0: float, h1: float, lake_range: float, q_lo: float, q_hi: float,
               res: int | None = None, row0: int = 0) -> np.ndarray | None:
    """The ``(TILE + 2)`` x ``2 (TILE + 2)`` RGBA image of tile ``(ti, tj)`` of
    a face level ``res`` cells a side, from arrays holding its rows ``[row0,
    row0 + len)``: the ground on top, the water below.  None when the tile has
    no land and no lake."""
    n = int(res) if res is not None else surf.shape[0]
    ii = np.clip(np.arange(ti * TILE - 1, ti * TILE + TILE + 1), 0, n - 1) - row0
    jj = np.clip(np.arange(tj * TILE - 1, tj * TILE + TILE + 1), 0, n - 1)
    oc = ocean_m[np.ix_(ii, jj)]
    lk = ld[np.ix_(ii, jj)]
    if not ((oc < 0.5).any() or (lk > 0.0).any()):
        return None
    h = encode_height_on(surf[np.ix_(ii, jj)], h0, h1)
    b = np.clip(np.round(255.0 * (lk + lake_range) / (2.0 * lake_range)), 0, 255).astype(np.uint8)
    a = (255 - np.clip(np.round(255.0 * oc), 0, 255)).astype(np.uint8)
    ground = np.stack([(h >> 8).astype(np.uint8), (h & 255).astype(np.uint8), b, a], axis=-1)
    q = np.asarray(disch[np.ix_(ii, jj)], np.float64)
    t = np.log(np.maximum(q, q_lo) / q_lo) / math.log(max(q_hi / q_lo, 1.0000001))
    qb = np.where(q > q_lo, np.clip(np.round(255.0 * t), 1, 255), 0).astype(np.uint8)
    zero = np.zeros_like(qb)
    water = np.stack([qb, zero, zero, np.full_like(qb, 255)], axis=-1)
    img = np.concatenate([ground, water], axis=1)            # cell (i, j): ground at j, water at j + TILE + 2
    return np.ascontiguousarray(img.transpose(1, 0, 2))      # image (y, x)


def webp_rgba_b64(img: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img, "RGBA").save(buf, "WEBP", lossless=True, exact=True, quality=70, method=3)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def levels_for(base_res: int, src_res: int, max_levels: int = 4) -> list[tuple[int, int]]:
    """``[(L, res)]`` between the base frame and the source (powers of two)."""
    out, L = [], 1
    while L <= max_levels and base_res << L <= src_res and src_res % (base_res << L) == 0:
        out.append((L, base_res << L))
        L += 1
    return out


def export_tiles(out: Path, src, base_res: int, lake_range: float, log=print) -> dict | None:
    """Write every level's tiles under ``out / "tiles"``; returns
    ``meta.detail`` or None when the source is no finer than the base."""
    import shutil

    levels = levels_for(int(base_res), int(src.res))
    tdir = Path(out) / "tiles"
    shutil.rmtree(tdir, ignore_errors=True)
    if not levels:
        return None
    lo, hi = np.inf, -np.inf
    step = 4 * TILE
    qs = []
    scale = getattr(src, "scale", None)
    for f in range(6):
        for r0 in range(0, src.res, step):
            d = src.rows(f, r0, min(src.res, r0 + step))
            s = d["surf"]
            lo, hi = min(lo, float(s.min())), max(hi, float(s.max()))
            q = d.get("discharge")
            if q is not None and scale is None:
                qq = np.asarray(q)[(np.asarray(s) > 0) & (np.asarray(q) > 0)]
                if qq.size:
                    qs.append(qq[::7])
    h0, h1 = height_grid(lo, hi)
    if scale is None:
        # the discharge byte: the land's own range, so the rivers at every
        # level are the same stream map the erosion left
        qall = np.concatenate(qs) if qs else np.array([1.0])
        q_lo = max(float(np.percentile(qall, 50)), 1e-9)
        scale = {"lo": q_lo, "hi": max(float(qall.max()), q_lo * 10.0),
                 "river_min": max(float(np.percentile(qall, 88)), q_lo), "river_full": max(float(np.percentile(qall, 99.5)), q_lo)}
    q_lo, q_hi = scale["lo"], scale["hi"]
    byte = lambda q: int(round(255.0 * math.log(max(q, q_lo) / q_lo) / math.log(q_hi / q_lo)))
    meta = {"tile": TILE, "pad": 1, "h0": h0, "h1": h1, "lake_range": float(lake_range), "q_lo": q_lo, "q_hi": q_hi,
            "river_min_byte": int(np.clip(byte(scale["river_min"]), 1, 254)),
            "river_span_byte": max(byte(scale["river_full"]) - byte(scale["river_min"]), 8),
            "levels": []}
    for L, res in levels:
        nT = -(-res // TILE)
        (tdir / f"L{L}").mkdir(parents=True, exist_ok=True)
        meta["levels"].append({"L": L, "res": res, "nT": nT})
    written = 0
    # a row of tiles at a time, with the rows the lake-shore and ocean-mask
    # filters reach beyond it (3 cells): a face is never read whole
    MARGIN = 4
    for f in range(6):
        for li, (L, res) in enumerate(levels):
            k = src.res // res
            nT = meta["levels"][li]["nT"]
            bits = meta["levels"][li].setdefault("_bits", np.zeros(6 * nT * nT, np.bool_))
            for ti in range(nT):
                lr0, lr1 = max(ti * TILE - MARGIN, 0), min((ti + 1) * TILE + MARGIN, res)
                d = reduce_face(src.rows(f, lr0 * k, lr1 * k), k)
                ld = face_lake_depth(d["surf"], d["ws"], d["water"], lake_range)
                om = face_smooth_mask(d["water"] == WATER_OCEAN)
                q = d.get("discharge", np.zeros_like(d["surf"]))
                for tj in range(nT):
                    img = tile_image(d["surf"], ld, om, q, ti, tj, h0, h1, lake_range, meta["q_lo"], meta["q_hi"], res=res, row0=lr0)
                    if img is None:
                        continue
                    key = f"{L}_{f}_{ti}_{tj}"
                    (tdir / f"L{L}" / f"{f}_{ti}_{tj}.js").write_text('GLOBE_VIEWER.tile("%s","%s");\n' % (key, webp_rgba_b64(img)))
                    bits[(f * nT + ti) * nT + tj] = True
                    written += 1
                del d, ld, om
        log(f"[viewer] detail tiles: face {f} done ({written:,} so far)")
    for lv in meta["levels"]:
        bits = lv.pop("_bits")
        lv["tiles"] = int(bits.sum())
        lv["exists"] = base64.b64encode(np.packbits(bits, bitorder="little").tobytes()).decode("ascii")
    return meta


__all__ = ["TILE", "RIVER_MIN_PCT", "RIVER_FULL_PCT", "RIVER_RADIUS", "RefinedSource", "PlanetSource", "river_field", "river_scale",
           "widen_rivers", "reduce_face", "face_lake_depth", "face_smooth_mask", "height_grid",
           "encode_height_on", "tile_image", "levels_for", "export_tiles"]
