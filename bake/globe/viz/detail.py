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
  ``R`` = the byte of the water the rivers are drawn from (a planet level's
  accumulated flow as a channel to find the line of, :func:`river_strength`;
  the erosion's discharge where there is no routed flow), so the stream map
  keeps the source's resolution at every zoom instead of the atlas' blur;
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
#: a hairline web), at full strength by the second
RIVER_MIN_PCT, RIVER_FULL_PCT = 98.5, 99.97
#: ...and drawn as lines (:func:`river_strength`): a channel cell is worth
#: ``RIVER_BASE`` at the threshold to 1 at full strength, and its path is
#: laid down as a ridge ``RIVER_SIGMA`` cells wide, after each cell of it has
#: been moved towards its neighbours along the path ``RIVER_PASSES`` times.
#: A routed path steps cell to cell, and the line the viewer finds on a
#: staircase's ridge is a row of hooks until its corners are rounded.  Rounded
#: by a blur wide enough to do it (1.3 cells), two channels within two or
#: three cells of each other are one ridge between them: a creek beside its
#: river, or coming in to it at a shallow angle, lost its line cells short of
#: the junction (earth-v32 at 1.2 km: a third of a view's channel cells had no
#: line).  Rounded along the path, the ridge can be as narrow as the cells
#: allow.  ``RIVER_REACH`` is how far a cell's ridge reaches, moved and
#: spread; the weakest channel is byte ``RIVER_MIN_BYTE`` + a quarter of
#: ``RIVER_SPAN_BYTE`` of the river channel, the strongest the top of it
RIVER_BASE, RIVER_SIGMA, RIVER_PASSES, RIVER_REACH = 0.4, 0.8, 2, 6
RIVER_MIN_BYTE, RIVER_SPAN_BYTE = 56, 170
#: a zoom window's rivers are widened up to this many cells instead (:func:`widen_rivers`)
RIVER_RADIUS = 3
#: the sediment byte of a tile's or zoom level's water half: log over this
#: range in metres, 0 at or below the low end
SED_LO_M, SED_HI_M = 0.5, 2000.0


def log_byte(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """0 at or below ``lo``; ``255 ln(x / lo) / ln(hi / lo)`` above, from 1."""
    x = np.asarray(x, np.float64)
    t = np.log(np.maximum(x, lo) / lo) / math.log(max(hi / lo, 1.0000001))
    return np.where(x > lo, np.clip(np.round(255.0 * t), 1, 255), 0).astype(np.uint8)


# --------------------------------------------------------------------------
# sources: per face, at the source's resolution
# --------------------------------------------------------------------------
class RefinedSource:
    """The refined grid (``fine/``), from the arrays the final frame was built
    from (surface, water surface, water code at full resolution)."""

    def __init__(self, surf: np.ndarray, ws: np.ndarray | None, water: np.ndarray, discharge: np.ndarray | None = None,
                 scale: dict | None = None):
        self.surf, self.ws, self.water, self.q = surf, ws, water, discharge
        self.res = int(surf.shape[1])
        self.scale = scale                # the river channel's byte scale when it is not the erosion's discharge (strength_scale)

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

        m = max(int(np.ceil(np.sqrt(self.min_cells))) + 2, RIVER_REACH)
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
        raw = q
        if self.scale is not None:
            q = river_strength(q, self.scale)
        sed = np.asarray(load("sediment"), np.float32)
        return {"surf": surf[c], "ws": ws[c], "water": water[c], "discharge": q[c], "flow_raw": raw[c], "sediment": sed[c]}


def river_field(pdir: Path, R: int) -> str:
    """``"flow"`` when every face of the planet level at ``pdir`` has its
    accumulated flow, else ``"discharge"``."""
    return "flow" if all((Path(pdir) / f"L{int(R)}.f{f}.flow.npy").exists() for f in range(6)) else "discharge"


def strength_scale(lo: float, hi: float, q_min: float, q_full: float) -> dict:
    """The byte scale of a river channel drawn as lines (:func:`river_strength`):
    the flow's own log scale -- ``lo`` byte 1, ``hi`` byte 255, so the channel
    sits beside the flow layer on one scale -- with ``river_min`` and
    ``river_full`` at bytes ``RIVER_MIN_BYTE`` and ``RIVER_MIN_BYTE +
    RIVER_SPAN_BYTE``, and the flow a channel starts at and is full strength
    by (``q_min``, ``q_full``).  ``lines`` tells the viewer every channel in it
    is a line and none a band."""
    lo = max(float(lo), 1e-9)
    hi = max(float(hi), lo * 10.0)
    at = lambda b: lo * (hi / lo) ** (b / 255.0)
    q_min = max(float(q_min), 1e-9)
    return {"lo": lo, "hi": hi, "river_min": at(RIVER_MIN_BYTE), "river_full": at(RIVER_MIN_BYTE + RIVER_SPAN_BYTE),
            "q_min": q_min, "q_full": max(float(q_full), q_min * 1.001), "lines": True}


def river_scale(pdir: Path, R: int, stride: int = 7) -> dict:
    """The scale of a planet level's rivers (:func:`strength_scale`) over all
    six faces (every ``stride``-th cell of land): ``lo`` the flow's median,
    ``hi`` its maximum, a channel from the ``RIVER_MIN_PCT`` percentile of
    land flow to full strength at ``RIVER_FULL_PCT``."""
    qs = []
    for f in range(6):
        q = np.asarray(np.load(Path(pdir) / f"L{int(R)}.f{f}.flow.npy", mmap_mode="r")[::stride, ::stride], np.float64)
        z = np.asarray(np.load(Path(pdir) / f"L{int(R)}.f{f}.height.npy", mmap_mode="r")[::stride, ::stride], np.float64)
        qs.append(q[(z > 0.0) & (q > 0.0)])
    q = np.concatenate(qs) if qs else np.array([1.0])
    lo = max(float(np.percentile(q, 50)), 1e-9)
    return strength_scale(lo, float(q.max()), max(float(np.percentile(q, RIVER_MIN_PCT)), lo * 1.001), float(np.percentile(q, RIVER_FULL_PCT)))


def channel_value(q: np.ndarray, scale: dict) -> np.ndarray:
    """What each cell of the flow ``q`` is worth as a river: 0 below
    ``scale["q_min"]``, ``RIVER_BASE`` there, 1 at ``q_full`` (log); float64."""
    q = np.asarray(q, np.float64)
    qmin, qfull = float(scale["q_min"]), float(scale["q_full"])
    s = np.clip(np.log(np.maximum(q, qmin) / qmin) / math.log(qfull / qmin), 0.0, 1.0)
    return np.where(q >= qmin, RIVER_BASE + (1.0 - RIVER_BASE) * s, 0.0)


_D8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def path_ridges(q: np.ndarray, v: np.ndarray, R: int = 1, sigma: float = RIVER_SIGMA, passes: int = RIVER_PASSES) -> np.ndarray:
    """The channels of the flow ``q`` as ridges, ``R`` times finer than its
    cells (float64, 0..1): ``v`` is each cell's strength, 0 off the channels
    (:func:`channel_value`).

    Every channel cell is a point of its path, joined to the channel cell
    beside it that carries the least more water (downstream) and the one that
    carries the most less (up the main stem).  The points are moved towards
    those neighbours ``passes`` times -- a quarter of the way to each -- which
    takes the steps out of the path *along* it; and every stretch from a
    point to the next downstream is laid down as a Gaussian ridge ``sigma``
    fine cells wide, as high on its line as the stretch is strong.  Smoothing
    across the path instead (a blur) has to be wide to round the steps, and
    then cannot tell two channels two cells apart.

    Hand it ``RIVER_REACH`` cells beyond the rows wanted: a point is moved by
    what lies up to ``passes`` cells along its path, and its ridge reaches
    ``3 sigma``."""
    q = np.asarray(q, np.float64)
    v = np.asarray(v, np.float64)
    n0, n1 = q.shape
    R = int(R)
    out = np.zeros((n0 * R, n1 * R), np.float64)
    ii, jj = np.nonzero(v > 0.0)
    K = ii.size
    if K == 0:
        return out
    idx = np.full(q.shape, -1, np.int64)
    idx[ii, jj] = np.arange(K)
    qc = q[ii, jj]
    nxt, prv = np.arange(K), np.arange(K)
    more, less = np.full(K, np.inf), np.full(K, -np.inf)
    for a, b in _D8:
        i2, j2 = np.clip(ii + a, 0, n0 - 1), np.clip(jj + b, 0, n1 - 1)
        k2 = np.where((i2 == ii + a) & (j2 == jj + b), idx[i2, j2], -1)
        q2 = q[i2, j2]
        down = (k2 >= 0) & (q2 > qc) & (q2 < more)
        nxt[down], more[down] = k2[down], q2[down]
        up = (k2 >= 0) & (q2 < qc) & (q2 > less)
        prv[up], less[up] = k2[up], q2[up]
    P = np.stack([ii, jj], axis=1).astype(np.float64)
    for _ in range(int(passes)):
        P = 0.25 * P[prv] + 0.5 * P + 0.25 * P[nxt]
    seg = P[nxt] - P
    L = np.hypot(seg[:, 0], seg[:, 1])
    vs = v[ii, jj]
    vn = vs[nxt]
    norm = 1.0 / (float(sigma) * math.sqrt(2.0 * math.pi))   # a long straight path of strength 1 is 1 on its line
    rad = int(math.ceil(3.0 * float(sigma)))
    shape = (n0 * R, n1 * R)
    for k in range(R):                                       # a stretch is R samples, about one a fine cell
        t = (k + 0.5) / R
        X = (P + t * seg) * R + 0.5 * (R - 1)                # fine cells, centres at integers
        w = ((1.0 - t) * vs + t * vn) * (L * R / R) * norm
        if k == 0:                                           # a path's last cell (the sea, a lake, the rows' edge): a point of its own
            end = L == 0.0
            w = np.where(end, vs * norm, w)
            X = np.where(end[:, None], P * R + 0.5 * (R - 1), X)
        c0 = np.round(X).astype(np.int64)
        for a in range(-rad, rad + 1):
            for b in range(-rad, rad + 1):
                y, x = c0[:, 0] + a, c0[:, 1] + b
                ok = (w > 0.0) & (y >= 0) & (y < shape[0]) & (x >= 0) & (x < shape[1])
                g = w * np.exp(-((y - X[:, 0]) ** 2 + (x - X[:, 1]) ** 2) / (2.0 * float(sigma) ** 2))
                np.add.at(out, (y[ok], x[ok]), g[ok])
    return np.minimum(out, 1.0)


def river_strength(q: np.ndarray, scale: dict, R: int = 1) -> np.ndarray:
    """The river channel of the flow ``q``, ``R`` times finer than its cells
    (float32, as flow on ``scale``): a routed river is a path one cell wide
    and the same water that fills the lakes -- it runs into each at its shore
    and out at its spill -- so it is drawn as a line and not as the cells it
    crosses.  Each channel's path as a smooth ridge as high as the channel is
    strong (:func:`channel_value`, :func:`path_ridges`), for the viewer to
    find the line of (viewer.html ``riverRidge``): a strongest river's line
    is byte ``RIVER_MIN_BYTE + RIVER_SPAN_BYTE`` of ``scale``, a weakest
    ``RIVER_BASE`` of that, the land between byte 0."""
    b = path_ridges(q, channel_value(q, scale), R)
    lo, hi = float(scale["lo"]), float(scale["hi"])
    top = (RIVER_MIN_BYTE + RIVER_SPAN_BYTE) / 255.0
    return (lo * (hi / lo) ** (b * top)).astype(np.float32)


def widen_rivers(q: np.ndarray, scale: dict, radius: int = RIVER_RADIUS) -> np.ndarray:
    """(The zoom windows' rivers, :mod:`globe.viz.zoomtex`: down to 5 m a cell a
    river is many cells wide and is drawn as the water it is; the planet's are
    lines, :func:`river_strength`.)

    ``q`` with each channel spread over a disc that grows with its
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
    if "flow_raw" in d:
        out_q["flow_raw"] = blk(d["flow_raw"]).max(axis=(1, 3)).astype(np.float32)
    if "sediment" in d:
        out_q["sediment"] = blk(d["sediment"]).mean(axis=(1, 3), dtype=np.float64).astype(np.float32)
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
               res: int | None = None, row0: int = 0, flow=None, sediment=None) -> np.ndarray | None:
    """The ``(TILE + 2)`` x ``2 (TILE + 2)`` RGBA image of tile ``(ti, tj)`` of
    a face level ``res`` cells a side, from arrays holding its rows ``[row0,
    row0 + len)``: the ground on top, the water below -- R the river channel's
    byte, G the flow itself on the same scale (the flow layer), B the sediment
    byte (:data:`SED_LO_M`).  None when the tile has no land and no lake."""
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
    qb = log_byte(disch[np.ix_(ii, jj)], q_lo, q_hi)
    zero = np.zeros_like(qb)
    fb = zero if flow is None else log_byte(flow[np.ix_(ii, jj)], q_lo, q_hi)
    sb = zero if sediment is None else log_byte(sediment[np.ix_(ii, jj)], SED_LO_M, SED_HI_M)
    water = np.stack([qb, fb, sb, np.full_like(qb, 255)], axis=-1)
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
            "water_gb": True, "sed_lo": SED_LO_M, "sed_hi": SED_HI_M,
            "river_min_byte": int(np.clip(byte(scale["river_min"]), 1, 254)),
            "river_span_byte": max(byte(scale["river_full"]) - byte(scale["river_min"]), 8),
            "river_lines": bool(scale.get("lines")),          # every channel a line (river_strength): the page draws no bands
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
                # a lake is drawn at its water, not its bed: encoding the bed hillshaded the
                # bottom like dry ground and left a hole with a painted floor in the 3-D view
                d["surf"] = np.where(d["water"] == WATER_LAKE, np.maximum(d["ws"], d["surf"]), d["surf"])
                om = face_smooth_mask(d["water"] == WATER_OCEAN)
                q = d.get("discharge", np.zeros_like(d["surf"]))
                for tj in range(nT):
                    img = tile_image(d["surf"], ld, om, q, ti, tj, h0, h1, lake_range, meta["q_lo"], meta["q_hi"], res=res, row0=lr0,
                                     flow=d.get("flow_raw"), sediment=d.get("sediment"))
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


__all__ = ["TILE", "RIVER_MIN_PCT", "RIVER_FULL_PCT", "RIVER_BASE", "RIVER_SIGMA", "RIVER_PASSES", "RIVER_REACH", "RIVER_MIN_BYTE", "RIVER_SPAN_BYTE", "SED_LO_M", "SED_HI_M", "log_byte", "RefinedSource", "PlanetSource", "river_field", "river_scale",
           "strength_scale", "channel_value", "path_ridges", "river_strength", "widen_rivers", "RIVER_RADIUS", "reduce_face", "face_lake_depth", "face_smooth_mask", "height_grid",
           "encode_height_on", "tile_image", "levels_for", "export_tiles"]
