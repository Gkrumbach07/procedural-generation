"""Tiles stage (PLAN.md section 10.3): rasterise the fine fields into
``(T+1)²`` vertex tiles and build the 2x LOD pyramid.

Inputs (``fine/<name>.f{0..5}.npy``, docs/DEVELOPING.md; missing = zeros):
``height, sediment, water_surface, discharge, hardness`` (f32),
``basin_id`` (i32), ``biome, vegetation, river_mask`` (u8).

Per-tile channels (see ``globe.io.tiles`` for the files)::

    height.png  terrain surface = height + sediment (metres; the sediment
                depth is in layers.R so bedrock = surface - sediment)
    water.png   lake surface: water_surface where it lies > 0.05 m above the
                surface and > 0, else 0 (ocean is implicit, sea level 0)
    layers.png  R sediment / LAYER_SEDIMENT_MAX, G hardness, B biome, A vegetation
    flow.png    R encode_discharge, G basin-local id = index into
                meta["basins"] (255 = ocean / not listed), B river mask
    meta.json   "basins": basin ids present (-1 excluded, most frequent
                first, at most 255 entries), "neighbors": [lod, face, x, y] of the tiles across
                sides +i, -i, +j, -j (cube edges crossed via cubesphere)

Vertex values: LOD 0 reduces the fine cells around each fine-cell corner
(2×2; 2+2 across a face edge, 3 at a cube corner): mean for the float
channels (then quantised), max for the lake surface and river mask, mode
for ids and biome, exact mean for vegetation.  LOD ``l+1`` reduces the 3×3
LOD-``l`` vertices around LOD-``l`` vertex ``(2k, 2m)``: height by picking
the centre (so LOD ``l+1`` heights *are* strided LOD-``l`` heights), water
and river mask by max, basin id and biome by mode, the other byte channels
by mean.  Reductions are permutation invariant and evaluated with
cross-face rings (``globe.refine.lod``), so a vertex shared by two tiles —
also across a cube edge — has bit-identical values in both.

Memory: fine faces are memmapped and processed one face and one channel at
a time; lattices go to ``tiles/_work`` memmaps (removed when the stage
finishes) so no more than a couple of fine-face-sized arrays are resident.

Also writes ``tiles/index.json`` for the Godot loader.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ..io.tiles import LAYER_SEDIMENT_MAX, Tile, encode_discharge, read_tile, tile_exists, tiles_per_face, write_tile
from . import lod as lodmod
from .lod import edge_links, side_row

OUTPUTS = ["tiles"]

LAKE_MIN_DEPTH_M = 0.05
NO_BASIN = 255  # flow.G for ocean vertices or basins beyond the 255 listed

#: fine fields: dtype (missing files read as zeros of this dtype)
FINE_FIELDS = {
    "height": np.float32,
    "sediment": np.float32,
    "water_surface": np.float32,
    "discharge": np.float32,
    "hardness": np.float32,
    "basin_id": np.int32,
    "biome": np.uint8,
    "vegetation": np.uint8,
    "river_mask": np.uint8,
}


@dataclass(frozen=True)
class Channel:
    name: str
    inputs: tuple[str, ...]
    transform: Callable[[dict], np.ndarray]  # cell values (elementwise) from the inputs
    lod0: str  # LOD-0 reduction of the 2x2 cells (mean/max/mode)
    encode: Callable[[np.ndarray], np.ndarray]  # lattice -> stored dtype
    pyramid: str  # reduction LOD l -> l+1 (stride/max/mode/mean)


def _surface(d):
    return d["height"] + d["sediment"]


def _lake(d):
    ws = d["water_surface"]
    surf = d["height"] + d["sediment"]
    return np.where((ws - surf > LAKE_MIN_DEPTH_M) & (ws > 0.0), ws, np.float32(0.0)).astype(np.float32)


def _u8(scale):
    return lambda v: np.clip(np.rint(v.astype(np.float64) * scale), 0, 255).astype(np.uint8)


CHANNELS = (
    Channel("height", ("height", "sediment"), _surface, "mean", lambda v: v.astype(np.float32), "stride"),
    Channel("water", ("height", "sediment", "water_surface"), _lake, "max", lambda v: v.astype(np.float32), "max"),
    Channel("sediment", ("sediment",), lambda d: d["sediment"], "mean", _u8(255.0 / LAYER_SEDIMENT_MAX), "mean"),
    Channel("hardness", ("hardness",), lambda d: d["hardness"], "mean", _u8(255.0), "mean"),
    Channel("discharge", ("discharge",), lambda d: d["discharge"], "mean", encode_discharge, "mean"),
    Channel("basin", ("basin_id",), lambda d: d["basin_id"], "mode", lambda v: v.astype(np.int32), "mode"),
    Channel("biome", ("biome",), lambda d: d["biome"], "mode", lambda v: v.astype(np.uint8), "mode"),
    Channel("vegetation", ("vegetation",), lambda d: d["vegetation"], "mean", lambda v: v.astype(np.uint8), "mean"),
    Channel("river", ("river_mask",), lambda d: d["river_mask"], "max", lambda v: v.astype(np.uint8), "max"),
)
CHANNEL_BY_NAME = {c.name: c for c in CHANNELS}


# --------------------------------------------------------------------------
# fine inputs
# --------------------------------------------------------------------------
class FineCells:
    """Memmapped access to ``fine/<name>.f{k}.npy`` (zeros when missing)."""

    def __init__(self, root: Path, N: int):
        self.dir = Path(root) / "fine"
        self.N = int(N)
        self._open: dict[tuple[str, int], np.ndarray | None] = {}

    def has(self, name: str) -> bool:
        return all((self.dir / f"{name}.f{k}.npy").exists() for k in range(6))

    def face(self, name: str, f: int) -> np.ndarray:
        key = (name, f)
        if key not in self._open:
            p = self.dir / f"{name}.f{f}.npy"
            if p.exists():
                a = np.load(p, mmap_mode="r")
                if a.shape != (self.N, self.N):
                    raise ValueError(f"{p}: shape {a.shape} != ({self.N}, {self.N})")
                self._open[key] = a
            else:
                self._open[key] = None
        a = self._open[key]
        return np.zeros((self.N, self.N), FINE_FIELDS[name]) if a is None else a

    def row(self, name: str, f: int, side: int, depth: int = 0) -> np.ndarray:
        if not (self.dir / f"{name}.f{f}.npy").exists():
            return np.zeros(self.N, FINE_FIELDS[name])
        return side_row(self.face(name, f), side, depth)

    def close(self) -> None:
        self._open.clear()


def channel_getter(cells: FineCells, ch: Channel):
    """``getter(face, side=None, depth=0)`` of a channel's transformed cell
    values, for :func:`lod.lod0_lattice`."""

    def getter(f, side=None, depth=0):
        if side is None:
            d = {n: np.asarray(cells.face(n, f)) for n in ch.inputs}
        else:
            d = {n: cells.row(n, f, side, depth) for n in ch.inputs}
        return ch.transform(d)

    return getter


# --------------------------------------------------------------------------
# lattice store (work memmaps)
# --------------------------------------------------------------------------
class LatticeStore:
    """``tiles/_work/<channel>.L<lod>.f<face>.npy`` memmaps."""

    def __init__(self, work: Path):
        self.work = Path(work)
        self.work.mkdir(parents=True, exist_ok=True)

    def path(self, chan: str, lod: int, f: int) -> Path:
        return self.work / f"{chan}.L{lod}.f{f}.npy"

    def save(self, chan: str, lod: int, f: int, arr: np.ndarray) -> None:
        out = np.lib.format.open_memmap(self.path(chan, lod, f), mode="w+", dtype=arr.dtype, shape=arr.shape)
        out[...] = arr
        out.flush()
        del out

    def load(self, chan: str, lod: int, f: int) -> np.ndarray:
        return np.load(self.path(chan, lod, f), mmap_mode="r")

    def getter(self, chan: str, lod: int):
        def g(f, side=None, depth=0):
            a = self.load(chan, lod, f)
            return a if side is None else side_row(a, side, depth)

        return g

    def remove(self) -> None:
        shutil.rmtree(self.work, ignore_errors=True)


def build_lattices(store, params, work: Path | None = None, log=print, channels=CHANNELS) -> LatticeStore:
    """Compute every channel's lattice at every LOD for all faces into a
    :class:`LatticeStore` (``tiles/_work`` by default)."""
    N, L = params.N_fine, params.max_lod
    links = edge_links(N)
    ls = LatticeStore(work or (store.tiles_dir / "_work"))
    cells = FineCells(store.root, N)
    missing = [n for n in FINE_FIELDS if not cells.has(n)]
    if missing:
        log(f"[tiles] fine fields missing (zeros): {', '.join(missing)}")
    for ch in channels:
        getter = channel_getter(cells, ch)
        for f in range(6):
            ls.save(ch.name, 0, f, ch.encode(lodmod.lod0_lattice(getter, f, ch.lod0, links)))
        for lod in range(L):
            g = ls.getter(ch.name, lod)
            for f in range(6):
                ls.save(ch.name, lod + 1, f, lodmod.next_lod(g, f, ch.pyramid, links))
        log(f"[tiles] lattice {ch.name}: LOD 0..{L}")
    cells.close()
    return ls


# --------------------------------------------------------------------------
# tiles
# --------------------------------------------------------------------------
def basin_local_ids(ids: np.ndarray) -> tuple[list[int], np.ndarray]:
    """Basin list (present ids, -1 excluded, most frequent first, ties by
    id; capped at ``NO_BASIN`` = 255 entries so index 255 never names a
    basin) and the uint8 index map (``NO_BASIN`` for ocean and for basins
    beyond the first 255)."""
    vals, inverse = np.unique(ids, return_inverse=True)
    counts = np.bincount(inverse.ravel(), minlength=len(vals))
    keep = vals >= 0
    order = np.lexsort((vals, -counts))
    order = order[keep[order]]
    basins = [int(v) for v in vals[order[:NO_BASIN]]]
    table = np.full(len(vals), NO_BASIN, dtype=np.uint8)
    table[order[:NO_BASIN]] = np.arange(len(basins), dtype=np.uint8)
    return basins, table[inverse].reshape(ids.shape)


def basin_lookup(basins) -> np.ndarray:
    """256-entry ``flow.G -> global basin id`` table (``-1`` for
    ``NO_BASIN`` and for every unused index)."""
    basins = list(basins)[:NO_BASIN]
    table = np.full(256, -1, dtype=np.int64)
    table[: len(basins)] = basins
    return table


def make_tile(ls, lod: int, f: int, x: int, y: int, T: int, n: int, links) -> Tile:
    """``ls`` is a :class:`LatticeStore` or a ``{channel: lattice}`` dict
    of this (lod, face)'s lattices (opened once per job by the writer)."""
    sl = (slice(x * T, x * T + T + 1), slice(y * T, y * T + T + 1))
    lat = ls if isinstance(ls, dict) else {c.name: ls.load(c.name, lod, f) for c in CHANNELS}
    ch = {c.name: np.ascontiguousarray(lat[c.name][sl]) for c in CHANNELS}
    layers = np.stack([ch["sediment"], ch["hardness"], ch["biome"], ch["vegetation"]], axis=-1)
    basins, local = basin_local_ids(ch["basin"])
    flow = np.stack([ch["discharge"], local, ch["river"]], axis=-1)
    meta = {"basins": basins, "neighbors": lodmod.tile_neighbors(lod, f, x, y, n, links)}
    return Tile(lod, f, x, y, ch["height"], ch["water"], layers, flow, meta)


def write_index(store, params) -> Path:
    N, T, L = params.N_fine, params.world.T, params.max_lod
    index = {
        "lods": [{"lod": l, "tiles_per_face": tiles_per_face(N, T, l), "size": T + 1} for l in range(L + 1)],
        "T": T,
        "N_fine": N,
        "R_planet": params.R_planet,
        "max_lod": L,
        "faces": 6,
        "sides": list(lodmod.SIDES),
        "vertex_samples": True,
    }
    return store.write_json("tiles/index.json", index)


def _write_tiles_job(args) -> int:
    """One (lod, face) worth of tiles (multiprocessing target; the face's
    lattice memmaps are opened once from disk, outputs are independent
    files)."""
    root, work, lod, f, T, n, N = args
    ls = LatticeStore(work)
    lat = {c.name: ls.load(c.name, lod, f) for c in CHANNELS}
    links = edge_links(N)
    for x in range(n):
        for y in range(n):
            write_tile(root, make_tile(lat, lod, f, x, y, T, n, links))
    return n * n


def write_all_tiles(store, params, ls: LatticeStore, log=print, workers: int | None = None) -> int:
    """Write every tile of every LOD from the lattice store; ``workers``
    processes (default ``params.workers()``; the output does not depend on
    the scheduling)."""
    N, T, L = params.N_fine, params.world.T, params.max_lod
    workers = params.workers() if workers is None else int(workers)
    jobs = [(store.root, ls.work, lod, f, T, tiles_per_face(N, T, lod), N) for lod in range(L + 1) for f in range(6)]
    n_written = 0
    if workers > 1 and len(jobs) > 1:
        import multiprocessing as mp

        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
        with ctx.Pool(min(workers, len(jobs))) as pool:
            for k in pool.imap_unordered(_write_tiles_job, jobs):
                n_written += k
    else:
        for job in jobs:
            n_written += _write_tiles_job(job)
    for lod in range(L + 1):
        n = tiles_per_face(N, T, lod)
        log(f"[tiles] LOD {lod}: {6 * n * n} tiles ({n}x{n} per face)")
    return n_written


def run(store, params, log=print) -> dict:
    import time

    t0 = time.time()
    ls = build_lattices(store, params, log=log)
    t1 = time.time()
    log(f"[tiles] lattices in {t1 - t0:.1f}s")
    n_written = write_all_tiles(store, params, ls, log=log)
    ls.remove()
    write_index(store, params)
    t2 = time.time()
    log(f"[tiles] {n_written} tiles written in {t2 - t1:.1f}s ({params.workers()} workers)")
    return {"tiles": n_written, "lods": params.max_lod + 1, "seconds_lattices": round(t1 - t0, 2), "seconds_tiles": round(t2 - t1, 2)}


# --------------------------------------------------------------------------
# reading back / quicklook
# --------------------------------------------------------------------------
def face_mosaic(root, lod: int, face: int, N_fine: int, T: int) -> dict[str, np.ndarray]:
    """Stitch a face's tiles at ``lod`` back into ``(n*T + 1)²`` lattices
    (shared columns taken once): height, water, layers, flow, basin (global
    ids, -1 = none)."""
    n = tiles_per_face(N_fine, T, lod)
    M = n * T + 1
    out = {
        "height": np.zeros((M, M), np.float32),
        "water": np.zeros((M, M), np.float32),
        "layers": np.zeros((M, M, 4), np.uint8),
        "flow": np.zeros((M, M, 3), np.uint8),
        "basin": np.full((M, M), -1, np.int32),
    }
    for x in range(n):
        for y in range(n):
            t = read_tile(root, lod, face, x, y)
            sl = (slice(x * T, x * T + T + 1), slice(y * T, y * T + T + 1))
            out["height"][sl] = t.height
            out["water"][sl] = t.water
            out["layers"][sl] = t.layers
            out["flow"][sl] = t.flow
            out["basin"][sl] = basin_lookup(t.meta.get("basins", []))[t.flow[..., 1]]
    return out


def quicklook(store, params, path) -> Path:
    """Unfolded net of the coarsest LOD whose face mosaic is <= 1024 px:
    hillshaded surface, lakes (water.png), rivers (flow.B / flow.R) and the
    tile grid; a second image ``<path>_basins.png`` colours basin ids
    reconstructed from flow.G + meta["basins"]."""
    from ..viz import quicklook as ql

    N, T, L = params.N_fine, params.world.T, params.max_lod
    lod = L
    for l in range(L + 1):
        if tiles_per_face(N, T, l) * T <= 1024:
            lod = l
            break
    if not tile_exists(store.root, lod, 0, 0, 0):
        return None
    mos = [face_mosaic(store.root, lod, f, N, T) for f in range(6)]
    h = np.stack([m["height"][:-1, :-1] for m in mos])
    w = np.stack([m["water"][:-1, :-1] for m in mos])
    fl = np.stack([m["flow"][:-1, :-1] for m in mos])
    bid = np.stack([m["basin"][:-1, :-1] for m in mos])
    cell = params.fine_cell_size_m * (1 << lod)
    img = ql.render_height(h, 0.0, cell)
    img = ql.overlay(img, w > 0, (60, 120, 220), 0.9)
    q = fl[..., 0].astype(np.float64) / 255.0
    thr = float(np.percentile(q, 97)) if q.max() > 0 else 1.0
    wgt = np.clip((q - thr) / max(q.max() - thr, 1e-9), 0, 1)
    img = ql.overlay(img, wgt > 0, (30, 80, 255), 0.95, weight=0.35 + 0.65 * wgt)
    img = ql.overlay(img, fl[..., 2] > 0, (20, 60, 255), 0.9)
    grid = np.zeros(h.shape, dtype=bool)
    grid[:, ::T, :] = True
    grid[:, :, ::T] = True
    img = ql.overlay(img, grid, (255, 255, 255), 0.35)
    out = ql.save_image(path, img)
    path = Path(path)
    ql.save_image(path.with_name(path.stem + "_basins.png"), ql.contour_lines(ql.render_labels(bid, seed=1), bid, (255, 255, 255)))
    return out


__all__ = ["OUTPUTS", "run", "quicklook", "CHANNELS", "Channel", "FineCells", "LatticeStore", "build_lattices", "write_all_tiles", "make_tile", "basin_local_ids", "basin_lookup", "face_mosaic", "write_index", "NO_BASIN", "LAKE_MIN_DEPTH_M"]
