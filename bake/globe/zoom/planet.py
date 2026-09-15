"""The whole planet at a zoom level's resolution: every cube face cut into
tiles and eroded the way a zoom level is, straight from the planet
(docs/zoom-windows.md, "The planet at 1.2 km").

    python scripts/planet_bake.py --world worlds/earth-v9            # R = 8: 1.2 km on the earth preset

Per face, a work raster covering the face and ``guard`` coarse cells beyond
its edges (``<out>/work.f{k}.*.npy`` memmaps: height, sediment, discharge,
momentum, done).  Tiles are square windows of ``tile`` fine cells plus
``margin`` on every side, aligned to the coarse grid, run in passes whose
windows do not overlap (:func:`bake.tile_passes`), each pass across worker
processes that read and write the memmaps directly.  A tile:

* upsamples the planet over its own window (:func:`refine.upsample.upsample_window`,
  point-wise, so overlapping tiles agree) and adds detail noise hashed from
  the fine cell's face coordinates (:func:`hashed_ridged`), so overlapping
  tiles add the same noise;
* starts from the memmaps where an earlier pass wrote, from the upsample
  elsewhere;
* takes the water of everything upstream where the planet's drainage
  crosses its window edge, across cube edges too (:func:`planet_inflow`);
* erodes as a zoom tile does (:func:`bake.erode_tile`: the zoom erosion
  profile, held to the upsample as it goes) and writes back as one
  (:func:`bake.write_tile`: new cells in the core and the inner half of the
  margin, cells an earlier pass wrote cross-faded).

Tiles whose window has no land on the face are skipped.  A tile's result
depends only on its window and what earlier passes wrote, so the raster
does not depend on the number of workers.  After the tiles
(:func:`finish_face`): the two faces' solutions of each cube-edge strip are
blended (:func:`blend_face_seams`), the water surface is a flood of each
face capped by the planet's lakes, and ``L{R}.f{k}.{height,sediment,
discharge,water_surface}.npy`` (the face itself) are written.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from ..config import WorldParams
from ..cubesphere import from_sphere_v, project_to_face_v, to_sphere_v
from ..hydro.priority_flood import priority_flood_flat
from ..io.world_store import WorldStore
from ..refine import basin_job as bj
from ..refine.upsample import Window, upsample_window
from ..refine.zoom import ZOOM_REFINE, zoom_params
from . import bake as zb

#: rng / hash sub-key of planet bakes
PLANET_KEY = 7_700_101
WORK_FIELDS = {"height": ((), np.float32), "sediment": ((), np.float32), "discharge": ((), np.float32),
               "momentum": ((2,), np.float32), "done": ((), np.bool_)}


@dataclass(frozen=True)
class PlanetLevel:
    R: int = 8
    iterations: int = 200
    tile: int = 1024  # core side, fine cells (a multiple of R)
    margin: int = 64  # fine cells, a multiple of R
    hold_every: int = 10
    hold_scale: float = 4.0
    seam_cells: int = 16  # fine cells either side of a cube edge blended between the two faces

    @property
    def guard(self) -> int:
        """Coarse cells of work raster beyond each face edge: a tile's margin
        and the cell of halo its window samples."""
        return self.margin // self.R + 1

    def zoom_level(self) -> zb.ZoomLevel:
        return zb.ZoomLevel(self.R, 0, self.iterations, tile=self.tile, margin=self.margin, hold_every=self.hold_every, hold_scale=self.hold_scale)


# --------------------------------------------------------------------------
# noise that overlapping tiles agree on
# --------------------------------------------------------------------------
_M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _mix(x: np.ndarray) -> np.ndarray:
    """splitmix64 finaliser (uint64 arrays, wrapping)."""
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def _lattice(seed: int, face: int, octave: int, ki: np.ndarray, kj: np.ndarray) -> np.ndarray:
    """Value in [-1, 1) at integer lattice points, a pure function of its
    arguments."""
    with np.errstate(over="ignore"):
        k = np.uint64((int(seed) * 0x9E3779B97F4A7C15 + int(face) * 0xC2B2AE3D27D4EB4F + int(octave) * 0x165667B19E3779F9) & 0xFFFFFFFFFFFFFFFF)
        x = _mix(ki.astype(np.int64).view(np.uint64) * np.uint64(0xD6E8FEB86659FD93) ^ k)
        x = _mix(x ^ kj.astype(np.int64).view(np.uint64) * np.uint64(0xA0761D6478BD642F))
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53) * 2.0 - 1.0


def _value_noise(seed: int, face: int, octave: int, gi: np.ndarray, gj: np.ndarray, wavelength: float) -> np.ndarray:
    x = (gi + 0.5) / wavelength
    y = (gj + 0.5) / wavelength
    i0 = np.floor(x).astype(np.int64)
    j0 = np.floor(y).astype(np.int64)
    tx = x - i0
    ty = y - j0
    tx = tx * tx * (3 - 2 * tx)
    ty = ty * ty * (3 - 2 * ty)
    I0, J0 = i0[:, None], j0[None, :]
    a = _lattice(seed, face, octave, I0, J0)
    b = _lattice(seed, face, octave, I0 + 1, J0)
    c = _lattice(seed, face, octave, I0, J0 + 1)
    d = _lattice(seed, face, octave, I0 + 1, J0 + 1)
    TX, TY = tx[:, None], ty[None, :]
    return a * (1 - TX) * (1 - TY) + b * TX * (1 - TY) + c * (1 - TX) * TY + d * TX * TY


_RIDGED_MEAN: dict[tuple, float] = {}


def hashed_ridged(seed: int, face: int, gi0: int, gj0: int, n0: int, n1: int, base_wavelength: float, min_wavelength: float = 2.0, gain: float = 0.5) -> np.ndarray:
    """:func:`refine.upsample.ridged_fbm` over the fine cells ``[gi0, gi0 +
    n0) x [gj0, gj0 + n1)`` of a face (indices may run beyond the face),
    from lattice values hashed from their coordinates, so two windows
    overlapping on a cell draw the same value there.  Normalised with a fixed
    mean (measured once on a large patch) instead of the array's own, for the
    same reason: ``(sum - mean) / (1 - mean)``, at most 1."""
    gi = np.arange(gi0, gi0 + n0, dtype=np.float64)
    gj = np.arange(gj0, gj0 + n1, dtype=np.float64)
    out = np.zeros((n0, n1), np.float64)
    w, lam, total, octave = 1.0, float(base_wavelength), 0.0, 0
    while lam >= min_wavelength:
        n = _value_noise(seed, face, octave, gi, gj, lam)
        r = 1.0 - np.abs(n)
        out += w * r * r
        total += w
        w *= gain
        lam *= 0.5
        octave += 1
    out /= max(total, 1e-12)
    key = (float(base_wavelength), float(min_wavelength), float(gain))
    if key not in _RIDGED_MEAN:
        big = 64 * max(int(base_wavelength), 1)
        _RIDGED_MEAN[key] = float(hashed_ridged_raw(987654321, big, base_wavelength, min_wavelength, gain).mean())
    mu = _RIDGED_MEAN[key]
    return ((out - mu) / max(1.0 - mu, 1e-6)).astype(np.float32)


def hashed_ridged_raw(seed: int, n: int, base_wavelength: float, min_wavelength: float, gain: float) -> np.ndarray:
    g = np.arange(n, dtype=np.float64)
    out = np.zeros((n, n))
    w, lam, total, octave = 1.0, float(base_wavelength), 0.0, 0
    while lam >= min_wavelength:
        r = 1.0 - np.abs(_value_noise(seed, 0, octave, g, g, lam))
        out += w * r * r
        total += w
        w *= gain
        lam *= 0.5
        octave += 1
    return out / max(total, 1e-12)


# --------------------------------------------------------------------------
# inflow across a window edge, across cube edges too
# --------------------------------------------------------------------------
_FLOW: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def planet_flow(root: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Hydro's ``flow_dir`` (u8, 255 = ocean) and ``flow_acc`` (precip
    volume) of every face, ``(6, N, N)``, loaded once per process."""
    key = str(Path(root).resolve())
    if key not in _FLOW:
        fd = np.stack([np.load(Path(root) / "coarse" / f"flow_dir.f{f}.npy") for f in range(6)]).astype(np.int64)
        fa = np.stack([np.load(Path(root) / "coarse" / f"flow_acc.f{f}.npy") for f in range(6)]).astype(np.float64)
        _FLOW.clear()
        _FLOW[key] = (fd, fa)
    return _FLOW[key]


def planet_inflow(fd: np.ndarray, fa: np.ndarray, face: int, a0: int, a1: int, b0: int, b1: int, R: int, surface: np.ndarray) -> np.ndarray:
    """Spawn weight (``surface.shape``: the fine window ``[a0, a1) x [b0,
    b1)`` coarse plus a one-cell ring) of the water the planet's drainage
    carries into the window: every coarse cell in the ring around it --
    looked up on whichever face owns it -- whose D8 receiver lies inside
    sends its whole accumulation, entering at the lowest cell by the
    crossing."""
    N = fd.shape[1]
    top = [(a0 - 1, j) for j in range(b0 - 1, b1 + 1)]
    bot = [(a1, j) for j in range(b0 - 1, b1 + 1)]
    lef = [(i, b0 - 1) for i in range(a0, a1)]
    rig = [(i, b1) for i in range(a0, a1)]
    ij = np.array(top + bot + lef + rig, dtype=np.float64)
    ii, jj = ij[:, 0], ij[:, 1]
    p = to_sphere_v(np.full(ii.shape, face), (ii + 0.5) / N, (jj + 0.5) / N)
    f2, u2, v2 = from_sphere_v(p)
    i2 = np.minimum((u2 * N).astype(np.int64), N - 1)
    j2 = np.minimum((v2 * N).astype(np.int64), N - 1)
    code = fd[f2, i2, j2]
    ok = code < 8
    k = np.minimum(code, 7)
    # the receiver in the owning face's own indices; off that face's grid
    # when the donor drains across its edge -- exactly the cross-face case,
    # so its centre is placed on the face's gnomonic extension, not dropped
    ri, rj = i2 + zb.D8[k, 0], j2 + zb.D8[k, 1]
    pr = to_sphere_v(f2, (ri + 0.5) / N, (rj + 0.5) / N)
    u, v = project_to_face_v(np.full(ii.shape, face), pr)
    ci, cj = u * N, v * N
    ok &= (np.floor(ci) >= a0) & (np.floor(ci) < a1) & (np.floor(cj) >= b0) & (np.floor(cj) < b1)
    ok &= np.sum(pr * p, axis=-1) > 0.0
    mi = ((ii + 0.5) + ci) / 2.0
    mj = ((jj + 0.5) + cj) / 2.0
    e_i = (mi - a0) * R + 0.5
    e_j = (mj - b0) * R + 0.5
    return zb.spawn_at(e_i[ok], e_j[ok], fa[f2[ok], i2[ok], j2[ok]], surface, max(R // 2, 1))


# --------------------------------------------------------------------------
# the work rasters
# --------------------------------------------------------------------------
def work_path(out: Path, face: int, name: str) -> Path:
    return Path(out) / f"work.f{face}.{name}.npy"


def open_work(out: Path, face: int, name: str, NF: int, mode: str = "r+") -> np.ndarray:
    extra, dtype = WORK_FIELDS[name]
    path = work_path(out, face, name)
    if mode == "w+" or not path.exists():
        return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(NF, NF) + extra)
    return np.load(path, mmap_mode=mode)


def planet_tile(root: str, params: WorldParams, level: PlanetLevel, out: str, face: int, ta0: int, tb0: int, cc: int) -> dict:
    """Erode the tile whose core is coarse ``[ta0 + mc, ta0 + mc + cc)``
    (window ``[ta0, ta0 + cc + 2 mc)``, ``mc = margin / R``) of ``face``,
    reading and writing the face's work memmaps."""
    t0 = time.time()
    R = level.R
    mc = level.margin // R
    nc = cc + 2 * mc
    ta1, tb1 = ta0 + nc, tb0 + nc
    lp = zoom_params(params, R)
    grid, fields, derived = bj.coarse_inputs(root, lp)
    N = grid.N
    win = Window(face, ta0, ta1, tb0, tb1, R)
    up = upsample_window(fields, derived, win, grid)
    t = slice(R - 1, win.NE - (R - 1))                          # the window plus one fine cell
    tr = {k: v[t, t] for k, v in up.items()}
    n = tr["height0"].shape[0]
    plain = (tr["height0"] + tr["sediment0"]).astype(np.float64)
    ocean = zb._ocean(plain, tr["basin_id"] < 0)
    coast_taper = float(lp.refine.coast_taper_m)
    amp = float(ZOOM_REFINE["detail_amp"]) * np.minimum(np.maximum(tr["slope"], 0.0) * grid.cell_size_m, np.maximum(tr["relief"], 0.0)) * (0.5 + 0.5 * np.clip(tr["hardness"], 0.0, 1.0))
    if coast_taper > 0.0:
        tt = np.clip(np.abs(plain) / coast_taper, 0.0, 1.0)
        amp = amp * tt * tt * (3.0 - 2.0 * tt)
    noise = np.where(ocean, 0.0, amp * hashed_ridged(int(params.world.seed) + PLANET_KEY, face, ta0 * R - 1, tb0 * R - 1, n, n, 2.0 * R))
    NF = (N + 2 * level.guard) * R
    mm = {k: open_work(Path(out), face, k, NF) for k in WORK_FIELDS}
    o = level.guard * R
    sl = (slice(ta0 * R - 1 + o, ta1 * R + 1 + o), slice(tb0 * R - 1 + o, tb1 * R + 1 + o))
    done = np.array(mm["done"][sl])
    cur = {
        "height": np.where(done, mm["height"][sl], tr["height0"] + noise),
        "sediment": np.where(done, mm["sediment"][sl], tr["sediment0"]),
        "discharge": np.where(done, mm["discharge"][sl], tr["discharge"]),
        "momentum": np.where(done[..., None], mm["momentum"][sl], tr["momentum"]),
    }
    inwin = np.zeros((n, n), bool)
    inwin[1:-1, 1:-1] = True
    land = inwin & ~ocean
    frozen = land & ndimage.binary_dilation(ocean & inwin, structure=np.ones((3, 3), bool))
    active = land & ~frozen
    stats = {"face": face, "window": [ta0, tb0, nc], "active_cells": int(active.sum())}
    if not active.any():
        return stats
    fd, fa = planet_flow(root)
    src = planet_inflow(fd, fa, face, ta0, ta1, tb0, tb1, R, plain)
    job = {"a": ta0 + mc, "b": tb0 + mc, "c": cc, "sl": sl, "land": land, "active": active, "inwin": inwin, "ocean": ocean, "src": src, "f": R,
           "arrays": {"height": cur["height"], "sediment": cur["sediment"], "discharge": cur["discharge"], "momentum": cur["momentum"],
                      "hardness": tr["hardness"], "precip": np.where(ocean, 0.0, np.maximum(tr["precip"], 0.0)), "evap": tr["evap"],
                      "metric": tr["metric"], "metric_inv": tr["metric_inv"], "plain": plain}}
    zl = level.zoom_level()
    res = zb.erode_tile(lp, zl, R, job, (PLANET_KEY, face, ta0, tb0))
    stats.update(res["stats"])
    sea = inwin & ocean & ~done                            # write_tile marks the window's sea done: give it the upsample
    for k, v in (("height", tr["height0"]), ("sediment", tr["sediment0"]), ("discharge", tr["discharge"])):
        view = mm[k][sl]
        view[sea] = v[sea]
    stats["blended_cells"] = zb.write_tile(zl, mm, mm["done"], job, res)
    for m in mm.values():
        m.flush()
    stats["seconds_total"] = round(time.time() - t0, 1)
    return stats


def _tile_job(args):
    return planet_tile(*args)


def face_tiles(level: PlanetLevel, N: int, land_c: np.ndarray, face: int) -> list[list[tuple[int, int, int]]]:
    """Passes of ``(ta0, tb0, cc)`` window origins (coarse) covering the
    face, tiles with no land in their on-face window left out."""
    R = level.R
    cc = level.tile // R
    mc = level.margin // R
    starts, c = zb.tile_starts(0, N, cc)
    windows = zb.tile_windows(starts, c, mc)
    passes = zb.tile_passes(windows, c, mc)
    out = []
    for ps in passes:
        keep = []
        for ti, tj, a, b in ps:
            a0, a1 = max(a - mc, 0), min(a + c + mc, N)
            b0, b1 = max(b - mc, 0), min(b + c + mc, N)
            if land_c[face, a0:a1, b0:b1].any():
                keep.append((a - mc, b - mc, c))
        if keep:
            out.append(keep)
    return out


def run_planet(root: str | Path, level: PlanetLevel = PlanetLevel(), out: str | Path | None = None, faces=None, workers: int = 0, log=None,
               finish: bool = True) -> Path:
    """Erode every face of the world at ``root`` at ``level.R`` into
    ``out`` (default ``<root>/zoom/planet_R{R}``).  Resumable: finished
    passes are recorded in ``progress.json`` with the level they ran with."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    root = Path(root)
    params = WorldParams.from_dict(WorldStore(root).manifest["params"])
    grid = params.coarse_grid()
    N, R = grid.N, level.R
    out = Path(out) if out is not None else root / "zoom" / f"planet_R{R}"
    out.mkdir(parents=True, exist_ok=True)
    prog_path = out / "progress.json"
    prog = json.loads(prog_path.read_text()) if prog_path.exists() else {}
    if prog.get("level") != asdict(level):
        prog = {"level": asdict(level), "passes": {}, "tiles": []}
    NF = (N + 2 * level.guard) * R
    bid = np.stack([np.load(root / "coarse" / f"basin_id.f{f}.npy") for f in range(6)])
    land_c = bid >= 0
    faces = list(range(6)) if faces is None else [int(f) for f in faces]
    t0 = time.time()
    for face in faces:
        passes = face_tiles(level, N, land_c, face)
        started = any(k.split(":")[0] == str(face) for k in prog["passes"])
        if not (started and all(work_path(out, face, k).exists() for k in WORK_FIELDS)):
            for k in WORK_FIELDS:
                open_work(out, face, k, NF, mode="w+").flush()
        widest = max((len(ps) for ps in passes), default=1)
        n_workers, threads = zb.pool_size(widest, workers)
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn"), initializer=zb._pool_init, initargs=(threads,)) as ex:
            for p_i, ps in enumerate(passes):
                key = f"{face}:{p_i}"
                if prog["passes"].get(key):
                    continue
                tp = time.time()
                args = [(str(root), params, level, str(out), face, ta0, tb0, cc) for ta0, tb0, cc in ps]
                stats = list(ex.map(_tile_job, args))
                prog["tiles"].extend(stats)
                prog["passes"][key] = {"tiles": len(ps), "seconds": round(time.time() - tp, 1)}
                prog_path.write_text(json.dumps(prog))
                if log is not None:
                    cells = sum(s.get("active_cells", 0) for s in stats)
                    log(f"face {face} pass {p_i + 1}/{len(passes)}: {len(ps)} tiles, {cells / 1e6:.1f} M cells, {time.time() - tp:.0f}s "
                        f"({n_workers} workers x {threads} threads; {time.time() - t0:.0f}s so far)")
    if finish and len(faces) == 6:
        # three faces at a time: a face's flood holds ~2 GB at R = 8
        n_workers, threads = zb.pool_size(6, 3)
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn"), initializer=zb._pool_init, initargs=(threads,)) as ex:
            fin = list(ex.map(_finish_job, [(root, out, level, f) for f in range(6)]))
        ql = quicklook(out, R)
        info = {"level": asdict(level), "world": root.name, "cell_m": grid.cell_size_m / R, "faces": fin,
                "tiles": len(prog["tiles"]), "seconds_tiles": round(sum(v["seconds"] for v in prog["passes"].values()), 1),
                "active_cells": int(sum(t_.get("active_cells", 0) for t_ in prog["tiles"])), "quicklook": ql.name}
        (out / "planet.json").write_text(json.dumps(info, indent=1))
        if log is not None:
            log(f"finished: {sum(f_['lake_cells'] for f_ in fin):,} lake cells, seams {sum(f_['seam_cells'] for f_ in fin):,} cells; {ql}")
    return out


def _finish_job(args):
    return finish_face(*args)


# --------------------------------------------------------------------------
# after the tiles
# --------------------------------------------------------------------------
OUT_FIELDS = ("height", "sediment", "discharge", "water_surface")


def out_path(out: Path, R: int, face: int, name: str) -> Path:
    return Path(out) / f"L{R}.f{face}.{name}.npy"


def _bilinear(a: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n0, n1 = a.shape[:2]
    x = np.clip(x, 0.0, n0 - 1.001)
    y = np.clip(y, 0.0, n1 - 1.001)
    i, j = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = x - i, y - j
    return (a[i, j] * (1 - fx) * (1 - fy) + a[i + 1, j] * fx * (1 - fy) + a[i, j + 1] * (1 - fx) * fy + a[i + 1, j + 1] * fx * fy)


def _neighbour_across(face: int, side: int, N: int) -> int:
    """The face across edge ``side`` (0: i = 0, 1: i = N, 2: j = 0, 3: j = N)."""
    u, v = {0: (-0.01, 0.5), 1: (1.01, 0.5), 2: (0.5, -0.01), 3: (0.5, 1.01)}[side]
    f2, _, _ = from_sphere_v(to_sphere_v(np.array([face]), np.array([u]), np.array([v])))
    return int(f2[0])


def blend_face_seams(out: Path, level: PlanetLevel, N: int, face: int, arrays: dict) -> int:
    """Blend a face's cells within ``seam_cells`` of each cube edge with the
    neighbour face's solution of the same ground (its work raster reaches
    ``margin / 2`` fine cells beyond its edge): weights
    ``smoothstep((dist + F) / 2F)`` for the face's own value and one minus
    that for the neighbour's, ``dist`` the cell centre's distance from the
    edge in cells, so the two faces meet at 1/2 each on the edge and the
    surface is continuous through it.  Discharge takes whichever face has the
    larger weight (a river is a position, not a quantity to average).
    ``arrays`` (the face's on-face height / sediment / discharge) are updated
    in place; the neighbours are read from their untouched work rasters.
    Returns the cells blended."""
    R, F = level.R, int(level.seam_cells)
    n = N * R
    g = level.guard * R
    NF = n + 2 * g
    blended = 0
    e = np.arange(F)
    for side in range(4):
        nb = _neighbour_across(face, side, N)
        if nb == face:
            continue
        if side == 0:
            I, J = np.meshgrid(e, np.arange(n), indexing="ij")
        elif side == 1:
            I, J = np.meshgrid(n - 1 - e, np.arange(n), indexing="ij")
        elif side == 2:
            I, J = np.meshgrid(np.arange(n), e, indexing="ij")
        else:
            I, J = np.meshgrid(np.arange(n), n - 1 - e, indexing="ij")
        dist = {0: I + 0.5, 1: n - I - 0.5, 2: J + 0.5, 3: n - J - 0.5}[side].astype(np.float64)
        p = to_sphere_v(np.full(I.shape, face), (I + 0.5) / n, (J + 0.5) / n)
        u, v = project_to_face_v(np.full(I.shape, nb), p)
        x = u * n - 0.5 + g
        y = v * n - 0.5 + g
        inside = (x >= 0) & (x <= NF - 2) & (y >= 0) & (y <= NF - 2)
        wn = {k: open_work(out, nb, k, NF, mode="r") for k in ("height", "sediment", "discharge", "done")}
        done_n = np.asarray(wn["done"])[np.clip(np.rint(x).astype(np.int64), 0, NF - 1), np.clip(np.rint(y).astype(np.int64), 0, NF - 1)] & inside
        t = np.clip((dist + F) / (2.0 * F), 0.0, 1.0)
        w = np.where(done_n, t * t * (3.0 - 2.0 * t), 1.0)
        for k in ("height", "sediment"):
            other = _bilinear(np.asarray(wn[k]), x, y)
            arrays[k][I, J] = np.where(done_n, w * arrays[k][I, J] + (1.0 - w) * other, arrays[k][I, J]).astype(np.float32)
        q_other = _bilinear(np.asarray(wn["discharge"]), x, y)
        arrays["discharge"][I, J] = np.where(done_n & (w < 0.5), q_other, arrays["discharge"][I, J]).astype(np.float32)
        blended += int(done_n.sum())
        del wn
    return blended


def finish_face(root: Path, out: Path, level: PlanetLevel, face: int, strip: int = 64) -> dict:
    """The face's outputs from its work raster: the on-face height,
    sediment and discharge (cells no tile wrote -- the open ocean -- are the
    plain upsample), the seams blended, and the water surface: a priority
    flood of the face draining to its border and the sea, no higher than the
    planet's own water surface where the planet has a lake."""
    from ..refine.upsample import upsample_face

    t0 = time.time()
    params = WorldParams.from_dict(WorldStore(root).manifest["params"])
    lp = zoom_params(params, level.R)
    grid, fields, derived = bj.coarse_inputs(root, lp)
    N, R = grid.N, level.R
    n = N * R
    g = level.guard * R
    NF = n + 2 * g
    wk = {k: open_work(out, face, k, NF, mode="r") for k in WORK_FIELDS}
    on = (slice(g, g + n), slice(g, g + n))
    arrays = {k: np.empty((n, n), np.float32) for k in OUT_FIELDS}
    plain_ws = np.empty((n, n), np.float32)
    for i0 in range(0, N, strip):
        i1 = min(N, i0 + strip)
        up = upsample_face(fields, derived, face, R, i0, i1)
        rs = slice(i0 * R, i1 * R)
        done = np.asarray(wk["done"][on][rs])
        for k in ("height", "sediment", "discharge"):
            arrays[k][rs] = np.where(done, wk[k][on][rs], up[k])
        plain_ws[rs] = up["water_surface"]
    seams = blend_face_seams(out, level, N, face, arrays)
    surf = arrays["height"] + arrays["sediment"]
    ocean_c = np.load(Path(root) / "coarse" / f"flow_dir.f{face}.npy") == 255
    sea_near = np.repeat(np.repeat(ndimage.binary_dilation(ocean_c, structure=np.ones((3, 3), bool)), R, 0), R, 1)
    ocean = (surf < 0.0) & sea_near
    drain = ocean.copy()
    drain[0, :] = drain[-1, :] = drain[:, 0] = drain[:, -1] = True
    fr = priority_flood_flat(surf, drain, None)
    ws = np.maximum(fr.filled.reshape(surf.shape), surf)
    del fr
    lake_c = plain_ws > surf + 1e-3                     # the planet's lakes, upsampled
    ws = np.where(lake_c, np.maximum(surf, np.minimum(ws, plain_ws)), ws)
    ws = np.where(ocean, 0.0, ws)
    arrays["water_surface"] = ws.astype(np.float32)
    for k, a in arrays.items():
        np.save(out_path(out, R, face, k), a)
    lake = (ws - surf > float(params.hydro.lake_min_depth)) & ~ocean
    return {"face": face, "seam_cells": seams, "land_cells": int((~ocean).sum()), "lake_cells": int(lake.sum()), "seconds": round(time.time() - t0, 1)}


def quicklook(out: Path, R: int, size: int = 1024) -> Path:
    """The six faces as a cross (hillshade, discharge in blue, water) at
    ``size`` cells per face."""
    from PIL import Image

    tiles = {}
    for f in range(6):
        h = np.load(out_path(out, R, f, "height"), mmap_mode="r")
        n = h.shape[0]
        k = max(1, n // size)
        m = (n // k) * k

        def red(name, how="mean"):
            a = np.asarray(np.load(out_path(out, R, f, name), mmap_mode="r")[:m, :m], np.float32).reshape(m // k, k, m // k, k)
            return a.max(axis=(1, 3)) if how == "max" else a.mean(axis=(1, 3))

        surf = red("height") + red("sediment")
        q = red("discharge", "max")
        ws = red("water_surface")
        gy, gx = np.gradient(surf, 1.0)
        shade = np.clip(1.0 - (gx + gy) * 0.002, 0.3, 1.5) / 1.5
        t = np.clip(surf / 4000.0, 0, 1)[..., None]
        img = (np.array([0.35, 0.5, 0.3]) * (1 - t) + np.array([0.8, 0.75, 0.65]) * t) * shade[..., None]
        lq = np.log10(np.maximum(q, 1e-6))
        land = surf >= 0
        lo, hi = (np.percentile(lq[land], [90, 99.8]) if land.any() else (0.0, 1.0))
        r = np.clip((lq - lo) / max(hi - lo, 1e-6), 0, 1)[..., None]
        img = img * (1 - r) + np.array([0.15, 0.35, 0.8]) * r
        img[ws > surf + 1.0] = [0.2, 0.4, 0.75]
        img[surf < 0] = [0.05, 0.12, 0.3]
        tiles[f] = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    s = tiles[0].shape[0]
    canvas = np.zeros((2 * s, 3 * s, 3), np.uint8)
    for f in range(6):
        canvas[(f // 3) * s:(f // 3 + 1) * s, (f % 3) * s:(f % 3 + 1) * s] = tiles[f]
    path = Path(out) / f"L{R}.png"
    Image.fromarray(canvas).save(path)
    return path


__all__ = ["PlanetLevel", "hashed_ridged", "planet_flow", "planet_inflow", "open_work", "planet_tile", "face_tiles", "run_planet", "blend_face_seams", "finish_face", "quicklook", "out_path"]
