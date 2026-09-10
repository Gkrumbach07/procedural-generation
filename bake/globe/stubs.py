"""Phase-0 stub implementations: every stage writes plausible noise with the
correct field names, shapes and dtypes so the pipeline and on-disk layout
can be exercised end to end.  Real stages replace the ``run`` in their own
module; the stubs stay here for tests of the runner."""
from __future__ import annotations

import numpy as np

from .config import WorldParams
from .field import FaceField, rotation_field
from .io.world_store import WorldStore


def fbm_at(p: np.ndarray, rng: np.random.Generator, octaves: int = 5, base_freq: float = 2.0) -> np.ndarray:
    """Smooth value noise sampled at arbitrary unit vectors ``p`` (..., 3).

    Sum of octaves of random 3-D lattices, trilinearly interpolated. Because
    the lattice lives in space rather than on a face, the result is
    continuous across cube edges by construction -- which is what makes it
    usable for anything that has to look right on the sphere rather than on
    the net.
    """
    out = np.zeros(p.shape[:-1], dtype=np.float64)
    amp = 1.0
    total = 0.0
    for o in range(octaves):
        f = base_freq * (2**o)
        n = int(np.ceil(f)) + 2
        lattice = rng.random((n + 1, n + 1, n + 1))
        q = (p + 1.0) * 0.5 * f  # [0, f]
        i0 = np.floor(q).astype(np.int64)
        t = q - i0
        t = t * t * (3 - 2 * t)
        i0 = np.clip(i0, 0, n - 1)
        acc = np.zeros_like(out)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = (t[..., 0] if dx else 1 - t[..., 0]) * (t[..., 1] if dy else 1 - t[..., 1]) * (t[..., 2] if dz else 1 - t[..., 2])
                    acc += w * lattice[i0[..., 0] + dx, i0[..., 1] + dy, i0[..., 2] + dz]
        out += amp * (acc * 2 - 1)
        total += amp
        amp *= 0.5
    return out / total


def fbm_noise(grid, rng: np.random.Generator, octaves: int = 5, base_freq: float = 2.0, seed_dims: int = 3) -> np.ndarray:
    """:func:`fbm_at` at a grid's extended cell centres -> (6, NE, NE) float32."""
    return fbm_at(grid.centers, rng, octaves, base_freq).astype(np.float32)


def _noise_field(grid, rng, name, scale=1.0, offset=0.0, octaves=5, dtype=np.float32):
    f = FaceField(grid, fbm_noise(grid, rng, octaves) * scale + offset, name=name)
    return f.astype(dtype) if dtype != np.float32 else f


def stub_tectonics(store: WorldStore, params: WorldParams, log=print) -> dict:
    grid = params.coarse_grid()
    rng = params.rng("stub", 1)
    bed = _noise_field(grid, rng, "bedrock", 2500.0, 0.0, 6)
    q = np.quantile(bed.interior, 1.0 - params.world.land_fraction)
    bed.data -= q
    store.save_field(bed)
    store.save_field(_noise_field(grid, rng, "uplift", 0.001, 0.001, 4))
    hard = _noise_field(grid, rng, "hardness", 0.25, 0.5, 4)
    hard.data[...] = np.clip(hard.data, 0, 1)
    store.save_field(hard)
    pid = _noise_field(grid, rng, "plate_id", 6.0, 6.0, 2)
    store.save_field(FaceField(grid, pid.data.astype(np.int16), name="plate_id"))
    # a rigid rotation is a seamless tangent field (magnitude arbitrary)
    store.save_field(rotation_field(grid, (0.0, 0.0, 0.01), "plate_vel"))
    return {"stub": True}


def stub_climate(store: WorldStore, params: WorldParams, log=print) -> dict:
    grid = params.coarse_grid()
    rng = params.rng("stub", 2)
    lat = grid.latitude()
    h = store.load_field("bedrock", grid)
    T = FaceField(grid, (params.climate.T_eq - params.climate.k_lat * np.abs(lat) ** 1.5 - params.climate.lapse * np.maximum(h.data, 0) / 1000.0).astype(np.float32), name="temperature")
    store.save_field(T)
    store.save_field(rotation_field(grid, (0.0, 1.0, 0.0), "wind"))
    pr = _noise_field(grid, rng, "precip", 0.5, 1.0, 4)
    pr.data[...] = np.maximum(pr.data, 0.05) * grid.cell_area / grid.cell_size_m**2
    store.save_field(pr)
    store.save_field(FaceField(grid, (params.climate.k_evap * np.maximum(T.data, 0)).astype(np.float32), name="evap"))
    return {"stub": True}


def stub_erosion(store: WorldStore, params: WorldParams, log=print) -> dict:
    grid = params.coarse_grid()
    rng = params.rng("stub", 3)
    bed = store.load_field("bedrock", grid)
    h = FaceField(grid, bed.data + fbm_noise(grid, rng, 6) * 100.0, name="height")
    store.save_field(h)
    sed = _noise_field(grid, rng, "sediment", 1.0, 1.0, 4)
    sed.data[...] = np.maximum(sed.data, 0)
    store.save_field(sed)
    q = _noise_field(grid, rng, "discharge", 1.0, 1.0, 5)
    q.data[...] = np.exp(3 * q.data)
    store.save_field(q)
    store.save_field(rotation_field(grid, (0.5, 0.0, 0.0), "momentum"))
    return {"stub": True}


def stub_hydro(store: WorldStore, params: WorldParams, log=print) -> dict:
    grid = params.coarse_grid()
    h = store.load_field("height", grid)
    sed = store.load_field("sediment", grid)
    ws = FaceField(grid, np.maximum(h.data + sed.data, 0.0), name="water_surface")
    store.save_field(ws)
    fd = FaceField.zeros(grid, 1, np.uint8, name="flow_dir")
    fd.data[...] = 255
    fd.data[h.data >= 0] = 0
    store.save_field(fd)
    store.save_field(FaceField(grid, np.maximum(h.data, 0) * 0.1, name="flow_acc"))
    store.write_json("graph/drainage.json", {"nodes": [], "edges": [], "stub": True})
    store.write_json("graph/lakes.json", {"lakes": [], "stub": True})
    return {"stub": True}


def stub_watersheds(store: WorldStore, params: WorldParams, log=print) -> dict:
    grid = params.coarse_grid()
    h = store.load_field("height", grid)
    N, H = grid.N, grid.H
    bid = FaceField.zeros(grid, 1, np.int32, name="basin_id")
    bid.data[...] = -1
    blk = max(8, N // 4)
    ii = (np.arange(grid.NE) - H) // blk
    fid = np.arange(6)[:, None, None] * 100 + ii[None, :, None] * 10 + ii[None, None, :]
    bid.data[...] = np.where(h.data >= 0, fid, -1)
    store.save_field(bid)
    basins = []
    for b in np.unique(bid.interior):
        if b < 0:
            continue
        basins.append({"id": int(b), "parent": -1, "face": int(b // 100), "outlet": [int(b // 100), 0, 0], "area_cells": int((bid.interior == b).sum()), "bbox": [0, 0, N, N], "order": 1})
    store.write_json("graph/basins.json", {"basins": basins, "stub": True})
    return {"stub": True, "n_basins": len(basins)}


def stub_refine(store: WorldStore, params: WorldParams, log=print) -> dict:
    """Stub: upsample the coarse height to the fine grid and store as
    ``fine/`` per-face npy (real refine writes per-basin results)."""
    grid = params.coarse_grid()
    R = params.world.R
    h = store.load_field("height", grid)
    sed = store.load_field("sediment", grid)
    ws = store.load_field("water_surface", grid)
    fine_dir = store.root / "fine"
    fine_dir.mkdir(exist_ok=True)
    q = store.load_field("discharge", grid)
    hard = store.load_field("hardness", grid)
    bid = store.load_field("basin_id", grid)
    for name, f in (("height", h), ("sediment", sed), ("water_surface", ws), ("discharge", q), ("hardness", hard), ("basin_id", bid)):
        a = f.interior
        up = np.repeat(np.repeat(a, R, axis=1), R, axis=2)
        up = up.astype(np.int32) if name == "basin_id" else up.astype(np.float32)
        for k in range(6):
            np.save(fine_dir / f"{name}.f{k}.npy", up[k])
    return {"stub": True}


def stub_derive(store: WorldStore, params: WorldParams, log=print) -> dict:
    grid = params.coarse_grid()
    T = store.load_field("temperature", grid)
    P = store.load_field("precip", grid)
    biome = FaceField.zeros(grid, 1, np.uint8, name="biome")
    biome.data[...] = np.clip((T.data + 10) / 10, 0, 4).astype(np.uint8) + 5 * np.clip(P.data / P.data.max() * 3, 0, 2).astype(np.uint8)
    store.save_field(biome)
    store.write_json("graph/rivers.json", {"rivers": [], "stub": True})
    return {"stub": True}


def fine_vertex_field(fine_grid, faces: np.ndarray, name: str = "") -> np.ndarray:
    """Fine cell-centre faces (6, N, N) -> vertex samples (6, N+1, N+1) on
    fine-cell corners (the tile sample convention, ``globe.io.tiles``):
    2x2 average of the surrounding cells, using cross-face halo data at
    face edges so the edge column agrees with the neighbouring face."""
    H, N = fine_grid.H, fine_grid.N
    ff = FaceField.from_interior(fine_grid, np.asarray(faces, dtype=np.float32), name=name, exchange=True)
    E = ff.data[:, H - 1 : H + N + 1, H - 1 : H + N + 1]
    return 0.25 * (E[:, :-1, :-1] + E[:, 1:, :-1] + E[:, :-1, 1:] + E[:, 1:, 1:])


def stub_tiles(store: WorldStore, params: WorldParams, log=print) -> dict:
    """Stub tiles: (T+1)^2 vertex samples per tile (see ``globe.io.tiles``),
    sample (k, l) of tile (lod, f, x, y) at fine-cell corner
    ((x*T + k) << lod, (y*T + l) << lod).  The real Phase 6 writer does the
    same on memmapped fine fields."""
    from .io.tiles import Tile, tiles_per_face, write_tile

    fine = params.fine_grid()
    T = params.world.T
    Nf = params.N_fine
    fine_dir = store.root / "fine"
    verts = {}
    for name in ("height", "sediment", "water_surface"):
        faces = np.stack([np.load(fine_dir / f"{name}.f{k}.npy") for k in range(6)])
        verts[name] = fine_vertex_field(fine, faces, name)
    n_written = 0
    for lod in range(params.max_lod + 1):
        npf = tiles_per_face(Nf, T, lod)
        step = 1 << lod
        for f in range(6):
            for x in range(npf):
                for y in range(npf):
                    sl = (slice(x * T * step, (x * T + T) * step + 1, step), slice(y * T * step, (y * T + T) * step + 1, step))
                    h = verts["height"][f][sl]
                    S = verts["sediment"][f][sl]
                    W = verts["water_surface"][f][sl]
                    layers = np.zeros(h.shape + (4,), np.uint8)
                    layers[..., 0] = np.clip(S * 10, 0, 255)
                    layers[..., 1] = 128
                    layers[..., 2] = 1
                    layers[..., 3] = 100
                    flow = np.zeros(h.shape + (3,), np.uint8)
                    water = np.where(W > h + 0.05, W, 0).astype(np.float32)
                    write_tile(store.root, Tile(lod, f, x, y, h.astype(np.float32), water, layers, flow, {"basins": []}))
                    n_written += 1
    return {"stub": True, "tiles": n_written}
