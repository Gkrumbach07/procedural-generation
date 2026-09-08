"""Animate a bake: plates drifting over the tectonic simulation, and the
drainage network growing over the erosion iterations.

    cd bake
    python3 scripts/animate.py --world worlds/demo --stage tectonics --out tect.webp
    python3 scripts/animate.py --world worlds/demo --stage erosion   --out ero.webp

Both stages are *re-run* from their inputs rather than read back from disk:
neither keeps its intermediate states (tectonics writes only the final
fields, erosion only its checkpoints), and re-running is deterministic, so
the frames are exactly the states the bake passed through.  The world's own
parameters come from its ``manifest.json``; ``--set`` overrides them the
same way ``bake.py`` does, which is the cheap way to animate a coarser
world than the one on disk.

Output is an animated WebP (Pillow; no ffmpeg needed).  ``--gif`` writes a
GIF instead, which every viewer plays but quantises to 256 colours.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from globe.config import WorldParams  # noqa: E402
from globe.io.world_store import WorldStore  # noqa: E402
from globe.viz import quicklook as ql  # noqa: E402


# --------------------------------------------------------------------------
# frames
# --------------------------------------------------------------------------
def _net(img: np.ndarray, width: int) -> Image.Image:
    """(6, N, N, 3) face images -> one unfolded-net PIL frame of `width`."""
    out = ql.to_net(img)
    im = Image.fromarray(np.ascontiguousarray(out))
    if im.width != width:
        h = max(1, round(im.height * width / im.width))
        im = im.resize((width, h), Image.LANCZOS)
    return im


def tectonic_frames(params: WorldParams, frames: int, width: int, log=print):
    """Yield one frame per ``steps / frames`` tectonic steps.

    Each frame is the current crust height splatted to the tect grid and
    tinted hypsometrically, so plates read as continents: the sea level is
    the same land-fraction quantile ``finalise`` uses, recomputed per frame
    (the crust is still growing, so a fixed datum would drown or beach the
    early steps).
    """
    from globe.tectonics.run import initialise, _smooth_field
    from globe.tectonics.collision import SmoothSplat, build_tree

    sim = initialise(params, log=lambda *a: None)
    tp, grid = sim.tp, sim.grid
    total = int(tp.steps)
    every = max(1, total // max(1, frames))
    done = 0
    while True:
        tree = build_tree(sim.seg)
        blend = SmoothSplat(tree, grid, tp.splat_sigma_factor * sim.spacing, int(tp.splat_knn))
        buoy = tp.ridge_height * np.exp(-sim.seg.age / max(float(tp.ridge_age), 1.0))
        bed = _smooth_field(grid, blend(sim.seg.height() + buoy), tp, cascade=True).interior
        sea = float(np.quantile(bed, 1.0 - params.world.land_fraction))
        # Unlike erosion, the palette is deliberately *not* pinned here: the
        # crust is still being created, so its relief grows by orders of
        # magnitude and a scale fixed at step 0 would flatten everything after.
        yield _net(ql.render_height(bed - sea, cell_size=grid.cell_size_m), width), done
        if done >= total:
            return
        n = min(every, total - done)
        sim.run(n, log=lambda *a: None)
        done += n


def erosion_frames(store: WorldStore, params: WorldParams, frames: int, width: int, log=print):
    """Yield one frame per ``iterations / frames`` erosion iterations, from
    bedrock.  Rivers are the live discharge map (log-weighted blue) and
    lakes the flooded routing surface, so the frames show the network
    organising itself rather than just the relief changing."""
    from globe.erosion import run as er
    from globe.erosion.maps import step

    p = params.with_overrides(erosion={"resume": False})
    state = er.build_state(store, p)
    total = int(p.erosion.iterations)
    every = max(1, total // max(1, frames))
    done = 0
    # One palette and one vertical exaggeration for the whole run, taken from
    # the starting bedrock: render_height derives them per call otherwise, and
    # a rescaling palette makes terrain that never moved look like it did.
    # Erosion only redistributes relief (the datum is held every iteration and
    # p99 height moves ~8 % over a full run), so the bedrock scale stays right.
    scale = ql.terrain_scale(state.surface()[state.interior] * state.height_unit_m, cell_size=state.grid.cell_size_m)
    while True:
        surf = state.surface()[state.interior] * state.height_unit_m
        img = ql.render_height(surf, cell_size=state.grid.cell_size_m, **scale)
        # Lakes: the epsilon-flooded routing surface standing above the
        # terrain — masked to LAND, because over the sea that surface sits at
        # sea level above every submerged cell and would paint the whole ocean.
        if state.route is not None:
            depth = (state.route - state.surface())[state.interior] * state.height_unit_m
            lake = (depth > 0.5) & (surf > 0.0)
            if lake.any():
                img = ql.overlay(img, lake, (60, 120, 220), 0.9)
        # rivers: the live discharge map, log-weighted above its 97th percentile
        q = np.where(surf > 0.0, state.discharge[state.interior].astype(np.float64), 0.0)
        lq = np.log1p(np.maximum(q, 0.0))
        if lq.max() > 0:
            thr = float(np.percentile(lq[surf > 0.0], 97)) if (surf > 0.0).any() else 0.0
            wgt = np.clip((lq - thr) / max(lq.max() - thr, 1e-9), 0, 1)
            img = ql.overlay(img, (wgt > 0) & (surf > 0.0), (30, 80, 255), 0.95, weight=0.35 + 0.65 * wgt)
        yield _net(img, width), done
        if done >= total:
            return
        for k in range(min(every, total - done)):
            step(state, p, (done + k,))
        done += min(every, total - done)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def write_animation(path: Path, frames: list[Image.Image], fps: float, gif: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(20, int(round(1000.0 / max(fps, 0.1))))
    head, rest = frames[0], frames[1:]
    if gif:
        head.save(str(path), save_all=True, append_images=rest, duration=duration, loop=0, optimize=True)
    else:
        head.save(str(path), format="WEBP", save_all=True, append_images=rest, duration=duration, loop=0, quality=88, method=4)
    return path


def parse_sets(sets) -> dict:
    over: dict = {}
    for s in sets or []:
        key, _, val = s.partition("=")
        group, _, name = key.partition(".")
        try:
            v = json.loads(val)
        except json.JSONDecodeError:
            v = val
        over.setdefault(group, {})[name] = v
    return over


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", required=True, help="baked world directory (its manifest supplies the parameters)")
    ap.add_argument("--stage", default="tectonics", choices=["tectonics", "erosion"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--width", type=int, default=900, help="frame width in pixels")
    ap.add_argument("--gif", action="store_true", help="write a GIF instead of a WebP")
    ap.add_argument("--set", action="append", default=[], metavar="GROUP.KEY=VALUE")
    a = ap.parse_args()

    store = WorldStore(a.world)
    params = WorldParams.from_dict(json.loads((Path(a.world) / "manifest.json").read_text())["params"])
    over = parse_sets(a.set)
    if over:
        params = params.with_overrides(**over)

    t0 = time.time()
    gen = tectonic_frames(params, a.frames, a.width) if a.stage == "tectonics" else erosion_frames(store, params, a.frames, a.width)
    frames: list[Image.Image] = []
    for im, done in gen:
        frames.append(im)
        print(f"[animate] {a.stage} frame {len(frames)} at step {done} ({time.time() - t0:.0f}s)", flush=True)
    write_animation(a.out, frames, a.fps, a.gif)
    mb = a.out.stat().st_size / 1e6
    print(f"[animate] {len(frames)} frames -> {a.out} ({mb:.1f} MB, {time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
