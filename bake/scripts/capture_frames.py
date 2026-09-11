#!/usr/bin/env python3
"""Backfill a world's viewer timeline without re-baking it.

    python scripts/capture_frames.py worlds/w-base --tectonics            # re-simulate, capture frames only
    python scripts/capture_frames.py worlds/w-base --checkpoints worlds/w-base/keep

``--tectonics`` re-runs the (deterministic) plate simulation with frame
capture on and writes nothing but ``frames/tectonics/`` -- the stage's
outputs are untouched.  ``--checkpoints DIR`` turns erosion checkpoints
(``erosion_iterNNNN.npz``) into erosion frames; ``erosion/run.py`` keeps only
the two newest, so a full timeline needs a directory they were copied to.
Then run ``scripts/export_viewer.py``.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from globe.io.world_store import WorldStore  # noqa: E402
from globe.viz import frames as vf  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("world")
    ap.add_argument("--tectonics", action="store_true", help="re-simulate tectonics and capture its frames")
    ap.add_argument("--checkpoints", default=None, help="directory of erosion checkpoints to turn into frames")
    ap.add_argument("--frames", type=int, default=None, help="tectonics frames (default: render.tectonics_frames)")
    ap.add_argument("--res", type=int, default=None, help="cells per face (default: render.frame_res)")
    args = ap.parse_args()
    store = WorldStore(args.world)
    params = store.params()
    res = int(args.res or params.render.frame_res)

    if args.tectonics:
        from globe.tectonics.run import initialise

        tp = params.tectonics
        n = int(args.frames or params.render.tectonics_frames or 60)
        rec = vf.FrameRecorder(store.root, "tectonics", min(res, int(tp.N_tect)))
        rec.clear()
        sim = initialise(params, print)
        sim.run(int(tp.steps), log=print, on_frame=lambda s, i: vf.tectonics_frame(s, rec, i, int(tp.steps)), frames=n)
        print(f"tectonics: {len(vf.list_frames(store.root, 'tectonics'))} frames -> {rec.dir}")

    if args.checkpoints:
        n_iter = int(params.erosion.iterations)
        rec = vf.FrameRecorder(store.root, "erosion", min(res, int(params.world.N_c)))
        for p in sorted(Path(args.checkpoints).glob("erosion_iter*.npz")):
            meta = json.loads(p.with_suffix(".json").read_text())
            unit = float(meta["height_unit_m"])
            with np.load(p) as z:
                H = (z["height"].shape[1] - int(params.world.N_c)) // 2
                I = (slice(None), slice(H, -H), slice(H, -H))
                surf = (z["height"][I] + z["sediment"][I]) * unit
                rec.write(int(meta["iteration"]), {"height": vf.downsample(surf, rec.res).astype(np.float16),
                                                   "discharge": vf.downsample(z["discharge"][I], rec.res, "max").astype(np.float16)},
                          iteration=int(meta["iteration"]), of=n_iter, units="m", source=p.name)
            print(f"  {p.name} -> frame {meta['iteration']}")
        print(f"erosion: {len(vf.list_frames(store.root, 'erosion'))} frames -> {rec.dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
