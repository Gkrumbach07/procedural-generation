#!/usr/bin/env python3
"""Export (or re-export) a baked world's HTML viewer.

    python scripts/export_viewer.py worlds/demo                    # -> worlds/demo/viewer/index.html
    python scripts/export_viewer.py worlds/demo --single           # + viewer/standalone.html (one file)
    python scripts/export_viewer.py worlds/demo --formats equirect,anim

A bake already does this at the end (``render.viewer``); this is for worlds
baked before that, or to change the resolution or add export formats.  A
world with no captured frames gets the final state only --
``scripts/capture_frames.py`` backfills the timeline.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from globe.viz.viewer import export_viewer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("world")
    ap.add_argument("--out", default=None, help="output directory (default: <world>/viewer)")
    ap.add_argument("--final-res", type=int, default=None, help="cells per face of the final frame (default: render.viewer_final_res)")
    ap.add_argument("--frame-res", type=int, default=None, help="downsample timeline frames to this many cells per face")
    ap.add_argument("--max-frames", type=int, default=None, help="thin the timeline to about this many frames")
    ap.add_argument("--single", action="store_true", help="also write standalone.html with every frame inlined")
    ap.add_argument("--formats", default="", help="extra exports: equirect, anim (comma separated)")
    args = ap.parse_args()
    p = export_viewer(args.world, args.out, formats=args.formats, final_res=args.final_res, frame_res=args.frame_res,
                      max_frames=args.max_frames, single=args.single)
    print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
