#!/usr/bin/env python3
"""3-D view of a saved zoom window (``window_bake.py --save``) or zoom level:
one self-contained HTML page (globe/zoom/view.py).

    python scripts/window_view.py scratch/window/mtn2/def76_R128.npz --out mtn76.html
    python scripts/window_view.py ... --shot mtn76.png --yaw 35 --pitch 32 --dist 0.9

Reads the arrays ``window_bake.py --save`` writes (height, sediment,
discharge, water_surface, mask) and the run's JSON beside them (cell size).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("BAKE") or os.path.join(os.path.dirname(__file__), ".."))
from globe.zoom.view import build_html, screenshot  # noqa: E402


def build(npz_path: Path, max_res: int, exag: float, min_lake_cells: int = 4, min_lake_depth: float = 1.0) -> tuple[str, dict]:
    z = np.load(npz_path)
    meta_path = npz_path.with_suffix(".json")
    cell_m = float(json.loads(meta_path.read_text())["cell_m"]) if meta_path.exists() else 1.0
    return build_html(z["height"], z["sediment"], z["discharge"], z["water_surface"], z["mask"], cell_m, title=npz_path.stem,
                      max_res=max_res, exag=exag, min_lake_cells=min_lake_cells, min_lake_depth=min_lake_depth)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--out", default=None, help="HTML path (default: beside the npz)")
    ap.add_argument("--max-res", type=int, default=1024, help="cells per side drawn (the window is block-averaged down to it)")
    ap.add_argument("--exag", type=float, default=1.0)
    ap.add_argument("--min-lake-cells", type=int, default=4, help="pools smaller than this (native cells) are drawn as ground")
    ap.add_argument("--min-lake-depth", type=float, default=1.0, help="metres of water a cell needs to be drawn as lake")
    ap.add_argument("--shot", default=None, help="also write a PNG screenshot (Playwright)")
    ap.add_argument("--yaw", type=float, default=35.0)
    ap.add_argument("--pitch", type=float, default=35.0)
    ap.add_argument("--dist", type=float, default=1.2)
    ap.add_argument("--tx", type=float, default=0.0)
    ap.add_argument("--ty", type=float, default=0.0)
    ap.add_argument("--size", default="1400x900")
    a = ap.parse_args()
    npz = Path(a.npz)
    html, meta = build(npz, a.max_res, a.exag, a.min_lake_cells, a.min_lake_depth)
    out = Path(a.out) if a.out else npz.with_name(npz.stem + ".view.html")
    out.write_text(html)
    print(f"{out}  ({meta['W']}x{meta['H']} drawn at {meta['cell_m']:.0f} m, relief {meta['h1'] - meta['h0']:.0f} m)")
    if a.shot:
        errors = screenshot(out, a.shot, a.yaw, a.pitch, a.dist, a.tx, a.ty, a.exag, tuple(int(v) for v in a.size.split("x")))
        if errors:
            print("page errors:", errors[:5])
        print(a.shot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
