"""Render a baked world's surface so the *shape* is readable.

    python3 scripts/render_terrain.py --world W [--km 6] [--style neutral]

Why not the quicklook palette: `viz/quicklook.py` tints by elevation into
greens and tans, which is a fine map but a poor instrument.  Colour and
shading then carry the same signal, so a slope reads as a change of
material -- a hillside darkens and looks like different ground rather than
like a hillside -- and the thing we are usually trying to judge (how the
surface is *shaped* between the rivers) is exactly what gets hidden.

`neutral` (the default) paints one achromatic ramp and puts every bit of
the contrast into the light: relief shading from two directions plus a
slope term, so a facet's tone means its angle and nothing else.  `natural`
keeps the map colours for when a picture rather than a measurement is
wanted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def hillshade(z: np.ndarray, cell: float, az: float, alt: float) -> np.ndarray:
    gy, gx = np.gradient(z, cell)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    a, z0 = np.radians(alt), np.radians(360.0 - az + 90.0)
    return np.clip(np.sin(a) * np.cos(slope) + np.cos(a) * np.sin(slope) * np.cos(z0 - aspect), 0.0, 1.0)


def shade(z: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Two-light relief shading, and the slope in rise/run.

    A single low sun leaves every face that points away from it flat black,
    which loses as much shape as it reveals.  A second, dimmer light from
    the opposite quarter keeps those faces legible without washing the
    first one out.
    """
    key = hillshade(z, cell, az=315.0, alt=45.0)
    fill = hillshade(z, cell, az=135.0, alt=60.0)
    gy, gx = np.gradient(z, cell)
    return np.clip(0.75 * key + 0.25 * fill, 0.0, 1.0), np.hypot(gx, gy)


def render(z, river, cell, style="neutral", px=1000, sea=0.0):
    land = z > sea
    relief, slope = shade(z, cell)
    if style == "neutral":
        # one achromatic ramp; all the contrast is in the light, so tone
        # means angle rather than material
        lo, hi = (np.percentile(z[land], [2, 98]) if land.any() else (0.0, 1.0))
        t = np.clip((z - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        base = 0.52 + 0.30 * t                       # light, so shading has room
        sh = 0.30 + 0.95 * relief
        # darken by slope as well as by facing: a steep face stays legible
        # even when it happens to point at the light
        sh *= 1.0 - 0.35 * np.clip(slope / 0.8, 0.0, 1.0)
        v = np.clip(base * sh, 0.0, 1.0)
        rgb = np.stack([v * 1.02, v, v * 0.96], axis=-1)   # a hair warm, still neutral
        water = np.array([0.07, 0.13, 0.24])
        riv_c = np.array([0.25, 0.62, 0.95])
    else:
        lo, hi = (np.percentile(z[land], [2, 98]) if land.any() else (0.0, 1.0))
        t = np.clip((z - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        rgb = np.stack([0.34 + 0.48 * t, 0.47 + 0.33 * t, 0.29 + 0.28 * t], axis=-1)
        rgb *= (0.32 + 0.78 * relief)[..., None]
        water = np.array([0.09, 0.20, 0.40])
        riv_c = np.array([0.22, 0.47, 0.88])
    rgb[~land] = water
    if river is not None:
        rgb[river] = riv_c
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    return img.resize((px, px), Image.LANCZOS)


def load(world: Path):
    m = json.loads((world / "manifest.json").read_text())
    R = int(m["params"]["world"]["R"])
    cell = float(m["params"]["world"]["cell_size_m"]) / R
    h = np.stack([np.load(world / "fine" / f"height.f{f}.npy") for f in range(6)]).astype(np.float64)
    rp = world / "fine" / "river_mask.f0.npy"
    riv = (np.stack([np.load(world / "fine" / f"river_mask.f{f}.npy") for f in range(6)]).astype(bool)
           if rp.exists() else np.zeros(h.shape, bool))
    return h, riv, cell


def best_patch(h, riv, side, land_min=0.55):
    """The patch with the most relief and river, so the render shows the
    landscape rather than whichever corner happened to be first."""
    best = None
    for f in range(6):
        L = h[f] > 0
        st = max(8, side // 12)
        for i in range(0, h.shape[1] - side, st):
            for j in range(0, h.shape[2] - side, st):
                w = L[i:i+side, j:j+side]
                if w.mean() < land_min:
                    continue
                s = float(np.ptp(h[f, i:i+side, j:j+side])) * w.mean() * (1.0 + 3.0 * riv[f, i:i+side, j:j+side].mean())
                if best is None or s > best[0]:
                    best = (s, f, i, j)
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", required=True, type=Path)
    ap.add_argument("--km", type=float, default=6.0, help="patch width in km")
    ap.add_argument("--style", choices=("neutral", "natural"), default="neutral")
    ap.add_argument("--px", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    h, riv, cell = load(a.world)
    side = min(h.shape[1], int(round(a.km * 1000.0 / cell)))
    b = best_patch(h, riv, side)
    if b is None:
        print("no land patch that size"); return 1
    _, f, i, j = b
    z = h[f, i:i+side, j:j+side]
    r = riv[f, i:i+side, j:j+side]
    out = a.out or (a.world / f"terrain_{a.style}.jpg")
    render(z, r, cell, a.style, a.px).save(out, quality=93)
    print(f"{side}^2 cells at {cell:g} m = {side*cell/1000:.1f} km, relief {np.ptp(z):.0f} m -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
