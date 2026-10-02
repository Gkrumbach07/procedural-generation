#!/usr/bin/env python3
"""Render a tectonic run as a plate-motion movie: where the trenches, ridges and
collisions are, which way each plate moves, and how that evolves.

    python scripts/plate_movie.py --out ../worlds/earth-v17/plates --seed 1423 --steps 4000 --every 25

Each frame is an equal-area (Mollweide) map, centred on the supercontinent's
starting longitude:

* ground -- continental crust in tan (darker = thicker: belts and plateaus),
  island-arc crust in violet, ocean floor in blue from pale (young, at the
  ridges) to navy (old);
* boundaries, from the force phase's own census (``TectonicSim.census_last``):
  red = subduction, with a tooth on the overriding side (the side the slab goes
  under); orange = continent-continent collision; yellow = spreading ridge;
  grey = transform / quiet;
* arrows -- each plate's surface velocity, omega x r, on a regular grid, length
  proportional to speed (the scale bar is 5 cm/yr at tectonics.myr_per_step).

The run is the stage's own simulation with the given preset and seed, so a
movie for seed 1423 shows the plates of the world baked at seed 1423.  Frames
are written as PNG plus an ``index.html`` that plays and scrubs them.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

from globe.config import PRESETS
import globe.tectonics.run as tect
from globe.tectonics.segments import CONTINENTAL, OCEANIC

W, H = 1600, 800                       # Mollweide ellipse fills W x H
PAD_TOP = 34                           # title strip
PAD_BOT = 46                           # legend strip


# ---------------------------------------------------------------- projection
def _moll_theta(lat):
    """Auxiliary angle theta for Mollweide: 2t + sin 2t = pi sin(lat) (Newton)."""
    t = np.array(lat, dtype=np.float64)
    target = np.pi * np.sin(lat)
    for _ in range(30):
        f = 2 * t + np.sin(2 * t) - target
        d = 2 + 2 * np.cos(2 * t)
        t = t - np.where(np.abs(d) > 1e-12, f / d, 0.0)
    return t


def project(lat, lon, lon0):
    """lat/lon (rad) -> pixel x, y."""
    lam = (lon - lon0 + np.pi) % (2 * np.pi) - np.pi
    t = _moll_theta(lat)
    x = (2 * np.sqrt(2) / np.pi) * lam * np.cos(t)      # in [-2 sqrt2, 2 sqrt2]
    y = np.sqrt(2) * np.sin(t)                           # in [-sqrt2, sqrt2]
    px = (x / (2 * np.sqrt(2)) + 1) * 0.5 * (W - 1)
    py = (1 - (y / np.sqrt(2) + 1) * 0.5) * (H - 1) + PAD_TOP
    return px, py


def unproject_grid(lon0):
    """Pixel grid -> unit vectors (H*W, 3) and a mask of pixels inside the ellipse."""
    xs = (np.arange(W) / (W - 1) * 2 - 1) * 2 * np.sqrt(2)
    ys = (1 - np.arange(H) / (H - 1) * 2) * np.sqrt(2)
    X, Y = np.meshgrid(xs, ys)
    inside = (X / (2 * np.sqrt(2))) ** 2 + (Y / np.sqrt(2)) ** 2 <= 1.0
    t = np.arcsin(np.clip(Y / np.sqrt(2), -1, 1))
    lat = np.arcsin(np.clip((2 * t + np.sin(2 * t)) / np.pi, -1, 1))
    lon = lon0 + np.pi * X / (2 * np.sqrt(2) * np.maximum(np.cos(t), 1e-9))
    p = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)
    return p.reshape(-1, 3), inside.reshape(-1)


def latlon(p):
    return np.arcsin(np.clip(p[..., 2], -1, 1)), np.arctan2(p[..., 1], p[..., 0])


# ---------------------------------------------------------------- colours
def ground_colours(seg, step, myr):
    """(M, 3) uint8 per segment."""
    M = seg.M
    col = np.zeros((M, 3), np.float64)
    cont = seg.kind == CONTINENTAL
    oc = ~cont
    # ocean: pale at age 0 -> navy by 150 My
    age_my = np.clip(seg.age * myr, 0, 200)
    a = np.clip(age_my / 150.0, 0, 1)[:, None]
    young, old = np.array([120, 190, 235]), np.array([18, 38, 92])
    col[oc] = (young * (1 - a) + old * a)[oc]
    # island-arc crust: oceanic columns thicker than ~14 km
    arc = oc & (seg.thickness >= 0.4)
    col[arc] = np.array([150, 110, 200])
    # continent: tan, darker with thickness (belts / plateaus)
    th = np.clip((seg.thickness - 0.8) / 1.2, 0, 1)[:, None]
    lo, hi = np.array([214, 196, 150]), np.array([120, 84, 52])
    col[cont] = (lo * (1 - th) + hi * th)[cont]
    return col.astype(np.uint8)


# ---------------------------------------------------------------- frame
def render(sim, lon0, grid_p, inside, font, myr, R_km, step):
    seg, plates = sim.seg, sim.plates
    tree = cKDTree(seg.pos)
    _, nn = tree.query(grid_p, k=1, workers=-1)
    cols = ground_colours(seg, step, myr)
    img = np.full((H * W, 3), 255, np.uint8)
    img[inside] = cols[nn[inside]]
    # faint plate tint: alternate per plate id so plates read as patches
    pid = seg.plate_id[nn]
    shade = np.where((pid % 2) == 0, 1.0, 0.93)
    img = (img.astype(np.float32) * np.where(inside, shade, 1.0)[:, None]).astype(np.uint8)
    canvas = Image.new("RGB", (W, H + PAD_TOP + PAD_BOT), (255, 255, 255))
    canvas.paste(Image.fromarray(img.reshape(H, W, 3)), (0, PAD_TOP))
    d = ImageDraw.Draw(canvas)

    # ---- boundaries from the force census
    cen = sim.census_last
    stats = {}
    if cen is not None and len(cen["i"]):
        i, j = cen["i"], cen["j"]
        appr = cen["appr"]                          # rad/step, > 0 converging
        v_cmyr = appr * R_km * 1e5 / (myr * 1e6)    # km->cm, My->yr
        ki, kj = cen["ki"], cen["kj"]
        mid = seg.pos[i] + seg.pos[j]
        mid /= np.linalg.norm(mid, axis=1, keepdims=True)
        la, lo = latlon(mid)
        mx, my = project(la, lo, lon0)
        conv = v_cmyr > 0.1
        div = v_cmyr < -0.1
        cc = conv & (ki == CONTINENTAL) & (kj == CONTINENTAL)
        sub = conv & ~cc
        # continuous boundary lines: pixels where the plate id changes, coloured by the
        # nearest census contact's class (grey where no contact is near: a quiet boundary)
        P2 = pid.reshape(H, W)
        ins = inside.reshape(H, W)
        edge = np.zeros((H, W), bool)
        edge[:, :-1] |= (P2[:, :-1] != P2[:, 1:]) & ins[:, :-1] & ins[:, 1:]
        edge[:-1, :] |= (P2[:-1, :] != P2[1:, :]) & ins[:-1, :] & ins[1:, :]
        e = np.flatnonzero(edge.ravel())
        cls = np.zeros(i.size, np.int8)                      # 0 quiet, 1 ridge, 2 C-C, 3 subduction
        cls[div] = 1; cls[cc] = 2; cls[sub] = 3
        dmid, kmid = cKDTree(mid).query(grid_p[e], k=1, workers=-1)
        c_e = np.where(dmid < 1.5 * sim.spacing, cls[kmid], 0)
        palette = np.array([[110, 110, 110], [250, 215, 30], [240, 130, 20], [215, 30, 30]], np.uint8)
        arr = np.asarray(canvas).copy()
        ey, ex = np.divmod(e, W)
        for dy in (0, 1):
            for dx in (0, 1):
                yy, xx = np.clip(ey + dy + PAD_TOP, 0, arr.shape[0] - 1), np.clip(ex + dx, 0, W - 1)
                arr[yy, xx] = palette[c_e]
        canvas.paste(Image.fromarray(arr))
        d = ImageDraw.Draw(canvas)
        # teeth: from the trench toward the overriding plate (the side that is not going down)
        down_i = cen["down_i"]
        si = np.flatnonzero(sub)
        if si.size:
            # keep roughly one tooth per ~2 spacings of trench
            si = si[np.argsort(np.random.default_rng(step).random(si.size))][: max(1, si.size // 4)]
            for k in si:
                over = j[k] if down_i[k] else i[k]
                under = i[k] if down_i[k] else j[k]
                a = mid[k]
                b = a + 0.9 * sim.spacing * (seg.pos[over] - seg.pos[under]) / max(np.linalg.norm(seg.pos[over] - seg.pos[under]), 1e-12)
                b /= np.linalg.norm(b)
                (bx,), (by,) = project(*latlon(b[None]), lon0)
                bx, by = float(bx), float(by)
                ax, ay = float(mx[k]), float(my[k])
                if abs(bx - ax) > 40:
                    continue
                nx, ny = -(by - ay), (bx - ax)
                d.polygon([(ax + 0.55 * nx, ay + 0.55 * ny), (ax - 0.55 * nx, ay - 0.55 * ny), (bx, by)], fill=(215, 30, 30))
        seg_len = 0.5 * sim.spacing * R_km
        stats = {"sub_km": float(np.unique(np.r_[i[sub], j[sub]]).size * seg_len),
                 "ridge_km": float(np.unique(np.r_[i[div], j[div]]).size * seg_len),
                 "cc_km": float(np.unique(np.r_[i[cc], j[cc]]).size * seg_len)}

    # ---- velocity arrows on a ~roughly equal-area lattice
    pts = []
    for lat in np.radians(np.arange(-72, 73, 9.0)):
        n = max(1, int(round(40 * math.cos(lat))))
        for lon in lon0 + np.linspace(-np.pi, np.pi, n, endpoint=False) + np.pi / n:
            pts.append((lat, lon))
    pts = np.array(pts)
    P = np.stack([np.cos(pts[:, 0]) * np.cos(pts[:, 1]), np.cos(pts[:, 0]) * np.sin(pts[:, 1]), np.sin(pts[:, 0])], -1)
    _, nb = tree.query(P, k=1, workers=-1)
    om = plates.omega[seg.plate_id[nb]]
    v = np.cross(om, P)                               # rad/step tangent
    sp = np.linalg.norm(v, axis=1) * R_km * 1e5 / (myr * 1e6)
    # arrow length: 5 cm/yr -> 7 degrees of arc
    k = math.radians(4.5) / 5.0
    for p, vv, s in zip(P, v, sp):
        if s < 0.2:
            continue
        u = vv / np.linalg.norm(vv)
        q = math.cos(k * s) * p + math.sin(k * s) * u
        (x0,), (y0,) = project(*latlon(p[None]), lon0)
        (x1,), (y1,) = project(*latlon(q[None]), lon0)
        if abs(x1 - x0) > 200:
            continue
        d.line([(x0, y0), (x1, y1)], fill=(0, 0, 0), width=4)
        d.line([(x0, y0), (x1, y1)], fill=(255, 255, 255), width=2)
        ang = math.atan2(y1 - y0, x1 - x0)
        for da in (2.6, -2.6):
            hx, hy = x1 + 7 * math.cos(ang + da), y1 + 7 * math.sin(ang + da)
            d.line([(x1, y1), (hx, hy)], fill=(0, 0, 0), width=4)
            d.line([(x1, y1), (hx, hy)], fill=(255, 255, 255), width=2)

    # ---- text
    alive = plates.alive
    spd = np.linalg.norm(plates.omega, axis=1) * R_km * 1e5 / (myr * 1e6)
    cont = seg.kind == CONTINENTAL
    title = (f"step {step}  ·  {step * myr:.0f} My  ·  {int(alive.sum())} plates  ·  "
             f"subduction {stats.get('sub_km', 0) / 1000:.0f}k km  ·  ridges {stats.get('ridge_km', 0) / 1000:.0f}k km  ·  "
             f"collisions {stats.get('cc_km', 0) / 1000:.0f}k km  ·  continental crust {float(seg.ext[cont].sum() / (4 * np.pi)):.2f}")
    d.text((10, 8), title, fill=(0, 0, 0), font=font)
    y = H + PAD_TOP + 10
    x = 10
    for label, c in (("subduction (teeth on the overriding side)", (215, 30, 30)), ("continent-continent collision", (240, 130, 20)),
                     ("spreading ridge", (250, 220, 40)), ("transform / quiet", (150, 150, 150)),
                     ("island-arc crust", (150, 110, 200)), ("young -> old sea floor", (60, 110, 170)), ("continent", (190, 160, 110))):
        d.rectangle([x, y + 3, x + 16, y + 17], fill=c)
        d.text((x + 22, y + 2), label, fill=(0, 0, 0), font=font)
        x += 30 + int(d.textlength(label, font=font))
    d.text((10, y + 20), "arrows: plate velocity (5 cm/yr = 4.5° of arc)", fill=(0, 0, 0), font=font)
    return canvas, stats


PLAYER = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Plate motion</title><style>
body{margin:0;background:#111;color:#eee;font:14px system-ui,sans-serif}
#wrap{max-width:1600px;margin:0 auto;padding:8px}
img{width:100%;height:auto;display:block;background:#fff}
#bar{display:flex;gap:10px;align-items:center;padding:8px 0}
input[type=range]{flex:1}
button{background:#333;color:#eee;border:1px solid #555;border-radius:4px;padding:4px 12px;cursor:pointer}
</style></head><body><div id="wrap">
<div id="bar"><button id="play">▶ play</button><button id="prev">◀</button><button id="next">▶</button>
<input id="s" type="range" min="0" max="0" value="0"><span id="lab"></span>
<select id="fps"><option value="4">4 fps</option><option value="8" selected>8 fps</option><option value="15">15 fps</option></select></div>
<img id="im"></div><script>
const F=__FRAMES__;const im=document.getElementById('im'),s=document.getElementById('s'),lab=document.getElementById('lab');
s.max=F.length-1;const cache=F.map(f=>{const i=new Image();i.src=f.file;return i});
function show(k){k=(k+F.length)%F.length;s.value=k;im.src=F[k].file;lab.textContent=`${F[k].step} steps · ${F[k].my} My`}
let t=null;document.getElementById('play').onclick=e=>{if(t){clearInterval(t);t=null;e.target.textContent='▶ play'}else{e.target.textContent='❚❚ pause';
t=setInterval(()=>show(+s.value+1),1000/+document.getElementById('fps').value)}};
document.getElementById('prev').onclick=()=>show(+s.value-1);document.getElementById('next').onclick=()=>show(+s.value+1);
s.oninput=()=>show(+s.value);document.onkeydown=e=>{if(e.key=='ArrowRight')show(+s.value+1);if(e.key=='ArrowLeft')show(+s.value-1)};show(0);
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", default="earth")
    ap.add_argument("--seed", type=int, default=1423)
    ap.add_argument("--steps", type=int, default=None, help="default: the preset's tectonics.steps")
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--set", action="append", default=[], help="tectonics.key=value")
    a = ap.parse_args()

    p = PRESETS[a.preset]()
    p.world.seed = a.seed
    for kv in a.set:
        k, v = kv.split("=", 1)
        g, key = k.split(".")
        t = type(getattr(getattr(p, g), key))
        setattr(getattr(p, g), key, (v.lower() in ("1", "true")) if t is bool else t(v))
    steps = int(a.steps or p.tectonics.steps)
    myr = float(p.tectonics.myr_per_step)
    R_km = float(getattr(p.tectonics, "tectonic_radius_km", 6371.0) or 6371.0)
    os.makedirs(a.out, exist_ok=True)

    sim = tect.initialise(p, log=None)
    cont = sim.seg.kind == CONTINENTAL
    c = sim.seg.pos[cont].mean(0) if cont.any() else np.array([1.0, 0, 0])
    lon0 = float(math.atan2(c[1], c[0]))
    grid_p, inside = unproject_grid(lon0)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()

    frames = []
    t0 = time.time()
    for k in range(steps + 1):
        if k % a.every == 0 or k == steps:
            if sim.census_last is None:
                sim.force_state() if hasattr(sim, "force_state") else None
            im, st = render(sim, lon0, grid_p, inside, font, myr, R_km, k)
            fn = f"frame_{k:05d}.png"
            im.save(os.path.join(a.out, fn), optimize=True)
            frames.append({"file": fn, "step": k, "my": round(k * myr), **{kk: round(v) for kk, v in st.items()}})
            print(f"[plates] step {k}/{steps}  {fn}  ({time.time() - t0:.0f}s)", flush=True)
        if k < steps:
            sim.step()
    with open(os.path.join(a.out, "index.html"), "w") as f:
        f.write(PLAYER.replace("__FRAMES__", json.dumps(frames)))
    json.dump(frames, open(os.path.join(a.out, "frames.json"), "w"))
    print(f"[plates] {len(frames)} frames -> {a.out}/index.html")


if __name__ == "__main__":
    main()
