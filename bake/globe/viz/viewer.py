"""Static HTML viewer for a baked world: globe and flat map, pan/zoom, and a
timeline through tectonics -> erosion -> final that can be scrubbed or played.

    export_viewer(world_dir)            # -> <world>/viewer/index.html

Open ``index.html`` straight from disk; no server is needed.  Browsers
refuse ``fetch`` on ``file://``, so every data file is a ``<script>`` that
hands its payload to ``window.GLOBE_VIEWER``: ``data/meta.js`` describes the
frames, and ``data/fNNNN.js`` carries one frame's textures as base64 WebP.
``single=True`` also writes ``viewer/standalone.html`` with everything
inlined -- one file to copy or attach.

Textures are cube-face atlases: 3 columns x 2 rows of ``(res + 2)²`` tiles,
face ``f`` at column ``f % 3``, row ``f // 3``, with a one-cell border taken
from the neighbouring face so bilinear filtering is seamless.  Pixel
``(x, y) = (1 + i, 1 + j)`` of a tile is cell ``(i, j)``.  Texture 0 is
``R, G`` = 16-bit height over the frame's ``[h0, h1]`` metres and ``B`` = an
overlay byte (plate id in tectonics frames, log discharge afterwards); the
final frame adds climate/biome and plate/sediment/crust textures.  The
shader (``viewer.html``) maps every pixel to a direction on the sphere and
then to ``(face, u, v)`` exactly as :mod:`globe.cubesphere` does, so no
reprojection happens anywhere.  Latitude has its pole on +Z, as in
:meth:`globe.cubesphere.Grid.latitude` and the climate stage.

Frames come from ``<world>/frames/`` (captured during the bake, see
:mod:`globe.viz.frames`); a world baked before frame capture existed still
gets a viewer with the final state, and ``scripts/capture_frames.py`` can
backfill its timeline.

Extra ``formats`` (comma separated): ``equirect`` writes lat/lon PNGs of the
final state (16-bit height, shaded relief, one per layer) to
``viewer/exports/``; ``anim`` writes an animated WebP of the whole timeline
as a flat map.
"""
from __future__ import annotations

import base64
import io
import json
import math
import shutil
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..cubesphere import from_sphere_v, to_sphere_v
from . import frames as vf

TEMPLATE = Path(__file__).with_name("viewer.html")
PLACEHOLDER = "<!--GLOBE_VIEWER_DATA-->"
PAD = 1

# colour stops shared with the shader in viewer.html (keep in sync)
LAND = np.array([[.30, .50, .28], [.58, .64, .38], [.78, .70, .48], [.58, .45, .34], [.96, .96, .97]])
SEA = np.array([[.62, .80, .86], [.36, .62, .80], [.18, .42, .68], [.09, .24, .50], [.03, .09, .26]])
CMAPS = {
    "thermal": np.array([[.19, .07, .55], [.16, .44, .90], [.94, .94, .90], [.95, .55, .18], [.62, .05, .10]]),
    "rain": np.array([[.93, .87, .72], [.72, .80, .45], [.30, .65, .40], [.15, .45, .70], [.10, .15, .50]]),
    "viridis": np.array([[.27, 0, .33], [.23, .32, .55], [.13, .57, .55], [.37, .79, .38], [.99, .91, .14]]),
}
LIGHT = np.array([-0.6, 0.6, 0.75]) / np.linalg.norm([-0.6, 0.6, 0.75])


# --------------------------------------------------------------------------
# cube-face atlas encoding
# --------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _pad_index(res: int, pad: int):
    """For every cell of a padded face (``pad`` cells beyond each edge), the
    ``(face, i, j)`` of the interior cell containing its centre."""
    k = np.arange(-pad, res + pad)
    I, J = np.meshgrid(k, k, indexing="ij")
    shape = (6,) + I.shape
    faces = np.broadcast_to(np.arange(6)[:, None, None], shape)
    p = to_sphere_v(faces, np.broadcast_to((I + 0.5) / res, shape), np.broadcast_to((J + 0.5) / res, shape))
    f2, u2, v2 = from_sphere_v(p)
    i2 = np.minimum((u2 * res).astype(np.int64), res - 1)
    j2 = np.minimum((v2 * res).astype(np.int64), res - 1)
    return f2, i2, j2


def pad_faces(a: np.ndarray, pad: int = PAD) -> np.ndarray:
    """``(6, r, r)`` -> ``(6, r + 2·pad, r + 2·pad)`` with the border cells
    read from the neighbouring faces (nearest cell)."""
    f, i, j = _pad_index(a.shape[1], pad)
    return a[f, i, j]


def atlas(channels: list[np.ndarray]) -> np.ndarray:
    """Padded ``(6, T, T)`` uint8 channels -> ``(2T, 3T, C)`` atlas image."""
    T = channels[0].shape[1]
    img = np.zeros((2 * T, 3 * T, len(channels)), np.uint8)
    for f in range(6):
        r, c = divmod(f, 3)
        for k, ch in enumerate(channels):
            img[r * T:(r + 1) * T, c * T:(c + 1) * T, k] = ch[f].T
    return img


def encode_height(h: np.ndarray):
    """Padded metres -> (hi byte, lo byte, h0, h1)."""
    h = np.asarray(h, np.float64)
    h0, h1 = float(np.floor(h.min())), float(np.ceil(h.max()))
    if h1 <= h0:
        h1 = h0 + 1.0
    q = np.clip(np.round((h - h0) / (h1 - h0) * 65535.0), 0, 65535).astype(np.uint16)
    return (q >> 8).astype(np.uint8), (q & 255).astype(np.uint8), h0, h1


def log_byte(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """0 at or below ``lo``; ``255·ln(x/lo)/ln(hi/lo)`` above (clipped)."""
    x = np.asarray(x, np.float64)
    t = np.log(np.maximum(x, lo) / lo) / math.log(hi / lo)
    b = np.clip(np.round(255.0 * t), 1, 255).astype(np.uint8)
    b[~(x > lo)] = 0
    return b


def lin_byte(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(np.round(255.0 * (np.asarray(x, np.float64) - lo) / (hi - lo)), 0, 255).astype(np.uint8)


def webp_b64(img: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img), "RGB").save(buf, "WEBP", lossless=True, quality=70, method=4)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@lru_cache(maxsize=8)
def _area(res: int) -> np.ndarray:
    """Relative cell areas of an equi-angular face (same for all 6)."""
    s = np.tan(((np.arange(res) + 0.5) / res - 0.5) * (np.pi / 2))
    S, T = np.meshgrid(s, s, indexing="ij")
    w = (1 + S * S) * (1 + T * T) / (1 + S * S + T * T) ** 1.5
    return np.broadcast_to(w, (6, res, res))


def relief_exaggeration(h: np.ndarray, cell_m: float) -> float:
    """Default hillshade exaggeration: a 90th-percentile land slope shades
    like a 0.35 gradient.  An Earth-scale world (10 km cells, slopes ~1 %)
    needs tens; a 50 m-cell test world needs none."""
    g = np.hypot(np.diff(h, axis=1)[:, :, :-1], np.diff(h, axis=2)[:, :-1, :]) / max(cell_m, 1e-9)
    p90 = _pctl(g[h[:, :-1, :-1] >= 0], 90, 0.01)
    return float(np.clip(round(0.35 / max(p90, 1e-6)), 1, 50))


def frame_stats(h: np.ndarray) -> dict:
    w = _area(h.shape[1])
    land = h >= 0.0
    wl = float(w[land].sum())
    return {
        "land_pct": round(100.0 * wl / float(w.sum()), 3),
        "max_m": round(float(h.max()), 1),
        "median_land_m": round(float(np.median(h[land])), 1) if land.any() else None,
        "above2k_pct": round(100.0 * float(w[h >= 2000.0].sum()) / wl, 3) if wl > 0 else 0.0,
    }


# --------------------------------------------------------------------------
# collecting the frames
# --------------------------------------------------------------------------
def _load_faces(root: Path, name: str, sub: str = "coarse"):
    paths = [root / sub / f"{name}.f{k}.npy" for k in range(6)]
    if not all(p.exists() for p in paths):
        return None
    return np.stack([np.load(p) for p in paths])


class _Frame:
    """One timeline entry before encoding: metres + optional channels."""

    def __init__(self, stage, key, label, height, **ch):
        self.stage, self.key, self.label = stage, key, label
        self.height = np.asarray(height, np.float32)
        self.ch = {k: v for k, v in ch.items() if v is not None}

    @property
    def res(self) -> int:
        return int(self.height.shape[1])


def _thin(items: list, n: int) -> list:
    """``n`` evenly spaced items, always keeping the first and the last."""
    if len(items) <= n:
        return items
    return [items[i] for i in sorted({round(k * (len(items) - 1) / (n - 1)) for k in range(n)})]


def collect_frames(root: Path, final_res: int | None, log=print, frame_res: int | None = None,
                   max_frames: int | None = None) -> tuple[list[_Frame], dict]:
    manifest = json.loads((root / "manifest.json").read_text())
    stages = manifest.get("stages", {})
    frames: list[_Frame] = []

    tect = vf.list_frames(root, "tectonics")
    ero = vf.list_frames(root, "erosion")
    # the last tectonics frame maps compact plate ids back to raw ones, so
    # read it before thinning can drop it
    alive_last = tect[-1][2].get("alive") if tect else None
    if max_frames:
        total = max(len(tect) + len(ero), 1)
        tect = _thin(tect, max(2, round(max_frames * len(tect) / total)))
        ero = _thin(ero, max(2, max_frames - len(tect)))
    lo = (lambda a, how="mean": vf.downsample(a, frame_res, how)) if frame_res else (lambda a, how="mean": a)
    scale = (stages.get("tectonics", {}).get("info", {}) or {}).get("scale_m_per_unit")
    if tect and scale:
        for key, p, meta in tect:
            with np.load(p) as z:
                frames.append(_Frame("tectonics", key, f"tectonics · step {key} / {meta.get('of', '?')}",
                                     lo(z["height"].astype(np.float32) * float(scale)), plate=lo(z["plate"], "nearest")))
    for key, p, meta in ero:
        with np.load(p) as z:
            frames.append(_Frame("erosion", key, f"erosion · iteration {key} / {meta.get('of', '?')}",
                                 lo(z["height"]), discharge=lo(z["discharge"], "max")))

    # the final state, from the coarse fields at (up to) full resolution
    h = _load_faces(root, "height")
    if h is not None:
        sed = _load_faces(root, "sediment")
        surf = h + (sed if sed is not None else 0.0)
        label = "final · " + ", ".join(s for s in ("erosion", "hydro", "derive") if stages.get(s, {}).get("done"))
    else:
        surf = _load_faces(root, "bedrock")
        sed = None
        label = "final · tectonics"
    if surf is None:
        raise FileNotFoundError(f"{root}: no height or bedrock on the coarse grid -- has tectonics run?")
    N = surf.shape[1]
    r = min(N, int(final_res or N))
    ds = lambda a, how="mean": None if a is None else vf.downsample(a, r, how)
    plate = _load_faces(root, "plate_id")
    if plate is not None:
        plate = plate.astype(np.int64)
        if alive_last is not None and plate.max() < len(alive_last):
            plate = np.asarray(alive_last, np.int64)[np.maximum(plate, 0)]  # compact ids -> raw, as in the frames
        plate = plate + 1
    crust = _load_faces(root, "crust_kind", "diagnostics")
    frames.append(_Frame(
        "final", N, label, ds(surf),
        discharge=ds(_load_faces(root, "discharge"), "max"),
        temperature=ds(_load_faces(root, "temperature")),
        precip=ds(_load_faces(root, "precip")),
        biome=ds(_load_faces(root, "biome"), "nearest"),
        plate=ds(plate, "nearest"),
        sediment=ds(sed),
        crust=ds(crust, "nearest") if crust is not None else None,
    ))
    log(f"[viewer] {len(tect) if scale else 0} tectonics + {sum(f.stage == 'erosion' for f in frames)} erosion frames + final ({r}² per face)")
    return frames, manifest


# --------------------------------------------------------------------------
# channel encoding
# --------------------------------------------------------------------------
def _pctl(a, q, default):
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if a.size else default


def channel_specs(final: _Frame) -> dict:
    """Byte encodings, fixed across the timeline so frames compare."""
    specs = {}
    land = final.height >= 0
    if "discharge" in final.ch:
        q = final.ch["discharge"].astype(np.float64)
        ql = q[land & (q > 0)]
        lo = max(_pctl(ql, 50, 1.0), 1e-6)
        hi = max(float(q.max()), lo * 10)
        specs["discharge"] = {"label": "Discharge", "kind": "log", "lo": lo, "hi": hi, "unit": "", "cmap": "viridis",
                              "river_min": _pctl(ql, 97, lo * 4)}
    if "temperature" in final.ch:
        specs["temperature"] = {"label": "Temperature", "kind": "linear", "lo": -45.0, "hi": 35.0, "unit": "°C", "cmap": "thermal"}
    if "precip" in final.ch:
        p = final.ch["precip"]
        specs["precip"] = {"label": "Precipitation", "kind": "log", "lo": 0.02, "hi": max(_pctl(p[land], 99.5, 5.0), 0.1),
                           "unit": "× land mean", "cmap": "rain"}
    if "biome" in final.ch:
        from ..derive.biomes import NAMES
        specs["biome"] = {"label": "Biome", "kind": "category", "cmap": "biome", "names": [n.replace("_", " ") for n in NAMES]}
    if "plate" in final.ch:
        specs["plate"] = {"label": "Plates", "kind": "category", "cmap": "plates", "offset": 1}
    if "sediment" in final.ch:
        s = final.ch["sediment"]
        specs["sediment"] = {"label": "Sediment", "kind": "log", "lo": 1.0, "hi": max(_pctl(s, 99.9, 100.0), 10.0), "unit": "m", "cmap": "viridis"}
    if "crust" in final.ch:
        specs["crust"] = {"label": "Crust", "kind": "category", "cmap": "plates", "names": ["oceanic", "continental"]}
    return specs


def _byte(name: str, a: np.ndarray, specs: dict) -> np.ndarray:
    s = specs[name]
    if name == "plate":
        return (np.asarray(a, np.int64) % 256).astype(np.uint8)
    if s["kind"] == "category":
        return np.clip(np.asarray(a, np.int64), 0, 255).astype(np.uint8)
    if s["kind"] == "log":
        return log_byte(a, s["lo"], s["hi"])
    return lin_byte(a, s["lo"], s["hi"])


# textures beyond texture 0 on the final frame: (R, G, B) channel names
FINAL_TEXTURES = (("temperature", "precip", "biome"), ("plate", "sediment", "crust"))


def encode_frame(fr: _Frame, specs: dict) -> tuple[list[np.ndarray], dict]:
    """-> ([atlas images], frame meta)."""
    hi, lo, h0, h1 = encode_height(pad_faces(fr.height))
    zero = np.zeros_like(hi)
    over = "plate" if fr.stage == "tectonics" else "discharge"
    b = pad_faces(_byte(over, fr.ch[over], specs)) if over in fr.ch and over in specs else zero
    images = [atlas([hi, lo, b])]
    layers = {over: [0, 2]} if over in fr.ch and over in specs else {}
    if fr.stage == "final":
        for names in FINAL_TEXTURES:
            present = [n for n in names if n in fr.ch and n in specs]
            if not present:
                continue
            chans = [pad_faces(_byte(n, fr.ch[n], specs)) if n in present else zero for n in names]
            k = len(images)
            images.append(atlas(chans))
            for c, n in enumerate(names):
                if n in present:
                    layers[n] = [k, c]
    meta = {"stage": fr.stage, "key": int(fr.key), "label": fr.label, "res": fr.res, "pad": PAD,
            "h0": h0, "h1": h1, "layers": layers, "stats": frame_stats(fr.height)}
    return images, meta


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------
def export_viewer(world_dir, out=None, *, formats: str = "", final_res: int | None = None,
                  frame_res: int | None = None, max_frames: int | None = None,
                  single: bool = False, log=print) -> Path:
    """``frame_res`` / ``max_frames`` downsample and thin the captured
    timeline -- a light export for a slow link or a phone."""
    t0 = time.time()
    root = Path(world_dir)
    out = Path(out) if out else root / "viewer"
    frames, manifest = collect_frames(root, final_res or _render_param(manifest_of(root), "viewer_final_res", 1024), log,
                                      frame_res=frame_res, max_frames=max_frames)
    final = frames[-1]
    specs = channel_specs(final)
    land_final = final.height[final.height >= 0]
    land_top = max(_pctl(land_final, 99.5, 1000.0), 200.0)
    ocean_bottom = min(_pctl(final.height[final.height < 0], 1.0, -4000.0), -200.0)
    default_exag = relief_exaggeration(final.height, float(manifest.get("cell_size_m", 1.0)) * int(manifest.get("N_c", final.res)) / final.res)

    shutil.rmtree(out / "data", ignore_errors=True)
    (out / "data").mkdir(parents=True, exist_ok=True)
    metas, scripts, sizes = [], [], 0
    for i, fr in enumerate(frames):
        images, meta = encode_frame(fr, specs)
        payload = "GLOBE_VIEWER.frame(%d, [%s]);\n" % (i, ",".join('"%s"' % webp_b64(im) for im in images))
        sizes += len(payload)
        meta["file"] = f"data/f{i:04d}.js"
        if len(images) > 1:
            meta["tex"] = [{"res": fr.res, "pad": PAD} for _ in images]
        (out / meta["file"]).write_text(payload)
        if single:
            scripts.append(payload)
        metas.append(meta)

    disc = specs.get("discharge")
    meta = {
        "world": root.resolve().name,
        "N_c": manifest.get("N_c"), "cell_size_m": manifest.get("cell_size_m"), "seed": manifest.get("seed"),
        "R_planet_m": float(manifest.get("R_planet") or manifest.get("N_c", 1024) * manifest.get("cell_size_m", 9773.0) * 2 / math.pi),
        "sea_level_m": 0.0, "land_top_m": land_top, "ocean_bottom_m": ocean_bottom, "default_exag": default_exag,
        "channels": specs,
        "river_byte_min": int(log_byte(np.array([disc["river_min"]]), disc["lo"], disc["hi"])[0]) if disc else 255,
        "biome_palette": _biome_palette() if "biome" in specs else [],
        "stage_seconds": {s: round(v.get("seconds", 0.0), 1) for s, v in manifest.get("stages", {}).items()},
        "exported": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "final_index": len(frames) - 1,
        "frames": metas,
    }
    meta_js = "GLOBE_VIEWER.setMeta(%s);\n" % json.dumps(meta, separators=(",", ":"))
    (out / "data" / "meta.js").write_text(meta_js)
    html = TEMPLATE.read_text()
    (out / "index.html").write_text(html.replace(PLACEHOLDER, '<script src="data/meta.js"></script>'))
    if single:
        inline_meta = json.loads(json.dumps(meta))
        for m in inline_meta["frames"]:
            m["file"] = None
        blocks = ["<script>GLOBE_VIEWER.setMeta(%s);</script>" % json.dumps(inline_meta, separators=(",", ":"))]
        blocks += ["<script>%s</script>" % s for s in scripts]
        (out / "standalone.html").write_text(html.replace(PLACEHOLDER, "\n".join(blocks)))
    log(f"[viewer] {len(frames)} frames, {sizes / 1e6:.1f} MB -> {out / 'index.html'} in {time.time() - t0:.1f}s")

    fmts = {f.strip() for f in (formats or "").split(",") if f.strip()}
    if fmts:
        export_extras(frames, specs, meta, out / "exports", fmts, log)
    return out / "index.html"


def manifest_of(root: Path) -> dict:
    return json.loads((Path(root) / "manifest.json").read_text())


def _render_param(manifest: dict, key: str, default):
    return (manifest.get("params", {}).get("render", {}) or {}).get(key, default)


def _biome_palette() -> list:
    from ..derive.biomes import PALETTE
    return PALETTE.astype(int).tolist()


# --------------------------------------------------------------------------
# extra formats: equirectangular PNGs, animated flat map
# --------------------------------------------------------------------------
def _ramp(t: np.ndarray, stops: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0) * (len(stops) - 1)
    k = np.minimum(t.astype(np.int64), len(stops) - 2)
    f = (t - k)[..., None]
    return stops[k] * (1 - f) + stops[k + 1] * f


def elevation_rgb(h: np.ndarray, land_top: float, ocean_bottom: float) -> np.ndarray:
    sea = _ramp(np.power(np.clip(-h / -ocean_bottom, 0, 1), 0.6), SEA)
    land = _ramp(np.power(np.clip(h / land_top, 0, 1), 0.75), LAND)
    return np.where((h < 0)[..., None], sea, land)


@lru_cache(maxsize=4)
def _equirect_map(width: int, res: int, pad: int):
    """``(face, x, y)`` atlas-tile coordinates of every lat/lon pixel."""
    H, W = width // 2, width
    lat = (0.5 - (np.arange(H) + 0.5) / H) * np.pi
    lon = ((np.arange(W) + 0.5) / W - 0.5) * 2 * np.pi
    cl = np.cos(lat)[:, None]
    p = np.stack([cl * np.cos(lon)[None, :], cl * np.sin(lon)[None, :], np.broadcast_to(np.sin(lat)[:, None], (H, W))], -1)
    f, u, v = from_sphere_v(p)
    return f, u * res - 0.5 + pad, v * res - 0.5 + pad


def equirect(padded: np.ndarray, width: int, pad: int = PAD, nearest: bool = False) -> np.ndarray:
    """Sample a padded ``(6, T, T)`` face array on a ``width x width/2``
    lat/lon grid (pole on +Z, longitude 0 at +X, north at the top)."""
    res = padded.shape[1] - 2 * pad
    f, x, y = _equirect_map(width, res, pad)
    if nearest:
        return padded[f, np.clip(np.round(x).astype(np.int64), 0, res + 2 * pad - 1), np.clip(np.round(y).astype(np.int64), 0, res + 2 * pad - 1)]
    x0 = np.clip(np.floor(x).astype(np.int64), 0, res + 2 * pad - 2)
    y0 = np.clip(np.floor(y).astype(np.int64), 0, res + 2 * pad - 2)
    fx, fy = x - x0, y - y0
    a = padded.astype(np.float64)
    return ((a[f, x0, y0] * (1 - fx) + a[f, x0 + 1, y0] * fx) * (1 - fy)
            + (a[f, x0, y0 + 1] * (1 - fx) + a[f, x0 + 1, y0 + 1] * fx) * fy)


def relief(h: np.ndarray, R: float, land_top: float, ocean_bottom: float, exag: float = 12.0,
           river: np.ndarray | None = None) -> np.ndarray:
    """Shaded relief of an equirect height map, lit as in the shader."""
    H, W = h.shape
    lat = (0.5 - (np.arange(H) + 0.5) / H) * np.pi
    dphi, dlam = np.pi / H, 2 * np.pi / W
    gx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) / (2 * R * dlam * np.maximum(np.cos(lat), 1e-3)[:, None])
    hp = np.pad(h, ((1, 1), (0, 0)), mode="edge")
    gy = (hp[:-2] - hp[2:]) / (2 * R * dphi)
    ex = np.where(h < 0, 0.35 * exag, exag)
    n = np.stack([-gx * ex, -gy * ex, np.ones_like(h)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    shade = np.clip(n @ LIGHT / LIGHT[2], 0, 2)
    col = elevation_rgb(h, land_top, ocean_bottom) * (0.15 + 0.85 * shade)[..., None]
    if river is not None:
        m = river & (h >= 0)
        col[m] = col[m] * 0.35 + np.array([0.16, 0.40, 0.90]) * 0.65
    return (np.clip(col, 0, 1) ** 0.95 * 255).astype(np.uint8)


def export_extras(frames: list[_Frame], specs: dict, meta: dict, out: Path, fmts: set, log=print) -> None:
    from PIL import Image

    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    R, top, bot = meta["R_planet_m"], meta["land_top_m"], meta["ocean_bottom_m"]
    final = frames[-1]
    if "equirect" in fmts:
        W = min(8192, 4 * final.res)
        h = equirect(pad_faces(final.height), W)
        h0, h1 = float(np.floor(h.min())), float(np.ceil(h.max()))
        q = np.round((h - h0) / (h1 - h0) * 65535).astype(np.uint16)
        Image.fromarray(q).save(out / "equirect_height16.png")
        (out / "equirect_height16.json").write_text(json.dumps({"h0_m": h0, "h1_m": h1, "encoding": "h0 + v/65535*(h1-h0)",
                                                                 "projection": "equirectangular, pole +Z, lon 0 = +X, north up"}))
        river = None
        if "discharge" in final.ch:
            river = equirect(pad_faces(final.ch["discharge"].astype(np.float32)), W, nearest=True) > specs["discharge"]["river_min"]
        Image.fromarray(relief(h, R, top, bot, river=river)).save(out / "equirect_relief.png")
        for name, s in specs.items():
            if name == "discharge":
                continue
            b = equirect(pad_faces(_byte(name, final.ch[name], specs)), W, nearest=True)
            if s["kind"] == "category":
                rgb = (np.array(meta["biome_palette"], np.uint8)[np.minimum(b, len(meta["biome_palette"]) - 1)]
                       if name == "biome" else _hash_rgb(b))
            else:
                rgb = (_ramp(b / 255.0, CMAPS.get(s["cmap"], CMAPS["viridis"])) * 255).astype(np.uint8)
            Image.fromarray(rgb).save(out / f"equirect_{name}.png")
        log(f"[viewer] equirect {W}x{W // 2} PNGs -> {out}")
    if "anim" in fmts:
        W = 1024
        imgs = []
        disc = specs.get("discharge")
        for fr in frames:
            h = equirect(pad_faces(fr.height), W)
            river = None
            if disc and "discharge" in fr.ch:
                river = equirect(pad_faces(fr.ch["discharge"].astype(np.float32)), W, nearest=True) > disc["river_min"]
            imgs.append(Image.fromarray(relief(h, R, top, bot, river=river)))
        dur = [100] * (len(imgs) - 1) + [2000]
        imgs[0].save(out / "timeline.webp", save_all=True, append_images=imgs[1:], duration=dur, loop=0, quality=80, method=4)
        log(f"[viewer] timeline.webp ({len(imgs)} frames) -> {out}")
    log(f"[viewer] extras in {time.time() - t0:.1f}s")


def _hash_rgb(b: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(7)
    pal = (rng.uniform(0.25, 0.9, (256, 3)) * 255).astype(np.uint8)
    pal[0] = 30
    return pal[b]
