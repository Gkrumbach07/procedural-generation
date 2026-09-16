"""Scout a baked planet level for places worth refining (docs/zoom-windows.md,
"Scouting"): square windows of ``window_km`` scored for a backpacking and
fly-fishing map, best first, far enough apart to be different places.

    python scripts/scout.py --world worlds/earth-v9            # -> <world>/zoom/scout/

The fine levels keep the planet level's ridges, valleys, lakes and river
courses (each is held to its parent a few parent cells out), so what a
window scores at 1.2 km is what a zoom of it will have; the fine levels add
texture.  A window's score is the weighted geometric mean of:

* ``relief``: max - min of the surface -- mountains, not a plain or a wall;
* ``walkable``: the share of cells gentler than ``walk_slope``;
* ``streams``: the share of cells that are creeks (discharge between the
  90th and 99th percentile of land) and whether a river (above the 99th)
  runs through;
* ``gradient``: along those streams, the share at riffle-and-pool slopes and
  the share at pocket-water slopes, both wanted;
* ``lakes``: the share of the window under a lake;
* ``climate``: mean temperature, rain against the land mean, and the share
  of the window's coarse cells that are forest.

Every component is a trapezoid in ``[0, 1]`` (0 outside ``(a, d)``, 1 on
``[b, c]``); a window needs ``land_min`` of land to be considered.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

from ..config import WorldParams
from ..cubesphere import to_sphere_v
from ..io.world_store import WorldStore

FOREST = (3, 5, 6, 10, 11, 14)      # boreal, temperate, temperate rain, tropical seasonal, tropical rain, riparian


@dataclass
class ScoutSpec:
    window_km: float = 24.0
    stride: float = 0.5  # of a window
    top: int = 24
    min_separation_km: float = 150.0
    land_min: float = 0.9
    walk_slope: float = 0.18  # rise / run over a level cell
    relief_m: tuple = (250.0, 700.0, 2200.0, 3600.0)
    walkable: tuple = (0.15, 0.4, 0.9, 1.01)
    creek_share: tuple = (0.01, 0.04, 0.2, 0.4)
    riffle_share: tuple = (0.05, 0.25, 1.01, 1.02)  # streams at 0.2-2 % slope
    pocket_share: tuple = (0.02, 0.1, 0.6, 0.9)  # streams at 2-8 %
    lake_share: tuple = (0.0005, 0.004, 0.06, 0.15)
    temperature_c: tuple = (-3.0, 3.0, 13.0, 19.0)
    wetness: tuple = (0.4, 0.8, 3.0, 6.0)  # precip over the land mean
    forest_share: tuple = (0.1, 0.5, 1.01, 1.02)
    weights: dict = field(default_factory=lambda: {"relief": 1.5, "walkable": 1.0, "streams": 1.5, "gradient": 1.0, "lakes": 0.7, "climate": 1.2})


def ramp(x, a, b, c, d):
    x = np.asarray(x, np.float64)
    up = np.clip((x - a) / max(b - a, 1e-12), 0.0, 1.0)
    down = np.clip((d - x) / max(d - c, 1e-12), 0.0, 1.0)
    return np.minimum(up, down)


def _level_arrays(pdir: Path, R: int, face: int) -> dict:
    load = lambda name: np.load(pdir / f"L{R}.f{face}.{name}.npy", mmap_mode="r")
    surf = np.asarray(load("height"), np.float32) + np.asarray(load("sediment"), np.float32)
    ws = np.maximum(np.asarray(load("water_surface"), np.float32), surf)
    return {"surf": surf, "ws": ws, "q": np.asarray(load("discharge"), np.float32)}


def score_face(root: Path, pdir: Path, R: int, face: int, spec: ScoutSpec, q_levels: tuple, coarse: dict, cell_m: float) -> dict:
    """Window centres (fine indices), their components and scores on one face."""
    a = _level_arrays(pdir, R, face)
    surf, ws, q = a["surf"], a["ws"], a["q"]
    n = surf.shape[0]
    N = coarse["temperature"].shape[1]
    w = max(4, int(round(spec.window_km * 1000.0 / cell_m)))
    st = max(1, int(round(w * spec.stride)))
    ocean_c = ndimage.binary_dilation(coarse["ocean"][face], structure=np.ones((3, 3), bool))
    ocean = (surf < 0.0) & np.repeat(np.repeat(ocean_c, R, 0), R, 1)[:n, :n]
    land = ~ocean
    lake = land & (ws - surf > 2.0)
    gy, gx = np.gradient(surf, cell_m)
    slope = np.hypot(gx, gy)
    del gy, gx
    q_creek, q_river = q_levels
    creek = land & ~lake & (q >= q_creek) & (q < q_river)
    river = land & ~lake & (q >= q_river)
    stream = creek | river
    riffle = stream & (slope >= 0.002) & (slope < 0.02)
    pocket = stream & (slope >= 0.02) & (slope < 0.08)
    walk = land & (slope < spec.walk_slope)
    mean = lambda m: ndimage.uniform_filter(m.astype(np.float32), size=w, mode="nearest")
    c = np.arange(w // 2, n - w // 2, st)
    pick = lambda arr: arr[np.ix_(c, c)]
    land_s = pick(mean(land))
    stream_s = np.maximum(pick(mean(stream)), 1e-9)
    comp = {
        "land": land_s,
        "walkable": pick(mean(walk)) / np.maximum(land_s, 1e-9),
        "creek_share": pick(mean(creek)) / np.maximum(land_s, 1e-9),
        "river": pick(mean(river)) * w * w,
        "riffle_share": pick(mean(riffle)) / stream_s,
        "pocket_share": pick(mean(pocket)) / stream_s,
        "lake_share": pick(mean(lake)),
    }
    hi = pick(ndimage.maximum_filter(np.where(land, surf, -1e4), size=w, mode="nearest"))
    lo = pick(ndimage.minimum_filter(np.where(land, surf, 1e4), size=w, mode="nearest"))
    comp["relief_m"] = np.maximum(hi - lo, 0.0)
    del slope, creek, river, stream, riffle, pocket, walk, lake, land, ocean
    # climate at the window's coarse cells
    ci = np.minimum(c // R, N - 1)
    k = max(1, int(round(w / (2 * R))))
    cw = lambda f: ndimage.uniform_filter(f.astype(np.float32), size=2 * k + 1, mode="nearest")[np.ix_(ci, ci)]
    comp["temperature_c"] = cw(coarse["temperature"][face])
    comp["wetness"] = cw(coarse["precip"][face]) / coarse["precip_land_mean"]
    comp["forest_share"] = cw(np.isin(coarse["biome"][face], FOREST))
    parts = {
        "relief": ramp(comp["relief_m"], *spec.relief_m),
        "walkable": ramp(comp["walkable"], *spec.walkable),
        "streams": ramp(comp["creek_share"], *spec.creek_share) * np.clip(comp["river"] / 3.0, 0.3, 1.0),
        "gradient": np.sqrt(ramp(comp["riffle_share"], *spec.riffle_share) * ramp(comp["pocket_share"], *spec.pocket_share)),
        # too few lakes is a small loss, too many (a lake plain) rules the window out
        "lakes": np.where(comp["lake_share"] > spec.lake_share[2], ramp(comp["lake_share"], *spec.lake_share), np.maximum(ramp(comp["lake_share"], *spec.lake_share), 0.3)),
        "climate": (ramp(comp["temperature_c"], *spec.temperature_c) * ramp(comp["wetness"], *spec.wetness)
                    * np.maximum(ramp(comp["forest_share"], *spec.forest_share), 0.3)) ** (1.0 / 3.0),
    }
    wsum = sum(spec.weights.values())
    logs = sum(spec.weights[kk] * np.log(1e-3 + parts[kk]) for kk in parts)
    score = np.exp(logs / wsum)
    score = np.where(comp["land"] >= spec.land_min, score, 0.0)
    return {"centres": c, "window": w, "score": score, "parts": parts, "comp": comp}


def _thumbnail(pdir: Path, R: int, face: int, i: int, j: int, w: int, path: Path, q_levels: tuple, cell_m: float) -> None:
    """The window (outlined) and two windows around it: hillshade, height
    tint, streams from creek size up, lakes and sea."""
    from PIL import Image

    a = _level_arrays(pdir, R, face)
    n = a["surf"].shape[0]
    h = 5 * w // 2
    i0, i1, j0, j1 = max(i - h, 0), min(i + h, n), max(j - h, 0), min(j + h, n)
    s = a["surf"][i0:i1, j0:j1].astype(np.float64)
    ws = a["ws"][i0:i1, j0:j1]
    q = a["q"][i0:i1, j0:j1].astype(np.float64)
    gy, gx = np.gradient(s, cell_m)
    shade = np.clip(0.62 + (-gx + gy) * 3.0, 0.25, 1.15)
    t = np.clip(s / 3000.0, 0.0, 1.0)[..., None]
    img = (np.array([0.36, 0.52, 0.30]) * (1 - t) + np.array([0.80, 0.76, 0.68]) * t) * shade[..., None]
    lo, hi = math.log(q_levels[0]), math.log(q_levels[1] * 30.0)
    r = np.clip((np.log(np.maximum(q, 1e-12)) - lo) / (hi - lo), 0.0, 1.0)
    r = np.where(q >= q_levels[0], 0.45 + 0.55 * r, 0.0)[..., None]
    img = img * (1 - r) + np.array([0.16, 0.36, 0.78]) * r
    img[ws - s > 2.0] = [0.28, 0.48, 0.86]
    img[s < 0.0] = [0.06, 0.14, 0.34]
    b0, b1 = i - w // 2 - i0, i + w // 2 - i0
    c0, c1 = j - w // 2 - j0, j + w // 2 - j0
    for x in (b0, b1):
        if 0 <= x < img.shape[0]:
            img[x, max(c0, 0):min(c1 + 1, img.shape[1])] = [1.0, 0.85, 0.3]
    for y in (c0, c1):
        if 0 <= y < img.shape[1]:
            img[max(b0, 0):min(b1 + 1, img.shape[0]), y] = [1.0, 0.85, 0.3]
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8).transpose(1, 0, 2))
    im.resize((360, 360), Image.BICUBIC).save(path)


def scout(root: str | Path, planet: str | Path | None = None, spec: ScoutSpec = ScoutSpec(), out: str | Path | None = None, log=None) -> dict:
    """Score the planet level (default: the finest finished ``zoom/planet_R*``)
    and write ``<out>/candidates.json`` with thumbnails, and the viewer's
    ``viewer/scout.js``.  Returns the record."""
    t0 = time.time()
    root = Path(root)
    params = WorldParams.from_dict(WorldStore(root).manifest["params"])
    grid = params.coarse_grid()
    N = grid.N
    if planet is None:
        found = sorted((int(p.name.split("_R")[1]), p) for p in (root / "zoom").glob("planet_R*")
                       if (p / "planet.json").exists() and p.name.split("_R")[1].isdigit())
        if not found:
            raise FileNotFoundError(f"no finished planet level under {root / 'zoom'}")
        pdir = found[-1][1]
    else:
        pdir = root / planet if not Path(planet).is_absolute() else Path(planet)
    info = json.loads((pdir / "planet.json").read_text())
    R = int(info["level"]["R"])
    cell_m = grid.cell_size_m / R
    out = Path(out) if out is not None else root / "zoom" / "scout"
    out.mkdir(parents=True, exist_ok=True)
    load6 = lambda name: np.stack([np.load(root / "coarse" / f"{name}.f{f}.npy") for f in range(6)])
    height_c = load6("height")
    coarse = {"temperature": load6("temperature").astype(np.float32), "precip": load6("precip").astype(np.float32),
              "biome": load6("biome"), "ocean": load6("flow_dir") == 255}
    coarse["precip_land_mean"] = float(coarse["precip"][height_c > 0].mean())
    # discharge levels of land over the whole planet level
    samples = []
    for f in range(6):
        q = np.load(pdir / f"L{R}.f{f}.discharge.npy", mmap_mode="r")[::7, ::7]
        s = np.load(pdir / f"L{R}.f{f}.height.npy", mmap_mode="r")[::7, ::7]
        samples.append(np.asarray(q)[np.asarray(s) > 0])
    qs = np.concatenate(samples)
    qs = qs[qs > 0]
    q_levels = (float(np.percentile(qs, 90)), float(np.percentile(qs, 99)))
    cands = []
    for f in range(6):
        r = score_face(root, pdir, R, f, spec, q_levels, coarse, cell_m)
        c = r["centres"]
        order = np.argsort(-r["score"].ravel())[: spec.top * 40]
        for k in order:
            a, b = divmod(int(k), c.size)
            sc = float(r["score"][a, b])
            if sc <= 0.0:
                break
            cands.append({"face": f, "i": int(c[a]), "j": int(c[b]), "window": r["window"], "score": sc,
                          "parts": {kk: round(float(v[a, b]), 3) for kk, v in r["parts"].items()},
                          "measures": {kk: round(float(v[a, b]), 4) for kk, v in r["comp"].items()}})
        if log is not None:
            log(f"scout: face {f} scored ({len(cands)} candidates so far)")
    cands.sort(key=lambda d: -d["score"])
    n = N * R
    chosen, dirs = [], []
    min_cos = math.cos(spec.min_separation_km * 1000.0 / grid.R_planet)
    for cand in cands:
        p = to_sphere_v(np.array([cand["face"]]), np.array([(cand["i"] + 0.5) / n]), np.array([(cand["j"] + 0.5) / n]))[0]
        if any(float(np.dot(p, d)) > min_cos for d in dirs):
            continue
        dirs.append(p)
        cand["lat"] = round(math.degrees(math.asin(max(-1.0, min(1.0, p[2])))), 4)
        cand["lon"] = round(math.degrees(math.atan2(p[1], p[0])), 4)
        cand["spot"] = [cand["face"], min(cand["i"] // R, N - 1), min(cand["j"] // R, N - 1)]
        cand["rank"] = len(chosen) + 1
        cand["thumbnail"] = f"cand_{cand['rank']:02d}.png"
        _thumbnail(pdir, R, cand["face"], cand["i"], cand["j"], cand["window"], out / cand["thumbnail"], q_levels, cell_m)
        chosen.append(cand)
        if len(chosen) >= spec.top:
            break
    record = {"world": root.name, "planet": pdir.name, "cell_m": cell_m, "spec": asdict(spec), "discharge_levels": q_levels,
              "seconds": round(time.time() - t0, 1), "candidates": chosen}
    (out / "candidates.json").write_text(json.dumps(record, indent=1))
    if (root / "viewer").is_dir():
        view = [{k: c_[k] for k in ("rank", "score", "lat", "lon", "spot", "parts", "measures")} | {"thumbnail": f"../zoom/{out.name}/{c_['thumbnail']}", "window_km": spec.window_km}
                for c_ in chosen]
        (root / "viewer" / "scout.js").write_text("GLOBE_VIEWER.setScout(%s);\n" % json.dumps(view, separators=(",", ":")))
    return record


__all__ = ["ScoutSpec", "ramp", "score_face", "scout"]
