"""The zooms a world has: ``<world>/zoom/index.json`` and the globe viewer's
``<world>/viewer/zooms.js`` (a ``<script>``, because the viewer runs from
file:// where fetch is not allowed).

A zoom is listed once its ``zoom.json`` and ``view.html`` exist.  Each
record carries the corners of every level's product square as latitude /
longitude, so the viewer can outline them without knowing the cube-sphere
grid, and the path of its pages relative to ``viewer/``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _corners(N: int, geo: dict) -> list[list[float]]:
    from ..cubesphere import to_sphere_v

    i0, j0, n = geo["ci0"], geo["cj0"], geo["cells"]
    ii = np.array([i0, i0 + n, i0 + n, i0], dtype=np.float64)
    jj = np.array([j0, j0, j0 + n, j0 + n], dtype=np.float64)
    p = to_sphere_v(np.full(4, int(geo["face"])), ii / N, jj / N)
    lat = np.degrees(np.arcsin(np.clip(p[:, 2], -1.0, 1.0)))
    lon = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
    return [[round(float(a), 5), round(float(b), 5)] for a, b in zip(lat, lon)]


def scan(root: str | Path) -> list[dict]:
    root = Path(root)
    zdir = root / "zoom"
    if not zdir.is_dir():
        return []
    try:
        N = int(json.loads((root / "manifest.json").read_text()).get("N_c"))
    except (OSError, ValueError, TypeError):
        return []
    out = []
    for info_path in sorted(zdir.glob("*/zoom.json")):
        d = info_path.parent
        if not (d / "view.html").exists():
            continue
        info = json.loads(info_path.read_text())
        levels = [{"R": lv["R"], "cell_m": lv["cell_m"], "corners": _corners(N, lv["geometry"]),
                   "href": f"../zoom/{d.name}/L{lv['R']}.html"} for lv in info.get("levels", [])]
        out.append({"name": info.get("name", d.name), "spot": info.get("spot"), "lat": info.get("lat"), "lon": info.get("lon"),
                    "href": f"../zoom/{d.name}/view.html", "levels": levels, "seconds": info.get("seconds")})
    return out


def write(root: str | Path) -> list[dict]:
    """Rewrite ``zoom/index.json`` (when there are zooms) and
    ``viewer/zooms.js`` (always, when the viewer exists)."""
    root = Path(root)
    zooms = scan(root)
    if zooms:
        (root / "zoom" / "index.json").write_text(json.dumps(zooms, indent=1))
    if (root / "viewer").is_dir():
        (root / "viewer" / "zooms.js").write_text(script(zooms))
    return zooms


def script(zooms: list[dict]) -> str:
    return "GLOBE_VIEWER.setZooms(%s);\n" % json.dumps(zooms, separators=(",", ":"))


__all__ = ["scan", "write", "script"]
