"""The zooms a world has: ``<world>/zoom/index.json`` and the globe viewer's
``<world>/viewer/zooms.js`` (a ``<script>``, because the viewer runs from
file:// where fetch is not allowed).

A zoom is listed once its ``zoom.json`` and ``view.html`` exist.  Each
record carries the corners of every level's product square as latitude /
longitude, so the viewer can outline them without knowing the cube-sphere
grid, and the path of its pages relative to ``viewer/``.  A level whose
texture exists (``viewer/zoomtex/``, :mod:`globe.viz.zoomtex`, written by
:func:`write`) also carries its ``tex`` record, read from the texture's
sidecar so a scan never opens a level's arrays.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _corners(N: int, geo: dict) -> list[list[float]]:
    from ..cubesphere import to_sphere_v
    from .bake import Geometry

    g = Geometry(**geo)
    a, b = g.product_origin          # fine cells: a fine-cell level's product need not start on a coarse cell
    i0, j0, n = a / g.R, b / g.R, g.n / g.R
    ii = np.array([i0, i0 + n, i0 + n, i0], dtype=np.float64)
    jj = np.array([j0, j0, j0 + n, j0 + n], dtype=np.float64)
    p = to_sphere_v(np.full(4, int(geo["face"])), ii / N, jj / N)
    lat = np.degrees(np.arcsin(np.clip(p[:, 2], -1.0, 1.0)))
    lon = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
    return [[round(float(a), 5), round(float(b), 5)] for a, b in zip(lat, lon)]


def _listed(root: Path) -> tuple[int, list[tuple[Path, dict]]]:
    """``(N, [(zoom dir, zoom.json)])`` of the zooms listed (both files there)."""
    zdir = root / "zoom"
    if not zdir.is_dir():
        return 0, []
    try:
        N = int(json.loads((root / "manifest.json").read_text()).get("N_c"))
    except (OSError, ValueError, TypeError):
        return 0, []
    out = []
    for info_path in sorted(zdir.glob("*/zoom.json")):
        d = info_path.parent
        if (d / "view.html").exists():
            out.append((d, json.loads(info_path.read_text())))
    return N, out


def scan(root: str | Path) -> list[dict]:
    from ..viz import zoomtex

    root = Path(root)
    N, listed = _listed(root)
    viewer = root / "viewer"
    has_viewer = viewer.is_dir()
    out = []
    for d, info in listed:
        name = info.get("name", d.name)
        levels = []
        for lv in info.get("levels", []):
            rec = {"R": lv["R"], "cell_m": lv["cell_m"], "corners": _corners(N, lv["geometry"]),
                   "href": f"../zoom/{d.name}/L{lv['R']}.html"}
            npz = d / f"L{lv['R']}.npz"
            tex = zoomtex.record(viewer, name, lv["R"], npz) if has_viewer and npz.exists() else None
            if tex is not None:
                rec["tex"] = tex
            levels.append(rec)
        out.append({"name": name, "spot": info.get("spot"), "lat": info.get("lat"), "lon": info.get("lon"),
                    "href": f"../zoom/{d.name}/view.html", "levels": levels, "seconds": info.get("seconds"),
                    # when it was baked: where two zooms overlap, the viewer draws the newer
                    "baked": round((d / "zoom.json").stat().st_mtime) if (d / "zoom.json").exists() else 0})
    return out


def write_textures(root: str | Path, force: bool = False, log=print) -> int:
    """Write the texture of every listed zoom level with an npz whose texture
    is missing or older than it (all of them with ``force``); returns how
    many were written.  A level that cannot be read is reported and skipped."""
    import zipfile

    from ..viz import zoomtex

    root = Path(root)
    viewer = root / "viewer"
    if not viewer.is_dir():
        return 0
    N, listed = _listed(root)
    written = 0
    for d, info in listed:
        name = info.get("name", d.name)
        for lv in info.get("levels", []):
            npz = d / f"L{lv['R']}.npz"
            if not npz.exists() or (not force and zoomtex.record(viewer, name, lv["R"], npz) is not None):
                continue
            try:
                zoomtex.write_level(viewer, name, d, lv["R"], N, cell_m=lv.get("cell_m"), geometry=lv.get("geometry"), force=True, log=log)
                written += 1
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as e:
                if log is not None:
                    log(f"[zoomtex] {name} L{lv['R']}: not written ({type(e).__name__}: {e})")
    return written


def write(root: str | Path, log=print) -> list[dict]:
    """Write the missing or stale level textures (when the viewer exists),
    then rewrite ``zoom/index.json`` (when there are zooms) and
    ``viewer/zooms.js`` (always, when the viewer exists)."""
    root = Path(root)
    write_textures(root, log=log)
    zooms = scan(root)
    if zooms:
        (root / "zoom" / "index.json").write_text(json.dumps(zooms, indent=1))
    if (root / "viewer").is_dir():
        (root / "viewer" / "zooms.js").write_text(script(zooms))
    return zooms


def script(zooms: list[dict]) -> str:
    return "GLOBE_VIEWER.setZooms(%s);\n" % json.dumps(zooms, separators=(",", ":"))


__all__ = ["scan", "write", "write_textures", "script"]
