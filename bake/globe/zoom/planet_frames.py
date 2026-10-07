"""The planet's own time lapse: the 1.2 km erosion, frame by frame.

A planet level erodes in tiles (:mod:`globe.zoom.planet`), each keeping its
core every few iterations (``PlanetLevel.snapshots``), block-averaged to its
share of a face's frame (``frame_res``).  :func:`build` stitches those into
whole faces -- frame ``k`` of every tile together -- and writes one
``L{R}.frames.npz`` beside the level, in the shape the viewer's timeline
reads: ``surface`` and ``discharge`` as ``(frames, 6, res, res)`` -- and
``lake``, the metres of water standing on the ground, where a tile had a
lake of the planet's in it.  A level's lakes are the planet's, at their
levels over the floors the level found, for the whole of its erosion
(:mod:`globe.zoom.parent_lakes`, :func:`globe.zoom.bake.erode_tile`): they
neither fill nor drain across the lapse, and what moves is the ground beside
them.

Tiles run in passes, so no two are at the same iteration at the same moment:
a frame holds every tile as far as it had got by that many iterations of its
own.  That is how the level was built, and it reads as the whole planet
carving itself at once -- which is what the coarse timeline shows too, an
iteration at a time rather than a clock.

Cells no tile covered (a window with no land never runs) keep the level's own
finished value: they are sea, and the sea does not move.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

#: a frame's own file, beside the level
FRAMES_NAME = "L{R}.frames.npz"
_TILE_RE = re.compile(r"^f(\d+)_(-?\d+)_(-?\d+)\.npz$")


def tile_frames(out: Path) -> list[tuple[int, int, int, Path]]:
    """``(face, ta0, tb0, path)`` of the per-tile lapses a bake left in
    ``<out>/frames``."""
    d = Path(out) / "frames"
    found = []
    for p in sorted(d.glob("f*.npz")) if d.is_dir() else []:
        m = _TILE_RE.match(p.name)
        if m:
            found.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), p))
    return found


def _face_base(out: Path, R: int, face: int, res: int) -> tuple[np.ndarray, np.ndarray]:
    """The level's finished surface and streams, reduced to ``res`` -- what a
    frame holds where no tile ran."""
    from . import planet as zp
    from .planet_finish import quicklook_rows

    n = np.load(zp.out_path(out, R, face, "height"), mmap_mode="r").shape[0]
    k = max(1, n // int(res))
    m = (n // k) * k
    surf = quicklook_rows(out, R, face, "height", k, m) + quicklook_rows(out, R, face, "sediment", k, m)
    river = "flow" if zp.out_path(out, R, face, "flow").exists() else "discharge"
    q = quicklook_rows(out, R, face, river, k, m, "max")
    return surf.astype(np.float32), q.astype(np.float32)


def build(out: str | Path, R: int | None = None, log=None) -> Path | None:
    """Stitch ``<out>/frames/f*.npz`` into ``<out>/L{R}.frames.npz``.  Returns
    the path, or None when the bake kept no frames."""
    out = Path(out)
    meta = json.loads((out / "planet.json").read_text()) if (out / "planet.json").exists() else {}
    if R is None:
        R = int((meta.get("level") or {}).get("R") or 0)
    if not R:
        raise ValueError(f"{out}: no planet.json to take R from")
    tiles = tile_frames(out)
    if not tiles:
        return None
    with np.load(tiles[0][3]) as z:
        iterations = np.asarray(z["iterations"], np.int32)
        nfr = len(iterations)
        factor = int(z["factor"][0])
    n_fine = np.load((out / f"L{R}.f0.height.npy"), mmap_mode="r").shape[0]
    res = n_fine // factor
    surface = np.empty((nfr, 6, res, res), np.float16)
    discharge = np.empty((nfr, 6, res, res), np.float16)
    lake = None                                       # made when the first tile with a lake turns up
    for f in range(6):
        base_s, base_q = _face_base(out, R, f, res)
        surface[:, f] = base_s.astype(np.float16)
        discharge[:, f] = np.minimum(base_q, 65000.0).astype(np.float16)
    for face, ta0, tb0, path in tiles:
        with np.load(path) as z:
            surf, q, core = z["surface"], z["discharge"], z["core"]
            wet = z["lake"] if "lake" in z.files else None
        a0, b0 = int(core[0]) // factor, int(core[1]) // factor
        a1, b1 = a0 + surf.shape[1], b0 + surf.shape[2]
        if a1 > res or b1 > res:                      # a level whose faces do not divide evenly
            surf, q = surf[:, :res - a0, :res - b0], q[:, :res - a0, :res - b0]
            wet = None if wet is None else wet[:, :res - a0, :res - b0]
            a1, b1 = min(a1, res), min(b1, res)
        surface[:, face, a0:a1, b0:b1] = surf[:nfr].astype(np.float16)
        discharge[:, face, a0:a1, b0:b1] = np.minimum(q[:nfr], 65000.0).astype(np.float16)
        if wet is not None:
            if lake is None:
                lake = np.zeros((nfr, 6, res, res), np.float16)
            lake[:, face, a0:a1, b0:b1] = wet[:nfr].astype(np.float16)
    path = out / FRAMES_NAME.format(R=R)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, surface=surface, discharge=discharge, iterations=iterations,
             factor=np.array([factor] * nfr, np.int32), **({} if lake is None else {"lake": lake}))
    tmp.replace(path)
    if log is not None:
        log(f"[planet] time lapse: {nfr} frames of {res}^2 a face from {len(tiles)} tiles -> {path.name}")
    return path


def load(out: str | Path, R: int) -> dict | None:
    """``{surface, discharge, iterations, factor}`` of a level's time lapse
    (and ``lake`` where it kept its lakes), or None when it has none."""
    path = Path(out) / FRAMES_NAME.format(R=int(R))
    if not path.exists():
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


__all__ = ["FRAMES_NAME", "tile_frames", "build", "load"]
