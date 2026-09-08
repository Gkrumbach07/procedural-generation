"""Quicklook PNGs: hillshade, scalar colour maps, overlays, unfolded nets.

This is the primary debugging tool (PLAN.md section 4).  Every stage
writes ``worlds/<name>/quicklook/<stage>.png``.

All functions accept *interior* arrays indexed ``[f, i, j]`` (6, N, N)
or a :class:`FaceField`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..field import FaceField
from .unfold import face_image, unfold


def _interior(x) -> np.ndarray:
    if isinstance(x, FaceField):
        return np.asarray(x.interior)
    return np.asarray(x)


# --------------------------------------------------------------------------
# colour helpers
# --------------------------------------------------------------------------
def _viridis(t: np.ndarray) -> np.ndarray:
    """Cheap viridis-like polynomial colour map, t in [0, 1] -> (…, 3) uint8."""
    t = np.clip(t, 0.0, 1.0)
    r = 0.267 + t * (0.005 + t * (-1.3 + t * (4.9 + t * (-5.7 + t * 2.9))))
    g = 0.005 + t * (1.28 + t * (-0.53 + t * (0.2 + t * -0.05)))
    b = 0.329 + t * (1.4 + t * (-3.4 + t * (2.9 + t * -1.1)))
    return (np.clip(np.stack([r, g, b], -1), 0, 1) * 255).astype(np.uint8)


def _terrain_cmap(h: np.ndarray, sea_level: float = 0.0, hmax: float | None = None, hmin: float | None = None) -> np.ndarray:
    """Hypsometric tint: blues below sea level, greens→browns→white above.

    The tint spans ``sea_level..hmax`` on land and ``sea_level..hmin`` at
    sea; both default to the 99.5th / 0.5th percentile of *this* array, so
    a single image always uses its full range.  Pass them explicitly to
    pin the palette across a series of images (an animation, or a
    before/after pair) — otherwise every frame rescales and terrain that
    never moved appears to change.
    """
    out = np.zeros(h.shape + (3,), dtype=np.float32)
    land = h > sea_level
    if hmax is None:
        hmax = float(np.nanpercentile(h[land], 99.5)) if land.any() else 1.0
    if hmin is None:
        hmin = float(np.nanpercentile(h[~land], 0.5)) if (~land).any() else -1.0
    t = np.clip((h - sea_level) / max(hmax - sea_level, 1e-6), 0, 1)
    stops = np.array([[0.30, 0.55, 0.25], [0.55, 0.65, 0.30], [0.72, 0.60, 0.40], [0.60, 0.50, 0.45], [0.95, 0.95, 0.95]])
    pos = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    for c in range(3):
        out[..., c] = np.interp(t, pos, stops[:, c])
    td = np.clip((h - sea_level) / min(hmin - sea_level, -1e-6), 0, 1)
    ocean = np.stack([0.25 - 0.15 * td, 0.45 - 0.25 * td, 0.75 - 0.3 * td], -1)
    out[~land] = ocean[~land]
    return out


def hillshade(h: np.ndarray, cell_size: float = 1.0, azimuth_deg: float = 315.0, altitude_deg: float = 45.0, z_factor: float = 1.0) -> np.ndarray:
    """Lambertian hillshade of a (…, N, N) height array indexed [i, j]
    (i = x/east, j = y/south in image terms). Returns [0, 1] floats."""
    h = np.asarray(h, dtype=np.float64) * z_factor
    dx = np.zeros_like(h)
    dy = np.zeros_like(h)
    dx[..., 1:-1, :] = (h[..., 2:, :] - h[..., :-2, :]) / (2 * cell_size)
    dy[..., :, 1:-1] = (h[..., :, 2:] - h[..., :, :-2]) / (2 * cell_size)
    az = np.deg2rad(azimuth_deg)
    alt = np.deg2rad(altitude_deg)
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(shade, 0.0, 1.0)


def terrain_scale(h, sea_level: float = 0.0, cell_size: float = 1.0) -> dict:
    """The three scale values :func:`render_height` would derive from ``h``,
    as a kwargs dict.  Compute once and splat into every call to keep a
    series of frames on one palette and one vertical exaggeration."""
    a = _interior(h).astype(np.float64)
    land = a > sea_level
    rng = float(np.nanpercentile(a, 99) - np.nanpercentile(a, 1)) or 1.0
    return {
        "z_factor": 0.5 * a.shape[-1] * cell_size / rng,
        "hmax": float(np.nanpercentile(a[land], 99.5)) if land.any() else 1.0,
        "hmin": float(np.nanpercentile(a[~land], 0.5)) if (~land).any() else -1.0,
    }


def render_height(h, sea_level: float = 0.0, cell_size: float = 1.0, z_factor: float | None = None, shade_strength: float = 0.7,
                  hmax: float | None = None, hmin: float | None = None) -> np.ndarray:
    """Hypsometric tint × hillshade → (6, N, N, 3) uint8 indexed [f, i, j].

    ``z_factor``, ``hmax`` and ``hmin`` all default to values derived from
    this array.  Pin all three (see :func:`terrain_scale`) to compare or
    animate a series of heights on one scale."""
    h = _interior(h).astype(np.float64)
    if z_factor is None:
        rng = float(np.nanpercentile(h, 99) - np.nanpercentile(h, 1)) or 1.0
        z_factor = 0.5 * h.shape[-1] * cell_size / rng  # relief ≈ half the face width
    hs = hillshade(h, cell_size, z_factor=z_factor)
    col = _terrain_cmap(h, sea_level, hmax, hmin)
    img = col * (1.0 - shade_strength + shade_strength * hs[..., None] * 1.3)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def render_scalar(x, vmin: float | None = None, vmax: float | None = None, log: bool = False, cmap: str = "viridis", mask=None) -> np.ndarray:
    """Colour-mapped scalar → (6, N, N, 3) uint8."""
    a = _interior(x).astype(np.float64)
    if log:
        a = np.log1p(np.maximum(a, 0))
    if vmin is None:
        vmin = float(np.nanpercentile(a, 1))
    if vmax is None:
        vmax = float(np.nanpercentile(a, 99))
    t = (a - vmin) / max(vmax - vmin, 1e-12)
    if cmap == "gray":
        img = np.repeat((np.clip(t, 0, 1) * 255).astype(np.uint8)[..., None], 3, -1)
    elif cmap == "bwr":
        t = np.clip(t, 0, 1)
        img = (np.stack([np.minimum(1, 2 * t), 1 - np.abs(2 * t - 1), np.minimum(1, 2 - 2 * t)], -1) * 255).astype(np.uint8)
    else:
        img = _viridis(t)
    if mask is not None:
        img[~_interior(mask).astype(bool)] = 0
    return img


def render_labels(ids, seed: int = 0, background: int = -1) -> np.ndarray:
    """Random colour per integer label → (6, N, N, 3) uint8."""
    ids = _interior(ids).astype(np.int64)
    rng = np.random.default_rng(seed)
    lut = rng.integers(40, 230, size=(max(int(ids.max()) + 2, 2), 3), dtype=np.int64)
    img = lut[np.clip(ids, 0, lut.shape[0] - 1)].astype(np.uint8)
    img[ids == background] = 0
    return img


def render_vector(vec, stride: int = 8, scale: float = 1.0, base=None) -> np.ndarray:
    """Draw a sparse arrow field (as short line strokes) on top of ``base``
    (or a dark background).  ``vec`` is a contravariant vector field."""
    v = _interior(vec).astype(np.float64)
    F, N = v.shape[0], v.shape[1]
    img = np.zeros((F, N, N, 3), dtype=np.uint8) if base is None else base.copy()
    mag = np.hypot(v[..., 0], v[..., 1])
    m = float(np.percentile(mag, 95)) or 1.0
    L = stride * 0.8 * scale
    for f in range(F):
        for i in range(stride // 2, N, stride):
            for j in range(stride // 2, N, stride):
                a, b = v[f, i, j] / m * L
                n = int(max(abs(a), abs(b))) + 1
                for s in range(n + 1):
                    ii = int(round(i + a * s / n))
                    jj = int(round(j + b * s / n))
                    if 0 <= ii < N and 0 <= jj < N:
                        c = 255 if s == n else 180
                        img[f, ii, jj] = (c, c, 60)
    return img


def overlay(base: np.ndarray, mask, color=(40, 90, 255), alpha: float = 0.85, weight=None) -> np.ndarray:
    """Blend ``color`` into ``base`` where ``mask`` is True (``weight`` in [0,1]
    modulates alpha per cell if given).  Arrays are [f, i, j(, 3)]."""
    m = _interior(mask).astype(bool)
    out = base.astype(np.float32)
    w = np.full(m.shape, alpha, dtype=np.float32) if weight is None else np.clip(_interior(weight), 0, 1).astype(np.float32) * alpha
    col = np.array(color, dtype=np.float32)
    out[m] = out[m] * (1 - w[m, None]) + col * w[m, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def contour_lines(base: np.ndarray, ids, color=(255, 255, 255)) -> np.ndarray:
    """Draw boundaries between differing integer labels (within faces)."""
    ids = _interior(ids)
    edge = np.zeros(ids.shape, dtype=bool)
    edge[:, 1:, :] |= ids[:, 1:, :] != ids[:, :-1, :]
    edge[:, :, 1:] |= ids[:, :, 1:] != ids[:, :, :-1]
    out = base.copy()
    out[edge] = color
    return out


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------
def to_net(img_fij: np.ndarray) -> np.ndarray:
    """(6, N, N[, C]) [f, i, j] → unfolded net image [row, col(, C)]."""
    return unfold([face_image(img_fij[f]) for f in range(6)])


def save_image(path: str | Path, img: np.ndarray, net: bool = True, max_size: int = 4096) -> Path:
    """Save a (6, N, N, 3) [f, i, j] image as an unfolded net PNG (or a
    row of faces if ``net=False``).  Down-samples if wider than max_size."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if img.ndim == 4 and img.shape[0] == 6:
        out = to_net(img) if net else np.concatenate([face_image(img[f]) for f in range(6)], axis=1)
    else:
        out = img
    im = Image.fromarray(np.ascontiguousarray(out))
    if max(im.size) > max_size:
        s = max_size / max(im.size)
        im = im.resize((max(1, int(im.size[0] * s)), max(1, int(im.size[1] * s))), Image.BILINEAR)
    im.save(str(path))
    return path


def save_face(path: str | Path, img_ij: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(face_image(img_ij))).save(str(path))
    return path


def quicklook_height(path, height, sea_level=0.0, cell_size=1.0, discharge=None, water=None, net=True):
    """Standard terrain quicklook: hypsometric hillshade + optional rivers
    (discharge, log-weighted blue) and lakes (water surface > height)."""
    img = render_height(height, sea_level, cell_size)
    if water is not None:
        h = _interior(height)
        w = _interior(water)
        lake = (w - h) > 0.05
        img = overlay(img, lake, (60, 120, 220), 0.9)
    if discharge is not None:
        q = _interior(discharge).astype(np.float64)
        lq = np.log1p(np.maximum(q, 0))
        thr = float(np.percentile(lq, 97)) if lq.max() > 0 else 1.0
        wgt = np.clip((lq - thr) / max(lq.max() - thr, 1e-9), 0, 1)
        img = overlay(img, wgt > 0, (30, 80, 255), 0.95, weight=0.35 + 0.65 * wgt)
    return save_image(path, img, net=net)
