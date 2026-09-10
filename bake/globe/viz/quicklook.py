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


def _ring(h: np.ndarray) -> np.ndarray:
    """(6, N, N) -> (6, N+2, N+2) with one ring of true neighbour values.

    Without it the outermost row and column of every face get a one-sided
    (here: zero) gradient, so each face is rimmed with flat shading. On the
    unfolded net that reads as a faint border; on a globe it is a dark seam
    tracing the cube, which is the one shape the render exists to hide.
    """
    from ..cubesphere import get_grid
    from ..field import FaceField

    return FaceField.from_interior(get_grid(h.shape[-1], 1), h, exchange=True).data


def hillshade(h: np.ndarray, cell_size: float = 1.0, azimuth_deg: float = 315.0, altitude_deg: float = 45.0, z_factor: float = 1.0) -> np.ndarray:
    """Lambertian hillshade of a (…, N, N) height array indexed [i, j]
    (i = x/east, j = y/south in image terms). Returns [0, 1] floats.

    A (6, N, N) array is shaded as a **sphere**: the surface normal is built
    in world space from the cell's own 3-D tangent frame and lit by a single
    fixed direction. Shading per face in face-local (i, j) instead gives each
    face its own idea of where the light is, so the tone jumps at every cube
    edge — invisible on the unfolded net, and on a globe a hard seam tracing
    the cube, which is the one shape the render exists to hide. (The gradients
    also need a halo, or each face is rimmed with flat shading.)
    """
    h = np.asarray(h, dtype=np.float64) * z_factor
    if h.ndim == 3 and h.shape[0] == 6:
        from ..cubesphere import get_grid

        g = get_grid(h.shape[-1], 1)
        p = g.centers.astype(np.float64)          # (6, N+2, N+2, 3) unit vectors
        q = _ring(h)
        r = p[:, 1:-1, 1:-1]
        ti = p[:, 2:, 1:-1] - p[:, :-2, 1:-1]
        tj = p[:, 1:-1, 2:] - p[:, 1:-1, :-2]

        def ortho(t):
            t = t - r * np.sum(t * r, axis=-1, keepdims=True)
            return t / np.maximum(np.linalg.norm(t, axis=-1, keepdims=True), 1e-12)

        xh, yh = ortho(ti), ortho(tj)
        dhx = (q[:, 2:, 1:-1] - q[:, :-2, 1:-1]) / (2.0 * cell_size)
        dhy = (q[:, 1:-1, 2:] - q[:, 1:-1, :-2]) / (2.0 * cell_size)
        n = r - dhx[..., None] * xh - dhy[..., None] * yh
        n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
        az, alt = np.deg2rad(azimuth_deg), np.deg2rad(altitude_deg)
        # a fixed world-space light: up-ish, from the -z/+x quarter
        L = np.array([np.cos(alt) * np.sin(az), np.sin(alt), np.cos(alt) * np.cos(az)])
        L = L / np.linalg.norm(L)
        return np.clip(np.sum(n * L, axis=-1) * 0.5 + 0.5, 0.0, 1.0)
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


def to_globe(img_fij: np.ndarray, size: int = 1024, lon0: float = 0.0, lat0: float = 20.0,
             two: bool = True, bg: int = 12) -> np.ndarray:
    """(6, N, N[, C]) [f, i, j] -> orthographic globe(s).

    The unfolded net is a debugging view: it shows all six faces at once at
    the cost of showing a shape the world does not have, and every quantity
    that matters here (how a continent is shaped, where a rift runs, whether
    an orogen is a belt or a blob) is distorted differently on each face.
    A globe is the thing itself.

    With `two`, draws the far side beside the near one, so a whole world is
    visible in a single image -- otherwise half of every world is
    permanently behind the camera.
    """
    from ..cubesphere import from_sphere_v

    def hemisphere(lon_c, lat_c):
        y, x = np.mgrid[0:size, 0:size]
        # unit disc, y up
        xs = (x - size / 2.0 + 0.5) / (size / 2.0 - 1.0)
        ys = -(y - size / 2.0 + 0.5) / (size / 2.0 - 1.0)
        r2 = xs * xs + ys * ys
        on = r2 <= 1.0
        zs = np.sqrt(np.clip(1.0 - r2, 0.0, None))
        la, lo = np.radians(lat_c), np.radians(lon_c)
        # rotate the disc's (x, y, z) into world coordinates
        ex = np.array([np.cos(lo), 0.0, -np.sin(lo)])
        ey = np.array([-np.sin(la) * np.sin(lo), np.cos(la), -np.sin(la) * np.cos(lo)])
        ez = np.array([np.cos(la) * np.sin(lo), np.sin(la), np.cos(la) * np.cos(lo)])
        p = (xs[..., None] * ex + ys[..., None] * ey + zs[..., None] * ez)
        n = np.linalg.norm(p, axis=-1, keepdims=True)
        p = p / np.maximum(n, 1e-12)
        f, u, v = from_sphere_v(np.ascontiguousarray(p.reshape(-1, 3)))
        N = img_fij.shape[1]
        # Grid.uv_cell maps u -> i and v -> j; swapping them transposes every
        # face, which on a globe reads as the data being wrong rather than the
        # indexing
        i = np.clip((u * N).astype(np.int64), 0, N - 1)
        j = np.clip((v * N).astype(np.int64), 0, N - 1)
        out = img_fij[f, i, j].reshape(size, size, -1) if img_fij.ndim == 4 else \
              img_fij[f, i, j].reshape(size, size, 1)
        out = np.where(on[..., None], out, np.uint8(bg))
        return out.astype(np.uint8)

    near = hemisphere(lon0, lat0)
    if not two:
        return near[..., 0] if near.shape[-1] == 1 else near
    far = hemisphere(lon0 + 180.0, lat0)
    gap = np.full((size, max(8, size // 64), near.shape[-1]), np.uint8(bg))
    out = np.concatenate([near, gap, far], axis=1)
    return out[..., 0] if out.shape[-1] == 1 else out


def save_image(path: str | Path, img: np.ndarray, net: bool = True, max_size: int = 4096,
               view: str = "globe", globe_size: int = 1024) -> Path:
    """Save a (6, N, N, 3) [f, i, j] image.

    `view` is "globe" (two orthographic hemispheres, the default), "net"
    (the unfolded cube, for debugging face seams and indexing) or "row".
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if img.ndim == 4 and img.shape[0] == 6:
        if view == "globe":
            out = to_globe(img, size=globe_size)
        elif view == "row" or not net:
            out = np.concatenate([face_image(img[f]) for f in range(6)], axis=1)
        else:
            out = to_net(img)
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
