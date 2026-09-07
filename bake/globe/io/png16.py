"""16-bit and 8-bit PNG helpers (Pillow)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def write_png16(path: str | Path, arr: np.ndarray) -> None:
    """Write a (rows, cols) uint16 array as a 16-bit grayscale PNG."""
    a = np.ascontiguousarray(arr, dtype=np.uint16)
    if a.ndim != 2:
        raise ValueError("write_png16 expects a 2-D array")
    Image.fromarray(a).save(str(path), format="PNG", compress_level=6)


def read_png16(path: str | Path) -> np.ndarray:
    im = Image.open(str(path))
    if im.mode not in ("I;16", "I;16B", "I;16L", "I"):
        im = im.convert("I")
    a = np.array(im)
    return a.astype(np.uint16)


def write_png8(path: str | Path, arr: np.ndarray) -> None:
    """Write (rows, cols) L, (rows, cols, 3) RGB or (rows, cols, 4) RGBA uint8."""
    a = np.ascontiguousarray(arr, dtype=np.uint8)
    if a.ndim == 2:
        mode = "L"
    elif a.shape[2] == 3:
        mode = "RGB"
    elif a.shape[2] == 4:
        mode = "RGBA"
    else:
        raise ValueError("unsupported channel count")
    Image.fromarray(a, mode=mode).save(str(path), format="PNG", compress_level=6)


def read_png8(path: str | Path) -> np.ndarray:
    return np.array(Image.open(str(path)))


def normalize_u16(arr: np.ndarray, lo: float | None = None, hi: float | None = None, reserve_zero: bool = False) -> tuple[np.ndarray, float, float]:
    """Map a float array to uint16 by (lo, hi); returns (u16, lo, hi).
    With ``reserve_zero`` the range maps to 1..65535 so 0 can mean "none"."""
    a = np.asarray(arr, dtype=np.float64)
    lo = float(np.nanmin(a)) if lo is None else float(lo)
    hi = float(np.nanmax(a)) if hi is None else float(hi)
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    q = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    if reserve_zero:
        return (1 + np.rint(q * 65534.0)).astype(np.uint16), lo, hi
    return np.rint(q * 65535.0).astype(np.uint16), lo, hi


def denormalize_u16(u16: np.ndarray, lo: float, hi: float, reserve_zero: bool = False) -> np.ndarray:
    if reserve_zero:
        return ((u16.astype(np.float32) - 1.0) / 65534.0) * np.float32(hi - lo) + np.float32(lo)
    return (u16.astype(np.float32) / 65535.0) * np.float32(hi - lo) + np.float32(lo)
