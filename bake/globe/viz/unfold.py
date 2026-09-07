"""Unfold the six faces into a cross-shaped net image.

Layout (image rows top to bottom)::

              [+Y]
        [-X]  [+Z]  [+X]  [-Z]
              [-Y]

Face orientations in the net are *derived*, not hand-coded: for every face
we try the eight dihedral transforms of its image and keep the one whose
edge cells coincide in 3-D with the already-placed neighbour's edge cells.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..cubesphere import Grid

# slot (row, col) in a 3x4 grid of face squares
SLOTS = {4: (1, 1), 0: (1, 2), 1: (1, 0), 2: (0, 1), 3: (2, 1), 5: (1, 3)}
# placement order and the neighbour each face must match against
ORDER = [(4, None), (0, 4), (1, 4), (2, 4), (3, 4), (5, 0)]


def _transforms():
    ops = []
    for k in range(4):
        ops.append((k, False))
        ops.append((k, True))
    return ops


def apply_transform(img: np.ndarray, op: tuple[int, bool]) -> np.ndarray:
    k, flip = op
    out = np.rot90(img, k, axes=(0, 1))
    if flip:
        out = out[:, ::-1]
    return out


def face_image(arr_ij: np.ndarray) -> np.ndarray:
    """Face array indexed [i, j] -> image indexed [row=j, col=i]."""
    if arr_ij.ndim == 2:
        return arr_ij.T
    return np.transpose(arr_ij, (1, 0, *range(2, arr_ij.ndim)))


@lru_cache(maxsize=None)
def net_transforms() -> dict[int, tuple[int, bool]]:
    """Per-face dihedral op (rot90 count, flip) so edges line up in the net."""
    g = Grid(16, 4)
    C = g.interior_centers  # (6, N, N, 3) indexed [f, i, j]
    imgs = {f: face_image(C[f]) for f in range(6)}  # [row, col, 3]
    placed: dict[int, np.ndarray] = {}
    ops: dict[int, tuple[int, bool]] = {}
    for f, nb in ORDER:
        if nb is None:
            ops[f] = (0, False)
            placed[f] = imgs[f]
            continue
        (r0, c0), (r1, c1) = SLOTS[nb], SLOTS[f]
        best = None
        for op in _transforms():
            im = apply_transform(imgs[f], op)
            nbim = placed[nb]
            if c1 == c0 + 1:  # f right of nb
                a, b = nbim[:, -1], im[:, 0]
            elif c1 == c0 - 1:  # f left of nb
                a, b = nbim[:, 0], im[:, -1]
            elif r1 == r0 - 1:  # f above nb
                a, b = nbim[0, :], im[-1, :]
            elif r1 == r0 + 1:  # f below nb
                a, b = nbim[-1, :], im[0, :]
            else:
                raise RuntimeError("slots not adjacent")
            d = float(np.linalg.norm(a - b, axis=-1).sum())
            if best is None or d < best[0]:
                best = (d, op)
        ops[f] = best[1]
        placed[f] = apply_transform(imgs[f], best[1])
    return ops


def unfold(face_images: np.ndarray | list, fill=0) -> np.ndarray:
    """``face_images``: (6, rows, cols[, C]) *image-oriented* face images
    (use :func:`face_image` first).  Returns the (3*rows, 4*cols[, C]) net."""
    imgs = [np.asarray(im) for im in face_images]
    n = imgs[0].shape[0]
    extra = imgs[0].shape[2:]
    out = np.full((3 * n, 4 * n, *extra), fill, dtype=imgs[0].dtype)
    ops = net_transforms()
    for f in range(6):
        r, c = SLOTS[f]
        out[r * n : (r + 1) * n, c * n : (c + 1) * n] = apply_transform(imgs[f], ops[f])
    return out


def unfold_field_interior(arr: np.ndarray, fill=0) -> np.ndarray:
    """Convenience: (6, N, N[, C]) array indexed [f, i, j] -> net image."""
    return unfold([face_image(arr[f]) for f in range(6)], fill=fill)
