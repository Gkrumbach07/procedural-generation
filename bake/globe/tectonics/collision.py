"""Label map, gap filling, subduction, crystallisation and the grid
post-processing (cascade, Gaussian) of PLAN.md section 6.2 / 6.4.

All functions are deterministic: KD-tree queries are exact per point (so
the thread count -- :func:`kd_workers` -- never changes a result), pair
lists are sorted before they are applied, and every sequential update runs
in index order inside numba.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree

from ..cubesphere import Grid, from_sphere_v
from ..field import FaceField
from .segments import CONTINENTAL, OCEANIC, Segments, _hash_insert, _voxel_coord, _voxel_resolution, greedy_accept


# --------------------------------------------------------------------------
# label map
# --------------------------------------------------------------------------
_CENTERS_CACHE: dict = {}


def interior_centers_flat(grid: Grid) -> np.ndarray:
    """Contiguous (6*N*N, 3) copy of ``grid.interior_centers`` (cached per
    grid: the reshape of the strided halo view costs ~30 ms at N = 512)."""
    key = (id(grid), grid.N, grid.H)
    c = _CENTERS_CACHE.get(key)
    if c is None:
        c = np.ascontiguousarray(grid.interior_centers.reshape(-1, 3))
        _CENTERS_CACHE[key] = c
    return c


#: Query sets smaller than this run on the calling thread.  scipy's ``workers=-1`` starts
#: one Python thread per core for every query, whatever its size, and most of a step's
#: queries are tiny: on the Earth preset (steps 1000-1200) the heat blobs, the label map's
#: gap fallback, the spawn and the orphan weld ask about 4-120 points, five queries a step,
#: ~85 thread starts.  Measured on this box (20 cores, load ~2), a 9-NN query against the
#: 19k-segment tree: 40 points 0.03 ms on one thread against 0.72 with all; break-even near
#: 1000 points; 2000 points 1.5 against 0.7; the whole cloud 14.4 against 2.3.  Under load
#: the threads start later, so the cut sits above the quiet-box break-even.  Results are
#: exact per point, so the count changes nothing but the time
KD_SERIAL_BELOW = 2000


def kd_workers(n: int) -> int:
    """``workers`` for a cKDTree query of ``n`` points (see :data:`KD_SERIAL_BELOW`)."""
    return -1 if int(n) >= KD_SERIAL_BELOW else 1


def build_tree(seg: Segments) -> cKDTree:
    """KD-tree on the current segment positions (one per step; reuse it for
    every query of that step)."""
    return cKDTree(seg.pos)


def label_map(tree: cKDTree, grid: Grid):
    """Nearest segment of every interior cell (exact KD-tree query).
    Returns ``(idx, dist)``: ``idx`` (6, N, N) int32 segment index, ``dist``
    (6, N, N) float64 chord distance.  Every cell gets a label (there are
    no holes by construction)."""
    c = interior_centers_flat(grid)
    dist, idx = tree.query(c, k=1, workers=kd_workers(c.shape[0]))
    N = grid.N
    return idx.reshape(6, N, N).astype(np.int32), dist.reshape(6, N, N)


@njit(cache=True)
def _build_hash(pts, G):
    head = np.full(G * G * G, -1, dtype=np.int32)
    nxt = np.empty(pts.shape[0], dtype=np.int32)
    for i in range(pts.shape[0]):
        _hash_insert(head, nxt, pts, i, G)
    return head, nxt


@njit(cache=True, parallel=True)
def _nearest_kernel(q, pts, head, nxt, G, cap2):
    """Nearest hashed point of every query (exact whenever the distance² is
    < cap2; otherwise idx = -1).  Ties resolve to the lowest index, so the
    parallel loop is deterministic."""
    n = q.shape[0]
    idx = np.full(n, -1, dtype=np.int32)
    d2o = np.full(n, cap2, dtype=np.float64)
    for a in prange(n):
        qx, qy, qz = q[a, 0], q[a, 1], q[a, 2]
        ix = _voxel_coord(qx, G)
        iy = _voxel_coord(qy, G)
        iz = _voxel_coord(qz, G)
        best = cap2
        bi = -1
        for dx in range(-1, 2):
            jx = ix + dx
            if jx < 0 or jx >= G:
                continue
            for dy in range(-1, 2):
                jy = iy + dy
                if jy < 0 or jy >= G:
                    continue
                for dz in range(-1, 2):
                    jz = iz + dz
                    if jz < 0 or jz >= G:
                        continue
                    k = head[(jx * G + jy) * G + jz]
                    while k >= 0:
                        ex = pts[k, 0] - qx
                        ey = pts[k, 1] - qy
                        ez = pts[k, 2] - qz
                        d2 = ex * ex + ey * ey + ez * ez
                        if d2 < best or (d2 == best and k < bi):
                            best = d2
                            bi = k
                        k = nxt[k]
        idx[a] = bi
        d2o[a] = best
    return idx, d2o


def label_map_fast(seg: Segments, grid: Grid, cap_radius: float, tree: cKDTree | None = None):
    """Same result as :func:`label_map` (nearest segment per interior cell,
    lowest index on exact ties) but ~3x faster: a numba voxel hash resolves
    every cell whose nearest segment is closer than ``cap_radius``; the few
    remaining cells (gaps) are resolved with the KD-tree (built here if
    ``tree`` is None).  ``cap_radius`` must be >= the gap radius so gap
    detection stays exact."""
    c = interior_centers_flat(grid)
    G = _voxel_resolution(cap_radius)
    head, nxt = _build_hash(seg.pos, G)
    idx, d2 = _nearest_kernel(c, seg.pos, head, nxt, G, float(cap_radius) ** 2)
    dist = np.sqrt(d2)
    miss = idx < 0
    if miss.any():
        tree = build_tree(seg) if tree is None else tree
        dm, im = tree.query(c[miss], k=1, workers=kd_workers(int(miss.sum())))
        idx[miss] = im
        dist[miss] = dm
    N = grid.N
    return idx.reshape(6, N, N), dist.reshape(6, N, N)


def cell_area_steradians(grid: Grid) -> np.ndarray:
    """Interior cell areas in steradians (6, N, N) float64."""
    return grid.interior_cell_area.astype(np.float64) / grid.R_planet**2


def accumulate_area(seg: Segments, idx: np.ndarray, area_sr: np.ndarray, blend: float) -> None:
    """``area = (1 - blend) * area + blend * measured`` with the measured
    steradians from the label map (PLAN 6.2.1, blend 0.01 == '0.99 rolling')."""
    measured = np.bincount(idx.ravel(), weights=area_sr.ravel(), minlength=seg.M)[: seg.M]
    if blend >= 1.0:
        seg.area = measured
    else:
        seg.area = (1.0 - blend) * seg.area + blend * measured


def splat(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Per-segment values (M,) -> (6, N, N) through the label map."""
    return np.asarray(values)[idx]


#: reconstruction kernels of :class:`SmoothSplat` (``tectonics.splat_kernel``)
SPLAT_KERNELS = ("gaussian", "wendland", "tapered")


def wendland_support(sigma: float, knn: int, spacing: float) -> float:
    """Support radius ``h`` (chord, unit sphere) of the Wendland C2 kernel
    with the same *central weight* as the Gaussian of ``sigma`` truncated to
    the ``knn`` nearest segments of mean spacing ``spacing``.

    The truncated Gaussian is renormalised over the disc of radius ``R``
    holding ``knn`` segments (``pi R^2 = knn spacing^2``), so its peak is ``1
    / (2 pi sigma^2 (1 - exp(-R^2 / 2 sigma^2)))``; the 2-D Wendland C2's is
    ``7 / (pi h^2)``.  Equal peaks: ``h^2 = 14 sigma^2 (1 - exp(-R^2 / 2
    sigma^2))``, h = 3.454 sigma at sigma = 1 spacing and knn = 12.  The
    peak is what a feature one spacing wide keeps, so this width leaves the
    top of the land -- and the vertical scale derived from it -- where the
    Gaussian put it; the discrete match (equal mean nearest-neighbour
    weight over the tect cells) lands at 3.456-3.473 sigma on `small` and
    `tiny`.  Matching the *untruncated* Gaussian's second moment instead (h
    = 3.795 sigma; the half-weight radius gives 3.752) is a wider kernel
    than the one in use -- at knn = 12 the truncation leaves the Gaussian
    1.34 sigma^2 of its 2 sigma^2 -- and it raised the vertical scale
    15-22 % (docs/coast-fringe.md section 6).  ``spacing`` is the design
    spacing (``sqrt(4 pi / segments)``), so ``h`` is fixed for a world and
    does not follow the segment count through the run."""
    r2 = int(knn) * float(spacing) ** 2 / math.pi
    return float(sigma) * math.sqrt(14.0 * (1.0 - math.exp(-r2 / (2.0 * float(sigma) ** 2))))


_SPLAT_CHUNK = 1 << 16


def _wendland_chunk(tree: cKDTree, pts: np.ndarray, h: float, kk: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Wendland C2 weights (unnormalised) of the points ``pts`` from a kNN
    list of ``kk``, re-queried at twice the count while any row's farthest
    listed neighbour is not beyond ``h``; columns past the chunk's last
    in-support neighbour dropped.  Returns ``(nb, w, rows short at kk)``."""
    m = pts.shape[0]
    short0 = -1
    while True:
        d, nb = tree.query(pts, k=kk, workers=kd_workers(m))
        d = np.atleast_2d(d).reshape(m, kk)
        nb = np.atleast_2d(nb).reshape(m, kk)
        n_short = int((d[:, -1] <= h).sum())
        short0 = n_short if short0 < 0 else short0
        if n_short == 0 or kk >= tree.n:
            break
        kk = int(min(tree.n, 2 * kk))
    keep = max(1, int((d < h).sum(axis=1).max(initial=1)))
    q = d[:, :keep] / h
    np.minimum(q, 1.0, out=q)
    w = 1.0 - q
    np.power(w, 4, out=w)
    q *= 4.0
    q += 1.0
    w *= q
    return np.ascontiguousarray(nb[:, :keep]), w, short0


def splat_weights(tree: cKDTree, pts: np.ndarray, sigma: float, knn: int = 12, kernel: str = "gaussian",
                  support: float | None = None, weight_k: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Neighbour indices and normalised weights of the reconstruction
    kernel at the (n, 3) points ``pts``: ``(nb, w, covered)``, rows sorted
    by distance, ``v(p) = sum_k w_k v_k``.

    * ``gaussian``: ``exp(-d^2 / 2 sigma^2)`` over the ``knn`` nearest,
      renormalised.  Truncated: at sigma = 1 spacing the 12th neighbour
      still carries 2.6 % mean / 5.7 % max of the weight, so the value
      jumps where a segment enters or leaves the list.
    * ``tapered``: the same Gaussian times ``(1 - (d / d_k)^2)^2`` with
      ``d_k`` the ``knn``-th neighbour's distance, so the last listed
      neighbour -- the one about to be swapped -- weighs exactly zero.
    * ``wendland``: ``(1 - d/h)^4 (4 d/h + 1)`` for ``d < h``, ``h =
      support`` (chord; default :func:`wendland_support` at the spacing
      implied by ``tree.n``), gathered from a kNN list that is re-queried
      wider (per chunk of cells) while any row's farthest listed neighbour
      is not beyond ``h``, so the whole support is always inside the list;
      columns past the last in-support neighbour are dropped (zero-weight
      padding keeps the rows one width).

    ``covered`` is the share of rows whose first kNN list already covered
    the support (1.0 for the other kernels).  The nearest neighbour always
    counts (a point with nothing in support takes its value)."""
    if kernel not in SPLAT_KERNELS:
        raise ValueError(f"tectonics.splat_kernel must be one of {SPLAT_KERNELS} (got {kernel!r})")
    n = pts.shape[0]
    sigma = float(sigma)
    covered = 1.0
    if kernel == "wendland":
        h = wendland_support(sigma, knn, math.sqrt(4.0 * math.pi / tree.n)) if support is None else float(support)
        # segments in the support disc at the mean density n_seg / 4 pi, with
        # headroom for the Poisson-disc packing's local fluctuation
        expect = h * h * tree.n / 4.0
        kk = int(min(tree.n, max(int(knn), math.ceil(1.5 * expect + 16))))
        parts, n_short = [], 0
        for lo in range(0, n, _SPLAT_CHUNK):   # bounded transient memory at Earth's 6x256^2 cells
            nb_c, w_c, short = _wendland_chunk(tree, pts[lo:lo + _SPLAT_CHUNK], h, kk)
            parts.append((nb_c, w_c))
            n_short += short
        covered = 1.0 - n_short / max(n, 1)
        keep = max(p[0].shape[1] for p in parts) if parts else 1
        nb = np.zeros((n, keep), dtype=np.intp)
        w = np.zeros((n, keep), dtype=np.float64)
        lo = 0
        for nb_c, w_c in parts:
            nb[lo:lo + nb_c.shape[0], :nb_c.shape[1]] = nb_c
            w[lo:lo + w_c.shape[0], :w_c.shape[1]] = w_c
            lo += nb_c.shape[0]
    else:
        kk = min(int(knn), tree.n)
        d, nb = tree.query(pts, k=kk, workers=kd_workers(n))
        d = np.atleast_2d(d).reshape(n, kk)
        nb = np.atleast_2d(nb).reshape(n, kk)
        w = np.exp(-(d * d) / (2.0 * sigma ** 2))
        if kernel == "tapered":
            dk = d[:, -1:]
            t = np.clip(1.0 - (d / np.maximum(dk, 1e-300)) ** 2, 0.0, 1.0)
            w = w * t * t
    if weight_k is not None:
        # How much ground a segment owns, as against how smooth the reconstruction is.  A
        # segment carrying twice the extent gets twice the weight at the same distance, so the
        # boundary between two of them moves to where ext_1 exp(-d1^2/2s^2) = ext_2
        # exp(-d2^2/2s^2) -- a power (Laguerre) cell whose area follows the extent -- while
        # every kernel keeps the width the packing needs (splat_sigma_factor: a kernel
        # narrower than the cloud's own spacing shows the Poisson disc through as worms, which
        # is what shrinking sigma per segment would have done).  All extents equal multiplies
        # by exactly one, so a uniform cloud is the old blend to the bit
        w = w * np.asarray(weight_k, np.float64)[nb]
    w[:, 0] = np.maximum(w[:, 0], 1e-300)  # the nearest always counts
    return nb, w / w.sum(axis=1, keepdims=True), covered


class SmoothSplat:
    """Kernel-weighted blend of the nearest segments for every interior
    cell: ``v(cell) = sum_k w_k v_k / sum_k w_k``; by default ``w_k =
    exp(-d_k^2 / 2 sigma^2)`` over the ``knn`` nearest (``kernel``, see
    :func:`splat_weights`).  Replaces the step function of the
    nearest-segment splat by a surface that is smooth at the
    segment-spacing scale while keeping features one spacing wide (belts).
    Query once, splat many fields."""

    def __init__(self, tree: cKDTree, grid: Grid, sigma: float, knn: int = 12, kernel: str = "gaussian",
                 support: float | None = None, weight_k: np.ndarray | None = None):
        self.nb, self.w, self.support_covered = splat_weights(
            tree, interior_centers_flat(grid), sigma, knn, kernel, support, weight_k)
        self.kernel = kernel
        self.shape = (6, grid.N, grid.N)

    def __call__(self, values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=np.float64)[self.nb]
        return (v * self.w).sum(axis=1).reshape(self.shape)


# --------------------------------------------------------------------------
# crystallisation (PLAN 6.2.6)
# --------------------------------------------------------------------------
def deposit_density(T: np.ndarray, k_D: float) -> np.ndarray:
    """``D = k_D (1 - T) / (1 - k_D (1 - T))`` in [0, 1] for T in [0, 1]."""
    x = k_D * (1.0 - T)
    return x / np.maximum(1.0 - x, 1e-6)


def crystallise(seg: Segments, T: np.ndarray, growth: float, density_base: float, k_D: float, dissolution_factor: float, max_thickness: float = 0.0, min_thickness: float = 1e-3) -> None:
    """PLAN 6.2.6, one step.  Growth ``G = k_G (1 - T)(1 - T - d_b)``:
    positive -> deposit thickness ``G`` at density ``D(T)`` (faded by
    ``exp(-thickness / max_thickness)`` if > 0, so cratons thicken
    logarithmically with age instead of linearly forever); negative -> dissolve
    ``dissolution_factor * |G|`` at the segment's own density (never more
    than half its thickness).  Mass changes only here (and in
    :func:`spawn_segments`); ``age += 1`` (steps).

    **Continental crust only.**  Oceanic crust ages but does not grow: real
    ocean floor is the same ~7 km thick when it subducts as when it was
    erupted, because it is a chilled melt layer rather than an accreting
    pile.  Letting it crystallise was what made the height distribution one
    continuum -- every segment integrating the heat it drifted through, with
    nothing to hold the seafloor down at its birth thickness."""
    T = np.clip(np.asarray(T, dtype=np.float64), 0.0, 1.0)
    G = growth * (1.0 - T) * (1.0 - T - density_base)
    D = deposit_density(T, k_D)
    grow = G > 0
    if max_thickness > 0:
        fade = np.exp(-seg.thickness / max_thickness)  # asymptotic: old crust keeps (slowly) getting thicker
        G = np.where(grow, G * fade, G)
    dth = np.where(grow, G, dissolution_factor * G)
    dth = np.maximum(dth, -0.5 * seg.thickness)
    dth = np.maximum(dth, min_thickness - seg.thickness)
    dth = np.where(seg.kind == CONTINENTAL, dth, 0.0)
    dm = np.where(grow, dth * D, dth * seg.density)
    seg.mass += dm
    seg.thickness += dth
    seg.density = np.clip(seg.mass / np.maximum(seg.thickness, 1e-12), 0.0, 1.0)
    seg.mass = seg.thickness * seg.density
    seg.age += 1.0
    seg.rework += 1.0                                             # and so does the last assembly
    seg.weld = np.maximum(seg.weld - np.int16(1), np.int16(0))     # a weld wears off


# --------------------------------------------------------------------------
# gaps -> new crust (PLAN 6.2.4)
# --------------------------------------------------------------------------
def spawn_segments(seg: Segments, idx: np.ndarray, dist: np.ndarray, grid: Grid, gap_radius: float, r_min: float, rng: np.random.Generator, heat: FaceField, new_thickness: float, oceanic_density: float, jitter_cells: float = 0.5, omega: np.ndarray | None = None, tree: cKDTree | None = None, ext: float | None = None, stretch: float = 0.0, thin_floor: float = 0.0,
                   void: str = "create", taken_out: list | None = None, net_outflow: bool = False, pair_gate: bool = False) -> tuple[Segments, np.ndarray]:
    """Cells farther than ``gap_radius`` from every segment are divergent
    boundaries — provided the nearest segment is moving *away* from the
    cell (``omega`` (P, 3) rad/step given; holes left by subduction at a
    convergent boundary are closed by the incoming plate instead of being
    filled with new crust).  Their (jittered) centres are candidate
    positions, walked in a random order and accepted greedily with minimum
    spacing ``r_min`` (also against the existing segments).  New segments
    are thin (``new_thickness``), have age 0 and the plate of the nearest
    existing segment.

    New crust is oceanic **only where the gap is a real divergent boundary**,
    meaning the segments around it belong to more than one plate. A gap whose
    neighbours are all one plate is not a rift: it is a void that opened in
    the point cloud as the plate rotated and its segments were nudged about,
    and filling it with ocean floor punches a hole through the middle of a
    continent. Measured before this rule, 6.2 % of land area at 20k segments
    and 17.7 % at 60k -- worse with more segments, which is the signature of
    a cloud artifact rather than a resolution limit. Such a gap is filled
    with crust continuing its surroundings instead.

    New crust at a divergent boundary is mid-ocean-ridge basalt: dense,
    compositionally uniform, and the same everywhere.  It used to be given
    ``deposit_density(T)``, which reads the local heat -- so crust born at a
    ridge, where the heat is by construction highest, came out *light* and
    therefore buoyant.  That is backwards, and it meant the model had no way
    to make ocean floor at all.

    ``void`` says what fills a void inside a continent when there is no
    ``stretch`` to pay for it:

    * ``'create'`` -- new continental crust at its neighbours' column, from
      nothing.  The fixed-area model, where nothing else can fill it, and bit
      for bit what this always did.
    * ``'split'`` -- the continental neighbours hand the new segment ground in
      proportion to their extent and the crust standing on it, so their
      columns, the crust and the planet's ground all stay as they were and the
      new column is their extent-weighted mean (the opposite of the
      ``extent_min`` merge).  ``variable_extent``'s choice.
    * ``'stretch'`` -- the neighbours keep their ground, take on the new
      segment's as well and thin to cover it, as extension does.
    * ``'ocean'`` -- sea floor.

    Measured on Earth seeds 0 / 1 at step 600 (everything else as shipped),
    sea enclosed by land: 'create' 9 / 3 pits, 0.70 / 0.14 % of the sphere;
    'split' 5 / 5, 0.52 / 0.20 %; 'stretch' 75 / 82, 4.6 / 5.7 %; 'ocean'
    83 / 90, 4.9 / 5.8 %.  There are 130-150 such voids in a run, almost all
    before step 150, and a column thinned by a seventh (stretch) or replaced
    by sea floor (ocean) drowns in the middle of the supercontinent.
    ``taken_out`` receives the crust taken from existing segments,
    ``(column units, crust units)``, so the caller can book what the new
    segments hold less what they took.

    Returns ``(new_segments, gap_mask)``; the caller appends the segments
    and cools the heat field under ``gap_mask``."""
    if void not in ("create", "split", "stretch", "ocean"):
        raise ValueError(f"spawn_segments: void must be 'create', 'split', 'stretch' or 'ocean' (got {void!r})")
    if taken_out is not None:
        taken_out.append((0.0, 0.0))
    gap = dist > gap_radius
    if omega is not None and gap.any():
        cells = np.nonzero(gap.ravel())[0]
        near = idx.ravel()[cells]
        ps = seg.pos[near]
        v = np.cross(omega[seg.plate_id[near]], ps)
        away = ps - interior_centers_flat(grid)[cells]
        div = np.sum(v * away, axis=1) > 0.0
        if net_outflow and tree is not None and seg.M > 6:
            # A gap between plates is a ridge only if the crust around it is leaving it.  The
            # hole a slab leaves at a trench has the incoming plate's crust closing on it and the
            # overriding plate's next to it, nearly still: the nearest-segment test above read
            # that as divergent whenever the overriding plate drifted the wrong way by a hair,
            # and filled the trench with new floor on the *overriding* plate (32 % of all new
            # sea floor was born within a spacing of a slab that had just gone down), which
            # grew an oceanic apron on the supercontinent and stepped the girdle off its margin.
            # So where the neighbours belong to more than one plate, the net outflow decides
            cen_ = interior_centers_flat(grid)[cells]
            _, nb6 = tree.query(cen_, k=6)
            pnb = seg.plate_id[nb6]
            mixed = (pnb != pnb[:, :1]).any(axis=1)
            if mixed.any():
                pk = seg.pos[nb6[mixed]]                                   # (m, 6, 3)
                vk = np.cross(omega[pnb[mixed]], pk)
                ok = pk - cen_[mixed][:, None, :]
                ok /= np.maximum(np.linalg.norm(ok, axis=2, keepdims=True), 1e-12)
                div[mixed] = np.sum(vk * ok, axis=2).mean(axis=1) > 0.0
        if pair_gate and tree is not None and seg.M > 8:
            # The arcs track's gate (proto/arcs spawn_relative + spawn_normal), the one rule all
            # three dynamics prototypes converged on: a gap with two plates around it is a ridge
            # only if the two plates separate there.  Both plates' velocities are taken at the
            # cell itself and compared across the boundary normal (the line from the nearest
            # plate's neighbours' centroid to the other plate's).  The nearest segment moving
            # away from the cell is not that: behind a slab the overriding plate often moves the
            # same way as the down-going one, only slower, and the hole the slab left was filled
            # with age-0 crust on the overrider -- trench froth, and on a nearly still
            # supercontinent an oceanic apron that stepped the girdle off its margin.  A gap with
            # one plate around it keeps the nearest-segment test
            x_all = interior_centers_flat(grid)[cells]
            kq = min(8, seg.M)
            _, nb = tree.query(x_all, k=kq, workers=kd_workers(cells.size))
            nb = np.atleast_2d(nb).reshape(cells.size, kq)
            pl = seg.plate_id[nb]
            other = pl != pl[:, :1]
            two = other.any(axis=1)
            if two.any():
                r = np.flatnonzero(two)
                a = nb[r, 0]
                b = nb[r, np.argmax(other[r], axis=1)]
                x = x_all[r]
                pa, pb = seg.plate_id[a], seg.plate_id[b]
                inA = (pl[r] == pa[:, None]).astype(np.float64)
                inB = (pl[r] == pb[:, None]).astype(np.float64)
                P3 = seg.pos[nb[r]]
                ca = np.einsum("nk,nkc->nc", inA, P3) / np.maximum(inA.sum(axis=1), 1.0)[:, None]
                cb = np.einsum("nk,nkc->nc", inB, P3) / np.maximum(inB.sum(axis=1), 1.0)[:, None]
                va = np.cross(omega[pa], x)
                vb = np.cross(omega[pb], x)
                div[r] = np.sum((vb - va) * (cb - ca), axis=1) > 0.0
        gap.ravel()[cells[~div]] = False
    n = int(gap.sum())
    if n == 0:
        return Segments(np.zeros((0, 3)), 0.0, 0.0, 0.0, 0, 0.0), gap
    cells = np.nonzero(gap.ravel())[0]
    cands = interior_centers_flat(grid)[cells]
    order = rng.permutation(n)
    cells = cells[order]
    cands = cands[order]
    if jitter_cells > 0:
        cell_rad = math.pi / (2.0 * grid.N)
        cands = cands + rng.normal(size=cands.shape) * (jitter_cells * cell_rad)
        cands /= np.linalg.norm(cands, axis=1, keepdims=True)
    acc = greedy_accept(cands, seg.pos, r_min)
    pos = cands[acc]
    plate = seg.plate_id[idx.ravel()[cells[acc]]]
    mean_area = float(seg.area.mean()) if seg.M else 0.0
    if tree is not None and pos.shape[0] and seg.M > 1:
        kk = min(6, tree.n)
        _, nb = tree.query(pos, k=kk, workers=kd_workers(pos.shape[0]))
        nb = np.atleast_2d(nb).reshape(pos.shape[0], kk)
        pl = seg.plate_id[nb]
        boundary = (pl != pl[:, :1]).any(axis=1)          # >1 plate -> a real rift
        kinds = seg.kind[nb]
        interior_cont = (~boundary) & ((kinds == CONTINENTAL).mean(axis=1) > 0.5)
        kind = np.where(interior_cont, CONTINENTAL, OCEANIC).astype(np.int8)
        # a void inside a continent is filled with crust that continues it,
        # at the local thickness and density rather than as new ocean floor
        th = np.where(interior_cont, seg.thickness[nb].mean(axis=1), new_thickness)
        de = np.where(interior_cont, seg.density[nb].mean(axis=1), oceanic_density)
        cr = np.where(interior_cont, seg.craton[nb][np.arange(pos.shape[0]), 0], 0).astype(np.int8)
        if stretch > 0.0 and interior_cont.any():
            # Extension, the other half of shortening.  With extent as state a void inside a
            # continent is not a hole to fill with new crust -- that manufactured continental
            # area out of nothing every time a shortened margin left room -- it is the crust
            # around it stretching into the gap: the neighbours take the ground and thin by
            # exactly as much, so sum(ext * thickness) does not move.  Rifted margins,
            # back-arc basins and the thinning behind a collapsing orogen are all this
            # ...out of the ground the collisions destroyed this step, and no further: the
            # planet's surface is fixed, so crust can only spread over area that shortening
            # took somewhere else.  Unbudgeted, the stretch grew the continents until they
            # were 0.80 of an Earth-scale planet where they settle at 0.44 on `small`, since
            # the number of interior gaps -- and so the area handed out -- goes with the
            # segment count while the collisions that pay for it do not
            share = min(float(ext), float(stretch) / max(int(interior_cont.sum()), 1))
            add = np.zeros(seg.M, dtype=np.float64)
            np.add.at(add, nb[interior_cont, 0], share)
            if thin_floor > 0.0:
                # ...and no further than the crust can stretch: at the floor it breaks, and
                # the ground it could not take is a rift, filled with sea floor below
                room = seg.ext * np.maximum(seg.thickness / thin_floor - 1.0, 0.0)
                add = np.minimum(add, room)
            grown = add > 0.0
            if grown.any():
                keep = seg.ext[grown] / (seg.ext[grown] + add[grown])       # volume conserved
                seg.thickness[grown] *= keep
                seg.mass[grown] *= keep
                seg.ext[grown] += add[grown]
            # a gap whose crust could not stretch to cover it is a rift after all: it spawns
            taken = add[nb[interior_cont, 0]] >= share * 0.999 if thin_floor > 0.0 else np.ones(int(interior_cont.sum()), bool)
            filled = interior_cont.copy()
            filled[interior_cont] = taken
            kind = np.where(filled, kind, OCEANIC).astype(np.int8)
            th = np.where(filled, th, new_thickness)
            de = np.where(filled, de, oceanic_density)
            cr = np.where(filled, cr, 0).astype(np.int8)
            pos, plate, th, de, cr = (x[~filled] for x in (pos, plate, th, de, cr))
            kind = kind[~filled]
        elif void != "create" and interior_cont.any():
            # No shortening this step to pay for the ground, and a void all the same -- 130-150
            # of them a run at Earth scale, nearly all in the first 150 steps.  Filling one with
            # new continental crust at its neighbours' column made crust from nothing (+0.09
            # units of volume a run: small, but the one continental source with no process
            # behind it)
            e_new = float(ext) if ext is not None else mean_area
            rift = np.zeros(pos.shape[0], dtype=bool)
            took = [0.0, 0.0]
            for v in np.nonzero(interior_cont)[0]:
                if void == "ocean":
                    rift[v] = True
                    continue
                js = nb[v][seg.kind[nb[v]] == CONTINENTAL]
                ej = seg.ext[js]
                e_sum = float(ej.sum())
                vol = float((ej * seg.thickness[js]).sum())
                vm = float((ej * seg.mass[js]).sum())
                if void == "split":
                    # the continental neighbours hand over ground, in proportion to what they
                    # hold, and the crust standing on it: their columns do not change, the new
                    # segment's is their extent-weighted mean, and the planet's ground total
                    # does not move either
                    if e_sum <= 0.0 or vol <= 0.0:
                        rift[v] = True
                        continue
                    d = e_new * ej / e_sum
                    if (ej - d).min() <= 0.0:
                        rift[v] = True
                        continue
                    v_new = float((d * seg.thickness[js]).sum())
                    m_new = float((d * seg.mass[js]).sum())
                    seg.ext[js] -= d
                    th[v] = v_new / e_new
                    de[v] = m_new / v_new
                    took[0] += m_new / e_new                                    # its column
                    took[1] += m_new                                            # crust units
                    continue
                # the continental neighbours stretch over the new ground: each thins by the
                # same factor, and what they lose is the new segment's column
                keep = e_sum / max(e_sum + e_new, 1e-12)
                if e_sum <= 0.0 or vol <= 0.0 or (thin_floor > 0.0 and keep * vol / e_sum < thin_floor):
                    rift[v] = True        # crust at its floor breaks: that is a rift
                    continue
                took[0] += (1.0 - keep) * float(seg.mass[js].sum())          # column units
                took[1] += (1.0 - keep) * vm                                 # crust units
                seg.thickness[js] *= keep
                seg.mass[js] *= keep
                th[v] = (1.0 - keep) * vol / e_new
                de[v] = vm / vol
            kind = np.where(rift, OCEANIC, kind).astype(np.int8)
            th = np.where(rift, new_thickness, th)
            de = np.where(rift, oceanic_density, de)
            cr = np.where(rift, 0, cr).astype(np.int8)
            if taken_out is not None:
                taken_out[-1] = (took[0], took[1])
        new = Segments(pos, th, de, 0.0, plate, mean_area, kind=kind, craton=cr, ext=ext if ext is not None else mean_area)
    else:
        new = Segments(pos, new_thickness, oceanic_density, 0.0, plate, mean_area, kind=OCEANIC, ext=ext if ext is not None else mean_area)
    return new, gap


# --------------------------------------------------------------------------
# collisions (PLAN 6.2.5)
# --------------------------------------------------------------------------
def differentiate(seg: Segments, survivors: np.ndarray, rate: float, floor: float) -> float:
    """Make collided crust lighter -- the process that builds continents.

    Earth's surface is famously bimodal: a peak at the continental shelf and
    another on the abyssal plain, with little between, because it carries two
    kinds of crust. Oceanic crust is basaltic, dense and thin; continental
    crust is granitic, light and thick, and floats about 4 km higher.

    Collision alone cannot produce that split. Merging two segments averages
    their mass and thickness, so density only ever moves toward the mean and
    the elevation histogram stays a single narrow spike (measured: land mean
    324 m and ocean mean -189 m, against Earth's 840 m and -3700 m).

    What separates the two populations on Earth is *differentiation*. Crust
    thickened at an arc partially melts; the light granitic fraction rises
    and stays, while the dense residue delaminates and is lost to the mantle.
    Crust that has been through a collision therefore comes out lighter than
    it went in, and repeated orogeny ratchets it toward continental.

    So this pulls each survivor's density a fraction `rate` toward `floor`,
    keeping thickness and dropping mass. The lost mass is returned so the
    caller can book it: it has left the crust for the mantle, which is
    physical rather than a leak.

    **Measured: this alone does not produce the bimodality.** Sweeping
    `differentiation` 0 -> 0.04 -> 0.10 moved the land/ocean mean gap
    1723 -> 1171 -> 1472 m, against Earth's 4540 m, with ocean mean depth
    stuck near -300 m rather than -3700 m. Making continents lighter also
    lifts the sea-level quantile they are measured against, and
    `relief_m` then rescales the whole field, so the ratio barely moves.
    Raising `deposit_density` (denser new oceanic crust) is likewise only
    marginal: 0.5 -> 0.9 took ocean-depth-over-land-relief from 0.03 to
    0.06, where Earth is 0.42.

    The deeper obstacle is that our ocean floor is not a basin. It spans
    about 400 m (-600 to -190) hugging sea level, where Earth's spans 3000
    (-5500 to -2500) and is *separated* from the shelf by a steep, narrow
    slope. Earth is bimodal because it carries two discrete crust
    populations with a sharp margin between them; a smooth splat of a
    continuously-varying thickness gives one continuum, and no amount of
    density tuning turns a continuum into two peaks. Kept because the
    mechanism is real and may matter alongside a genuine crust-type split,
    but it is not the lever on its own.
    """
    if rate <= 0.0 or survivors.size == 0:
        return 0.0
    su = np.unique(survivors)
    su = su[(su >= 0) & (su < seg.M)]
    if su.size == 0:
        return 0.0
    d0 = seg.density[su]
    d1 = np.maximum(d0 - rate * (d0 - float(floor)), float(floor))
    lost = float(((d0 - d1) * seg.thickness[su]).sum())
    seg.density[su] = d1
    seg.mass[su] = seg.thickness[su] * d1
    return lost


#: Density difference below which two segments count as the same rock.  Crust
#: of one kind *is* one density here (oceanic 0.88 everywhere), and a collision
#: recomputes it as ``mass / thickness``, so the two sides of an ocean-ocean
#: pair differ only in the last bits of the division.  Anything above this is a
#: real compositional difference (belt 0.804 against craton 0.856).
DENSITY_EPS = 1e-9


def plate_pair_polarity(plate_id: np.ndarray, age: np.ndarray, kind: np.ndarray, pairs: np.ndarray, P: int) -> np.ndarray:
    """Which of two plates goes down where they meet: ``pol[p, q] == 1``
    means a segment of plate ``p`` subducts under one of plate ``q``.

    A subduction zone has a *polarity*: one plate dives and the other
    overrides, and it is the same one along the whole trench, because what
    decides it is a property of the plates -- which carries the older,
    colder, denser lithosphere -- and not of the particular pair of rocks
    in contact.

    Deciding it per pair instead is what braided the ocean plates into
    interleaved strands.  Within one crust type every segment has the same
    density, so ``density[i] > density[j]`` was comparing the rounding error
    of ``mass / thickness``: measured over 300 steps of the Earth preset,
    28.7 % of ocean-on-ocean collisions were exact ties (broken by segment
    index) and the rest were decided by the last bits of a division.  Either
    way the choice is uncorrelated with which plate the segment belongs to,
    so along a trench each plate wins about half the contacts and drives a
    finger into the other.  By step 1500 the worst plate's largest connected
    piece held 25 % of its area and the total boundary length had grown 3.7x.

    The mean age of each side's crust *along that boundary* is the
    discriminator, so a plate that is old where it meets one neighbour and
    young where it meets another gets a different polarity at each -- which
    is the real behaviour.  Ties go to the higher-numbered plate, so the
    result is deterministic.
    """
    pi = plate_id[pairs[:, 0]].astype(np.int64)
    pj = plate_id[pairs[:, 1]].astype(np.int64)
    pol = np.zeros((P, P), dtype=np.int8)
    sel = (pi != pj) & (kind[pairs[:, 0]] == kind[pairs[:, 1]])
    if not sel.any():
        return pol
    a, b = pi[sel], pj[sel]
    swap = a > b
    lo, hi = np.where(swap, b, a), np.where(swap, a, b)
    age_i, age_j = age[pairs[sel, 0]], age[pairs[sel, 1]]
    key = lo * P + hi
    cnt = np.bincount(key, minlength=P * P)
    s_lo = np.bincount(key, weights=np.where(swap, age_j, age_i), minlength=P * P)
    s_hi = np.bincount(key, weights=np.where(swap, age_i, age_j), minlength=P * P)
    k = np.nonzero(cnt)[0]
    older_lo = s_lo[k] > s_hi[k]
    k_lo, k_hi = k // P, k % P
    pol[k_lo[older_lo], k_hi[older_lo]] = 1
    pol[k_hi[~older_lo], k_lo[~older_lo]] = 1
    return pol


# numba closes over module globals as compile-time constants; the int8 typing
# has to match ``Segments.kind`` for the comparisons inside the kernel
OCEANIC_K = np.int8(OCEANIC)
CONTINENTAL_K = np.int8(CONTINENTAL)


@njit(cache=True)
def _is_rift_pair(a, b, rift_a, rift_b):
    for r in range(rift_a.shape[0]):
        if (rift_a[r] == a and rift_b[r] == b) or (rift_a[r] == b and rift_b[r] == a):
            return True
    return False


@njit(cache=True)
def _apply_collisions(pairs, plate_id, omega_dt, pos, mass, thickness, density, age, rework, kind, craton, weld, ext, spent, polarity, alive, overlap2, accretion, arc_birth, birth_draw, shortening, radius, weld_steps, extent_min, arc_thickness, arc_density,
                      arc_dock, arc_keep, ocean_base, rift_a, rift_b, dock_keep, terrane):
    n = pairs.shape[0]
    losers = np.empty(n, dtype=np.int64)
    survivors = np.empty(n, dtype=np.int64)
    # what the survivor was actually handed, thickness and mass -- the change in its column,
    # new minus old, which a merge that thins the survivor makes negative -- where the extent
    # model decided it (NaN: the fixed-area rule, `_received_fraction` of the loser's column)
    recv_th = np.full(n, np.nan)
    recv_m = np.full(n, np.nan)
    # every event is a fixed-area one or an extent one for the whole run
    var_ext = extent_min > 0.0
    k = 0
    for e in range(n):
        i = pairs[e, 0]
        j = pairs[e, 1]
        if not alive[i] or not alive[j]:
            continue
        pi = plate_id[i]
        pj = plate_id[j]
        if pi == pj:
            continue
        # approaching?  (v_i - v_j) . (p_j - p_i) > 0 with v = (omega dt) x p
        vix = omega_dt[pi, 1] * pos[i, 2] - omega_dt[pi, 2] * pos[i, 1]
        viy = omega_dt[pi, 2] * pos[i, 0] - omega_dt[pi, 0] * pos[i, 2]
        viz = omega_dt[pi, 0] * pos[i, 1] - omega_dt[pi, 1] * pos[i, 0]
        vjx = omega_dt[pj, 1] * pos[j, 2] - omega_dt[pj, 2] * pos[j, 1]
        vjy = omega_dt[pj, 2] * pos[j, 0] - omega_dt[pj, 0] * pos[j, 2]
        vjz = omega_dt[pj, 0] * pos[j, 1] - omega_dt[pj, 1] * pos[j, 0]
        dx = pos[j, 0] - pos[i, 0]
        dy = pos[j, 1] - pos[i, 1]
        dz = pos[j, 2] - pos[i, 2]
        if arc_dock > 0.0:
            # A docked terrane stays docked while it touches anything: its weld is renewed for
            # as long as the contact lasts and wears off weld_steps after it ends.  A fixed
            # weld_steps (9 My) let the same terrane be handed back and forth once the weld wore
            # off under a boundary still in contact (proto/arcs: 922 segments docked 2,519
            # times on one seed, the worst 26 times)
            if kind[i] == OCEANIC_K and weld[i] > 0 and thickness[i] >= arc_dock:
                weld[i] = weld_steps
            if kind[j] == OCEANIC_K and weld[j] > 0 and thickness[j] >= arc_dock:
                weld[j] = weld_steps
        approaching = (vix - vjx) * dx + (viy - vjy) * dy + (viz - vjz) * dz > 0.0
        if not approaching:
            # receding / sliding past: only collide once they overlap deeply
            if dx * dx + dy * dy + dz * dz > overlap2:
                continue
        # Who goes down.  Crust type first: a continent cannot be subducted
        # under ocean floor at any density, which is the irreversibility that
        # separates the two populations.  Within a type, the denser (older,
        # colder) slab sinks, as before.
        if kind[i] != kind[j]:
            if kind[i] == OCEANIC_K:
                lo, su = i, j
            else:
                lo, su = j, i
        elif kind[i] == CONTINENTAL_K and craton[i] != craton[j]:
            # A craton against a mobile belt: the belt deforms. Cratons are
            # cold, thick and depleted, so they behave as rigid indenters --
            # which is why the same Archean nuclei have survived every cycle
            # while the crust welded between them has been reworked
            # repeatedly. Without this a craton is just another continental
            # segment and gets consumed at the same rate as its surroundings.
            if craton[i] == 0:
                lo, su = i, j
            elif craton[j] == 0 or extent_min <= 0.0:
                lo, su = j, i
            # Craton against craton: the weaker lithosphere yields -- the thinner column,
            # then the one with less ground.  Which went under was the pair's array order
            # (`j`, the higher index, always lost), and array order says nothing about the
            # two cratons, so along one suture they took turns to lose, contact by contact,
            # instead of the stronger indenting the weaker
            # (variable extent only: the fixed-area model keeps the old order, bit for bit)
            elif thickness[i] < thickness[j] or (thickness[i] == thickness[j] and ext[i] < ext[j]):
                lo, su = i, j
            else:
                lo, su = j, i
        elif density[i] > density[j] + DENSITY_EPS:
            lo, su = i, j
        elif density[j] > density[i] + DENSITY_EPS:
            lo, su = j, i
        elif polarity[pi, pj] == 1:
            # Same rock on both sides: the *boundary* decides, not the pair.
            # Comparing the two densities here compares rounding error, and
            # breaking the tie by segment index gives each plate half the
            # contacts along a trench, which interleaves them (see
            # `plate_pair_polarity`).
            lo, su = i, j
        else:
            lo, su = j, i

        if arc_dock > 0.0 and kind[lo] == OCEANIC_K and kind[su] == OCEANIC_K and weld[lo] > 0 \
                and thickness[lo] >= arc_dock:
            # A terrane that docked is part of its new plate's margin: it is never handed back
            # because the boundary's polarity flickered: against anything but another terrane
            # the roles swap -- thinner floor goes down under the terrane, and a thick arc that
            # has not docked anywhere (the case that fell through to the dock below and handed
            # the terrane back) docks onto it instead.
            if weld[su] > 0 and thickness[su] >= arc_dock:
                # Two terranes of two plates: an arc-arc collision (the Molucca Sea).  Neither
                # can go down, so the crust stacks: the thinner is shortened whole into the
                # thicker, whose column thickens by its volume (arc root foundering, arc_max_km,
                # takes back what passes 35 km).  Left alone they passed through each other, each
                # renewing the other's weld, and clusters of them packed 2.7x the design density
                # with the columns of three plates overlapping: arc "islands" of 0.2-0.5 Mkm2 (a
                # 30 km thick plateau, its crest p90 to +7.6 km) on four of six Earth seeds at
                # 300-600 My.  Docking the thinner onto the thicker's plate instead handed
                # terranes in a cluster back and forth, often back within the step: 88-96 % of
                # the ocean docks on the same six seeds were redocks.  A rift's halves keep apart
                if _is_rift_pair(plate_id[lo], plate_id[su], rift_a, rift_b):
                    continue
                if thickness[lo] > thickness[su]:
                    t_ = lo
                    lo = su
                    su = t_
                if var_ext:
                    g_ = ext[lo] / max(ext[su], 1e-12)
                else:
                    g_ = 1.0
                m_lo = g_ * mass[lo]
                thickness[su] += g_ * thickness[lo]
                mass[su] += m_lo
                density[su] = mass[su] / max(thickness[su], 1e-12)
                rework[su] *= 1.0 - min(m_lo / max(mass[su], 1e-12), 1.0)
                if age[lo] > age[su]:
                    age[su] = age[lo]
                alive[lo] = False
                spent[15] += 1.0
                continue
            t_ = lo
            lo = su
            su = t_
        if arc_dock > 0.0 and kind[lo] == OCEANIC_K and thickness[lo] >= arc_dock and approaching \
                and not _is_rift_pair(plate_id[lo], plate_id[su], rift_a, rift_b):
            # Crust this thick on the down-going side -- an island arc, mostly -- is too buoyant
            # to follow its slab (Earth: > ~17 km jams a trench).  It docks: it becomes part of
            # the overriding plate where it touches it, and the trench steps out behind it.
            # Onto ocean floor it stays an arc, a terrane; onto a continent it is continental
            # crust from here on (the books' `docked`), which is how continents grow, and the
            # same contact is a continental collision that shortens it into the margin
            # (arc-continent collision thickens the margin, Taiwan, rather than leaving a low
            # terrane).  A rift's two halves never dock onto each other
            if dock_keep < 1.0 and (terrane[lo] == 0 or kind[su] == CONTINENTAL_K):
                # A colliding arc does not accrete whole: its forearc, its dense lower crust and
                # the slab under it go down the trench (the Luzon arc under Taiwan, the Halmahera
                # arc under the Sangihe in the Molucca Sea), and only the rest is scraped onto
                # the overriding plate.  Whole, the docked arcs were 12 % of the planet's ground
                # and +27 % of the continental crust over 600 My on the first Earth seed
                # (continents 0.40 -> 0.48), and arcs docked onto ocean plates piled up into
                # plateaus no trench could take.
                # Once per terrane onto ocean floor: a terrane that docks onto an ocean plate
                # again (its weld wore off, or it was welded onto the plate around it when its own
                # was subducted from under it, intraplate.split_disconnected) is the same crust at
                # the same trench, not another collision.  Halved at every dock, terranes that
                # docked 4-5 times were 0.05-0.22 of a design segment's ground, below extent_min,
                # and 878 of 2462 ocean docks on one seed were repeats.  Onto a continent it is a
                # collision every time (and the last: it is continental crust after it); exempt
                # there too, the docked terranes were 3.4-6.3 % of the planet at 600 My on six
                # Earth seeds and the continental crust grew 1.03-1.29x.  Nor is a docking arc left
                # with less ground than extent_min
                if extent_min > 0.0:
                    kept = max(dock_keep * ext[lo], min(ext[lo], extent_min))
                    lost = 1.0 - kept / max(ext[lo], 1e-300)
                else:
                    lost = 1.0 - dock_keep
                spent[3] += lost * mass[lo]
                spent[4] += lost * ext[lo] * mass[lo]
                if extent_min > 0.0:
                    ext[lo] *= 1.0 - lost
                else:
                    thickness[lo] *= 1.0 - lost
                    mass[lo] *= 1.0 - lost
            elif terrane[lo] != 0:
                spent[14] += 1.0
            terrane[lo] = 1
            # it moves with the overriding plate from here on, welded to it
            plate_id[lo] = plate_id[su]
            weld[lo] = weld_steps
            if kind[su] == CONTINENTAL_K:
                spent[9] += mass[lo]
                spent[10] += ext[lo] * mass[lo]
                spent[11] += 1.0
                spent[12] += ext[lo]
                kind[lo] = CONTINENTAL_K
                rework[lo] = 0.0                     # juvenile crust, assembled now
                # ...and falls through to the continental shortening below.  It is the
                # continent's from here whether or not this contact spends any ground on it:
                # relabelled only when the shortening took some, a disc that no longer reached
                # the margin turned continental but stayed on its ocean plate, unwelded
                # (a continental speck riding away on the plate it docked from)
            else:
                spent[13] += 1.0
                continue

        # How much of the slab stays at the surface.  Ocean floor going down a
        # trench mostly leaves the system: its sediment and a melt fraction are
        # welded onto the overriding plate as an arc, the rest returns to the
        # mantle.  Handing over 100 %, as this did, made the crust a monotone
        # accumulator -- 1500 steps of it drove thickness to 8.3 against an
        # initial 0.4, and every collision zone toward the same saturated
        # height.  Continent-on-continent keeps everything: nothing subducts,
        # the crust doubles, and that is what a Tibet is.
        f = 1.0 if kind[lo] == CONTINENTAL_K else accretion
        if kind[lo] == OCEANIC_K and arc_keep >= 0.0:
            # A thinner arc does go down, and the overriding plate scrapes part of it off on the
            # way: `arc_keep` of the column above the oceanic birth thickness and `accretion` of
            # the sea floor under it; the rest goes to the mantle.  Plain sea floor has no arc
            # and gets exactly `accretion`, as before
            th_lo = thickness[lo]
            arc_c = max(th_lo - ocean_base, 0.0)
            f = (accretion * min(th_lo, ocean_base) + arc_keep * arc_c) / max(th_lo, 1e-12)
        if kind[lo] == CONTINENTAL_K and extent_min > 0.0:
            # Crustal shortening, area spent rather than a segment deleted.  The two discs of
            # equal area overlap by a geometric amount that goes to zero as they separate, so
            # the ground consumed at a convergent boundary is the boundary's length times how
            # far the plates actually converged -- the physical rate -- instead of one whole
            # segment per contact whatever the convergence (which is what halved the
            # continental crust every 1500 steps).  The crust that was standing on the
            # consumed ground goes into the survivor as volume, so sum(ext * thickness) does
            # not move: the belt thickens by exactly what the margin lost
            th_was = thickness[su]
            m_was = mass[su]
            dn2 = dx * dx + dy * dy + dz * dz
            dn = np.sqrt(dn2)
            r_lo = np.sqrt(ext[lo] / np.pi)
            r_su = np.sqrt(ext[su] / np.pi)
            # how fast the two are closing, radians a step, along the line of centres
            approach = ((vix - vjx) * dx + (viy - vjy) * dy + (viz - vjz) * dz) / max(dn, 1e-12)
            took = 0.0
            if dn < r_lo + r_su and dn > 1e-12:
                # circle-circle lens area, planar at these radii
                c1 = (dn2 + r_lo * r_lo - r_su * r_su) / (2.0 * dn * r_lo)
                c2 = (dn2 + r_su * r_su - r_lo * r_lo) / (2.0 * dn * r_su)
                c1 = min(1.0, max(-1.0, c1))
                c2 = min(1.0, max(-1.0, c2))
                lens = (r_lo * r_lo * (np.arccos(c1) - c1 * np.sqrt(max(1.0 - c1 * c1, 0.0)))
                        + r_su * r_su * (np.arccos(c2) - c2 * np.sqrt(max(1.0 - c2 * c2, 0.0))))
                # The ground a contact consumes is the margin's width times how far the
                # plates actually converged this step -- not the overlap of the two discs,
                # which is a penetration depth set by the collision radius and so by the
                # segment size: at 20000 segments that consumed a quarter of the area per
                # unit of boundary that it did at 1500, and the continents ran to 0.80 of
                # the planet at Earth scale where they settle at 0.44 on `small`
                take = min(lens, 2.0 * r_lo * max(approach, 0.0))
                take = min(take, ext[lo])
                if take > 0.0:
                    vol = take * thickness[lo]                       # crust standing on it
                    mvol = take * mass[lo]
                    ext[lo] -= take
                    spent[0] += take
                    thickness[su] += vol / max(ext[su], 1e-12)
                    mass[su] += mvol / max(ext[su], 1e-12)
                    density[su] = mass[su] / max(thickness[su], 1e-12)
                    # The belt is new crust.  The column now holds its own protolith and
                    # what has just been stacked into it, so the age it would be *mapped*
                    # at -- when this crust was last assembled -- is the two mixed by mass.
                    # `age` does not move: the rock is as old as it ever was
                    rework[su] *= 1.0 - min((mvol / max(ext[su], 1e-12)) / max(mass[su], 1e-12), 1.0)
                    took = take
            if took > 0.0:
                # The margin that shortened joins the plate it shortened onto.  Only then:
                # relabelling every contact -- the receding and sliding ones inside
                # `overlap_fraction` too, and approaching pairs whose discs do not yet meet --
                # moved crust between plates that no convergence had spent any ground on
                plate_id[lo] = plate_id[su]
                weld[lo] = weld_steps
            if ext[lo] <= extent_min:
                # Nothing left to shorten: what remains joins the belt and the point goes.  The
                # survivor's column becomes the extent-weighted mean of the two, and only then
                # does its extent grow by the loser's.  Raising the survivor's column by the
                # remnant's volume over its *old* extent and then adding the remnant's extent as
                # well -- which is what this did -- counted the remnant's ground twice and made
                # ext[lo] * th[su] of crust per merge: +19.4 % of the pair's crust in one
                # merge, +3.6 to +4.0 units of continental volume over 8000 Earth steps
                # (1 unit = 1.42e9 km3), about a fifth of what orogen decay takes
                e_su = ext[su]
                e_lo = ext[lo]
                e_new = max(e_su + e_lo, 1e-12)
                m_lo = e_lo * mass[lo]
                m_all = e_su * mass[su] + m_lo
                thickness[su] = (e_su * thickness[su] + e_lo * thickness[lo]) / e_new
                mass[su] = m_all / e_new
                density[su] = mass[su] / max(thickness[su], 1e-12)
                rework[su] *= 1.0 - min(m_lo / max(m_all, 1e-12), 1.0)
                ext[su] = e_su + e_lo
                ext[lo] = 0.0
                alive[lo] = False
            elif took <= 0.0:
                # No ground spent and nothing merged: the pair touched and nothing happened,
                # so it is not a collision either.  Reported as one, it went on to the belt
                # builder with nothing received, and the contact was relabelled every step the
                # pair stayed in range.  The slot is left as it was found (NaN) for the next one
                continue
            # what the survivor's column holds now against what it held: the belt lays out a
            # positive change and nothing else (a merge with a thinner remnant thins it, and a
            # pair too far apart to overlap changes nothing -- where the -1 this used to leave
            # sent the whole of the loser's column to the belt, out of the survivor)
            recv_th[k] = thickness[su] - th_was
            recv_m[k] = mass[su] - m_was
            losers[k] = lo
            survivors[k] = su
            k += 1
            continue
        if kind[lo] == CONTINENTAL_K and shortening > 0.0:
            # crustal shortening, area conserved: part of the segment is
            # stacked into the belt, the rest stays where it is and moves with
            # the plate it has just been welded to (see continental_shortening)
            f = shortening
            mass[su] += f * mass[lo]
            thickness[su] += f * thickness[lo]
            density[su] = mass[su] / thickness[su]
            rework[su] *= 1.0 - min(f * mass[lo] / max(mass[su], 1e-12), 1.0)
            mass[lo] *= 1.0 - f
            thickness[lo] *= 1.0 - f
            plate_id[lo] = plate_id[su]
            # and it stays that plate's for a while.  It sits inside the plate it came from,
            # so split_disconnected saw a sliver embedded in a foreign plate and welded it
            # straight back -- which made the same pair a collision again the next step, and
            # the one after (9 collisions a step became 100, and the arc births that ride on
            # them turned the planet continental).  The weld is what says otherwise
            weld[lo] = weld_steps
            # the crust between the two has shortened: the loser's centre
            # retreats to the edge of the collision radius, so the same pair
            # is not a collision again next step (it would be, every step,
            # until nothing was left of it -- measured at 14 million
            # continent-on-continent events in a run against 9 thousand)
            dn = np.sqrt(dx * dx + dy * dy + dz * dz)
            if dn > 1e-12:
                s = 1.05 * radius / dn
                px = pos[su, 0] - dx * s
                py = pos[su, 1] - dy * s
                pz = pos[su, 2] - dz * s
                pn = np.sqrt(px * px + py * py + pz * pz)
                pos[lo, 0] = px / pn
                pos[lo, 1] = py / pn
                pos[lo, 2] = pz / pn
            losers[k] = lo
            survivors[k] = su
            k += 1
            continue
        # The books (column units, then crust units -- column x extent): what the slab sends
        # to the mantle, and what of it an ocean-going-under-continent contact turns into
        # continental crust.  Exact amounts, so the per-kind ledger can be checked against the
        # state rather than read off it (run.py, KIND_KEYS)
        if kind[lo] == OCEANIC_K:
            spent[3] += (1.0 - f) * mass[lo]
            spent[4] += (1.0 - f) * ext[lo] * mass[lo]
            if kind[su] == CONTINENTAL_K:
                spent[5] += f * mass[lo]
                spent[6] += f * ext[lo] * mass[lo]
        if var_ext:
            # The accreted crust stood on the slab's ground and is spread over the survivor's,
            # so the survivor's column rises by the ratio of the two.  Column for column, as
            # the fixed-area rule does, overcounted it by ext[su] / ext[lo] -- 1.15 on average
            # at Earth scale and 1.22 after step 4000, ~0.55 of the +4.3 units of continental
            # volume accretion books over 8000 steps
            g = f * ext[lo] / max(ext[su], 1e-12)
            dm = g * mass[lo]
            dth = g * thickness[lo]
            mass[su] += dm
            thickness[su] += dth
            recv_th[k] = dth
            recv_m[k] = dm
        else:
            dm = f * mass[lo]
            mass[su] += f * mass[lo]
            thickness[su] += f * thickness[lo]
        density[su] = mass[su] / thickness[su]
        # the survivor's age is the older of the two only when it keeps the
        # whole slab; an arc is new crust welded to old, not old crust
        if f >= 1.0 and age[lo] > age[su]:
            age[su] = age[lo]
        # ...and whatever was stacked into it, of either kind, is new crust in the column:
        # the accreted fraction of a slab is what builds an accretionary margin
        rework[su] *= 1.0 - min(dm / max(mass[su], 1e-12), 1.0)
        # Island arcs: repeated ocean-on-ocean subduction is how continental
        # crust is *born* (the Japans, the Aleutians, the Andean margin before
        # it was a margin).  Without a birth channel the continental area can
        # only shrink from whatever the initial condition seeded, and the
        # supercontinent cycle runs down.
        if kind[su] == OCEANIC_K and kind[lo] == OCEANIC_K and birth_draw[e] < arc_birth:
            kind[su] = CONTINENTAL_K            # island arc -> new continental crust, not craton
            rework[su] = 0.0                    # juvenile: the crust is being made right now
            # the column it had, slab accretion and all, changes kind with it
            spent[7] += mass[su]
            spent[8] += ext[su] * mass[su]
            if arc_thickness > 0.0:
                # born as arc crust, not as a relabelled slab: the column an arc has and the
                # belt's composition, the extra drawn from the mantle (spent[1] keeps the sum)
                m0 = mass[su]
                thickness[su] = max(thickness[su], arc_thickness)
                density[su] = arc_density
                mass[su] = thickness[su] * density[su]
                spent[1] += mass[su] - m0
                spent[2] += ext[su] * (mass[su] - m0)
        alive[lo] = False
        losers[k] = lo
        survivors[k] = su
        k += 1
    return losers[:k], survivors[:k], recv_th[:k], recv_m[:k]


def collide(seg: Segments, tree: cKDTree, radius: float, omega_dt: np.ndarray, alive: np.ndarray, overlap_fraction: float = 0.5, accretion: float = 1.0, arc_birth: float = 0.0, rng: np.random.Generator | None = None, shortening: float = 0.0, weld_steps: int = 0, extent_min: float = 0.0, spent_out: list | None = None,
            arc_thickness: float = 0.0, arc_density: float = 0.804, arc_out: list | None = None, recv_out: list | None = None,
            books_out: list | None = None, arc_dock: float = 0.0, arc_keep: float = -1.0, ocean_base: float = 0.2,
            rift_pairs=None, dock_out: list | None = None, dock_keep: float = 1.0):
    """Subduction: for every pair of segments of different plates within
    chord ``radius`` (KD-tree pair query, applied in sorted order) that are
    *approaching* — or closer than ``overlap_fraction * radius`` whatever
    their relative motion, so plates sliding past each other cannot
    interleave — one subducts: the oceanic member of a mixed pair whatever
    its density, else the denser one, and where both sides are the same rock
    (which is every ocean-ocean pair) the plate whose crust is older along
    that boundary (:func:`plate_pair_polarity`).  A fraction ``accretion`` of its mass
    and thickness goes to the survivor and the rest is lost to the mantle
    (all of it, and the older age, when the loser is continental — that
    crust cannot sink, so a continent-continent collision doubles the crust
    instead of destroying half of it).  With probability ``arc_birth`` an
    ocean-on-ocean survivor becomes continental.  The loser is flagged dead
    in ``alive`` (in place).  The loser's own arrays keep the
    transferred amounts until ``seg.compress(alive)``; the total mass of
    live segments is conserved.  Returns ``(losers, survivors)`` index
    arrays (into the current arrays; a survivor may appear several times).
    With ``shortening > 0`` a continental loser is not killed: that fraction
    of it goes to the survivor, it keeps the rest and joins the survivor's
    plate (``plate_id`` is updated in place).

    ``books_out`` receives the crust the step moved between the books, each as
    ``(column units, crust units)`` -- crust being column x extent, the
    quantity ``variable_extent`` conserves: ``subducted`` (oceanic slab sent
    to the mantle, positive), ``accreted`` (oceanic slab welded onto a
    continent, which becomes continental crust) and ``arc_born`` (an oceanic
    column relabelled continental by ``arc_birth``, before its extra arc crust
    is drawn from the mantle; that draw is ``arc_out``) and ``docked`` (an
    oceanic column at least ``arc_dock`` thick that docked onto a continent and
    turned continental).

    Island arcs (te/arcs): with ``arc_dock > 0`` an oceanic column that thick
    on the down-going side of an approaching pair docks instead of subducting
    (onto ocean floor it joins the overriding plate, welded; onto a continent it
    turns continental and shortens into the margin), a welded terrane is never
    the loser of an ocean-ocean pair and stays welded while it is in contact,
    and the two halves of a rift (``rift_pairs``, plate id pairs) never dock
    onto each other.  Of a docking arc only ``dock_keep`` accretes -- of its
    ground with variable extent (never less than ``extent_min``), of its column
    without -- and the rest goes down with its slab (booked ``subducted``): at its
    first dock onto ocean floor (``seg.terrane`` is set there), not when the same
    terrane docks onto an ocean plate again, and at its dock onto a continent.
    ``dock_out`` receives ``(docked onto continents, the ground they
    kept, docked onto ocean plates, docks of terranes that had docked before,
    terranes stacked into another plate's terrane)``: two terranes of two plates in
    contact are an arc-arc collision, the thinner shortened whole into the thicker.
    With ``arc_keep >= 0`` a
    thinner arc going down hands the overriding plate ``arc_keep`` of its crust
    above ``ocean_base`` and ``accretion`` of the sea floor under it."""
    pairs = tree.query_pairs(radius, output_type="ndarray")
    if pairs.shape[0] == 0:
        if spent_out is not None:
            spent_out.append(0.0)
        if recv_out is not None:
            recv_out.append((np.zeros(0), np.zeros(0)))
        if books_out is not None:
            books_out.append({"subducted": (0.0, 0.0), "accreted": (0.0, 0.0), "arc_born": (0.0, 0.0), "docked": (0.0, 0.0)})
        if dock_out is not None:
            dock_out.append((0.0, 0.0, 0.0, 0.0, 0.0))
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    pairs = np.sort(pairs, axis=1)
    pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    draw = rng.random(pairs.shape[0]) if (rng is not None and arc_birth > 0.0) else np.zeros(pairs.shape[0])
    P = int(seg.plate_id.max()) + 1 if seg.M else 1
    pol = plate_pair_polarity(seg.plate_id, seg.age, seg.kind, pairs, P)
    # [ground crustal shortening consumed, mass arcs drew from the mantle by column and
    #  by crust (column x extent) -- the ledger is kept in whichever its mode conserves --
    #  then the same two for slab lost to the mantle, slab accreted to a continent,
    #  oceanic columns an arc birth relabelled, and oceanic columns that docked onto a
    #  continent; then the docks as counts and ground: onto continents (count, ground),
    #  onto ocean plates (count), docks of a terrane that had docked before (count), and
    #  terranes stacked into terranes of another plate (count)]
    spent = np.zeros(16, dtype=np.float64)
    ra = np.zeros(0, np.int64)
    rb = np.zeros(0, np.int64)
    if rift_pairs:
        ra = np.asarray([int(a) for a, _ in rift_pairs], np.int64)
        rb = np.asarray([int(b) for _, b in rift_pairs], np.int64)
    out = _apply_collisions(np.ascontiguousarray(pairs), seg.plate_id, np.ascontiguousarray(omega_dt), seg.pos, seg.mass, seg.thickness, seg.density, seg.age, seg.rework, seg.kind, seg.craton, seg.weld, seg.ext, spent, pol, alive, (float(overlap_fraction) * float(radius)) ** 2, float(accretion), float(arc_birth), np.ascontiguousarray(draw), float(shortening), float(radius), int(weld_steps), float(extent_min), float(arc_thickness), float(arc_density),
                            float(arc_dock), float(arc_keep), float(ocean_base), ra, rb, float(dock_keep), seg.terrane)
    if spent_out is not None:
        spent_out.append(float(spent[0]))
    if arc_out is not None:
        arc_out.append((float(spent[1]), float(spent[2])))
    if books_out is not None:
        books_out.append({"subducted": (float(spent[3]), float(spent[4])), "accreted": (float(spent[5]), float(spent[6])),
                          "arc_born": (float(spent[7]), float(spent[8])), "docked": (float(spent[9]), float(spent[10]))})
    if dock_out is not None:
        dock_out.append((float(spent[11]), float(spent[12]), float(spent[13]), float(spent[14]), float(spent[15])))
    losers, survivors, recv_th, recv_m = out
    if recv_out is not None:
        recv_out.append((recv_th, recv_m))     # what each survivor was handed (NaN: fixed-area rule)
    return losers, survivors


@njit(cache=True)
def _received_fraction(kind_lo, alive_lo, accretion, shortening):
    """What the survivor received, as a multiple of what the loser's arrays
    hold *now*: all of a dead continental partner, ``accretion`` of a slab,
    and for a continental loser that is still alive (shortening) the part
    that moved over the part that stayed."""
    if kind_lo == CONTINENTAL_K:
        if alive_lo and shortening > 0.0:
            return shortening / max(1.0 - shortening, 1e-9)
        return 1.0
    return accretion


@njit(cache=True)
def _spread_kernel(losers, survivors, nbrs, pos, plate_id, kind, mass, thickness, density, alive, inv2s2, accretion, shortening, recv_th, recv_m, ext, use_ext):
    K = nbrs.shape[1]
    w = np.empty(K, dtype=np.float64)
    for e in range(losers.shape[0]):
        su = survivors[e]
        lo = losers[e]
        if not alive[su]:
            continue  # the survivor was subducted later in this step; its mass already moved on
        # only what the survivor actually received: an oceanic slab hands over
        # `accretion` of itself and the rest goes to the mantle, so spreading
        # the whole slab would create mass that was never accreted
        if not np.isnan(recv_th[e]):
            # the extent model said exactly what moved: a sliver of the loser's column, not
            # the whole of it -- spreading the whole stripped the survivor to nothing.  And
            # only a gain: a merge that thinned the survivor handed it nothing to spread
            m = recv_m[e]
            th = recv_th[e]
            if th <= 0.0 or m <= 0.0:
                continue
        else:
            f = _received_fraction(kind[lo], alive[lo], accretion, shortening)
            m = f * mass[lo]
            th = f * thickness[lo]
        tot = 0.0
        tot_e = 0.0
        for q in range(K):
            n = nbrs[e, q]
            w[q] = 0.0
            if n < 0 or not alive[n] or plate_id[n] != plate_id[su] or kind[n] != kind[su]:
                continue
            dx = pos[n, 0] - pos[su, 0]
            dy = pos[n, 1] - pos[su, 1]
            dz = pos[n, 2] - pos[su, 2]
            w[q] = np.exp(-(dx * dx + dy * dy + dz * dz) * inv2s2)
            tot += w[q]
            if use_ext:
                tot_e += w[q] * ext[n]
        if tot <= 0.0:
            continue
        # the survivor already holds (m, th); hand the neighbours their share
        for q in range(K):
            if w[q] <= 0.0:
                continue
            n = nbrs[e, q]
            if n == su:
                continue
            if use_ext:
                # The Gaussian says how much each neighbour rises, so the crust (volume)
                # lands by weight x the ground under it: every neighbour's column rises by its
                # weight's share whatever its extent, and none is amplified for being small
                g = w[q] * ext[n] / max(tot_e, 1e-300)
                mass[su] -= g * m
                thickness[su] -= g * th
                r = w[q] * ext[su] / max(tot_e, 1e-300)
                mass[n] += r * m
                thickness[n] += r * th
            else:
                f = w[q] / tot
                mass[su] -= f * m
                thickness[su] -= f * th
                mass[n] += f * m
                thickness[n] += f * th
            density[n] = mass[n] / thickness[n]
        density[su] = mass[su] / thickness[su]


def spread_collisions(seg: Segments, tree: cKDTree, losers: np.ndarray, survivors: np.ndarray, alive: np.ndarray, sigma: float, knn: int = 12, accretion: float = 1.0, shortening: float = 0.0, received=None,
                      conserve_volume: bool = False) -> None:
    """Belt formation: the mass and thickness a survivor just received from
    a subducted segment are shared, with Gaussian weights ``exp(-d²/2σ²)``,
    among the survivor and its ``knn`` nearest *live, same-plate, same-kind* segments
    (the survivor itself is among them with weight 1), so repeated
    collisions along a boundary build a belt ~2σ wide instead of isolated
    peaks.  Mass conserving.  Must run before ``seg.compress`` (the dead
    losers' arrays still hold the transferred amounts).

    ``conserve_volume`` (``tectonics.variable_extent``): the crust is shared
    as volume, by Gaussian weight times the receiver's extent, so every
    neighbour's column rises by its weight's share and ``sum(ext *
    thickness)`` and ``sum(ext * mass)`` do not move.  Off, a column moves
    column for column, which conserves only the plain sums -- right when
    every extent is the same, and the fixed-area model bit for bit."""
    if losers.size == 0:
        return
    kk = min(int(knn), tree.n)
    _, nb = tree.query(seg.pos[survivors], k=kk, workers=kd_workers(survivors.size))
    nb = np.atleast_2d(nb).reshape(survivors.size, kk).astype(np.int64)
    rt = np.ascontiguousarray(received[0], dtype=np.float64) if received is not None else np.full(len(losers), np.nan)
    rm = np.ascontiguousarray(received[1], dtype=np.float64) if received is not None else np.full(len(losers), np.nan)
    _spread_kernel(np.ascontiguousarray(losers), np.ascontiguousarray(survivors), np.ascontiguousarray(nb), seg.pos, seg.plate_id, seg.kind, seg.mass, seg.thickness, seg.density, alive, 1.0 / (2.0 * float(sigma) ** 2), float(accretion), float(shortening), rt, rm,
                   seg.ext, bool(conserve_volume))


@njit(cache=True)
def _segment_cascade(order, nbrs, kind, thickness, mass, density, alive, rate, thr):
    K = nbrs.shape[1]
    for e in range(order.shape[0]):
        s = order[e]
        if not alive[s]:
            continue
        for q in range(K):
            n = nbrs[e, q]
            if n == s or n < 0 or not alive[n] or kind[n] != kind[s]:
                continue
            ds = density[s]
            hs = thickness[s] * (1.0 - ds)
            hn = thickness[n] * (1.0 - density[n])
            delta = hs - hn
            if delta <= thr or ds >= 0.999:
                continue
            dh = rate * (delta - thr) * 0.5 / K
            dth = dh / (1.0 - ds)
            if dth > 0.5 * thickness[s]:
                dth = 0.5 * thickness[s]  # oceanic crust is thin; unclamped this went negative
            thickness[s] -= dth
            mass[s] -= dth * ds
            thickness[n] += dth
            mass[n] += dth * ds
            density[s] = mass[s] / thickness[s]
            density[n] = mass[n] / thickness[n]


def segment_cascade(seg: Segments, tree: cKDTree, survivors: np.ndarray, alive: np.ndarray, rate: float, threshold: float, knn: int = 8) -> None:
    """PLAN 6.2.5 / 6.4 on segments: for each survivor (in order) move
    height ``rate * (Δh - threshold) / 2`` (split over its ``knn`` nearest
    live neighbours) to lower neighbours as thickness at the survivor's
    density — mass conserving."""
    if survivors.size == 0:
        return
    order = np.unique(survivors)
    _, nb = tree.query(seg.pos[order], k=min(knn + 1, tree.n), workers=kd_workers(order.size))
    nb = np.atleast_2d(nb).astype(np.int64)
    _segment_cascade(order, np.ascontiguousarray(nb), seg.kind, seg.thickness, seg.mass, seg.density, alive, float(rate), float(threshold))


@njit(cache=True)
def _relax_kernel(nbrs, pos, kind, thickness, mass, density, rate, thr_per_rad, ext, use_ext):
    M, K = nbrs.shape
    for s in range(M):
        for q in range(K):
            n = nbrs[s, q]
            if n == s or n < 0 or kind[n] != kind[s]:
                continue
            ds = density[s]
            if ds >= 0.999:
                continue
            hs = thickness[s] * (1.0 - ds)
            hn = thickness[n] * (1.0 - density[n])
            dx = pos[n, 0] - pos[s, 0]
            dy = pos[n, 1] - pos[s, 1]
            dz = pos[n, 2] - pos[s, 2]
            thr = thr_per_rad * np.sqrt(dx * dx + dy * dy + dz * dz)
            delta = hs - hn
            if delta <= thr:
                continue
            if use_ext:
                # The step closes by the same share as ever, split between the two by their
                # ground: the giver drops ext[n] / (ext[s] + ext[n]) of it and the receiver
                # rises by the rest, so the crust that leaves one column is the crust that
                # arrives on the other (column for column lost 2.3-2.8 units of continental
                # volume over 8000 Earth steps) and a small receiver is not driven past the
                # giver.  Equal extents give the old half and half
                dh = rate * (delta - thr) / K * ext[n] / max(ext[s] + ext[n], 1e-300)
            else:
                dh = rate * (delta - thr) * 0.5 / K
            dth = dh / (1.0 - ds)
            if dth > 0.5 * thickness[s]:
                dth = 0.5 * thickness[s]
            thickness[s] -= dth
            mass[s] -= dth * ds
            if use_ext:
                r = ext[s] / max(ext[n], 1e-12)
                thickness[n] += dth * r
                mass[n] += dth * ds * r
            else:
                thickness[n] += dth
                mass[n] += dth * ds
            density[s] = mass[s] / thickness[s]
            density[n] = mass[n] / thickness[n]


def relax_segments(seg: Segments, tree: cKDTree, rate: float, threshold_per_spacing: float, spacing: float, knn: int = 8,
                   conserve_volume: bool = False) -> None:
    """PLAN 6.4 cascade applied to the segment cloud every step: for every
    segment (index order) and each of its ``knn`` nearest *same-kind*
    neighbours whose bedrock height is lower by more than
    ``threshold_per_spacing`` × their
    distance (in spacings), move ``rate * (Δh - thr) / 2 / knn`` of height
    downhill as thickness at the giver's density (mass conserving).  Turns
    stacked collision peaks into belts with foothills; the threshold is
    the maximum stable slope in bedrock units per spacing.

    Restricted to same-kind pairs because the continent-ocean contact is a
    ~4 km step in bedrock height, far above any plausible threshold, so an
    unrestricted cascade drains every coastal continental segment into the
    seafloor beside it -- exactly the leak that closes the gap the crust
    types exist to open.

    ``conserve_volume`` (``tectonics.variable_extent``): what leaves a column
    of extent ``ext[s]`` arrives on one of ``ext[n]`` scaled by their ratio,
    so ``sum(ext * thickness)`` does not move, and the height step is split
    between the two by their extents rather than half and half, so a small
    receiver does not overshoot; off, the plain sum is what is conserved (the
    fixed-area model, bit for bit)."""
    if seg.M < 2:
        return
    kk = min(int(knn) + 1, tree.n)
    _, nb = tree.query(seg.pos, k=kk, workers=kd_workers(seg.M))
    nb = np.atleast_2d(nb).reshape(seg.M, kk).astype(np.int64)
    _relax_kernel(np.ascontiguousarray(nb), seg.pos, seg.kind, seg.thickness, seg.mass, seg.density, float(rate), float(threshold_per_spacing) / float(spacing),
                  seg.ext, bool(conserve_volume))


def delaminate(seg: Segments, limit: float, rate: float) -> float:
    """Shed the root of over-thickened crust; returns the mass lost.

    Continent-on-continent collision stacks crust with nothing to stop it:
    every orogeny doubles the survivor, so measured over 1500 steps a
    handful of segments reached thickness 8.3 against a continental median
    of 2.2.  That tail is not harmless -- the vertical scale in
    :func:`~globe.tectonics.run.finalise` is pinned to a high percentile of
    land, so a few runaway spikes squash the whole map beneath them (the
    land/ocean gap came out at 400 m, worse than before crust types
    existed).

    Earth does not do this: crustal thickness saturates near 70 km, about
    twice normal, even under Tibet after 50 My of the largest collision
    going.  The reason is that a thick root is pushed into the eclogite
    field, becomes denser than the mantle it sits in, and founders.  So
    thickness above ``limit`` decays toward it at ``rate`` per step, and the
    mass goes to the mantle rather than to a neighbour -- delamination is a
    loss, not a transfer.
    """
    if limit <= 0.0 or rate <= 0.0:
        return 0.0
    over = seg.thickness > limit
    if not over.any():
        return 0.0
    excess = (seg.thickness[over] - limit) * float(rate)
    lost = float((excess * seg.density[over]).sum())
    seg.thickness[over] -= excess
    seg.mass[over] = seg.thickness[over] * seg.density[over]
    return lost


# --------------------------------------------------------------------------
# heat sources
# --------------------------------------------------------------------------
class CellTree:
    """Static KD-tree of the interior cell centres of a grid, for
    seamless 'all cells within r of a point' queries (heat blobs, collision
    zones)."""

    def __init__(self, grid: Grid):
        self.grid = grid
        self.centers = np.ascontiguousarray(grid.interior_centers.reshape(-1, 3))
        self.tree = cKDTree(self.centers)

    def add_blobs(self, flat_interior: np.ndarray, points: np.ndarray, peak: float, radius: float) -> None:
        """``flat_interior += peak * exp(-d² / (2 (radius/2)²))`` for every
        cell within ``radius`` of every point (accumulating, order-free)."""
        if points.shape[0] == 0:
            return
        sigma2 = (0.5 * radius) ** 2
        lists = self.tree.query_ball_point(points, radius, workers=kd_workers(points.shape[0]))
        cells = np.concatenate([np.asarray(l, dtype=np.int64) for l in lists]) if len(lists) else np.zeros(0, np.int64)
        if cells.size == 0:
            return
        src = np.repeat(np.arange(points.shape[0]), [len(l) for l in lists])
        d2 = np.sum((self.centers[cells] - points[src]) ** 2, axis=1)
        np.add.at(flat_interior, cells, peak * np.exp(-d2 / (2.0 * sigma2)))

    def within(self, points: np.ndarray, radius: float) -> np.ndarray:
        """Bool mask (6, N, N) of cells within ``radius`` of any point."""
        N = self.grid.N
        mask = np.zeros(6 * N * N, dtype=bool)
        if points.shape[0]:
            lists = self.tree.query_ball_point(points, radius, workers=kd_workers(points.shape[0]))
            for l in lists:
                mask[np.asarray(l, dtype=np.int64)] = True
        return mask.reshape(6, N, N)


# --------------------------------------------------------------------------
# grid post-processing (PLAN 6.3 / 6.4)
# --------------------------------------------------------------------------
@njit(cache=True, parallel=True)
def _cascade_pass(d, out, rate, thr):
    F, NE, _ = d.shape
    for f in prange(F):
        for i in range(1, NE - 1):
            for j in range(1, NE - 1):
                h = d[f, i, j]
                acc = 0.0
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        if di == 0 and dj == 0:
                            continue
                        t = thr * math.sqrt(float(di * di + dj * dj))
                        delta = h - d[f, i + di, j + dj]
                        if delta > t:
                            acc -= rate * (delta - t) * 0.5 / 8.0
                        elif -delta > t:
                            acc += rate * (-delta - t) * 0.5 / 8.0
                out[f, i, j] = h + acc


def grid_cascade(field: FaceField, passes: int, rate: float, threshold: float) -> FaceField:
    """PLAN 6.4 height cascade on a grid: between 8-neighbours whose
    heights differ by more than ``threshold`` (× neighbour distance) move
    ``rate * (Δh - threshold) / 2`` (split over the 8 neighbours) from high
    to low; symmetric, hence conservative.  Halos exchanged between passes."""
    d = field.data.astype(np.float64)
    out = d.copy()
    f = FaceField(field.grid, d, name=field.name)
    for _ in range(int(passes)):
        f.exchange_halos()
        _cascade_pass(f.data, out, float(rate), float(threshold))
        f.data, out = out, f.data
    f.exchange_halos()
    return f


@njit(cache=True, parallel=True)
def _blur_i(d, out, w, r):
    F, NE, _ = d.shape
    for f in prange(F):
        for i in range(r, NE - r):
            for j in range(NE):
                acc = 0.0
                for k in range(-r, r + 1):
                    acc += w[k + r] * d[f, i + k, j]
                out[f, i, j] = acc


@njit(cache=True, parallel=True)
def _blur_j(d, out, w, r):
    F, NE, _ = d.shape
    for f in prange(F):
        for i in range(NE):
            for j in range(r, NE - r):
                acc = 0.0
                for k in range(-r, r + 1):
                    acc += w[k + r] * d[f, i, j + k]
                out[f, i, j] = acc


def gaussian_smooth(field: FaceField, sigma_cells: float) -> FaceField:
    """Separable Gaussian blur with halo exchange (seamless).  The kernel
    radius is limited to ``H - 1`` cells per pass, so larger sigmas are
    reached with several passes (``n = ceil(sigma²)``, each ``sigma/√n``)."""
    d = field.data.astype(np.float64)
    f = FaceField(field.grid, d, name=field.name)
    if sigma_cells <= 1e-3:
        f.exchange_halos()
        return f
    H = field.grid.H
    n = max(1, int(math.ceil(sigma_cells**2)))
    s = sigma_cells / math.sqrt(n)
    r = int(max(1, min(H - 1, math.ceil(3.0 * s))))
    k = np.arange(-r, r + 1, dtype=np.float64)
    w = np.exp(-0.5 * (k / s) ** 2)
    w /= w.sum()
    out = np.empty_like(d)
    for _ in range(n):
        f.exchange_halos()
        _blur_i(f.data, out, w, r)
        f.data, out = out, f.data
        f.exchange_halos()
        _blur_j(f.data, out, w, r)
        f.data, out = out, f.data
    f.exchange_halos()
    return f


def boundary_distance(tree: cKDTree, seg: Segments, grid: Grid, idx: np.ndarray, k: int = 12) -> np.ndarray:
    """Chord distance from every interior cell to the nearest segment of a
    plate *different* from the cell's own (6, N, N); cells with no foreign
    segment among the ``k`` nearest get the distance of the k-th."""
    c = interior_centers_flat(grid)
    kk = min(k, tree.n)
    dist, nb = tree.query(c, k=kk, workers=kd_workers(c.shape[0]))
    dist = np.atleast_2d(dist).reshape(c.shape[0], kk)
    nb = np.atleast_2d(nb).reshape(c.shape[0], kk)
    own = seg.plate_id[idx.ravel()]
    foreign = seg.plate_id[nb] != own[:, None]
    first = np.where(foreign.any(axis=1), np.argmax(foreign, axis=1), kk - 1)
    return dist[np.arange(c.shape[0]), first].reshape(6, grid.N, grid.N)


def resample_to(field: FaceField, grid: Grid, nearest: bool = False) -> np.ndarray:
    """Sample a (halo-exchanged) field of one resolution on the interior
    cell centres of another grid -> (6, N, N[, C]) array."""
    p = grid.interior_centers
    if nearest:
        face, u, v = from_sphere_v(p)
        return field.sample_nearest(face, u, v)
    return field.sample_sphere(p)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Area-weighted quantile (``q`` in [0, 1]) of ``values``."""
    v = np.asarray(values, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    order = np.argsort(v, kind="stable")
    cw = np.cumsum(w[order])
    return float(v[order][min(int(np.searchsorted(cw, q * cw[-1])), v.size - 1)])


__all__ = [
    "build_tree", "kd_workers", "KD_SERIAL_BELOW", "label_map", "label_map_fast", "cell_area_steradians", "accumulate_area", "splat",
    "SmoothSplat", "splat_weights", "wendland_support", "SPLAT_KERNELS", "deposit_density", "crystallise", "spawn_segments", "collide", "spread_collisions", "segment_cascade", "relax_segments",
    "CellTree", "grid_cascade", "gaussian_smooth", "boundary_distance", "resample_to", "weighted_quantile",
]
