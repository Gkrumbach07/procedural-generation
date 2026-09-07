"""Erosion state, one-iteration driver, discharge/momentum EMA, thermal
erosion and uplift (PLAN 8.1 / 8.2).

:class:`ErosionState` is the plain-array state the kernels operate on.  It
is built either from the global coarse fields (:meth:`ErosionState.from_grid`)
or, by the refine stage, from a single-face basin window
(:meth:`ErosionState.window`).  Heights inside the state are in *cell units*
(``metres / height_unit_m``, ``height_unit_m = cell_size_m`` by default).

Typical use::

    state = ErosionState.from_grid(grid, bedrock_m, hardness, precip, evap, uplift_m, params.erosion)
    for it in range(n):
        step(state, params, it)          # particles + thermal + uplift + halos
    height_m, sediment_m = state.height_m(), state.sediment_m()

Everything random comes from ``params.rng(rng_stage, *iteration_key)``.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from numba import get_num_threads, parallel_chunksize

from ..config import ErosionParams, WorldParams
from ..cubesphere import Grid
from ..field import FaceField
from . import particle as pk


@dataclass
class ErosionState:
    """Plain-array erosion state (see module docstring).

    All arrays are extended ``(F, NE, NE[, 2])``; ``height``/``sediment``
    are float64 in cell units, the maps float32/float64 as noted.

    ``mask``: uint8, 0 outside (particles die, never written), 1 active,
    2 frozen (sampled but never modified — basin divides in refinement).
    ``spherical``: True for the 6-face global state (particles transfer
    across faces), False for a basin window (particles leaving the array or
    the mask die).
    """

    height: np.ndarray  # (F, NE, NE) float64, cell units
    sediment: np.ndarray  # (F, NE, NE) float64, cell units
    discharge: np.ndarray  # (F, NE, NE) float64, EMA of volume per iteration
    momentum: np.ndarray  # (F, NE, NE, 2) float64, EMA of volume-weighted velocity
    hardness: np.ndarray  # (F, NE, NE) float32 [0, 1]
    precip: np.ndarray  # (F, NE, NE) float32 spawn weight (volume per cell per iteration)
    evap: np.ndarray  # (F, NE, NE) float32 evaporation multiplier
    uplift: np.ndarray  # (F, NE, NE) float64 cell units per iteration
    mask: np.ndarray  # (F, NE, NE) uint8
    metric: np.ndarray  # (F, NE, NE, 3) float32 dimensionless metric (g_ii, g_ij, g_jj)
    metric_inv: np.ndarray  # (F, NE, NE, 3) float32
    N: int
    H: int
    spherical: bool
    height_unit_m: float
    grid: Grid | None = None  # needed for halo exchange in spherical mode
    route: np.ndarray | None = None  # (F, NE, NE) float64 routing surface (see route.py); None = steer on the terrain
    disch_track: np.ndarray = field(default=None, repr=False)
    mom_track: np.ndarray = field(default=None, repr=False)
    samp: np.ndarray = field(default=None, repr=False)  # packed float32 samples (F, NE, NE, NS), rebuilt every iteration
    acc: np.ndarray = field(default=None, repr=False)  # float64 net terrain change of the current iteration (cell units), zero between iterations
    pending: np.ndarray = field(default=None, repr=False)  # float64 sediment stockpile per cell (cell units) that found no room this iteration (particle.apply_changes); re-injected as a loaded particle next iteration; part of the mass balance
    _owner: np.ndarray = field(default=None, repr=False)
    iteration: int = 0
    deposit_on_exit: bool = False  # window mode: deposit the load at the last active cell when leaving

    def __post_init__(self):
        F, NE = self.height.shape[0], self.height.shape[1]
        if self.disch_track is None:
            self.disch_track = np.zeros((F, NE, NE), dtype=np.float64)
        if self.mom_track is None:
            self.mom_track = np.zeros((F, NE, NE, 2), dtype=np.float64)
        if self.samp is None:
            self.samp = np.zeros((F, NE, NE, pk.NS), dtype=np.float32)
        if self.acc is None:
            self.acc = np.zeros((F, NE, NE), dtype=np.float64)
        if self.pending is None:
            self.pending = np.zeros((F, NE, NE), dtype=np.float64)
        assert NE == self.N + 2 * self.H, "array width must be N + 2H"
        for a, shape in (
            (self.sediment, (F, NE, NE)),
            (self.discharge, (F, NE, NE)),
            (self.momentum, (F, NE, NE, 2)),
            (self.hardness, (F, NE, NE)),
            (self.precip, (F, NE, NE)),
            (self.evap, (F, NE, NE)),
            (self.uplift, (F, NE, NE)),
            (self.mask, (F, NE, NE)),
            (self.metric, (F, NE, NE, 3)),
            (self.metric_inv, (F, NE, NE, 3)),
        ):
            assert tuple(a.shape) == shape, (a.shape, shape)

    # -- constructors --------------------------------------------------------
    @classmethod
    def from_grid(
        cls,
        grid: Grid,
        bedrock_m,
        hardness,
        precip,
        evap,
        uplift_m,
        eparams: ErosionParams | None = None,
        mask: np.ndarray | None = None,
    ) -> "ErosionState":
        """Global 6-face state from extended arrays / FaceFields in metres."""
        unit = height_unit(grid, eparams)

        def arr(x, dtype):
            d = x.data if isinstance(x, FaceField) else x
            return np.ascontiguousarray(np.asarray(d), dtype=dtype)

        F, NE = 6, grid.NE
        height = arr(bedrock_m, np.float64) / unit
        m = np.ones((F, NE, NE), dtype=np.uint8) if mask is None else np.ascontiguousarray(mask, dtype=np.uint8)
        return cls(
            height=np.ascontiguousarray(height),
            sediment=np.zeros((F, NE, NE), dtype=np.float64),
            discharge=np.zeros((F, NE, NE), dtype=np.float64),
            momentum=np.zeros((F, NE, NE, 2), dtype=np.float64),
            hardness=arr(hardness, np.float32),
            precip=arr(precip, np.float32),
            evap=arr(evap, np.float32),
            uplift=np.ascontiguousarray(arr(uplift_m, np.float64) / unit),
            mask=m,
            metric=np.ascontiguousarray(grid.metric, dtype=np.float32),
            metric_inv=np.ascontiguousarray(grid.metric_inv, dtype=np.float32),
            N=grid.N,
            H=grid.H,
            spherical=True,
            height_unit_m=unit,
            grid=grid,
        )

    @classmethod
    def window(
        cls,
        height_m: np.ndarray,
        sediment_m: np.ndarray,
        discharge: np.ndarray,
        momentum: np.ndarray,
        hardness: np.ndarray,
        precip: np.ndarray,
        evap: np.ndarray,
        uplift_m: np.ndarray,
        mask: np.ndarray,
        metric: np.ndarray,
        metric_inv: np.ndarray,
        H: int,
        height_unit_m: float,
        deposit_on_exit: bool = False,
        route_m: np.ndarray | None = None,
    ) -> "ErosionState":
        """Single-face window state (``F = 1``, ``spherical=False``) from
        ``(NE, NE[, 2])`` arrays in metres (``NE = N + 2H``).  Particles
        leaving the array or a mask-0 cell die; with ``deposit_on_exit``
        they drop their load at the last active cell first (conservation)."""
        NE = height_m.shape[0]
        N = NE - 2 * H

        def a(x, dtype, extra=()):
            return np.ascontiguousarray(np.asarray(x), dtype=dtype).reshape((1, NE, NE) + tuple(extra))

        return cls(
            height=a(height_m, np.float64) / height_unit_m,
            sediment=a(sediment_m, np.float64) / height_unit_m,
            discharge=a(discharge, np.float64),
            momentum=a(momentum, np.float64, (2,)),
            hardness=a(hardness, np.float32),
            precip=a(precip, np.float32),
            evap=a(evap, np.float32),
            uplift=a(uplift_m, np.float64) / height_unit_m,
            mask=a(mask, np.uint8),
            metric=a(metric, np.float32, (3,)),
            metric_inv=a(metric_inv, np.float32, (3,)),
            N=N,
            H=H,
            spherical=False,
            height_unit_m=float(height_unit_m),
            deposit_on_exit=deposit_on_exit,
            route=None if route_m is None else a(route_m, np.float64) / height_unit_m,
        )

    # -- views ---------------------------------------------------------------
    @property
    def F(self) -> int:
        return self.height.shape[0]

    @property
    def NE(self) -> int:
        return self.height.shape[1]

    @property
    def interior(self):
        H, N = self.H, self.N
        return (slice(None), slice(H, H + N), slice(H, H + N))

    def surface(self) -> np.ndarray:
        return self.height + self.sediment

    def pack(self) -> None:
        """Refresh the packed float32 sample array from the state arrays."""
        use_route = self.route is not None
        pk.pack_samples(self.samp, self.height, self.sediment, self.route if use_route else self.height, use_route, self.discharge, self.momentum, self.evap, self.hardness, self.metric, self.metric_inv)

    def height_m(self) -> np.ndarray:
        return self.height * self.height_unit_m

    def sediment_m(self) -> np.ndarray:
        return self.sediment * self.height_unit_m

    def total_mass(self) -> float:
        """Σ (height + sediment + pending) over active interior cells (cell
        units) — the quantity the particle pass conserves exactly."""
        act = self.mask[self.interior] == pk.MASK_ACTIVE
        return float(np.sum((self.height + self.sediment + self.pending)[self.interior][act]))

    # -- halos ---------------------------------------------------------------
    def exchange_halos(self) -> None:
        """Spherical mode: refresh the halos of every evolving field (the
        vector momentum is rotated into the neighbouring face's basis)."""
        if not self.spherical:
            return
        if self.grid is None:
            raise ValueError("spherical ErosionState needs its Grid for halo exchange")
        for arr in (self.height, self.sediment, self.discharge):
            FaceField(self.grid, arr).exchange_halos()
        FaceField(self.grid, self.momentum, is_vector=True).exchange_halos()
        if self.route is not None:
            FaceField(self.grid, self.route).exchange_halos(linear=True)

    def owner_table(self) -> np.ndarray:
        """Flat index of the interior cell owning every extended cell
        (``grid.owner`` in spherical mode, identity for a window)."""
        if self._owner is None:
            from .route import window_owner

            self._owner = np.ascontiguousarray(self.grid.owner if self.spherical else window_owner(self.F, self.NE), dtype=np.int64)
        return self._owner

    def refresh_route(self, eps: float) -> None:
        """Recompute the epsilon-filled routing surface from the current
        terrain (:mod:`globe.erosion.route`)."""
        from .route import priority_flood_eps

        self.route = priority_flood_eps(self.surface(), self.mask, self.owner_table(), self.H, self.N, float(eps))
        if self.spherical:
            FaceField(self.grid, self.route).exchange_halos(linear=True)

    def fields(self, names=("height", "sediment", "discharge", "momentum")) -> dict[str, FaceField]:
        """Output FaceFields in metres (spherical mode)."""
        if self.grid is None:
            raise ValueError("fields() needs a Grid")
        out = {}
        for n in names:
            if n == "height":
                out[n] = FaceField(self.grid, self.height_m().astype(np.float32), name="height")
            elif n == "sediment":
                out[n] = FaceField(self.grid, self.sediment_m().astype(np.float32), name="sediment")
            elif n == "discharge":
                out[n] = FaceField(self.grid, self.discharge.astype(np.float32), name="discharge")
            elif n == "momentum":
                out[n] = FaceField(self.grid, self.momentum.astype(np.float32), is_vector=True, name="momentum")
        return out


def height_unit(grid: Grid, eparams: ErosionParams | None) -> float:
    """Kernel height unit: the grid's cell size.  ``erosion.height_unit_m``
    accepts only 0, 1 (both = cell size) or the cell size itself — every
    cap, talus slope and the gravity term are defined in cell units, so a
    different unit would silently rescale the model."""
    u = float(eparams.height_unit_m) if eparams is not None else 0.0
    if u in (0.0, 1.0) or u == grid.cell_size_m:
        return grid.cell_size_m
    raise ValueError(f"erosion.height_unit_m={u} is not supported: the kernel works in cell units ({grid.cell_size_m} m); set 0")


def _eparams(params) -> ErosionParams:
    return params.erosion if isinstance(params, WorldParams) else params


def max_steps_of(ep: ErosionParams, N: int) -> int:
    return int(ep.max_steps) if ep.max_steps > 0 else 2 * int(N)


#: an iteration is split into at least this many chunks (small runs would
#: otherwise freeze the terrain for the whole iteration)
MIN_CHUNKS = 16


def chunk_size(ep: ErosionParams, n_particles: int) -> int:
    return max(1, min(int(ep.chunk), -(-int(n_particles) // MIN_CHUNKS)))


# --------------------------------------------------------------------------
# spawning
# --------------------------------------------------------------------------
def spawn_weights(state: ErosionState) -> tuple[np.ndarray, np.ndarray]:
    """Flat indices of spawnable interior cells (land, mask 1 — frozen
    divides never spawn — precip > 0) and their weights (precip)."""
    F, NE, H, N = state.F, state.NE, state.H, state.N
    inter = np.zeros((F, NE, NE), dtype=bool)
    inter[state.interior] = True
    ok = inter & (state.mask == pk.MASK_ACTIVE) & (state.height + state.sediment >= 0.0) & (state.precip > 0)
    idx = np.flatnonzero(ok)
    w = state.precip.reshape(-1)[idx].astype(np.float64)
    return idx, w


def spawn_particles(state: ErosionState, n_particles: int, rng: np.random.Generator):
    """Deterministic spawn: cell ∝ precip via inverse CDF on sorted
    uniforms (so particles come out sorted by cell — cache friendly), a
    uniform offset inside the cell, and ``volume = total weight / n``
    (PLAN 8.2's ``precip[cell] / expected_hits``).  Returns
    ``(face, x, y, volume)`` or ``None`` if nothing can spawn."""
    idx, w = spawn_weights(state)
    if idx.size == 0 or n_particles <= 0:
        return None
    total = float(w.sum())
    if total <= 0.0:
        return None
    cdf = np.cumsum(w)
    u = np.sort(rng.random(n_particles))
    off = rng.random((n_particles, 2))
    k = pk.spawn_cells(cdf, u)
    flat = idx[k]
    NE = state.NE
    f = flat // (NE * NE)
    rem = flat - f * NE * NE
    ei = rem // NE
    ej = rem - ei * NE
    x = (ei - state.H).astype(np.float64) + off[:, 0]
    y = (ej - state.H).astype(np.float64) + off[:, 1]
    return f.astype(np.int64), x, y, total / n_particles


# --------------------------------------------------------------------------
# one iteration
# --------------------------------------------------------------------------
def run_iteration(
    state: ErosionState,
    params,
    iteration_key,
    n_particles: int | None = None,
    particles_per_cell: float | None = None,
    rng_stage: str = "erosion",
    log=None,
) -> dict:
    """PLAN 8.2, particle part: spawn (rain + re-injected stockpiles),
    trace in chunks, apply each change list serially on the live terrain
    (:func:`particle.apply_changes`), fold the iteration's net change into
    height/sediment, then EMA-update ``discharge``/``momentum``.
    ``iteration_key`` is an int or a tuple of ints mixed into
    ``params.rng(rng_stage, *key)``.  ``params`` may be a ``WorldParams`` or
    an ``ErosionParams`` (then ``rng`` must be reachable: pass a WorldParams
    for real runs).  Does *not* do thermal erosion, uplift or halos — see
    :func:`step`."""
    ep = _eparams(params)
    key = iteration_key if isinstance(iteration_key, (tuple, list)) else (int(iteration_key),)
    rng = params.rng(rng_stage, *key)
    t0 = time.time()
    n_cells = int(np.count_nonzero(state.mask[state.interior] == pk.MASK_ACTIVE))
    if n_particles is None:
        ppc = ep.particles_per_cell if particles_per_cell is None else particles_per_cell
        n_particles = max(1, int(round(ppc * n_cells)))
    sp = spawn_particles(state, n_particles, rng)
    stats = {"particles": 0, "entries": 0, "steps_mean": 0.0}
    if sp is None:
        return stats
    sp_face, sp_x, sp_y, volume0 = sp
    sp_sed = np.zeros(sp_face.shape[0], dtype=np.float64)
    sp_dir = np.zeros(sp_face.shape[0], dtype=np.float64)
    # re-inject the pending stockpiles as particles that start with that
    # load (appended after the rain, in cell order: deterministic)
    pend_cells = np.flatnonzero(state.pending.reshape(-1) > 0.0)
    released = 0.0
    if pend_cells.size:
        NE = state.NE
        pf = pend_cells // (NE * NE)
        rem = pend_cells - pf * NE * NE
        pei = rem // NE
        pej = rem - pei * NE
        loads = state.pending.reshape(-1)[pend_cells]
        released = float(loads.sum())
        sp_face = np.concatenate([sp_face, pf.astype(np.int64)])
        sp_x = np.concatenate([sp_x, (pei - state.H) + 0.5])
        sp_y = np.concatenate([sp_y, (pej - state.H) + 0.5])
        sp_sed = np.concatenate([sp_sed, loads])
        sp_dir = np.concatenate([sp_dir, rng.random(pend_cells.size) * (2.0 * np.pi)])
        state.pending.reshape(-1)[pend_cells] = 0.0
    state.pack()
    state.acc[...] = 0.0
    max_steps = max_steps_of(ep, state.N)
    cap = max_steps + 2 * pk.SPREAD
    P = sp_face.shape[0]
    chunk = chunk_size(ep, P)
    # change list buffers (reused for every chunk)
    cl_cell = np.empty(chunk * cap, dtype=np.int32)
    cl_delta = np.empty(chunk * cap, dtype=np.float64)
    cl_vol = np.empty(chunk * cap, dtype=np.float32)
    cl_mom = np.empty((chunk * cap, 2), dtype=np.float32)
    cl_count = np.zeros(chunk, dtype=np.int64)
    cl_lo = np.zeros(chunk, dtype=np.int64)
    cl_hi = np.zeros(chunk, dtype=np.int64)
    sp_death = np.zeros(chunk, dtype=np.int8)
    deaths = np.zeros(len(pk.DEATH_NAMES), dtype=np.int64)
    entries = 0
    n_clamp = 0
    to_pending = 0.0
    lost = 0.0
    t_trace = 0.0
    t_apply = 0.0
    for c0 in range(0, P, chunk):
        c1 = min(P, c0 + chunk)
        m = c1 - c0
        t1 = time.time()
        with parallel_chunksize(4):  # dynamic scheduling: particle lifetimes vary a lot
            pk.trace_particles(
                sp_face[c0:c1],
                sp_x[c0:c1],
                sp_y[c0:c1],
                sp_sed[c0:c1],
                sp_dir[c0:c1],
                float(volume0),
                state.samp,
                state.route is not None,
                state.mask,
                int(state.N),
                int(state.H),
                bool(state.spherical),
                float(ep.dt),
                float(ep.density),
                float(ep.friction),
                float(ep.deposition_rate),
                float(ep.evap_rate),
                float(ep.k_mom),
                float(ep.k_disc),
                float(ep.disc_saturation),
                float(ep.slope_gain),
                float(ep.slope_saturation),
                float(ep.erodibility),
                float(ep.min_volume),
                int(max_steps),
                float(ep.max_erode),
                float(chunk / (volume0 * P)),
                int(ep.pit_steps),
                bool(state.deposit_on_exit),
                float(ep.ocean_deposition_rate),
                int(ep.ocean_steps),
                cl_cell,
                cl_delta,
                cl_vol,
                cl_mom,
                cl_count[:m],
                cl_lo[:m],
                cl_hi[:m],
                sp_death[:m],
            )
        t2 = time.time()
        nc, tp, lo = pk.apply_changes(
            cl_cell, cl_delta, cl_vol, cl_mom, cl_count[:m], cap,
            state.height, state.sediment, state.acc, state.pending, state.samp, state.disch_track, state.mom_track, state.mask,
            float(ep.iter_erode), float(ep.iter_deposit), float(ep.fan_slope),
        )
        n_clamp += int(nc)
        to_pending += tp
        lost += lo
        t_trace += t2 - t1
        t_apply += time.time() - t2
        entries += int(cl_count[:m].sum())
        deaths += np.bincount(sp_death[:m], minlength=deaths.size)
    pk.fold_changes(state.height, state.sediment, state.acc, state.mask)
    pk.ema_update(state.discharge, state.momentum, state.disch_track, state.mom_track, state.mask, float(ep.ema))
    stats.update({"particles": int(P), "entries": entries, "steps_mean": entries / max(P, 1), "volume": float(volume0), "seconds": time.time() - t0})
    stats["deaths"] = {n: int(c) for n, c in zip(pk.DEATH_NAMES, deaths)}
    stats["clamped"] = n_clamp
    stats["to_pending"] = to_pending
    stats["released"] = float(released)
    stats["pending_total"] = float(state.pending[state.interior].sum())
    stats["deficit_out"] = lost
    stats["seconds_trace"] = t_trace
    stats["seconds_apply"] = t_apply
    stats["chunk"] = int(chunk)
    if log is not None:
        log(f"  particles {P} volume {volume0:.3f} mean steps {stats['steps_mean']:.1f} deaths {stats['deaths']} clamped {n_clamp} pending {stats['pending_total']:.1f} ({stats['seconds']:.2f}s)")
    return stats


def thermal_erosion(state: ErosionState, params) -> None:
    """One pass of talus-limited mass wasting (PLAN 8.2): material above the
    talus slope ``soft + (hard - soft) * hardness`` (rise/run, cell units)
    moves to lower neighbours, sediment first, at most ``thermal_max`` per
    cell and pass, and never above ``-DEP_FLOOR`` into a submerged cell;
    conservative between active cells."""
    ep = _eparams(params)
    talus = (ep.talus_slope_soft + (ep.talus_slope_hard - ep.talus_slope_soft) * state.hardness).astype(np.float64)
    out_total = np.zeros_like(state.height)
    scale = np.zeros_like(state.height)
    pk.thermal_pass_a(state.height, state.sediment, state.metric, talus, state.mask, float(ep.thermal_rate), float(ep.thermal_max), out_total, scale)
    rscale = np.ones_like(state.height)
    pk.thermal_pass_r(state.height, state.sediment, state.metric, talus, state.mask, float(ep.thermal_rate), scale, rscale)
    new_h = state.height.copy()
    new_s = state.sediment.copy()
    pk.thermal_pass_b(state.height, state.sediment, state.metric, talus, state.mask, float(ep.thermal_rate), out_total, scale, rscale, new_h, new_s)
    state.height[...] = new_h
    state.sediment[...] = new_s


def apply_uplift(state: ErosionState) -> None:
    act = state.mask == pk.MASK_ACTIVE
    state.height[act] += state.uplift[act]


def step(state: ErosionState, params, iteration_key, log=None, **kw) -> dict:
    """One full PLAN 8.2 iteration: particles, thermal erosion, uplift,
    halo exchange.  Increments ``state.iteration``."""
    ep = _eparams(params)
    t0 = time.time()
    if ep.flood_every > 0 and (state.route is None or state.iteration % ep.flood_every == 0):
        state.refresh_route(ep.route_eps)
    tr = time.time() - t0
    st = run_iteration(state, params, iteration_key, log=log, **kw)
    t1 = time.time()
    thermal_erosion(state, params)
    apply_uplift(state)
    state.exchange_halos()
    state.iteration += 1
    st["seconds_route"] = tr
    st["seconds_particles"] = t1 - t0 - tr
    st["seconds_total"] = time.time() - t0
    return st


__all__ = ["ErosionState", "run_iteration", "thermal_erosion", "apply_uplift", "step", "spawn_particles", "height_unit", "max_steps_of"]
