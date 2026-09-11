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

Model notes (PLAN milestone 2 tuning, docs/erosion-tuning.md):

* ``erosion.creep_rate`` adds hillslope creep (a talus-0 mass-wasting pass,
  submerged cells inert).  Without it every particle path incises its own
  rill and the drainage density saturates at one channel per ~3 cells;
  with it interfluves round off and a dendritic trunk network forms.
* ``erosion.disc_exponent`` (0.5) makes the transport capacity grow with
  discharge without saturating (``1 + k_disc * sqrt(q / disc_saturation)``,
  stream-power concavity): trunk rivers grade to gentler slopes than
  rills, so long profiles are concave, valleys widen downstream and a
  channel survives on a floodplain.  With the saturating erf law every
  plain became an alluvial fan and nothing could meander.
* Sediment a submarine fan cannot place (a shelf filled to sea level, which
  never becomes land) is lost to the deep ocean (``lost_offshore`` in the
  iteration stats) instead of accumulating in ``pending`` forever.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from numba import get_num_threads, parallel_chunksize

from ..config import cell_units, ErosionParams, WorldParams
from ..cubesphere import Grid
from ..field import FaceField
from . import glacial
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
    diag=None,
) -> dict:
    """PLAN 8.2, particle part: spawn (rain + re-injected stockpiles),
    trace in chunks, apply each change list serially on the live terrain
    (:func:`particle.apply_changes`), fold the iteration's net change into
    height/sediment, then EMA-update ``discharge``/``momentum``.
    ``iteration_key`` is an int or a tuple of ints mixed into
    ``params.rng(rng_stage, *key)``.  ``params`` may be a ``WorldParams`` or
    an ``ErosionParams`` (then ``rng`` must be reachable: pass a WorldParams
    for real runs).  Does *not* do thermal erosion, uplift or halos — see
    :func:`step`.

    ``diag``, when given, is called once per chunk as
    ``diag(state, cl_cell, cl_vol, cl_count, cap, sp_death)`` with that
    chunk's raw change list, *before* :func:`particle.apply_changes` folds
    it in.  It is how a diagnostic gets at what the kernel did without
    re-implementing the loop around it: a particle's entries are
    ``cl_cell[p*cap : p*cap + cl_count[p]]``, its seafloor steps carry
    ``cl_vol == 0``, its final deposits ``cl_vol < 0``, and so its death
    cell is the first entry with ``cl_vol < 0``.  Production passes
    nothing and the branch costs a single ``is None``."""
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
    lost_offshore = 0.0
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
                float(ep.disc_exponent),
                float(ep.slope_gain),
                float(ep.slope_saturation),
                float(ep.erodibility),
                cell_units(ep, "cover_depth", state.height_unit_m),
                cell_units(ep, "min_volume", state.height_unit_m),
                int(max_steps),
                cell_units(ep, "max_erode", state.height_unit_m),
                float(chunk / (volume0 * P)),
                int(ep.pit_steps),
                bool(state.deposit_on_exit),
                float(ep.ocean_deposition_rate),
                int(ep.ocean_steps),
                cell_units(ep, "fan_room", state.height_unit_m),
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
        if diag is not None:
            diag(state, cl_cell, cl_vol, cl_count[:m], cap, sp_death[:m])
        nc, tp, lo, lo_off = pk.apply_changes(
            cl_cell, cl_delta, cl_vol, cl_mom, cl_count[:m], cap,
            state.height, state.sediment, state.acc, state.pending, state.samp, state.disch_track, state.mom_track, state.mask,
            cell_units(ep, "iter_erode", state.height_unit_m),
            cell_units(ep, "iter_deposit", state.height_unit_m),
            float(ep.fan_slope), state.route is not None,
        )
        n_clamp += int(nc)
        to_pending += tp
        lost += lo
        lost_offshore += lo_off
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
    stats["lost_offshore"] = lost_offshore  # load a seafloor walk could not place: left the modelled surface (deep ocean)
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
    conservative between active cells.  With ``creep_rate > 0`` a second
    pass with talus 0 follows (hillslope creep, a linear diffusion of the
    surface at that rate): without it every particle path incises its own
    rill and the drainage density saturates at one channel per ~3 cells
    (parallel micro-rills, no coherent trunk network)."""
    ep = _eparams(params)
    talus = (ep.talus_slope_soft + (ep.talus_slope_hard - ep.talus_slope_soft) * state.hardness).astype(np.float64)
    _mass_wasting_pass(state, talus, float(ep.thermal_rate), cell_units(ep, "thermal_max", state.height_unit_m))
    if ep.creep_rate > 0.0:
        # hillslope creep: the same conservative pass with talus 0 (every
        # lower neighbour receives creep_rate/2 of the height difference).
        # Submerged cells are inert for it (frozen in a temporary mask):
        # creep is a hillslope process, it must not diffuse the coast into
        # the sea (the talus pass still lets sea cliffs collapse).
        cmask = np.where(state.height + state.sediment < 0.0, np.uint8(pk.MASK_FROZEN), state.mask).astype(np.uint8)
        _mass_wasting_pass(state, np.zeros_like(talus), float(ep.creep_rate), cell_units(ep, "thermal_max", state.height_unit_m), cmask)


def _mass_wasting_pass(state: ErosionState, talus: np.ndarray, rate: float, cap: float, mask: np.ndarray | None = None) -> None:
    mask = state.mask if mask is None else mask
    out_total = np.zeros_like(state.height)
    scale = np.zeros_like(state.height)
    pk.thermal_pass_a(state.height, state.sediment, state.metric, talus, mask, rate, cap, out_total, scale)
    rscale = np.ones_like(state.height)
    pk.thermal_pass_r(state.height, state.sediment, state.metric, talus, mask, rate, scale, rscale)
    new_h = state.height.copy()
    new_s = state.sediment.copy()
    pk.thermal_pass_b(state.height, state.sediment, state.metric, talus, mask, rate, out_total, scale, rscale, new_h, new_s)
    state.height[...] = new_h
    state.sediment[...] = new_s


def apply_uplift(state: ErosionState) -> None:
    """Add the per-iteration uplift, **mean-free** over the active interior.

    The raw tectonic uplift has a positive area mean (it is a growth rate,
    not a redistribution), so adding it verbatim inflates the whole planet
    a little every iteration and the erosion kernel's base level (z = 0)
    drifts away from sea level; hydro's re-quantile then has to drop the
    finished surface by that accumulated amount and drowns the lowlands
    erosion just built.  Removing the mean is a rigid global shift — the
    same thing as raising the datum instead of the crust — and makes the
    applied mass change exactly zero, so uplift only redistributes relief.

    The mean is taken over *interior* active cells of the whole sphere: halo
    cells are copies of neighbouring interior cells (``exchange_halos``
    overwrites them right after) and must not bias it.  No area weighting:
    the rest of the kernel treats cells as equal-mass units
    (:meth:`ErosionState.total_mass`), so an unweighted mean is the
    consistent choice and makes the applied mass change exactly zero.

    Only the global (``spherical``) state has that planetary datum, so a
    window (refinement) applies its uplift as given — a *window* mean would
    subside the basin against a base level it does not own.  Refine passes
    ``uplift = 0``; a windowed run with a real uplift field must be handed
    one that is already mean-free over the whole sphere.
    """
    act = state.mask == pk.MASK_ACTIVE
    mean = 0.0
    if state.spherical:
        iact = state.mask[state.interior] == pk.MASK_ACTIVE
        mean = float(state.uplift[state.interior][iact].mean()) if iact.any() else 0.0
    state.height[act] += state.uplift[act] - mean


def hold_datum(state: ErosionState, land_fraction: float) -> float:
    """Hold the planetary datum: shift ``height`` (a rigid global shift, no
    mass moves between cells) so that exactly ``round(land_fraction * M)``
    of the ``M`` interior cells have ``height + sediment >= 0``.  This is
    the same order statistic :func:`globe.hydro.run.requantile_height`
    applies at the end of the bake, here in cell units and once per
    iteration.  Returns the shift applied (cell units).

    Uplift alone is not the only thing that moves the datum: the surface
    also exports mass to the deep ocean (``lost_offshore``) and buries the
    shelf, so even with a mean-free :func:`apply_uplift` the coastline
    drifts (measured on the small e2e world, 60 iterations at
    ``N_c = 128``: land fraction 0.30 -> 0.248, i.e. hydro still had to
    shift the finished surface by +5.8 m; before the mean-free uplift it
    was 0.30 -> 0.56 and -163.8 m).  Any such late one-shot shift drops
    sea level onto terrain that was sculpted against a different base
    level and drowns the drainage network erosion built, which PLAN 9-11
    then have to work with.  Holding the datum every iteration keeps the
    coastline erosion sees equal to the one hydro will use, and makes
    ``hydro.requantile_land_fraction`` (kept as the final guarantee) a
    no-op.

    One ``np.partition`` over the interior — 2 ms at ``N_c = 256``, 70 ms
    at ``N_c = 1024``, i.e. 0.2 % of an iteration (1.3 s / 29 s, see
    ``tests/test_perf.py``) — so it runs every iteration rather than on a
    stride, and never on ``erosion.checkpoint_every``, which is
    hash-exempt (``config.RUNTIME_KNOBS``) and must never change results.

    Global (``spherical``) pass only: a refinement window is one basin,
    not a planet, and has no land fraction of its own.
    """
    inter = state.interior
    surf = (state.height[inter] + state.sediment[inter]).ravel()
    M = surf.size
    n_land = min(max(int(round(float(land_fraction) * M)), 0), M)
    if n_land == 0:
        q = float(surf.max()) + 1.0
    elif n_land == M:
        q = float(surf.min())
    else:
        k = M - n_land  # index of the lowest land cell in sorted order
        q = float(np.partition(surf, k)[k])
    if q != 0.0:
        state.height -= q
        if state.route is not None:  # the cached routing surface must stay registered with the terrain
            state.route -= q
    return q


def step(state: ErosionState, params, iteration_key, log=None, **kw) -> dict:
    """One full PLAN 8.2 iteration: particles, thermal erosion, uplift,
    datum hold (:func:`hold_datum`, global pass only), halo exchange.
    Increments ``state.iteration``."""
    ep = _eparams(params)
    t0 = time.time()
    if ep.flood_every > 0 and (state.route is None or state.iteration % ep.flood_every == 0):
        state.refresh_route(ep.route_eps)
    tr = time.time() - t0
    st = run_iteration(state, params, iteration_key, log=log, **kw)
    t1 = time.time()
    thermal_erosion(state, params)
    apply_uplift(state)
    # Glacial carving: the only pass that may leave a closed depression, so
    # the only one that can produce a lake.  Global pass only — a refinement
    # window inherits the coarse result rather than re-carving it.
    if state.spherical and ep.glacial_every > 0 and (state.iteration + 1) % int(ep.glacial_every) == 0 \
            and (state.iteration + 1) >= float(ep.glacial_from) * int(ep.iterations):
        st["glacial"] = glacial.carve(state, params)
    if state.spherical and isinstance(params, WorldParams):
        st["datum_shift"] = hold_datum(state, params.world.land_fraction)
    state.exchange_halos()
    state.iteration += 1
    st["seconds_route"] = tr
    st["seconds_particles"] = t1 - t0 - tr
    st["seconds_total"] = time.time() - t0
    return st


__all__ = ["ErosionState", "run_iteration", "thermal_erosion", "apply_uplift", "hold_datum", "step", "spawn_particles", "height_unit", "max_steps_of"]
