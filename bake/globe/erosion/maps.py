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
* Sediment a submarine fan cannot place (a shelf filled to the waterline
  floor, which never becomes land) is parked in the death cell's
  ``pending`` like a land pit's, less the ``erosion.offshore_writeoff``
  share written off to the deep ocean (``lost_offshore`` in the iteration
  stats), so it walks again next iteration without accumulating forever.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from numba import get_num_threads, njit, parallel_chunksize

from ..config import cell_units, ErosionParams, WorldParams
from ..cubesphere import Grid
from ..field import FaceField
from . import glacial
from . import particle as pk


#: the roots argument of a state without plants (erosion.vegetation)
_NO_ROOTS = np.zeros((1, 1, 1), dtype=np.float32)

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
    lake_flag: np.ndarray | None = None  # (F, NE, NE) uint8, 1 on an overflowing lake cell (`refresh_lakes`); closed lakes live in `base` instead
    disch_track: np.ndarray = field(default=None, repr=False)
    mom_track: np.ndarray = field(default=None, repr=False)
    samp: np.ndarray = field(default=None, repr=False)  # packed float32 samples (F, NE, NE, NS), rebuilt every iteration
    acc: np.ndarray = field(default=None, repr=False)  # float64 net terrain change of the current iteration (cell units), zero between iterations
    pending: np.ndarray = field(default=None, repr=False)  # float64 sediment stockpile per cell (cell units) that found no room this iteration (particle.apply_changes: land pits, and the seafloor less the offshore write-off); re-injected as a loaded particle next iteration; part of the mass balance
    lake_pet: np.ndarray | None = None  # (F, NE, NE) float32: the potential evaporation the lake balance runs on where it is not `evap` (hydro.pet_t0; erosion.run.build_state); a field of the inputs, not of the run, so not checkpointed
    lake_load: np.ndarray = field(default=None, repr=False)  # float64 sediment per cell (cell units) that particles brought into a lake and its shore had no room for (particle.trace_particles, `erosion.lake_fill`): parked on the lake cell entered until `settle_lake_loads` lays it over the lake's floor
    lake_id: np.ndarray = field(default=None, repr=False)  # int32 per cell: the lake of `lake_room` a cell belongs to, -1 where a load cannot be parked
    lake_room: np.ndarray = field(default=None, repr=False)  # float64 per lake: the room it has left for its rivers' load (cell units), counted down by particle.apply_changes
    base: np.ndarray = field(default=None, repr=False)  # float64 local base level per cell (cell units); see `refresh_base`.  0 everywhere until it is refreshed, which is exactly the old "sea = surface < 0" behaviour
    _owner: np.ndarray = field(default=None, repr=False)
    iteration: int = 0
    land_target: float | None = None  # the land fraction `hold_datum` holds (`datum_land_fraction`, set by erosion.run.build_state); None = world.land_fraction
    deposit_on_exit: bool = False  # window mode: deposit the load at the last active cell when leaving
    roots: np.ndarray | None = None  # (F, NE, NE) float32 [0, 1]: the share of the particles' exchange with the bed the plants hold back (erosion.vegetation); None = bare ground
    inflow_volume: float = 0.0  # window mode: the part of the spawn weight (`precip`) that is not rain on the window but water flowing in across its edge (zoom tiles, globe/zoom/bake.py); excluded from the rain per cell the cell-count discharge scales use

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
        if self.lake_load is None:
            self.lake_load = np.zeros((F, NE, NE), dtype=np.float64)
        if self.lake_id is None:
            self.lake_id = np.full((F, NE, NE), -1, dtype=np.int32)
        if self.lake_room is None:
            self.lake_room = np.zeros(1, dtype=np.float64)
        if self.base is None:
            self.base = np.zeros((F, NE, NE), dtype=np.float64)
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
        if self.lake_flag is None:
            self.lake_flag = np.zeros(self.height.shape, dtype=np.uint8)
        pk.pack_samples(self.samp, self.height, self.sediment, self.route if use_route else self.height, use_route, self.discharge, self.momentum, self.evap, self.hardness, self.metric, self.metric_inv, self.base, self.lake_flag)

    def height_m(self) -> np.ndarray:
        return self.height * self.height_unit_m

    def sediment_m(self) -> np.ndarray:
        return self.sediment * self.height_unit_m

    def total_mass(self) -> float:
        """Σ (height + sediment + pending + lake_load) over active interior
        cells (cell units) — the quantity the particle pass conserves exactly."""
        act = self.mask[self.interior] == pk.MASK_ACTIVE
        return float(np.sum((self.height + self.sediment + self.pending + self.lake_load)[self.interior][act]))

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

        self.route = priority_flood_eps(self.surface(), self.mask, self.owner_table(), self.H, self.N, float(eps), self.base)
        if self.spherical:
            FaceField(self.grid, self.route).exchange_halos(linear=True)

    def refresh_base(self, min_fraction: float) -> dict:
        """Recompute the per-cell base level (:data:`particle.S_BASE`).

        **Being under the waterline is not what makes a cell sea; being
        joined to the sea is.**  ``hydro.open_ocean`` has said so since
        docs/plates-rifts-and-water.md, but erosion runs *before* hydro and
        kept reading the sign of the height, so every closed basin below sea
        level was a marine sink for the whole stage: deposit-only seafloor
        walks into it, no erosion of its floor, no hillslope creep on it, and
        whatever a submarine fan could not place inside it deleted as
        ``lost_offshore``.

        The base level is 0 on the open ocean and on ordinary land -- so the
        kernel's ``surface < base`` is bit-identical to the old
        ``surface < 0`` there -- and, inside a closed basin, the surface of
        that basin's **deepest cell**.  A closed basin is then land with an
        internal base level: particles cross it and deposit as they do in any
        pit, its floor may be eroded down to (never below) its own low point,
        and its mass stays in the model.

        The coastline moves as erosion runs, so this is recomputed on the
        ``erosion.sea_mask_every`` stride rather than once; the drift between
        refreshes is measured in docs/erosion-and-the-sea.md.  Global pass
        only: a refinement window is one basin and has no sea of its own.
        """
        from ..hydro.run import below_sea_components

        base = np.zeros_like(self.height)
        info = {"closed_cells": 0, "closed_basins": 0, "sea_cells": 0, "deepest_basin_floor": 0.0}
        if not self.spherical or self.grid is None:
            self.base = base
            self.base_at = self.iteration
            return info
        inter = self.interior
        surf = (self.height[inter] + self.sediment[inter]).astype(np.float64)
        below, labels, keep = below_sea_components(surf.astype(np.float32), self.grid, float(min_fraction))
        sea = None
        if labels is not None:
            sea = below & keep[np.maximum(labels, 0)]
            closed = below & ~sea
            info["sea_cells"] = int(sea.sum())
            info["closed_cells"] = int(closed.sum())
            if closed.any():
                lab = labels[closed]
                floor = np.zeros(keep.size, dtype=np.float64)
                np.minimum.at(floor, lab, surf.reshape(-1)[closed])
                inner = np.zeros(surf.size, dtype=np.float64)
                inner[closed] = floor[lab]
                base[inter] = inner.reshape(surf.shape)
                info["closed_basins"] = int(np.unique(lab).size)
                info["deepest_basin_floor"] = float(floor.min())
        # piecewise constant with sharp jumps at a basin rim: the halo must be
        # an exact copy of the cell it mirrors, not an interpolation of four
        # (`FaceField.exchange_halos` does this for integer fields)
        hm = self.grid.halo
        flat = base.reshape(6 * self.grid.NE * self.grid.NE, 1)
        flat[hm.dst] = flat[hm.nearest]
        self.base = base
        self.base_at = self.iteration
        self.sea = sea
        return info

    def refresh_lakes(self, lake_evap: float, min_depth: float, min_fraction: float, fill: tuple | None = None, land_evap: float = 0.0) -> dict:
        """Solve every depression's water level on the current surface and
        hand it to the kernel (``erosion.lake_balance``).

        Hydro's own sequence -- priority flood, D8 on the filled DEM,
        upstream-first accumulation with the evaporation balance
        (:func:`globe.hydro.balance.balance_lakes`) -- run on the eroding
        surface, so the lakes erosion works with are the lakes hydro will
        find.  Two kinds come out of it.  A depression whose balance
        overflows stands at its spill point: its cells are flagged
        (``lake_flag``, ``S_RFLAG == 2``) and particles cross it on the
        routing surface, dropping their load at the shore and leaving the
        bed alone.  One drawn down below its rim is a closed sea: its level
        goes into ``base`` and the kernel's sea rules do the rest (particles
        entering it settle their load and stop).  Must run after
        :meth:`refresh_base` (which rebuilds ``base`` from scratch) and
        after :meth:`refresh_route`.

        With ``fill`` = ``(grade, ice_evap)`` (``erosion.lake_fill``) the
        overflowing lakes also take the load their rivers parked on them
        since the last refresh, and are flagged 2 while they have room for
        more (:meth:`settle_lake_loads`)."""
        from ..hydro.balance import balance_lakes
        from ..hydro.priority_flood import priority_flood_sphere
        from ..hydro.routing import downstream_table, flow_directions
        from ..hydro.run import open_ocean

        H = self.H
        inter = (slice(None), slice(H, -H), slice(H, -H))
        surf = np.ascontiguousarray(self.surface()[inter], dtype=np.float32)
        sea = getattr(self, "sea", None)
        ocean = np.asarray(sea, bool).reshape(surf.shape) if sea is not None else np.asarray(open_ocean(surf, self.grid, min_fraction), bool).reshape(surf.shape)
        flood = priority_flood_sphere(surf, ocean, self.grid)
        fd = flow_directions(flood.filled, ocean, flood.parent, self.grid)
        down = downstream_table(fd, self.grid.owner, H)
        topo = flood.pop_seq[::-1]
        topo = topo[~ocean.reshape(-1)[topo]]
        pet = self.evap if self.lake_pet is None else self.lake_pet
        water, _acc, bal = balance_lakes(surf, flood.filled, ocean, flood.order, down, topo,
                                         np.ascontiguousarray(self.precip[inter]), np.ascontiguousarray(pet[inter]),
                                         self.grid, float(lake_evap), float(land_evap))
        lake = ((water - surf) > min_depth) & ~ocean
        closed = lake & (water < flood.filled - 1e-4)
        flag = np.zeros(self.height.shape, dtype=np.uint8)
        settled = {}
        if fill is not None:
            settled, takes = self.settle_lake_loads(lake & ~closed, water, surf, flood, *(float(v) for v in fill))
            surf = np.ascontiguousarray(self.surface()[inter], dtype=np.float32)
            lake &= (water - surf) > min_depth          # what the load raised to its plain is ground again
            flag[inter] = (lake & ~closed).astype(np.uint8) + (takes & lake & ~closed)
        else:
            flag[inter] = (lake & ~closed).astype(np.uint8)
        if closed.any():
            b = self.base[inter]
            b[closed & lake] = water[closed & lake]
        hm = self.grid.halo
        for arr in (flag, self.base):
            flat = arr.reshape(-1, 1)
            flat[hm.dst] = flat[hm.nearest]
        self.lake_flag = flag
        return {"lake_cells": int(lake.sum()), "lake_cells_closed": int((closed & lake).sum()),
                "depressions": int(bal.get("depressions", 0)), "overflowing": int(bal.get("overflowing", 0)),
                "closed": int(bal.get("closed", 0)), "dry": int(bal.get("dry", 0)), **{f"load_{k}": v for k, v in settled.items()}}

    def settle_lake_loads(self, lake: np.ndarray, water: np.ndarray, surf: np.ndarray, flood, grade: float, ice_evap: float,
                          rise: float = 0.0, load_share: float = 1.0):
        """Lay the sediment the lakes kept (``lake_load``) over their floors,
        and say what each lake has room for until the next refresh.

        ``lake`` (6, N, N) bool are the overflowing lakes of this refresh,
        ``water`` their level and ``surf`` the ground under them (cell
        units); ``flood`` is the priority flood they came from.  A lake --
        cells joined at one level -- fills towards a plain that rises
        ``grade`` (cell units per cell) from its outlet along the flood's own
        tree, the surface :func:`fill_basins` lays at the hand-off: ground
        filled only to the waterline is dead flat at the spill level, and the
        next few metres of tilt put 300,000 km2 of it back under a metre of
        water.  Its room is what that plain would take; it fills from the far
        side towards the outlet, as a delta does, so what is left of the lake
        is by its outlet at the level it had.

        The plain stands at most ``rise`` (cell units) above the water.  At
        the grade alone the far side of a lake 300 cells long was laid 290 m
        above its water, across the mouth of every river that came in there:
        the lake upstream of it, whose outlet that mouth was, rose with the
        wedge (36 m in earth-v20, over 171,000 km2 of new water).  A metre is
        enough to keep the ground from flooding again, now that standing
        water under ``hydro.marsh_depth`` is marsh.

        ``load_share`` of the deposit is load for the isostasy: sediment laid
        in a lake stands where water stood, and only the difference of their
        densities is new weight on the crust (about 0.55 of the sediment's).

        Then the lakes are numbered for the kernel: ``lake_id`` per cell and
        ``lake_room`` per lake, the room still left, which
        :func:`particle.apply_changes` counts down as particles park their
        loads, so the rivers of a refresh never bring a lake more than it
        holds -- a full lake's load rides on downstream as it always did.  A
        frozen lake (``evap <= ice_evap``, where the glacial pass has ice) is
        under ice, not at the mouth of a river: it takes nothing and keeps the
        basin the ice cut.

        A load stays with its lake when the shore has moved off the cell it
        was parked on (the lake of the last refresh hands it to the one that
        covers most of it now); only where the lake has drained altogether
        does it go to ``pending``.  The deposit is sediment
        and a load like any other (``iso_acc``), so the crust under it
        subsides at the next isostasy pass.  Returns ``(volumes, takes)``:
        the volumes ``parked``, ``laid`` and ``stray`` (cell units), the
        ``room`` left in the ``lakes`` that take load, and ``takes`` (6, N, N)
        bool, their cells."""
        from ..hydro.lakes import label_components

        H, N = self.H, self.N
        inter = (slice(None), slice(H, -H), slice(H, -H))
        load = self.lake_load[inter]
        out = {"parked": float(load.sum()), "laid": 0.0, "stray": 0.0, "room": 0.0, "lakes": 0}
        takes = np.zeros(lake.shape, bool)
        lake_id = np.full(lake.shape, -1, np.int32)
        self.lake_room = np.zeros(1, dtype=np.float64)

        def add(arr, flat_cells, values):
            # interior flat ids -> the extended array (a reshape of the interior view is a copy)
            f, i, j = np.unravel_index(flat_cells, (lake.shape[0], N, N))
            arr[f, i + H, j + H] += values

        lab, n = label_components(np.ascontiguousarray(lake.reshape(-1)), np.ascontiguousarray(water, np.float32).reshape(-1), self.grid.owner, N, H)
        # A load belongs to the lake it was parked in, and that lake is still there when its
        # shore has moved off the cell: each of the last refresh's lakes hands what it holds
        # to the lake of this one that covers most of it
        lflat = load.reshape(-1)
        held = np.flatnonzero(lflat > 0.0)
        have = np.zeros(max(n, 1), dtype=np.float64)
        if held.size:
            was = self.lake_id[inter].reshape(-1)
            n_was = int(was.max()) + 1 if was.size else 0
            heir = np.full(max(n_was, 1), -1, dtype=np.int64)
            both = np.flatnonzero((was >= 0) & (lab >= 0))
            if both.size and n > 0:
                key, cnt = np.unique(was[both].astype(np.int64) * n + lab[both], return_counts=True)
                order = np.lexsort((cnt, key // n))              # per old lake, the largest overlap last
                k_old = (key // n)[order]
                last = np.r_[k_old[1:] != k_old[:-1], True]
                heir[k_old[last]] = (key % n)[order][last]
            to = np.where(lab[held] >= 0, lab[held], np.where(was[held] >= 0, heir[np.maximum(was[held], 0)], -1))
            ok = to >= 0
            np.add.at(have, to[ok], lflat[held[ok]])
            if (~ok).any():
                # its lake has drained altogether: on downstream as a pending particle
                add(self.pending, held[~ok], lflat[held[~ok]])
                out["stray"] = float(lflat[held[~ok]].sum())
        cells = np.flatnonzero(lab >= 0)
        if cells.size:
            s64 = np.asarray(surf, np.float64).reshape(-1)
            w64 = np.asarray(water, np.float64).reshape(-1)
            plain = _lake_plain_kernel(w64, lab, flood.parent, flood.pop_seq, float(grade))
            room = np.maximum(np.minimum(plain[cells], w64[cells] + float(rise)) - s64[cells], 0.0)
            # farthest from the outlet first: a delta builds out from where the rivers come in
            # towards the outlet, which stays open at the level it had.  Filled from the outlet
            # side the new plain is a dam, each refresh a little higher (a 119,000 km2 lake at
            # 4.5 m became 244,000 km2 at 20.9 m).  Far as the crow flies, not in the flood
            # tree's steps: across flat water those count cells along the grid, and what was
            # left of a half-filled lake had straight sides (earth-v22)
            order_f = np.asarray(flood.order).reshape(-1)[cells]
            first = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
            np.minimum.at(first, lab[cells], order_f)
            outlet = np.zeros(n, dtype=np.int64)
            at = order_f == first[lab[cells]]
            outlet[lab[cells][at]] = cells[at]
            centres = self.grid.interior_centers.reshape(-1, 3)
            far = np.linalg.norm(centres[cells] - centres[outlet[lab[cells]]], axis=1)
            order = np.lexsort((-far, lab[cells]))
            cells, room = cells[order], room[order]
            L = lab[cells].astype(np.int64)
            ptr = np.searchsorted(L, np.arange(n))
            csum = np.cumsum(room)
            before = csum - room - (csum[ptr] - room[ptr])[L]    # room in the lake's cells farther from the outlet
            lay = np.clip(have[L] - before, 0.0, room)
            add(self.sediment, cells, lay.astype(self.sediment.dtype))
            if getattr(self, "iso_acc", None) is not None:
                add(self.iso_acc, cells, float(load_share) * lay)
            left = np.bincount(L, weights=room - lay, minlength=n)
            over = have - np.bincount(L, weights=lay, minlength=n)    # rounding only: apply_changes holds a lake to its room
            spill = over > 1e-9
            if spill.any():
                add(self.pending, cells[ptr[spill]], over[spill])
                out["stray"] += float(over[spill].sum())
            warm = np.ones(n, bool)
            cold = np.asarray(self.evap[inter], np.float64).reshape(-1)[cells] <= ice_evap
            warm[np.unique(L[cold])] = False
            open_ = warm & (left > 0.0)
            takes.reshape(-1)[cells[open_[L]]] = True
            ids = np.cumsum(open_) - 1
            lake_id.reshape(-1)[cells[open_[L]]] = ids[L[open_[L]]].astype(np.int32)
            if open_.any():
                self.lake_room = np.ascontiguousarray(left[open_], dtype=np.float64)
            out.update(laid=float(lay.sum()), room=float(left[open_].sum()), lakes=int(open_.sum()))
        load[...] = 0.0
        full = np.full(self.height.shape, -1, np.int32)
        full[inter] = lake_id
        self.lake_id = full
        return out, takes

    def refresh_lakes_window(self, min_depth: float, lake_evap: float = 0.0) -> dict:
        """:meth:`refresh_lakes` for a window (``erosion.window_lakes``):
        every depression the routing flood fills more than ``min_depth``
        (cell units) deep is an overflowing lake -- flagged, so particles
        cross it on the routing surface without touching the bed, drop their
        load at its shore and are not killed for climbing inside it, and its
        bed is never eroded below the water.  The level is the routing
        surface's (spill point plus ``route_eps`` per cell), so lakes and
        routes agree by construction; there is no evaporation balance, a
        window having no climate of its own to draw a basin down with.

        Without it a window's particle pass dams its own channels (92-98 % of
        new depressions at 305 m cells, docs/zoom-windows.md) and a particle
        reaching the dam climbs, dies and drops its load behind it, so the dam
        is never incised.  Must run after :meth:`refresh_route`.

        With ``lake_evap > 0`` (``erosion.window_lake_evap``) the level is the
        evaporation balance instead of the spill point, as :meth:`refresh_lakes`
        solves it on the planet: the window's own rain is accumulated down the
        flood tree, and a depression settles where the inflow arriving at it
        equals ``lake_evap`` times the evaporative demand of the water it
        covers, so a divot with no catchment holds no lake.  Every depression
        keeps the routing surface it had -- a particle still crosses it and
        leaves, which is what cuts the outlet -- but the water, and with it
        the bed the kernel will not touch, reaches only the balanced level;
        the ground between that level and the rim is dry land again.

        (Making a balanced lake a sink in ``base``, as the planet does, is
        wrong here: at 1.2 km most depressions are noise the erosion should
        drain, and trapping the particles in them left more, not fewer --
        pits 2.6 % -> 4.3 % of a tile's land, lakes 2.8 % -> 5.4 %.)"""
        flag = np.zeros(self.height.shape, dtype=np.uint8)
        if self.route is None:
            self.lake_flag = flag
            return {"lake_cells": 0}
        surf = self.surface()
        lake = ((self.route - surf) > float(min_depth)) & (self.mask == pk.MASK_ACTIVE)
        H = self.H
        inter = (slice(None), slice(H, -H), slice(H, -H))
        st = {}
        if float(lake_evap) > 0.0:
            lake, st = self._balance_window_lakes(surf, float(min_depth), float(lake_evap))
        flag[inter] = lake[inter]
        self.lake_flag = flag
        return {"lake_cells": int(lake[inter].sum()), **st}

    def _balance_window_lakes(self, surf: np.ndarray, min_depth: float, lake_evap: float):
        """The evaporation balance of every depression of a window (F = 1).
        Returns ``(overflowing lake mask, counts)`` and writes the level of a
        closed one into ``base``."""
        from scipy import ndimage

        from ..hydro.priority_flood import priority_flood_flat
        from ..zoom.bake import _accumulate

        s = np.ascontiguousarray(surf[0], np.float32)
        active = self.mask[0] == pk.MASK_ACTIVE
        drain = ~active
        drain[0, :] = drain[-1, :] = drain[:, 0] = drain[:, -1] = True
        fr = priority_flood_flat(s, drain, None)
        filled = fr.filled.reshape(s.shape)
        acc = _accumulate(fr.pop_seq, fr.parent, np.ascontiguousarray(self.precip[0], np.float64).ravel()).reshape(s.shape)
        dep = (filled > s + 1e-6) & active
        lab, n = ndimage.label(dep, structure=np.ones((3, 3), bool))
        out = np.zeros(s.shape, bool)
        if n == 0:
            return out[None], {"depressions": 0, "overflowing": 0, "closed": 0, "dry": 0}
        cells = np.flatnonzero(lab.ravel() > 0)
        order = np.lexsort((s.ravel()[cells], lab.ravel()[cells]))
        cells = cells[order]
        lab_s = lab.ravel()[cells].astype(np.int64)
        ptr = np.searchsorted(lab_s, np.arange(1, n + 2)).astype(np.int64)
        demand = np.cumsum(lake_evap * np.maximum(self.evap[0].ravel()[cells].astype(np.float64), 0.0))
        z = s.ravel()[cells].astype(np.float64)
        spill = ndimage.maximum(filled, lab, np.arange(1, n + 1))
        inflow = ndimage.maximum(acc, lab, np.arange(1, n + 1))
        level = np.zeros(n)
        counts = {"depressions": n, "overflowing": 0, "closed": 0, "dry": 0}
        for k in range(n):
            a, b = ptr[k], ptr[k + 1]
            if b <= a:
                continue
            need = float(np.atleast_1d(inflow)[k])
            sp = float(np.atleast_1d(spill)[k])
            before = demand[a - 1] if a > 0 else 0.0
            loss = demand[a:b] - before                      # evaporation with the water at each cell's own level
            if need <= 0.0:
                level[k] = z[a] - 1.0
                counts["dry"] += 1
            elif loss[-1] <= need:                           # cannot evaporate what arrives: still spills
                level[k] = sp
                counts["overflowing"] += 1
                continue
            else:
                i = int(np.searchsorted(loss, need))
                level[k] = float(z[a + min(i, b - a - 1)])
                counts["closed"] += 1
        lv = np.zeros(n + 1)
        lv[1:] = level
        water = np.where(lab > 0, lv[lab], s)
        out = (water - s > min_depth) & active
        # the kernel reads the water level off the routing surface: lower it to
        # the balanced level on the water itself, and leave the dry bed of the
        # depression at the spill level so a particle can still climb out
        if self.route is not None:
            r = self.route[0]
            r[out] = np.minimum(r[out], water[out])
        return out[None], counts

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


def _ocean_min_fraction(params) -> float:
    """``hydro.ocean_min_fraction``, so erosion and hydro draw the same
    coastline.  An ``ErosionParams`` alone (the window/test path) has no
    hydro group; it falls back to the shipped default."""
    return float(params.hydro.ocean_min_fraction) if isinstance(params, WorldParams) else 0.02


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
    divides never spawn — precip > 0) and their weights (precip).

    Land is ``surface >= base`` (:meth:`ErosionState.refresh_base`), so the
    dry floor of a closed basin below sea level catches rain like any other
    ground; only the open ocean and standing water do not."""
    F, NE, H, N = state.F, state.NE, state.H, state.N
    inter = np.zeros((F, NE, NE), dtype=bool)
    inter[state.interior] = True
    ok = inter & (state.mask == pk.MASK_ACTIVE) & (state.height + state.sediment >= state.base) & (state.precip > 0)
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
    ``cl_vol == 0``, its final deposits ``cl_vol == -1``, and so its death
    cell is the first entry with ``cl_vol == -1`` (a bank cut beside the path
    is ``-2`` with ``erosion.lateral_rate``, a load a lake kept ``-3`` with
    ``erosion.lake_fill``; without them every negative entry is a final
    deposit).  Production passes nothing and the branch costs a single
    ``is None``."""
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
    # discharge scales in cells of upstream area (McDonald's are in cells:
    # his discharge is the volume of the 512 particles a cycle sends over a
    # 512^2 map, so erf(0.4 q) saturates at ~1280 cells and the momentum push
    # is at half strength at ~512).  Ours is in rain volume, one cell's worth
    # of rain per iteration = the rain spawn weight over the cells it falls on
    disc_sat = float(ep.disc_saturation)
    mom_scale = 1.0
    if float(ep.disc_saturation_cells) > 0.0 or float(ep.momentum_saturation_cells) > 0.0:
        total = float(volume0) * float(n_particles)
        n_rain = max(int(spawn_weights(state)[0].size), 1)
        rain_per_cell = max(total - float(state.inflow_volume), 0.0) / n_rain
        rain_per_cell = rain_per_cell if rain_per_cell > 0.0 else total / n_rain
        if float(ep.disc_saturation_cells) > 0.0:
            disc_sat = float(ep.disc_saturation_cells) * rain_per_cell
        if float(ep.momentum_saturation_cells) > 0.0:
            mom_scale = float(volume0) / (float(ep.momentum_saturation_cells) * rain_per_cell)
    use_ecap = float(ep.slope_limit_erode) > 0.0
    if use_ecap:
        ecap = np.empty_like(state.height)
        pk.slope_erode_cap(state.height, state.sediment, state.mask, float(ep.slope_limit_erode), float(ep.slope_limit_exit), ecap)
    else:
        ecap = np.zeros(1, dtype=np.float64)
    iter_deposit = cell_units(ep, "iter_deposit", state.height_unit_m)
    if float(ep.slope_limit_deposit) > 0.0:
        # soillib: deposition <= 0.25 L critSlopeSediment per step
        iter_deposit = min(iter_deposit, float(ep.slope_limit_deposit) * math.sqrt(2.0) * 0.3)
    max_steps = max_steps_of(ep, state.N)
    cap = int(pk.change_list_cap(max_steps, float(getattr(ep, "lateral_rate", 0.0)) > 0.0))
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
                mom_scale,
                float(ep.k_disc),
                disc_sat,
                float(ep.disc_exponent),
                float(ep.slope_gain),
                float(ep.slope_saturation),
                float(ep.erodibility),
                cell_units(ep, "cover_depth", state.height_unit_m),
                float(ep.min_volume_frac) * float(volume0),
                int(max_steps),
                cell_units(ep, "max_erode", state.height_unit_m),
                float(chunk / (volume0 * P)),
                int(ep.pit_steps),
                bool(state.deposit_on_exit),
                float(ep.ocean_deposition_rate),
                int(ep.ocean_steps),
                cell_units(ep, "fan_room", state.height_unit_m),
                cell_units(ep, "dep_floor_m", state.height_unit_m),
                float(ep.lake_trap),
                1.0 if lake_fill_args(state, params) is not None else 0.0,
                float(getattr(ep, "lateral_rate", 0.0)),
                state.roots if state.roots is not None else _NO_ROOTS,
                state.roots is not None,
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
            iter_deposit,
            float(ep.fan_slope), state.route is not None,
            cell_units(ep, "dep_floor_m", state.height_unit_m),
            float(ep.offshore_writeoff),
            ecap, use_ecap, state.lake_load, state.lake_id, state.lake_room,
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
    stats["lost_offshore"] = lost_offshore  # the offshore_writeoff share of the load seafloor walks could not place: left the modelled surface (deep ocean); the rest is in pending_total
    stats["seconds_trace"] = t_trace
    stats["seconds_apply"] = t_apply
    stats["chunk"] = int(chunk)
    stats["disc_saturation"] = disc_sat
    stats["mom_scale"] = mom_scale
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
    dep_floor = cell_units(ep, "dep_floor_m", state.height_unit_m)
    _mass_wasting_pass(state, talus, float(ep.thermal_rate), cell_units(ep, "thermal_max", state.height_unit_m), dep_floor=dep_floor)
    if ep.creep_rate > 0.0:
        # hillslope creep: the same conservative pass with talus 0 (every
        # lower neighbour receives creep_rate/2 of the height difference).
        # Submerged cells are inert for it (frozen in a temporary mask):
        # creep is a hillslope process, it must not diffuse the coast into
        # the sea (the talus pass still lets sea cliffs collapse).
        cmask = np.where(state.height + state.sediment < state.base, np.uint8(pk.MASK_FROZEN), state.mask).astype(np.uint8)
        _mass_wasting_pass(state, np.zeros_like(talus), float(ep.creep_rate), cell_units(ep, "thermal_max", state.height_unit_m), cmask, dep_floor=dep_floor)


def _mass_wasting_pass(state: ErosionState, talus: np.ndarray, rate: float, cap: float, mask: np.ndarray | None = None, dep_floor: float = pk.DEP_FLOOR) -> None:
    mask = state.mask if mask is None else mask
    out_total = np.zeros_like(state.height)
    scale = np.zeros_like(state.height)
    pk.thermal_pass_a(state.height, state.sediment, state.metric, talus, mask, rate, cap, out_total, scale)
    rscale = np.ones_like(state.height)
    pk.thermal_pass_r(state.height, state.sediment, state.metric, talus, mask, rate, scale, rscale, float(dep_floor))
    new_h = state.height.copy()
    new_s = state.sediment.copy()
    pk.thermal_pass_b(state.height, state.sediment, state.metric, talus, mask, rate, out_total, scale, rscale, new_h, new_s)
    state.height[...] = new_h
    state.sediment[...] = new_s


def apply_uplift(state: ErosionState, cap: float | None = None) -> None:
    """Add the per-iteration uplift, **mean-free** over the active interior,
    each cell's share bounded by ``cap`` (cell units) when one is given.

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

    The field itself has no ceiling: tectonics writes the height a segment
    gained over its last ``uplift_window`` steps divided by the iteration
    count, so a belt still rising at the last tectonic step keeps rising at
    that rate for the whole stage, and at a ridge crest -- no discharge, no
    talus failure at Earth's cell size, the glacial carve tapering to zero
    -- nothing opposes it (earth-v5: 4.51 m/it x 800 iterations = 3.6 km on
    top of a 6.7 km bedrock; docs/uplift-ceiling.md).  ``cap`` is that
    ceiling (``erosion.uplift_max_m`` in cell units): the field is clipped
    *before* the mean is taken, so the applied change is still exactly
    mass-free.  It is a guard on the tectonics field and belongs to the
    global pass; :func:`step` passes it only for a ``spherical`` state (the
    window tests hand in their own synthetic field, refine passes 0).  With
    ``cap=None`` this is byte-identical to the uncapped rule.

    What the stage starts from is :func:`start_replay`'s business
    (``erosion.uplift_mode``): this function adds the same field in both
    modes.
    """
    act = state.mask == pk.MASK_ACTIVE
    u, mean = _applied_uplift(state, cap)
    state.height[act] += u[act] - mean


def _applied_uplift(state: ErosionState, cap: float | None) -> tuple[np.ndarray, float]:
    """The (capped) field and the datum :func:`apply_uplift` removes from it:
    one iteration adds ``u - mean`` to every active cell.  Shared with
    :func:`uplift_replay`, so the start state is lowered by exactly what the
    run will add back."""
    u = state.uplift if cap is None else np.minimum(state.uplift, float(cap))
    mean = 0.0
    if state.spherical:
        iact = state.mask[state.interior] == pk.MASK_ACTIVE
        mean = float(u[state.interior][iact].mean()) if iact.any() else 0.0
    return u, mean


def uplift_cap(state: ErosionState, ep) -> float | None:
    """``erosion.uplift_max_m`` in this state's cell units, or ``None`` when
    it does not apply: off (``<= 0``) or a window state (the cap guards the
    tectonics field the global pass applies; see :func:`apply_uplift`)."""
    if not state.spherical or float(getattr(ep, "uplift_max_m", 0.0)) <= 0.0:
        return None
    return cell_units(ep, "uplift_max_m", state.height_unit_m)


def uplift_replay(state: ErosionState, ep) -> np.ndarray | None:
    """The total uplift the stage will apply to each cell (cell units, an
    extended array, zero off the active cells), or ``None`` when
    ``erosion.uplift_mode`` is ``'stack'`` or the state is a window.

    ``erosion.iterations`` times the per-iteration change :func:`apply_uplift`
    makes -- the capped field less its active-interior mean -- so it is
    mass-free like the rule it inverts, and a run that adds the field
    ``iterations`` times over a start lowered by this lands on the start's
    own height plus this, cell for cell."""
    if not state.spherical or str(getattr(ep, "uplift_mode", "stack")) != "replay":
        return None
    u, mean = _applied_uplift(state, uplift_cap(state, ep))
    act = state.mask == pk.MASK_ACTIVE
    total = np.zeros_like(state.height)
    total[act] = float(int(ep.iterations)) * (u[act] - mean)
    return total


def start_replay(state: ErosionState, params) -> dict | None:
    """``erosion.uplift_mode = 'replay'``: turn a state built from
    ``bedrock`` into the crust as it stood at the uplift reference step.

    Tectonics writes ``bedrock`` at its last step and ``uplift`` as what a
    segment gained over the last ``uplift_window`` steps, spread over
    ``erosion.iterations``.  Starting from ``bedrock`` and applying
    ``uplift`` counts that window twice (docs/uplift-ceiling.md: earth-v5's
    top cell is 5798 m of bedrock + 3162 m of applied uplift).  This lowers
    the start by :func:`uplift_replay` -- exactly what :func:`apply_uplift`
    will add back over the run -- refreshes the halos, and holds the datum
    on the result (:func:`hold_datum`, a rigid shift), so the coastline
    iteration 0's sea mask and routing flood see is the reference step's at
    the held land fraction (:func:`held_land_fraction`), which is what the loop holds from iteration 1
    on.  With no surface process the stage then ends at ``bedrock`` less a
    single global constant (the accumulated datum hold).

    Call it once, on a freshly built state: a checkpoint already carries the
    replayed height, and :func:`globe.erosion.run.load_checkpoint`
    overwrites it.  Returns ``None`` (and changes nothing) in ``'stack'``
    mode or on a window; otherwise ``replay_max`` / ``replay_min`` (cell
    units, the most a cell was lowered / raised), ``land_fraction`` (of the
    lowered surface, before the hold) and ``datum_shift`` (cell units, what
    :func:`hold_datum` subtracted; 0 when ``params`` has no world group)."""
    ep = _eparams(params)
    total = uplift_replay(state, ep)
    if total is None:
        return None
    state.height -= total
    state.exchange_halos()
    inter = state.interior
    info = {"replay_max": float(total[inter].max()), "replay_min": float(total[inter].min()),
            "land_fraction": float(np.mean((state.height + state.sediment)[inter] >= 0)), "datum_shift": 0.0}
    if isinstance(params, WorldParams):
        info["datum_shift"] = hold_datum(state, held_land_fraction(state, params))
    return info


@njit(cache=True)
def _graded_fill_kernel(s, parent, pop_seq, grade):
    """The flood's surface with a slope on it: every cell stands at least
    ``grade`` above the cell that flooded it (``parent``), walked in pop order
    so a parent is always done before its children."""
    g = s.copy()
    for k in range(pop_seq.size):
        c = pop_seq[k]
        q = parent[c]
        if q >= 0:
            v = g[q] + grade
            if v > g[c]:
                g[c] = v
    return g


@njit(cache=True)
def _lake_plain_kernel(water, lab, parent, pop_seq, grade):
    """The plain a lake fills towards: its water level at the outlet, and
    ``grade`` higher with every cell the flood took to reach a cell from
    there (``parent`` inside the same lake ``lab``; pop order, so a parent is
    done before its children)."""
    g = water.copy()
    for k in range(pop_seq.size):
        c = pop_seq[k]
        if lab[c] >= 0:
            q = parent[c]
            if q >= 0 and lab[q] == lab[c]:
                g[c] = g[q] + grade
    return g


def lake_fill_args(state: ErosionState, params):
    """``refresh_lakes``' ``fill`` for this state: ``(grade in cell units per
    cell, ice_evap, the plain's rise above the water in cell units, the share
    of the fill that is load)`` with ``erosion.lake_fill`` on the planet, else
    None."""
    ep = _eparams(params)
    if not (state.spherical and bool(getattr(ep, "lake_fill", False)) and bool(getattr(ep, "lake_balance", False))):
        return None
    cell_km = float(state.grid.cell_size_m) / 1000.0
    return (float(getattr(ep, "basin_fill_grade", 0.0)) * cell_km / state.height_unit_m, float(getattr(ep, "ice_evap", 0.0)),
            float(getattr(ep, "lake_fill_rise_m", 1.0)) / state.height_unit_m, float(getattr(ep, "lake_fill_load", 0.55)))


def settle_lakes(state: ErosionState, params) -> dict | None:
    """Lay what the lakes hold after the last iteration: a run ends between
    two lake refreshes, and the load its last iterations parked would stay
    out of the surface the next stage reads."""
    fill = lake_fill_args(state, params)
    if fill is None or not isinstance(params, WorldParams):
        return None
    ep = _eparams(params)
    if int(getattr(ep, "sea_mask_every", 0)) > 0:
        state.refresh_base(_ocean_min_fraction(params))
    state.refresh_route(cell_units(ep, "route_eps", state.height_unit_m))
    st = state.refresh_lakes(float(params.hydro.lake_evap), float(params.hydro.lake_min_depth) / float(state.height_unit_m),
                             _ocean_min_fraction(params), fill, float(getattr(params.hydro, "land_evap", 0.0)))
    state.exchange_halos()
    return st


def fill_basins(state: ErosionState, params) -> dict | None:
    """``erosion.basin_fill_grade`` > 0: lay sediment into the closed basins the
    tectonic bedrock arrives with, up to a plain that rises ``basin_fill_grade``
    (metres per km) from each basin's outlet.

    Tectonics runs its whole history with no surface process: nothing fills a
    basin and nothing grades it to an outlet, so the bedrock reaches erosion
    with a fifth of its land inside closed depressions (earth-v18: 31 Mkm2 --
    a 5.3 Mkm2 interior 117 m below its rim, a stretched-crust hole 1 km deep,
    plateau basins of 0.6-0.7 Mkm2), and the stage's few hundred iterations
    cannot fill them: they came out as lakes of up to 1.2 Mkm2, eight above
    100,000 km2 where Earth has one.  On Earth those basins are what the
    continental interiors are made of -- the West Siberian plain, the Congo and
    Amazon basins, the Great Plains: kilometres of sediment under a plain that
    rivers cross.  This is that sediment, laid at the hand-off.

    A basin still subsiding is not lost by it: in ``uplift_mode`` 'replay' the
    start has the last tectonic window's motion taken out, the fill is laid on
    that surface, and the subsidence replayed during the run re-opens the
    basin under it, where a lake belongs (Baikal, Tanganyika).

    Call once on a fresh state, after :func:`start_replay`.  The sediment is
    new mass (the highlands that shed it over tectonic time are not lowered);
    its volume is returned and logged.  Returns ``None`` when off or on a window."""
    ep = _eparams(params)
    grade = float(getattr(ep, "basin_fill_grade", 0.0))
    if grade <= 0.0 or not state.spherical or not isinstance(params, WorldParams):
        return None
    from ..hydro.priority_flood import priority_flood_sphere
    from ..hydro.run import open_ocean

    H = state.H
    inter = (slice(None), slice(H, -H), slice(H, -H))
    surf = np.ascontiguousarray(state.surface()[inter], dtype=np.float32)
    ocean = np.asarray(open_ocean(surf, state.grid, float(params.hydro.ocean_min_fraction)), bool).reshape(surf.shape)
    flood = priority_flood_sphere(surf, ocean, state.grid)
    cell_km = float(state.grid.cell_size_m) / 1000.0
    step = grade * cell_km / state.height_unit_m                     # cell units per cell
    g = _graded_fill_kernel(surf.reshape(-1).astype(np.float64), flood.parent, flood.pop_seq, step).reshape(surf.shape)
    # only where the flood found a basin: a plain already draining is left as it is
    basin = (flood.filled - surf > float(ep.basin_fill_min_m) / state.height_unit_m) & ~ocean
    add = np.where(basin, np.maximum(g - surf, 0.0), 0.0)
    state.sediment[inter] += add.astype(state.sediment.dtype)
    state.exchange_halos()
    area = state.grid.interior_cell_area
    return {"basin_cells": int(basin.sum()), "basin_area_km2": float(area[basin].sum() / 1e6),
            "fill_km3": float((add * state.height_unit_m * area).sum() / 1e9),
            "fill_mean_m": float(add[basin].mean() * state.height_unit_m) if basin.any() else 0.0,
            "fill_max_m": float(add.max() * state.height_unit_m)}


_KAPPA_CACHE: dict = {}


def _smoothing_kappa(grid: Grid) -> float:
    """Largest *monotone* explicit step for the Laplace-Beltrami smoother.

    Explicit Euler on ``u' = L u`` is stable only while ``kappa * lam_max <=
    2`` and free of ringing only while ``kappa * lam_max <= 1``, where
    ``lam_max`` is the largest eigenvalue of ``-L`` in cell units.  On a flat
    5-point grid that is 8, so the textbook 0.25 is safe; on the cube-sphere
    the metric inflates the stencil and it is not.  Measured at the earth
    preset: kappa = 0.2 grows unit white noise to 4e12 in 95 steps, and it is
    what blew the first isostasy run up to a 1.7e16 m peak by iteration 50.
    So ``lam_max`` is estimated once per grid by power iteration (the
    Laplacian of noise converges on its highest mode in a few dozen
    applications) and the step is 0.9 of the monotone bound -- itself half
    the stability bound, so an estimate that is 2x low is still stable.
    """
    key = (grid.N, grid.H, float(grid.cell_size_m))
    if key not in _KAPPA_CACHE:
        rng = np.random.default_rng(0)
        f = FaceField.from_interior(grid, rng.standard_normal((6, grid.N, grid.N)), exchange=True)
        f.data = f.data.astype(np.float64)
        H, c2, lam = grid.H, float(grid.cell_size_m) ** 2, 8.0
        for _ in range(40):
            f.exchange_halos(linear=True)
            Lv = f.laplacian().data.astype(np.float64) * c2
            num = float(np.sqrt((Lv[:, H:-H, H:-H] ** 2).sum()))
            den = float(np.sqrt((f.interior ** 2).sum()))
            if num <= 0.0 or den <= 0.0:
                break
            lam = num / den
            f.data = Lv / num
        _KAPPA_CACHE[key] = 0.9 / max(lam, 1e-6)
    return _KAPPA_CACHE[key]


def apply_isostasy(state: ErosionState, params) -> dict:
    """Flexural isostatic response to the mass surface processes moved.

    ``state.iso_acc`` holds the net change of ``height + sediment`` that
    particles, mass wasting and the glacial pass made since the last call.
    A column that lost rock rebounds by ``isostasy`` of it and one that
    gained sediment subsides by the same fraction: Airy isostasy, with the
    densities normalised to the mantle exactly as the tectonics stage uses
    them (``h = t(1 - rho)``).  Without this the kernel lowers the surface
    one metre for every metre of rock it removes, where a real continent
    loses only ``1 - rho`` ~ 0.2 of it, and a continental interior planes
    straight down to base level (docs/missing-relief.md).

    The response is *regional*: the load is spread by a Gaussian of sigma
    ``flexure_km`` before it is applied, as a plate with flexural rigidity
    spreads it.  A per-cell rebound would refill every valley as fast as it
    was cut; a flexural one lifts the interfluves and peaks around it.  The
    smoothing is explicit diffusion with the metric-correct, seam-aware
    Laplace-Beltrami operator, so it has no face seams.  Mean-free over the
    active interior, like :func:`apply_uplift`: its mean is a datum shift,
    which :func:`hold_datum` owns.  Global pass only.
    """
    ep = _eparams(params)
    d = state.iso_acc
    k = float(ep.isostasy)
    load = (-k * d).astype(np.float64)          # erosion (d < 0) -> rebound up
    cell = float(state.height_unit_m)
    sig = float(ep.flexure_km) * 1000.0 / cell
    sig = min(sig, state.N / 8.0)               # a planet smaller than one plate
    kappa = _smoothing_kappa(state.grid)
    n = int(np.ceil(sig * sig / (2.0 * kappa))) if sig >= 0.5 else 0
    f = FaceField(state.grid, load)
    for _ in range(n):
        f.exchange_halos(linear=True)       # monotone halos for a smoother
        f.data += kappa * f.laplacian().data.astype(np.float64) * (cell * cell)
    f.exchange_halos(linear=True)
    act = state.mask == pk.MASK_ACTIVE
    iact = state.mask[state.interior] == pk.MASK_ACTIVE
    mean = float(f.data[state.interior][iact].mean()) if iact.any() else 0.0
    state.height[act] += (f.data[act] - mean).astype(state.height.dtype)
    moved = float(np.abs(d[state.interior]).sum())
    state.iso_acc[...] = 0.0
    return {"sigma_cells": sig, "diffusion_steps": n, "kappa": kappa, "load_abs_cells": moved,
            "rebound_max": float((f.data[state.interior] - mean).max()),
            "rebound_min": float((f.data[state.interior] - mean).min())}


def datum_land_fraction(params: WorldParams, bedrock) -> float:
    """The land fraction the planetary datum is held at, erosion and hydro.

    ``world.land_fraction`` while ``tectonics.shelf_fraction`` is 0.  In
    shelf mode tectonics places sea level against the continental crust and
    the land area is an output (:class:`globe.config.WorldGroup`,
    docs/crust-types.md), so the hold keeps the area tectonics produced: the
    share of cells whose ``bedrock`` is at or above 0 (any units -- only the
    sign is read; unused, and may be None, with shelf mode off).  A share of
    *cells*, the order statistic the hold works on, not of area:
    `scripts/hypsometry.py`'s "land % of globe" is area-weighted and reads
    ~0.8 points lower on the Earth grid.  Holding ``world.land_fraction`` there instead re-placed
    sea level on every seed: it lifted seed 0's 24 %-land Earth by ~550 m
    and lowered seeds 1-3 (34-36 %), and against a sea floor at its
    half-space depth it lifted seed 0 by ~1 km (docs/ocean-depth.md)."""
    if float(params.tectonics.shelf_fraction) > 0.0:
        return float(np.mean(np.asarray(bedrock) >= 0))
    return float(params.world.land_fraction)


def held_land_fraction(state: ErosionState, params: WorldParams) -> float:
    """``state.land_target`` when the driver set it, else ``world.land_fraction``
    -- which is the wrong rule in shelf mode, so a shelf-mode state that was
    not built by :func:`globe.erosion.run.build_state` is refused rather
    than held at a sea level tectonics did not put there."""
    if state.land_target is not None:
        return float(state.land_target)
    if float(params.tectonics.shelf_fraction) > 0.0:
        raise ValueError("shelf mode holds the bedrock's land fraction: set state.land_target "
                         "(erosion.maps.datum_land_fraction) or build the state with erosion.run.build_state")
    return float(params.world.land_fraction)


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
    then have to work with.  The cached routing surface and the per-cell
    base level are shifted with it: both are registered to the terrain.  Holding the datum every iteration keeps the
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
        # ... and so must the base level, for the same reason.  Sea level is
        # 0 in the new datum as it was in the old, so the 0 entries stay put;
        # a closed basin's base level is the elevation of its floor, and the
        # floor just moved with everything else.  Left un-shifted it drifts
        # against the terrain by the accumulated hold between refreshes
        # (-0.63 m an iteration on earth-full, so ~6 m at a stride of 10).
        if state.base is not None:
            np.subtract(state.base, q, out=state.base, where=state.base < 0.0)
    return q


def step(state: ErosionState, params, iteration_key, log=None, **kw) -> dict:
    """One full PLAN 8.2 iteration: particles, thermal erosion, uplift,
    datum hold (:func:`hold_datum`, global pass only), halo exchange.
    Increments ``state.iteration``."""
    ep = _eparams(params)
    t0 = time.time()
    sea_every = int(getattr(ep, "sea_mask_every", 0))
    if state.spherical and sea_every > 0 and (getattr(state, "base_at", None) is None or state.iteration % sea_every == 0):
        # before the route: the flood seeds on the sea, and the sea is what
        # `refresh_base` decides (see `_seed_base` in route.py)
        st_base = state.refresh_base(_ocean_min_fraction(params))
    else:
        st_base = None
    if ep.flood_every > 0 and (state.route is None or state.iteration % ep.flood_every == 0):
        state.refresh_route(cell_units(ep, "route_eps", state.height_unit_m))
    st_lake = None
    if state.spherical and bool(getattr(ep, "lake_balance", False)) and ep.flood_every > 0 and \
            (state.lake_flag is None or state.iteration % ep.flood_every == 0):
        # after base (rebuilt from scratch) and route (seeded on base)
        st_lake = state.refresh_lakes(float(params.hydro.lake_evap),
                                      float(params.hydro.lake_min_depth) / float(state.height_unit_m),
                                      _ocean_min_fraction(params), lake_fill_args(state, params),
                                      float(getattr(params.hydro, "land_evap", 0.0)))
    elif not state.spherical and bool(getattr(ep, "window_lakes", False)) and ep.flood_every > 0 and \
            (state.lake_flag is None or state.iteration % ep.flood_every == 0):
        min_depth = float(params.hydro.lake_min_depth) if isinstance(params, WorldParams) else 0.5
        st_lake = state.refresh_lakes_window(min_depth / float(state.height_unit_m), float(getattr(ep, "window_lake_evap", 0.0)))
    tr = time.time() - t0
    iso = state.spherical and float(getattr(ep, "isostasy", 0.0)) > 0.0
    if iso:
        if getattr(state, "iso_acc", None) is None:
            state.iso_acc = np.zeros_like(state.height)
        s0 = state.height + state.sediment
    st = run_iteration(state, params, iteration_key, log=log, **kw)
    t1 = time.time()
    thermal_erosion(state, params)
    if iso:
        # only what surface processes moved: uplift and the datum shift are
        # not loads, and must not be compensated
        state.iso_acc += (state.height + state.sediment) - s0
    apply_uplift(state, uplift_cap(state, ep))
    # Glacial carving: the only pass that may leave a closed depression, so
    # the only one that can produce a lake.  Global pass only — a refinement
    # window inherits the coarse result rather than re-carving it.
    if state.spherical and ep.glacial_every > 0 and (state.iteration + 1) % int(ep.glacial_every) == 0 \
            and (state.iteration + 1) >= float(ep.glacial_from) * int(ep.iterations):
        if iso:
            s1 = state.height + state.sediment
        st["glacial"] = glacial.carve(state, params)
        if iso:
            state.iso_acc += (state.height + state.sediment) - s1
    if iso and (state.iteration + 1) % max(int(getattr(ep, "isostasy_every", 10)), 1) == 0:
        st["isostasy"] = apply_isostasy(state, params)
    if state.spherical and isinstance(params, WorldParams):
        st["datum_shift"] = hold_datum(state, held_land_fraction(state, params))
    state.exchange_halos()
    state.iteration += 1
    if st_base is not None:
        st["sea"] = st_base
    if st_lake is not None:
        st["lakes"] = st_lake
    st["seconds_route"] = tr
    st["seconds_particles"] = t1 - t0 - tr
    st["seconds_total"] = time.time() - t0
    return st


__all__ = ["ErosionState", "run_iteration", "thermal_erosion", "apply_uplift", "uplift_replay", "start_replay", "apply_isostasy", "hold_datum", "step", "settle_lakes", "lake_fill_args", "spawn_particles", "height_unit", "max_steps_of"]
