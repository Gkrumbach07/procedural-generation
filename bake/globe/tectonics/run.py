"""Tectonics stage driver (PLAN.md section 6): clustered convection plate
tectonics on the unit sphere.

State lives on ``params.tect_grid()`` (``N_tect`` cells per face edge) and
in a :class:`~globe.tectonics.segments.Segments` point cloud; the outputs
are resampled to the coarse grid at the end.

Per step (:meth:`TectonicSim.step`)::

    1. rotate every plate rigidly about its Euler pole (omega, rad/step)
    2. KD-tree pair query -> subduction of the denser segment of every
       approaching pair from different plates (mass/thickness transfer,
       Gaussian sharing with the survivor's same-plate neighbours, heat
       blob at the subduction point, dead segments removed); then the
       segment-height relaxation (cascade) over the whole cloud
    3. label map (nearest segment per tect cell, voxel hash) -> Voronoi
       areas (rolling blend); cells farther than gap_radius from every
       segment are divergent gaps: new thin, young crust is spawned there
       (greedy Poisson acceptance) and the heat field is cooled under it
    4. crystallisation (growth / dissolution from the local heat)
    5. heat diffusion (explicit Laplace-Beltrami, sub-stepped)
    6. forces: plate torque from the heat gradient at the segments,
       omega update with damping and a speed cap

Units
-----
* Lengths on the sphere are chord/arc lengths in radians; ``spacing`` is
  the mean segment spacing ``sqrt(4π / M0)``.  Time is in tectonic steps.
* Bedrock height per segment is ``thickness * (1 - density)`` ("bedrock
  units") plus the thermal buoyancy of young crust
  ``ridge_height * exp(-age / ridge_age)`` (mid-ocean ridges, subsidence
  with age; not part of the mass); the map to metres puts the 99.9th
  percentile of land at ``relief_spacings`` mean segment spacings (the
  horizontal scale of the tectonic pattern, in metres), or at ``relief_m``
  when that is set, or scales by ``height_scale_m`` per unit when both are
  0, see :func:`finalise`.
* The tect grid is reconstructed from the cloud by a Gaussian blend of the
  ``splat_knn`` nearest segments (sigma ``splat_sigma_factor`` spacings).
  The kernel is truncated: at sigma = 1 spacing the 12th neighbour still
  carries ~3 % of the weight, so the blend jumps where a segment enters or
  leaves the twelve, and on the flank of the continental/oceanic step those
  jumps are the lacy coastline.  With ``splat_knn_base > splat_knn`` the
  *base* height (``min(h, belt height)`` on continental crust, all of
  oceanic) is blended from the wider neighbourhood at the same sigma and
  only the orogenic excess from the ``splat_knn`` nearest, so belts keep
  their width; the crust-type fraction ``c`` stays on the narrow blend
  (docs/coast-fringe.md section 5).  ``splat_kernel`` swaps the truncated
  Gaussian for a kernel without the jump: ``'wendland'`` (compact support,
  width matched to the truncated Gaussian's central weight, the whole
  support always gathered) or ``'tapered'`` (the Gaussian times a taper to
  zero at the ``splat_knn``-th neighbour); section 6.  At a continental
  margin the narrow blend resolves the individual boundary segments, so
  :func:`margin_ramp` can also re-position the continental/oceanic step
  with a kernel ``margin_sigma_factor`` spacings wide and fill the oceanic
  side up to it (raise-only: land, belts and sea level are as the blend
  made them).
* The heat field lives on a coarser grid (``N_tect / heat_grid_divisor``).
* ``uplift`` is *metres per erosion iteration*: the per-segment height
  gained since the reference step (``steps - uplift_window``) in metres,
  times ``uplift_scale``, divided by ``erosion.iterations`` — i.e. the
  erosion stage applying it every iteration reproduces the window's
  tectonic uplift over its run.  The difference is Lagrangian (per
  segment, follows the moving crust) so plate translation does not
  register as uplift/subsidence.  That window is already in ``bedrock``,
  so erosion by default starts from ``bedrock`` less the total it will
  apply and replays it (``erosion.uplift_mode`` 'replay', erosion/maps.py
  ``start_replay``, docs/uplift-replay.md); nothing here bounds the rate
  (``erosion.uplift_max_m``, docs/uplift-ceiling.md).
* ``plate_vel`` is ``omega × pos`` per step expressed as contravariant
  coarse cell components (coarse cells per tectonic step).
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

from ..config import WorldParams
from ..cubesphere import Grid, from_sphere_v
from ..field import FaceField
from ..io.world_store import WorldStore
from ..stubs import fbm_at, fbm_noise
from . import intraplate
from . import orogeny
from .collision import (
    CellTree,
    SmoothSplat,
    accumulate_area,
    boundary_distance,
    build_tree,
    cell_area_steradians,
    collide,
    crystallise,
    delaminate,
    differentiate,
    deposit_density,
    gaussian_smooth,
    grid_cascade,
    label_map_fast,
    relax_segments,
    spawn_segments,
    spread_collisions,
    splat,
    weighted_quantile,
    wendland_support,
)
from .plates import (
    Plates,
    cluster_plates,
    heat_gradient_3d,
    boundary_torques,
    plate_torques,
    slab_pull_torques,
    random_initial_omega,
    rotate_segments,
    seed_supercontinent,
    snap_cratons,
    supercontinent_plates,
    tangent_to_cell_components,
    update_omega,
)
from .segments import CONTINENTAL, OCEANIC, Segments, best_candidate_sphere, mean_spacing

OUTPUTS = ["bedrock", "uplift", "hardness", "plate_id", "plate_vel"]


# --------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------
#: Ledger keys that are *mass*: they sum to the segment total, so a new one
#: has to be added here or the books stop balancing. Sinks are negative by
#: convention -- crust that went back to the mantle -- and only `initial`,
#: `spawned` and `crystallised` may be positive.
MASS_KEYS = ("initial", "spawned", "crystallised", "subducted", "delaminated", "orogen_decayed", "arc_mantle",
             "extent_closed")
#: Ledger keys that are *counters*: how much crust a process relocated. They
#: are diagnostics and must be left out of any mass balance.
COUNTER_KEYS = ("orogen_shaped", "differentiated",
                # the ground (steradians) moved by kind and phase, the closure's factor, and the
                # count of thin continental segments each phase leaves -- variable_extent's
                # own bookkeeping (docs/plate-forces.md section 4d), none of it mass
                "ext_coll_cont", "ext_coll_ocean", "ext_spawn_cont", "ext_spawn_ocean", "extent_close", "extent_close_net",
                "thin_move", "thin_collide", "thin_spawn", "thin_delam", "thin_heat", "thin_rest")
SINK_KEYS = ("subducted", "delaminated", "orogen_decayed")


def _padded_sum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sum of two (P, 3) torque tables whose P may differ (a split this step
    appended plates); the shorter is zero-padded."""
    P = max(a.shape[0], b.shape[0])
    out = np.zeros((P, 3), dtype=np.float64)
    out[:a.shape[0]] += a
    out[:b.shape[0]] += b
    return out


class TectonicSim:
    """Tectonic state and stepping.  Construct with :func:`initialise`.

    Attributes: ``grid`` (tect Grid), ``seg`` (Segments), ``plates``
    (Plates), ``heat`` (float64 FaceField in [0, 1]), ``spacing`` (rad),
    ``step_index``, ``subduction_pts`` (list of (k, 3) arrays of subduction
    positions since the uplift reference step), ``stats`` (per-step dict
    list: M, alive plates, mean speed, gaps, collisions)."""

    def __init__(self, params: WorldParams, grid: Grid, seg: Segments, plates: Plates, heat: FaceField, spacing: float):
        self.params = params
        self.tp = params.tectonics
        self.grid = grid
        self.seg = seg
        self.plates = plates
        self.heat = heat
        self.heat_bg = heat.data.copy()
        self.spacing = float(spacing)
        self.step_index = 0
        self.subduction_pts: list[np.ndarray] = []
        self.stats: list[dict] = []
        self.cell_tree = CellTree(grid)
        self.heat_tree = CellTree(heat.grid)
        self.area_sr = cell_area_steradians(grid)
        tp = self.tp
        self.r_coll = tp.collision_radius_factor * self.spacing
        self.r_gap = tp.gap_radius_factor * self.spacing
        self.r_spawn = tp.spawn_spacing_factor * self.spacing
        self.r_cap = max(self.r_gap, self.r_coll) * 1.0001
        self.gain = tp.convection * tp.force_scale * self.spacing
        self.max_omega = tp.max_speed * self.spacing
        # explicit diffusion: k <= 0.2 * cell² per sub-step (cell in radians, smallest EAC cell ~0.75 of nominal)
        cell_rad = 0.75 * (math.pi / 2.0) / heat.grid.N
        self.diff_substeps = 0 if tp.heat_diffusion <= 0 else int(math.ceil(tp.heat_diffusion / (0.2 * cell_rad**2)))
        self.ref_step = max(0, int(tp.steps) - int(tp.uplift_window))
        #: the quantity the ledger is kept in, and the one this mode conserves.  With
        #: `variable_extent` on, crust moves between columns of different extent, so the
        #: plain sum of the column masses is not it -- `Segments.crust_mass` is
        self.crust_mass = seg.crust_mass if tp.variable_extent else seg.total_mass
        self.idx = None
        self.dist = None
        # mass bookkeeping: the crust mass == initial + spawned + crystallised (collisions/relaxation conserve)
        # `subducted` is a real sink, not a drift term: an oceanic slab hands the
        # overriding plate only `arc_accretion` of itself and the rest goes back
        # to the mantle. It used to be zero to within rounding, because
        # collisions transferred 100 %.
        self.ledger = {k: 0.0 for k in MASS_KEYS}
        self.ledger["initial"] = self.crust_mass()
        #: how many belts of each :mod:`~globe.tectonics.orogeny` type the run
        #: built, and the crust-type pairing of every collision that reached
        #: the classifier (`pair_oc` = oceanic slab under continental
        #: survivor).  Counts, not mass -- a diagnostic, never in the ledger.
        self.belt_census: dict[str, int] = {}
        # fixed points in the mantle frame; plates drift over them and come
        # out with a track of thickened crust (see tectonics/intraplate.py)
        self.hotspot_pos = intraplate.seed_hotspots(int(self.tp.hotspots), self.params.rng("tectonics", 7))
        self.events: list = []
        #: decayed count of continent-on-continent collisions per plate pair
        #: (min, max), for :func:`intraplate.suture`
        self.suture_count: dict[tuple[int, int], float] = {}
        self.suture_block: dict[tuple[int, int], int] = {}   # pair -> step before which it may not weld
        self.sutures = 0

    # -- helpers ------------------------------------------------------------
    def heat_at(self, pos: np.ndarray, fuv=None) -> np.ndarray:
        """Heat at unit vectors ``pos``; ``fuv`` is ``from_sphere_v(pos)`` when
        the caller already has it."""
        if fuv is None:
            return np.clip(self.heat.sample_sphere(pos).astype(np.float64), 0.0, 1.0)
        return np.clip(self.heat.sample_bilinear(*fuv).astype(np.float64), 0.0, 1.0)

    def _heat_blobs(self, pts: np.ndarray, peak: float) -> None:
        """Add Gaussian heat blobs (one spacing wide) at ``pts``."""
        flat = self.heat.interior.reshape(-1)
        self.heat_tree.add_blobs(flat, pts, peak, self.spacing)
        self.heat.interior[...] = flat.reshape(self.heat.interior.shape)

    def _diffuse_heat(self) -> None:
        if self.diff_substeps == 0:
            return
        k = self.tp.heat_diffusion / self.diff_substeps
        R2 = self.heat.grid.R_planet**2
        for _ in range(self.diff_substeps):
            self.heat.exchange_halos()
            lap = self.heat.laplacian().data.astype(np.float64) * R2  # per rad²
            self.heat.data += k * lap
        np.clip(self.heat.data, 0.0, 1.0, out=self.heat.data)

    # -- one step -------------------------------------------------------------
    def step(self) -> dict:
        tp = self.tp
        seg, plates, grid = self.seg, self.plates, self.grid
        k = self.step_index
        rng = self.params.rng("tectonics", 1, k)

        # --- intraplate relief: reorganisation, rifting, hotspots --------
        # Run before the plates move so the new poles take effect this step.
        # Each draws its own rng stream keyed on the step, so enabling one
        # does not shift the others' randomness.
        self.events = []
        if tp.reorganise_every > 0 and k > 0 and k % int(tp.reorganise_every) == 0:
            n = int(tp.reorganise_plates) or int(tp.initial_plates)
            self.events.append(intraplate.reorganise(self, n, self.params.rng("tectonics", 5, k)))
            plates = self.plates
        if tp.rift_every > 0 and k > 0 and k % int(tp.rift_every) == 0:
            ev = intraplate.rift(self, self.params.rng("tectonics", 6, k), int(tp.rift_plates))
            self.events.append(ev)
            plates = self.plates
            for a, b in ev.get("pairs", ()):
                self.suture_block[(min(a, b), max(a, b))] = k + int(tp.suture_cooldown)
        if self.hotspot_pos.shape[0] and tp.hotspot_rate > 0:
            self.events.append(intraplate.apply_hotspots(
                self, self.hotspot_pos, float(tp.hotspot_rate),
                float(tp.hotspot_radius_factor) * self.spacing))

        if k == self.ref_step:
            seg.h_ref = seg.height()
            self.subduction_pts = []

        # how many continental segments are thinner than half a column, after each phase: the
        # ledger that says which process thins the crust (only with variable_extent)
        def _thin():
            return int((seg.thickness[seg.kind == CONTINENTAL] < 0.5 * float(tp.continental_thickness)).sum()) if tp.variable_extent else 0

        def _tick(name, prev):
            now = _thin()
            self.ledger[name] = self.ledger.get(name, 0) + (now - prev)
            return now

        t_ = _thin()
        # 1. move
        mass0 = self.crust_mass()
        tau_slab = None
        cc_pair_max = 0
        rotate_segments(seg, plates)

        t_ = _tick("thin_move", t_)
        # 2. collisions
        ext_before = float(seg.ext.sum()) if tp.variable_extent else 0.0
        # where the ground goes, by kind and by phase, so a drift in the continental share
        # can be read off instead of guessed at (only with variable_extent)
        def _ext_by_kind():
            c = float(seg.ext[seg.kind == CONTINENTAL].sum())
            return c, float(seg.ext.sum()) - c

        if tp.variable_extent:
            c0, o0 = _ext_by_kind()
        tree = build_tree(seg)
        alive = np.ones(seg.M, dtype=bool)
        spent_out: list = []
        arc_out: list = []
        recv_out: list = []
        losers, survivors = collide(seg, tree, self.r_coll, plates.omega, alive, tp.overlap_fraction,
                                    float(tp.arc_accretion), float(tp.arc_birth), self.params.rng("tectonics", 7, k),
                                    shortening=float(tp.continental_shortening), weld_steps=int(tp.weld_steps),
                                    extent_min=(float(tp.extent_min) * self.spacing ** 2) if tp.variable_extent else 0.0,
                                    spent_out=spent_out, arc_out=arc_out, recv_out=recv_out,
                                    arc_thickness=float(tp.arc_thickness) * float(tp.continental_thickness),
                                    arc_density=float(tp.continental_density))
        received = recv_out[0] if (recv_out and tp.variable_extent) else None
        arc_now = float(arc_out[0][1 if tp.variable_extent else 0]) if arc_out else 0.0
        if arc_now:
            # by column or by crust, whichever this mode's ledger is kept in
            self.ledger["arc_mantle"] = self.ledger.get("arc_mantle", 0.0) + arc_now
        n_coll = int(losers.size)
        if n_coll:
            if tp.orogen_shaping > 0.0:
                # the accreted crust *is* the belt: lay it out along the
                # orogen's own cross-section rather than as a Gaussian bump
                self.ledger["orogen_shaped"] = self.ledger.get("orogen_shaped", 0.0) + orogeny.shape_belt(
                    seg, tree, losers, survivors, alive, self.spacing, self.params.R_planet,
                    float(tp.height_scale_m), float(tp.orogen_shaping), CONTINENTAL,
                    accretion=float(tp.arc_accretion), flat_slab_age=float(tp.flat_slab_age),
                    along_strike=float(tp.orogen_along_strike), census=self.belt_census,
                    shortening=float(tp.continental_shortening), width_scale=float(tp.orogen_width_scale), received=received)
            else:
                spread_collisions(seg, tree, losers, survivors, alive, tp.belt_width_factor * self.spacing,
                                  accretion=float(tp.arc_accretion), shortening=float(tp.continental_shortening), received=received)
            # crust that has been through a collision comes out lighter: the
            # light melt stays, the dense residue goes to the mantle.  This is
            # what separates continental from oceanic crust, and so what makes
            # the elevation histogram bimodal instead of one spike.
            if tp.differentiation > 0.0:
                self.ledger["differentiated"] = self.ledger.get("differentiated", 0.0) + differentiate(
                    seg, survivors, float(tp.differentiation), float(tp.density_continental))
            pts = seg.pos[losers].copy()
            if tp.suture_collisions > 0.0:
                cc = (seg.kind[losers] == CONTINENTAL) & (seg.kind[survivors] == CONTINENTAL)
                if cc.any():
                    pa = seg.plate_id[losers[cc]].astype(np.int64)
                    pb = seg.plate_id[survivors[cc]].astype(np.int64)
                    key = np.minimum(pa, pb) * plates.P + np.maximum(pa, pb)
                    for kk, cnt in zip(*np.unique(key, return_counts=True)):
                        pair = (int(kk) // plates.P, int(kk) % plates.P)
                        if self.suture_block.get(pair, -1) > k:
                            continue      # freshly rifted: the teeth grinding is not a collision
                        self.suture_count[pair] = self.suture_count.get(pair, 0.0) + float(cnt)
                        cc_pair_max = max(cc_pair_max, int(cnt))
            if tp.slab_pull > 0.0:
                # before compress: `losers` index the pre-collision cloud
                tau_slab = slab_pull_torques(seg, losers, plates, float(tp.slab_pull), float(tp.ridge_age), OCEANIC,
                                             survivors=survivors, suction=float(tp.trench_suction))
            if k >= self.ref_step:
                self.subduction_pts.append(pts)
            if tp.subduction_heating > 0:
                self._heat_blobs(pts, tp.subduction_heating)
            seg.compress(alive)
            tree = build_tree(seg)
        if tp.relax_rate > 0:
            relax_segments(seg, tree, tp.relax_rate, tp.relax_threshold, self.spacing, int(tp.relax_knn))
        # everything the collision phase changed *except* what the arcs drew from the
        # mantle, which this difference also contains and `arc_mantle` has already taken
        self.ledger["subducted"] += self.crust_mass() - mass0 - arc_now

        # two continents that have been grinding together long enough are one plate
        if tp.suture_collisions > 0.0 and self.suture_count:
            decay = 1.0 - 1.0 / max(float(tp.suture_window), 1.0)
            due = [pr for pr, c in self.suture_count.items() if c >= float(tp.suture_collisions)]
            if due:
                # only continents weld; an ocean plate carrying an arc does not
                pid_all = seg.plate_id.astype(np.int64)
                tot = np.bincount(pid_all, weights=seg.area, minlength=plates.P)[:plates.P]
                cont = np.bincount(pid_all[seg.kind == CONTINENTAL], weights=seg.area[seg.kind == CONTINENTAL], minlength=plates.P)[:plates.P]
                cont_share = cont / np.maximum(tot, 1e-12)
            for pr in due:
                a, b = pr
                if a < plates.P and b < plates.P and plates.alive[a] and plates.alive[b] and a != b \
                        and min(cont_share[a], cont_share[b]) >= float(tp.suture_continental):
                    self.events.append(intraplate.suture(self, a, b, self.params.rng("tectonics", 10, k)))
                    plates = self.plates
                    self.sutures += 1
                    # the joined plate's contacts are the merged plate's now
                    self.suture_count = {q: v for q, v in self.suture_count.items() if b not in q}
                    if tau_slab is not None:
                        tau_slab[a] += tau_slab[b] if b < tau_slab.shape[0] else 0.0
                        if b < tau_slab.shape[0]:
                            tau_slab[b] = 0.0
                else:
                    self.suture_count.pop(pr, None)
            self.suture_count = {q: v * decay for q, v in self.suture_count.items() if v * decay > 0.5}

        # a plate the trenches have just cut in two is two plates from here on
        if tp.plate_split_every > 0 and k % int(tp.plate_split_every) == 0:
            # `tree` is this cloud's: built after the collisions, and nothing since has moved a segment
            ev = intraplate.split_disconnected(self, int(tp.plate_split_min), rng=self.params.rng("tectonics", 9, k), tree=tree)
            if ev["split"] or ev.get("welded"):
                self.events.append(ev)
                plates = self.plates

        # active orogens become former ones: what convergence stops feeding,
        # erosion and root delamination take back down.  After the `subducted`
        # accounting above, which attributes every mass change since `mass0`.
        if tp.orogen_decay > 0.0:
            m_before = self.crust_mass()
            orogeny.relax_orogens(
                seg, float(tp.belt_thickness * (1.0 - tp.continental_density) * tp.height_scale_m),
                float(tp.orogen_floor_m), float(tp.height_scale_m),
                float(tp.orogen_decay), CONTINENTAL)
            self.ledger["orogen_decayed"] += self.crust_mass() - m_before

        if tp.variable_extent:
            c1, o1 = _ext_by_kind()
            self.ledger["ext_coll_cont"] = self.ledger.get("ext_coll_cont", 0.0) + (c1 - c0)
            self.ledger["ext_coll_ocean"] = self.ledger.get("ext_coll_ocean", 0.0) + (o1 - o0)
        t_ = _tick("thin_collide", t_)
        # 3. label map, areas, gaps
        n_gap = 0
        n_new = 0
        # what the collisions destroyed is what extension has to spend: the planet's surface
        # does not grow, so crust spreads only over ground that shortening took elsewhere
        # what *crustal shortening* consumed, not what every collision did: the ground a
        # trench swallows is ocean floor and comes back at a ridge as ocean (the spawn below),
        # and spending it on continental stretch instead handed the continents a share of
        # every subducted plate -- +6.5 steradians of a 12.6 planet over 400 Earth-scale steps
        stretch_budget = float(sum(spent_out)) if tp.variable_extent else 0.0
        if k % max(1, int(tp.label_every)) == 0 or self.idx is None:
            idx, dist = label_map_fast(seg, grid, self.r_cap, tree)
            accumulate_area(seg, idx, self.area_sr, tp.area_blend)
            new, gap = spawn_segments(seg, idx, dist, grid, self.r_gap, self.r_spawn, rng, self.heat, tp.oceanic_thickness, tp.oceanic_density, omega=plates.omega, tree=tree, ext=self.spacing ** 2, stretch=stretch_budget, thin_floor=float(tp.extent_thin_floor) * float(tp.continental_thickness) if tp.variable_extent else 0.0)
            n_gap = int(gap.sum())
            n_new = new.M
            if tp.variable_extent and n_gap:
                # The ground the cloud has left open goes to whoever is nearest, less what the
                # new segments took.  At a ridge that is the sea floor either side of it,
                # which is where the area a trench swallowed belongs: handing the deficit to
                # the whole cloud in proportion instead (the closure below) gave the
                # continents a share of every subducted plate, and at Earth scale -- where
                # there is far more trench per continent than on `small` -- that alone drove
                # them from 0.6 of the planet to 0.75
                g = gap.ravel()
                open_area = float(self.area_sr.ravel()[g].sum()) - n_new * self.spacing ** 2
                if open_area > 0.0:
                    share = np.bincount(idx.ravel()[g], weights=self.area_sr.ravel()[g], minlength=seg.M)
                    tot = float(share.sum())
                    if tot > 0.0:
                        add = share * (open_area / tot)
                        # continental crust at the thinning floor takes no more (see spawn)
                        floor = float(tp.extent_thin_floor) * float(tp.continental_thickness)
                        room = np.where(seg.kind == CONTINENTAL, seg.ext * np.maximum(seg.thickness / max(floor, 1e-9) - 1.0, 0.0), np.inf)
                        add = np.minimum(add, room)
                        keep = seg.ext / np.maximum(seg.ext + add, 1e-12)
                        seg.thickness *= keep                   # the crust on it does not change
                        seg.mass *= keep
                        seg.ext += add
            if n_new and tp.gap_cooling > 0:
                self._heat_blobs(new.pos, -tp.gap_cooling)
            if n_new and tp.ridge_push > 0.0:
                # the crust a ridge makes pushes its own plates off the ridge
                tr = boundary_torques(new.pos, new.plate_id, float(tp.ridge_push) * new.area, plates.com, plates.P, towards=False)
                tau_slab = tr if tau_slab is None else _padded_sum(tau_slab, tr)
            if n_new:
                self.ledger["spawned"] += new.crust_mass() if tp.variable_extent else new.total_mass()
                seg.append(new)
            self.idx, self.dist = idx, dist

        t_ = _tick("thin_spawn", t_)
        # 4. crystallisation, then delamination of over-thickened roots
        # the segments do not move again this step, so their face projection serves the
        # crystallisation's heat and the forces' heat gradient alike (1.5 ms a step at Earth)
        fuv = from_sphere_v(seg.pos)
        T = self.heat_at(seg.pos, fuv)
        mass1 = self.crust_mass()
        crystallise(seg, T, tp.growth, tp.density_base, tp.deposit_density, tp.dissolution_factor, tp.max_thickness)
        self.ledger["crystallised"] += self.crust_mass() - mass1
        if tp.max_crust_thickness > 0 and tp.delamination > 0:
            m_before = self.crust_mass()
            delaminate(seg, float(tp.max_crust_thickness), float(tp.delamination))
            self.ledger["delaminated"] += self.crust_mass() - m_before

        t_ = _tick("thin_delam", t_)
        # 5. heat diffusion + slow relaxation towards the background field
        if tp.heat_insulation > 0 and self.idx is not None:
            # the background itself moves: cold grows under continents, warm
            # under ocean floor, so the attractor the plates feel is not the
            # step-0 noise for the whole run (heat_insulation)
            idx = self.idx
            cont = (idx >= 0) & (seg.kind[np.maximum(idx, 0)] == CONTINENTAL)
            Nh = self.heat.grid.N
            r = idx.shape[1] // Nh
            frac = cont.reshape(6, Nh, r, Nh, r).mean(axis=(2, 4))
            H = self.heat.grid.H
            bg = self.heat_bg[:, H:-H, H:-H]
            bg -= float(tp.heat_insulation) * (2.0 * frac - 1.0)
            np.clip(bg, 0.0, 1.0, out=bg)
        if tp.heat_relax > 0:
            self.heat.data += tp.heat_relax * (self.heat_bg - self.heat.data)
        np.clip(self.heat.data, 0.0, 1.0, out=self.heat.data)
        self._diffuse_heat()

        t_ = _tick("thin_heat", t_)
        # 6. forces
        self.heat.exchange_halos()
        grad3 = heat_gradient_3d(self.heat, seg.pos, fuv)
        plates.update_stats(seg)
        tau = plate_torques(seg, grad3, plates.P)
        slab_ratio = 0.0
        if tau_slab is not None:
            # a split this step appended plates; they get no slab torque until next step
            ts = np.zeros_like(tau)
            n = min(tau_slab.shape[0], plates.P)
            ts[:n] = tau_slab[:n]
            heat_mag = float(np.linalg.norm(tau, axis=1).sum())
            slab_ratio = float(np.linalg.norm(ts, axis=1).sum()) / max(heat_mag, 1e-30)
            tau = tau + ts
        update_omega(plates, tau, self.gain, tp.damping, self.max_omega)

        self.step_index += 1
        spd = plates.speeds()[plates.alive]
        if tp.variable_extent and seg.M:
            c2, o2 = _ext_by_kind()
            self.ledger["ext_spawn_cont"] = self.ledger.get("ext_spawn_cont", 0.0) + (c2 - c1)
            self.ledger["ext_spawn_ocean"] = self.ledger.get("ext_spawn_ocean", 0.0) + (o2 - o1)
            # Close the extent budget: the sphere is covered, whatever the step did to the
            # margins.  Shortening spends ground at convergent boundaries and extension and
            # new sea floor give it back, but the two do not balance step for step, and the
            # splat only reads extents relative to each other -- so the residual is spread
            # over the cloud in proportion and logged.  A large or one-signed `extent_close`
            # means the local processes are not keeping up and the number is doing the work
            tot = float(seg.ext.sum())
            close = (4.0 * math.pi) / max(tot, 1e-12)
            m_before = self.crust_mass()
            seg.ext *= close
            # and the crust is not touched.  Thinning it by the same factor -- on the argument
            # that the volume on a segment is fixed -- was wrong twice over: the closure is a
            # change of units, not of ground (the splat reads extents only against each
            # other), and the factor is one-signed, so it compounded.  Measured on the first
            # Earth bake with the knob on: the sea floor came out at -1461 m against -4095,
            # the bedrock range -2750..10949 against -5705..8704, and refine ran in a sixth
            # of the time because there was hardly any relief left to refine
            # the crust the change of units carries with it.  A mass term, not a counter:
            # it is the ledger's measure of how much of the planet's crust the closure is
            # moving, which is the number to look at when the continents drain (section 4e)
            self.ledger["extent_closed"] = self.ledger.get("extent_closed", 0.0) + (self.crust_mass() - m_before)
            self.ledger["extent_close"] = self.ledger.get("extent_close", 0.0) + abs(close - 1.0)
            self.ledger["extent_close_net"] = self.ledger.get("extent_close_net", 0.0) + (close - 1.0)
        t_ = _tick("thin_rest", t_)              # everything after the forces: events, closure
        info = {
            "step": k,
            "M": seg.M,
            "plates": plates.n_alive(),
            "collisions": n_coll,
            "gap_cells": n_gap,
            "spawned": n_new,
            "speed_mean": float(spd.mean() / self.spacing) if spd.size else 0.0,
            "speed_max": float(spd.max() / self.spacing) if spd.size else 0.0,
            "mass": seg.total_mass(),
            "slab_ratio": slab_ratio,
            "sutures": self.sutures,
            "cc_pair_max": cc_pair_max,
        }
        self.stats.append(info)
        return info

    def run(self, steps: int | None = None, log=print, log_every: int = 50, on_frame=None, frames: int = 0) -> "TectonicSim":
        """Advance `steps` steps.

        With `frames > 0`, `on_frame(sim, index)` is called `frames` times at
        even intervals (and once at the end), so an animation can be captured
        during the run instead of re-simulating for it afterwards -- which
        costs a second full simulation, 39 minutes at Earth scale.
        """
        steps = int(self.tp.steps) if steps is None else int(steps)
        every = max(1, steps // max(1, frames)) if frames > 0 else 0
        if every:
            on_frame(self, 0)
        t0 = time.time()
        for i in range(steps):
            info = self.step()
            if every and ((i + 1) % every == 0 or i == steps - 1):
                on_frame(self, i + 1)
            if log is not None and log_every and (self.step_index % log_every == 0 or self.step_index == steps):
                log(
                    f"[tectonics] step {self.step_index}/{steps}: M={info['M']} plates={info['plates']} "
                    f"coll={info['collisions']} gaps={info['gap_cells']} new={info['spawned']} "
                    f"v={info['speed_mean']:.3f}/{info['speed_max']:.3f} sp/step mass={info['mass']:.1f} "
                    + (f"slab/heat={info['slab_ratio']:.2f} " if info.get('slab_ratio') else "")
                    + (f"sutures={info['sutures']} " if info.get('sutures') else "")
                    + (f"cc_pair_max={info['cc_pair_max']} " if info.get('cc_pair_max') else "")
                    + f"({time.time() - t0:.1f}s)"
                )
        return self



#: The continents come with a past the run did not simulate.  They are all there at the first
#: step, so every one of their segments carries the same age at the end and a crust-age map
#: shows one flat colour over every continent -- where a real one has cratons of three billion
#: years against belts of a few hundred million.  :func:`inherited_age` gives the crust that
#: was there from the start an age of its own -- the craton nuclei oldest, the belts welded
#: between them younger, varying smoothly in between -- spanning this many times the run's own
#: length.  Drawn from a stream of its own and read by nothing but the diagnostic: the
#: simulation, and every output of this stage, is bit for bit what it was without it.
PREHISTORY_RUNS = 8.0
#: how much age structure a run has to make for itself before the invented past fades out,
#: as a share of the run's length (the interquartile spread of the continental ages).  A run
#: that assembles, rifts and re-assembles welds a belt at every collision and leaves the old
#: cores between them -- measured at 0.39 of a run on `small` over 9000 steps with
#: tectonics.variable_extent on -- and needs no past handed to it; a run that starts with its
#: continents assembled and never breaks them up makes none at all, and without a past every
#: continent is one flat colour.  So the two are mixed by how much the run did itself
PREHISTORY_FADE = 0.25
#: sub-key of the stream the inherited ages come from
PREHISTORY_KEY = 5717


def inherited_age(sim) -> np.ndarray:
    """Age (steps) the continental crust brought with it, per segment: 0 for
    the ocean floor and for anything born during the run (an island arc is as
    young as it looks), and :data:`PREHISTORY_RUNS` runs' worth for a craton."""
    seg = sim.seg
    at_start = seg.age >= float(sim.step_index) - 0.5      # there before the first step ran
    # ...and how much of that column is still the crust it started with, rather than a belt
    # the run stacked into it.  A share, not a flag: a craton that has had a tenth of its
    # column reworked keeps nine tenths of its past, where a flag would drop it off a cliff
    # to nothing and speckle the map wherever the two kinds of crust met
    keep = np.clip(seg.rework / np.maximum(seg.age, 1.0), 0.0, 1.0)
    # what the run made for itself, and so how much of a past it still needs.  Measured on
    # the assembly age, the same number the map draws: a run whose collisions rework the
    # continents has structure of its own there even while every protolith `age` is the run
    own = seg.rework[seg.kind != OCEANIC]
    made = float(np.percentile(own, 75) - np.percentile(own, 25)) / max(float(sim.step_index), 1.0) if own.size else 0.0
    fade = min(max(1.0 - made / max(PREHISTORY_FADE, 1e-9), 0.0), 1.0)
    if fade <= 0.0:
        return np.zeros(seg.M)
    # the run's whole length, not how much of it has gone: a craton's past is a number it
    # carries, and scaling it with the step would have the continents ageing as the animation
    # plays while the map's scale stayed still, which reads as one flat colour early on
    span = PREHISTORY_RUNS * max(float(sim.params.tectonics.steps), 1.0)
    rng = np.random.default_rng(int(sim.params.world.seed) + PREHISTORY_KEY)
    f = np.clip(0.5 + 0.5 * fbm_at(seg.pos, rng, octaves=3, base_freq=2.0), 0.0, 1.0)
    old = np.where(seg.craton > 0, 0.70 + 0.30 * f, 0.12 + 0.48 * f)
    return np.where((seg.kind != OCEANIC) & at_start, fade * span * old * keep, 0.0)


def ridge_buoyancy(seg, tp) -> np.ndarray:
    """Thermal buoyancy of young oceanic crust, in bedrock units.

    Half-space cooling: the ocean floor subsides as the square root of its
    age, ~2500 m at a ridge crest deepening to ~5500 m by 80 My and then
    flattening.  That 3000 m ramp is most of the ocean's hypsometric range,
    and reproducing it is the difference between an abyssal plain and a
    uniform shelf -- measured without it, our seafloor spanned 1846 m
    where Earth's spans ~3000.

    ``max(0, 1 - sqrt(age / ridge_age))`` rather than the exponential this
    used to be, for two reasons: it is the actual half-space result, and it
    reaches zero at a finite age instead of asymptotically, so `ridge_age`
    means "the age at which subsidence is done" and can be read off the
    seafloor's measured lifetime.  Applied to oceanic crust only --
    continental crust is in isostatic equilibrium and does not subside with
    age; giving it a ridge term put a spurious 0.15 on every young craton.
    """
    tau = max(float(tp.ridge_age), 1.0)
    b = float(tp.ridge_height) * np.maximum(0.0, 1.0 - np.sqrt(np.maximum(seg.age, 0.0) / tau))
    return np.where(seg.kind == OCEANIC, b - float(tp.abyss_depth), 0.0)


def sea_level(bed, area, continental, params) -> float:
    """Where the water stops, in bedrock units.

    Two placements. The default is a quantile of *area*: put
    ``world.land_fraction`` of the surface above the waterline. That works
    when height is a single continuum, and stops working the moment it is
    not -- with two humps ~4 km apart, an area quantile that misses the
    continental hump lands in the near-empty trough between them, and every
    continent then reads as a plateau standing its full crustal buoyancy
    above the sea.

    Measured across three seeds of an otherwise identical Earth-scale
    configuration, the continental fraction at step 1500 came out 0.351,
    0.462 and 0.260. Against a fixed ``land_fraction = 0.25`` the first two
    give land under 1 km of 78.0 % and 75.6 % (Earth: 71 %) and the third
    gives **17.4 %** -- its margin was 0.01, so the quantile fell into the
    trough. The knob cannot be tuned around that: the thing it has to cut
    is a different size every seed.

    So with ``tectonics.shelf_fraction > 0`` sea level is instead a quantile
    of the *continental* crust: drown that fraction of it and let the land
    area fall out. That is what sea level physically is -- Earth's oceans
    hold just enough water to cover the shelves, ~27 % of the continental
    crust, leaving 29 % of the globe dry -- and it self-corrects, because
    the quantity it is measured against is the one that varies.
    """
    f = float(params.tectonics.shelf_fraction)
    if f > 0.0 and continental is not None and continental.any():
        return float(weighted_quantile(bed[continental], area[continental], f))
    return float(weighted_quantile(bed, area, 1.0 - params.world.land_fraction))


def frame_bed(sim, tree=None) -> np.ndarray:
    """The current crust as a sea-levelled bed on the tect grid.

    Shared by the in-simulation animation capture and the quicklook, so a
    frame drawn during the run is identical to the finished map.  ``tree``
    is the segment KD-tree when the caller has already built it.
    """
    from .collision import SmoothSplat, build_tree

    tp, grid = sim.tp, sim.grid
    tree = build_tree(sim.seg) if tree is None else tree
    blend, base_blend = splat_blends(tree, grid, tp, sim.spacing, sim.seg)
    buoy = ridge_buoyancy(sim.seg, tp)
    raw, c, _ = margin_ramp(grid, blend, sim.seg, sim.seg.height() + buoy, tp, sim.spacing, base_blend=base_blend)
    bed = _smooth_field(grid, raw, tp, cascade=True).interior
    cont = c > 0.5 if tp.shelf_fraction > 0 else None
    sea = sea_level(bed, grid.interior_cell_area.astype(np.float64), cont, sim.params)
    return bed - sea


def frame_image(sim, width: int = 900):
    """One animation frame: the unfolded cube net, as a PIL image.

    The palette is deliberately re-derived per frame rather than pinned.
    The crust is still being created, so its relief grows by orders of
    magnitude and a scale fixed at step 0 would flatten everything after --
    at the cost that step 0, where relief is still near zero, gets its ramp
    stretched and paints all land as high ground.
    """
    from PIL import Image

    from ..viz import quicklook as ql

    img = ql.render_height(frame_bed(sim), cell_size=sim.grid.cell_size_m)
    im = Image.fromarray(np.ascontiguousarray(ql.to_globe(img, size=max(256, width // 2))))
    if im.width != width:
        im = im.resize((width, max(1, round(im.height * width / im.width))), Image.LANCZOS)
    return im


def heat_grid(params: WorldParams) -> Grid:
    """The heat field lives on a coarser grid (``N_tect / heat_grid_divisor``,
    at least 2H+8 cells): it is a low-frequency driver, and explicit
    diffusion on it costs a handful of Laplacians per step at any N_tect."""
    from ..cubesphere import get_grid

    tp, w = params.tectonics, params.world
    n = max(2 * w.halo + 8, int(tp.N_tect) // max(1, int(tp.heat_grid_divisor)))
    return get_grid(n, w.halo, w.cell_size_m * w.N_c / n, params.R_planet)


def initialise(params: WorldParams, log=print) -> TectonicSim:
    """Segments (best-candidate sampling), plates (spherical k-means with
    jittered sizes and noisy boundaries), heat (low-frequency noise
    normalised to [0, 1]) and random initial Euler poles."""
    tp = params.tectonics
    grid = params.tect_grid()
    hgrid = heat_grid(params)
    rng = params.rng("tectonics", 0)
    M = int(tp.segments)
    spacing = mean_spacing(M)
    pos = best_candidate_sphere(M, rng)
    noise = fbm_noise(hgrid, rng, int(tp.heat_noise_octaves), float(tp.heat_noise_freq)).astype(np.float64)
    lo, hi = float(noise.min()), float(noise.max())
    heat = FaceField(hgrid, (noise - lo) / max(hi - lo, 1e-9), name="heat")
    heat.exchange_halos()
    kind, craton, taper = seed_supercontinent(
        pos, float(tp.continental_fraction), float(tp.craton_fraction), int(tp.cratons), rng,
        float(tp.supercontinent_roughness), float(tp.craton_roughness),
        float(tp.margin_taper), float(tp.margin_thinning))
    cont = kind == CONTINENTAL
    # Continental crust is not one thickness or one density. Cratons carry
    # 40-45 km of crust against 30-35 for the mobile belts welded between
    # them, but they are also *denser* (a mafic granulite lower crust), and
    # on Earth the two very nearly cancel: shields are thick and low, not
    # high. Height comes from the margin taper -- crust stretched thin at a
    # rifted edge, which is what a shelf is -- and from orogeny thickening
    # crust in the present, not from age.
    is_cr = craton > 0
    cont_t = np.where(is_cr, tp.craton_thickness, tp.belt_thickness) * tp.continental_thickness
    cont_t = cont_t * taper
    cont_t = cont_t * (1.0 + float(tp.continental_spread) * fbm_at(pos, rng, octaves=4, base_freq=3.0))
    thickness = np.where(cont, cont_t, tp.oceanic_thickness * (1.0 + 0.2 * (rng.random(M) - 0.5)))
    cont_rho = np.where(is_cr, tp.craton_density, tp.continental_density)
    density = np.where(cont, cont_rho, tp.oceanic_density)
    plate_id = supercontinent_plates(pos, kind, int(tp.initial_plates), rng,
                                     size_jitter=float(tp.plate_size_jitter))
    seg = Segments(pos, thickness, density, 0.0, plate_id, 4.0 * math.pi / M, kind=kind, craton=craton)
    snap_cratons(seg)          # a boundary goes around a craton, not through it
    plates = Plates(int(tp.initial_plates))
    plates.update_stats(seg)
    random_initial_omega(plates, rng, tp.initial_speed * spacing)
    sim = TectonicSim(params, grid, seg, plates, heat, spacing)
    if log is not None:
        log(
            f"[tectonics] crust: {int(cont.sum())} continental / {M} segments "
            f"({cont.mean() * 100:.1f} %) as one supercontinent, "
            f"{int((craton > 0).sum())} craton segments "
            f"({(craton > 0).sum() / max(cont.sum(), 1) * 100:.0f} % of it) in {len(np.unique(craton[craton > 0]))} nuclei; "
            f"h_craton={tp.craton_thickness * (1.0 - tp.craton_density):.3f} "
            f"h_belt={tp.belt_thickness * (1.0 - tp.continental_density):.3f} "
            f"h_ocean={tp.oceanic_thickness * (1.0 - tp.oceanic_density):.3f} bedrock units"
        )
        log(
            f"[tectonics] {grid.describe()}; M={M} segments, spacing={spacing:.4f} rad "
            f"({spacing * grid.N / (math.pi / 2):.1f} tect cells), plates={plates.n_alive()}, "
            f"r_coll={sim.r_coll:.4f}, r_gap={sim.r_gap:.4f}, diffusion substeps={sim.diff_substeps}"
        )
    return sim


def simulate(params: WorldParams, log=print, steps: int | None = None) -> TectonicSim:
    return initialise(params, log).run(steps, log)


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------
def _smooth_field(grid: Grid, values: np.ndarray, tp, cascade: bool) -> FaceField:
    """(6, N, N) cell values -> cascaded (optional) + Gaussian-smoothed
    float64 FaceField with exchanged halos."""
    f = FaceField.from_interior(grid, np.asarray(values, dtype=np.float64), exchange=True)
    if cascade and tp.cascade_passes > 0:
        f = grid_cascade(f, int(tp.cascade_passes), tp.cascade_rate, tp.cascade_threshold)
    return gaussian_smooth(f, float(tp.smooth_sigma))


def splat_blends(tree, grid: Grid, tp, spacing: float, seg=None) -> tuple[SmoothSplat, SmoothSplat | None]:
    """The narrow ``splat_knn`` blend that rasterises the segment cloud and,
    when ``splat_knn_base > splat_knn``, the wider blend of the same sigma
    for the base height (else ``None``).  One place, so :func:`finalise`
    and :func:`frame_bed` build the same pair and the timeline's last frame
    is the finished map.  Both use ``tectonics.splat_kernel``; every field
    :func:`finalise` rasterises from the cloud (height, the crust-type
    fraction ``c``, ``dh`` for uplift, age and density for hardness) goes
    through them, so crust_kind, uplift and hardness see the kernel the bed
    does.  For ``'wendland'`` the support radius comes from the design
    spacing (:func:`~globe.tectonics.collision.wendland_support`), not from
    the current segment count, so it is the same in every frame."""
    sigma = tp.splat_sigma_factor * spacing
    kernel = str(tp.splat_kernel)
    knn, kb = int(tp.splat_knn), int(tp.splat_knn_base)

    def support(k):  # the Wendland radius is fixed per world, not per frame
        return wendland_support(sigma, k, spacing) if kernel == "wendland" else None

    # with variable extent every segment brings its own width, sqrt(ext) scaled to the design
    # spacing, floored so the kernel never gets narrower than the cloud's own packing length
    # each segment weighted by the ground it covers, in units of the design extent: a cloud
    # that is one extent everywhere multiplies by one and is the old blend to the bit
    weight_k = None
    if seg is not None and bool(getattr(tp, "variable_extent", False)) and seg.M:
        weight_k = seg.ext / (spacing ** 2)
    blend = SmoothSplat(tree, grid, sigma, knn, kernel, support(knn), weight_k)
    base_blend = SmoothSplat(tree, grid, sigma, kb, kernel, support(kb), weight_k) if kb > knn else None
    return blend, base_blend


def margin_ramp(grid: Grid, blend: SmoothSplat, seg: Segments, h: np.ndarray, tp, spacing: float,
                base_blend: SmoothSplat | None = None):
    """Per-segment height ``h`` -> (6, N, N) bed on the tect grid with a
    smooth continental margin.  Returns ``(bed, c, lift)``: the bed, the
    narrow continental fraction ``c`` (the crust-type boundary, unchanged),
    and the raise that was applied (zeros when the knob is off).

    ``base_blend`` (``tectonics.splat_knn_base``) is the same Gaussian over
    more neighbours: when given, the bed is ``base_blend(h_base) + blend(h -
    h_base)`` -- the base height without the kernel truncation's jumps, the
    orogenic excess at the narrow resolution -- and ``c`` is still
    ``blend(kind)``.  ``None`` is the old ``blend(h)``, bit for bit.

    ``blend(h)`` is exactly ``c * hc + (1 - c) * ho`` -- the continental
    fraction of the nearest segments times their mean height plus the
    oceanic complement -- and ``c`` steps from 1 to 0 over about one
    spacing, at the positions of the individual boundary segments.  The
    continental/oceanic step is the largest on the map (~0.06 units, 1.5 km
    on Earth), so the shelf edge came out scalloped one segment at a time.
    Here the *step* term alone is re-positioned: ``cw`` is ``c`` blurred by
    ``margin_sigma_factor`` spacings and ``(cw - c) * (hc_base - ho)`` is
    what the blend would have added had it seen the crust-type boundary at
    that resolution.  Only the continental *base* (``min(h, belt height)``)
    enters the step height, so belt cross-sections keep the narrow
    resolution; the correction is raise-only, so land, belts and sea level
    are exactly as the blend made them and the oceanic side climbs to the
    ramp (a continental rise); and it is exactly zero where ``cw == c`` --
    the interior of either crust.  Measured on `small`: shelf-edge isoline
    L/sqrt(A) 7.79 -> 6.38 with the coastline unchanged (docs/coast-fringe.md).
    """
    kind = (seg.kind == CONTINENTAL).astype(np.float64)
    # the nominal belt / craton height, the same product as the seeding
    # (both float at 0.180); anything above it on continental crust is
    # orogenic thickening and stays out of the step (and of the wide base)
    h_belt = tp.continental_thickness * tp.belt_thickness * (1.0 - tp.continental_density)
    h_base = np.where(kind > 0, np.minimum(h, h_belt), h)
    if base_blend is None:
        bed0 = blend(h)
    else:
        bed0 = base_blend(h_base) + blend(h - h_base)
    c = blend(kind)
    f = float(tp.margin_sigma_factor)
    if f <= 0.0:
        return bed0, c, np.zeros_like(bed0)
    cb = blend(kind * h_base)      # c * mean continental base height
    ob = blend((1.0 - kind) * h)   # (1 - c) * mean oceanic height
    sig = f * spacing * grid.N / (math.pi / 2)  # spacings -> tect cells

    def blur(a):
        return gaussian_smooth(FaceField.from_interior(grid, a, exchange=True), sig).interior

    cw = np.clip(blur(c), 0.0, 1.0)  # the cubic halo exchange can overshoot [0, 1]
    hc_w = np.where(cw > 1e-12, blur(cb) / np.maximum(cw, 1e-12), 0.0)
    ho_w = np.where(cw < 1.0 - 1e-12, blur(ob) / np.maximum(1.0 - cw, 1e-12), 0.0)
    # the step is not sign-definite (young, buoyant ocean floor can stand
    # above a thinned margin base): only a downward step is a shelf edge
    corr = (cw - c) * np.maximum(hc_w - ho_w, 0.0)
    corr = np.where(np.abs(cw - c) < 1e-9, 0.0, corr)
    lift = np.maximum(corr, 0.0)
    return bed0 + lift, c, lift


class _Resampler:
    """Tect-grid field -> (6, N_c, N_c) interior values on the coarse grid
    (cubic by default; ``order=0`` = nearest, for integer maps).  The
    (face, u, v) of the coarse cell centres are computed once."""

    def __init__(self, coarse: Grid):
        self.face, self.u, self.v = from_sphere_v(coarse.interior_centers)

    def __call__(self, field: FaceField, order: int = 3) -> np.ndarray:
        if order == 0:
            return field.sample_nearest(self.face, self.u, self.v)
        if order == 1:
            return field.sample_bilinear(self.face, self.u, self.v)
        return field.sample_cubic(self.face, self.u, self.v)


def _land_slope_info(coarse: Grid, bed_m: np.ndarray, params: WorldParams) -> dict:
    """Median / p90 land slope (rise/run) of the tectonic bedrock and the
    fraction of the land above ``erosion.talus_slope_hard``.  The vertical
    scale is only sane if this stays well below the talus angle: erosion
    cannot carve a landscape out of a surface that already stands at it."""
    land = bed_m > 0.0
    if not land.any():
        return {"land_slope_median": None, "land_slope_p90": None, "land_above_talus_fraction": None}
    slope = FaceField.from_interior(coarse, bed_m.astype(np.float32), name="bed").gradient().vec_norm().interior[land]
    hard = float(params.erosion.talus_slope_hard)
    return {
        "land_slope_median": float(np.median(slope)),
        "land_slope_p90": float(np.percentile(slope, 90)),
        "land_above_talus_fraction": float((slope > hard).mean()),
    }


def inject_detail(bed: np.ndarray, coarse: Grid, tp, rng) -> np.ndarray:
    """Give the bedrock the fractal detail erosion cannot manufacture.

    Erosion reworks the spectrum it is handed; it cannot add variance that
    was never there.  Measured through the whole chain on one world, over
    200-1600 m:

        bedrock leaving tectonics   beta 12.99
        after coarse erosion        beta  6.47
        after refine (R = 2, 4, 8)  beta  6.0 - 6.3

    Refinement does not touch that band -- the band is already resolved on
    the coarse grid, so refine only adds detail *below* it, and R makes no
    difference (6.22 / 6.26 / 5.97 at R = 2 / 4 / 8).  Real topography sits
    near 2.  The deficit is inherited from here, so here is where it has to
    be fixed.

    Three things that experiment left unsolved, and how this handles them:

    * **Seams.** It generated noise per face, so every cube edge showed.
      :func:`~globe.stubs.fbm_noise` evaluates a 3-D lattice at the cell
      centres *on the sphere*, so it is continuous across faces by
      construction -- the same generator the heat field already uses.
    * **Amplitude.** A flat fraction of global relief roughens plains as
      hard as mountains.  The amplitude here follows the *local* relief
      (max - min over `detail_relief_cells`), the way the refine stage
      scales its own detail by local slope and relief, so a craton stays a
      craton.
    * **The mass budget.** Injecting detail moves the land fraction (14.4 %
      -> 14.0 % in the experiment) because sea level was fixed before the
      noise went in.  The caller re-derives sea level afterwards.

    `bed` is in bedrock units, already sea-levelled; the return is the same
    array with detail added.
    """
    if tp.detail_amp <= 0.0 or tp.detail_octaves <= 0:
        return bed
    # coarsest injected octave ~= `detail_cells` coarse cells.  A 3-D lattice
    # of frequency f has wavelength 2/f in the [-1, 1] cube, i.e. ~2/f radians
    # on the unit sphere, and a coarse cell subtends (pi/2)/N.
    k = max(2.0, float(tp.detail_cells))
    freq = 4.0 * coarse.N / (k * math.pi)
    noise = fbm_noise(coarse, rng, int(tp.detail_octaves), freq)
    noise = FaceField(coarse, noise, name="detail").interior.astype(np.float64)
    n = noise.std()
    if n > 0:
        noise /= n
    # local relief, so plains stay flat and mountains get rough
    r = max(1, int(tp.detail_relief_cells))
    hi = ndimage.maximum_filter(bed, size=(1, r, r), mode="nearest")
    lo = ndimage.minimum_filter(bed, size=(1, r, r), mode="nearest")
    return bed + float(tp.detail_amp) * (hi - lo) * noise


def finalise(sim: TectonicSim) -> dict[str, FaceField]:
    """Coarse-grid outputs (docs/DEVELOPING.md): ``bedrock`` (m), ``uplift``
    (m per erosion iteration), ``hardness`` [0, 1], ``plate_id`` (int16),
    ``plate_vel`` (coarse cells per step, vector).  Also returns the
    diagnostic ``collision_zone`` (uint8) and ``heat`` (resampled)."""
    params, tp = sim.params, sim.tp
    grid, seg, plates = sim.grid, sim.seg, sim.plates
    coarse = params.coarse_grid()
    _resample = _Resampler(coarse)
    tree = build_tree(seg)
    idx, dist = label_map_fast(seg, grid, sim.r_cap, tree)

    # -- bedrock and uplift (bedrock units on the tect grid) --
    # the narrow blend rasterises everything; the wide one (splat_knn_base,
    # None when off) only the base height, see margin_ramp.  dh, age and
    # density stay on the narrow blend: uplift and hardness are untouched
    blend, base_blend = splat_blends(tree, grid, tp, sim.spacing, sim.seg)
    # thermal buoyancy of young crust: ridges at rifts, subsidence with age
    elapsed = float(sim.step_index - sim.ref_step)
    buoy = ridge_buoyancy(seg, tp)
    ref = Segments(seg.pos, seg.thickness, seg.density, np.maximum(seg.age - elapsed, 0.0), seg.plate_id, seg.area, kind=seg.kind)
    buoy_ref = ridge_buoyancy(ref, tp)
    h = seg.height() + buoy
    raw, c_t, lift = margin_ramp(grid, blend, seg, h, tp, sim.spacing, base_blend=base_blend)
    bed_t = _smooth_field(grid, raw, tp, cascade=True)
    dh_t = _smooth_field(grid, blend(seg.height() - seg.h_ref + buoy - buoy_ref), tp, cascade=True)
    bed = _resample(bed_t).astype(np.float64)
    dh = _resample(dh_t).astype(np.float64)
    area = coarse.interior_cell_area.astype(np.float64)
    # Crust type on the coarse grid.  `sea_level` uses it only in shelf mode,
    # but it is saved as a diagnostic either way (see `crust_kind` below), so
    # it is computed unconditionally and `shelf_mask` -- not `cont_c` -- is
    # what reaches `sea_level`, which keeps the placement unchanged.
    cont_t = FaceField.from_interior(grid, c_t, exchange=True)
    cont_c = _resample(cont_t).astype(np.float64) > 0.5
    shelf_mask = cont_c if tp.shelf_fraction > 0 else None
    q = sea_level(bed, area, shelf_mask, params)
    bed -= q
    # fractal detail, then sea level again: the noise shifts how much of the
    # surface is above water, so the quantile has to be re-derived or the
    # land fraction drifts (measured 14.4 % -> 14.0 % when it was not)
    bed = inject_detail(bed, coarse, tp, params.rng("tectonics", 8))
    bed -= sea_level(bed, area, shelf_mask, params)
    if float(tp.ranges_amp) > 0.0:
        # ranges and basins along the belts' strike (globe/tectonics/ranges.py)
        from .ranges import inject_ranges

        bed = inject_ranges(bed, bed_t.interior, coarse, grid, tp, params.rng("tectonics", 9), params.R_planet, float(tp.height_scale_m))
        bed -= sea_level(bed, area, shelf_mask, params)
    land = bed > 0
    # vertical scale: tie the relief to the *horizontal* scale of the
    # tectonic pattern (the mean segment spacing, in metres) unless an
    # explicit relief_m is given.  A constant relief_m puts the same 5 km
    # on a pattern whose feature width in cells is preset-independent, so
    # the land ends up at the talus angle everywhere.
    target = float(tp.relief_m) if tp.relief_m > 0 else float(tp.relief_spacings) * sim.spacing * params.R_planet
    if target > 0 and land.any():
        top = weighted_quantile(bed[land], area[land], 0.999)
        scale = target / max(top, 1e-6)
    else:
        scale = tp.height_scale_m
    bedrock = FaceField.from_interior(coarse, (bed * scale).astype(np.float32), name="bedrock")

    # collision zones of the uplift window
    if sim.subduction_pts:
        pts = np.concatenate(sim.subduction_pts, axis=0)
        zone_t = sim.cell_tree.within(pts, tp.collision_zone_factor * sim.spacing)
    else:
        zone_t = np.zeros((6, grid.N, grid.N), dtype=bool)
    zone = _resample(FaceField.from_interior(grid, zone_t.astype(np.uint8), exchange=True), order=0).astype(bool)
    slope_info = _land_slope_info(coarse, bed * scale, params)
    n_iter = max(int(params.erosion.iterations), 1)
    up = dh * scale * tp.uplift_scale / n_iter
    pos_up = up[up > 0]
    baseline = tp.uplift_baseline * float(np.percentile(pos_up, 99)) if pos_up.size else 0.0
    up = np.where(zone, np.maximum(up, 0.0), np.where(up < 0.0, up, np.maximum(up, baseline)))
    uplift = FaceField.from_interior(coarse, up.astype(np.float32), name="uplift")

    # -- hardness --
    age_t = _smooth_field(grid, blend(seg.age), tp, cascade=False)
    den_t = _smooth_field(grid, blend(seg.density), tp, cascade=False)
    bd_t = FaceField.from_interior(grid, boundary_distance(tree, seg, grid, idx), exchange=True)
    age_c = _resample(age_t).astype(np.float64)
    den_c = _resample(den_t).astype(np.float64)
    bd_c = _resample(bd_t, order=1).astype(np.float64)
    age_norm = np.clip(age_c / max(float(np.percentile(age_c, 99)), 1.0), 0.0, 1.0)
    d_lo, d_hi = np.percentile(den_c, [1, 99])
    den_norm = np.clip((den_c - d_lo) / max(d_hi - d_lo, 1e-6), 0.0, 1.0)
    prox = np.clip(1.0 - bd_c / (tp.boundary_width_factor * sim.spacing), 0.0, 1.0)
    # Bedrock fabric.  Crust accretes at plate boundaries, so lines of equal
    # age are the strata it was laid down in: modulating hardness along age
    # lays bands parallel to the boundary that built them, narrow where
    # accretion was fast and wide where it was slow.  Without this the field
    # is a smooth two-tone blend (autocorrelation 0.53 at 64 cells) with
    # nothing for drainage to organise around, and every continent develops
    # the same radial network.  `tanh` sharpens the contacts: real strata
    # meet at a contact, not a gradient.
    strata = 0.0
    if tp.strata_amp > 0.0 and tp.strata_period > 0.0:
        wave = np.sin(2.0 * np.pi * age_c / float(tp.strata_period))
        strata = np.tanh(2.5 * wave) / np.tanh(2.5)
    hard = np.clip(0.3 + 0.5 * age_norm + 0.3 * den_norm - 0.4 * prox + float(tp.strata_amp) * strata, 0.0, 1.0)
    hardness = FaceField.from_interior(coarse, hard.astype(np.float32), name="hardness")

    # -- plate ids (compacted to 0..P'-1 in original order) and velocities --
    alive_ids = np.nonzero(plates.alive)[0]
    remap = np.full(plates.P, -1, dtype=np.int32)
    remap[alive_ids] = np.arange(alive_ids.size, dtype=np.int32)
    pid_t = FaceField.from_interior(grid, splat(seg.plate_id, idx).astype(np.int32), exchange=True)
    pid_c = _resample(pid_t, order=0).astype(np.int64)
    plate_id = FaceField.from_interior(coarse, remap[pid_c].astype(np.int16), name="plate_id")
    om = plates.omega[pid_c]  # (6, N, N, 3) rad/step of the owning plate
    p3 = coarse.interior_centers
    v3 = np.cross(om, p3)
    H, N = coarse.H, coarse.N
    ei, ej = np.meshgrid(np.arange(H, H + N), np.arange(H, H + N), indexing="ij")
    u, v = coarse.cell_uv(ei, ej)
    faces = np.broadcast_to(np.arange(6)[:, None, None], (6, N, N))
    comp = tangent_to_cell_components(coarse, faces, np.broadcast_to(u, (6, N, N)), np.broadcast_to(v, (6, N, N)), v3)
    plate_vel = FaceField.from_interior(coarse, comp.astype(np.float32), is_vector=True, name="plate_vel")

    heat_c = FaceField.from_interior(coarse, _resample(sim.heat, order=1).astype(np.float32), name="heat")
    zone_f = FaceField.from_interior(coarse, zone.astype(np.uint8), name="collision_zone")
    # 1 where the crust under a cell is continental.  Without it on disk there
    # is no way to ask which *kind* of crust a submerged cell sits on, and
    # "the ocean is too shallow" cannot be separated from "much of the ocean
    # is drowned continental shelf" -- which want opposite fixes (see
    # scripts/ocean_depth.py).  Diagnostic only: not in OUTPUTS, so the stage
    # hash and every world already baked are unchanged.
    crust_kind = FaceField.from_interior(coarse, cont_c.astype(np.uint8), name="crust_kind")
    # How old the crust under a cell is, in steps since the segment formed (a
    # segment ages one a step, collision.py; the ocean floor is born at a rift
    # and dies at a trench, so it reads young and striped, while a craton
    # carries the whole run).  The splat's weighted mean, so a cell between
    # segments of different ages lands between them.  Diagnostic, like
    # crust_kind: not in OUTPUTS, so no stage hash and no baked world changes
    # each kind's own splat, and the cell takes the one it is (cont_c, the same flag
    # crust_kind saves).  One blend over both would carry a continent's inherited age out into
    # the sea floor beside it and put a ring of impossibly old crust round every coast
    # the sea floor is mapped at the age of the rock, which is the age it was born at a ridge;
    # the continents at the age they were last assembled (`Segments.rework`), which is what a
    # crust-age map of the land plots -- a craton's number is its own, an orogen's is its last
    # orogeny, and the difference between the two is the structure the map is for
    ages = np.where(seg.kind == OCEANIC, seg.age, seg.rework) + inherited_age(sim)
    ocean_m = (seg.kind == OCEANIC).astype(np.float64)

    def kind_age(mask):
        w = _resample(FaceField.from_interior(grid, blend(mask), exchange=True), order=1)
        a = _resample(FaceField.from_interior(grid, blend(ages * mask), exchange=True), order=1)
        return a / np.maximum(w, 1e-9), w

    age_o, w_o = kind_age(ocean_m)
    age_c, w_c = kind_age(1.0 - ocean_m)
    take_c = cont_c & (w_c > 1e-6)                      # a continental cell with no continental
    take_c |= ~cont_c & (w_o <= 1e-6)                   # crust near it falls back to the other
    crust_age = FaceField.from_interior(coarse, np.maximum(np.where(take_c, age_c, age_o), 0.0).astype(np.float32),
                                        name="crust_age")
    return {
        "bedrock": bedrock,
        "uplift": uplift,
        "hardness": hardness,
        "plate_id": plate_id,
        "plate_vel": plate_vel,
        "collision_zone": zone_f,
        "crust_kind": crust_kind,
        "crust_age": crust_age,
        "heat": heat_c,
        "_land_slope": slope_info,
        "_scale_m_per_unit": scale,
        "_sea_level_units": q,
        # the margin ramp's footprint: share of tect cells raised and the
        # median raise (m at the final scale); 0 / 0 with the knob off
        "_margin_raised_fraction": float((lift > 0).mean()),
        "_margin_lift_median_m": float(np.median(lift[lift > 0]) * scale) if (lift > 0).any() else 0.0,
        # which kernel rasterised the base height: the neighbour count of the
        # base blend (splat_knn when off) and the share of the bed that came
        # from the wide blend rather than the narrow excess term
        "_splat_knn_base": int(base_blend.nb.shape[1]) if base_blend is not None else int(blend.nb.shape[1]),
        "_splat_base_fraction": _base_fraction(seg, h, tp, blend, base_blend),
        # the reconstruction kernel and, for 'wendland', the share of tect
        # cells whose first kNN list already held the whole support (the
        # rest were re-queried wider; 1.0 for the other kernels)
        "_splat_kernel": str(blend.kernel),
        "_splat_support_covered": float(blend.support_covered),
    }


def _base_fraction(seg, h, tp, blend, base_blend) -> float:
    """|base| / (|base| + |excess|) of the rasterised bed: 1.0 when every
    segment sits at or below the nominal belt height, 0 with the knob off
    (nothing was split)."""
    if base_blend is None:
        return 0.0
    kind = (seg.kind == CONTINENTAL).astype(np.float64)
    h_belt = tp.continental_thickness * tp.belt_thickness * (1.0 - tp.continental_density)
    h_base = np.where(kind > 0, np.minimum(h, h_belt), h)
    b = float(np.abs(base_blend(h_base)).sum())
    e = float(np.abs(blend(h - h_base)).sum())
    return b / max(b + e, 1e-300)


# --------------------------------------------------------------------------
# stage entry points
# --------------------------------------------------------------------------
def run(store: WorldStore, params: WorldParams, log=print) -> dict:
    from ..viz import frames as vf

    t0 = time.time()
    tp = params.tectonics
    frames: list = []
    # viewer timeline frames (hash-exempt: capture only reads the sim).  A
    # rerun of tectonics owns every frame downstream of it, erosion's too.
    vf.clear_all(store.root)
    n_view = int(params.render.tectonics_frames) if params.render.viewer else 0
    rec = vf.FrameRecorder(store.root, "tectonics", min(int(params.render.frame_res), int(tp.N_tect)))
    n_frames = max(int(tp.animate_frames), n_view)

    def on_frame(s, i):
        if int(tp.animate_frames) > 0:
            frames.append(frame_image(s, int(tp.animate_width)))
        if n_view > 0:
            vf.tectonics_frame(s, rec, i, int(tp.steps))

    if n_frames > 0:
        sim = initialise(params, log)
        sim.run(int(tp.steps), log=log, on_frame=on_frame, frames=n_frames)
    else:
        sim = simulate(params, log)
    t_sim = time.time() - t0
    if frames:
        out_path = store.root / "quicklook" / "tectonics.webp"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        step_ms = max(20, int(round(1000.0 / max(float(tp.animate_fps), 0.1))))
        # hold the last frame: the loop is not cyclic, so a hard cut from a
        # finished world back to bare crust reads as a glitch
        dur = [step_ms] * (len(frames) - 1) + [max(step_ms, 1500)]
        frames[0].save(str(out_path), save_all=True, append_images=frames[1:],
                       duration=dur, loop=0, format="WEBP", quality=88, method=4)
        log(f"[tectonics] animation -> {out_path} ({len(frames)} frames, {out_path.stat().st_size / 1e6:.1f} MB)")
    out = finalise(sim)
    for name in OUTPUTS:
        store.save_field(out[name])
    # diagnostics (not part of the hashed outputs)
    diag_dir = store.root / "diagnostics"
    out["heat"].save(diag_dir)
    out["collision_zone"].save(diag_dir)
    out["crust_kind"].save(diag_dir)
    out["crust_age"].save(diag_dir)
    last = sim.stats[-1] if sim.stats else {}
    info = {
        "segments_final": int(sim.seg.M),
        "plates_alive": int(sim.plates.n_alive()),
        "steps": int(sim.step_index),
        "spacing_rad": sim.spacing,
        "scale_m_per_unit": float(out["_scale_m_per_unit"]),
        "sea_level_units": float(out["_sea_level_units"]),
        "margin_raised_fraction": float(out["_margin_raised_fraction"]),
        "margin_lift_median_m": float(out["_margin_lift_median_m"]),
        "splat_knn_base": int(out["_splat_knn_base"]),
        "splat_base_fraction": float(out["_splat_base_fraction"]),
        "splat_kernel": out["_splat_kernel"],
        "splat_support_covered": float(out["_splat_support_covered"]),
        "collisions_total": int(sum(s["collisions"] for s in sim.stats)),
        "spawned_total": int(sum(s["spawned"] for s in sim.stats)),
        "final_mass": float(last.get("mass", 0.0)),
        "sim_seconds": t_sim,
        "seconds_per_step": t_sim / max(sim.step_index, 1),
        "belts": dict(sorted(sim.belt_census.items())),
        "bedrock_max_m": float(out["bedrock"].interior.max()),
        "bedrock_min_m": float(out["bedrock"].interior.min()),
        "uplift_max": float(out["uplift"].interior.max()),
        "uplift_min": float(out["uplift"].interior.min()),
        **out["_land_slope"],
    }
    med, hard = info["land_slope_median"], float(params.erosion.talus_slope_hard)
    if med is not None and med > hard:
        log(
            f"[tectonics] WARNING: median land slope {med:.2f} exceeds erosion.talus_slope_hard {hard:.2f} — "
            f"the bedrock already stands at the talus angle; lower tectonics.relief_spacings ({params.tectonics.relief_spacings}) "
            f"or relief_m ({params.tectonics.relief_m})"
        )
    log(f"[tectonics] done: {info}")
    return info


def quicklook(store: WorldStore, params: WorldParams, path) -> Path:
    """``tectonics.png``: hypsometric hillshade of bedrock with plate
    boundaries (dark red); ``tectonics_plates.png``: plate ids with the
    plate velocity field; ``tectonics_uplift.png`` and
    ``tectonics_hardness.png``."""
    from ..viz import quicklook as ql

    grid = params.coarse_grid()
    bed = store.load_field("bedrock", grid)
    pid = store.load_field("plate_id", grid)
    img = ql.render_height(bed, 0.0, grid.cell_size_m)
    img = ql.contour_lines(img, pid, color=(180, 20, 20))
    out = ql.save_image(path, img)
    base = ql.render_labels(pid, seed=3)
    if store.has_field("plate_vel"):
        vel = store.load_field("plate_vel", grid, is_vector=True)
        base = ql.render_vector(vel, stride=max(4, grid.N // 16), scale=1.0, base=base)
    base = ql.contour_lines(base, pid, color=(255, 255, 255))
    ql.save_image(store.quicklook_path("tectonics", "plates"), base)
    if store.has_field("uplift"):
        up = store.load_field("uplift", grid)
        a = up.interior.astype(np.float64)
        m = max(float(np.percentile(np.abs(a), 99)), 1e-9)
        ql.save_image(store.quicklook_path("tectonics", "uplift"), ql.render_scalar(a, -m, m, cmap="bwr"))
    if store.has_field("hardness"):
        ql.save_image(store.quicklook_path("tectonics", "hardness"), ql.render_scalar(store.load_field("hardness", grid), 0.0, 1.0, cmap="gray"))
    return out


__all__ = ["OUTPUTS", "TectonicSim", "initialise", "simulate", "finalise", "run", "quicklook"]
