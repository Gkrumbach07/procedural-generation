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
* The heat field lives on a coarser grid (``N_tect / heat_grid_divisor``).
* ``uplift`` is *metres per erosion iteration*: the per-segment height
  gained since the reference step (``steps - uplift_window``) in metres,
  times ``uplift_scale``, divided by ``erosion.iterations`` — i.e. the
  erosion stage applying it every iteration reproduces the window's
  tectonic uplift over its run.  The difference is Lagrangian (per
  segment, follows the moving crust) so plate translation does not
  register as uplift/subsidence.
* ``plate_vel`` is ``omega × pos`` per step expressed as contravariant
  coarse cell components (coarse cells per tectonic step).
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from ..config import WorldParams
from ..cubesphere import Grid, from_sphere_v
from ..field import FaceField
from ..io.world_store import WorldStore
from ..stubs import fbm_noise
from . import intraplate
from .collision import (
    CellTree,
    SmoothSplat,
    accumulate_area,
    boundary_distance,
    build_tree,
    cell_area_steradians,
    collide,
    crystallise,
    deposit_density,
    gaussian_smooth,
    grid_cascade,
    label_map_fast,
    relax_segments,
    spawn_segments,
    spread_collisions,
    splat,
    weighted_quantile,
)
from .plates import (
    Plates,
    cluster_plates,
    heat_gradient_3d,
    plate_torques,
    random_initial_omega,
    rotate_segments,
    tangent_to_cell_components,
    update_omega,
)
from .segments import Segments, best_candidate_sphere, mean_spacing

OUTPUTS = ["bedrock", "uplift", "hardness", "plate_id", "plate_vel"]


# --------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------
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
        self.idx = None
        self.dist = None
        # mass bookkeeping: total segment mass == initial + spawned + crystallised (collisions/relaxation conserve)
        self.ledger = {"initial": seg.total_mass(), "spawned": 0.0, "crystallised": 0.0, "collision_drift": 0.0}
        # fixed points in the mantle frame; plates drift over them and come
        # out with a track of thickened crust (see tectonics/intraplate.py)
        self.hotspot_pos = intraplate.seed_hotspots(int(self.tp.hotspots), self.params.rng("tectonics", 7))
        self.events: list = []

    # -- helpers ------------------------------------------------------------
    def heat_at(self, pos: np.ndarray) -> np.ndarray:
        return np.clip(self.heat.sample_sphere(pos).astype(np.float64), 0.0, 1.0)

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
            self.events.append(intraplate.rift(self, self.params.rng("tectonics", 6, k)))
            plates = self.plates
        if self.hotspot_pos.shape[0] and tp.hotspot_rate > 0:
            self.events.append(intraplate.apply_hotspots(
                self, self.hotspot_pos, float(tp.hotspot_rate),
                float(tp.hotspot_radius_factor) * self.spacing))

        if k == self.ref_step:
            seg.h_ref = seg.height()
            self.subduction_pts = []

        # 1. move
        mass0 = seg.total_mass()
        rotate_segments(seg, plates)

        # 2. collisions
        tree = build_tree(seg)
        alive = np.ones(seg.M, dtype=bool)
        losers, survivors = collide(seg, tree, self.r_coll, plates.omega, alive, tp.overlap_fraction)
        n_coll = int(losers.size)
        if n_coll:
            spread_collisions(seg, tree, losers, survivors, alive, tp.belt_width_factor * self.spacing)
            pts = seg.pos[losers].copy()
            if k >= self.ref_step:
                self.subduction_pts.append(pts)
            if tp.subduction_heating > 0:
                self._heat_blobs(pts, tp.subduction_heating)
            seg.compress(alive)
            tree = build_tree(seg)
        if tp.relax_rate > 0:
            relax_segments(seg, tree, tp.relax_rate, tp.relax_threshold, self.spacing, int(tp.relax_knn))
        self.ledger["collision_drift"] += seg.total_mass() - mass0

        # 3. label map, areas, gaps
        n_gap = 0
        n_new = 0
        if k % max(1, int(tp.label_every)) == 0 or self.idx is None:
            idx, dist = label_map_fast(seg, grid, self.r_cap, tree)
            accumulate_area(seg, idx, self.area_sr, tp.area_blend)
            new, gap = spawn_segments(seg, idx, dist, grid, self.r_gap, self.r_spawn, rng, self.heat, tp.new_thickness, tp.deposit_density, omega=plates.omega)
            n_gap = int(gap.sum())
            n_new = new.M
            if n_new and tp.gap_cooling > 0:
                self._heat_blobs(new.pos, -tp.gap_cooling)
            if n_new:
                self.ledger["spawned"] += new.total_mass()
                seg.append(new)
            self.idx, self.dist = idx, dist

        # 4. crystallisation
        T = self.heat_at(seg.pos)
        mass1 = seg.total_mass()
        crystallise(seg, T, tp.growth, tp.density_base, tp.deposit_density, tp.dissolution_factor, tp.max_thickness)
        self.ledger["crystallised"] += seg.total_mass() - mass1

        # 5. heat diffusion + slow relaxation towards the background field
        if tp.heat_relax > 0:
            self.heat.data += tp.heat_relax * (self.heat_bg - self.heat.data)
        np.clip(self.heat.data, 0.0, 1.0, out=self.heat.data)
        self._diffuse_heat()

        # 6. forces
        self.heat.exchange_halos()
        grad3 = heat_gradient_3d(self.heat, seg.pos)
        plates.update_stats(seg)
        tau = plate_torques(seg, grad3, plates.P)
        update_omega(plates, tau, self.gain, tp.damping, self.max_omega)

        self.step_index += 1
        spd = plates.speeds()[plates.alive]
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
                    f"({time.time() - t0:.1f}s)"
                )
        return self



def frame_bed(sim) -> np.ndarray:
    """The current crust as a sea-levelled bed on the tect grid.

    Shared by the in-simulation animation capture and `scripts/animate.py`,
    so a frame drawn during the run is identical to one drawn by re-running
    the sim afterwards.
    """
    from .collision import SmoothSplat, build_tree

    tp, grid = sim.tp, sim.grid
    tree = build_tree(sim.seg)
    blend = SmoothSplat(tree, grid, tp.splat_sigma_factor * sim.spacing, int(tp.splat_knn))
    buoy = tp.ridge_height * np.exp(-sim.seg.age / max(float(tp.ridge_age), 1.0))
    bed = _smooth_field(grid, blend(sim.seg.height() + buoy), tp, cascade=True).interior
    sea = float(np.quantile(bed, 1.0 - sim.params.world.land_fraction))
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
    im = Image.fromarray(np.ascontiguousarray(ql.to_net(img)))
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
    T0 = np.clip(heat.sample_sphere(pos).astype(np.float64), 0.0, 1.0)
    density = np.clip(deposit_density(T0, tp.deposit_density), 0.2, 0.95)
    thickness = tp.initial_thickness * (1.0 + 0.2 * (rng.random(M) - 0.5))
    plate_id = cluster_plates(pos, int(tp.initial_plates), rng, size_jitter=float(tp.plate_size_jitter))
    seg = Segments(pos, thickness, density, 0.0, plate_id, 4.0 * math.pi / M)
    plates = Plates(int(tp.initial_plates))
    plates.update_stats(seg)
    random_initial_omega(plates, rng, tp.initial_speed * spacing)
    sim = TectonicSim(params, grid, seg, plates, heat, spacing)
    if log is not None:
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
    blend = SmoothSplat(tree, grid, tp.splat_sigma_factor * sim.spacing, int(tp.splat_knn))
    # thermal buoyancy of young crust: ridges at rifts, subsidence with age
    elapsed = float(sim.step_index - sim.ref_step)
    tau = max(float(tp.ridge_age), 1.0)
    buoy = tp.ridge_height * np.exp(-seg.age / tau)
    buoy_ref = tp.ridge_height * np.exp(-np.maximum(seg.age - elapsed, 0.0) / tau)
    h = seg.height() + buoy
    bed_t = _smooth_field(grid, blend(h), tp, cascade=True)
    dh_t = _smooth_field(grid, blend(seg.height() - seg.h_ref + buoy - buoy_ref), tp, cascade=True)
    bed = _resample(bed_t).astype(np.float64)
    dh = _resample(dh_t).astype(np.float64)
    area = coarse.interior_cell_area.astype(np.float64)
    q = weighted_quantile(bed, area, 1.0 - params.world.land_fraction)
    bed -= q
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
    return {
        "bedrock": bedrock,
        "uplift": uplift,
        "hardness": hardness,
        "plate_id": plate_id,
        "plate_vel": plate_vel,
        "collision_zone": zone_f,
        "heat": heat_c,
        "_land_slope": slope_info,
        "_scale_m_per_unit": scale,
        "_sea_level_units": q,
    }


# --------------------------------------------------------------------------
# stage entry points
# --------------------------------------------------------------------------
def run(store: WorldStore, params: WorldParams, log=print) -> dict:
    t0 = time.time()
    tp = params.tectonics
    frames: list = []
    if int(tp.animate_frames) > 0:
        sim = initialise(params, log)
        sim.run(int(tp.steps), log=log,
                on_frame=lambda s, i: frames.append(frame_image(s, int(tp.animate_width))),
                frames=int(tp.animate_frames))
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
    last = sim.stats[-1] if sim.stats else {}
    info = {
        "segments_final": int(sim.seg.M),
        "plates_alive": int(sim.plates.n_alive()),
        "steps": int(sim.step_index),
        "spacing_rad": sim.spacing,
        "scale_m_per_unit": float(out["_scale_m_per_unit"]),
        "sea_level_units": float(out["_sea_level_units"]),
        "collisions_total": int(sum(s["collisions"] for s in sim.stats)),
        "spawned_total": int(sum(s["spawned"] for s in sim.stats)),
        "final_mass": float(last.get("mass", 0.0)),
        "sim_seconds": t_sim,
        "seconds_per_step": t_sim / max(sim.step_index, 1),
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
