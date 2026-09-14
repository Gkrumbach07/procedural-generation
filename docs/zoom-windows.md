# Zoom windows: what the refine machinery does at 1 km, 305 m and 76 m

Phase B of the scale plan (the rivers-and-lakes review of 2026-09-14): before
building nested tiles, measure one window at finer cell sizes -- cost, how
many iterations it needs, whether rivers stay connected, and what the
terrain looks like.

`scripts/window_bake.py` takes the D8 catchment upstream of one coarse cell
of `worlds/earth-v9` and hands it to the refine stage's own basin job
(`refine.basin_job._run_basin`) as a synthetic basin whose only exit is the
outlet, with `world.R` overridden. Nothing in the kernel is changed. `R`
has to be a power of two (the tile pyramid), so the levels are 4.9 km, 1.2
km, 305 m and 76 m.

Two catchments on face 5 of earth-v9, both a plateau at ~1.8-2.0 km with
peaks to 2.3-2.4 km:

* `c439`: outlet (58, 868), 439 coarse cells, ~42,000 km^2, two coarse lakes;
* `trib`: outlet (67, 873), 121 coarse cells, ~11,500 km^2, a tributary of it.

150 iterations unless noted, 20 numba threads, `refine.halo_cells` 2.

## Results

| run | cell | active cells | steps / particle | CPU µs / cell / it | wall µs / cell / it | cores busy | relief over 3 km p50 / p90 (upsample) | lakes / km^2 | ≥100 km^2 channels on the outlet's network | pit / evap deaths |
|---|---|---|---|---|---|---|---|---|---|---|
| c439, detail 0.3 | 4.9 km | 1,532 | 24 | (tiny) | | | | 2 / 167 | 100 % | 0 / 31 % |
| c439, detail 0.3 | 1.2 km | 27,188 | 92 | 2.3 | 0.77 | 3.1 | | 20 / 410 | 100 % | 0 / 27 % |
| c439, detail 0.3 | 305 m | 445,892 | 355 | 7.7 | 1.70 | 4.5 | 24 / 37 m (22 / 36) | 140 / 426 | 81 % | 1 / 21 % |
| c439, detail 3 | 305 m | 445,892 | 304 | 6.8 | 1.55 | 4.4 | 33 / 64 m | 422 / 1,189 | 72 % | 16 / 17 % |
| c439, detail 10 | 305 m | 445,892 | 256 | 5.8 | 1.40 | 4.2 | 67 / 161 m | 487 / **7,117** | 71 % | 39 / 10 % |
| c439, detail 3, **600 it** | 305 m | 445,892 | 370 | 7.9 | 2.06 | 3.8 | 30 / 52 m | 430 / 858 | 77 % | 6 / 16 % |
| trib, detail 3 | 305 m | 122,116 | 136 | 3.2 | 1.01 | 3.2 | 41 / 83 m (29 / 43) | 122 / 252 | 88 % | 10 / 0 % |
| trib, detail 3 | 76 m | 1,975,300 | **3** | 0.46 | 0.37 | 1.2 | 63 / 135 m | 226 / 2,397 | -- | 0 / **100 %** |
| trib, detail 3, `min_volume` 1e-6 | 76 m | 1,975,300 | 268 | 5.8 | 1.49 | 3.9 | 52 / 121 m (28 / 41) | 663 / 707 | **26 %** | **62 %** / 0 |

(Detail = `refine.detail_amp`, shipped 0.3. Relief is max - min in a 3 km
square around each active cell. Channels are cells whose discharge exceeds
the outlet's discharge per km^2 of catchment times 100 km^2; the share is
of those cells in the component that reaches the outlet.)

## What it says

**1. Particle walks grow with the level, so cost per cell does.** On the
same catchment a particle walks 24, 92 and 355 steps at 4.9 km, 1.2 km and
305 m: it crosses the window to the outlet, and the window is R times more
cells across. CPU per cell-iteration rose from 2.3 to 7.7 µs between 1.2 km
and 305 m. A smaller window (trib at 305 m, 640 cells across) walks 136
steps at 3.2 µs. So windows must be fixed-size tiles, not whole
catchments; at a 2,560-cell tile and 76 m it was 268-391 steps and 5.8 µs
CPU.

**2. One process uses ~4 of 20 cores.** Wall time per cell-iteration is
1.4-2.1 µs with 3-4.5 cores busy; the serial `apply_changes` is the floor
(docs/bake-performance.md). Tiles should run one per process, not one
process with more threads. At 5.8 µs CPU, every land cell at 76 m for 150
iterations is ~5,800 core-hours, in line with the phase-D estimate.

**3. `min_volume` is a volume declared as a length (a bug).** A fine cell
spawns rain in proportion to its area (`upsample`: precip / R^2), so the
spawn volume shrinks as 1/R^2, while `min_volume` is converted as a length
(0.5 m / cell size, `config.LENGTH_PARAMS_M`) and shrinks as 1/R. The
margin between them is, by that arithmetic at a land-mean spawn, about
80,000x on the planet grid, 2.4x at 305 m (an evaporation lifetime of ~730
steps; 21 % of c439's particles died that way) and 0.04x at 76 m: every particle died of
evaporation within 3 steps and nothing eroded by water at all. With
`min_volume` 1e-6 the 76 m window runs. The fix is a threshold relative to
the spawn volume, which the planet run already behaves like.

**4. The upsampled surface has no relief below a coarse cell, and erosion
does not make it.** With the shipped detail noise the 305 m surface's
3 km relief is 24 / 37 m against the plain upsample's 22 / 36: the detail is
3.7 m (std). Erosion changes the surface by ~0.13 m per iteration, ~20 m
over the run, and particles on a smooth ramp only print streaks -- this
kernel's capacity is proportional to slope. Real plateau terrain at ~2 km
has a few hundred metres of relief over 3 km.

**5. More noise makes pits, and pits make lakes and break rivers.** Detail
3 and 10 raise the relief (33 / 64, then 67 / 161 m) but the ridged noise
leaves closed depressions that 150 iterations do not drain: at detail 10 a
sixth of the catchment is lake, and 39 % of particles die in pits. At 76 m
with detail 3, 62 % die in pits, the ≥100 km^2 network is 483 pieces and a
quarter of it reaches the outlet. Lakes do form during erosion, as they
should -- 2, 20, 140 and 663 as the cells shrink -- but most are noise pits,
not landforms.

**6. More iterations are not the lever.** 600 iterations instead of 150 at
305 m left the surface changing at the same ~0.2 m per iteration, lowered
the relief (33 / 64 to 30 / 52 m) and moved outlet connectivity from 72 to
77 %. The landscape does not converge; the starting relief's structure
decides what comes out.

**7. The drift correction prints the coarse grid.** `block_drift` removes
the per-coarse-cell mean change and interpolates it bilinearly; with strong
detail the correction is visible as a faint square grid at the coarse-cell
spacing (the detail-10 and 76 m quicklooks).

## What the nested tiles need, in order

1. `min_volume` relative to the spawn volume (3). Small, and nothing below
   ~300 m works without it.
2. Sub-cell relief that already drains: noise whose valleys follow the
   parent's flow network and whose pits are breached (carved to their
   outlet) rather than left to fill (4, 5). This is the unsolved part and
   the one that decides whether the result looks like Nick's.
3. Fixed-size tiles, one per process, with inflow from the parent at tile
   edges so walks stay bounded (1, 2).
4. A drift correction that does not follow the coarse cells (7).

Iteration count (6) and per-level erosion constants come after these: none
of the measured problems were about how long erosion ran.
