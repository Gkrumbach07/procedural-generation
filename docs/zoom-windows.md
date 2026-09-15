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

## Item 1 done: the evaporation floor follows the spawn volume

`erosion.min_volume_frac` (5e-5 of the iteration's spawn volume) replaces
`min_volume` (retired, ignored, kept so old manifests load), and
`KERNEL_VERSION` is 10. 5e-5 is the margin the old floor had on earth-v9's
planet grid (spawn volume 0.91, floor 5.1e-5 cell units = 5.6e-5 of it), so
the planet pass is unchanged in practice. `test_evaporation_floor_follows_the_spawn_volume`
steps one window with its rain at 1 and at 1/16384 (R = 128) and requires
the scaled one to walk at least half as far without more evaporation deaths;
with the old floor it walked 2.9 steps against 34 and 97 % of its particles
evaporated.

## Item 2, first prototype: relief that drains, measured

`scripts/drainage_relief.py`, swapped in for the job's detail noise with
`window_bake.py --relief drainage`:

1. perturb the plain upsample with smooth fbm (0.35 x the amplitude);
2. priority-flood it towards the job's own drains (the exit cells and
   the sea -- the first try drained through the frozen divide ring, which
   the job treats as a wall, and ponded 600 lakes against it);
3. accumulate upstream area along the flood tree;
4. carve each cell by a depth rising with log upstream area, to the
   amplitude at one coarse cell of area: a child never has more area than
   its parent or a lower filled level, so the carve cannot make a pit;
5. subtract a Gaussian low-pass (sigma one coarse cell) of the change to
   hold the parent's elevation, then raise any cell below its parent + 1 cm.

Amplitude: 0.1 x elevation + 0.5 x the coarse 3x3 relief, times
(0.5 + 0.5 hardness), faded at the coast.

**Before erosion it drains.** trib at 305 m, 0 iterations: 7 lakes (the
catchment's own) with the job's drift correction off, relief over 3 km
190 / 350 m (p50 / p90) against the plain upsample's 29 / 43. With the drift
correction on, the same surface has **190** lakes: `block_drift`'s bilinear
per-coarse-cell correction cuts pits into a surface that has relief at the
coarse-cell scale (item 4).

**Erosion halves that relief and puts the pits back within five
iterations**, and no knob tried takes the pits out (trib, 305 m, drift
correction off; 0 iterations is 190 / 350 m and 7 lakes):

| erosion | iterations | relief 3 km | lakes / km^2 | pit deaths | ≥100 km^2 channels on the outlet's network |
|---|---|---|---|---|---|
| shipped | 5 | 96 / 152 m | 570 / 674 | 20 % | 100 % |
| shipped | 150 | 37 / 65 m | 100 / 100 | 6 % | 72 % |
| creep 0 | 40 | 108 / 178 m | 995 / 1,267 | 17 % | 85 % |
| creep 0 | 150 | 86 / 145 m | 774 / 669 | 13 % | 75 % |
| creep 0, thermal 0.1 | 150 | 85 / 145 m | 715 / 676 | 14 % | 68 % |
| creep 0, routing flood every iteration | 60 | 107 / 186 m | 933 / 1,285 | 15 % | 49 % |
| creep 0, `iter_deposit` 5 m | 60 | 110 / 183 m | 903 / 1,031 | 22 % | 69 % |
| creep 0, `deposition_rate` 0.02 | 60 | 139 / 229 m | 834 / 1,188 | 21 % | 58 % |
| creep 0, metre caps x 305 / 9773 | 60 | 147 / 274 m | 488 / 469 | 34 % | 25 % |

Creep is what wears the relief away (37 against 86 m after 150 iterations):
it is a diffusion in cell units, so at any cell size it erases features a
few cells wide. Without it the pits stay, and neither refreshing the
routing surface every iteration, nor less deposition, nor caps scaled to
the cell removes them -- so they are not sediment dams the router fails to
see. Where in the kernel they come from is the next thing to find.

**At 76 m** (trib, creep 0, drift correction off, 60 iterations), against the
noise run above:

| | relief 3 km | lakes (median size) | pit deaths | ≥100 km^2 channels on the outlet's network | CPU µs / cell / it |
|---|---|---|---|---|---|
| detail noise 3 | 52 / 121 m | 663 (0.017 km^2) | 62 % | 26 % | 5.8 |
| drainage relief | 69 / 101 m | **10,670** (0.017 km^2) | 37 % | 52 % | 12.6 |

The numbers improve and the picture gets worse. Pit deaths fall and twice
as much of the network reaches the outlet, but the surface is fine parallel
grooves peppered with 1-3-cell lakes: on a smooth regional ramp the flood
tree's flow lines run parallel, and carving each cell by its own upstream
area cuts a one-cell groove along each instead of branching valleys. The
routing noise is too weak to make flow converge, and a valley has no
cross-section -- its depth is a per-cell function of area, not of distance
from a channel. Longer walks (509 steps) also doubled the cost per cell.

### What the next prototype needs

* valleys with a cross-section: depth from distance to the channel network
  at a few stream orders, width growing with upstream area, rather than a
  per-cell function of area;
* routing noise at the coarse-cell wavelength strong enough to make flow
  converge into a branching network before carving;
* the kernel's pit source found (a death-cell census of where particles
  die in pits and what the terrain did there in the preceding iterations);
* a drift correction that does not print pits (item 4), e.g. the Gaussian
  low-pass the relief step already uses.

## Item 3: where the pits come from

`window_bake.py --census N` snapshots the surface before the particle pass,
after it and after the thermal pass, floods the result towards the job's
drains, and classifies every depression that is new that iteration: *dug*
if its bottom went down by more than its spill cell (the lowest rim cell)
came up, *dammed* otherwise, and by which pass (`scripts/pit_census.py`).
trib at 305 m, drainage relief, creep 0, drift correction off:

| iteration | depressions | new | dammed by the particle pass | spill cell raised (median) | new depth p50 / p90 |
|---|---|---|---|---|---|
| 1 | 1,252 | 1,252 | 98 % | 21 m (the bottom rose 9.7 m) | 4.0 / 17 m |
| 2 | 1,437 | 903 | 99 % | 15 m | 5.0 / 29 m |
| 4 | 1,430 | 585 | 97 % | 7.5 m | 7.3 / 45 m |
| 8 | 1,338 | 438 | 92 % | 3.2 m | 10 / 61 m |

**The pits are dams built by the particle pass**: under 3 % are dug, dams
by the thermal pass are 1-6 %, and a depression's rim goes up tens of
metres in a single iteration. Ruled out, each by measurement on the same
census:

| candidate | test | new depressions, iteration 1 |
|---|---|---|
| (baseline) | | 1,252 |
| particles in a chunk depositing against the same frozen terrain | `chunk` 128 / 16 (from 2048) | 1,260 / 1,409 |
| deposits of particles crossing a filled pit | `dep_floor_m` 0.001 (from 1) | 1,324 |
| trunk channels left flat by the relief step (55 % of channels over one coarse cell had gradient < 0.001) | a minimum channel gradient, 0.003 at one coarse cell x area^-0.5 | 1,359 |
| a deposit closing a neighbour's only outlet | a kernel rule forbidding it, in the trace and in `apply_changes` | 622 (but 965 lakes at 60 iterations against 930 without) |
| (repair, not prevention) take back this iteration's deposit at each new dam's spill cell into `pending` | `--breach` | 60 iterations: 488 lakes, 79 % of the ≥100 km^2 network on the outlet |

The no-pit rule halves the first iteration's dams but not the end state
(depressions of a few cells get closed by raises no single-cell check
sees), so it was reverted (`scratch/window/deposit-no-pits.diff`). The
minimum channel gradient stays in `drainage_relief.py`: it is right for a
river profile even though it was not the cause.

What is left is magnitude. At 305 m on relief with 190 / 350 m over 3 km
the particle pass moves tens of metres of material per cell per iteration
off the hillslopes and onto the valley floors -- the per-iteration caps
are metres, which bind hard on the planet grid (9.8 km cells) and barely
at all here -- and a particle that meets the resulting bar climbs for
`pit_steps` and dies, dropping its load into the pit behind it, so a dam
never gets incised (reasoning from the 13-22 % pit-death shares, not traced
particle by particle). The planet pass has machinery for exactly this that
windows do not use: `ErosionState.refresh_lakes` turns every depression
into a lake with a level, particles cross it without touching the bed and
drop their load at its shore, and an overflowing lake spills downstream.
That is the next step: lakes-in-erosion for windows, so a dam behaves like
a lake that fills with sediment and spills, not like a trap.

## Lakes-in-erosion for windows

`erosion.window_lakes` (`ErosionState.refresh_lakes_window`, off by default
so the refine stage is unchanged): every depression the routing flood fills
deeper than `hydro.lake_min_depth` is a flagged, overflowing lake, refreshed
on the route's `flood_every` stride -- particles cross it without touching
the bed and without being killed for climbing, drop 90 % of their load at
its shore, and its bed is never eroded below the water. The level is the
routing surface's own, so lakes and routes agree; there is no evaporation
balance. `test_window_lakes_flag_the_depressions_the_route_fills`.

**On its own it does not stop the dams.** The census with the lakes
refreshed every 10 iterations is identical to without (the first refresh
finds the pit-free relief and the next comes after iteration 8); refreshed
every iteration the depressions grow, 1,839 against 1,304 by iteration 8,
and they are dams just the same (89-98 % by the particle pass, spill raised
30 m in the first iteration -- before any lake exists).

So the size of a deposit is the lever, relative to the cell: a 30 m bar
dams a few cells of a planet channel dropping ~10 m per 9.8 km cell and
~30 cells (9 km) of a 305 m channel dropping ~1 m per cell, while
`iter_deposit` is 50 m at every cell size. 60 iterations, trib, 305 m,
drainage relief, creep 0, drift correction off:

| | relief 3 km | lakes / km^2 | pit deaths | ≥100 km^2 channels on the outlet's network |
|---|---|---|---|---|
| window lakes, refreshed every iteration | 203 / 315 m | 2,027 / 1,079 | 11 % | 67 % |
| `iter_deposit` 1.5 m (= 50 m x 305 / 9773) | 179 / 282 m | 517 / 610 | 21 % | 45 % |
| both | 172 / 273 m | 704 / 370 | 25 % | 31 % |
| both + dam breach | 172 / 271 m | **226 / 114** | 24 % | 37 % |

The last is the first configuration with a lake area in a sensible range
(1 % of the catchment, against 3-9 % for the other rows and 4-17 % for
the earlier creep-0 runs), but the river
network is worse: a quarter of the particles still die in pits and only a
third of the large channels reach the outlet. Two caveats on that number:
the window's routing treats the catchment's divide ring as an outlet (a
cell next to the outside of the mask is a seed), so 66-88 % of particles
leave through the ring rather than the outlet cell, and a channel that
does is counted as disconnected. And the surface is crumpled into parallel
ridges with channels that look broken (`scratch/window/sweep/D_vs_base.png`),
where the shipped erosion's is a smooth fan of converging valleys.

**At 76 m the same configuration trades dams for runaway incision** (trib,
60 iterations, `iter_deposit` 0.39 m = 50 m x 76.4 / 9773, window lakes
refreshed every iteration, dam breach; the other two rows as above):

| | relief 3 km | lakes / km^2 (median size) | pit deaths | exits | ≥100 km^2 channels on the outlet's network (pieces) | steps | CPU µs / cell / it |
|---|---|---|---|---|---|---|---|
| detail noise 3 | 52 / 121 m | 663 / 707 (0.017 km^2) | 62 % | 38 % | 26 % (483) | 268 | 5.8 |
| drainage relief | 69 / 101 m | 10,670 / 1,006 (0.017) | 37 % | 62 % | 52 % (763) | 509 | 12.6 |
| **+ lakes, scaled deposit cap, breach** | **299 / 393 m** | 6,384 / 1,441 (0.052) | 34 % | 24 % | **1 %** (1,862) | **2,810** | **87.9** |

With deposition throttled to a cell-sized cap and erosion's caps left at
12.5 / 25 m, the channels incise without limit -- the relief ends above what
the relief step put there, and the valleys become chains of pools
(`scratch/window/runs/D76_vs_dr76.png`: a densely dissected, water-threaded
surface, the texture of McDonald's maps, on a network in 1,862 pieces).
Particles that no longer die in lakes walk the window instead: 2,810 steps
and 15x the cost per cell, 36 minutes for one 11,500 km^2 catchment.

## Where this leaves the particle kernel at fine cells

Every lever measured moves one failure into another: creep on wears the
relief away; creep off keeps it and the particle pass dams the channels;
lakes let particles past the dams but not through them; a cell-sized
deposit cap stops the dams and lets incision run away, at 15x the cost.
The kernel was calibrated on 9.8 km cells, where its metre caps bind hard
and a bar or a trench is small against a cell, and no single rescaling
found here carries that balance down to 76 m. The two ways forward are a
consistent re-calibration of *all* of its length scales per level (erosion
and deposition caps, cover depth, `route_eps`, the talus and creep terms)
judged on these metrics, or a different model for the fine levels: an
implicit stream-power solver on the flood tree (Braun & Willett 2013's
FastScape scheme -- O(n) per step, unconditionally stable, drainage-
consistent by construction, so no pits and no walk cost), with the
particles' discharge replaced by flow accumulation on the same tree for
the rendered stream map.

## McDonald's own parameters

The post in `docs/ref` gives only its lake flood's numbers (`volumeFactor`
100, stream-map rate 0.01, drainage 0.001, approach 0.5, spill 5) and two
modifiers (friction x (1 - 0.5 stream), evaporation x (1 - 0.2 stream)).
The particle parameters are in his code, SimpleHydrology (current
`master`, read 2026-09-15; `source/water.h`, `world.h`, `cellpool.h`):

| his | value | ours (earth preset) |
|---|---|---|
| `depositionRate` | 0.1 per step | `deposition_rate` 0.1 |
| `evapRate` | 0.001 per step | `evap_rate` 0.001 x `dt` 1.2 |
| `momentumTransfer` | 1 | `k_mom` 1 |
| `lrate` (discharge / momentum EMA) | 0.1 | `ema` 0.1 |
| entrainment | `c_eq = (1 + 10 erf(0.4 q)) x (h - h2)` | `0.2 x drop x (1 + (q / 32)^0.5)` |
| `maxAge` | **500 steps** | `max_steps` 2 x window cells |
| `minVol` | 0.01 of the spawn volume | `min_volume_frac` 5e-5 |
| inertia | full: `speed += gravity n / volume`, no friction | `friction` 0.25 |
| erosion / deposition caps | **none** | 12.5 m / step, 25 / 50 m per iteration |
| cascade | `maxdiff` 0.01 x 80 = 0.8 cell slope, `settling` 0.8 | talus 0.6 / 1.2, rate 0.5, plus creep 0.1 |
| terrain | 8-octave simplex (gain 0.6) in [0, 1] x `mapscale` 80 on a 512-cell tile | the parent's upsample + detail |

His heights are fractions of `mapscale`, so his constants are cell units,
as the kernel's are: erodibility 1.0 and `k_disc` 10 with `erf` entrainment
(`disc_exponent` 0). What does not carry: his discharge is particle volume
per cycle, ours is closer to upstream area, so the `erf` scale is a guess
(`disc_saturation` 32, ours); and his terrain stands up to 80 cells high,
where the 76 m plateau window stands ~4.

"Nick mode" on the kernel: `creep_rate 0, erodibility 1, k_disc 10,
disc_exponent 0, disc_saturation 32, max_steps 500, min_volume_frac 0.01,
friction 0, evap_rate 0.00083, max_erode / iter_erode / iter_deposit /
thermal_max 1e9 (off), thermal_rate 0.8, talus 0.8 / 0.8`. 60 iterations,
drift correction off:

| window | start | relief 3 km | lakes / % of area | pit / age deaths | ≥100 km^2 network on the outlet | CPU µs / cell / it |
|---|---|---|---|---|---|---|
| trib 305 m | detail noise 3 | 86 / 237 m | 930 / 8 % | 22 / 0 % | 81 % | |
| trib 305 m | drainage relief | 126 / 257 m | 1,028 / 8 % | 17 / 1 % | 80 % | |
| trib 305 m | drainage relief + window lakes | 128 / 235 m | 1,502 / 8 % | 13 / 1 % | 83 % | |
| trib 76 m | drainage relief | 143 / 227 m | 12,472 / 6 % | 39 / 39 % | **62 %** | 9.1 |
| trib 76 m | detail noise 3 | 68 / 156 m | 7,203 / 8 % | 50 / 22 % | 18 % | 6.5 |
| range 305 m (5: 216, 928; 174-7,803 m) | drainage relief | 133 / 429 m (upsample 51 / 175) | 1,453 / 5 % | 23 / 7 % | 26 % | 5.4 |
| range 305 m, shipped refine (150 it, detail 0.3, drift on) | | 51 / 168 m | 45 / 0.1 % | 12 / 0 % | 79 % | 4.3 |

Against every earlier configuration, McDonald's settings give the best
river network at both cell sizes on the plateau (80-83 % at 305 m against
31-72 %; 62 % at 76 m against 1-52 %) and keep the cost bounded by
`maxAge` (9.1 µs at 76 m against 12.6-87.9). They are also the first to
cut branching, incised valleys (`scratch/window/nick/nick_cmp.png`) and, on
the mountain catchment, real detail: ridges and gullies off the range and
texture over the lowland, where the shipped refine leaves the upsample
smooth (`mtn_crop_cmp.png`). What they do not fix: 5-8 % of the area is
still small lakes in pits, the plateau still streaks along the regional
slope, the coarse-scale drift grows without the correction (38-62 m
median), and the range catchment is a poor connectivity test -- most of
its steep part is a one-coarse-cell-wide strip.

## The zoom-window setup

`bake/globe/refine/zoom.py` now holds the two things a window below the
planet grid needs, outside the refine stage until zoom windows are one:
`ZOOM_EROSION` (the McDonald settings above) and `smooth_drift`, a drift
correction made of normalised Gaussian low-passes (sigma one coarse cell)
instead of `basin_job.block_drift`'s per-coarse-cell bilinear field, whose
gradient jumps on every block line (`test_smooth_drift_removes_the_coarse_scale_without_the_grid`:
3x the second difference on the node lines for the bilinear field, none for
the Gaussian). `window_bake.py --profile zoom --drift smooth`.

**The drift correction.** On the pit-free relief surface before any erosion
(trib, 305 m): no correction leaves 7 lakes and 34 / 67 m of coarse-scale
drift; the block correction 107 lakes and 0 m; the smooth one 71 lakes and
6 / 17 m. Uncorrected, erosion's own drift grows with the run (30 / 70 m at
60 iterations, 104 / 178 m at 150), so a correction is needed; the smooth one
holds it to 7-9 / 15-25 m and neither changes the river network, which it
follows.

**Longer runs with lakes** (trib, 305 m, zoom profile, drainage relief,
smooth drift, window lakes every 5 iterations):

| iterations | relief 3 km | lakes / km^2 | ≥100 km^2 / ≥10 km^2 network on the outlet (pieces) |
|---|---|---|---|
| 60 | 127 / 235 m | 1,545 / 1,052 | 80 % / 94 % (140) |
| 150 | 85 / 182 m | 1,160 / 632 | 85 % / 85 % (81) |
| 300 | 72 / 168 m | 1,055 / 463 | 75 % / 99 % (99) |

Lake area falls with the run (9 %, 5.5 %, 4 % of the catchment): the pools
fill with the load dropped at their shores. The surface after 150-300
iterations is a branching network converging on the outlet with a scatter
of small lakes (`scratch/window/drift/drift_cmp.png`).

**A compact mountain catchment** (face 5, outlet (239, 913): 220 coarse
cells in a 30 x 27 box, 224 m to 6,979 m, median 3,747 m), 305 m, 150
iterations:

| | relief 3 km (upsample 89 / 199 m) | lakes / km^2 | ≥100 / ≥10 km^2 network on the outlet |
|---|---|---|---|
| shipped refine (detail 0.3, block drift) | 89 / 198 m | 39 / 8 | 57 % / 97 % |
| zoom, drainage relief | 221 / 594 m | 2,835 / 915 | 20 % / 13 % |
| **zoom, detail noise 3** | 177 / 652 m | 2,795 / 1,349 | 54 % / 95 % |

The drainage relief step fails on a steep regional slope: its flood tree's
flow lines run parallel down the range and the carve cuts them into
corduroy -- parallel ridges and grooves with the network in pieces. The
same zoom settings from plain detail noise give mountains: branching
ridges, rounded massifs, valleys with lakes in them, the network mostly
connected (`scratch/window/mtn2/mtn2_crops.png`, shipped / drainage relief
/ noise). On the plateau the two starts tied under these settings (81 and
80 %), so the relief step is not earning its place; the setup is **zoom
profile, detail noise 3, smooth drift, window lakes**. What is left is the
lake count (6 % of the mountain catchment, most of it small pools) and the
coarse drift at the p90 (58 m on the mountain).

## McDonald's later learnings, checked against the kernel

His last hydrology post (*Procedural Hydrology: Improvements and Meandering
Rivers in Particle-Based Hydraulic Erosion Simulations*, 2023-12-12) and his
successor library soillib (`erosiv/soillib`: `source/soillib/model/path/erosion.{hpp,cu}`,
`example/erosion_gpu_multiscale.py`, read 2026-09-15):

| learning | where | the kernel |
|---|---|---|
| thermal avalanching at an angle of repose | 2023 post | has it (`thermal_erosion`, talus by hardness) |
| surface gradient by finite differences, not mesh normals | 2023 post | has it (`_sbilin_grad`) |
| dynamic time step: normalise the speed to one cell per step, so mass cannot tunnel | 2023 post | has it (the step is exactly one cell) |
| discharge = EMA of the volume passing, used as `erf(0.4 q)` | 2023 post | `ZOOM_EROSION` (`disc_exponent` 0); the planet uses a power law |
| momentum map, pushing particles along the stream: the meander mechanism | 2023 post | has it, the same formula (`k_mom cos / (vol + q)`) |
| **lakes removed**: the flood fill was costly and "ill-posed" | 2023 post | `window_lakes` measured the other way here (lake area 9 -> 4 %, network 80-85 %); kept |
| cache-friendly cell struct | 2023 post | n/a |
| **parameters in physical units with the cell's scale in metres**, so "the simulation occurs correctly at scale" | soillib | lengths only (`LENGTH_PARAMS_M`) |
| **multiscale: a long run coarse, then ~4 steps at each finer resolution** with every map (discharge, momentum, suspended mass) resized up | soillib example (128 px x 2048, 256 x 4, 1000 x 4 over 20 km) | windows start from the planet's maps (`discharge0`); the chain of levels is not built |
| **"The erosion system is not permitted to generate a pit, because pits become self-reinforcing and numerically unstable"**: erosion per step at most `0.25 L slope`, deposition at most `0.25 L 0.3` (~0.1 cell) | soillib `mass_transfer` | the kernel caps per particle-step against the next / previous cell; per iteration in metres (off in `ZOOM_EROSION`) |
| particles as transport samples (water, suspended mass, velocity fluxes with exponential attenuation); erosion and deposition computed per cell from wall shear stress `fD rho v^2 / 8`, slope and the suspended mass | soillib | a different model: the kernel exchanges mass along each particle's path |
| two layers, bedrock and sediment, critical slopes 0.57 and 0.3; debris flows / landslides above the critical slope | soillib | one layer with a sediment thickness and hardness |
| a slope boundary condition at the map edge (`exitSlope` 0.02-0.025) | soillib | windows have a frozen ring |
| pigment / albedo carried with the sediment, printing stream history on the surface | both | not yet: a viewer idea |

Two of these were cheap to test on the mountain catchment (305 m, zoom
defaults):

| | lakes / km^2 | ≥100 / ≥10 km^2 network on the outlet |
|---|---|---|
| 150 iterations (the default) | 2,795 / 1,349 (6 %) | 54 % / 95 % |
| **40 iterations** (few fine steps, as the multiscale example) | 3,044 / 2,790 (13 %) | **91 %** / 96 % |
| 150 iterations, `iter_deposit` 32 m (= 0.25 L 0.3 at 305 m) | 2,780 / 1,191 (6 %) | 57 % / 94 % |

Few iterations keep the network the parent routed (91 % on the outlet) but
leave the pools unfilled (13 % lake); a cell-sized deposit cap changes
little with the particle caps already off. Neither is a new default. The
pit-free limits and the multiscale chain are the two worth building: the
first is the soillib answer to the dams this doc spent three sections on,
the second is the zoom pyramid itself.

## The finest level: chained, not direct

The mountain catchment at 76 m with the zoom defaults, straight from the
planet as every earlier window was, against the same level chained from
the 305 m window (`window_bake.py --parent a_R32.npz --parent-R 32`: the
parent's surface and discharge upsampled, detail noise only below the
parent's cell at 0.5 x min(parent slope x 305 m, parent 3x3 relief), the
drift held to the parent at the parent's cell, 40 iterations -- soillib's
multiscale procedure):

| | 305 m, 150 it | 76 m direct, 150 it | **76 m chained, 40 it** |
|---|---|---|---|
| relief over 3 km | 177 / 652 m | 156 / 485 m | 173 / 656 m |
| lakes / km^2 (share) | 2,795 / 1,349 (6.4 %) | 29,108 / 1,631 (7.8 %) | 11,419 / 1,437 (6.8 %) |
| pit / age deaths | 15 / 6 % | 26 / 32 % | 35 / 24 % |
| ≥100 / ≥10 / ≥1 km^2 network on the outlet (pieces at 100) | 54 / 95 / 96 % (149) | 24 / 18 / 15 % (231) | **100 / 75 / 97 %** (3) |
| erosion wall time (CPU µs / cell / it) | 37 s | 1,295 s (8.6) | **322 s** (7.5) |

Direct, the 76 m window has to organise noise from 20 km down to 150 m in
one go and does not: 29,108 lakes, a fifth of the network on the outlet,
the relief lower than the 305 m level's. Chained, it inherits the 305 m
level's valleys and only adds what is below them: the large network is in
3 pieces (100 % on the outlet), the relief is the parent's, the lakes are
fewer than half, and it costs a quarter of the time for 40 iterations
instead of 150 (`scratch/window/mtn2/def76_view_d.png` against
`chain76_view_d.png`, both drawn by `scripts/window_view.py`). The water
it leaves is mostly pools strung along the valley floors.
