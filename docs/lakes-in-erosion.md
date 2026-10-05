# One water surface: lakes as erosion state

docs/earth-v3-review.md section 2 found that nothing that drew a river and
nothing that drew a lake shared a water surface: the viewer's rivers were
the erosion stage's particle discharge, its lakes hydro's flood after
erosion, the fine rivers a thresholded fine discharge that ran straight
through every lake, and the fine lakes refine's re-flood of every basin to
its spill point. This is the sequence of changes that made them one thing,
and what each cost. The viewer half (rivers from hydro's flood, coloured by
order) is in the review; the rest is here.

## 1. Lakes inside the particle loop (`erosion.lake_balance`)

McDonald's 2020 post filled pools inside the particle loop and his 2023
model removed them again; docs/earth-v3-review.md has the argument for
why this loop was already half way there (a routing flood every
`flood_every` iterations that particles steer on). The other half:

**`ErosionState.refresh_lakes`** (`erosion/maps.py`), every `flood_every`
iterations after `refresh_base` and `refresh_route`, runs hydro's own
sequence on the eroding surface: `priority_flood_sphere`, D8 on the
filled DEM, and the upstream-first accumulation with the evaporation
balance (`hydro/balance.py`, `hydro.lake_evap`). That gives every
depression a water level. Two kinds come out:

* **Overflowing** (level at the spill point): the cells under water are
  flagged, `S_RFLAG = 2` in the packed sample (no new channel; `NS = 16`
  is one cache line and stays so). The kernel:
  - drops `lake_trap = 0.9` of a particle's load at the shore cell as it
    steps into a lake (`trace_particles`, the land branch): a delta;
  - exchanges nothing with the bed while the particle crosses on the
    routing surface, and does not count the crossing as climbing out of a
    pit;
  - never erodes a flagged cell below its routing surface
    (`apply_changes`), so a lake bed is not cut down but its outlet is,
    and a lake drains itself by incising its sill.
  Discharge and momentum are still recorded across the lake, so the
  river below it keeps its discharge.
* **Closed** (drawn down below the rim): the level goes into `base`, the
  kernel's sea test (`surface < base`) does the rest -- particles entering
  settle their load as a fan and stop, no rain spawns on it, no creep
  crosses it. A Caspian is a sea to the kernel, which is what it is.

Hydro is unchanged: it runs the same flood and balance on the final
surface, so its lakes are erosion's lakes up to the last ten iterations
of change. The flags are checkpointed (`lake_flag`) so a resumed run sees
what the continuous run saw; `KERNEL_VERSION` is 7.

Cost at the `earth` preset: the flood and balance are the ~0.7 s hydro
already spends, every tenth iteration.

## 2. The deposition floor in metres (`erosion.dep_floor_m`)

`DEP_FLOOR` was 0.02 *cells*: 1 m of water kept over any deposit on the
50 m preset the model was tuned on, **195 m** at the `earth` preset, which
forbade deposition in the whole top of the water column -- no delta,
coastal plain or shelf wedge could build (docs/crust-audit.md,
docs/earth-bake.md). It is now `dep_floor_m = 1.0`, converted to cell
units where the kernels take it (particle deposition, `apply_changes`'
ceilings, and mass wasting into submerged cells). This is the coarse-grid
half of the coast fringe in the review's section 3; the refine half
(`refine.coast_taper_m`) fades the detail noise out at the shoreline.

## 3. Refine and derive read the same lakes

* `refine/basin_job.py`: the fine flood inside a basin is capped at the
  coarse water surface (the balance, upsampled) rather than refilling the
  depression to its spill point. `earth-v3` had 510,341 fine lake cells
  against 124,276 coarse ones for that reason.
* `derive/run.py`: rivers are the drainage graph's reaches traced through
  the fine grid (`derive.river_source = graph`), each cut on hydro's lake
  mask and walked to the shore, so a river ends at a shore and starts
  again at the outlet; the older thresholded path (`discharge`, kept for
  comparison) excludes lake cells from its candidate mask instead of
  being skeletonised across the flat and the land bridges between a
  lake's pieces.

## 4. Measured: `worlds/earth-v5`

The first bake with all of this, on the 3000-step tectonics of
docs/plate-forces.md section 7 (63 minutes: erosion 3389 s, refine 330 s).
`earth-v4` is the same pipeline before any of it, on 1500-step tectonics,
so the tectonics differ too; the lake and coast numbers below are still
the ones this change moves.

| | earth-v3 | earth-v4 | **earth-v5** |
|---|---|---|---|
| lakes in erosion at the last refresh: overflowing / closed / dry depressions | -- | -- | 2426 / 58 / 54 |
| coarse lake cells (hydro) | 124,276 | 103,111 | **41,406** |
| of which spill-point fill would have given | 214,423 | 121,978 | 77,918 |
| fine lake cells / 4 (refine, in coarse-cell equivalents) | 127,585 | 95,315 | **40,200** |
| rivers (derive) / max Strahler order | 7976 / 6 | 6041 / 4 | 6565 / 6 |
| mean sediment thickness | 129 m | 59 m | **175 m** |
| sediment on the shelf (−200..0 m by eroded surface), share / concentration | -- | 12.5 % / 0.98x | **56.4 % / 11.4x** |
| sediment on land (> 50 m) | -- | 17.2 % / 1.05x | 10.5 % / 0.41x |
| `lost_offshore` (mass deleted; parked instead since kernel 8, docs/sea-death-stockpile.md) | 1142 Mm | 1052 Mm | **3141 Mm** |

What the numbers say, and what the viewer shows:

* **Lakes are a third of what they were**, because depressions fill
  while erosion runs instead of being discovered full afterwards: the
  delta at the shore is where the load goes, and the bed is not cut. In
  the viewer the big lakes have clean shores, rivers enter them and a
  single river leaves at the outlet, and no river runs across a lake or
  along a land bridge between its pieces. The refine and coarse counts
  now agree to 3 %: the "four times as many fine lake cells" of
  docs/erosion-and-the-sea.md was fine cells against coarse cells, four
  per coarse cell, and never a gap. (The cap is still right, for closed
  lakes.)
* **The shelf receives sediment**: 56 % of it, at 11x the mean
  concentration, against 12.5 % and 1x with the 195 m floor. That is the
  shelf wedge that could not build before, and it is a lot of wedge;
  whether it is too much wants the depth profile across a margin.
* **Two things went the wrong way**, both from the same floor. With the
  ceiling on a step deposit now 1 m above the upstream cell instead of
  195 m, a particle on a flat floodplain or a flat seafloor can place
  almost nothing per visit: land keeps 10.5 % of the sediment instead of
  17 %, and `lost_offshore` -- the load a seafloor walk could not place,
  which `apply_changes` deletes -- triples. The deletion is the defect
  (that mass should go to `pending`, as it does on land), and the 1 m
  floodplain ceiling at 9.8 km cells is a 1e-4 gradient, which is a
  real floodplain but a slow one. The deletion is fixed in kernel 8
  (`erosion.offshore_writeoff`, docs/sea-death-stockpile.md: the
  surplus is parked less a write-off), which also corrects the seafloor
  half of this reading: a seafloor step's ceiling is the previous cell
  minus `fan_slope` whatever the floor, and what strands the load is a
  shelf filled to the floor -- the 1 m floor's part in the tripling is
  the flux it sends to the shelf. The floodplain ceiling is for the
  next erosion pass.
* **The coast fringe is not erosion's.** With the floor in metres the
  spiky light band along every coast is still there, and the
  tectonics-only bakes in docs/plate-forces.md show it before erosion
  has run: it is the continental margin as the segment cloud rasterises
  it -- `margin_taper` thins the outer 35 % of the continent to 45 %
  thickness, that band sits at the waterline, and its outer edge is the
  individual 160 km segments splatted one blob at a time. The review's
  section 3 blamed refine's noise and then erosion's floor; it is the
  tectonics stage's splat. A smoother margin (a wider splat kernel at the
  margin, or a taper in the label map rather than per segment) is the
  fix, in `tectonics/run.py`.

  > **Tried, and off** (`tectonics.margin_sigma_factor = 0`, `margin_ramp`
  > in `tectonics/run.py`): re-positioning the continental/oceanic step
  > with a wide kernel and filling the oceanic side up to it shallowed the
  > sea near every margin without smoothing the edge at equal area. The
  > diagnosis and the instrument (`scripts/coastline.py`) are in
  > docs/coast-fringe.md; the lace at the coastline is the narrow splat
  > resolving per-segment history, still open.

## 5. Sheets of water: marsh, and a lake fill that is off

Measured on earth-v19 by labelling the lake cells themselves (a mask of "cells
at this lake's level" merges lakes that happen to share one, and had given a
mean depth of 571 m for a lake that averages 9): its largest lakes were
sheets. 322,000 km2 averaging 9 m, 220,000 km2 averaging 3 m, both over
basins already holding 500-600 m of sediment, and 30 % of all lake area under
3 m deep (0.93 % of the land). The frozen ones are deeper: 233,000 km2
averaging 48 m.

**`hydro.marsh_depth` (3 m).** A coarse cell is kilometres across, and half a
metre of water over one is a plain tilted by a metre. Standing water no
deeper than `marsh_depth` is marsh: dry in `water_surface`, marked in the
coarse `marsh` field, wetland in derive's biomes; the flood's routing does
not change. Earth's own are the Pantanal, the Sudd, the West Siberian and
Hudson Bay lowlands.

The refine pass floods the same shallow ground again at its own cells, so
the line is drawn on the refined grid too: derive's lakes, its wetland and
the viewer's water take water deeper than `marsh_depth` for a lake. On the
refined grid, which is what the viewer draws, earth-v19 had 3.76 % of its
land under lakes, four of them over 100,000 km2 (the largest 485,000);
earth-v21 has 2.29 %, three over 100,000 km2 (163,000, 136,000 and 119,000).

**`erosion.lake_fill`.** A particle entering a lake drops what the shore
has room for -- on a plain, a metre a visit -- and carries the rest across
and out, so a lake on flat ground never silts up. With the fill a lake that
has room keeps that load (`lake_load`, at most `lake_room`) and each refresh
lays it over the floor, the side far from the outlet first, up to a plain a
metre over the water (`lake_fill_rise_m`); a frozen lake takes nothing.

| seed 1423, same tectonics | lakes, % of land | marsh | over 100,000 km2 | largest | deeper than 10 m |
|---|---|---|---|---|---|
| **refined grid, as the viewer draws them** | | | | | |
| earth-v19 (neither) | 3.76 | | 4 | 485k | |
| earth-v21 (marsh) | 2.29 | | 3 | 163k | |
| earth-v22 (marsh and fill) | **2.26** | | **1** | 117k | |
| **coarse grid** | | | | | |
| earth-v19 | 3.13 | -- | 4 | 322k | 1.36 % |
| earth-v21 | 2.20 | 0.93 | 2 | 212k (frozen), 192k | 1.36 % |
| earth-v22 | 2.03 | 0.74 | 3 | 122k, 121k, 106k | 1.16 % |
| earth-v20 (the fill's first form) | 2.98 | 0.78 | 6 | 280k (frozen), 231k | 1.89 % |

The fill takes the giants more than the area: one seed, and a frozen lake of
212,000 km2 that the fill never touches came out as two of 53,000 km2, so
part of the difference is the run's own scatter.

**What the first form got wrong (earth-v20).** Around earth-v19's largest
lakes it had dried half the cells of the 322,000 km2 one -- whose level then
stood 4 m higher, with 99,000 km2 of new water round it; the frozen
233,000 km2 one stood 36 m higher with 171,000 km2 more. Two things raised
them:

* the plain a lake filled towards rose at `basin_fill_grade` from the outlet
  without limit, 292 m above the water at the far side of a long lake, across
  the mouth of every river entering there -- which is the outlet of the next
  lake up;
* a load a full lake refused became the particle's surplus, which its next
  deposits placed on the cells it crossed in the lake, up to a metre above the
  one before, the outlet among them.

The plain is capped a metre over the water (the marsh rule is what makes that
enough: ground that floods again by less than 3 m is marsh), and a refused
load goes where the particle ends. The deposit loads the crust by 0.55 of its
thickness (`lake_fill_load`): it stands where water stood.

What was left of a half-filled lake in earth-v22 had straight sides, the fill
having gone farthest-first in the flood tree's steps, which across flat water
count cells along the grid; it goes by distance from the outlet now
(earth-v23).

**The quarter-resolution world is not a test of lakes.** It had the first
form halving the deep water (0.88 -> 0.50 % with the marsh rule) where the
full bake doubled the giants.

Three earlier forms were worse still on the small world and are not in the
code: filling to the waterline before there was a marsh rule (ground dead
flat at the spill level floods again with the next tilt); sending a full
lake's surplus on as one `pending` particle (it dams the outlet); filling
from the outlet side (the new plain is the dam: 119,000 km2 at 4.5 m became
244,000 km2 at 20.9 m).

**`hydro.land_evap` (off).** The balance evaporates water off lakes only, and
all the rain a catchment receives arrives at its lake. With `land_evap` the
land evaporates too: a cell's rain runs off in the share Budyko's curve
leaves, and a lake loses the potential evaporation less what the ground
under it would have lost. On finished surfaces, hydro stage alone: earth-v21
2.20 -> 2.05 % of land at 2.5 and 1.93 % at 3.0; earth-v22 2.03 -> 1.90 %
(giants of 170k and 141k) and 1.69 % with no lake over 100,000 km2 at 3.0.
Not monotonic and not baked: erosion solves the same balance every ten
iterations, so a world wants it from the start of the stage.

What is left in earth-v21: one frozen lake of 212,000 km2 averaging 52 m (the
Great Lakes together are 244,000 km2); a tropical one of 192,000 km2
averaging 13 m, its floor below sea level 700 km inland; frozen basins of
30-70,000 km2 up to 350 m deep; and an arid plateau lake of 32,000 km2
(rain 0.08 of the land mean) that a real climate would dry to a pan.

A trap for the next change to the kernel: a particle's slice of the change
list is sized one entry a step (`particle.change_list_cap`). The fill's
entries needed room of their own there; without it earth-v20's first bake
died at iteration 540 with a corrupted heap, after every small world had
passed.
