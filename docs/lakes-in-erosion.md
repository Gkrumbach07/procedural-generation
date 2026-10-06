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

**`derive.lake_agree_cells` (4).** The refine pass floods its own surface to
the spill point and caps the level at the coarse one only where the coarse
grid *has* a lake. Where hydro's balance found none -- a closed basin
evaporated down to a small lake, a marsh, a lake the fill had laid dry -- the
whole basin stood full again on the refined grid: earth-v23 had a lake of
425,000 km2 on an arid plateau whose coarse lake was 80,000, and 276,000 km2
of water 7 m deep where the coarse grid had marsh. A refined lake now stays
where its coarse cell is a lake or touches one, and elsewhere only as a pond
of at most four coarse cells (`lakes.agree_with_coarse`, derive and the
viewer alike).

| the lakes the viewer draws (refined grid), seed 1423 | % of land | over 100,000 km2 | largest |
|---|---|---|---|
| earth-v19 | 3.76 | 4 | 485k |
| earth-v21: marsh, agreement | 1.88 | 2 | 163k, 136k |
| earth-v22: and the fill | 1.72 | 0 | 85k |
| earth-v23: and the fill by distance | **1.53** | 1 | 142k, then 71k |

**`erosion.lake_fill`.** A particle entering a lake drops what the shore
has room for -- on a plain, a metre a visit -- and carries the rest across
and out, so a lake on flat ground never silts up. With the fill a lake that
has room keeps that load (`lake_load`, at most `lake_room`) and each refresh
lays it over the floor, the side far from the outlet first, up to a plain a
metre over the water (`lake_fill_rise_m`); a frozen lake takes nothing.

| seed 1423, same tectonics | lakes, % of land | marsh | over 100,000 km2 | largest | deeper than 10 m |
|---|---|---|---|---|---|
| **coarse grid** | | | | | |
| earth-v19 (neither) | 3.13 | -- | 4 | 322k | 1.36 % |
| earth-v21 (marsh) | 2.20 | 0.93 | 2 | 212k (frozen), 192k | 1.36 % |
| earth-v22 (marsh and fill) | 2.03 | 0.74 | 3 | 122k, 121k, 106k | 1.16 % |
| earth-v23 (the fill by distance) | **1.79** | 0.75 | 2 | 152k, 147k | 1.00 % |
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

## 6. Below the coarse grid: the planet's lakes, and no others

A level below the coarse grid -- the planet at 1.2 km (`zoom/planet.py`), a
zoom window (`zoom/bake.py`) -- upsamples its parent, adds detail noise and
erodes. None of that is a cause for a lake. On Earth every lake has one that
is recent or still at work -- ice (more lake basins than every other origin
together), a sinking or blocked basin, a dry climate with no way out to the
sea -- because a river-cut landscape drains: a lake silts up or its outlet
cuts down within a geological moment. The planet's stages are where the
model's causes are (tectonics' basins, the glacial pass, the water balance).
What a level's own flood finds are its erosion's dams, the pits of its noise
and the pools of its hold. So a level takes its lakes from its parent
(`zoom/parent_lakes.py`) and treats them as water from start to end.

What was wrong before, measured on earth-v24 at 1.2 km:

* **Square corners, stair edges.** The level's water was its own flood capped
  at the *upsampled* coarse water surface -- not a level surface: 73 % of the
  lake area lay in water bodies whose surface varied by more than a metre --
  then cut to the coarse lake cells and one beside them: 15.7 % of the shore
  ran along coarse-cell lines (12.5 % by chance).
* **Ponds without a cause**, strung along rivers and across plains: 26 % of
  the lake cells.
* **Large lakes in fingers and loops.** `refine.zoom.drain_noise` fills the
  closed depressions of the starting surface "because the upsample of a
  coarser level drains everywhere". It does not: the planet's lakes are
  closed depressions. A 31 m-deep plateau lake had 97 % of its water volume
  laid as ground before the first particle moved; the erosion cut that flat
  by 40 m rms (uncorrelated with the detail noise, r = 0.1); the finish
  flooded whatever pieces lay under the lake's level.

What a level does now:

* **At the start** (`parent_lakes.standing`): the parent's lakes on the plain
  (its surface upsampled), each at its one level. The detail noise is nothing
  under a lake and fades to nothing at the lake's level on the ground within
  two parent cells (`lake_quiet`, the rule the sea has had as
  `refine.coast_taper_m`). `drain_noise` takes sinks: the lakes' water and
  the floor of every basin the planet left dry (`derived["floor"]`, the
  coarse cells no neighbour is lower than), so it fills the noise's pits and
  leaves the planet's basins open.
* **While it erodes**: a lake's cells are not active (the bed stays as the
  parent left it; a particle that ends on one leaves its load on the shore it
  came by) and are flagged, so a particle lays `lake_trap` of its load on the
  shore as it enters. On the same plateau tile the bed under the lakes
  changes by 0.00 m at the 90th percentile (5.8 m at the 99th: the deltas);
  flagged but active, the same cells took +23 m on average, and single dying
  particles speckled the shallows with islands.
* **At the end** (`parent_lakes.level_lakes`): a depression of the level
  holds water only where at least `LAKE_SHARE` (a quarter) of the water it
  would hold lies over the parent's lake cells, at that lake's level or its
  own spill point where its outlet has cut lower. One level surface per
  lake; the shore is the level's own ground meeting it. A lake that runs off
  a block (a tile, a cube face, a window) keeps its level there: the border
  cells it covers drain at the lake's level, not along its bed.

The refined grid (R = 2) has the matching rule in `derive.lakes.agree_with_coarse`:
a refined lake piece is kept whole where a quarter of it lies over coarse
lake cells, and otherwise only as a pond under `derive.lake_agree_cells`
coarse cells; nothing is clipped along coarse cells.

After the finish alone (the first two points; the level's erosion as it was)
earth-v24 at 1.2 km: no water body with a tilted surface, 12.9 % of the shore
on coarse-cell lines (chance), lake cells 2.18 M -> 1.53 M.

**What this does not fix.** A level now shows the planet's lakes as they are,
and some of the large ones are an odd shape on the coarse grid already: long
stripes, a triangle and a block on a sediment plateau at 44 S. The first
reading of that -- rivers running on beds they had raised, the lows between
them flooded, wanting a rule that makes a perched river break out -- did not
survive measurement: of the river cells on plains, 1 % stand more than 5 m
above the lower of the two cells beside them, on the coarse grid and at
1.2 km alike, and none above both. Section 7 has what it was.

## 7. Lakes evaporate in the cold too (`hydro.pet_t0`)

The water balance took its potential evaporation from the climate's `evap`,
`k_evap max(T, 0)`: nothing at freezing, a ninth of the tropics' at 3 C. That
is the kernel's particle decay and its ice line. It is not what open water
loses (mm a year, roughly as measured / `potential_evaporation` / `evap`,
scaled to Lake Chad's 2200):

| lake | latitude, mean T | measured | the law | `evap` |
|---|---|---|---|---|
| Caspian | 42 N, 12 C | ~1000 | 1090 | 940 |
| Titicaca | 16 S, 8 C | ~1700 | 1220 | 630 |
| Nam Co (Tibet) | 31 N, 0 C | ~900 | 750 | 0 |
| Superior | 47 N, 4 C | ~600 | 750 | 310 |
| Baikal | 53 N, -1 C | ~400 | 530 | 0 |
| Great Bear | 66 N, -7 C | ~300 | 280 | 0 |

So nothing dried a cold basin, however little rain it got. earth-v24 had
lakes on 1.83 % of its land with under a quarter of the mean rain and on
1.55 % of the rest -- the dry land the wetter in lakes -- and the plateau
above (3 C, a tenth of the mean rain, tectonic lows over 42 % of it) stood
13 % under water. Its lakes were in 1.4 % of it in earth-v19 and 17 % in
earth-v24: the lake fill is not what keeps them (rerunning earth-v24's last
50 iterations without it leaves more closed lows there, 20.0 against 17.5 %
of the tile, and more on the planet, 2.00 against 1.34 % of land), the two
histories simply differ; no climate dries them in either.

`hydro.pet_t0` > 0 puts the balance -- hydro's, and erosion's own every
`flood_every` (`ErosionState.lake_pet`) -- on Hargreaves' law of temperature
under the latitude's annual insolation,

    (T + pet_t0) / (T_eq + pet_t0) x Q(lat) / Q(0)        (17.8 is Hargreaves' constant)

1 at a sea-level cell on the equator, as `evap` is, so `lake_evap` and
`land_evap` keep their meaning. Cold comes two ways: ground cold for its
latitude has little sun and loses little, and the lake country of the shields
keeps its water; ground cold for its height has the sun of its latitude, and
a dry plateau's basins are pans (Tibet's lakes cover about 2 % of it).

Hydro stage alone on earth-v24's surface, at 17.8:

| lake cover of | `evap` | the law |
|---|---|---|
| all land | 1.66 % | 1.14 % |
| land with under a quarter of the mean rain | 1.83 % | 0.60 % |
| the rest | 1.55 % | 1.47 % |
| cold (T <= 0) and not dry | 4.88 % | 4.74 % |
| cold below 50 degrees of latitude | 4.72 % | 1.67 % |
| cold above 50 degrees | 3.32 % | 2.71 % |
| the plateau | 13.1 % | 0.0 % |

Lakes over 10,000 km2: 26 -> 14; the three largest unchanged. Off by default
until earth-v25 (earth-v24 with it from the start of erosion) is measured.

### The cold follows the ground (`erosion.climate_at_surface`)

The climate runs before erosion, on the tectonic bedrock, and its
`temperature` and `evap` stay on that surface. Erosion's ice line is `evap <=
ice_evap`, so a range the stage takes down a kilometre is glaciated to the end
for the height it no longer has, and the glacial pass, which has no base
level, keeps cutting it. On earth-v24:

* 23.5 % of land is ice ground by the bedrock's temperature, 20.8 % by the
  temperature at the surface the stage ended with;
* 19 % of the ice ground (4.5 % of land) is above freezing at that surface --
  a median 794 m under its bedrock, now at +2.3 C -- and it has the highest
  lake cover of any ground: 5.9 %, against 3.1 % on ice ground still cold and
  1.1 % off it. That is 16 % of all lake area;
* rerun from the iteration-750 checkpoint, the last 50 iterations without the
  glacial pass end with closed lows on 0.88 % of land against 1.34 % with it
  (on the plateau, 10.3 against 17.5 % of the tile): its striped lakes lie
  along tectonic ranges whose crests were ice by their bedrock height, cut
  into troughs and dammed at the ice's margin. The same reruns clear the
  rest: without the lake fill 2.00 %, without the fill's load on the crust
  1.30 %, without isostasy 1.66 %.

With `erosion.climate_at_surface` the erosion state carries the temperature
itself and moves it by the lapse rate to the surface as it stands
(`ErosionState.temperature`): the ice line (`cold`), the lakes that are
frozen and take no fill, and with `pet_t0` the lakes' evaporation all read
it. Hydro and `zoom/ice.py` then read it at the final surface
(`hydro.run.surface_temperature`), as derive's biomes always have.

### Ice needs snow (`erosion.ice_aridity`)

Every cell at or below the ice line was ice. On earth-v24 that is 20.8 % of
land at the final surface, and half of it gets under 0.15 of the mean rain:
on average a cold steppe at -4 C and 56 degrees of latitude -- Siberia and
Tibet, not Canada. Ice sheets grew over wet cold ground and not over those
(north-east Siberia and the Tibetan interior were largely ice-free at the
last glacial maximum for want of snow).

With `erosion.ice_aridity` > 0 cold ground is ice only where its aridity --
the water balance's potential evaporation over its rain
(`hydro.balance.aridity`, on `pet_t0` and `land_evap`) -- is under it. A
polar desert stays ice whatever its rain: at -18 C and under the law
evaporates nothing. On earth-v24's surface, ice on 8.4 / 9.4 / 10.7 / 11.8 /
13.6 % of land at 0.75 / 1 / 1.5 / 2 / 3 (Earth today: 10 %; at the last
glacial maximum about 25 %); earth-v25 has 1.5. The plateau of section 6 has
an aridity of 8 to 9 and no ice at any of these. `zoom/ice.py` reads the
same line.

The three together, rerun from earth-v24's iteration-750 checkpoint for its
last 50 iterations (closed lows over 5 m, share of land / of the plateau
tile; lake cells by erosion's own count):

| | planet | plateau | lake cells |
|---|---|---|---|
| as baked | 1.34 % | 17.5 % | 35,025 |
| `pet_t0` 17.8 | 1.22 % | 7.6 % | 27,144 |
| `climate_at_surface` | 1.11 % | 15.9 % | 28,138 |
| both | 1.14 % | 16.7 % | 24,228 |
| both and `ice_aridity` 1.5 | 1.21 % | 4.2 % | 25,160 |
| no glacial pass at all | 0.88 % | 10.3 % | 27,635 |

The cold following the ground takes the ice off the ranges it had cut down
(their closed lows 3.81 -> 0.62 % of that ground) and puts it on ground the
stage built up into the cold (2.72 -> 6.46 %), which on the plateau is a
desert; the snow rule takes it off the desert again. earth-v25 is earth-v24
with all three from the first iteration.

## 8. The ice's small lakes (`zoom/ice.py`)

What the planet's lakes are made by, on earth-v24's coarse grid (share of
lake area; a cell goes to the first class it fits):

| cause | where it is in the model | share |
|---|---|---|
| ice-carved basin on ground at or below 0 C | `erosion.glacial`, the last quarter of the run | 50 % |
| closed low the tectonic bedrock arrived with | tectonics; 20 % of its land, of which 3.5 % still holds water | 24 % |
| basin held below its rim by evaporation, elsewhere | the water balance | 10 % |
| basin below sea level behind a sill | `open_ocean` | 1 % |
| none of these, overflowing | -- | 14 % |

The census at the large end is close to Earth's (1431 lakes over 100 km2,
351 over 1,000, 26 over 10,000, 1 over 100,000). The small end is missing
altogether: a coarse cell is 95 km2 and a level below it adds no lakes of its
own (section 6), so a cold wet lowland of 1.5 million km2 had 55 lakes at
1.2 km. Earth has a third of a million lakes over a square kilometre, most
of them on ground an ice sheet has crossed: ice quarries the rock under it
wherever the rock is weak and leaves hollows no river could cut and none has
yet drained.

The first level below the coarse grid (the planet at 1.2 km, the first level
of a zoom window; their children inherit the ground) lets the ice finish its
work, after its own erosion:

* **where**: the climate's ice (`evap <= erosion.ice_evap`), tapered in from
  its margin over `erosion.glacial_ramp` coarse cells as the coarse carve is;
  not below the level of a lake of the planet's beside it, not at the sea;
* **how deep**: up to `SCOUR_M` (40 m) times the rock's *grain* -- a ridged
  fractal field on the sphere (seamless across cube faces, the same at every
  level) standing for the fracture spacing the model does not carry, deepest
  along its lineaments -- and `1 - hardness / 2`; no base level;
* **which hollows hold water**: those on ground that was dry before the cut
  (the lows the level's rivers had left are not the ice's), where a lake
  covering 15 % of its own catchment gets what it evaporates by the planet's
  water balance; each at its own spill point
  (`parent_lakes.level_lakes(cut=...)`).

Set against Earth's lake country (the Shield, Finland: about a tenth under
water, N(> A) ~ A^-1, mean depths of 5-15 m) on that lowland at 1.2 km: the
ice's own lakes cover 5.1 % of it in 5,300 lakes (1,840 over 3 km2, 98 over
100), N(> A) ~ A^-0.88 from 3 to 300 km2, mean depth 6.6 m; with the planet's
lakes 8.3 %. The grain's gain (0.7) and base wavelength (20 km) were chosen
for that slope: 0.6 gives 0.82, 0.8 gives 0.92, 30 km gives 0.76.

Not here yet: lakes in glacial *valleys* (a carve by ice flux at 1.2 km cut
hair-thin lakes along every flow line of the level's own rivers, and was
dropped), the elongation of real shield lakes along the ice's flow, ice
ground that is warm today (the model's ice is where it is cold now, 23.5 % of
land, nearer Earth's last glacial maximum than its present ice), crater lakes
and oxbows.
