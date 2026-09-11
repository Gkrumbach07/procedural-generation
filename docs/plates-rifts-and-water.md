# Four defects the globe viewer showed

Looking at `worlds/earth-full` in the timeline viewer turned up four things
at once: lakes that were not drawn as lakes, inland seas classified as
ocean, ocean plates braided into interleaved ribbons, and rifts that
rearranged the map in a single step.  They are four separate bugs with one
thing in common -- each is a place where a *classification* was being
inferred from the sign of a number instead of from the thing that decides
it.

All numbers below are the shipped `earth` preset (1024² per face, 9773 m
cells, 1500 tectonic steps), seed 0, measured on `worlds/earth-full` and on
tectonics-only re-runs of the same parameters.

## 1. A rift sheared instead of opening, at 33x the speed limit

`intraplate._rift_one` gave the two halves of a rifted plate
`omega = ±n · convection · spacing · rift_speed_factor`, with `n` the normal
of the cut plane.  Two things wrong with that vector:

**The magnitude was not a speed.** `convection` is the force gain of the
model (10.0); the speed cap is `max_speed = 0.3` spacings per step.  A step
runs *rotate, collide, forces*, so the plate moves before `update_omega`
can cap anything: the step that opened a rift turned each half by **14.36°**
against a cap of 0.431°, about 1600 km of crust teleported in one step.

**The axis was wrong.** Rotating about the cut's normal keeps every segment
at its own distance from the cut plane -- the two halves slide *along* the
rift rather than apart.  A spreading pair's Euler pole lies on the rift, 90°
from its middle, so the axis is `com × n`.

The visible cost was a burst of collisions as the zig-zag teeth of the cut
ground through each other:

| | step 400 | step 800 | step 1200 |
|---|---|---|---|
| collisions on the rift step (typical step: ~70) | 895 | 947 | 1475 |
| continent-continent collisions, 50 steps before | 64 | 139 | 278 |
| ... 50 steps after | **602** | **1017** | **1552** |

That is also where "rifting consumes continental crust" in
docs/crust-types.md comes from: land fraction fell 1.4-2.3 points within
25 steps of every rift.

With the pole on the rift and the rate taken from `max_speed`
(`rift_speed_factor` now multiplies the cap, and the halves keep their
parent's motion and add the opening to it), the rift step turns the halves
0.23-0.30° and the collision burst is gone (93/54/105 collisions), with
continent-continent collisions after a rift down to 101/320/632.

## 2. Ocean-on-ocean subduction was decided by rounding error

Within one crust type every segment has the same density -- oceanic crust
is 0.88 everywhere -- and a collision recomputes density as
`mass / thickness`.  So `density[i] > density[j]` in `_apply_collisions` was
comparing the last bits of a division.  Measured over 300 steps: **28.7 %**
of ocean-on-ocean collisions were exact ties (broken by segment *index*) and
the remaining 71 % differed by one ulp; the older segment went down in only
22.6 % of them.

Nothing about that is correlated with which plate a segment belongs to, so
along a trench each plate won about half the contacts.  Subduction has a
polarity -- one plate dives along the whole trench -- and now so does this:
`plate_pair_polarity` takes the mean age of each side's crust along that
boundary and sends the older one down.  Ties go to the higher-numbered
plate, so it stays deterministic.

**This alone did not fix the braiding** (22 plate pieces before, 22 after),
which is what sent the investigation to the real cause below.  It is still
right -- trench polarity should not be a coin flip -- but it was not the
defect.

## 3. A plate cut in two kept rotating as one plate

A plate is a rigid rotation about one pole, and a rigid rotation preserves
distances.  When a trench ate right across a plate, the remaining pieces
kept their separation *for the rest of the run* and swept the planet locked
together, dragging each piece through whatever lay in its path.  Measured
at step 800 on the tect grid:

| plate | pieces (share of its area @ angular separation) |
|---|---|
| 2 | 30 % @ 0° · 24 % @ 49° · 16 % @ 46° · 11 % @ 83° |
| 6 | 44 % @ 0° · 34 % @ **97°** · 10 % @ 47° |

At step 1500 the 10 live plates occupied **22 disconnected pieces**, and the
worst plate's largest piece held 25-49 % of its own area.  That is what the
ribbons in the plate view are: not one plate interleaved with another, but
one plate scattered across the map.

`intraplate.split_disconnected` runs every step: connected components of
each plate's segment cloud (link radius 1.6 spacings), the largest keeps the
plate, any other piece of at least `plate_split_min` (16) segments becomes a
plate of its own inheriting its parent's motion, and smaller fragments are
welded onto the plate around them.

| step | as baked: plates / pieces / largest piece | with splitting |
|---|---|---|
| 250 | 8 / 12 / 55 % | 9 / 9 / 100 % |
| 750 | 7 / 16 / 34 % | 5 / 5 / 99.8 % |
| 1250 | 10 / 21 / 41 % | 8 / 8 / 99.0 % |
| 1500 | 10 / 22 / 49 % | 8 / 8 / 99.3 % |

Every plate is now a single body.  Segment-level interleaving (the share of
segments whose nearest neighbour belongs to another plate) falls from
0.017-0.035 to 0.005-0.010, and the plate count moves the way a
supercontinent cycle should -- 8 at the start, up to 18 mid-run, 8-10 at the
end -- instead of creeping monotonically from 8 to 10.

It costs **10 ms per step** at Earth scale (55 ms/step), about 15 s over a
1500-step run.

Two side effects worth re-measuring after an erosion run: total collisions
over the run fall from 103,090 to 55,203 (plates now stop pushing where they
have been severed), and the tectonic maximum rises from 6604 m to **8783 m**.

### What "the plates all move as one" is *not*

Worth recording, because it was the first hypothesis and it was wrong: the
plates are not co-rotating.  Net rotation (the area-weighted resultant of
every plate's `omega`, over the sum of their speeds) is 0.15-0.20, and the
mean pairwise similarity of their poles is *negative* (-0.1 to -0.4).  The
impression came from the braiding in 3, plus the deliberate fact that the
supercontinent starts as one plate covering ~60 % of the globe.

## 4. Below sea level is not the same thing as ocean

`hydro` took `ocean = surface < 0`, and so did `derive`, `biomes` and the
viewer.  Earth has several million km² of land and inland sea under the
waterline -- the Caspian depression, Qattara, the Dead Sea, Turpan, Death
Valley -- because a rim of higher ground stands between them and the sea.
The same basins arise here whenever plate motion traps a piece of ocean
floor inside a continent or a rift drops a floor below the waterline.

Measured on `earth-full`: **504 enclosed basins, 6.94 M km²** -- 1.36 % of
the globe, 4.53 % of the land -- all classified as ocean.  The largest is
5.84 M km² (78,144 cells) of crust that is 100 % oceanic by `crust_kind`,
with a median surface of -165 m; the second, 290,617 km², sits on
continental crust at a median bedrock of -1016 m.  Both kinds are real
(the South Caspian basin is trapped oceanic crust); neither is sea.

`hydro.open_ocean` now takes the connected components of the below-sea-level
cells and keeps those covering at least `hydro.ocean_min_fraction` (0.02) of
the globe, plus the largest whatever its size.  Everything else is land with
a closed depression in it, which the priority flood already fills to its
spill point.  `flow_dir == OCEAN` is the mask; `watersheds` and `refine`
already read it, and `derive`, `biomes` and the viewer now do too (on the
fine grid, sea is a cell below sea level within one coarse cell of open
ocean -- the coarse mask decides what is connected, the fine surface decides
where the coastline runs).

Re-running hydro on the same erosion output:

| | before | after |
|---|---|---|
| land cells | 30.00 % | 31.46 % |
| closed-basin cells below sea level | (all ocean) | 91,583 |
| lakes | 2134 | 2362 |
| lake cells | 51,419 | **204,806** |
| lake area, share of globe | 0.83 % | 2.69 % |

Carried through refine and derive on the same erosion output
(`worlds/earth-full-water`, hydro onward re-run):

| | before | after |
|---|---|---|
| coarse biome: ocean | 4,404,019 | 4,312,436 |
| coarse biome: lake | 51,419 | **204,806** |
| coarse biome: wetland | 123,091 | 156,691 |
| fine lakes (`graph/lakes.json`) | 2157 | 2405 |
| fine lake cells (refine) | 201,491 | **665,104** |
| rivers | 7296 | 10,132 |

The rivers are the second-order effect worth noting: a closed basin is now
land that drains, so it grows a network running into its lake instead of
being a hole in the map.  Derive costs the same (9.8 s against 10.3 s) —
the fine sea mask is a coarse lookup per cell, not another flood.

## 5. And a lake is above sea level, so it was drawn as ground

The viewer coloured elevation by `h < sea_level`, which misses a lake
entirely: a lake is a filled depression, so its water stands at the *spill
point*, above sea level almost everywhere.  A lake with a visible shoreline
in the biome view was ordinary terrain in the elevation view.

The final viewer frame now carries a `water` channel (0 land, 1 lake, 2
ocean) built from `flow_dir` and `water_surface`, and the shader colours
from it; the timeline frames, which are captured before hydro has run, still
fall back to the height sign because there is nothing else to use.

## What this does not fix

* **Erosion still believes the sign.** `erosion/particle.py` treats any cell
  with `h < 0` as sea (`in_sea`), so a closed basin is still a marine
  sediment sink while erosion runs -- which is why the trapped basin's floor
  sits at a median -165 m against a bedrock -433 m.  The classification is
  now right everywhere downstream of erosion; the physics inside it is not.
  Fixing that means carrying the mask into the particle kernel and re-running
  erosion (~73 min at Earth scale).
* **Lake level is the spill point.** There is no evaporation-limited level,
  so an arid closed basin fills to its rim and overflows instead of settling
  below it.  That is exactly why the Caspian is 28 m *below* sea level, and
  it is the next thing to model if these basins are to read as endorheic
  seas rather than as very large lakes.
* **The tectonics fixes have not been through erosion.** Everything in 1-3
  was measured on tectonics-only runs; a full re-bake is needed before the
  hypsometry table in docs/earth-bake.md can be compared against.
