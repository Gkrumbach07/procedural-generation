# The first complete Earth-scale bake

One end-to-end run of the shipped `earth` preset — 1024² per face, 9773 m
cells, 1500 tectonic steps, 800 erosion iterations — measured at every
stage against Earth. This is the run that the crust-type, craton and orogen
work was building toward, and the first time the pipeline has been taken
past erosion at this scale.

Stage costs on 4 cores: tectonics 247 s, climate 56 s, **erosion 7038 s**,
hydro 3.7 s, watersheds 3.9 s. Refine was stopped deliberately (see below).

## The headline: the low ground is right, the high ground is gone

Elevation bands as a share of land area, against the classic hypsographic
curve:

| band | Earth | after tectonics | after erosion |
|---|---|---|---|
| 0–1 km | 71.6 | 59.7 | **76.2** |
| 1–2 | 15.4 | 27.1 | **19.7** |
| 2–3 | 7.5 | 7.6 | 3.2 |
| 3–4 | 3.8 | 3.5 | 0.7 |
| 4–5 | 1.7 | 1.5 | 0.2 |
| >5 | 0.3 | 0.7 | **0.0** |
| land % of globe | 29.2 | 28 | 27 |
| land mean | 840 m | 980 | 696 |
| land median | ~350 m | 741 | 585 |
| max | 8849 m | 6619 | **5355** |
| ocean median | −3700 m | −2859 | −2142 |
| within ±50 m of sea level | 1–2 % | 3 | 5 |

**The continental interior behaves.** The 1–2 km band was too full after
tectonics (27.1 against 15.4) and erosion moved it almost exactly where it
belongs. That was the prediction made when the orogen decay rate was chosen
— that the excess was crust awaiting erosion rather than an orogen defect —
and it held.

> **Retracted.** That agreement is an artifact. `glacial_rate` and
> `glacial_max` are lengths consumed in cell units and tuned on a 50 m-cell
> preset, so at 9773 m cells the glacial pass carves 195× too hard; what it
> was doing to these bands was hauling the lowlands back up from where the
> fluvial pass had over-planed them, and it happened to stop near Earth's
> figures. Correct the scale and the same run gives 87.9 % in the 0–1 km
> band and a land median of 241 m, against Earth's 71.6 % and ~350 m. The
> fluvial/uplift balance was the defect underneath, and this reading hid it.
> See docs/missing-relief.md.

**Nothing survives above 5 km.** Every band above 2 km comes out at roughly
half of Earth's, and the highest point on the planet is 5355 m. Two
contributors, and they need separating before either knob is turned:
`orogen_decay = 0.008` left tectonics with a 6619 m maximum (Earth 8849),
and erosion then removed another 1264 m of that.

> **Separated, and it was neither of the two.** `orogen_decay` hands over
> exactly Earth's high-ground distribution (13.2 % of land above 2 km
> against 13.3). Of the 10.3 points erosion then removes, the fluvial pass
> and uplift take 9.0 and the glacial pass 1.3. The glacial pass does cost
> 2263 m of the *highest point* in fifty iterations, so which contributor
> looks dominant depends on which statistic you read.
> See docs/missing-relief.md.

**The ocean is too shallow** at a −2142 m median against −3700. Related to
the sediment problem below rather than to anything on land.

## Erosion has two regimes, and the glacial one reverses every trend

`glacial_from = 0.75` starts the glacial pass at iteration 600 of 800. It is
the only pass with no base-level limit — it is *supposed* to carve closed
depressions, that being how lakes are made — and it changes the run
completely:

| iter | land % | mean | median | >2 km | max | ±50 m (land side) |
|---|---|---|---|---|---|---|
| 0 (tect) | 27.8 | 980 | 741 | 13.2 | 6619 | 2.24 |
| 400 | 29.5 | 573 | 254 | 6.9 | 7312 | 6.04 |
| 500 | 29.5 | 525 | 226 | 6.4 | ~7300 | 6.86 |
| **600** | — | — | — | — | — | *glaciation begins* |
| 650 | 26.3 | 578 | 463 | 2.9 | 4900 | 2.55 |
| 700 | 25.7 | 698 | 623 | 3.6 | 5126 | 1.34 |
| 750 | 26.6 | 736 | 666 | 4.0 | 5293 | 1.04 |

Through iterations 150–500 the mean, median and mountain bands all fall and
decelerate, while the land-side ±50 m band grows almost linearly — 3.0 →
6.86 %, which reads unmistakably as the whole landscape being planed to base
level. **Every one of those trends reverses after 600.** Mean and median
climb, and the ±50 m band collapses to 1.04 %, inside Earth's range.

The lesson is procedural rather than geological: *do not extrapolate an
erosion trajectory across iteration 600.* Anything measured before that
point is a pre-glacial transient. This was got wrong in the session that
produced these numbers, with the regime change identified in writing and
then extrapolated straight through anyway.

Glacial carving costs about 4 s/iteration on top of the ~7 s base, and takes
the maximum elevation down 7300 → 4900 in fifty iterations. Whether that is
too strong is the open question above.

## Offshore sediment: a mild bias, and a hypothesis that failed

Where eroded material ends up at the end of the run, **normalised by the
area of each zone** — which is the only way to read it:

| zone | % of globe | % of sediment | concentration |
|---|---|---|---|
| deep (< −200 m) | 65.08 | 86.4 | **1.33×** |
| shelf (−200..0) | 8.33 | 4.9 | **0.59×** |
| coast (0..50) | 1.89 | 0.5 | 0.26× |
| land (> 50) | 24.70 | 8.2 | 0.33× |

Sediment is about **2.3× more concentrated in deep water than on the
shelf**. On Earth the shelves and slopes are the *more* concentrated of the
two, so this is a real bias and worth chasing eventually — but a mild one.

**Read the raw shares and you will overstate this badly.** "86 % goes to the
abyss and 5 % to the shelf" invites the conclusion that the margins are
being bypassed wholesale; in fact deep water is 65 % of the globe, so 86 %
is a 1.33× concentration. The shelf is 8.33 % of the surface here, squarely
inside Earth's 7–8 %, so it is not unusually narrow either. The session that
produced these numbers built three rounds of analysis, a proposed fix and a
hand-off note on the un-normalised version before catching it.

### `fan_slope` is not the cause — tested and refuted

The proposed mechanism was structural. In `erosion/particle.py` the
deposition ceiling on the seafloor is defined *relative to the previous cell
on the particle's path*:

```python
ceil = hflat[ref] + sflat[ref] + aflat[ref] + DEP_FLOOR
if s_c < 0.0 and is_step:
    ceil -= DEP_FLOOR + fan_slope   # a fan descends away from its source
```

`fan_slope` is a genuine dimensionless gradient (height in cell units, per
cell), so the shipped 0.05 is a 5 % slope where real submarine fans run
0.1–1 %. At 9773 m cells that is 489 m of descent per cell, which appeared
to put the ceiling below any shelf within one cell and make `lim = ceil -
s_c` negative there — i.e. deposition on a shelf would be arithmetically
impossible.

Forking the erosion state at iteration 800 and running 25 iterations at each
value says otherwise:

| | shelf share | deep share | land median |
|---|---|---|---|
| `fan_slope = 0.05` | 4.92 → 4.89 | 86.37 → 86.73 | 585 → 540 |
| `fan_slope = 0.002` | 4.92 → 4.86 | 86.37 → **86.80** | 585 → 540 |

A 25× reduction changes nothing; every figure matches to two decimals. The
ceiling is not the binding constraint. Candidates not yet eliminated: the
`is_step` gate may mean most seafloor deposition never takes that branch;
`iter_deposit`, `fan_room` or the fill-to-just-below-sea-level clamp may
bind first; or the distribution may be set by where particles *die*
(`DEATH_OCEAN` after `ocean_steps = 64`) rather than by any ceiling.

The harness is `maps.step` on a forked `ErosionState` — cheap, ~7 minutes
for both arms — and is the right shape for testing the remaining candidates.

## Refine is not runnable as a pre-bake, confirmed at Earth scale

The refine stage reported **9684 basins**, and the first took **109 s** on
one of 4 workers. Even assuming that basin is among the largest and the
average is ten times cheaper, the stage is 6–7 hours; at the observed rate
it is ~73 hours.

That is the same conclusion `docs/pipeline-cost.md` reaches from a cost
model, now measured at scale: **the tile pyramid is not something to
pre-bake.** Basins should be refined ahead of the camera at play time. The
`export_godot.py` globe export reads the coarse grid and needs neither
refine nor tiles, so nothing that has been asked of the pipeline depends on
this stage completing.

## What to measure next — and what was measured

The three questions this document ended on have since been taken up. Their
answers are in **docs/missing-relief.md** (1) and **docs/crust-audit.md**
(2, 3), and they changed the shape of the problem:

1. **Split the missing 3.5 km of relief.** Done, and the split is not the
   one this document guessed. `orogen_decay = 0.008` is *not* the culprit:
   swept against 0.004 and 0.002, the shipped value puts the pre-erosion
   high ground exactly on Earth's post-erosion curve (13.2 % of land above
   2 km against Earth's 13.3). Erosion then removes 69 % of it. Of that,
   the glacial pass is carving in the wrong units — `glacial_rate` and
   `glacial_max` are lengths tuned on a 50 m-cell preset and consumed in
   cell units, so at 9773 m cells one pass takes 848 m off the average
   glaciated cell and 3909 m off the deepest. See docs/missing-relief.md.
2. **Offshore sediment.** The particle census exists now
   (`scripts/fork_erosion.py --deaths`, through a `diag` hook on
   `maps.run_iteration`) and 94.7 % of particles die in the ocean. A
   candidate mechanism turned up on the way: only 2.5 % of the globe is
   drowned continent above −200 m, so the margin is a ramp rather than a
   shelf and there is nowhere for a particle to stop.
3. **Ocean depth.** Not sediment fill, and not the arc plateaus that were
   the obvious suspect (tested: correcting them made the ocean *shallower*).
   Fully subsided sea floor is pinned at −4018 m by
   `(h_ocean − sea_level) × height_scale_m`, which is where Earth's abyssal
   plain starts, and only 22.8 % of the sea floor has got that far.

Also settled by the same work, both in docs/crust-audit.md: the
supercontinent cycle **does** happen — split at step ~400, reassembly by
~600, a second split from ~1200 — but the oceans it opens are only 176 to
256 km wide, so the cycle is invisible to any measurement that calls
anything within 256 km one landmass. And `classify` was building Andean
plateaus out of ocean floor for 44 % of all collisions; it now has an
`island_arc` branch.
