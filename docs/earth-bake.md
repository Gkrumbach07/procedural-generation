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

**Nothing survives above 5 km.** Every band above 2 km comes out at roughly
half of Earth's, and the highest point on the planet is 5355 m. Two
contributors, and they need separating before either knob is turned:
`orogen_decay = 0.008` left tectonics with a 6619 m maximum (Earth 8849),
and erosion then removed another 1264 m of that.

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

## The continental shelf is arithmetically excluded from deposition

Where eroded material ends up, at the end of the run:

```
deep water (< −200 m)   86.4 %
shelf (−200..0)          4.9 %
coast (0..50)            0.4 %
land (> 50)              8.7 %
```

On Earth the shelves and slopes trap the great majority of terrigenous
sediment and the abyssal plains get comparatively little. Here it is close
to inverted, and the cause looks structural rather than a matter of tuning.

In `erosion/particle.py` the deposition ceiling on the seafloor is defined
*relative to the previous cell on the particle's path*:

```python
ceil = hflat[ref] + sflat[ref] + aflat[ref] + DEP_FLOOR
if s_c < 0.0 and is_step:
    ceil -= DEP_FLOOR + fan_slope   # a fan descends away from its source
```

`fan_slope` is a genuine dimensionless gradient (height in cell units, per
cell), so the shipped 0.05 is a **5 % slope** — continental-slope steep,
where real submarine fans run 0.1–1 %. At 9773 m cells that is 489 m of
descent per cell: one cell offshore the ceiling is at −489 m, two cells at
−978 m. A shelf is ~200 m deep, so `lim = ceil - s_c` is negative there and
**no particle can unload on a shelf at all**. It walks — up to
`ocean_steps = 64`, a 626 km runout over which the ceiling falls 31 km —
until the real seafloor drops below that staircase, which is deep water.

Something near `fan_slope = 0.002` would let a fan run its full 64 cells and
descend ~1.25 km, which is a fan rather than a cliff. **Untested**; the
prediction is that shelf share rises well above 5 %.

Why it matters beyond realism: sediment that leaves the continental margin
can never backfill a valley or build a coastal plain, so base level is never
locally raised and incision never slows. It is a plausible contributor to
the land-planing seen in the pre-glacial regime.

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

## What to measure next

1. **Split the missing 3.5 km of relief** between `orogen_decay` (which cost
   2.2 km before erosion started) and glacial carving (which cost 1.3 km
   after). Re-run tectonics alone at 0.004 and 0.002 to isolate the first;
   vary `glacial_rate` / `glacial_max` on a checkpoint at iteration 600 to
   isolate the second. Do not tune both at once.
2. **Test `fan_slope`.** Fork the erosion state at iteration 800 and run ~25
   iterations at 0.05 and 0.002, comparing the sediment split. Prediction:
   shelf share rises well above 5 %.
3. **Ocean depth** (−2142 m against −3700) — check whether it is sediment
   fill from (2), or the ridge-buoyancy / oceanic-thickness pair.
