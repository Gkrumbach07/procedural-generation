# World scale, and why the continents look wrong

Two separate problems that read as the same symptom ("the continents are
small and each has one mountain ridge"). One is physical scale, the other is
hypsometry. Fixing either alone is not enough.

## 1. The worlds are asteroid-sized

Planet radius is derived, not set: `R_planet = N_c · cell_size_m · 4/(2π)`.

| preset | N_c | cell | **planet radius** |
|--------|-----|------|-------------------|
| `default` | 1024 | 50 m | **32.6 km** |
| `small` | 128 | 50 m | **4.1 km** |
| `final512` (shipped world) | 512 | 50 m | **16.3 km** |

Earth is 6371 km. At `default` the entire planet has ~13,000 km² of surface
and ~4,000 km² of land — about one English county, for the whole world. There
is no room for a continent, so there are no continents.

**Scale is nearly free.** Radius is `N_c × cell_size_m`, so raising
`cell_size_m` grows the world at *identical* cell count and identical compute.
An Earth-radius world at `N_c = 1024` needs 9.8 km cells and costs exactly
what `default` costs today. What you trade is resolution, not time.

## 2. But relief is tied to radius, so scaling alone breaks it

`tectonics/run.py`:

```
target = relief_m if relief_m > 0 else relief_spacings · spacing · R_planet
```

With `relief_m = 0` (the default) relief is a **fixed fraction of planet
radius** — about 0.038·R with 20,000 segments. Measured: raising
`cell_size_m` from 50 m to 5 km grew relief from 263 m to 45 km. Extrapolated
to Earth radius it would ask for ~240 km of relief.

That is not how planets work. Real relief is set by rock strength and gravity
(isostasy), not by radius: Earth is 9 km on 6371 km, i.e. 0.0014·R — 27×
less than this formula gives.

**So any scale-up must set `relief_m` explicitly.** With
`relief_m = 9000` an Earth-radius world comes out at 8995 m of land relief and
30.6 % land, against Earth's 8850 m and 29 %.

## 3. Plate structure: what is established, and what is not

**Established (a structural fact about the model, measured from
`cluster_plates` directly).** Our plates are near-uniform in size at any
count, where Earth's span two orders of magnitude:

| | largest plate | top 7 cover | smallest | span |
|---|---|---|---|---|
| **Earth** (7 major, 8 minor, dozens of micro) | 20.3 % | **92 %** | 0.22 % | **94×** |
| ours, 16 plates | 10.4 % | 56 % | 3.23 % | **3.2×** |
| ours, 48 plates | 3.6 % | 21 % | 1.14 % | **3.2×** |

Earth's seven largest plates cover 92 % of the globe. That is what creates
vast interiors far from any boundary — stable platform — with microplates
scattered between. A uniform tiling cannot do both at once: raising the plate
count shrinks every plate rather than adding small ones beside big ones.

The cause is `plate_size_jitter` (was a hardcoded 0.35 in `cluster_plates`,
now exposed in `TectonicsParams`). It sets per-plate distance weights of
1 ± jitter, so 0.35 gives a 3.2× span. Measured, ~0.8 reaches Earth's ~94×.

**NOT established: that any of this improves hypsometry.** An earlier
revision of this file claimed 48 plates reproduced Earth's land-elevation
distribution almost exactly (71.8 % of land under 1 km against Earth's 71 %).
That was one seed. Across seeds:

| config | per-seed % of land under 1 km | mean ± sd |
|--------|-------------------------------|-----------|
| 16 plates, jitter 0.35 (default) | 50.2, 84.6, 80.1 | **71.6 ± 15.3** |
| 16 plates, jitter 0.60 | 50.0, 55.8, 37.0 | 47.6 ± 7.8 |
| 48 plates, jitter 0.35 | 45.8, 66.3, 57.8 | 56.6 ± 8.4 |

The seed-to-seed spread (±8–15 points) is larger than the gaps between
configurations, so none of these differences is real on this evidence. The
single 71.8 % that looked like a match was the top of its own range.

Note how unstable the ranking is: on two seeds the default averaged 67.4 and
looked worse than 48 plates; the third seed moved it to 71.6, the closest of
the three to Earth, and made raising the jitter look actively harmful. Any
ordering drawn from a handful of seeds here is noise.

At `N_c = 128` with 16 plates the sample is tiny — a handful of plates over
six faces — so one lucky arrangement moves the whole statistic. Settling
whether the hierarchy actually helps needs many seeds, a larger world, or
both. Treat the plate-size span as the thing to fix and hypsometry as the
thing to then measure properly, not as a result already in hand.

## What to change

1. Set `world.cell_size_m` for the planet size you want — free.
2. Set `tectonics.relief_m` explicitly (~9000 for Earth-like) — mandatory
   once the world is large, or relief scales into the hundreds of km.
3. Plate structure is a real defect but **not yet a recipe.** The near-uniform
   plate sizes are established and worth fixing via `tectonics.plate_size_jitter`
   (~0.8 for an Earth-like span); what that does to hypsometry is not measured
   yet, and plate *count* is the wrong knob — the differences it produced were
   inside seed noise.

Untested at the time of writing: whether the erosion parameters, which are
tuned in cell units at 50 m cells, still behave at kilometre cells. Talus
angles are dimensionless and should carry over, but uplift, erodibility and
the discharge scales are all calibrated against the old cell size and will
need re-checking. Nothing here has been run through the full pipeline —
these are tectonics-stage measurements only.
