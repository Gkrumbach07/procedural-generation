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

## 3. The real cause of "one mountain ridge": hypsometry

Scale fixed, the continents still looked wrong, and hypsometry says why.
Distribution of land elevation, Earth-radius worlds, `relief_m = 9000`:

| land elevation | 16 plates | 48 plates | **Earth** |
|----------------|-----------|-----------|-----------|
| 0–200 m | 13.2 % | 34.0 % | 28 % |
| 200–500 m | 13.5 % | 19.4 % | 24 % |
| 500–1000 m | 17.5 % | 18.3 % | 19 % |
| 1000–2000 m | 24.9 % | 19.3 % | 17 % |
| 2000–4000 m | 22.3 % | 5.5 % | 10 % |
| above 4000 m | 8.6 % | 3.5 % | 2 % |
| **under 1000 m** | **44.2 %** | **71.8 %** | **~71 %** |

At the shipped 16 plates, only 44 % of land is below 1 km and 31 % is above
2 km. The continents *are* the mountain belts: land exists where collision
thickened the crust, and there is no low-lying platform around it. That is
exactly the "one ridge per continent" symptom — the ridge is not on the
continent, it *is* the continent.

At 48 plates the hypsometry lands on Earth's almost exactly. More, smaller
plates give more collision zones, each thickening less, so the land that
emerges includes broad low ground as well as ridges.

Worth noting: judged by eye the 48-plate net looks *blobbier* and less
dramatic than the 16-plate one, which is the opposite of what the numbers say.
The eye is drawn to the ridges; hypsometry counts the ground.

## What to change

1. Set `world.cell_size_m` for the planet size you want — free.
2. Set `tectonics.relief_m` explicitly (~9000 for Earth-like) — mandatory
   once the world is large, or relief scales into the hundreds of km.
3. Raise `tectonics.initial_plates` (16 → ~48) for Earth-like hypsometry.

Untested at the time of writing: whether the erosion parameters, which are
tuned in cell units at 50 m cells, still behave at kilometre cells. Talus
angles are dimensionless and should carry over, but uplift, erodibility and
the discharge scales are all calibrated against the old cell size and will
need re-checking. Nothing here has been run through the full pipeline —
these are tectonics-stage measurements only.
