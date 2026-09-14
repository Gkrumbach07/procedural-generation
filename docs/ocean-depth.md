# Ocean depth and the 4-5 km band, on four seeds

`worlds/earth-v8` (seed 0) ends with an ocean median of -2763 m and 0.8 %
of its land at 4-5 km, against Earth's ~-4070 m and 1.7 %. Both gaps sit in
the bedrock (docs/uplift-replay.md), and docs/land-median.md showed seed 0
is a poor judge of the bedrock on its own. Everything here is measured on
seeds 0-3 of the `earth` preset.

Three changes, each for a reason found on the way, not tuned to a number:

| | was | now | why |
|---|---|---|---|
| `tectonics.abyss_depth` | (none) | 0.062 | old sea floor could not sink below the crust-only Airy height, -4018 m |
| `tectonics.ridge_height` | 0.085 | 0.1155 | 3050 m ridge-to-old-floor step, the GDH1 age-depth curve |
| `tectonics.max_crust_thickness` | 2.0 | 2.3 | the cap held the 3-5 km crust ~1 km too low |
| datum hold in shelf mode | `world.land_fraction` | the bedrock's own land fraction | the land area is an output in shelf mode; forcing 30 % re-placed sea level per seed |

## Method

Plate motion never reads the sea-floor terms, so the ocean calibration was
evaluated on pickled end states: `scratch/abyss/pickle_sim.py` runs the
3000-step simulation once per seed (~185 s) and `variants.py` re-runs only
`finalise` with other parameters. The crust cap does change the trajectory
and was run in full for each value and seed.

## Earth's ocean median is -4070 m, not -3700

`scripts/hypsometry.py` carried -3700 m as Earth's ocean median. That is the
ocean's *mean* depth (3688 m). The classic hypsographic curve (% of the
surface: 0..-200 m 5.4, -1000 3.6, -2000 4.0, -3000 6.8, -4000 14.0, -5000
22.8, -6000 12.5, deeper 1.7) puts the median 7 % into the -4000..-5000 m
band, -4070 m, with 52 % of the sea floor below -4000 m. The script now
prints both, and `scripts/ocean_depth.py` uses -4070.

## The sea floor had no abyss

docs/crust-audit.md found the floor: oceanic crust floats at 0.024 bedrock
units against a sea level of 0.1762 on that bake, so fully subsided sea floor sits at
-4018 m -- where Earth's abyssal plain starts -- and the ridge crest at
0.024 + 0.085 - 0.176, -1770 m, 800 m shallower than Earth's. The Airy
column has crust over a uniform mantle; what it lacks is the cooled mantle
lithosphere that makes old sea floor dense, which is what thermal
subsidence *is*.

GDH1 (Stein & Stein 1992) puts the crest at 2600 m and the fully cooled
floor at 5650 m. At the fixed Earth scale (26400 m per unit) that is
`ridge_height = 3050 / 26400 = 0.1155` and an offset that puts the old
floor at -5650: `abyss_depth = 5650 / 26400 - (0.176 - 0.024) = 0.062`.
`ridge_age` stays 400: 300 deepens every seed by another 160-220 m and puts
49-68 % of the sea floor below -4 km.

Bedrock at the tectonics stage's own sea level, crust cap 2.0:

| seed | ocean median, was | **with GDH1** | ocean mean | oceanic crust median | sea floor below -4 km | land bands 1-2 / 2-3 / 3-4 km |
|---|---|---|---|---|---|---|
| 0 | -3225 m | **-4472** | -4162 | -4695 | 64 % | 27.8 / 9.9 / 3.8 (was 28.1 / 10.2 / 3.8) |
| 1 | -2692 | **-3792** | -3461 | -4156 | 44 | 13.8 / 5.2 / 1.8 (14.0 / 5.1 / 1.8) |
| 2 | -2872 | **-4038** | -3691 | -4408 | 51 | 15.9 / 4.3 / 1.0 (16.4 / 4.3 / 1.0) |
| 3 | -2727 | **-3797** | -3486 | -4128 | 44 | 16.6 / 3.3 / 1.4 (16.0 / 3.3 / 1.3) |
| Earth | | ~-4070 | -3688 | ~-4300 (abyssal) | 52 | 15.4 / 7.5 / 3.8 |

The mean over seeds is -4025 m with 51 % below -4 km; the land moves by
0.6 points at most. The calibration was set from GDH1, and the hypsometry
was only read afterwards.

## The 30 % hold was re-placing sea level

In shelf mode (`shelf_fraction` 0.275, the default) tectonics puts sea
level against the continental crust and the land area falls out:
`WorldGroup` says `land_fraction` is honoured "only while shelf_fraction is
0". Erosion and hydro held `world.land_fraction` = 0.30 anyway. Seed 0's
tectonics leaves 24 % land, so the hold lifted the planet ~550 m and
erosion planed the lifted margins to sea level (earth-v8's deep ocean sits
+564 m above its bedrock with no sediment on it); seeds 1-3 leave 34-36 %
and were lowered.

With the sea floor at GDH1 depth this becomes a defect, not a nuisance:
holding 30 % on seed 0 lifted the land by ~1 km and put 19.5 % of it at
0-1 km (Earth 71.6). The hold now keeps the bedrock's own land fraction
in shelf mode (`erosion.maps.datum_land_fraction`, used by the loop, the
replay start and hydro's re-quantile; `KERNEL_VERSION` 9). With shelf mode
off -- the small and tiny presets -- nothing changes.

The hold counts cells (`>= 0`); the tables here are area-weighted, which
reads ~0.8 points lower on this grid. The land fraction therefore varies by seed again, 28-34 % at the new
defaults. It is a property of the seed's collision history
(docs/land-median.md), and it was already there under the old hold.

## The crust cap held the mountains down

Census of the end-state segments, crust cap 2.0, as land bands above sea
level (`scratch/abyss/census.py`):

| seed | 3-4 km: at the cap | 4-5 km: at the cap | craton-flagged | median density |
|---|---|---|---|---|
| 0 | 53 % | 90 % | 80-87 % | 0.84 |
| 1 | 70 | 74 | 81-85 | 0.85 |
| 2 | 12 | 93 | 83-100 | 0.84 |
| 3 | 74 | 100 | 91-98 | 0.85 |

Collision belts are mostly thickened craton (density 0.856, mixed with
belts to ~0.84), not the 0.804 mobile-belt crust the 2.0 cap was reasoned
about. At 0.84, twice normal thickness stands 2.0 x 0.16 x 26400 - 4650 ≈
3800 m above sea level, so the 4-5 km band could only hold crust that was
momentarily over the cap between delamination steps. 2.3 is ~80 km, the
thickest crust on Earth (southern Tibet), and puts that crust at ~5 km.

Full 3000-step runs, bedrock at its own sea level, with GDH1:

| cap | seed | 0-1 | 1-2 | 2-3 | 3-4 | 4-5 | > 5 km | land % | land median | max | ocean median |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Earth | | 71.6 | 15.4 | 7.5 | 3.8 | 1.7 | 0.3 | 29.2 | ~350 m | 8849 m | ~-4070 m |
| 2.0 | 0 | 57.3 | 27.8 | 9.9 | 3.8 | 0.9 | 0.4 | 23.7 | 754 | 6801 | -4472 |
| 2.0 | 1 | 78.7 | 13.8 | 5.2 | 1.8 | 0.5 | 0.1 | 35.8 | 304 | 7321 | -3792 |
| 2.0 | 2 | 78.2 | 15.9 | 4.3 | 1.0 | 0.5 | 0.1 | 34.3 | 258 | 7054 | -4038 |
| 2.0 | 3 | 78.4 | 16.6 | 3.3 | 1.4 | 0.2 | 0.0 | 34.1 | 414 | 4949 | -3797 |
| **2.3** | 0 | 62.5 | 25.9 | 6.9 | 2.7 | 1.3 | 0.7 | 27.5 | 574 | 6745 | -4289 |
| **2.3** | 1 | 75.4 | 14.3 | 5.4 | 3.3 | 1.4 | 0.2 | 34.1 | 319 | 6157 | -3960 |
| **2.3** | 2 | 71.6 | 17.4 | 4.4 | 3.6 | 2.1 | 0.9 | 33.6 | 321 | 6169 | -4422 |
| **2.3** | 3 | 71.2 | 17.4 | 4.4 | 3.4 | 2.2 | 1.4 | 33.6 | 371 | 7083 | -3937 |
| 2.5 | 0 | 62.9 | 25.9 | 5.0 | 2.4 | 1.7 | 2.0 | 27.0 | 523 | 7622 | -4463 |
| 2.5 | 1 | 81.8 | 10.4 | 3.4 | 2.3 | 1.5 | 0.6 | 37.9 | 257 | 6380 | -3850 |
| 2.5 | 2 | 77.2 | 15.7 | 2.8 | 2.1 | 1.7 | 0.6 | 37.0 | 269 | 6060 | -4281 |
| 2.5 | 3 | 74.1 | 17.5 | 3.0 | 2.1 | 1.7 | 1.6 | 32.8 | 310 | 9372 | -3939 |

At 2.3 the 3-4 km band goes from 1.0-3.8 % to 2.7-3.6 % and the 4-5 km band
from 0.2-0.9 % to 1.3-2.2 % on every seed, and seed 0 stops being the
outlier: its land median falls from 754 to 574 m, its 1-2 km band from
27.8 to 25.9 %, and its land fraction rises from 23.7 to 27.5 %. What it
costs: > 5 km overshoots Earth's 0.3 % on three seeds (0.7-1.4 %, before
erosion, which takes the most off the highest ground), and the 2-3 km band
stays thin at 4.4-6.9 % against 7.5. 2.5 thins 2-3 km further and puts 1.6-
2.0 % above 5 km on seeds 0 and 3, with a 9.4 km peak on seed 3.
