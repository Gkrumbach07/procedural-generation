# The land median: a seed, not a parameter

`worlds/earth-v7` (seed 0) ends erosion with a land median of 695 m
against Earth's ~350 and 19 % of its land at 1-2 km against 15 %. This
note finds where that land sits, tests the obvious lever, and finds the
excess belongs to seed 0's history rather than to the defaults.

## Where the 1-2 km land is (earth-v7, read-only)

| land on | share of land | bedrock p25 / p50 / p75 / p90 | surface p50 |
|---|---|---|---|
| continental crust outside collision zones | 66.8 % | 28 / 239 / 1186 / 1692 m | 631 m |
| continental crust in collision zones | 29.9 % | 415 / 1075 / 2017 / 2971 m | 1175 m |
| oceanic crust (arcs, filled basins) | 3.2 % | -1374 / -850 / -394 / 72 m | 104 m |

The 1-2 km band is 99 % on continental crust and 61 % outside the active
collision zones, so it is mostly not the belts. It is not the
double-counted uplift either: subtracting the applied uplift from the
surface lowers the median only from 695 to 638 m and the band from 19.4
to 18.3 %. The bedrock outside the collision zones has a spike: 11.7 % of
it lies at 1200-1400 m, against 2.7-3.3 % in every neighbouring 100 m
bin. That is `tectonics.orogen_floor_m = 1200`: a dead belt relaxes
towards the `ural` crest height above the continental baseline and stops
there (`orogeny.relax_orogens`), and after 3000 steps at `orogen_decay =
0.004` every dead belt on seed 0 has reached it.

## The floor, swept

Tectonics-only Earth bakes (`--only tectonics`, 194 s each), bedrock
hypsometry:

    python scripts/bake.py --world scratch/floor/floorF_sS --preset earth --seed S --only tectonics --set tectonics.orogen_floor_m=F
    python scripts/hypsometry.py scratch/floor/...

| floor | seed | 0-1 km | 1-2 km | 2-3 km | 3-4 km | 4-5 km | land % | land median | max |
|---|---|---|---|---|---|---|---|---|---|
| Earth | | 71.6 | 15.4 | 7.5 | 3.8 | 1.7 | 29.2 | ~350 m | 8849 m |
| **1200** | 0 | 56.7 | 28.1 | 10.2 | 3.8 | 0.9 | 24 | 786 | 6747 |
| 800 | 0 | 70.9 | 21.3 | 4.7 | 1.7 | 0.9 | 28 | 511 | 6776 |
| 600 | 0 | 78.6 | 16.3 | 3.8 | 1.2 | 0.2 | 30 | 480 | 4600 |
| 400 | 0 | 84.1 | 8.9 | 4.0 | 2.1 | 0.8 | 27 | 349 | 4875 |
| **1200** | 1 | 78.6 | 14.0 | 5.1 | 1.8 | 0.5 | 36 | 264 | 7283 |
| 800 | 1 | 83.6 | 11.0 | 3.7 | 1.2 | 0.4 | 38 | 253 | 5968 |
| 600 | 1 | 74.8 | 13.7 | 5.9 | 3.3 | 1.5 | 33 | 333 | 6071 |
| **1200** | 2 | 77.5 | 16.4 | 4.3 | 1.0 | 0.5 | 34 | 252 | 7200 |
| 800 | 2 | 80.8 | 12.7 | 3.8 | 2.1 | 0.6 | 33 | 322 | 5042 |
| **1200** | 3 | 79.2 | 16.0 | 3.3 | 1.3 | 0.2 | 34 | 340 | 4865 |
| 800 | 3 | 80.5 | 13.4 | 3.4 | 1.9 | 0.7 | 37 | 316 | 5568 |

Seed 0 is the outlier. At the shipped floor, seeds 1-3 already put 14-16 %
of their land at 1-2 km and a median of 252-340 m, on Earth's figures;
lowering the floor to 800 m takes their 1-2 km band to 11-13 %, below
Earth, and the 2-5 km bands move non-monotonically with the floor
because any parameter change sends the simulation down a different
trajectory. The spread between seeds is larger than the effect of the
floor. `orogen_floor_m` stays 1200.

## What the other seeds say instead

Two things are systematic across seeds 1-3 and were hidden by tuning on
seed 0:

* **Too much land**: 34-36 % of the globe against Earth's 29 % (seed 0:
  24 %). The land fraction comes from the continental share of the crust
  and the sea level placement (`shelf_fraction`), not from the floor.
* **Thin mountain bands before erosion**: 2-3 / 3-4 / 4-5 km at
  3.3-5.1 / 1.0-1.8 / 0.2-0.5 % against 7.5 / 3.8 / 1.7. Seed 0's
  well-populated bands (10.2 / 3.8 / 0.9) are its collision history.
  Erosion's uplift adds to them (seed 0: 6.9 / 3.1 / 2.5 after), and with
  that uplift no longer counted twice (docs/uplift-replay.md) the other
  seeds' mountains will be thinner still.

Every tuning in docs/plate-forces.md and docs/uplift-ceiling.md was made
on seed 0 alone. The next tectonics change should be judged on seeds 0-3
together; a tectonics-only run is three minutes, and four run in
parallel on this machine without slowing each other.
