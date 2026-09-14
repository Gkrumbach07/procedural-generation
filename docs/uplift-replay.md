# Uplift replay: the last 100 steps of orogeny, counted once

Tectonics writes two fields the erosion stage starts from. `bedrock` is
the crust at the last step (3000 at Earth scale, 300 on the small preset).
`uplift` is what each segment gained between the reference step `steps -
uplift_window` and the end, divided by `erosion.iterations`
(`tectonics/run.py`, `seg.h_ref`). Until now the stage started from
`bedrock` and added `uplift` every iteration, so the window was in the
surface twice: once in the bedrock and once more over the run.
docs/uplift-ceiling.md measured the result on earth-v5 (top cell 9862 m =
5798 m bedrock + 3162 m of applied uplift; surface minus applied uplift
lands on Earth's 0.3 % of land above 5 km) and added a 2 m/it cap, which
halved the excess (earth-v7: 1.6 % above 5 km, 9195 m). The cap clips a
double count. It does not remove one.

`erosion.uplift_mode` removes it. `'stack'` is the old rule, byte for
byte. `'replay'` starts from the crust as it stood at the reference step,
`bedrock` less the total the stage will apply, and replays the window over
the run, so with no erosion the surface ends at `bedrock`. **Replay is the
default now, and `erosion.uplift_max_m` defaults to 0 (off)**. The
measurements that decided both are below.

## The rule

`maps.uplift_replay(state, ep)` is `iterations x (u_c - mean)`: the field
`apply_uplift` adds each iteration (capped when a cap is on, less its
active-interior mean) times the number of iterations. `maps.start_replay`
subtracts it from the freshly built state, exchanges halos and holds the
datum once (`hold_datum`). `run.build_state` calls it, so
`fork_erosion.py --fresh` gets the same start. `apply_uplift` is unchanged
and shares its field with the replay through `_applied_uplift`, so the
start is lowered by exactly what the run adds back.

What had to be decided:

* **The mean.** The subtraction uses the mean-free field. Subtracting the
  raw `n x u` would give the same end state, because the difference is a
  rigid shift the datum hold removes. But it would push the start
  `n x mean` off the datum, and the first iteration's hold (and
  `datum_drift_m`) would carry it. With the mean-free field the replayed
  total is mass-free like the rule it inverts: the test checks that the
  only mass change over a run without surface processes is the datum's.
* **The cap.** When a cap is on, the start is lowered by the capped field,
  so the no-erosion end state is still `bedrock`. Under replay the cap
  therefore bounds nothing at the end. It only changes the path: capped
  crests start nearer their final height and spend longer under erosion.
* **The sea at iteration 0.** The replayed start is the step-200 (small) /
  step-2900 (Earth) crust. Its land fraction is not `world.land_fraction`:
  it is 0.223 / 0.254 on the small preset, seeds 0 / 1, against 0.297 /
  0.309 for the bedrock. `start_replay` holds the datum on it (a rigid lift of
  21.4 / 17.3 m), so the first `refresh_base` and routing flood see the
  reference-step coastline at 0.30 land, which is what the loop holds from
  iteration 1 on. This turned out not to matter (see "What did not
  matter"). It is kept because it keeps the stage's invariant true from
  iteration 0 rather than from iteration 1. The start hold is reported
  separately as `replay_datum_start_m` and is not part of `datum_drift_m`.
* **Isostasy.** Nothing changes. `iso_acc` accumulates only what surface
  processes move, and the replayed start is not a load. The test runs with
  isostasy on, and with nothing eroding the rebound is exactly zero.
* **Checkpoints and resume.** The start is built once. A checkpoint
  carries the replayed height, and `load_checkpoint` overwrites the
  freshly built one, so a resume never subtracts again. The test checks
  that a resumed run writes the same bytes as a continuous one, and that
  moving `start_replay` after the load fails it. One thing had to change:
  `_ckpt_hash` normalised `iterations` out, so a shorter run could resume
  a longer run's checkpoint. Under replay that would be wrong. The start
  depends on `iterations`, so an 800-iteration state resumed as a
  600-iteration run would end `200 x uplift` below `bedrock`. In replay
  mode `iterations` stays in the checkpoint hash. Stack keeps the old
  normalisation.
* **Windows.** `uplift_replay` returns `None` for a non-spherical state.
  Refine builds windows with `ErosionState.window` (never `build_state`)
  and passes `uplift = 0`, so replay is a no-op there twice over.
* **Hashes.** `uplift_mode` is a new `ErosionParams` field, and
  `uplift_max_m`'s default moved from 2 to 0. Both change
  `group_hash("world", "erosion")`, every world's `params_hash` and every
  erosion checkpoint hash. `bake` refuses an existing world without
  `--force`, and `--force --from erosion` re-runs erosion and everything
  downstream. Tectonics and climate are untouched. `KERNEL_VERSION` stays
  8: the particle kernel did not move. YAML with the new field cannot be
  read by older source. Older YAML loads with the new defaults, and a
  world that should reproduce earth-v7 needs `erosion.uplift_mode: stack`
  and `erosion.uplift_max_m: 2.0`.
* **Stage info.** `uplift_mode` is always reported. In replay mode the
  info also has `replay_lowered_max_m` / `replay_raised_max_m` (the most
  a cell was lowered / raised to build the start),
  `land_fraction_reference` (the start before its hold) and
  `replay_datum_start_m`. The log line reads, for the small preset seed 0:
  `uplift replay: start lowered by up to 502.0 m and raised by up to
  258.9 m to the reference-step crust (land fraction 0.223 before the
  datum hold moved the surface +21.4 m)`. Negative uplift is subsidence
  over the window (trenches, ageing ocean floor), so those cells start
  higher.

## Measured on the small preset

N_c 128, 50 m cells, 300 tectonic steps (reference step 200), 60 erosion
iterations, seeds 0 and 1. Tectonics and climate were baked once per seed
from af58d28. Each arm re-ran only erosion on a copy. On this preset the
window is most of the relief (seed 0: uplift max 8.23 m/it = 494 m over
the run on a 585 m bedrock maximum; seed 1: 13.24 m/it), so it is a harsh
test of any uplift rule. Bands are relative to each seed's own bedrock
maximum, because every absolute band is 0-1 km here.

```
cd bake
python scripts/bake.py --world scratch/replay/head-s$S --preset small --seed $S --to erosion        # af58d28
cp -r head-s$S <arm>-s$S; rm -r <arm>-s$S/checkpoints
python scripts/bake.py --world scratch/replay/stack-s$S      --preset small --seed $S --from erosion --to erosion --force --set erosion.uplift_mode=stack
python scripts/bake.py --world scratch/replay/stackcap0-s$S  --preset small --seed $S --from erosion --to erosion --force --set erosion.uplift_mode=stack --set erosion.uplift_max_m=0
python scripts/bake.py --world scratch/replay/replay-s$S     --preset small --seed $S --from erosion --to erosion --force --set erosion.uplift_mode=replay --set erosion.uplift_max_m=2
python scripts/bake.py --world scratch/replay/default-s$S    --preset small --seed $S --from erosion --to erosion --force     # replay, cap 0
python scratch/replay/measure.py bake <worlds...>
python scripts/hypsometry.py --preset small --checkpoints <worlds...>
```

(The `stack` and `replay` arms above ran before the default moved. At the
time `stack` was the default and 2 m/it the cap. The `--set`s spell out
what each arm was.)

**Stack is byte-identical to af58d28.** The erosion output hashes of the
stack arm equal the af58d28 bake's: `23d9d6136dcab32a` (seed 0) and
`6ad9974de09010b3` (seed 1). The uncapped stack arm on seed 0 reproduces
b45422e's `43aca3e0aec2ba44` from docs/uplift-ceiling.md. The default bake
(`default-s0/1`, no `--set`) reproduces the replay-cap-0 arm's
`6ad2eadd53aa8e9d` / `4bf1309b6bf8ce4a`.

### Seed 0 (bedrock max 584.6 m, bedrock land p99.9 557.9 m)

| arm | max | land > 0.75 x bedmax | > 0.5 x bedmax | > bedrock p99.9 | land p99.9 | land median | ocean median | `lost_offshore` |
|---|---|---|---|---|---|---|---|---|
| bedrock | 584.6 m | 0.94 % | 2.66 % | (0.10) | 557.9 m | 35.0 m | -516 m | |
| stack, cap 2 (af58d28) | 556.4 | 0.414 | 1.56 | 0.000 | 522.6 | 22.9 | -557 | 51,227 m |
| stack, cap 0 (b45422e) | **744.2** | **1.231** | 2.75 | **0.573** | 686.7 | 23.9 | -557 | 61,719 |
| replay, cap 2 | 441.9 | 0.007 | 0.74 | 0.000 | 414.7 | 21.2 | -523 | 33,474 |
| **replay, cap 0 (default)** | **454.6** | **0.061** | **1.10** | **0.000** | 429.8 | 22.0 | -522 | 32,877 |

### Seed 1 (bedrock max 584.0 m, bedrock land p99.9 555.0 m)

| arm | max | land > 0.75 x bedmax | > 0.5 x bedmax | > bedrock p99.9 | land p99.9 | land median | ocean median | `lost_offshore` |
|---|---|---|---|---|---|---|---|---|
| bedrock | 584.0 m | 0.73 % | 2.78 % | (0.10) | 555.0 m | 62.1 m | -731 m | |
| stack, cap 2 (af58d28) | 509.6 | 0.098 | 1.65 | 0.000 | 435.5 | 39.3 | -770 | 45,144 m |
| stack, cap 0 | **889.9** | **1.543** | 3.75 | **0.566** | 739.0 | 41.4 | -770 | 51,611 |
| replay, cap 2 | 419.2 | 0.000 | 0.52 | 0.000 | 351.2 | 38.7 | -716 | 25,645 |
| **replay, cap 0 (default)** | **478.5** | **0.048** | **1.11** | **0.000** | 410.1 | 40.2 | -716 | 28,769 |

Land fraction is 29.9998 % in every arm, because the datum hold holds it.

### Hypsometric bands against the bedrock's own

Share of land, bands in fractions of the bedrock maximum:

| band (x bedmax) | 0-0.05 | 0.05-0.1 | 0.1-0.25 | 0.25-0.5 | 0.5-0.75 | > 0.75 |
|---|---|---|---|---|---|---|
| seed 0 bedrock | 41.4 | 39.3 | 12.3 | 4.34 | 1.72 | 0.94 |
| stack, cap 2 | 58.7 | 26.7 | 9.8 | 3.29 | 1.15 | 0.41 |
| stack, cap 0 | 57.1 | 26.0 | 10.4 | 3.75 | 1.52 | 1.23 |
| replay, cap 2 | 61.2 | 27.5 | 8.1 | 2.52 | 0.73 | 0.01 |
| replay, cap 0 | 59.8 | 27.7 | 8.7 | 2.76 | 1.04 | 0.06 |
| seed 1 bedrock | 22.7 | 24.3 | 43.9 | 6.43 | 2.06 | 0.73 |
| stack, cap 2 | 41.0 | 21.6 | 29.5 | 6.35 | 1.55 | 0.10 |
| stack, cap 0 | 39.6 | 21.1 | 28.8 | 6.75 | 2.21 | 1.54 |
| replay, cap 2 | 41.2 | 22.4 | 32.3 | 3.70 | 0.52 | 0.00 |
| replay, cap 0 | 40.3 | 22.2 | 32.1 | 4.32 | 1.06 | 0.05 |

`scripts/hypsometry.py --preset small --checkpoints` on the same worlds
(iteration 30 / 60, eroded surface):

| | seed 0 max @30 / @60 | land mean @60 | seed 1 max @30 / @60 | land mean @60 | deep-water sediment share, seed 0 (by bedrock) |
|---|---|---|---|---|---|
| stack, cap 2 | 541 / 556 m | 39 m | 566 / 510 m | 60 m | 51.6 % (1.01x) |
| stack, cap 0 | 671 / 744 | 46 | 789 / 890 | 71 | 53.4 % (1.04x) |
| replay, cap 2 | 428 / 442 | 33 | 460 / 419 | 53 | 44.4 % (0.87x) |
| replay, cap 0 | 370 / 455 | 35 | 444 / 479 | 56 | 42.1 % (0.82x) |

### What the numbers say

**The high tail.** Uncapped, stack ends 160-306 m above its own bedrock
maximum, with 0.57 % of land above the bedrock's 99.9th percentile. That is
the double count on this preset. Replay never puts a cell above that
percentile, with or without the cap, and its maximum ends 105-130 m below
the bedrock's. The capped stack also stays under the bedrock here. On the
small preset the cap and the replay both bound the tail; only the replay
does it for the right reason.

**Elsewhere.** Land median: replay at cap 0 is -0.9 / +0.9 m against the
capped stack, which is noise on 22 / 39 m. The ocean median ends within
6 / 15 m of the bedrock's (-522 vs -516, -716 vs -731) instead of ~40 m
below it: the window's subsidence was counted twice on the sea floor as
well. `lost_offshore` falls by a third (51.2 -> 32.9 k m, 45.1 -> 28.8
k m) and the deep-water sediment concentration falls from 1.01x to 0.82x,
so more of the load stays on the shelf. The ledger is unchanged (below).

**The cost.** Replay's upper-middle bands (0.25-0.75 x bedmax) end lower
than both stack arms and lower than the bedrock's: 2.76 / 1.04 % against
the capped stack's 3.29 / 1.15 % (seed 0), and 4.32 / 1.06 against 6.35 /
1.55 (seed 1). Under replay a belt is still being built while erosion
runs, and on this preset the window is most of the belt. The stack arms
sit closer to the bedrock in those bands partly because the double count
tops them up. So this is erosion acting on the right path, not a
regression of the rule. It is still the number to read first on the next
Earth bake (see the estimate below).

**Replay is right, so it is the default.**

### The cap under replay: off

Replay at 2 m/it against 0 m/it, same seeds:

| | cap 2 | cap 0 |
|---|---|---|
| land > 0.5 x bedmax, seed 0 / 1 | 0.74 / 0.52 % | **1.10 / 1.11 %** (bedrock 2.66 / 2.78) |
| land > 0.75 x bedmax | 0.007 / 0.000 | 0.061 / 0.048 (bedrock 0.94 / 0.73) |
| above bedrock p99.9 | 0 / 0 | **0 / 0** |
| max | 442 / 419 m | 455 / 479 m (bedrock 585 / 584) |
| land median | 21.2 / 38.7 m | 22.0 / 40.2 m |

With replay the end state cannot exceed `bedrock` plus rebound and datum,
whatever the field's size. The cap takes a third to a half of the land
above half the bedrock maximum, and the uncapped replay still stays under
the bedrock tail in both seeds. There is nothing left for it to guard, so
**`uplift_max_m` defaults to 0**. It stays available: under `stack` it is
still the only bound, and a replayed run can still set it.

## What did not matter

**The datum hold on the start.** `scratch/replay/nohold.py` drives
`maps.step` exactly as `run.py` does, once from `start_replay` and once
from the same subtraction without the hold. The hold arm reproduces the
default bake (seed 0: 454.6 m, median 22.0 m).

| | start land | max | > 0.5 x bedmax | median | ocean median | `lost_offshore` | loop `datum_drift_m` |
|---|---|---|---|---|---|---|---|
| seed 0, held | 0.300 | 454.6 m | 1.10 % | 22.0 m | -521.9 m | 32,877 m | 25.3 m |
| seed 0, not held | 0.223 | 455.0 | 1.08 | 21.9 | -521.9 | 28,445 | 4.1 |
| seed 1, held | 0.300 | 478.5 | 1.11 | 40.2 | -716.1 | 28,769 | 25.3 |
| seed 1, not held | 0.254 | 477.2 | 1.11 | 40.1 | -716.1 | 36,371 | 8.0 |

The per-cell difference is mean 0.07 / 0.09 m, p1-p99 about ±14 m, and at
most 175 m in a handful of cells. Without the start hold, iteration 1's
hold takes the same lift, which is why the loop drift differs by exactly
the start hold (25.3 - 21.4 = 3.9 m). `lost_offshore` moves ±15 % in
opposite directions across the two seeds, which is noise.

**Time.** Erosion stage, 60 iterations: 12.0 / 12.1 s (stack, cap 2, seed
0 / 1) against 12.1 / 12.8 s (replay, cap 0) in the first batch. A second
batch on seed 0 gave 8.5, 9.0 s (stack) against 9.5, 9.1 s (replay) and
9.3, 9.1 s (replay cap 2). The machine's load moved whole batches by 30 %;
within a batch replay is 0-10 % slower, the size of the noise.
`start_replay` is one subtraction, one halo exchange and one
`np.partition`, done once per run.

## The mass ledger

`scratch/replay/ledger.py` wraps every sub-pass of `maps.step` and sums
its change of `total_mass()`, with `lost_offshore` added back for the
particle pass and `q x M` for the datum hold. The result is relative to
Σ|height|, over 60 iterations:

| pass | stack cap 2, seed 0 / 1 | replay cap 2, seed 0 / 1 |
|---|---|---|
| particles (`run_iteration`) | -6e-16 / -6e-16 | -1e-15 / -2e-16 |
| `apply_uplift` | 2e-15 / -2e-15 | 6e-16 / -9e-16 |
| `apply_isostasy`, `glacial.carve`, `hold_datum` | <= 1.5e-15 | <= 1.5e-15 |
| **`thermal_erosion`** | **-3.0e-5 / -2.7e-4** | **+2.6e-5 / -1.6e-4** |

Every pass closes to rounding except the mass-wasting pass. It leaks
1e-5 to 1e-4 of the column in both modes, with either sign. That leak
predates this change, is the same order under both rules, and is not
investigated here. The end-state ledger read off the checkpoints
(`measure.py`: start mass - M x Σq - `lost_offshore` against the final
height + sediment + pending) agrees: -2.8e-5 / -2.6e-4 (stack) and
+3.2e-5 / -1.5e-4 (replay, cap 0).

## What to expect at Earth scale

This is not a bake. `scratch/replay/estimate.py` takes a baked stack world's
final checkpoint, subtracts the applied total `800 x (u_c - mean)` and
re-holds the datum at 0.30. It ignores the erosion the lower path would
have caused. Calibrated on the small preset against the real replay
bakes:

| | estimate | baked |
|---|---|---|
| replay cap 2, max seed 0 / 1 | 446 / 404 m | 442 / 419 m |
| replay cap 2, > 0.5 x bedmax | 0.65 / 0.34 % | 0.74 / 0.52 % |
| replay cap 0, max | 422 / 380 m | 455 / 479 m |
| replay cap 0, > 0.5 x bedmax | 0.53 / 0.13 % | 1.10 / 1.11 % |
| land median (cap 0) | 19.6 / 31.8 m | 22.0 / 40.2 m |

So the estimate is a **lower bound** on the high tail and on the median.
It is close with a cap. It misses badly without one, because the cores
that rise late in an uncapped replay are the ones erosion never reaches.

Applied to the read-only reference worlds (earth-v7 at its 2 m/it cap;
earth-v5, uncapped, for the cap-0 default); bands are area-weighted % of
land:

| | 0-1 km | 1-2 | 2-3 | 3-4 | 4-5 | > 5 | max | land median | ocean median |
|---|---|---|---|---|---|---|---|---|---|
| Earth | 71.6 | 15.4 | 7.5 | 3.8 | 1.7 | 0.3 | 8849 m | ~350 m | -3700 m |
| earth-v5/v7 bedrock | 56.7 | 28.1 | 10.2 | 3.76 | 0.86 | 0.40 | 6747 | 786 | -3227 |
| earth-v7 (stack, cap 2) | 66.7 | 19.2 | 6.9 | 3.10 | 2.46 | 1.64 | 9195 | 695 | -3195 |
| replay cap 2, estimate | 70.2 | 18.1 | 7.1 | 3.43 | 0.87 | 0.39 | ~7600 | ~637 | ~-2746 |
| earth-v5 (stack, cap 0) | 67.7 | 18.7 | 6.5 | 2.82 | 2.27 | 2.10 | 9862 | 655 | -3247 |
| **replay cap 0, estimate** | 71.0 | 17.7 | 6.8 | 3.34 | 0.80 | 0.36 | ~7470 | ~606 | ~-2804 |

Read with the calibration. The share above 5 km should land near the
bedrock's own 0.4 % (Earth 0.3), where stack left it at 1.6-2.1 %. The
4-5 km band will be above the estimate's 0.8 %, by how much only the bake
can say. The land median is not fixed by this: the estimate is 606-637 m,
the real value higher, against ~350. docs/plate-forces.md section 7's
1-2 km problem is the bedrock's (28 % of its land is at 1-2 km).

**The ocean median gets shallower, by about 450 m, and that is the fix.**
The first version of this paragraph blamed the datum hold. A review
retracted that, measured on earth-v7's final checkpoint with this note's
own estimate:

* the applied total over the ocean has a median of +72 m but is two-sided:
  p5 / p10 / p25 are -702 / -566 / -274 m, 23 % of ocean cells are below
  -300 m, and the cells in stack's -3400..-3000 m band (a quarter of the
  ocean) carry a median of -404 m. That is the window's ageing sea-floor
  subsidence counted twice, the same mechanism as on land and as on the
  small preset's ocean;
* the datum rule does not bring the depth back: re-holding at the
  bedrock's own land fraction (24.6 %) instead of 30 % gives -2764 m
  against -2746 m, because the eroded coastal band between the two
  fractions sits near 0 m by the end of the run;
* a small world in shelf mode, where tectonics leaves 45 % land (the
  opposite mismatch), shows the same shift: replay's ocean ~14 m from the
  bedrock's, stack's ~44 m below it. The shift follows the double count,
  not the land-fraction mismatch.

So earth-v7's Earth-like -3195 m was mostly the double count, and replay's
~-2750 m is the bedrock's own ocean held at 30 % land (-2678 m). What is
left of the gap to Earth's -3700 m belongs to tectonics, in both modes:
the abyss the bedrock builds, and the 30 % hold lifting a surface that
tectonics left at 24.6 % land on seed 0 (a ~550 m lift measured against
the raw bedrock; seeds 1-3 leave 34-36 % land, so the hold lowers them
instead, docs/land-median.md). Expect -2700 to -2800 m on the next Earth
bake of seed 0.

## Measured at Earth scale: `worlds/earth-v8`

The full `earth` bake with replay and the cap off (commit 43577e1, with the
seam blend of docs/cross-face-basins.md section 5), on earth-v7's
tectonics (hash `1e0618134ba4586a` in both). The replay start lowered the
fastest cell by 3674 m and raised the most subsided one by 1183 m; the
reference-step crust had 19.9 % land and the start hold lifted it 712 m.
Eroded surface at iteration 800:

| | Earth | earth-v7 (stack, cap 2) | **earth-v8 (replay)** | estimate above |
|---|---|---|---|---|
| 0-1 km | 71.6 | 66.7 | **69.2** | 71.0 |
| 1-2 km | 15.4 | 19.2 | **18.5** | 17.7 |
| 2-3 km | 7.5 | 6.9 | **7.7** | 6.8 |
| 3-4 km | 3.8 | 3.1 | **3.5** | 3.34 |
| 4-5 km | 1.7 | 2.5 | **0.8** | 0.80 |
| > 5 km | 0.3 | 1.6 | **0.4** | 0.36 |
| highest point | 8849 m | 9195 | **7666** | ~7470 |
| land median | ~350 m | 695 | **636** | ~606 |
| ocean median | -3700 m | -3195 | **-2763** | ~-2804 |
| `lost_offshore` | | 1974 Mm | 1798 Mm | |
| loop datum drift | | -755 m | -81 m | |
| erosion stage | | 3652 s | 3459 s | |

The estimate was right to within a tenth of a point on every band and
40 m on the ocean. Five of the six bands move towards Earth's, the land
above 5 km lands on the bedrock's own 0.4 %, and the highest point comes
down 1.5 km. It is still 919 m above the bedrock's maximum (6747 m): the
no-erosion invariant says a column ends at its bedrock only when nothing
moves mass, and with erosion on, the flexural isostatic rebound lifts the
ground around incised valleys and the datum hold shifts the whole
surface, so a crest can end above its bedrock. That is ~920 m against the
~2450 m the double count added on earth-v7, and it has not been split
between the two causes. The small
preset's worry did not carry over: the 2-4 km bands, what a viewer reads as
mountains, went *up* (6.9 / 3.1 -> 7.7 / 3.5), because at Earth scale
erosion barely touches the crests while the double count had been
inflating the 4-5 km band at the 2-3 km band's expense. The 4-5 km band is
now half of Earth's, and the ocean is 560 m shallower than Earth's, both
properties of the bedrock (docs/land-median.md lists the seed-dependent
ones). The datum hold, which had to pull the planet down 755 m under stack,
now moves it 81 m.

## Side effects to know about

* **Changing `iterations` recomputes erosion.** In replay mode
  `iterations` is part of the checkpoint hash, because the start depends
  on it, so a shorter or longer run no longer resumes a longer run's
  checkpoint. Checkpoint file names still carry only the iteration number,
  so the new run overwrites the old family's files at the same iterations.
* **The replay covers what tectonics divided by.** Tectonics divides
  `uplift` by `erosion.iterations` when it runs, and `--from erosion` does
  not re-run it: an erosion stage with a different `iterations` than the
  tectonics bake replays that fraction of the window (a 40-iteration run
  on a 60-iteration tectonics bake lowered the start by two thirds of the
  window). The no-erosion-ends-at-bedrock invariant holds either way. The
  field also carries `uplift_baseline` and the collision-zone clamp, so
  "the crust at the reference step" is a close description, not an exact
  one.
* **The viewer's erosion timeline starts lower.** Erosion frame 0 is the
  replayed start, so at the tectonics/erosion boundary the belts drop by
  up to the window's growth (~3.6 km at the fastest Earth cell) and then
  rise back over the run.
* **A partial replay is not a good middle.** Starting from `bedrock - k x
  total` with k = 0.5 on the small preset gives a maximum of 601 / 715 m on
  a 585 / 584 m bedrock and 0.10-0.14 % of land above the bedrock's p99.9;
  the static Earth estimate gives 1.62 / 0.91 % in the 4-5 km / > 5 km
  bands at k = 0.5, 1.21 / 0.54 % at 0.75, 0.80 / 0.36 % at 1 (Earth 1.7 /
  0.3). It brings back part of the double count as a tuning knob. At Earth
  scale erosion barely touches the crests (earth-v5's top cell lost 36 m),
  so the small preset's band loss under full replay is unlikely to carry
  over; the Earth bake decides.

## Tests

`tests/test_erosion.py`:

* `test_uplift_replay_without_erosion_ends_at_bedrock`: a stub world with
  ~1 m/it uplift and a 5 m/it core, and particles and mass wasting
  monkeypatched out, run through `maps.step` for 12 iterations with the
  base, route and lake refreshes, isostasy and the datum hold. The replay
  start is `bedrock - applied` + a constant (ptp < 1e-9 cells) at 0.30
  land, and the end is `bedrock` + one constant (ptp < 1e-9), with the cap
  off and with it binding on the core. The only mass change is the
  datum's. The same run in stack starts at `bedrock` exactly and ends at
  `bedrock + applied` + a constant.
* `test_uplift_stack_mode_is_the_old_rule_bit_for_bit`: stack's
  `build_state` equals `ErosionState.from_grid` on the untouched bedrock,
  and `start_replay` / `uplift_replay` are no-ops. Four steps equal, byte
  for byte in height, sediment, discharge, momentum, pending, base and
  route, four steps driven with af58d28's `apply_uplift` body inlined,
  with the 2 m/it cap binding. `iterations` is still out of stack's
  checkpoint hash.
* `test_uplift_replay_resume_equals_continuous_run`: a 10-iteration replay
  run, then a run resumed from its iteration-5 checkpoint, writes the same
  output bytes. With a second `start_replay` after the checkpoint load,
  this test fails. A 6-iteration replay and a stack run of the same world
  both refuse the checkpoint. An unknown `uplift_mode` raises.
* `test_uplift_replay_is_a_no_op_on_a_window`.
* Adjusted: `test_checkpoints_are_pruned_and_a_shorter_run_resumes` pins
  `uplift_mode = 'stack'` (the behaviour it documents), and
  `test_uplift_cap_bounds_the_rise_and_stays_mean_free` sets
  `uplift_max_m = 2` explicitly now that the default is 0.
