# Sea-death surplus: parked, not deleted (erosion kernel 8)

Until kernel 7, a particle that died in the sea with a load its cell would
not take had that load deleted: `apply_changes` counted it in
`lost_offshore` and it left the model. Over the earth-v5 bake that was
3141 Mm of crust (docs/lakes-in-erosion.md), tripled in the bake that brought
in the 1 m deposition floor. Kernel 8 parks it instead, the way a land pit's
surplus has always been parked: `1 - erosion.offshore_writeoff` of it goes to
the death cell's `pending` and is re-injected next iteration as a seafloor
particle, the rest is written off as `lost_offshore`. The default write-off
is 0.25; 1.0 is the old rule and reproduces it bit for bit.

Everything below is the small preset (N_c 128, 50 m cells, seed 0, 60
erosion iterations unless stated), the real kernel, no instrumentation.
Earth is a one-hour bake and is not measured here: the first bake on
kernel 8 (earth-v6) is where the 3141 Mm gets its after-number.

## The rule, and why a write-off

A sea death lists one final entry (`trace_particles`: `nout = 1` for
`DEATH_OCEAN`), because a load that reached the sea never comes back on
land, so unlike a land death it has no overflow slots up its path. Its
overflow slot is now the pending stockpile of its own cell. Parking *all*
of it would reintroduce the deadlock the deletion was written against
(commit b4f24d4, docs/erosion-tuning.md: 34,915 cells of pending on a shelf
that had filled to sea level) -- a stockpile on a shelf that can never take
anything would come back every iteration and grow with the inflow. With a
write-off `w` per failure the stockpile that keeps failing decays as
`(1 - w)^n`, so seafloor pending is a geometric series bounded by
`inflow / w`, and `lost_offshore` keeps its job as the counter of what is
finally unplaceable.

This is stateless (no per-cell retry count to checkpoint, no format change)
and the ledger closes without new bookkeeping: `total_mass` already sums
height + sediment + pending. `to_pending` / `pending_total` now include
seafloor stockpiles; the stage info gains `pending_sea_m`, the part still
walking when the run ended (absent from the outputs, like the land part).

## Which ceiling strands the load

The review asked whether it is the fan ceiling (`fan_slope`) rather than
the 1 m floor (`dep_floor_m`) that leaves no room on a flat seafloor. Read
against the kernel: the fan ceiling (`prev - fan_slope`) applies to
seafloor *steps* only; a sea death's final deposit has no reference cell
and no fan ceiling (`ref = -1`), it is bounded by the `iter_deposit` rate
cap and by the waterline, `base - dep_floor - surface`. So the fan
ceiling is what clamps the walk (the load arrives at the death cell still
carried), but what strands it at the end is a shelf already filled to the
floor. Measured with `fork_erosion.py --fresh` arms at `offshore_writeoff
= 1.0`, so `lost` is the whole surplus:

| arm (60 iterations, `offshore_writeoff = 1.0`) | `lost_offshore` |
|---|---|
| defaults (the old rule) | 96,803 m |
| `fan_slope = 0` (a flat fan: the step ceiling is the previous cell's surface, which on a flat shelf is still no room) | 85,814 m (-11 %) |
| `dep_floor_m = 5` (the shelf fills to 5 m below the waterline, not 1 m) | 120,200 m (+24 %) |
| `dep_floor_m = 0` | 253 m -- degenerate, see below |

The fan ceiling is 11 % of the loss. The floor moves it the way shelf
*capacity* does -- a deeper floor leaves less room over the shelf, so it
fills sooner and strands more -- which is the signature of a filled shelf,
not of a per-visit ceiling: a seafloor step's ceiling is `prev - fan_slope`
whatever the floor (`apply_changes` cancels `dep_floor` out of it for a
submerged cell), so the reading in docs/lakes-in-erosion.md that the 1 m
floor made a *flat seafloor* accept almost nothing per visit does not
hold. What the 1 m floor did at Earth is send the shelf 56 % of the
sediment instead of 12.5 %: more flux arriving at a shelf that then fills.
`dep_floor_m = 0` is not an arm: a shelf cell then fills to exactly `base`,
the kernel classifies it as land (`surface >= base`), the sea turns into
land (pit deaths 18,783 to 28,140) and the surplus is parked as a land
pit's. The floor is the guard that keeps sea as sea, which is why it
cannot be zero.

The death census (`--deaths`) says the same from the other side: sea
deaths average 9-14 seafloor steps out of the 64 allowed (coast 8.5, shelf
10.3, slope 13.6), so walks end by exhausting the load or bumping land,
not by the step cap; half the sea deaths are on the slope (51.7 %), a
third at the coast (31.2 %), a sixth on the shelf (16.0 %), and two of
1.46 M in the deep.

## Before and after

`scripts/bake.py --preset small --to erosion` three times into `/tmp`:
the tree before this change, the new tree with `offshore_writeoff = 1.0`,
and the new tree at its default. Stage info from `manifest.json`:

| | before (kernel 7) | kernel 8, `offshore_writeoff = 1.0` | kernel 8, default 0.25 |
|---|---|---|---|
| `lost_offshore_m` | 96,802.69 | 96,802.69 | **72,902.83** (-24.7 %) |
| `pending_sea_m` at the end | -- (always 0) | 0 | 5,418.6 |
| `pending_total_m` at the end | 38,234 | 38,234 | 44,253 |
| `sediment_mean_m` | 9.840 | 9.840 | 10.010 |
| `sediment_p99_m` | 154.9 | 154.9 | 152.8 |
| `clamped_entries` | 8,044,455 | 8,044,455 | 8,779,799 |
| `land_fraction` | 0.3000 | 0.3000 | 0.3000 |
| output arrays | -- | byte-identical to before | differ |

The `1.0` arm is the switch-off equivalence: every output array under
`coarse/` is byte-identical to the kernel-7 bake, only the checkpoint hash
(`:k7` to `:k8`, plus the new field) differs. `pending_total_m` is mostly
the moraine of the final glacial pass (iteration 60 is a `glacial_every`
multiple; 262 land cells hold it under the old rule, 458 under the new),
which an iteration 61 would have re-injected; the seafloor part sits on
931 of 73,728 sea cells. At 120 iterations (`fork_erosion.py --fresh
--iterations 120 --set erosion.iterations=120`, so the glacial pass starts
at 90 in both arms): old rule 290,436 m lost, default 194,293 m (-33 %);
the recovery grows with the run because a parked stockpile gets more
chances.

Where the recovered mass goes (`--deaths`, change in sediment by the depth
zone before each iteration, 60 iterations, m): coast 195,018 to 206,165
(+11.1 k), shelf 289,676 to 291,614 (+1.9 k), slope 430,628 to 435,076
(+4.4 k), land 47,379 to 46,579, deep 4,629 to 4,603. A parked stockpile
re-enters at its death cell in a random direction on a filled shelf, so it
is a random walk of a few cells: the recovered sediment is coastal, and the
abyss is unchanged. Feeding the deep ocean would need a slope-following
start; that is a different change.

## The bound, measured

The runaway case is a shelf nothing can be placed on. Two views:

* The small preset over 120 iterations: total pending after every tenth
  iteration, read before that iteration's glacial pass (m): 3,768, 5,660,
  3,605, 2,793, 3,770, 5,272, 5,254, 5,193, 4,003, 9,445, 3,703, 8,625.
  Seafloor pending is three times the iteration's write-off by
  construction (verified exactly every iteration); it climbs while a
  shelf is filling and is knocked back on the iterations where isostasy
  or the base refresh opens room, and the two high readings are the
  glacial outwash once the pass starts at iteration 90 (the old rule
  spikes there too, to 6,244 m, from moraine the margin cannot take). No
  trend over 120 iterations. At the end of the
  60-iteration bake 931 of 73,728 sea cells hold 5,419 m.
* `tests/test_erosion.py::test_seafloor_stockpile_is_bounded`: a 48-cell
  window whose sea strip is filled to `dep_floor_m` below the waterline
  from the start and walled off, so every load reaching it is surplus,
  every iteration. Over 40 iterations seafloor pending is 9.8, 51.0, 83.5,
  96.6, 91.2, 86.2 cells after iterations 1, 5, 10, 20, 30, 40, against a
  geometric ceiling `(1 - w) / w * max inflow` of 102 and a linear pile-up
  (parking everything) of 886. After every iteration the stockpile on
  submerged cells is exactly `(1 - w) / w` times that iteration's
  `lost_offshore` (1e-9), which is the split rule observed from outside.

On the open-coast `tilt` window (`test_offshore_loss_is_accounted`) the old
rule deletes 0.2216 cells over 25 iterations and the new one 0.0295: the
strip has room, so 87 % of what used to be deleted is placed. Ledger error
1.3e-15 and 1.95e-15 relative.

## Cost

Every sea cell holding a stockpile is one more particle next iteration.
Small preset, 60 iterations: ocean deaths 1,455,906 to 1,497,733 (+2.9 %,
and +3.7 % over 120), i.e. ~700 extra particles an iteration on 24,576;
`clamped_entries` +9 %
(the re-injected loads are mostly clamped again). Seconds per iteration in
the census forks, same conditions: 0.238 (old rule) against 0.211
(default) -- noise. At Earth the extra particles scale with the sea deaths
that leave a surplus (14 % of ocean deaths here); check `particles` in the
erosion log against the rain count on the first kernel-8 bake.

## Invalidation

`KERNEL_VERSION` 7 to 8 and a new `ErosionParams` field: every erosion
checkpoint hash (`:k7` to `:k8`) and every world's `params_hash` -- the
manifest's and the erosion stage's (`group_hash` of the erosion group now
carries `offshore_writeoff`) -- changes, so `bake` refuses a resume and
existing worlds need `bake --force --from erosion`. The stage's output
`hash` is the content hash of the output arrays, which is why the `1.0`
arm above prints the kernel-7 one. `fork_erosion.py` still loads an old
checkpoint explicitly, but a fork of a kernel-7 state runs the new rule on
a stockpile-free seafloor.

## Commands

    cd bake
    .venv/bin/python scripts/bake.py --world /tmp/pg/small-before --preset small --to erosion   # from the kernel-7 tree
    .venv/bin/python scripts/bake.py --world /tmp/pg/small-w1 --preset small --to erosion --set erosion.offshore_writeoff=1.0
    .venv/bin/python scripts/bake.py --world /tmp/pg/small-after --preset small --to erosion
    python3 -c "import json; [print(w, {k: json.load(open(f'/tmp/pg/{w}/manifest.json'))['stages']['erosion']['info'].get(k) for k in ('lost_offshore_m','pending_sea_m','pending_total_m','sediment_mean_m')}) for w in ('small-before','small-w1','small-after')]"
    .venv/bin/python scripts/fork_erosion.py /tmp/pg/small-after --fresh --preset small --iterations 60 --label after --deaths
    .venv/bin/python scripts/fork_erosion.py /tmp/pg/small-after --fresh --preset small --iterations 60 --label w1 --deaths --set erosion.offshore_writeoff=1.0
    .venv/bin/python scripts/fork_erosion.py /tmp/pg/small-after --fresh --preset small --iterations 60 --label w1-fan0 --set erosion.offshore_writeoff=1.0 --set erosion.fan_slope=0
    .venv/bin/python scripts/fork_erosion.py /tmp/pg/small-after --fresh --preset small --iterations 60 --label w1-floor5 --set erosion.offshore_writeoff=1.0 --set erosion.dep_floor_m=5
    .venv/bin/python scripts/fork_erosion.py /tmp/pg/small-after --fresh --preset small --iterations 120 --label after120 --set erosion.iterations=120
    .venv/bin/python scripts/fork_erosion.py /tmp/pg/small-after --fresh --preset small --iterations 120 --label w1-120 --set erosion.iterations=120 --set erosion.offshore_writeoff=1.0
    # lost_offshore_m, deaths, death_zones, sediment_change_m_by_zone and the per-iteration pending are in <world>/forks/<label>/diagnostics.json
