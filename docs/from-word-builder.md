# What is worth taking from `word-builder`

`Gkrumbach07/word-builder`, branch `claude/procedural-world-generator-qipmos`,
is a browser world generator (TypeScript, ~3400 lines, Vite, no runtime deps)
built on the same source material as this project: weigert/SimpleTectonics for
clustered-convection plate tectonics and weigert/SimpleHydrology for
particle erosion. Two independent ports of one lineage, so most of it is
convergent rather than new.

This is a review of what it does that this repo does not, written after
reading both. The honest tally is shorter than a first pass suggested — three
of the four things that looked like gaps turned out to be already solved here,
in one case better. What survives is one real feature, one conditional
architectural lesson, and one idea that only matters if a GPU path is ever
wanted.

## 1. Vegetation modulates erodibility — a real gap

`word-builder` runs biome → vegetation density → rock erodibility:

```ts
let v = vegetationDensity(t, m);      // Whittaker cell: temperature x moisture
v *= Math.max(0, 1 - relief * 2.5);   // steep ground holds less soil
erodibility[i] = (1.4 - hard) * (1 - 0.65 * veg[i]);
```

Rooted soil resists erosion, so forests weather gently and bare badlands
gully. It is a genuine geomorphic feedback, it is roughly fifteen lines, and
it is **not** in the SimpleHydrology lineage — this is the author's addition.

This repo has `vegetation` only as an *output*: `derive` computes it and
`refine/tiles.py` writes it to the alpha channel of `layers.png`. Nothing
feeds it back into `hardness`, so erosion treats a rainforest and a desert
identically given equal rock. Adding the feedback means computing vegetation
before erosion rather than after, which is an ordering change (`derive`
currently runs last), not just a formula.

Worth doing, and cheap. It would also give the biome map something to do
besides colour.

## 2. Refining an arbitrary window needs river inlets — conditional, but real

`word-builder`'s zoom cuts a 16x16 coarse window anywhere the user clicks,
upsamples it, and then scans the ring of coarse cells *outside* the window for
ones whose D8 flow points *in*, injecting each as a boundary inflow with
position, direction, and strength scaled by upstream discharge
(`src/sim/tiles.ts`).

**This repo does not need that today, and the reason is structural.** `refine`
partitions by *watershed*: a basin is by definition a region no river flows
into, so there is no cross-boundary inflow to model. What continuity it does
need it already has — `basin_job` starts fine `discharge` and `momentum` from
the upsampled coarse maps in the same volume units, so fine rivers continue
coarse ones and the EMA relaxes from there, with a frozen divide ring and
every exit a sink.

The lesson is conditional and lands the moment the camera drives refinement.
`docs/pipeline-cost.md` establishes that the tile pyramid cannot be pre-baked
(9684 basins, ~109 s each). The alternative is refining ahead of the camera —
and a camera does not sit inside one order-8 basin, nor can a whole basin be
refined in real time. Camera-local refinement means **arbitrary windows**,
which is exactly `word-builder`'s regime, and the inlet mechanism is the
answer to the problem that regime creates: a window that does not know what
drains into it grows a drainage network contradicting its surroundings.

So: not a defect here, but the design to reach for when refine becomes
interactive. Its inputs already exist — `hydro` produces `flowTo` and
`discharge` on the coarse grid.

## 3. Fixed-point atomics for GPU determinism — only if a GPU path is wanted

`word-builder`'s WebGPU kernel accumulates every particle's deposits into an
`atomic<i32>` buffer at a fixed scale (1e6 for rock, 1e4 for the discharge and
momentum tracks) rather than using float atomics, because float addition is
not associative and parallel float atomics therefore give a different answer
each run. Integer sums are order-independent, so a seeded world reproduces.

This repo reaches determinism differently and, for its purposes, better:
particles trace in parallel against *frozen* maps into a per-particle change
list, and `apply_changes` applies that list **serially in particle order**
(`erosion/particle.py`). That buys something commutative atomics cannot
express — order-dependent caps. `iter_erode` and `iter_deposit` are enforced
against the live terrain in apply order, and a particle's excess moves back up
its own path. An atomic-sum kernel has no "apply order" to enforce them in.

Keep the current design. The technique is on record here in case a GPU path is
ever built, where serialisation is not available and the tradeoff flips.

## 4. Already solved here, for the record

Two things that look like `word-builder` advantages and are not:

* **Relief-conditioned detail amplitude.** Its zoom scales fractal detail by
  local coarse relief and damps the seafloor to 0.25x. `refine/upsample.py`'s
  `detail_noise` conditions on slope, relief *and* hardness
  (`detail_amp * min(slope * cell, relief) * (0.5 + 0.5 * hardness)`), with
  ridged fBm octave-matched to what survives the job's drift removal. Strictly
  more.
* **Crust thickness driving erodibility.** Both do it; this repo derives
  `hardness` from crustal age and strata in `tectonics/run.py`.

## 5. Where this repo is ahead

Stated to keep the comparison honest, not to score points. `word-builder` is a
flat clamped 256² square about 2000 km across — a subcontinent called a
planet, with no wrap and no sphere. This repo runs a 6x1024² cube-sphere at
9773 m cells with rigid plate motion about Euler poles, plus crust types,
cratons, orogens with real cross-sections, and a mass ledger that closes.
`word-builder` has no tests (`npm run build` is `tsc --noEmit` plus vite)
against 157 here, and its branch is a single squashed commit.

Its tectonics is the 2020 SimpleTectonics model — one crust population, no
craton rule, no orogen shape. Everything in `docs/crust-types.md` is
downstream of finding that model insufficient.

## Recommendation

1. **Do** the vegetation-erodibility feedback. Small, real, and the one clear
   gap. Requires moving vegetation ahead of erosion in stage order.
2. **Remember** the inlet design for when refine goes camera-driven. Nothing
   to build now.
3. **Do not** adopt fixed-point atomics on the CPU path; it would cost the
   order-dependent caps.
