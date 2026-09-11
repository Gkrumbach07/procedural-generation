# Pipeline cost: what each stage does, and what it costs

Where the time and memory actually go, so it is obvious which stages have
room to grow and which do not.

> **Superseded for absolute numbers.** Everything below was measured on 4
> cores, and partly on older erosion defaults. docs/bake-performance.md has
> the 20-thread measurements, including the first complete Earth bake
> (84.5 min, refine 7 min — the "~73 hours" projection below was a
> scheduling artefact). The structure of each stage described here still
> holds.

All measurements are the shipped `final512` world: `N_c = 512`, `R = 4`,
50 m coarse cells, 30 % land. Scaling exponents are empirical, from
comparing against `med1` (`N_c = 256`, `R = 2`), where coarse cells go up 4×
and fine cells 16×.

## Notation

| symbol | meaning | final512 |
|--------|---------|----------|
| `C` | coarse cells, `6·N_c²` | 1,572,864 |
| `L` | coarse **land** cells (`land_fraction` 0.3) | 471,859 |
| `F` | fine cells, `R²·C` | 25,165,824 |
| `T` | tectonic cells, `6·N_tect²` | 393,216 |
| `S` | tectonic segments (point cloud) | ~plates × seg/plate |
| `I` | erosion iterations | 400 |
| `P` | particles per iteration, `particles_per_cell · L` | ~118,000 |
| `s̄` | mean steps a particle survives (cap `2·N_c` = 1024) | ~80 |

## The whole pipeline at a glance

Total 1057 s ≈ 17.6 min, ~1.0 GB on disk.

| # | stage | time | share | measured scaling | dominant term |
|---|-------|------|-------|------------------|---------------|
| 1 | tectonics | 121.5 s | 11.5 % | ×13.2 per 4× C | `O(steps · (S log S + T))` |
| 2 | climate | 6.1 s | 0.6 % | ×2.7 | `O(C)` |
| 3 | **erosion** | **571.1 s** | **54.0 %** | ×4.1 (linear in C) | `O(I · P · s̄)` |
| 4 | hydro | 0.7 s | 0.07 % | ×3.5 | `O(C log C)` |
| 5 | watersheds | 1.2 s | 0.1 % | ×4.0 | `O(C)` |
| 6 | **refine** | **331.3 s** | **31.3 %** | ×15.8 (linear in F) | `O(I_r · F_land)` |
| 7 | derive | 11.5 s | 1.1 % | ×8.2 | `O(F)` |
| 8 | tiles | 13.7 s | 1.3 % | ×6.9 | `O(4/3 · F)` |

**Two stages are 85 % of the runtime. The other six together are 3.1 %.**

---

## 1. Tectonics — 121.5 s

Clustered convection on the sphere: a segment point cloud on its own
`N_tect` grid, 1500 steps, resampled to the coarse grid at the end.

Per step: rigid plate rotation `O(S)`; KD-tree build and pair query
`O(S log S)` for subduction; nearest-segment label map via voxel hash
`O(T)`; Voronoi areas and a cascade relaxation over the cloud.

* **Time** `O(steps · (S log S + T))`
* **Space** `O(S + T)` — small; the point cloud dominates
* Measured ×13.2 for 4× coarse cells: superlinear because `N_tect` and the
  segment count both grow with the world.

Runs once and feeds everything. 11 % is a reasonable price, but note it is
the stage that sets the **long-wavelength relief** the rest of the pipeline
inherits — see docs/terrain-realism.md.

## 2. Climate — 6.1 s

Temperature (closed form), wind advection, orographic precipitation,
evaporation — all fixed-pass sweeps over the coarse grid.

* **Time** `O(k · C)` with small `k`; measured *sublinear*, so fixed costs
  dominate at this size
* **Space** `O(C)`, a handful of fields

**0.6 % of runtime.** This is effectively free, and it produces `precip`,
which is a direct input to the erosion kernel's spawn weighting. A far more
elaborate climate model would not show up in the budget.

## 3. Erosion — 571.1 s ← the dominant stage

The global particle pass on the coarse grid, 400 iterations.

Per iteration:

| work | cost | final512 |
|------|------|----------|
| particle descent | `O(P · s̄)` | 118k × 80 ≈ 9.4 M particle-steps |
| thermal + creep (2 mass-wasting passes) | `O(C)` | 1.6 M |
| priority flood, every `flood_every`=10 | `O(C log C)` / 10 | amortised ~3 M |

* **Time** `O(I · P · s̄)` ≈ **3.8 × 10⁹ particle-steps** — about 6.6 M
  steps/s on this box. Particle descent dominates everything else by ~6×.
* **Space** `O(C)` fields, plus the change list:
  `chunk · (max_steps + 2·SPREAD) · 24 B` ≈ 2048 × 1040 × 24 ≈ **51 MB**
* Measured ×4.13 for 4× coarse cells: **linear in C**, as expected since
  `P ∝ L` and `s̄` is capped.

Levers and what they cost:

* `iterations` (400) — linear.
* `particles_per_cell` (0.25) — linear. **Measured to change nothing**: 4×
  particles left β at 4.86 → 4.90 and the network unchanged. Do not spend
  here.
* `max_steps` (auto `2·N_c` = 1024) — caps `s̄`, currently ~80, so raising
  it is nearly free while lowering it would bite.

## 4. Hydro — 0.7 s

Sea level, priority flood, D8, flow accumulation in topological order,
drainage tree, lakes.

* **Time** `O(C log C)`, the flood's heap dominating
* **Space** `O(C)`

**0.07 %.** Free. There is room here for far more elaborate hydrology —
dynamic lakes, multi-flow-direction routing, sediment routing on the graph —
without touching the budget.

## 5. Watersheds — 1.2 s

Labels basins from the drainage tree and writes `basins.json`.

* **Time** `O(C)` · **Space** `O(C)`

**0.1 %.** Free — but see the load-balance note under refine: *what this
stage emits determines refine's parallel efficiency.*

## 6. Refine — 331.3 s ← the second dominant stage

Per-basin refinement into the `fine/` rasters: each basin's coarse cells are
expanded `R² = 16×`, given a halo, and run for `refine_iterations = 40`
particle iterations, then a local priority flood and `block_drift`.

* **Time** `O(I_r · F_land)` where `F_land = R²·L ≈ 7.5 M` fine cells;
  measured ×15.8 for 16× fine cells, i.e. **linear in F**
* `block_drift` adds Jacobi passes to tolerance, ≤64, over each window
* **Space** per basin window + halo, **multiplied by the worker count**

### The load imbalance is the real issue here

2300 basins, and the distribution is brutally skewed:

| | |
|---|---|
| median basin | **2 coarse cells** |
| basins ≤ 4 cells | **1676 (73 % of basins, 0.57 % of the area)** |
| top 100 basins | **65 % of the area** |
| largest basin | 12,406 coarse → **198,496 fine cells** |

So 73 % of the jobs are scheduling overhead for half a percent of the work,
while wall-clock is floored by a single basin that alone is ~7.9 M
fine-cell-iterations. Merging tiny basins into their neighbours and
splitting the giants would improve parallel efficiency without changing any
physics — the cheapest real speedup available.

## 7. Derive — 11.5 s

Rivers, lakes, soil, biomes over the fine grid.

* **Time** `O(F)` sweeps · **Space** `O(F)` per field
* **1.1 %.** Free.

## 8. Tiles — 13.7 s

The LOD pyramid, each level a quarter of the last, so `4/3 · F` total.

* **Time** `O(F)` · **Space** 243 MB written
* **1.3 %.** Free.

---

## Storage

| | size |
|---|---|
| `coarse/` (18 fields) | 115 MB |
| `fine/` (9 fields) | 649 MB |
| `tiles/` | 243 MB |
| `graph/` | 7.8 MB |
| **total** | **~1.0 GB** |

One fine field is `6 · 2048² · 4 B` = 100 MB, so fine-grid fields are the
memory story. Adding a fine field costs 100 MB on disk at this world size,
and `R² = 16×` more than a coarse one.

## Where the headroom is

**Free (3.1 % combined).** Climate, hydro, watersheds, derive, tiles. Any of
these could get 10× more expensive and add under a minute. If a fix needs
richer climate forcing, smarter routing, real lake dynamics, or more derived
layers, the budget is already there.

**Expensive (85 %).** Erosion and refine — and awkwardly, these are exactly
the two stages that would have to change to fix what
docs/terrain-realism.md identifies as the actual defect (too little relief
amplitude at 100 m – 3 km). Both are linear in their grid, so cost scales
predictably, but neither is cheap to iterate on.

**Two cheap wins that need no physics change:**

1. **Rebalance the basin partition** (stage 5's output) — merge the 1676
   sub-5-cell basins, split the giants. Refine's wall-clock is set by its
   largest job.
2. **Stop spending on particle count.** `particles_per_cell` is a linear
   cost with a measured null effect. If erosion needs to get more expensive,
   it should be in iterations or in the kernel, not in particles.

## Measured at Earth scale (the first complete run)

The cost model above is borne out. On 4 cores, `earth` preset, 1024² faces:

| stage | wall clock | share |
|---|---|---|
| tectonics | 247 s | 3.3 % |
| climate | 56 s | 0.8 % |
| **erosion** | **7038 s** | **94.9 %** |
| hydro | 3.7 s | 0.05 % |
| watersheds | 3.9 s | 0.05 % |
| refine | *stopped* | — |

Erosion is 800 iterations at ~7 s each pre-glaciation and ~11 s each after
iteration 600, when the glacial pass switches on and adds ~4 s/iteration.

**Refine, measured rather than modelled.** 9684 basins, 4 workers, and the
first basin took 109 s (53482 cells, 752² window, 150 iterations, 7.9 M
particles). At that rate the stage is **~73 hours**; even assuming that
basin is among the largest and the mean is ten times cheaper, 6–7 hours.

This confirms the architectural conclusion at the scale that matters:
**do not pre-bake the tile pyramid.** Refine basins ahead of the camera.
`export_godot.py` reads the coarse grid, so the globe export does not depend
on refine or tiles at all.
