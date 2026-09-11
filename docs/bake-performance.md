# Bake performance: where the time goes, and what should run where

Measured on the development box: 20 hardware threads (numba runs 20 threads,
GNU OpenMP layer), 30 GB RAM, and an RTX 5060 Ti with 8 GB (Blackwell,
compute capability 12.0, driver CUDA 13.3). Earlier cost notes
(docs/pipeline-cost.md, the README) were measured on 4 cores and on older
erosion defaults; this document replaces their numbers for this machine.

## The short version

* **Erosion is the bake.** Tectonics, climate, hydro and watersheds together
  take about 2 minutes at Earth scale; erosion took 73.
* **The erosion stage keeps 3 of the 20 cores busy.** `/usr/bin/time` gives
  291–354 % CPU over a run of iterations. The particle trace is parallel;
  most of the rest is serial, or parallel over the 6 cube faces only.
* **The single largest cost is the isostasy smoother**, 25 s every 20
  iterations (31 % of pre-glacial erosion). It is 283 explicit diffusion
  steps, each costing about ten full-array allocations and dtype conversions
  in Python. A fused kernel removes almost all of it. This is the one
  change worth making first.
* **The hard floor is `apply_changes`**, serial by construction (it is what
  makes the output byte-reproducible and independent of the thread count):
  1.1 s/iteration before glaciation and 1.7 s after, which comes to 16 minutes
  of every Earth bake whatever else is done.
* **The GPU is the right tool for two things only**: the stencil loops
  (isostasy, and to a lesser degree thermal/halo work), and the viewer. The
  particle kernel is not worth porting while the serial change-list
  application sits between every chunk of it.
* **Refine's "73 hours" was a scheduling artefact. Measured: 7 minutes.**
  The old figure was extrapolated from the first basin to *finish*, which was
  one of the four largest submitted first. On 20 workers every basin but one
  was done after 155 s. The last, the largest, then ran alone on a single
  thread for another 4.4 minutes.
* **The first complete Earth bake took 84.5 minutes** on this machine
  (`worlds/earth-full`, below).

## The first complete Earth bake (`worlds/earth-full`, 2026-09-11)

Shipped defaults, every stage from tectonics to tiles, plus viewer capture
and export. The run had two phases: `--to watersheds`, then the rest.

| stage | seconds | notes |
|---|---|---|
| tectonics | 100 | includes 61 viewer frames; 74 s in w-base without them (the test suite was also running) |
| climate | 17 | 12 s uncontended |
| erosion | **4380** | 73 min, 5.5 s/iteration mean, 339 % CPU |
| hydro | 1.3 | |
| watersheds | 1.2 | 518 basins |
| refine | **419** | all 20 cores; one basin alone was 397 s (below) |
| derive | 9.8 | |
| tiles | 3.0 | 8190 tiles, 210 MB |
| viewer | 55 per export | 30 s for 143 frames (98 MB), 25 s for the equirect PNGs and animated WebP |
| **total** | **≈ 84.5 min** | 2.4 GB on disk |

**Erosion ran 25 % slower than the profile below predicted** (73 min against
58). The profile forked w-base checkpoints, whose terrain came from the old
defaults. On the new defaults the terrain keeps more relief, and particle
paths average 110–137 steps instead of about 90. Trace and apply both scale
with path length.

**Refine is minutes, not hours.** It used all 20 cores (1924 % CPU summed
over 22 processes) and 517 of the 518 basins were done after 155 s. The
last basin (222,621 coarse cells, a 1622² window, 33 M particles) took 397 s
on its own on **one thread**. The per-worker thread budget is fixed at
`cpu_count // workers` = 1 when the pool starts, so the largest job ran
single-threaded on an idle machine for more than four minutes. It is also
the most memory-hungry stage: the 20 workers each load the coarse fields,
20.6 GB resident against 30 GB of RAM.

## Measured stage costs, Earth preset (`N_c = 1024`, 9773 m cells)

From `worlds/w-base` (baked on this machine, *old* erosion defaults: no
isostasy, glacial lengths in cell units):

| stage | seconds | notes |
|---|---|---|
| tectonics | 74 | 58 s simulation (1500 steps × 0.039 s) + setup, finalise, resample |
| climate | 12 | 8.3 s precipitation advection (728 sweeps), 3.5 s wind |
| erosion | 3711 | 800 iterations, 4.6 s mean |
| hydro | ~4 | |
| watersheds | ~4 | |
| refine / derive / tiles | — | never run at Earth scale before `worlds/earth-full` |

The small preset takes 51 s end to end on a cold numba cache, and 20 s with a
warm one.

## One erosion iteration, by phase

`/tmp`-style harness: load a w-base checkpoint, run 20 iterations with the
*current* defaults, and time every kernel (inclusive; the first iteration is
discarded as JIT warm-up). Pre-glacial from iteration 200, glacial from 750.

| phase | pre-glacial s/iter | glacial s/iter | parallel? |
|---|---|---|---|
| `apply_changes` (change list → terrain) | 1.09 | 1.66 | **serial by design** |
| `trace_particles` | 1.01 | 1.33 | prange over particles, 20 threads |
| `apply_isostasy` (25 s every 20 iterations) | 1.27 | 1.28 | Python loop of 283 steps; kernel prange over 6 faces |
| thermal + creep | 0.21 | 0.22 | prange over 6 faces |
| halo exchanges (5 per step, einsum) | 0.11 | 0.19 | single-threaded numpy |
| route flood (1.28 s every 10 iterations) | 0.13 | 0.13 | serial heap |
| spawn | 0.09 | 0.09 | numpy + serial njit |
| glacial carve (~1 s every 10 iterations) | — | 0.10 | numpy |
| uplift, datum hold, pack, fold, EMA | 0.10 | 0.10 | mixed |
| **total** | **4.09** | **5.13** | |

At 600 pre-glacial and 200 glacial iterations that is **≈ 58 minutes** of
erosion with the shipped defaults. Two corrections to the earlier record:

* The glacial pass itself costs about 1 s a pass (0.1 s/iteration). The big
  spikes seen every 20 iterations are the isostasy smoother, not the glacier.
  The glacial phase costs more per iteration because particle paths get
  longer (72 → 96 steps), which slows both trace and apply.
* The first iteration of every run costs 16 s: numba loading its cache and
  touching memory for the first time. `seconds_per_iteration` in the
  manifest includes it.

### Why the cores are idle

* `apply_changes` runs ~768 times an iteration (once per 2048-particle
  chunk). While it runs, the other 19 threads wait, and the next chunk's
  trace cannot start because it must see the terrain the apply just changed.
* Seven kernels (`pack_samples`, `fold_changes`, `ema_update`, the three
  thermal passes, `_laplacian_kernel`) use `prange(F)` with `F = 6`: at most 6
  threads each.
* Halo exchange, spawn sorting, `np.partition`, `np.roll` and the other
  numpy work are single-threaded, and many of them allocate full-grid
  temporaries every iteration for inputs that never change: the spawn masks,
  `talus`, the uplift mask and mean, and the halo weight tables.

## The isostasy smoother, measured four ways

283 steps of `f += κ L f cell²` with a linear halo exchange before each, at
`N_c = 1024` (`/tmp/gpu_iso.py`, run while an Earth bake was using the CPU,
so the CPU rows are pessimistic; uncontended, the current code measured
25 s per application):

| implementation | ms/step | s per application | max diff vs current |
|---|---|---|---|
| current (`FaceField.laplacian` in a Python loop) | 129 | 36.6 | — |
| one fused numba kernel, prange over rows | 16.0 | 4.5 | 1.1e-16 |
| CuPy FP64, FMA contraction off | 3.7 | 1.06 | 1.1e-16 |
| CuPy FP64, FMA on | 3.4 | 0.95 | 4e-10 |
| CuPy FP32 | 1.6 | 0.47 | 1e-7 |

The 1.1e-16 difference is the same for CPU and GPU. It comes from the order
in which the 16 halo weights are summed (numpy's einsum against a plain
loop), not from the stencil. Keeping the einsum halo exchange and fusing
only the Laplacian and the update should therefore be byte-identical, while
keeping most of the 8× gain.

## What should run where

| component | share of a bake | GPU fit | verdict |
|---|---|---|---|
| isostasy smoother | 20–30 % of erosion | excellent: a pure stencil loop | **fused numba kernel now**; CuPy is a further 4× if it is ever worth a CUDA dependency |
| `apply_changes` | 27–32 % | none: every entry depends on all earlier ones | CPU, serial. The floor. |
| `trace_particles` | 25 % | poor today: float64 math (consumer FP64 is 1/64 rate), divergent paths, and a CPU apply between every chunk, i.e. ~768 PCIe round trips per iteration | CPU. Revisit only with a GPU-native erosion design, which changes output (the `erosion.backend` knob already reserves this). |
| route / priority flood | 3 % | none: sequential heap | CPU; use `hydro/priority_flood.py`'s array heap instead of the tuple `heapq` |
| thermal, pack, fold, EMA, uplift, halo | ~12 % together | good (stencils and maps) | CPU: prange over rows, not faces; cache constants; drop the per-call `astype` copies |
| tectonics step | 1.5 min total | poor: 20 k points, order-dependent kernels, latency-bound | CPU. Jit `shape_belt`, reuse one voxel hash per step. |
| climate advection | 12 s total | excellent | not worth it at 12 s |
| refine | unmeasured at Earth, estimated tens of minutes on 20 cores | batched basins could work | CPU scheduling first (below) |
| viewer rendering | — | it is WebGL | done: `globe/viz/viewer.html` |

**GPU tooling on this machine.** `numba-cuda` 0.30.4 does not import with
numpy 2.5 (it references the removed `np.row_stack`). CuPy 14.2
(`cupy-cuda13x`) works through raw CUDA kernels once `CUDA_PATH` and
`LD_LIBRARY_PATH` point at the pip `nvidia/cu13` libraries; measured 431 GB/s
on a plain streaming kernel. Pass `--fmad=false` to nvrtc when comparing
against the CPU, or results differ through FMA contraction alone.

## Recommended order of work

Each step is independent. The expected gains are per Earth bake, measured or
derived from the table above.

1. **Fuse the isostasy smoother** (numba, keep the einsum halo exchange).
   Saves about 1.1 s/iteration → erosion **58 → ~43 minutes**. Should be
   byte-identical; verify with the existing determinism tests before and
   after.
2. **Widen the six-face kernels to prange over rows** and cache the per-run
   constants (spawn masks, `talus`, uplift mean, halo weight casts). Saves
   about 0.3 s/iteration → **~38 minutes**. Byte-identical.
3. **The route flood on the array heap.** Saves about 0.1 s/iteration.
   Tie-breaking must stay `(value, cell)` to keep bytes.
4. **Refine scheduling.** Give the largest basins the whole thread budget,
   and hand threads back as the pool drains, instead of fixing
   `cpu_count // workers` at pool start. The measured tail is one basin on
   one thread for 264 of the stage's 419 s; with the trace on 20 threads it
   should take refine to about 3 minutes. Output is unchanged (the refine
   tests already check byte identity across thread counts). Sharing the
   coarse inputs through memory-maps would also cut refine's 20.6 GB peak.
5. **Only if more is needed: pipeline trace and apply.** Trace chunk k+1
   against the terrain *before* chunk k's apply (a one-chunk lag), so the
   serial apply overlaps the parallel trace. This takes most of the trace
   time off the critical path (up to ~1 s/iteration) but **changes the
   output**, so it needs a `KERNEL_VERSION` bump and a fresh baseline.

After 1–3 the bake is bounded by `apply_changes`: about 16 of the remaining
~38 erosion minutes.

## Defects found while measuring

These are not performance problems, and none of them is fixed here.

* `bake.py -v` sets the root logger to DEBUG, so numba logs its compiler
  passes: 1.08 million lines on a small bake.
* `tectonics.animate_*` is in the tectonics hash group, so switching the
  WebP animation on invalidates the whole world, although capture does not
  change the output. The viewer's frame knobs are `render.*` and are exempt.
* The glacial pass writes the ice flag into `state.mask` but
  `ErosionState.exchange_halos` never exchanges the mask, so ice does not
  cross face seams.
* The thermal passes run over the halo without an exchange after the
  particle pass, so seam cells read the previous iteration's halo.
* `scripts/export_godot.py` and every globe quicklook put the pole on +Y,
  while climate (and `Grid.latitude`) put it on +Z. The climate bands in
  those images lie along the wrong axis. The new viewer uses +Z.
* `export_godot.py` reads `meta["R_planet_m"]`; the manifest key is
  `R_planet`.
* `scripts/hypsometry.py <world>` measures `coarse/bedrock`, the tectonics
  output, not the eroded surface. Only `--checkpoints` reads erosion. It is
  easy to misread as the final world (docs/earth-bake.md).
* docs/pipeline-cost.md quotes `refine_iterations = 40`; the config has
  always said 150. (The README's reference to the deleted
  `scripts/animate.py` has been replaced by the viewer.)
* The ocean floor carries staircase terraces about 4 coarse cells wide, the
  tectonics grid resolution. They show plainly in the viewer at the default
  relief exaggeration.
