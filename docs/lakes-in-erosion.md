# One water surface: lakes as erosion state

docs/earth-v3-review.md section 2 found that nothing that drew a river and
nothing that drew a lake shared a water surface: the viewer's rivers were
the erosion stage's particle discharge, its lakes hydro's flood after
erosion, the fine rivers a thresholded fine discharge that ran straight
through every lake, and the fine lakes refine's re-flood of every basin to
its spill point. This is the sequence of changes that made them one thing,
and what each cost. The viewer half (rivers from hydro's flood, coloured by
order) is in the review; the rest is here.

## 1. Lakes inside the particle loop (`erosion.lake_balance`)

McDonald's 2020 post filled pools inside the particle loop and his 2023
model removed them again; docs/earth-v3-review.md has the argument for
why this loop was already half way there (a routing flood every
`flood_every` iterations that particles steer on). The other half:

**`ErosionState.refresh_lakes`** (`erosion/maps.py`), every `flood_every`
iterations after `refresh_base` and `refresh_route`, runs hydro's own
sequence on the eroding surface: `priority_flood_sphere`, D8 on the
filled DEM, and the upstream-first accumulation with the evaporation
balance (`hydro/balance.py`, `hydro.lake_evap`). That gives every
depression a water level. Two kinds come out:

* **Overflowing** (level at the spill point): the cells under water are
  flagged, `S_RFLAG = 2` in the packed sample (no new channel; `NS = 16`
  is one cache line and stays so). The kernel:
  - drops `lake_trap = 0.9` of a particle's load at the shore cell as it
    steps into a lake (`trace_particles`, the land branch): a delta;
  - exchanges nothing with the bed while the particle crosses on the
    routing surface, and does not count the crossing as climbing out of a
    pit;
  - never erodes a flagged cell below its routing surface
    (`apply_changes`), so a lake bed is not cut down but its outlet is,
    and a lake drains itself by incising its sill.
  Discharge and momentum are still recorded across the lake, so the
  river below it keeps its discharge.
* **Closed** (drawn down below the rim): the level goes into `base`, the
  kernel's sea test (`surface < base`) does the rest -- particles entering
  settle their load as a fan and stop, no rain spawns on it, no creep
  crosses it. A Caspian is a sea to the kernel, which is what it is.

Hydro is unchanged: it runs the same flood and balance on the final
surface, so its lakes are erosion's lakes up to the last ten iterations
of change. The flags are checkpointed (`lake_flag`) so a resumed run sees
what the continuous run saw; `KERNEL_VERSION` is 7.

Cost at the `earth` preset: the flood and balance are the ~0.7 s hydro
already spends, every tenth iteration.

## 2. The deposition floor in metres (`erosion.dep_floor_m`)

`DEP_FLOOR` was 0.02 *cells*: 1 m of water kept over any deposit on the
50 m preset the model was tuned on, **195 m** at the `earth` preset, which
forbade deposition in the whole top of the water column -- no delta,
coastal plain or shelf wedge could build (docs/crust-audit.md,
docs/earth-bake.md). It is now `dep_floor_m = 1.0`, converted to cell
units where the kernels take it (particle deposition, `apply_changes`'
ceilings, and mass wasting into submerged cells). This is the coarse-grid
half of the coast fringe in the review's section 3; the refine half
(`refine.coast_taper_m`) fades the detail noise out at the shoreline.

## 3. Refine and derive read the same lakes

* `refine/basin_job.py`: the fine flood inside a basin is capped at the
  coarse water surface (the balance, upsampled) rather than refilling the
  depression to its spill point. `earth-v3` had 510,341 fine lake cells
  against 124,276 coarse ones for that reason.
* `derive/run.py`: lake cells are excluded from the river candidate mask
  in both the connectivity pass and the extraction, so a river ends at a
  shore and starts again at the outlet instead of being skeletonised
  across the flat and the land bridges between a lake's pieces.

## 4. Measured

`worlds/earth-v5` is the first bake with all of this (and the 3000-step
tectonics of docs/plate-forces.md section 7). Results follow.
