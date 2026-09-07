# Developer guide: cross-stage contract

PLAN.md is the design.  This file pins the details the plan leaves open so
that stages written independently fit together.  Change it here first if
you must change a contract.

## Grids

| name | N | cell size | where |
|---|---|---|---|
| coarse | `N_c` | `cell_size_m` | `params.coarse_grid()` — all `coarse/*` fields |
| tect | `N_tect` (≤ N_c) | `cell_size_m·N_c/N_tect` | `params.tect_grid()` — tectonics internal only |
| fine | `N_c·R` | `cell_size_m/R` | `params.fine_grid()` — `fine/*` arrays and tiles |

All `Grid`s share `R_planet` (derived from the coarse grid).

## Coarse fields (`coarse/<name>.f{0..5}.npy`, interior `N_c×N_c`, `[i, j]`)

| stage | field | dtype | units / meaning |
|---|---|---|---|
| tectonics | `bedrock` | f32 | metres; shifted so `land_fraction` of cells are ≥ 0 |
| | `uplift` | f32 | metres per **erosion iteration** (already divided by `erosion.iterations`, scaled by `uplift_scale`) |
| | `hardness` | f32 | [0,1], 1 = hardest |
| | `plate_id` | i16 | plate index; every cell has one |
| | `plate_vel` | f32 (2) | contravariant coarse cells per tectonic step (vector field) |
| climate | `temperature` | f32 | °C |
| | `wind` | f32 (2) | contravariant coarse cells per advection step (vector field) |
| | `precip` | f32 | *volume* per cell per erosion iteration: rain × `cell_area/cell_size_m²`; mean over land = `precip_mean` (nondimensional particle-volume units) |
| | `evap` | f32 | per-iteration volume decay rate for particles (`k_evap·max(T,0)`) |
| erosion | `height` | f32 | bedrock surface after erosion, metres, sea level 0 |
| | `sediment` | f32 | metres of loose sediment on top of `height`; the terrain surface is `height + sediment` |
| | `discharge` | f32 | EMA of particle volume passing through the cell per iteration |
| | `momentum` | f32 (2) | EMA of volume-weighted particle velocity (contravariant cells/step) |
| hydro | `water_surface` | f32 | metres; `≥ surface`; equal to surface except in lakes; ocean cells = 0 |
| | `flow_dir` | u8 | D8 code 0..7 (below), 255 = ocean / no outflow (endorheic pit is impossible after priority flood, so 255 ⇒ ocean) |
| | `flow_acc` | f32 | accumulated `precip` volume (includes the cell's own) |
| watersheds | `basin_id` | i32 | −1 = ocean; otherwise a basin id from `graph/basins.json` |
| derive | `biome` | u8 | biome code (see `derive/biomes.py`); 0 = ocean |

`surface = height + sediment` everywhere; `ocean = surface < 0` (hydro may
re-quantile `height` so that exactly `land_fraction` is land).

## D8

```
D8_OFFSETS = [(1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1)]  # (di, dj) for code 0..7
```
The downstream cell of `(f, i, j)` with code `k` is the extended cell
`(f, i+H+di, j+H+dj)` mapped through `grid.owner` (a flat index into the
`(6, NE, NE)` extended array; use `grid.unflat_index` to get `(f', ei', ej')`
and subtract `H`).  Codes are defined in `globe/hydro/d8.py` together with
`downstream_flat(owner, N, H, f, i, j, code)`.

## Fine data (`fine/<name>.f{0..5}.npy`, `N_fine×N_fine`, `[i, j]`, no halo)

Written with `np.lib.format.open_memmap` so a face is never fully resident
when it is large.  Producers and consumers:

| field | dtype | producer | consumers |
|---|---|---|---|
| `height`, `sediment`, `water_surface`, `discharge` | f32 | refine | derive, tiles |
| `hardness` | f32 | refine (upsampled) | tiles |
| `basin_id` | i32 | refine (upsampled, −1 ocean) | derive, tiles |
| `biome`, `vegetation`, `river_mask` | u8 | derive | tiles |

`tiles` treats a missing fine field as zeros.  The stub `refine` writes
plain upsampled coarse fields so `derive`/`tiles` can be developed alone.

## Basins

* Basins never cross a cube-face edge: a channel crossing an edge is cut
  there and the upstream part becomes a child basin whose outlet is the
  first cell on the other face.  A refinement window is therefore a
  rectangle on one face (plus a halo that may extend past the edge; sample
  it with `FaceField.sample_window`).
* `graph/basins.json`: `{"basins": [{"id", "parent" (−1 root), "face",
  "outlet": [face, i, j], "downstream_basin" (id or −1 ocean), "area_cells",
  "bbox": [i0, j0, i1, j1] (exclusive), "order" (Strahler at outlet),
  "tiles": [[lod0_x, lod0_y], ...]}]}`

## Graph JSON

* `graph/drainage.json`: `{"nodes": [{"id", "cell": [f,i,j], "kind": "source|junction|outlet|lake_in|lake_out", "acc"}],
  "edges": [{"id", "from", "to", "order", "length_m", "mean_discharge", "cells": [[f,i,j], ...]}]}`
* `graph/lakes_coarse.json` (hydro) and `graph/lakes.json` (derive, fine):
  `{"lakes": [{"id", "surface_m", "area_m2", "outlet": [f,i,j] or null,
  "polygon": [[f,u,v], ...]}]}`
* `graph/rivers.json` (derive): `{"rivers": [{"id", "edge_id", "order",
  "points": [[f,u,v,width_m], ...]}]}`

## Erosion kernel contract (`globe/erosion/particle.py`)

The same kernel runs globally (6 faces, particles transfer across edges
with `cubesphere.transfer_velocity`) and inside a basin window (1 face
array, particles leaving the window or the mask die).  Inputs are plain
arrays shaped `(F, NE, NE[, 2])` plus `N, H`, `metric_inv`, `mask`
(uint8: 0 outside, 1 active, 2 active-but-frozen divide), and a flag
`spherical`.  Heights inside the kernel are in **cell units**
(`metres / cell_size_m`) so the ★ parameters apply as published.

## Quicklooks

Each stage module defines `quicklook(store, params, path)`.  Use
`globe.viz.quicklook` helpers; write the unfolded net.

## Tests

`bake/tests/test_<stage>.py`, runnable in < 60 s on the `tiny`/`small`
presets; never depend on another stage's real implementation when its stub
suffices.
