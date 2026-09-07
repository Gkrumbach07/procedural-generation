# Globe Terrain Generation — Implementation Plan
Procedurally generate a finite, globe-shaped world by simulating plate tectonics, climate, and particle-based hydraulic erosion (after Nick McDonald / weigert's work), partition the result into watersheds, refine each watershed in parallel, and bake everything to tiles that a Godot 4 project streams and renders.
This document is written to be handed to a coding agent. Everything in **Decisions** is locked; everything else is the recommended route with reasons, and the agent should push back in a PR description if it finds a materially better approach rather than silently diverging.
---
## 0. Goals, non-goals, decisions
### Goals
1. A deterministic offline **bake tool** (`bake/`) that turns `(seed, params)` into a complete world on disk: heights, water, rock/soil, climate, a drainage graph, and a watershed partition, at two resolutions (coarse global and fine per-tile).
2. A **Godot 4.x** project (`game/`) that streams the baked tiles, renders them as flat local terrain with correct wrap-around on the sphere, and exposes the drainage graph to gameplay.
3. Terrain that visibly has tectonic mountain belts, rain shadows, dendritic river networks with meanders, lakes, deltas and alluvial fans, and coastlines — not noise with erosion sprinkled on top.
### Non-goals (for now)
- Infinite or streaming *generation*. Everything is precomputed.
- Runtime erosion in-game. (Data formats should not preclude it later.)
- Visible planetary curvature. The sphere is topology only.
- Physical fidelity beyond "looks and behaves right." We are borrowing the *structure* of geomorphology models, not validating against real data.
### Decisions (locked)
| Topic | Decision |
|---|---|
| Bake language | Python 3.11+, NumPy, **Numba** for kernels. No `soillib` dependency. Kernels written so they can be ported to `numba.cuda` later (see §13). |
| Sphere representation | **Cube-sphere**, 6 faces, equi-angular projection (EAC). |
| Coarse global grid | `6 × N_c²`, default `N_c = 1024`. Configurable. |
| Fine grid | Refinement factor `R = 4` (default) → `6 × (N_c·R)²` fine cells, produced per watershed, stored as square tiles. |
| Tile size | `T = 256` fine cells per tile edge, with a 2× downsample LOD pyramid down to 1 tile per face. |
| Erosion model | McDonald 2023 "meandering" particle model (momentum map + discharge map + thermal erosion), extended with an **uplift** term and **hardness**-scaled erodibility. Not the 2026 SIGGRAPH momentum-conservation model (v2 candidate). |
| Tectonics | Clustered convection (McDonald 2020) reformulated on the unit sphere: plates are rigid rotations about Euler poles. |
| Lakes | Priority-flood depression filling (Barnes et al. 2014) as a post-pass, not in the particle loop. |
| Watershed partition | D8 flow routing → outlet labelling → size-bounded recursive splitting at tributary junctions (Pfafstetter-style). |
| Godot stack | Godot 4.x, GDScript for glue, **GDExtension (C++)** for tile decode and any hot loop. |
| Rendering | Flat local tiles. Anchor-based gnomonic projection into a local tangent frame; re-anchor as the player moves. |
---
## 1. Repository layout
```
globe-terrain/
├── README.md
├── PLAN.md                          # this file
├── bake/                            # Python bake tool
│   ├── pyproject.toml
│   ├── globe/
│   │   ├── __init__.py
│   │   ├── config.py                # dataclass params + YAML load, seed handling
│   │   ├── cubesphere.py            # face bases, EAC mapping, halo maps, face transitions
│   │   ├── field.py                 # FaceField: 6 arrays + halo exchange + gradient/laplacian
│   │   ├── tectonics/
│   │   │   ├── segments.py          # point cloud on sphere, Poisson-disc sampling
│   │   │   ├── plates.py            # plate clustering, rigid rotation, forces
│   │   │   ├── collision.py         # subduction / crystallization
│   │   │   └── run.py               # driver → bedrock, uplift, hardness, plate_velocity
│   │   ├── climate/
│   │   │   ├── temperature.py
│   │   │   ├── wind.py
│   │   │   └── precipitation.py     # orographic moisture advection
│   │   ├── erosion/
│   │   │   ├── particle.py          # numba kernels: descend, momentum, thermal
│   │   │   ├── maps.py              # discharge/momentum EMA update
│   │   │   └── run.py               # driver with uplift + hardness
│   │   ├── hydro/
│   │   │   ├── priority_flood.py
│   │   │   ├── routing.py           # D8, accumulation, drainage tree
│   │   │   └── watersheds.py        # partition + size-bounded splitting
│   │   ├── refine/
│   │   │   ├── upsample.py          # bilinear + detail noise conditioned on slope/hardness
│   │   │   ├── basin_job.py         # one watershed → fine cells (multiprocessing target)
│   │   │   └── rasterize.py         # basin results → tiles
│   │   ├── derive/
│   │   │   ├── rivers.py            # centerlines, Strahler order, width
│   │   │   ├── lakes.py
│   │   │   ├── biomes.py
│   │   │   └── soil.py
│   │   ├── io/
│   │   │   ├── world_store.py       # on-disk layout, npy/zarr, tiles, manifest
│   │   │   └── png16.py
│   │   └── viz/
│   │       ├── quicklook.py         # PNG dumps of any FaceField, hillshade, overlays
│   │       └── unfold.py            # 6 faces → cross-shaped net image
│   ├── scripts/
│   │   ├── bake.py                  # CLI: run all stages or a stage range, resume from cache
│   │   └── inspect.py               # dump stats / images for a baked world
│   └── tests/
├── gdextension/                     # C++ GDExtension
│   ├── SConstruct
│   ├── src/
│   │   ├── register_types.cpp
│   │   ├── tile_loader.cpp/.h       # decode 16-bit PNG/raw → PackedFloat32Array, build mesh arrays
│   │   ├── cubesphere.cpp/.h        # same math as bake/globe/cubesphere.py (must match bit-for-bit in tests)
│   │   └── drainage_graph.cpp/.h    # load graph, downstream/upstream queries
│   └── godot-cpp/                   # submodule
└── game/                            # Godot 4 project
    ├── project.godot
    ├── addons/globe/                 # the GDExtension .gdextension file + binaries
    ├── world/
    │   ├── WorldRoot.tscn / .gd      # anchor management, streaming
    │   ├── TerrainTile.tscn / .gd
    │   ├── terrain.gdshader          # vertex displacement + gnomonic placement
    │   ├── water.gdshader
    │   └── DebugOverlay.gd           # basin ids, flow arrows, tile bounds
    └── data/worlds/<world_name>/     # baked output copied/symlinked here
```
---
## 2. Cube-sphere coordinate system
All simulation and storage is on six square grids. Every stage that needs neighbors (gradients, diffusion, particle movement, flow routing) must work correctly across face edges. Get this right in Phase 1 and never think about it again.
### 2.1 Faces and bases
Faces indexed `0..5` = `+X, -X, +Y, -Y, +Z, -Z`. Each face has an orthonormal basis `(right, up, normal)`. Use the standard OpenGL cubemap convention so that any reference cubemap tooling agrees with us.
Local face coordinates `(u, v) ∈ [0,1)²` map to a cell `(i, j) = (floor(u·N), floor(v·N))`.
### 2.2 Equi-angular mapping (EAC)
Plain cube mapping has ~2.6× cell-area variation between face center and corner. Use the equi-angular variant to bring that to ~1.3×:
```
s = tan((u - 0.5) · π/2)        # ∈ (-1, 1)
t = tan((v - 0.5) · π/2)
p = normalize(normal + s·right + t·up)        # 3D point on unit sphere
```
Inverse: given unit vector `p`, face = argmax |component| (with sign); then project onto that face's plane, `s = dot(p, right)/dot(p, normal)`, `u = atan(s)·2/π + 0.5`, same for `v`.
Provide `cubesphere.to_sphere(face, u, v) -> vec3` and `cubesphere.from_sphere(vec3) -> (face, u, v)`. These two functions are the entire seam abstraction. Implement once in Python (numba-jitted, vectorized) and once in C++; a test must confirm they agree to 1e-6 on 100k random points.
### 2.3 Cell metrics
Precompute per face, once per resolution, and cache:
- `cell_area[i,j]` (steradians × R²) — needed to make precipitation and erosion volume-correct.
- `metric[i,j]` = local (du, dv) → meters scale, for gradients in physical units.
The planet radius `R_planet` is derived from `N_c` and the target coarse cell size: `R_planet = N_c · cell_size_m · 4 / (2π)` approximately (face edge ≈ quarter circumference). Default cell size 50 m at `N_c = 1024` → ~32 km radius; at `N_c = 2048` → ~65 km. Log the derived radius at bake start.
### 2.4 Halo exchange
Every `FaceField` stores each face as an `(N + 2H) × (N + 2H)` array with a halo of width `H` (default 4). `exchange_halos()` fills the halo cells from neighboring faces.
Neighbor faces are rotated relative to each other, so do not hand-code the eight edge cases. Instead, at init, for every halo cell compute its 3D position via `to_sphere` (extrapolating `u`/`v` outside `[0,1)`), call `from_sphere`, and store `(src_face, src_u, src_v)`. `exchange_halos()` then bilinearly samples. Precompute the index/weight tables once per resolution (`HaloMap`), so exchange is a gather with fixed indices.
Corner halo cells (the 8 cube corners have only 3 faces meeting) are undefined by this scheme; fill them by averaging the two valid neighbors. They only affect stencils at 8 cells on the planet.
### 2.5 Stencils
Provide on `FaceField`, all halo-aware and metric-aware:
- `gradient()` → `(∂/∂x, ∂/∂y)` in m/m, central differences; **upwind** variant for use inside the erosion step.
- `laplacian()`.
- `sample_bilinear(face, u, v)` and `sample_bilinear_sphere(vec3)`.
### 2.6 Particle face transitions
Particles carry `(face, u, v)` and a velocity in local face tangent coordinates `(vu, vv)`. After each step, if `u` or `v` leaves `[0,1)`:
```
p3   = to_sphere(face, u, v)                     # still valid slightly outside the face
vel3 = vu · right[face] + vv · up[face]          # approximate tangent vector
face', u', v' = from_sphere(p3)
vu' = dot(vel3, right[face']); vv' = dot(vel3, up[face'])
```
Because the faces are rotated, the velocity re-expression is what keeps momentum continuous across the seam. Unit test: a particle moving in a straight line across an edge should trace a great circle to within one cell.
---
## 3. Data model and on-disk format
One world = one directory. All arrays are `float32` unless noted. All coarse fields are `FaceField`s stored as `.npy` per face (`<name>.f{0..5}.npy`) — simple, memory-mappable, and NumPy-native. Switch to zarr only if a single face exceeds ~1 GB.
```
worlds/<name>/
├── manifest.json            # seed, params, N_c, R, T, R_planet, stage completion + hashes
├── coarse/
│   ├── bedrock.f*.npy       # tectonics output, m
│   ├── uplift.f*.npy        # m per erosion-step
│   ├── hardness.f*.npy      # [0,1], 1 = hardest
│   ├── plate_id.f*.npy      # int16
│   ├── plate_vel.f*.npy     # (2,) tangent m/step — for future interleaved v2
│   ├── temperature.f*.npy   # °C
│   ├── wind.f*.npy          # (2,) tangent
│   ├── precip.f*.npy        # m/step (already area-weighted)
│   ├── height.f*.npy        # post-erosion, m, sea level = 0
│   ├── sediment.f*.npy      # m of loose sediment on top of bedrock
│   ├── discharge.f*.npy
│   ├── momentum.f*.npy      # (2,)
│   ├── water_surface.f*.npy # priority-flood surface; > height where lake
│   ├── flow_dir.f*.npy      # int8, D8 code 0..7, 255 = sink/ocean (cross-face aware)
│   ├── flow_acc.f*.npy      # accumulated precip volume
│   ├── basin_id.f*.npy      # int32, -1 = ocean
│   └── biome.f*.npy         # uint8
├── graph/
│   ├── drainage.json        # nodes: junctions/outlets; edges: reaches with order, length, mean discharge
│   ├── basins.json          # per basin: id, parent, outlet cell, area, tiles touched, bbox, order
│   └── lakes.json           # id, surface elevation, area, outlet, polygon (coarse cell ring)
└── tiles/
    └── L{lod}/f{face}/{x}_{y}/
        ├── height.png       # 16-bit PNG, normalized by tile min/max in meta
        ├── water.png        # 16-bit, water surface height (0 where none)
        ├── layers.png       # RGBA8: R=sediment depth, G=hardness, B=biome id, A=vegetation density
        ├── flow.png         # RG8: discharge (log-scaled), basin-local id
        └── meta.json        # min/max height, min/max water, basin ids present, neighbors
```
Tile addressing: `(lod, face, x, y)`, LOD 0 = finest, `T` cells per tile. Tiles include a 1-cell overlap on the +x/+y edges so meshes share vertices with neighbors and don't crack.
`manifest.json` records a content hash per stage so `bake.py --from erosion` can resume from cached upstream stages.
---
## 4. Phase 0 — Scaffold, config, viewer
**Deliverables**
- `bake/` package installable with `pip install -e .`; deps: `numpy`, `numba`, `scipy`, `pillow`, `pyyaml`, `tqdm`. Nothing else without justification.
- `config.py`: one `@dataclass WorldParams` with all knobs (§14) and `from_yaml`. A `seed: int` field; every RNG in the tool must derive from `np.random.default_rng(seed + stage_offset)`. Same seed + params ⇒ byte-identical output (test this).
- `scripts/bake.py --world NAME --params p.yaml [--from STAGE] [--to STAGE]`. Stages in order: `tectonics, climate, erosion, hydro, watersheds, refine, derive, tiles`.
- `viz/quicklook.py`: dump any `FaceField` as a hillshaded PNG, single face or unfolded net. Every stage writes a quicklook to `worlds/<name>/quicklook/<stage>.png`. This is the primary debugging tool; do it first.
- `tests/` with pytest; CI-ready but no CI required.
**Acceptance**: `bake.py` runs end-to-end on a stub pipeline (each stage writes noise) and produces the directory structure in §3.
---
## 5. Phase 1 — Cube-sphere core
**Deliverables**: everything in §2 as `cubesphere.py` and `field.py`.
**Tests**
- `to_sphere`/`from_sphere` round-trip at 1e-6 for 1M random points, including points within 1e-4 of edges and corners.
- Halo exchange of a smooth analytic function on the sphere (e.g. `f = p.x·p.y + p.z²`): halo values match the analytic value to bilinear-interp tolerance.
- `laplacian()` of a spherical harmonic is proportional to it (eigenfunction test) with no visible seam artifacts in the quicklook.
- Straight-line particle crossing an edge traces a great circle.
- Bit-identical results between the Python and C++ `to_sphere`/`from_sphere` on a fixed test vector file checked into the repo (`tests/data/cubesphere_vectors.bin`). The C++ side of this test lives in `gdextension/tests/`.
---
## 6. Phase 2 — Tectonics (clustered convection on a sphere)
Faithful to McDonald's 2020 method, but the sphere removes the boundary-artifact and gap-filling headaches: the domain has no edges, and rigid motion is a rotation.
### 6.1 State
- **Heat field** `heat: FaceField` on the coarse grid (or on a `N_c/4` grid — tectonics doesn't need erosion resolution; default `N_tect = N_c/2`). Initialized with low-frequency noise.
- **Segments**: `M` points on the unit sphere (`M ≈ 20k` default) from Poisson-disc sampling on the sphere (Mitchell's best-candidate on the sphere is fine; exact blue noise not required). Each: `pos: vec3`, `mass`, `thickness`, `density`, `age`, `plate_id`, `area`.
- **Plates**: `P` initial plates (`P ≈ 12–20`) by k-means on segment positions or flood-fill grouping. Each: `omega: vec3` (angular velocity vector; the axis is the Euler pole), `center_of_mass: vec3`, `mass`, `inertia`.
### 6.2 Per-step
1. **Label map**: nearest-segment for each grid cell → `segment_id: FaceField[int32]`. Use `scipy.spatial.cKDTree` on the 3D segment positions queried with the cell centers' 3D positions. This is the "GPU Voronoi" replaced by a KD-tree; at `6 × 512²` cells × 20k points it's well under a second. Accumulate `segment.area` from labels (rolling blend 0.99 as in the source).
2. **Forces**: sample `∇heat` at each segment (via `field.gradient` + `sample_bilinear_sphere`). Tangent force `f = convection · ∇heat`. Torque on the plate about the sphere center `τ = Σ pos × f`. `omega += dt · τ / inertia`, with damping.
3. **Move**: rotate every segment of the plate: `pos = rotate(pos, axis = omega/|omega|, angle = |omega|·dt)`. Recompute plate `center_of_mass`, `inertia`.
4. **Gaps**: cells whose nearest segment is farther than `gap_radius` are divergent boundaries. Spawn new segments there (Poisson-style, rejecting candidates too close to existing segments) with low thickness, high temperature-derived density, `age = 0`, assigned to the plate of the nearest existing segment. Cool the heat field under them (new crust) — this is the diffusion/subduction shader pair from the original.
5. **Collisions**: for each pair of segments from *different* plates within `collision_radius` (KD-tree pair query), the denser one subducts: transfer its mass and thickness to the survivor, remove it, and warm the heat field at that location. Cascade the height difference to the survivor's neighbors (§6.4).
6. **Crystallization**: per segment, `T = heat sampled at pos`; growth `G = k_G(1−T)(1−T−d_b)`, deposit density `D = k_D(1−T)/(1−k_D(1−T))`, dissolution at `0.05·G`. Update `mass`, `thickness`, `density`, `age += dt`.
7. **Heat**: diffuse (`heat += dt·k_diff·laplacian(heat)`), plus the local cool/warm events above.
### 6.3 Outputs (all on the coarse `N_c` grid, resampled from `N_tect` if different)
- `bedrock = thickness·(1 − density)` per segment, splatted to the label map, then smoothed with 2–4 cascade passes (§6.4) and a small Gaussian. Shift so the target land fraction (`land_fraction`, default 0.3) sits above 0 using a quantile.
- `uplift`: `(bedrock_now − bedrock_k_steps_ago) / k`, clamped `≥ 0` in collision zones, small positive baseline elsewhere, negative allowed in subsiding basins. Then divided by `erosion_steps` so it is "meters per erosion step".
- `hardness ∈ [0,1]`: increasing with `age` and `density` (old cool cratons hard), decreasing near current collision/rift zones (hot, fractured). `hardness = clamp(0.3 + 0.5·age_norm + 0.3·density_norm − 0.4·boundary_proximity)`.
- `plate_id`, `plate_vel` (tangent velocity `omega × pos` in m/step) — stored for v2 interleaving.
### 6.4 Cascading
Between adjacent label cells whose heights differ by more than `cascade_threshold`, move `cascade_rate · (Δh − threshold)/2` from high to low. 2–4 passes. This is the original's height cascade shader; it's what turns a step-function of segment heights into a plausible range with foothills.
### 6.5 Tests / acceptance
- Run 1000 steps at `N_tect = 256`. Quicklook must show: several distinct plates, linear mountain belts at convergent boundaries, ridges/scarps at divergent ones, no square-domain artifacts, no plate-less holes.
- Total segment mass changes only through crystallization/dissolution (conservation test on subduction transfers).
- Runtime target: < 5 min for 1000 steps at `N_tect = 512` on a laptop CPU.
---
## 7. Phase 3 — Climate
Cheap, but it decides where the interesting terrain happens. All fields on the coarse grid.
- **Temperature** `T(lat, h) = T_eq − k_lat·|lat|^1.5 − lapse·max(h, 0)`, lapse 6.5 °C/km. Latitude from `p.z` of the cell's sphere point (choose `+Z` as the pole).
- **Wind**: three-cell circulation by latitude band: easterlies 0–30°, westerlies 30–60°, easterlies 60–90°, with sign flip across the equator, smoothly blended at band edges. Express as a tangent vector per cell. Add a small deflection around high terrain (rotate wind by `k·∇h⊥`) so ranges steer it a little.
- **Precipitation** (orographic advection). Start a moisture field `m = m_ocean` over ocean cells, `0` on land. Iterate `n_advect` sweeps: `m_new = m − dt·(wind·∇m)` (semi-Lagrangian: sample `m` at `pos − wind·dt`), then rain out `rain = m·(k_base + k_oro·max(0, wind·∇h))`, `m −= rain`, re-source `m = m_ocean` over ocean. Accumulate `precip += rain`. Also add a latitude prior (ITCZ wet band near the equator, dry bands ~25°) as a multiplier. Normalize so mean land precipitation equals `precip_mean`.
- **Evaporation** `evap = k_evap · max(0, T)` — used by erosion as the particle volume decay rate multiplier.
Multiply `precip` by `cell_area` so it is a *volume* per cell per step. Erosion spawns particles proportional to it.
**Acceptance**: quicklook `precip` shows wet windward slopes, dry lee sides, and a wet equatorial band. Hand-check with a synthetic single ridge under uniform wind.
---
## 8. Phase 4 — Erosion
McDonald 2023 particle model with uplift and hardness. Runs on the coarse grid for the global pass; the same kernel runs inside each watershed for refinement (Phase 6), so write it once, parameterized by grid and by a *mask* (cells outside the mask are frozen).
### 8.1 State
`height` (starts as `bedrock`), `sediment` (0), `discharge`, `discharge_track`, `momentum (2,)`, `momentum_track (2,)`, `hardness`, `uplift`, `precip`, `evap`, `mask`.
### 8.2 One iteration
```
spawn N particles: cell chosen ∝ precip (alias-table sampling), volume = precip[cell]/expected_hits
for each particle (numba parallel loop; per-thread RNG):
    while volume > min_volume and steps < max_steps:
        n  = -upwind_gradient(height + sediment) at pos       # descent direction in tangent coords
        # momentum coupling (2023): stream momentum pushes the particle
        fm = momentum at pos (bilinear)
        if |fm| > 0 and |speed| > 0:
            speed += k_mom · dot(normalize(fm), normalize(speed)) / (volume + discharge_at_pos) · fm
        speed += dt · n / (volume · density) - dt · friction · speed
        # dynamic timestep: normalize so we land in an adjacent cell
        step = speed / max(|speed|, eps) · cell_size
        prev_pos = pos; pos += step                             # handle face transition (§2.6)
        if outside mask or into ocean: break
        # equilibrium sediment with discharge scaling
        dh   = h(prev_pos) - h(pos)
        c_eq = max(0, volume · |speed| · dh · (1 + k_disc · discharge_at_pos))
        # hardness lowers erodibility, sediment layer is fully erodible
        k_e  = deposition_rate · (1 - hardness_at_pos) if sediment_at_pos <= 0 else deposition_rate
        cdiff = k_e · (c_eq - sed_carried)
        sed_carried += cdiff
        remove_or_deposit(prev_pos, -cdiff · volume)             # write to sediment first, then bedrock; atomic
        # accumulate tracks
        discharge_track[cell] += volume
        momentum_track[cell]  += volume · speed
        volume *= (1 - dt · evap_at_pos)
    deposit remaining sed_carried at final pos
# after all particles:
discharge = lerp(discharge, discharge_track, ema);  momentum = lerp(momentum, momentum_track, ema); zero tracks
thermal_erosion(height + sediment): for each cell, for each of 8 neighbors, if slope > talus(hardness): move
    excess · thermal_rate downhill (sediment first). 1 pass.
height += uplift                                                # tectonic forcing
exchange_halos on everything
```
Notes for the implementer:
- Writes to grid cells from parallel particles need atomics. Under `numba.njit(parallel=True)` there are no float atomics; instead give each thread a private accumulation buffer for `discharge_track`/`momentum_track` (reduce at the end) and apply height changes through a per-particle **change list** that is applied serially after the parallel loop (cells × delta). This is also exactly how a `numba.cuda` port would look with `cuda.atomic.add`, so keep the kernel shape.
- `remove_or_deposit`: deposition always adds to `sediment`. Erosion removes from `sediment` first, then from `height` (bedrock), and bedrock removal is additionally scaled by `(1 − hardness)`.
- Particles that end on a lake cell (see §9) in the *refinement* pass deposit their load and stop; in the global pass there are no lakes yet.
- Keep every parameter in `WorldParams`; the values in the 2023 post are a good starting set (§14).
### 8.3 Global pass driver
`iterations` (default 800) of the above at `N_c`. Every 50 iterations write a quicklook and a checkpoint (`--resume`). Particle count per iteration `N = particles_per_cell · cells` (default 0.25).
### 8.4 Acceptance
- On a flat-ish noise + uplift test: dendritic networks form, ridgelines sharpen, valleys widen, and rivers meander in low-gradient zones with visible cutoff scars in the quicklook of `discharge` overlaid on hillshade.
- Uplifted belts stay mountainous through the run (uplift vs erosion balance); with `uplift = 0` they decay. Expose `uplift_scale` to tune.
- No seam artifacts across cube faces in `discharge` (particles cross edges).
- Runtime target: ≤ 2 s per iteration at `N_c = 1024` on 8 cores (≈ 30 min global). If far off, profile before optimizing; the change-list apply and the bilinear samples are the usual hot spots.
---
## 9. Phase 5 — Sea level, lakes, routing, drainage tree
Operates on the final coarse `height + sediment`.
1. **Sea level**: `height` is already shifted so 0 is sea level; `ocean = height < 0`. Optionally re-quantile to hit `land_fraction` exactly.
2. **Priority-flood** (Barnes 2014): seed a min-heap with all ocean-adjacent land cells at their elevation; pop lowest; for each unvisited neighbor (8-connected, **cross-face via halo index maps**) set `water_surface = max(neighbor height, current)` and push. Result: `water_surface ≥ height` everywhere, equal except in depressions. `lake = water_surface − height > lake_min_depth`. Endorheic basins are handled naturally (they fill to their spill point). Also produce the *filled* DEM for routing.
3. **D8 flow direction** on the filled DEM: steepest of 8 neighbors, with the cross-face neighbor table. Flat areas (lake surfaces) get directions via the priority-flood pop order (each cell drains toward the cell that flooded it), which guarantees no cycles.
4. **Flow accumulation**: topological order (sort by filled elevation descending, or process in reverse priority-flood order), `acc[cell] += precip[cell]; acc[downstream] += acc[cell]`. Store `flow_acc`.
5. **Drainage tree**: cells with `acc > river_threshold` are channel cells. Build reaches between junctions; compute Strahler order; outlet nodes at the ocean. Write `graph/drainage.json`.
6. **Lakes**: connected components of `lake` cells → `lakes.json` with surface elevation, outlet cell, coarse polygon.
**Acceptance**: every land cell has a path to an ocean cell or an endorheic lake outlet (assert, no cycles). Overlay of channels on hillshade looks like a river map. Priority-flood over `6 × 1024²` in < 30 s (heap in numba or use a Python heap on the filtered set; if too slow, implement the Barnes "priority-flood + plain queue" optimization).
---
## 10. Phase 6 — Watershed partition and per-basin refinement
### 10.1 Partition
- `basin_id[cell] = id of outlet cell` by following `flow_dir` (memoized). Ocean = −1.
- **Size-bound**: `basin_max_cells` (default 512² coarse), `basin_min_cells` (default 64²).
  - Recursively split any basin > max: find the channel junction in the basin whose upstream sub-tree is closest to half the basin's area, cut there, and the upstream sub-tree becomes a child basin with its own outlet (that junction cell). Repeat until all pieces ≤ max. Record `parent` so the hierarchy is preserved (Pfafstetter-style nesting).
  - Merge any basin < min into an adjacent basin sharing the longest boundary (coastal micro-basins), *or* into a "coastal strip" basin per stretch of coast.
- Output `basin_id` (post-split ids), `basins.json` (id, parent, outlet cell, area, bbox in coarse cells, list of tiles touched at LOD 0, Strahler order at outlet).
### 10.2 Refinement job (one basin, one process)
Input: the basin's bbox expanded by a halo of `H_b = 8` coarse cells, all coarse fields cropped to it, and the basin mask. Output: fine arrays over the bbox at `R×` resolution.
1. **Upsample** all fields bilinearly to `R×`. For height, use bicubic to avoid facets.
2. **Detail noise**: add FBM ridged noise with amplitude `∝ slope · (0.5 + 0.5·hardness)` and frequency tied to `cell_size_fine`, seeded by `(seed, basin_id)`. Keep it small (amplitude ≤ 0.5 × coarse cell height range); its job is to break bilinear smoothness, not to invent terrain.
3. **Local erosion**: run the Phase 4 kernel for `refine_iterations` (default 150) with:
   - `mask` = basin cells (+ halo) — particles leaving the basin are killed; divide cells (mask boundary) are **frozen** (no height change) so basins never capture each other's territory.
   - `precip` from the upsampled climate field; particles spawn only inside the basin.
   - `momentum`/`discharge` **initialized from the upsampled coarse maps** so the fine rivers start where the coarse ones were and meander from there rather than re-forming.
   - `uplift = 0` (already in the coarse result), thermal erosion on.
   - Particles reaching a lake cell (fine `water_surface`) deposit and stop; particles reaching the outlet cell are removed (outflow).
4. **Local lakes**: re-run priority-flood *within* the basin with the outlet as the only drain, so sub-coarse depressions get water surfaces consistent with the coarse ones.
5. Return the fine arrays for the bbox; the parent process rasterizes into tiles (only writing cells whose fine `basin_id` matches, so overlapping bboxes don't clobber each other) and blends a 2-cell feather across divides to hide any residual step.
Run with `multiprocessing.Pool` (processes = cores), largest basins first for load balance. Each job also writes a quicklook. Determinism: the basin RNG depends only on `(seed, basin_id)`, and the job's result must not depend on which worker ran it (test by running one basin twice).
### 10.3 Tiles and LOD pyramid
Rasterize LOD 0 tiles as in §3, then build LOD 1..L by 2×2 box downsampling of height (min/max preserved in meta), max-pooling of water depth, mode-pooling of ids. Write `height.png` as 16-bit normalized by tile min/max (stored in `meta.json`) — this gives ~cm precision per tile.
**Acceptance**: a full refine at `N_c = 1024, R = 4` finishes in < 1 h on 8 cores; seams at basin divides are invisible in a hillshade at LOD 0; rivers are continuous across tile borders and across basin outlets.
---
## 11. Phase 7 — Derived layers
- **Rivers**: from fine `discharge` (refined) thresholded by Strahler order; extract centerlines (thinning) per reach, fit Catmull-Rom splines, width `w = a · Q^b` (default `a = 2, b = 0.5` in cell units). Write `graph/rivers.json` with polylines in `(face, u, v)` and per-vertex width; also burn a river mask into `flow.png`.
- **Lakes**: polygons from fine water surface; write `lakes.json`.
- **Soil**: `sediment` depth from erosion (fine) — alluvium in valley floors and fans; `hardness` for rock.
- **Biomes** (coarse, then upsampled): Whittaker-style lookup on `(temperature, precip)` with elevation/slope overrides (alpine, cliff), plus `riparian` within `k` cells of a channel and `wetland` near lakes. Vegetation density from precip × (1 − slope) × biome factor. Written into `layers.png`.
---
## 12. Phase 8 — Godot runtime
### 12.1 GDExtension (`gdextension/`)
Build with `godot-cpp` matching the installed Godot 4.x minor version. Expose:
- `CubeSphere` static: `to_sphere(face,u,v) -> Vector3`, `from_sphere(Vector3) -> [face,u,v]`, `tile_of(face,u,v,lod) -> Vector3i`.
- `TileLoader`: `load(world_dir, lod, face, x, y) -> Dictionary{height: PackedFloat32Array, water: PackedFloat32Array, layers: Image, meta: Dictionary}`; decodes 16-bit PNG via Godot's `Image` and de-normalizes with meta. Runs on a worker thread (`WorkerThreadPool`), returns via signal. Also `build_mesh_arrays(height, T) -> Array` for `ArrayMesh` if we don't use vertex-texture displacement (see 12.3).
- `DrainageGraph`: `load(world_dir)`, `downstream(cell) -> Array`, `basin_of(face,u,v) -> int`, `nearest_reach(Vector3) -> Dictionary`, `outlet(basin) -> ...`.
### 12.2 World coordinates and the anchor
The player has a sphere position `P` (unit `Vector3`) and heading. The scene has an **anchor** `A` (unit `Vector3`) and an orthonormal tangent frame `(e1, e2, n = A)`. A world point `Q` on the sphere at height `h` renders at:
```
gnomonic:  d = Q / dot(Q, A)                 # project onto tangent plane at A
local xyz  = ( dot(d, e1)·R_planet, h, dot(d, e2)·R_planet )
```
Re-anchor when `|P − A| · R_planet > reanchor_distance` (default 1 tile edge): set `A = P`, rebuild the frame, and update the uniforms on every loaded tile. Distortion at distance `d` is ~`(d/R)²`; at 10 km on a 32 km planet that is ~10%, at 65 km radius ~2.4%. If this reads badly in play, increase `R_planet` (via `N_c`) or reduce view distance; do **not** try to fix it with a curved renderer in v1.
Wrap-around comes free: `P` is on the sphere, tiles are looked up by `from_sphere`.
### 12.3 Terrain rendering
- Each visible tile is a `MeshInstance3D` with a shared flat `T×T` grid mesh (positions in tile-local `(i, j)`), and a `ShaderMaterial` with uniforms: `face, tile_x, tile_y, lod, height_tex, water_tex, layers_tex, anchor frame, min/max height`.
- `terrain.gdshader` vertex stage: `(i,j)` → `(u,v)` → `to_sphere` (GLSL port of §2.2) → gnomonic → local xyz, `y = height sampled from height_tex`. Normals from finite differences of the height texture in the fragment stage. Skirts on tile edges hide LOD cracks.
- LOD: choose `lod` per tile by distance to the player with a hysteresis band; keep a quadtree per face.
- Streaming: `WorldRoot.gd` maintains the set of tiles within `view_distance` at the appropriate LOD, requests loads from `TileLoader`, and frees tiles outside a larger unload radius. Budget 2–4 loads per frame.
- Water: a second mesh per tile using `water_tex`, only where water depth > 0, simple shader.
- Collision: `HeightMapShape3D` only for LOD-0 tiles within `physics_radius`; created from the decoded `PackedFloat32Array`.
### 12.4 Debug overlay
Toggleable: tile bounds and ids, basin id color tint, flow arrows on channel cells, a minimap showing the unfolded cube net with the player and anchor. This is the tool that catches seam and wrap bugs; build it with the streaming, not after.
**Acceptance**: walk in a straight line around the planet and return to the start without a hitch; cross a cube edge with no visible seam; river spline from `DrainageGraph` follows the rendered river.
---
## 13. Phase 9 — Performance and hardening
- Profile the bake. Likely candidates: particle kernel change-list apply (sort by cell and reduce), bilinear sampling (precompute index/weight), priority-flood heap.
- Optional `numba.cuda` backend for `erosion/particle.py`: same kernel with `cuda.atomic.add` on `float32` grids. Gate behind `params.backend = "cuda"`. Expected 20–50× on the global pass.
- Optional interleaved tectonics/erosion (v2): every `k` erosion iterations run a tectonics step and semi-Lagrangian advect `height`, `sediment` with `plate_vel`. Only attempt after v1 output is judged good.
- Optional swap of the erosion model for the 2026 momentum-conservation formulation (quasi-static transport, Monte Carlo streamline integration, depth-averaged momentum equation). The data model already carries `discharge`/`momentum`; this changes only `erosion/`.
---
## 14. Default parameters (`WorldParams`)
Start with these; all are tunable via YAML. Values marked ★ are taken directly from McDonald's published code and should only be changed with a reason.
| Group | Param | Default |
|---|---|---|
| world | `seed` | 0 |
| | `N_c` | 1024 |
| | `cell_size_m` | 50 |
| | `R` (refine factor) | 4 |
| | `T` (tile edge) | 256 |
| | `land_fraction` | 0.30 |
| tectonics | `N_tect` | 512 |
| | `segments` | 20000 |
| | `initial_plates` | 16 |
| | `steps` | 1500 |
| | `convection` ★ | 10.0 |
| | `growth` ★ | 0.05 |
| | `dissolution_factor` ★ | 0.05 |
| | `collision_radius` | 1.5 × mean segment spacing |
| | `cascade_rate` ★ | 0.3 |
| | `uplift_scale` | 1.0 |
| climate | `T_eq`, `k_lat`, `lapse` | 28, 45, 6.5 |
| | `m_ocean`, `k_base`, `k_oro`, `n_advect` | 1.0, 0.02, 0.5, 200 |
| | `precip_mean` | 1.0 (nondimensional; particle volume unit) |
| erosion | `iterations` | 800 |
| | `particles_per_cell` | 0.25 |
| | `dt` ★ | 1.2 |
| | `density` ★ | 1.0 |
| | `friction` ★ | 0.05 |
| | `deposition_rate` ★ | 0.1 |
| | `evap_rate` ★ | 0.001 |
| | `k_mom` (momentumTransfer) ★ | 1.0 |
| | `k_disc` | 1.0 |
| | `ema` (map lerp) ★ | 0.1 |
| | `thermal_rate`, `talus_slope_soft/hard` | 0.5, 0.6/1.2 (rise/run) |
| | `min_volume` ★, `max_steps` | 0.01, 2·N |
| hydro | `lake_min_depth`, `river_threshold` | 0.5 m, 200 cells of accumulation |
| watersheds | `basin_max_cells`, `basin_min_cells` | 512², 64² |
| refine | `refine_iterations`, `detail_amp`, `halo_cells` | 150, 0.3, 8 |
| render | `view_distance`, `reanchor_distance`, `physics_radius` | 8 km, 1 tile, 500 m |
---
## 15. Testing and validation strategy
- **Determinism**: `bake.py` twice with the same seed → identical hashes for every stage (part of `tests/test_determinism.py`, run at `N_c = 128`).
- **Seams**: every stage's quicklook is also rendered as an unfolded net; a test computes the max gradient discontinuity across face edges relative to the interior and fails if > 3×.
- **Conservation**: erosion test asserts `Σ(height + sediment)·area + Σ sediment_in_flight` is constant when `uplift = 0` and no particles exit to ocean.
- **Hydrology**: no cycles in `flow_dir`; every land cell reaches an outlet; `flow_acc` at an outlet equals the sum of precip over its basin.
- **Watersheds**: partition covers all land, no overlaps, all basins within size bounds, hierarchy is a tree.
- **Refinement**: divide cells unchanged; rerunning one basin gives identical output; fine rivers coincide with coarse rivers at LOD 2.
- **Godot**: cube-sphere vector test vs the Python file; a headless scene test that streams a ring of tiles around a point on a cube edge and asserts no missing tiles.
Small-world profile for tests: `N_c = 128, N_tect = 64, R = 2, T = 64` runs the whole pipeline in ~1 minute.
---
## 16. Milestones (suggested order of work)
1. Phase 0 + Phase 1 (scaffold, viewer, cube-sphere) — nothing else is testable without these.
2. Phase 4 erosion on a **single flat face** with synthetic uplift, to validate the kernel visually before the sphere plumbing matters. Then Phase 5 on that face. Look at the watershed partition of that result and decide whether basin sizes feel right before continuing.
3. Phase 2 tectonics, then Phase 3 climate, then rerun Phases 4–5 globally.
4. Phase 6 refinement + tiles.
5. Phase 8 Godot loader + renderer, with the debug overlay. Walk around the planet.
6. Phase 7 derived layers once terrain is signed off (biomes and rivers are cheap to redo).
7. Phase 9 as needed.
---
## 17. References
- McDonald, N. "Simple Particle-Based Hydraulic Erosion" (2020). https://nickmcd.me/2020/04/10/simple-particle-based-hydraulic-erosion/
- McDonald, N. "Procedural Hydrology" (2020). https://nickmcd.me/2020/04/15/procedural-hydrology/
- McDonald, N. "GPU Accelerated Voronoi" (2020). https://nickmcd.me/2020/08/01/gpu-accelerated-voronoi/
- McDonald, N. "Clustered Convection for Simulating Plate Tectonics" (2020). https://nickmcd.me/2020/12/03/clustered-convection-for-simulating-plate-tectonics/
- McDonald, N. "SoilMachine" (2022). https://nickmcd.me/2022/04/15/soilmachine/
- McDonald, N. "Meandering Rivers in Particle-Based Hydraulic Erosion Simulations" (2023). https://nickmcd.me/2023/12/12/meandering-rivers-in-particle-based-hydraulic-erosion-simulations/
- McDonald, N. & Cordonnier, G. "Stochastic Geomorphological Transport for Terrain Erosion Simulation." ACM TOG 45(4), SIGGRAPH 2026. https://erosiv.studio/publications/stochastic-geomorphological-transport
- Code: https://github.com/weigert/SimpleHydrology, https://github.com/weigert/SimpleTectonics, https://github.com/weigert/SoilMachine, https://github.com/weigert/soillib, https://github.com/erosiv/geotransport
- Barnes, R., Lehman, C., Mulla, D. "Priority-flood: An optimal depression-filling and watershed-labeling algorithm for digital elevation models." Computers & Geosciences (2014).
- Cordonnier, G. et al. "Large Scale Terrain Generation from Tectonic Uplift and Fluvial Erosion." Eurographics (2016). — the canonical tectonics + fluvial coupling.
- Verdin, K. & Verdin, J. "A topological system for delineation and codification of the Earth's river basins" (Pfafstetter coding). J. Hydrology (1999).
- Equi-angular cube mapping: Google "EAC" (Equi-Angular Cubemap) projection notes.
