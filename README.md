# Globe Terrain Generation

Procedurally generated, finite, globe-shaped worlds: plate tectonics → climate →
particle hydraulic erosion → hydrology → watershed refinement → tiles, streamed
and rendered by a Godot 4 project.  See [PLAN.md](PLAN.md) for the full design.

```
bake/          Python bake tool (numpy + numba)          python -m pytest bake/tests
gdextension/   C++ GDExtension (tile decode, cube-sphere, drainage graph)
game/          Godot 4 project (streaming, gnomonic tile renderer, debug overlay)
```

## Bake

```sh
cd bake && pip install -e .
python scripts/bake.py --world demo --preset small           # ~1 minute test world
python scripts/bake.py --world big  --params my.yaml         # full world: hours, see below
python scripts/bake.py --world big  --from erosion           # resume from a stage
python scripts/inspect_world.py worlds/demo                  # stats + images
```

A default `big` bake is not a coffee break: at the default `N_c = 1024` an
erosion iteration measures 29 s on 4 cores (~20 s projected on 8), so the
default `erosion.iterations = 800` is ~6 h of erosion on 4 cores and ~4 h on
8 — not PLAN 8.4's "≈ 30 min global" (see *Deviations from PLAN.md*).  The
stage checkpoints every `erosion.checkpoint_every` iterations and `--from
erosion` resumes, so a long bake can be interrupted (only the two newest
checkpoints per parameter hash are kept; `erosion.iterations` is not part of
that hash, so lowering it resumes rather than recomputing from bedrock).

Stages: `tectonics, climate, erosion, hydro, watersheds, refine, derive, tiles`.
Every stage writes `worlds/<name>/quicklook/<stage>.png` (an unfolded cube net);
`manifest.json` records parameters and a content hash per stage.

`scripts/animate.py` re-runs a baked world's tectonics or erosion and writes
an animated WebP of it — plates drifting and colliding, or the drainage
network organising itself over the erosion iterations:

```sh
python scripts/animate.py --world worlds/demo --stage tectonics --out tect.webp
python scripts/animate.py --world worlds/demo --stage erosion   --out ero.webp
```

Further reading: [docs/DEVELOPING.md](docs/DEVELOPING.md) is the binding
cross-stage contract (field names, dtypes, units, JSON schemas).
[docs/erosion-tuning.md](docs/erosion-tuning.md) is the sweep that chose the
erosion defaults; re-run any row of it with
`python scripts/erosion_face_experiment.py`.

## Conventions (read before touching any stage)

* Faces `0..5 = +X, -X, +Y, -Y, +Z, -Z`, OpenGL cubemap bases; equi-angular
  cube mapping.  `globe.cubesphere` is the only module that knows how faces
  are oriented relative to each other.
* Arrays are indexed `[face, i, j]` with `i` along `u` and `j` along `v`;
  images are the transpose.  Every field carries an `H`-cell halo:
  `data[f, i + H, j + H]` is interior cell `(i, j)`; `field.interior` is a view.
* Tangent-vector fields (`wind`, `momentum`, `plate_vel`) hold *contravariant
  cell components* `(a, b)`: `(1, 0)` is "one cell along +i".  Halo exchange
  rotates them exactly into the neighbouring face's basis.
* `field.gradient()` is the physical gradient as such a vector (per metre);
  `vec_norm()` gives |∇f|, `vec_dot()` the metric dot product,
  `directional_derivative(vec)` the change per unit step along `vec`.
* Particles use `cubesphere.transfer_velocity` when they leave `[0, 1)²`.
* `from_sphere` returns `(u, v)` in the *closed* interval `[0, 1]`: a point
  exactly on a cube edge goes to the higher-priority face (X > Y > Z) with
  `u` or `v == 1.0`, so every cell index computed from it is
  `min(floor(u·N), N − 1)` (`FaceField.sample_*`, `Grid.halo` and the C++
  `tile_of` already clamp).
* D8 / neighbour lookups use `grid.owner` to map halo cells to their real
  owner cell on the neighbouring face.
* RNG: only `params.rng(stage, *keys)`; same seed + params ⇒ identical output.

## Deviations from PLAN.md

* `scripts/inspect.py` is named `scripts/inspect_world.py` because a script
  called `inspect.py` shadows the standard-library module.
* PLAN 2.1's half-open `[0, 1)` is the storage/cell convention;
  `from_sphere` returns the closed interval and consumers clamp (above).
* `flow_dir` is `uint8` (PLAN says `int8`; the 255 sentinel does not fit).
* `tiles/.../flow.png` is RGB8: R = log discharge, G = basin-local id,
  B = river mask (PLAN: RG8).
* `tiles/.../meta.json` carries `neighbors`: the `[lod, face, x, y]` of the
  4 edge-adjacent tiles (sides `+i, -i, +j, -j`, crossing cube edges via
  cubesphere), and `basins` (ids indexed by `flow.G`).  Tiles are `(T+1)²`
  *vertex* samples on fine-cell corners; a vertex shared by two tiles —
  also across a cube edge — is bit-identical in both (reduction of the
  same fine cells, no seam interpolation); `height.png` is the surface
  `height + sediment` (sediment depth in `layers.R`).  The LOD pyramid
  pools 3×3 vertex neighbourhoods across face edges (PLAN 10.3 says 2×2
  box filtering): heights are strided LOD-0 vertices, water/river mask
  max, ids/biome mode, other bytes mean.  `tiles/index.json` lists the
  LODs for the Godot loader (see `globe/io/tiles.py`, `globe/refine/lod.py`,
  docs/DEVELOPING.md).
* Climate writes an extra coarse field `evap`: the dimensionless
  evaporation multiplier `k_evap·max(T,0)` (~1 at `T_eq`) that erosion
  applies to `erosion.evap_rate`.
* The stub refine stage writes an intermediate `fine/<name>.f{0..5}.npy`
  per-face layout that the stub tiles stage reads; the real refine stage
  will replace this with per-basin output (see `stubs.stub_refine`).
* Halo exchange: edge halos use 4×4 cubic Lagrange interpolation by default
  (`exchange_halos(linear=True)` gives bilinear); corner halos use a local
  least-squares quadratic fit (`linear=True`: convex inverse-distance
  weights) rather than PLAN 2.4's "average of the two neighbours".
  Particle transitions use the exact `transfer_vector` rotation rather than
  PLAN 2.6's approximate re-expression.  PLAN 2.5's upwind gradient is
  deferred to the erosion stage.
* Collision uses a `ConcavePolygonShape3D` built from the tile's own
  projected vertices instead of PLAN 12.3's `HeightMapShape3D`: a
  gnomonically projected tile is a skewed quadrilateral and physics shapes
  cannot be skewed (the error was ~100 m on the small test planet).  Only
  LOD-0 tiles within `physics_radius_m` get a body; they are rebuilt on
  re-anchor, synchronously (collision must never lag the anchor frame).
  The faces come from `CubeSphere.build_collision_faces` (an extra
  GDExtension entry point beyond PLAN 12.1) and the body and its shape are
  reused in place; the GDScript fallback caches the anchor-independent
  unit-sphere direction of each vertex.  At `T = 64` that is 0.1 ms of
  projection plus ~5 ms of `ConcavePolygonShape3D.set_faces` per tile
  (was ~20 ms), i.e. ~24 ms for a 4-body re-anchor; the remainder is
  Godot's own shape upload and scales with `T²`, so a large `T` still
  costs a visible hitch every `reanchor_distance_tiles` of travel.
* `WorldRoot.effective_view_distance()` caps `view_distance_m` at
  `0.5 · R_planet` (0.5 rad of arc).  PLAN 12.2's flat gnomonic frame only
  holds near the anchor and `gnomonic_local` clamps `dot(Q, A)` at 0.05
  instead of rejecting far points, so without the cap a small planet
  streams the back hemisphere and smears it across the sky (on the `small`
  preset, `R_planet = 4074 m`: 57 tiles, vertices down to
  `dot(vertex, anchor) = -0.98` and 20× `R_planet` from the origin;
  with it 13 tiles, 0.56 and 1.5×).  It is a no-op on PLAN's default
  world (`R_planet ≈ 32.6 km` ≥ 2 × the 8 km default).  The sea-level
  plane keeps the raw `view_distance_m`: it is flat, so covering the
  horizon beyond the streamed tiles costs nothing.
* Debug mode 1 (F3) tints by the **global** basin id: each tile uploads
  its `meta["basins"]` map as the `basin_ids` uniform, since `flow.G` is
  only a tile-local index (a basin that crosses a tile edge has a
  different index on each side — 100 % of the 6223 shared-edge land
  samples on the medium test world).
* PLAN 8.4's erosion runtime target (≤ 2 s per iteration at `N_c = 1024`
  on 8 cores, "≈ 30 min global") is not met, and nothing in the repo
  measured the cost above `N_c = 256` before `bake/tests/test_perf.py`.
  Measured on 4 cores with the shipped `erosion` defaults (mean of 2
  iterations after warm-up, stub upstream fields): 1.30 s/iteration at
  `N_c = 256`, 4.94 s at 512, 29.0 s at 1024 (2.6 GB RSS).  Cost grows as
  ~cells^1.1 because the mean particle path grows with the grid (50 → 76
  → 119 steps); the `max_steps = 2·N` cap is not the driver (< 0.1 % of
  particles ever reach it).  A quarter of an iteration (8.1 s of the 29 s
  at `N_c = 1024`) is `particle.apply_changes`, which is serial by
  construction — the change list is applied in particle order, which is
  what makes a run byte-reproducible — so more cores cannot take the
  iteration below ~8 s there, and 8 cores project to ~20 s/iteration
  (trace halves, the rest does not).  The default
  `erosion.iterations = 800` at `N_c = 1024` is therefore a multi-hour
  stage; budget for it or lower `world.N_c` / `erosion.iterations`.
* Erosion holds the planetary datum every iteration
  (`erosion/maps.py: hold_datum`, plus a mean-free `apply_uplift`) instead
  of PLAN 8.2's plain `height += uplift`: uplift is a forcing with a
  positive area mean and mass leaves the surface for the deep ocean, so
  the surface drifts (measured on the small e2e world: land fraction
  0.30 → 0.56 over 60 iterations, a −164 m one-shot correction in hydro)
  and PLAN 9.1's "optionally re-quantile" then drowned the terrain
  erosion had just sculpted.  With the hold, hydro's re-quantile is a
  no-op and the coarse network keeps ~6 % more channel cells (2305 vs
  2183 at `N_c = 256`, 200 iterations) and a 30 % larger trunk basin.
* Content hashes ignore runtime-only knobs (`refine.workers`,
  `erosion.checkpoint_every/quicklook_every/resume`, `render.*`), and every stage
  records the hash of its own parameter group so a resume detects upstream
  parameter drift.

## Godot runtime (`game/`, `gdextension/`)

```sh
# 1. bake a world and make it visible to the project
(cd bake && python scripts/bake.py --world demo --preset small)   # -> bake/worlds/demo
ln -s "$PWD/bake/worlds/demo" game/data/worlds/demo      # or copy; also set globe/world_dir in project.godot

# 2. (optional but recommended) build the GDExtension for fast tile decode
git submodule update --init                              # godot-cpp (godot-4.4-stable)
(cd gdextension && scons platform=linux target=template_debug)  # -> game/addons/globe/bin

# 3. run
godot --path game                                        # WASD/QE move, Shift sprint, Esc mouse, F fly/walk, F3 debug

# headless tests
godot --headless --path game -s tests/test_stream.gd -- --world=/abs/path/to/world
make -C gdextension/tests test                           # C++ cube-sphere vs Python vector file
```

Without the compiled extension the project still runs: `TileSource` falls
back to a pure-GDScript 16-bit PNG decoder (slower loads, same output).

Rendering model (PLAN 12): tiles are flat `(T+1)²` grids; the vertex shader
maps `(i, j)` → face `(u, v)` → unit sphere (EAC) → gnomonic projection into
the anchor's tangent frame (`globe_e1/n/e2` shader globals), so wrap-around
and cube-edge crossing need no special cases.  `WorldRoot` re-anchors when
the player is more than one tile from the anchor.
