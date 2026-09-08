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
python scripts/bake.py --world big  --params my.yaml         # full world
python scripts/bake.py --world big  --from erosion           # resume from a stage
python scripts/inspect_world.py worlds/demo                  # stats + images
```

Stages: `tectonics, climate, erosion, hydro, watersheds, refine, derive, tiles`.
Every stage writes `worlds/<name>/quicklook/<stage>.png` (an unfolded cube net);
`manifest.json` records parameters and a content hash per stage.

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
  re-anchor.
* Content hashes ignore runtime-only knobs (`refine.workers`,
  `erosion.checkpoint_every/quicklook_every`, `render.*`), and every stage
  records the hash of its own parameter group so a resume detects upstream
  parameter drift.

## Godot runtime (`game/`, `gdextension/`)

```sh
# 1. bake a world and make it visible to the project
python bake/scripts/bake.py --world demo --preset small
ln -s "$PWD/bake/worlds/demo" game/data/worlds/demo      # or copy; also set globe/world_dir in project.godot

# 2. (optional but recommended) build the GDExtension for fast tile decode
git submodule update --init                              # godot-cpp (godot-4.4-stable)
cd gdextension && scons platform=linux target=template_debug   # -> game/addons/globe/bin

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
