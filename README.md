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
* D8 / neighbour lookups use `grid.owner` to map halo cells to their real
  owner cell on the neighbouring face.
* RNG: only `params.rng(stage, *keys)`; same seed + params ⇒ identical output.

Deviation from PLAN.md: `scripts/inspect.py` is named `scripts/inspect_world.py`
because a script called `inspect.py` shadows the standard-library module.
