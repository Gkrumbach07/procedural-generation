"""Globe terrain bake tool.

Deterministic offline pipeline: tectonics -> climate -> erosion -> hydro ->
watersheds -> refine -> derive -> tiles, all on a cube-sphere.

Conventions that every module relies on (see PLAN.md section 2):

* Six faces 0..5 = +X, -X, +Y, -Y, +Z, -Z, OpenGL cubemap bases.
* Face-local coordinates (u, v) in [0, 1); cell (i, j) = (floor(u*N), floor(v*N)).
* Arrays are indexed ``data[face, i, j]`` (first grid axis is u/i, second is v/j).
  Images are therefore the transpose of a face array.
* Every field carries a halo of ``H`` cells on each side, so a face array has
  shape ``(N + 2H, N + 2H)``; interior cell (i, j) lives at ``[i + H, j + H]``.
* Tangent-vector fields hold *contravariant cell components* ``(a, b)``:
  the vector ``a * E_i + b * E_j`` where ``E_i = d(position)/d(i)``; i.e.
  ``(1, 0)`` is "one cell along +i".  Halo exchange re-expresses them exactly
  in the neighbouring face's basis.
"""

__version__ = "0.1.0"
