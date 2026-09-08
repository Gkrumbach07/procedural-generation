"""Erosion runtime scaling (PLAN.md section 8.4, "runtime target").

PLAN 8.4 asks for ≤ 2 s per erosion iteration at ``N_c = 1024`` on 8 cores.
The shipped kernel does not meet that (README, "Deviations from PLAN.md",
records the measurement and the resulting bake budget), and until this file
nothing in the repo measured the cost above ``N_c = 256`` at all — the other
erosion tests all run on ``N ≤ 128`` grids, where the per-iteration cost is
milliseconds and the trace/apply split is invisible.

So this test pins the one number the plan cares about, one grid step above
the rest of the suite: a full ``maps.step`` at ``N_c = 512`` with the
shipped ``erosion`` defaults.  The ceiling is deliberately loose (a factor
~2 over the measured 6 s on 4 cores) — it is there to catch an order-of-
magnitude regression in ``particle.trace_particles`` / ``apply_changes``
(a lost ``parallel=True``, a re-JIT every iteration, an accidental
``max_steps`` blow-up), not to police a few per cent.

Marked ``slow`` (~1 min): run the suite with ``-m "not slow"`` to skip it.
"""
from __future__ import annotations

import time

import numpy as np
import pytest
from numba import get_num_threads

from globe.config import WorldGroup, WorldParams
from globe.erosion import run as erosion_run
from globe.erosion.maps import step
from globe.io.world_store import WorldStore
from globe.stubs import stub_climate, stub_tectonics

#: grid and per-iteration ceiling (seconds, mean of TIMED iterations)
N_C = 512
MAX_SECONDS = 15.0
TIMED = 2


def _log(_msg):
    pass


@pytest.mark.slow
def test_erosion_iteration_cost_at_n512(scratch):
    if get_num_threads() < 2:
        pytest.skip("needs at least 2 numba threads to be comparable to the recorded measurement")
    p = WorldParams()
    p.world = WorldGroup(seed=1, N_c=N_C, cell_size_m=50.0, R=2, T=64, land_fraction=0.3)
    store = WorldStore(scratch / "perf_n512", create=True)
    store.init_manifest(p, p.coarse_grid())
    for name, fn in (("tectonics", stub_tectonics), ("climate", stub_climate)):
        fn(store, p, _log)
        store.mark_stage(name, [], {"stub": True}, 0.0, params=p)
    state = erosion_run.build_state(store, p)

    step(state, p, 0)  # warm-up: JIT (the kernels are cache=True, but the first call still binds)
    ts = []
    for it in range(1, 1 + TIMED):
        t0 = time.time()
        st = step(state, p, it)
        ts.append(time.time() - t0)
        assert st["particles"] == round(p.erosion.particles_per_cell * 6 * N_C * N_C)
    mean = float(np.mean(ts))
    print(f"\n[perf] N_c={N_C}, {get_num_threads()} threads: {mean:.2f} s/iteration {ts}")
    assert mean < MAX_SECONDS, (mean, ts)
    # the particle pass is the iteration: routing/thermal/uplift are minor
    assert st["seconds_particles"] < st["seconds_total"]
