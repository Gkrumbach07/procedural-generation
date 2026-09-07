"""Hydro stage driver (PLAN.md section 9).  STUB — replaced by the real implementation."""
from __future__ import annotations

from ..stubs import stub_hydro

OUTPUTS = ["water_surface", "flow_dir", "flow_acc", "graph/drainage.json", "graph/lakes.json"]


def run(store, params, log=print) -> dict:
    return stub_hydro(store, params, log)
