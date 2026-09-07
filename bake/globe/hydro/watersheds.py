"""Watershed partition stage (PLAN.md section 10.1).  STUB — replaced by the real implementation."""
from __future__ import annotations

from ..stubs import stub_watersheds

OUTPUTS = ["basin_id", "graph/basins.json"]


def run(store, params, log=print) -> dict:
    return stub_watersheds(store, params, log)
