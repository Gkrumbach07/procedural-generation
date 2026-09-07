"""Climate stage driver (PLAN.md section 7).  STUB — replaced by the real implementation."""
from __future__ import annotations

from ..stubs import stub_climate

OUTPUTS = ["temperature", "wind", "precip", "evap"]


def run(store, params, log=print) -> dict:
    return stub_climate(store, params, log)
