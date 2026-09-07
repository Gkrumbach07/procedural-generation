"""Erosion stage driver (PLAN.md section 8).  STUB — replaced by the real implementation."""
from __future__ import annotations

from ..stubs import stub_erosion

OUTPUTS = ["height", "sediment", "discharge", "momentum"]


def run(store, params, log=print) -> dict:
    return stub_erosion(store, params, log)
