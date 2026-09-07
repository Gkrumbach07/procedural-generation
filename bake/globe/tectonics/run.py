"""Tectonics stage driver (PLAN.md section 6).  STUB — replaced by the real implementation."""
from __future__ import annotations

from ..stubs import stub_tectonics

OUTPUTS = ["bedrock", "uplift", "hardness", "plate_id", "plate_vel"]


def run(store, params, log=print) -> dict:
    return stub_tectonics(store, params, log)
