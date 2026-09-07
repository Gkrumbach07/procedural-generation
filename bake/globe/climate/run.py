"""Climate stage driver (PLAN.md section 7): temperature, wind, orographic
precipitation and evaporation on the coarse grid.

Input: ``bedrock`` (metres; the plan runs climate before erosion, so the
surface *is* the bedrock and ocean = ``bedrock < 0``).  Outputs (units in
docs/DEVELOPING.md): ``temperature`` °C, ``wind`` contravariant cells per
advection step (vector), ``precip`` volume per cell per erosion iteration
with land mean ``precip_mean``, ``evap`` dimensionless multiplier.

The stage draws no random numbers; it is deterministic by construction.
"""
from __future__ import annotations

import time

import numpy as np

from ..field import FaceField
from .precipitation import precipitation
from .temperature import evaporation, temperature
from .wind import cells_to_tangent3, geographic_frame, wind_field

OUTPUTS = ["temperature", "wind", "precip", "evap"]


def compute(grid, bedrock: FaceField, params, log=None) -> tuple[dict[str, FaceField], dict]:
    """Compute all four fields from a bedrock FaceField (halos exchanged).
    Returns ``({name: FaceField}, info)``."""
    cp = params.climate
    t0 = time.time()
    T = FaceField(grid, temperature(grid, bedrock.data, cp), name="temperature")
    wind = wind_field(grid, bedrock, cp)
    t1 = time.time()
    precip, pinfo = precipitation(grid, bedrock, wind, cp, log=log)
    t2 = time.time()
    evap = FaceField(grid, evaporation(T.data, cp), name="evap")
    li = bedrock.interior >= 0.0
    info = {
        "seconds_wind": round(t1 - t0, 3),
        "seconds_precip": round(t2 - t1, 3),
        "T_land_mean": float(T.interior[li].mean()) if li.any() else None,
        "T_min": float(T.interior.min()),
        "T_max": float(T.interior.max()),
        "wind_speed_median_cells": float(np.median(wind.vec_norm().interior) / grid.cell_size_m),
        **pinfo,
    }
    return {"temperature": T, "wind": wind, "precip": precip, "evap": evap}, info


def run(store, params, log=print) -> dict:
    grid = params.coarse_grid()
    bedrock = store.load_field("bedrock", grid)
    fields, info = compute(grid, bedrock, params, log=log)
    for name in OUTPUTS:
        store.save_field(fields[name])
    if log is not None:
        log(
            f"[climate] T {info['T_min']:.1f}..{info['T_max']:.1f} °C; precip land mean {info['precip_land_mean']:.3f} "
            f"median {info['precip_land_median']:.3f} max {info['precip_land_max']:.2f} (dry fraction {info['land_dry_fraction']:.2f}); "
            f"advection {info['seconds_precip']:.2f}s for {info['n_sweeps']} sweeps"
        )
    return info


def quicklook(store, params, path):
    """``climate.png``: land precipitation colour map (ocean dimmed to 30 %
    so only land carries the colour scale) + wind strokes + coastline;
    ``climate_temperature.png``: temperature with the coastline."""
    from ..viz import quicklook as ql

    grid = params.coarse_grid()
    bed = store.load_field("bedrock", grid)
    ocean = (bed.interior < 0.0).astype(np.int32)
    pr = store.load_field("precip", grid)
    wind = store.load_field("wind", grid)
    # sqrt scale: the field is peaked on windward coasts; sqrt keeps the
    # inland structure visible while the peaks still read as the wettest
    land = bed.interior >= 0.0
    top = float(np.percentile(pr.interior[land], 99)) if land.any() else float(pr.interior.max())
    img = ql.render_scalar(np.sqrt(np.maximum(pr.interior, 0.0)), vmin=0.0, vmax=float(np.sqrt(max(top, 1e-9))))
    img[ocean.astype(bool)] = (img[ocean.astype(bool)].astype(np.float32) * 0.3).astype(np.uint8)
    img = ql.render_vector(wind, stride=max(6, grid.N // 16), base=img)
    img = ql.contour_lines(img, ocean, (255, 255, 255))
    ql.save_image(path, img)
    T = store.load_field("temperature", grid)
    timg = ql.render_scalar(T, vmin=-30.0, vmax=30.0, cmap="bwr")
    timg = ql.contour_lines(timg, ocean, (0, 0, 0))
    tpath = path.with_name(f"{path.stem}_temperature{path.suffix}")
    ql.save_image(tpath, timg)
    return path


__all__ = ["OUTPUTS", "run", "compute", "quicklook", "cells_to_tangent3", "geographic_frame"]
