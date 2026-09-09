"""Add fractal detail to the bedrock that tectonics hands to erosion.

    python3 scripts/inject_bedrock_detail.py --world W [--beta 2.0] [--amp 0.15]
    python3 -m globe.cli --world W --from erosion --to erosion --force \
        --set erosion.resume=false

An experiment, not a pipeline stage — it rewrites `coarse/bedrock` in place,
so run it on a copy.

Why it exists: docs/terrain-realism.md measures the bedrock leaving
tectonics at a spectral slope of about 10.7 over 200–1600 m, where real
topography sits near 2. Erosion pulls that down (to ~4.6 on the coarse
grid) but cannot manufacture variance that was never in its input, and
every erosion parameter measured inert against it. McDonald's simulations
start from fractal noise, which has the right slope by construction; ours
start from a tectonic field that is almost pure long-wavelength swell.
That difference — not the erosion, and not glaciers or lakes, which
measured as no-ops — is what separates the two.

Measured on the `small` preset, 60 iterations, same tectonics and climate:

    tectonics as-is                   beta 3.71  (200-1600 m)
    tectonics, no glaciers/lakes      beta 3.61
    tectonics + fBm detail            beta 1.84   <- real-topography range

Caveats before this becomes a stage:

* Noise is generated per face, so it does NOT match across face seams.
  A real version has to be generated on the sphere (or in a seam-aware
  way) or every cube edge will show.
* `--amp` is a fraction of the bedrock's own 5–95 % relief; 0.15 landed
  beta at 1.84 but overshot at long wavelengths (0.30 over 800–6400 m),
  so the amplitude and the low-frequency cutoff both want tuning.
* Injecting detail into bedrock changes the mass budget that `hold_datum`
  and the land-fraction quantile balance against. Land fraction moved
  14.4 % -> 14.0 % in the test, which is small but not nothing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def fbm_like(n: int, beta: float, rng: np.random.RandomState, cut_cells: float = 32.0) -> np.ndarray:
    """Unit-variance field with power ~ k^-beta, with wavelengths longer
    than `cut_cells` removed — tectonics already owns those, and leaving
    them in would fight the plate-scale relief rather than add detail."""
    ky, kx = np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n), indexing="ij")
    kr = np.hypot(kx, ky)
    kr[0, 0] = 1e-9
    F = (rng.randn(n, n) + 1j * rng.randn(n, n)) * kr ** (-beta / 2.0)
    F[0, 0] = 0.0
    F[kr < 1.0 / cut_cells] = 0.0
    out = np.real(np.fft.ifft2(F))
    s = out.std()
    return out / s if s > 0 else out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", required=True, type=Path)
    ap.add_argument("--beta", type=float, default=2.0, help="target spectral slope of the injected detail")
    ap.add_argument("--amp", type=float, default=0.15, help="1 sigma, as a fraction of bedrock 5-95%% relief")
    ap.add_argument("--cut-cells", type=float, default=32.0, help="drop injected wavelengths longer than this")
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()

    rng = np.random.RandomState(a.seed)
    coarse = a.world / "coarse"
    for f in range(6):
        p = coarse / f"bedrock.f{f}.npy"
        z = np.load(p).astype(np.float64)
        land = z > 0
        if not land.any():
            continue
        span = float(np.percentile(z[land], 95) - np.percentile(z[land], 5))
        nz = fbm_like(z.shape[0], a.beta, rng, a.cut_cells)
        amp = a.amp * span
        np.save(p, (z + amp * nz).astype(np.float32))
        print(f"  face {f}: relief {np.ptp(z):8.1f} m  ->  injected {amp:6.1f} m (1 sigma)")
    print("bedrock rewritten; re-run erosion with erosion.resume=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
