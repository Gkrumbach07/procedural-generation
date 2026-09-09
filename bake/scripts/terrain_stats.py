"""Objective terrain-realism metrics: how dissected is a height field?

    python3 scripts/terrain_stats.py <run>/final.npz [...]

Three numbers, each against a real-world reference, so "the terrain looks
bland" becomes something measurable:

* **beta** -- radial power-spectrum slope, P(k) ~ k^-beta, fitted over
  60-600 m wavelengths.  Real topography sits near 2.  Higher means
  variance dies off too fast toward small scales, which is what "smooth"
  and "bland" actually are.  Validated on synthetic fBm of known slope:
  accurate to +/-0.1 over beta = 1..5.
* **drainage density** -- channel length per unit area.  Real landscapes
  run 2-20 km/km2, humid temperate 4-8.
* **hillslope length** -- median distance from a cell to its nearest
  channel.  Real values are 50-200 m.

The last two are the ones that bite: a landscape can have the right
relief and the right slopes and still read as bland if the valley network
is too coarse, leaving large undissected interfluves.  Measured on the
shipped final512 world: beta 4-6, drainage density 1.89 km/km2, median
hillslope 403 m -- under-dissected by 2-8x.  See docs/terrain-realism.md.
"""
import os
import sys

import numpy as np

def radial_psd(z, dx):
    z = np.asarray(z, np.float64); z = z - z.mean()
    n = min(z.shape); z = z[:n, :n]
    w = np.hanning(n)[:, None] * np.hanning(n)[None, :]
    F = np.fft.fftshift(np.abs(np.fft.fft2(z * w))**2)
    ky, kx = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(n, dx)),
                         np.fft.fftshift(np.fft.fftfreq(n, dx)), indexing='ij')
    kr = np.hypot(kx, ky).ravel(); P = F.ravel()
    bins = np.logspace(np.log10(2.0/(n*dx)), np.log10(kr.max()), 30)
    idx = np.digitize(kr, bins); ks, Ps = [], []
    for b in range(1, len(bins)):
        m = idx == b
        if m.sum() > 6: ks.append(kr[m].mean()); Ps.append(P[m].mean())
    return np.array(ks), np.array(Ps)

def beta(z, dx, lo=60.0, hi=600.0):
    """-slope of log P vs log k over wavelengths [lo, hi] metres."""
    k, P = radial_psd(z, dx)
    lam = 1.0 / k
    m = (lam >= lo) & (lam <= hi) & (P > 0)
    if m.sum() < 4: return float('nan'), 0
    A = np.polyfit(np.log10(k[m]), np.log10(P[m]), 1)
    return -A[0], int(m.sum())

def dissection(h, q, active, dx, support=12.0):
    """Drainage density (km/km2) and median hillslope length (m).

    A channel is a cell whose discharge exceeds `support` times the mean --
    a support-area threshold, the standard way channel heads are picked out
    of a DEM.  Precipitation is uniform, so discharge is proportional to
    upstream area and q/mean(q) is the area ratio; that makes the threshold
    independent of particle count, which absolute discharge is not.

    Do NOT use a percentile of q here.  "Above the 90th percentile" marks
    10 % of cells by construction, so every run scores an identical density
    (measured: exactly 2.00 km/km2 for configs whose networks plainly
    differ) and the metric cannot see the thing it is named after.
    """
    from scipy import ndimage
    qm = float(q[active].mean())
    chan = (q > support * qm) & active if qm > 0 else np.zeros_like(active)
    area_km2 = active.sum() * dx * dx / 1e6
    length_km = chan.sum() * dx / 1000.0
    dens = length_km / max(area_km2, 1e-9)
    dist = ndimage.distance_transform_edt(~chan) * dx
    return dens, float(np.median(dist[active]))


def _load(path):
    """(height+sediment, discharge, active) from a run's final.npz."""
    d = np.load(path)
    h = np.squeeze(d['height'] + d['sediment'])
    q = np.squeeze(d['discharge'])
    m = np.squeeze(d['mask'])
    return h, q, m == 1


if __name__ == '__main__':
    dx = float(os.environ.get('CELL_M', '50'))
    print('%-24s %6s %8s %10s %13s %12s'
          % ('config', 'beta', 'relief', 'p99 slope', 'drain.dens', 'hillslope'))
    for f in sorted(sys.argv[1:]):
        try:
            h, q, act = _load(f)
        except Exception as exc:            # noqa: BLE001 - report and continue
            print('%-24s  !! %s' % (f.split('/')[-2], exc))
            continue
        if not act.any():
            continue
        # crop to the active window, one cell in: the frozen rim and the
        # mask-0 halo carry step changes that are not terrain and would
        # dominate a p99 slope
        rr, cc = np.where(act)
        r0, r1 = rr.min() + 1, rr.max()
        c0, c1 = cc.min() + 1, cc.max()
        if r1 - r0 < 8 or c1 - c0 < 8:
            continue
        h, q, act = h[r0:r1, c0:c1], q[r0:r1, c0:c1], act[r0:r1, c0:c1]
        b, _ = beta(h, dx)
        # slope in cell units: height_unit == cell_size, so this is a true tangent
        gy, gx = np.gradient(h)
        sl = np.percentile(np.hypot(gx, gy)[act], 99)
        dens, hill = dissection(h, q, act, dx)
        flag = '' if sl <= 1.2 else '  <- past talus, does not count'
        print('%-24s %6.2f %8.1f %10.2f %8.2f km/km2 %8.0f m%s'
              % (f.split('/')[-2], b, np.ptp(h), sl, dens, hill, flag))
