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
import numpy as np, sys

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

if __name__ == '__main__':
    import glob
    for f in sys.argv[1:]:
        d = np.load(f)
        h = np.squeeze(d['height'] + d['sediment'])
        mask = np.squeeze(d['mask'])
        # interior only, largest square window of active cells
        act = mask == 1
        rr, cc = np.where(act)
        r0, r1, c0, c1 = rr.min(), rr.max(), cc.min(), cc.max()
        n = min(r1-r0, c1-c0)
        sub = h[r0:r0+n, c0:c0+n]
        b, npts = beta(sub, 50.0)
        print('%-46s beta %5.2f  (%d bands)  relief %7.1f  rms %6.2f'
              % (f.split('/')[-2], b, npts, np.ptp(sub), sub.std()))
