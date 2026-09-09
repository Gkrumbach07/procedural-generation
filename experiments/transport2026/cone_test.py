"""The paper's validation case (Sec 7.1): water transported across a
convergent cone of radius R, uniform source S, no decay, velocity along the
height gradient.  Analytic solution phi = S/(2 r |v|) (R^2 - r^2)."""
import numpy as np, transport

def run(K, n_per_cell):
    dx = 1.0
    c = (K - 1) / 2.0
    i, j = np.meshgrid(np.arange(K), np.arange(K), indexing='ij')
    r = np.hypot(i - c, j - c)
    Rad = c
    # cone descending to the centre; velocity is the unit inward radial
    vx = np.where(r > 1e-9, -(i - c) / np.maximum(r, 1e-9), 0.0)
    vy = np.where(r > 1e-9, -(j - c) / np.maximum(r, 1e-9), 0.0)
    S = np.ones((K, K)); Rdec = np.zeros((K, K))
    mask = (r <= Rad).astype(np.uint8)
    phi = transport.solve(S, Rdec, vx, vy, mask, dx, n_per_cell=n_per_cell, seed=1)
    exact = np.where(r > 1e-9, 1.0 / (2 * np.maximum(r, 1e-9)) * (Rad**2 - r**2), 0.0)
    band = (r > 0.15 * Rad) & (r < 0.85 * Rad)
    msre = float(np.mean(((phi[band] - exact[band]) / exact[band])**2))
    return msre, phi, exact, band

print("paper Fig 2: MSRE falls linearly with samples/cell, and with resolution at fixed ratio")
print("%8s %10s %12s" % ("K", "samples/cell", "MSRE"))
for K in (128, 256):
    for npc in (0.25, 1.0, 4.0):
        m, *_ = run(K, npc)
        print("%8d %10.2f %12.4f" % (K, npc, m))
