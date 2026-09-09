"""Slope-area analysis: where do channels actually begin?

    python3 -c "import slope_area, numpy as np; ..."   (see docs/terrain-realism.md)

Use this rather than a discharge threshold.  `discharge` is an EMA of
particle volume, not an upstream-cell count, so thresholding it does not
give a contributing area and its "drainage density" is not comparable to
published values -- that mistake produced a bogus "5x under-dissected"
reading before this existed.  Here the area comes from real D8 flow
accumulation over a priority-flood-filled surface.

The standard, threshold-free way to locate channel heads in a DEM.  On
hillslopes, gradient rises with contributing area (more area -> steeper
convex-up nose).  Once flow concentrates into a channel it reverses and
follows S ~ A^-theta with theta about 0.4-0.5.  The turnover is the
channel head, and its area is the drainage density expressed as a length
scale -- no arbitrary threshold anywhere.

Real landscapes put the turnover at 1e3-1e4 m2 (30-100 m of hillslope).
"""
import numpy as np
from scipy import ndimage

D8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

def fill_sinks(z, land, eps=1e-6):
    """Priority-flood: raise pits just enough to drain."""
    import heapq
    n0, n1 = z.shape
    out = np.where(land, np.inf, z).astype(np.float64)
    seen = np.zeros(z.shape, bool)
    h = []
    # seed from the domain edge and from any non-land neighbour
    for i in range(n0):
        for j in (0, n1-1):
            if land[i,j]: heapq.heappush(h,(z[i,j],i,j)); out[i,j]=z[i,j]; seen[i,j]=True
    for j in range(n1):
        for i in (0, n0-1):
            if land[i,j] and not seen[i,j]: heapq.heappush(h,(z[i,j],i,j)); out[i,j]=z[i,j]; seen[i,j]=True
    ii,jj = np.where(land)
    for i,j in zip(ii,jj):
        if seen[i,j]: continue
        for di,dj in D8:
            a,b = i+di, j+dj
            if 0<=a<n0 and 0<=b<n1 and not land[a,b]:
                heapq.heappush(h,(z[i,j],i,j)); out[i,j]=z[i,j]; seen[i,j]=True; break
    while h:
        zv,i,j = heapq.heappop(h)
        for di,dj in D8:
            a,b = i+di, j+dj
            if not (0<=a<n0 and 0<=b<n1) or seen[a,b] or not land[a,b]: continue
            nz = max(z[a,b], zv+eps)
            out[a,b] = nz; seen[a,b] = True
            heapq.heappush(h,(nz,a,b))
    return out

def d8_accum(zf, land, cell):
    """D8 steepest descent + flow accumulation (contributing area, m2)."""
    n0,n1 = zf.shape
    order = np.argsort(np.where(land, zf, -np.inf).ravel())[::-1]
    down = -np.ones(zf.size, np.int64)
    slope = np.zeros(zf.shape)
    for idx in order:
        i,j = divmod(idx, n1)
        if not land[i,j]: continue
        best, bs = -1, 0.0
        for di,dj in D8:
            a,b = i+di, j+dj
            if not (0<=a<n0 and 0<=b<n1): continue
            d = cell*np.hypot(di,dj)
            s = (zf[i,j]-zf[a,b])/d
            if s > bs: bs, best = s, a*n1+b
        down[idx] = best; slope[i,j] = bs
    acc = np.where(land, cell*cell, 0.0).ravel()
    for idx in order:
        if not land.ravel()[idx]: continue
        d = down[idx]
        if d >= 0: acc[d] += acc[idx]
    return acc.reshape(zf.shape), slope

def curve(z, land, cell, label):
    zf = fill_sinks(z, land)
    A, S = d8_accum(zf, land, cell)
    m = land & (S > 0) & (A > 0)
    la = np.log10(A[m]); ls = np.log10(S[m])
    bins = np.arange(la.min(), la.max()+0.25, 0.25)
    idx = np.digitize(la, bins)
    print('\n%s' % label)
    print('  %12s %10s %8s' % ('area (m2)', 'med slope', 'n'))
    prev = None; peak = None
    for b in range(1, len(bins)):
        k = idx == b
        if k.sum() < 25: continue
        a_ = 10**((bins[b-1]+bins[b])/2); s_ = 10**np.median(ls[k])
        mark = ''
        if prev is not None and peak is None and s_ < prev:
            peak = a_; mark = '   <-- turnover (channel head)'
        prev = s_ if peak is None else prev
        print('  %12.0f %10.4f %8d%s' % (a_, s_, k.sum(), mark))
    return peak
