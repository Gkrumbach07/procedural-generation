import numpy as np, json
from pathlib import Path
W = Path('worlds/final512/fine')
best = None
for f in range(6):
    bid = np.load(W/f'basin_id.f{f}.npy'); h = np.load(W/f'height.f{f}.npy')
    ids, cnt = np.unique(bid[bid > 0], return_counts=True)
    if not len(ids): continue
    for i in np.argsort(cnt)[::-1][:3]:
        b, c = int(ids[i]), int(cnt[i])
        m = bid == b
        rr, cc = np.where(m)
        H, Wd = np.ptp(rr)+1, np.ptp(cc)+1
        # want a compact basin that fits a square window
        if max(H, Wd) < 700 and c > 40000 and (best is None or c > best[1]):
            best = (b, c, f, rr.min(), rr.max(), cc.min(), cc.max())
print('picked basin', best)
b, c, f, r0, r1, c0, c1 = best
bid = np.load(W/f'basin_id.f{f}.npy'); h = np.load(W/f'height.f{f}.npy')
pad = 8
r0, r1 = max(0, r0-pad), min(h.shape[0]-1, r1+pad)
c0, c1 = max(0, c0-pad), min(h.shape[1]-1, c1+pad)
z = h[r0:r1+1, c0:c1+1].astype(np.float64)
m = (bid[r0:r1+1, c0:c1+1] == b).astype(np.uint8)
print('window %dx%d, basin cells %d (%.1f%% of window)' % (z.shape[0], z.shape[1], m.sum(), 100*m.mean()))
print('height range %.1f .. %.1f m, relief %.1f m' % (z.min(), z.max(), np.ptp(z[m==1])))
np.save('basin_z.npy', z); np.save('basin_m.npy', m)
