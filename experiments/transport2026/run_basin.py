import sys, time, numpy as np
sys.path.insert(0,'.')
import geoerode
from geoerode import P
steps = int(sys.argv[1]) if len(sys.argv)>1 else 400
z = np.load('basin_z.npy'); m = np.load('basin_m.npy')
dx = 12.5
U = np.where(m==1, P.U, 0.0)
ins = m==1
print('basin %dx%d at %.1f m = %.2f km2 | relief %.1f m'%(z.shape[0],z.shape[1],dx,z.size*dx*dx/1e6,np.ptp(z[ins])))
# start inside the angle of repose: the baked basin has 12.5 m cells with
# local steps well past it, and E_f is quartic in |v|
z = geoerode.repose_relax(z, dx, np.tan(P.theta), ins)
print('after repose pre-pass: relief %.1f m, max slope %.2f' % (np.ptp(z[ins]), np.abs(np.gradient(z, dx)).max()))
np.save('basin_z0.npy', z)
t0=time.time()
z2, info = geoerode.erode(z, m, dx, steps, npc=1.0, seed=0, uplift=U, log=print)
print("RESULT %d steps %.0fs | finite %s nan_free %s | sim %.1f ky | relief %.1f -> %.1f m"
      %(steps,time.time()-t0,bool(np.all(np.isfinite(z2))),info['nan_free'],
        info['t_sim']/(365.25*24*3600)/1000,np.ptp(z[ins]),np.ptp(z2[ins])))
np.save('basin_out.npy', z2)
for k in ('h_f','h_s'): np.save('basin_%s.npy'%k, info[k])
