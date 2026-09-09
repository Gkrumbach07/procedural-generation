"""Paper Fig. 3 schedule: flat start, uniform uplift, watch relief develop."""
import sys, time, numpy as np
sys.path.insert(0,'.')
import geoerode
from geoerode import P

K = int(sys.argv[1]) if len(sys.argv)>1 else 192
steps = int(sys.argv[2]) if len(sys.argv)>2 else 400
dx = float(sys.argv[3]) if len(sys.argv)>3 else 5.0
rs = np.random.RandomState(7)
z = rs.rand(K,K)*0.1
m = np.ones((K,K),np.uint8); m[0,:]=m[-1,:]=m[:,0]=m[:,-1]=0
U = np.full_like(z, P.U)
t0=time.time()
z2, info = geoerode.erode(z, m, dx, steps, npc=1.0, seed=0, uplift=U, log=print)
ins = m==1
print("RESULT %d steps in %.0f s | finite %s | nan_free %s (first bad it %d) | sim %.0f ky | relief %.1f m"
      % (steps, time.time()-t0, bool(np.all(np.isfinite(z2))), info['nan_free'], info['first_bad_it'],
         info['t_sim']/(365.25*24*3600)/1000, float(np.ptp(z2[ins]))))
np.save('flat_z_%d.npy'%K, z2)
