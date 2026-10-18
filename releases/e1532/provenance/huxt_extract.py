"""Extract HUXt Earth trajectories (and the 1 AU longitude ring) from the E1466 causal TD-HUXt field cache (read-only)."""
import glob, time
from pathlib import Path
import h5py, numpy as np
D = Path(glob.glob('/home/t-lab01/sunrun/research/expos/runs/E1466/tdwsahuxt-state/td-state-*')[0])
OUT = Path(__file__).resolve().parent / 'cache/huxt_earth.npz'
f = h5py.File(D / 'lead_fields.h5', 'r')
x, eli = f['fields'], f['earth_lon_index'][:]
n = x.shape[0]
ring = np.empty((n, 72, 128), np.float16)
t0 = time.time()
for i in range(0, n, 500):
    ring[i:i + 500] = x[i:i + 500, :, -1, :]
    if i % 5000 == 0:
        print(i, round(time.time() - t0), flush=True)
traj = np.take_along_axis(ring, eli[:, :, None].astype(np.int64), 2)[..., 0].astype(np.float32)
np.savez(OUT, origins_ns=f['origins_ns'][:], done=f['done'][:], earth_lon_index=eli, traj=traj, ring=ring)
print('saved', OUT, traj.shape, np.nanmean(traj), round(time.time() - t0), flush=True)
