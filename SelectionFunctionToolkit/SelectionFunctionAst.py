import sys, h5py, numpy as np, scipy.stats, healpy as hp, tqdm

eps=1e-10


# lmax:100, nside:32, M:0.5 --> ~2.5Gb, 54s
# lmax:100, nside:64, M:0.5 - ~12Gb, 800s

# lmax:100, nside:16, M:17 - ~?Gb, ?s
# lmax:100, nside:16, M:85 - ~3.9Gb, 6167s - 1h55m
# lmax:100, nside:32, M:85 - ~9.4Gb, 13933s - 4h
# lmax:100, nside:64, M:85 - ~30.3Gb, 34212s - 10.5h
# lmax:100, nside:128, M:85 - ~?Gb, ?s

lmax = 100

M = 85
C = 1
nside=64

box={};
with h5py.File('/data/asfe2/Projects/astrometry/gaia3_astcounts_arr_hpx128.h', 'r') as hf:
    box['n'] = hf['n'][...]
    box['k'] = hf['k'][...]
    M_bins = hf['magbins'][...]
print("Mag bins:", np.linspace(M_bins[0], M_bins[-1], M+1))

lengthscale = 0.3
lengthscale_m = lengthscale/(M_bins[1]-M_bins[0])
lengthscale_c = 1.

file_root = f"lmax{lmax}_nside{nside}_M{M}_C{C}_l{lengthscale}"
print(file_root)

# Import hammer
from SelectionFunctionHammer import Hammer
hammer = Hammer(k = box['k'],
                n = box['n'],
                axes = ['magnitude','colour','position'],
                nest = True,
                lmax = lmax,
                lengthscale_m = lengthscale_m,
                lengthscale_c = lengthscale_c,
                M = M,
                C = C,
                nside = nside,
                sparse = True,
                pivot = True,
                mu = 0.0,
                sigma = [-0.81489922, -2.55429039],
                Mlim = [M_bins[0], M_bins[-1]],
                Clim = [-100,100],
                file_root = file_root,
                spherical_harmonics_directory='/data/asfe2/Projects/astrometry/SphericalHarmonics/',
                stan_output_directory='/data/asfe2/Projects/astrometry/StanOutput/'
                )

# Run hammer
hammer.optimize(number_of_iterations = 10000)

# Print convergence information
hammer.print_convergence(number_of_lines = 10)
