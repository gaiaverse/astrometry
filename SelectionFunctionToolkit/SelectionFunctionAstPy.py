import sys, h5py, numpy as np, scipy.stats, healpy as hp, tqdm

eps=1e-10

M = 17
C = 1
nside=4

box={};
with h5py.File('/data/asfe2/Projects/astrometry/gaia3_astcounts_arr_hpx128.h', 'r') as hf:
    box['n'] = hf['n'][...]
    box['k'] = hf['k'][...]
    M_bins = hf['magbins'][...]
print("Mag bins:", np.linspace(M_bins[0], M_bins[-1], M+1))

lengthscale = 0.3
lengthscale_m = 3.#lengthscale/(M_bins[1]-M_bins[0])
lengthscale_c = 1.

jmax=2; B=2.
file_root = f"chisquare_jmax{jmax}_nside{nside}_M{M}_C{C}_l{lengthscale}_B{B}"
print(file_root)
basis_options = {'needlet':'chisquare', 'j':jmax, 'B':B, 'p':1.0, 'wavelet_tol':1e-2}

# Import chisel
from SelectionFunctionPython import pyChisel
pychisel = pyChisel(box['k'], box['n'],
                basis_options,file_root,
                axes = ['magnitude','colour','position'],
                nest = True,
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
                spherical_basis_directory='/data/asfe2/Projects/astrometry/SphericalWavelets/',
                stan_output_directory='/data/asfe2/Projects/astrometry/StanOutput/'
                )

if True:

    z0 = np.random.rand(pychisel.S, pychisel.M, pychisel.C)-0.5
    z0 = np.zeros((pychisel.S, pychisel.M, pychisel.C))

    res = pychisel.minimize_ray(z0, ncores=3, method='Newton-CG', \
                                  options={'disp':True, 'maxiter':50, 'xtol':1e-5})

    print(res)


if False:
    z0 = np.zeros((pychisel.S, pychisel.M, pychisel.C))
    bounds=np.zeros((len(z0.flatten()), 2))
    bounds[:,0]=-5
    bounds[:,1]=5

    pychisel.minimize(z0, method='BFGS', options={'disp':True, 'iprint':10, 'maxfun':1000})
