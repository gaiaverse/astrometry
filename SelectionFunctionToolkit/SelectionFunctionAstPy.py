import sys, h5py, numpy as np, scipy.stats, healpy as hp, tqdm


# M = 17; C = 1; nside=4; jmax=2; B=2.; ncores=5:   250s
# M = 17; C = 1; nside=8; jmax=3; B=2.; ncores=20:   500s
# M = 17; C = 1; nside=8; jmax=3; B=2.; ncores=20 BFGS: 1701s
# M = 17; C = 1; nside=32; jmax=3; B=2.; ncores=30: 1800s
# M = 17; C = 1; nside=64; jmax=4; B=2.; ncores=30: 8500s
# M = 85; C = 1; nside=64; jmax=4; B=2.; ncores=30:         32Gb

eps=1e-10
M = 17; C = 1; nside=8; jmax=2; B=4.

box={};
with h5py.File('/data/asfe2/Projects/astrometry/gaia3_astcounts_arr_hpx128.h', 'r') as hf:
    box['n'] = hf['n'][...]
    box['k'] = hf['k'][...]
    M_bins = hf['magbins'][...]
    C_bins = np.array([-100,100])
print("Mag bins:", np.linspace(M_bins[0], M_bins[-1], M+1))

lengthscale = 0.3
# Calculate lengthscales in units of bins
M_original, C_original = box['k'].shape[:2]
lengthscale_m = lengthscale/((M_bins[1]-M_bins[0])*(M_original/M))
lengthscale_c = lengthscale/((C_bins[1]-C_bins[0])*(C_original/C))
print(f"lengthscales m:{lengthscale_m} , c:{lengthscale_c}")


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
                Clim = [C_bins[0], C_bins[-1]],
                spherical_basis_directory='/data/asfe2/Projects/astrometry/SphericalWavelets/',
                stan_output_directory='/data/asfe2/Projects/astrometry/StanOutput/'
                )

if True:

    z0 = np.random.rand(pychisel.S, pychisel.M, pychisel.C)-0.5
    z0 = np.zeros((pychisel.S, pychisel.M, pychisel.C))

    #res = pychisel.minimize_ray(z0, ncores=2, method='Newton-CG', options={'disp':True, 'maxiter':50, 'xtol':1e-5})
    res = pychisel.minimize_mp(z0, ncores=2, method='Newton-CG', options={'disp':True, 'maxiter':50, 'xtol':1e-5})
    #res = pychisel.minimize(z0, method='Newton-CG', options={'disp':True, 'maxiter':50, 'xtol':1e-5})

    print(res)


if False:
    z0 = np.zeros((pychisel.S, pychisel.M, pychisel.C))
    bounds=np.zeros((len(z0.flatten()), 2))
    bounds[:,0]=-5
    bounds[:,1]=5

    pychisel.minimize(z0, method='BFGS', options={'disp':True, 'iprint':10, 'maxfun':1000})
