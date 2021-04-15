import sys, os, pickle, time, warnings
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import h5py, numpy as np, scipy.stats, healpy as hp, tqdm


# M = 17; C = 1; nside=4; jmax=2; B=2.; ncores=5:   250s
# M = 17; C = 1; nside=8; jmax=3; B=2.; ncores=20:   500s
# M = 17; C = 1; nside=8; jmax=3; B=2.; ncores=20 BFGS: 1701s
# M = 17; C = 1; nside=32; jmax=3; B=2.; ncores=30: 1800s
# M = 17; C = 1; nside=64; jmax=4; B=2.; ncores=30: 8500s
# M = 85; C = 1; nside=64; jmax=4; B=2.; ncores=30: 62h   32Gb

eps=1e-10
M = 85; C = 1; nside=64; jmax=5; B=3.
#M = 17; C = 1; nside=8; jmax=2; B=3.
ncores=50

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


file_root = f"chisquare_jmax{jmax}_nside{nside}_M{M}_C{C}_l{lengthscale}_B{B}_ncores{ncores}mp_lbfgsb"
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
                stan_output_directory='/data/asfe2/Projects/astrometry/PyOutput/'
                )

if True:

    #z0 = np.random.rand(pychisel.S, pychisel.M, pychisel.C)-0.5

    z0 = np.random.normal(0, 1, size=(pychisel.S, pychisel.M, pychisel.C)).flatten()

    if True:
        print('Hot Start!')
        with h5py.File(f'/data/asfe2/Projects/astrometry/PyOutput/{file_root}_progress.h', 'r') as hf:
            keys = list(hf.keys())
            z0 = hf[keys[np.argmax(np.array(keys).astype(int))]][...]
            last_iteration = np.max(np.array(keys).astype(int))
    # z0 = np.zeros((pychisel.S, pychisel.M, pychisel.C))
    # with h5py.File('/data/asfe2/Projects/astrometry/StanOutput/chisquare_jmax3_nside32_M17_C1_l0.3_B2.0_scipy_results.h5', 'r') as hf:
    #     print('Hot Start!')
    #     z0 = hf['z'][...]

    # x_tol = 1e-5
    # print(f'x_tol = {x_tol:.0e}')
    # res = pychisel.minimize_mp(z0, ncores=ncores, method='Newton-CG', force=False, nfev_init=0, options={'disp':True, 'maxiter':50, 'xtol':x_tol})

    f_tol = 1e-10
    print(f'f_tol = {f_tol:.0e}')
    bounds=np.zeros((len(z0.flatten()), 2)); bounds[:,0]=-50; bounds[:,1]=50
    res = pychisel.minimize_mp(z0, ncores=ncores, bounds=bounds, method='L-BFGS-B', force=True, nfev_init=last_iteration,
                                    options={'disp':False, 'maxiter':20000, 'ftol':f_tol, 'gtol':1})

    #res = pychisel.minimize_ray(z0, ncores=ncores, method='Newton-CG', options={'disp':True, 'maxiter':50, 'xtol':1e-5})
    #res = pychisel.minimize(z0, method='Newton-CG', options={'disp':True, 'maxiter':50, 'xtol':1e-5})

    # print(f'f_tol = {1e-5:.0e}')
    # bounds=np.zeros((len(z0.flatten()), 2)); bounds[:,0]=-50; bounds[:,1]=50
    # res = pychisel.minimize_ray(z0, bounds=bounds, ncores=ncores, method='L-BFGS-B', options={'disp':True, 'maxiter':100, 'ftol':1e-5})
    # res = pychisel.minimize_ray(z0, ncores=10, method='BFGS', options={'disp':True, 'maxiter':50, 'gnorm':1e-5})
    # res = pychisel.minimize_mp(z0, ncores=2, method='Newton-CG', options={'disp':True, 'maxiter':50, 'xtol':1e-5})

    print(res)


if False:
    z0 = np.zeros((pychisel.S, pychisel.M, pychisel.C))
    bounds=np.zeros((len(z0.flatten()), 2))
    bounds[:,0]=-5
    bounds[:,1]=5

    pychisel.minimize(z0, method='BFGS', options={'disp':True, 'iprint':10, 'maxfun':1000})
