import sys, os, pickle, time, warnings
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import h5py, numpy as np, scipy.stats, healpy as hp, tqdm

eps=1e-10

M = 16; Mlims = [2,18.];
C = 4; Clims = [-1,3];
nside=16; jmax=3; B=2.

colour=True
mag_res = 0.1;
M_bins = np.arange(Mlims[0], Mlims[1], mag_res)
col_res = 0.1;
C_bins = np.arange(Clims[0], Clims[1], col_res)

data_M=int((Mlims[1]-Mlims[0])/mag_res + eps);
data_C=int((Clims[1]-Clims[0])/col_res + eps);
data_nside = pow(2,7)
data_res=(data_M, data_C, hp.nside2npix(data_nside))
print('data_res: ', data_res)
sample="rvs"; file="Gres10hpx7"
box={};
with h5py.File(f'/data/asfe2/Projects/astrometry/gaiaedr3_{sample}_kncounts_{file}.h', 'r') as hf:
    box['n'] = np.zeros(data_res, dtype=np.int64)
    box['k'] = np.zeros(data_res, dtype=np.int64)

    Midx = hf['magnitude'][...] - int(Mlims[0]/mag_res + eps)
    try: Cidx = hf['colour'][...] - int(Clims[0]/mag_res + eps)
    except KeyError: Cidx = np.zeros(len(Midx), dtype=np.int64)
    Pidx = hf['position'][...]
    in_range = (Midx>-1)&(Midx<data_M)&(Cidx>-1)&(Cidx<data_C)
    for key in ['n','k']:
        box[key][Midx[in_range], Cidx[in_range], Pidx[in_range]] = hf[key][...][in_range]
print(box['n'].shape)


lengthscale = 0.3
# Calculate lengthscales in units of bins
M_original, C_original = box['k'].shape[:2]
lengthscale_m = lengthscale/((M_bins[1]-M_bins[0])*(M_original/M))
lengthscale_c = lengthscale/((C_bins[1]-C_bins[0])*(C_original/C))
print(f"lengthscales m:{lengthscale_m} , c:{lengthscale_c}")

file_root = f"chisquare_{sample}_jmax{jmax}_nside{nside}_M{M}_C{C}_l{lengthscale}_B{B}_ncores{ncores}mp_lbfgsb"
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
                sigma = [-0.81489922, -2.516],
                Mlim = [M_bins[0], M_bins[-1]],
                Clim = [C_bins[0], C_bins[-1]],
                spherical_basis_directory='/data/asfe2/Projects/astrometry/SphericalWavelets/',
                stan_output_directory='/data/asfe2/Projects/astrometry/PyOutput/'
                )
if True:
    z0 = np.random.normal(0, 1, size=(pychisel.S, pychisel.M_subspace, pychisel.C_subspace)).flatten()
    last_iteration=0
    force=False
else:
    print('Hot Start!')
    with h5py.File(f'/data/asfe2/Projects/astrometry/PyOutput/{file_root}_progress.h', 'r') as hf:
        keys = list(hf.keys())
        z0 = hf[keys[np.argmax(np.array(keys).astype(int))]][...]
        last_iteration = np.max(np.array(keys).astype(int))
    force=True

f_tol = 1e-10
print(f'f_tol = {f_tol:.0e}')
bounds=np.zeros((len(z0.flatten()), 2)); bounds[:,0]=-50; bounds[:,1]=50
res = pychisel.minimize_mp(z0, ncores=ncores, bounds=bounds, method='L-BFGS-B', force=force, nfev_init=last_iteration,
                               options={'disp':False, 'maxiter':20000, 'ftol':f_tol, 'gtol':1})
print(res)
