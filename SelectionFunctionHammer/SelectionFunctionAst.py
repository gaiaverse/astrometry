import sys, h5py, numpy as np, scipy.stats, healpy as hp, tqdm

eps=1e-10

lmax = 20
lengthscale = 0.3

M_bins = np.arange(10,21.1,1)
M = M_bins.shape[0]-1
C = 1
nside=32

nside_original = 128
resize = int((nside_original/nside)**2+0.1)

# Load in n,k data in magnitude-hpx bins
box={}; nside_original = 128
box['n']=np.zeros((M, C, hp.nside2npix(nside)), dtype=np.int)
box['k']=np.zeros((M, C, hp.nside2npix(nside)), dtype=np.int)
with h5py.File('/data/asfe2/Projects/astrometry/gaia3_astcounts_hpx10.h', 'r') as hf:
    for i, mag in tqdm.tqdm(enumerate(hf['magval']), total=len(hf['magval'])):
        if (mag+eps<M_bins[0])|(mag+eps>M_bins[-1]): continue
        hpx = hf[str(i)]['hpx'][...]//resize
        M_idx = np.sum(M_bins<mag+eps).astype(int) - 1
        box['n'][M_idx,0] = scipy.stats.binned_statistic(hpx, hf[str(i)]['n'][...],
                                                     bins=np.arange(hp.nside2npix(nside)+1)-0.5,
                                                     statistic='sum').statistic.astype(int)
        box['k'][M_idx,0] = scipy.stats.binned_statistic(hpx, hf[str(i)]['k_ast'][...],
                                                     bins=np.arange(hp.nside2npix(nside)+1)-0.5,
                                                     statistic='sum').statistic.astype(int)
lengthscale_m = lengthscale/(M_bins[1]-M_bins[0])
lengthscale_c = 1.

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
                file_root = f"lmax{lmax}_nside{nside}_M{M}_C{C}_l{lengthscale}",
                )

# Run hammer
hammer.optimize(number_of_iterations = 10000)

# Print convergence information
hammer.print_convergence(number_of_lines = 10)
