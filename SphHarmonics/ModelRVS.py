import numpy as np
import healpy as hp
import tqdm
import h5py
import sys
import os
from cmdstanpy import CmdStanModel

lmax = int(sys.argv[1])
nside = int(sys.argv[2])
P = hp.nside2npix(nside)
M = int(sys.argv[3])
C = int(sys.argv[4])
lengthscale = float(sys.argv[5]) # in units of magnitudes


# If spherical harmonics have not already been generated then generate them
sphox_file = f'./SphericalHarmonics/sphericalharmonics_decomposed_quick_nside{nside}_lmax{lmax}.h5'
if not os.path.isfile(sphox_file):
    import subprocess
    subprocess.Popen(['python3','GenerateSphHarm_decomposed_quick.py',str(nside),str(lmax)]).wait()

# Load spherical harmonics
with h5py.File(sphox_file, 'r') as f:
    sphox = {k:v[:] for k,v in f.items()}

# Function to shrink a 3D numpy array along each dimension
def shrink3d(data, height, width, depth):
    return data.reshape(height, int(data.shape[0]/height), width, int(data.shape[1]/width), depth, int(data.shape[2]/depth)).sum(axis=(1,3,5))

# Load in data and format
with h5py.File('./rvs_grid.h5', 'r') as g:
    
    # Load data
    box = {k:v[:] for k,v in g.items()}
    
    # Shrink along three axes
    P_original,M_original,C_original = box['counts'].shape[1:]
    print(f"Shrinking from {P_original,M_original,C_original} to {P,M,C}")
    L, H, R = 2 * lmax + 1, (lmax + 1) ** 2, 4 * nside - 1

    counts_shrunk = np.zeros((2, P, M, C))
    counts_shrunk[0] = shrink3d(box['counts'][0], P, M, C)
    counts_shrunk[1] = shrink3d(box['counts'][1], P, M, C)
    
    # Generate k and n, sorting by healpix
    nest_indices = hp.pixelfunc.ring2nest(nside, np.arange(P))
    box['k'] = counts_shrunk[1][nest_indices].astype(int)
    box['n'] = (counts_shrunk[0] + counts_shrunk[1])[nest_indices].astype(int)
    del counts_shrunk, box['counts']
    
    # Generate lengthscales
    lengthscale_m = lengthscale/((box['phot_rp_mean_mag'][1]-box['phot_rp_mean_mag'][0])*(M_original/M))
    lengthscale_c = lengthscale/((box['bp_rp'][1]-box['bp_rp'][0])*(C_original/C))
    print('Lengthscales',lengthscale_m,lengthscale_c)
    
    
# Create Cholesky matrices
dx = np.arange(max(M,C))
covariance_m = np.exp(-np.square(dx[:M,None]-dx[None,:M])/(2.0*lengthscale_m**2.0))
cholesky_m = np.linalg.cholesky(covariance_m+1e-15*np.diag(np.ones(M)))
covariance_c = np.exp(-np.square(dx[:C,None]-dx[None,:C])/(2.0*lengthscale_c**2.0))
cholesky_c = np.linalg.cholesky(covariance_c+1e-15*np.diag(np.ones(C)))

# Create input to Stan
mcp_data = {"P":P, "M":M, "C":C, "L":L, "H":H, "R":R,
            "lambda":sphox['lambda'].T, "azimuth":sphox['azimuth'],
            "pixel_to_ring":sphox['jpix']+1, "lower":sphox['lower'].astype(int)+1, "upper":sphox['upper'].astype(int)+1,
            "cholesky_m":cholesky_m, "cholesky_c_T":cholesky_c.T,
            "mu":np.zeros(H), "sigma":1.0/np.power(1.0+sphox['l'],1.5),
            "k": np.moveaxis(box['k'],0,2), "n": np.moveaxis(box['n'],0,2)}
del sphox, box

# Compile model if it doesn't already exist
mcp_model = CmdStanModel(stan_file='./StanModels/magnitude_colour_position.stan')

# Run optimisation
import time
print('Running optimisation')
t1 = time.time()
fit_opt = mcp_model.optimize(data=mcp_data,iter=10000,output_dir='./StanOutput')
t2 = time.time()
print(f'Finished optimisation, it took {t2-t1} seconds')

# Extract maxima
fit_opt_x = fit_opt.optimized_params_np[1+H*M*C:].reshape((P,C,M))

# Save maxima
save_file = f"./ModelOutputs/lmax{lmax}_nside{nside}_M{M}_C{C}_l{lengthscale}.h5"
with h5py.File(save_file, 'w') as f:
    f.create_dataset('x', data = fit_opt_x, shape = (P, C, M), dtype = np.float64, compression = 'lzf', chunks = True)

# Print result
def tail(filename, lines=1, _buffer=4098):
    """Tail a file and get X lines from the end"""
    # place holder for the lines found
    lines_found = []

    # block counter will be multiplied by buffer
    # to get the block size from the end
    block_counter = -1
    
    with open(filename,'r') as f:
        # loop until we find X lines
        while len(lines_found) < lines:
            try:
                f.seek(block_counter * _buffer, os.SEEK_END)
            except IOError:  # either file is too small, or too many lines requested
                f.seek(0)
                lines_found = f.readlines()
                break

            lines_found = f.readlines()

            block_counter -= 1

    return lines_found[-lines:]

for line in tail('./StanOutput/'+str(fit_opt).split('/')[-1],10):
    print(line)
    
