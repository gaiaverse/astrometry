import sys
import h5py
import numpy as np

jmax = int(sys.argv[1])
p = float(sys.argv[2])
B = float(sys.argv[3])
nside = int(sys.argv[4])
M = int(sys.argv[5])
C = int(sys.argv[6])
lengthscale = float(sys.argv[7]) # in units of magnitudes

# Load in data and format
with h5py.File('./rvs_grid.h5', 'r') as g:
    
    # Load data
    box = {k:v[:] for k,v in g.items()}
    print(box.keys())
    M_original, C_original = box['k'].shape[:2]
    
    # Calculate lengthscales in units of bins
    lengthscale_m = lengthscale/((box['phot_rp_mean_mag'][1]-box['phot_rp_mean_mag'][0])*(M_original/M))
    lengthscale_c = lengthscale/((box['bp_rp'][1]-box['bp_rp'][0])*(C_original/C))
    print('Lengthscales',lengthscale_m,lengthscale_c)
    
# Import chisel
from SelectionFunctionChisel import Chisel
chisel = Chisel(k = box['k'],
                n = box['n'],
                axes = ['magnitude','colour','position'],
                nest = True,
                basis_options = {'j':[-1]+[i for i in range(jmax)],'p':p,'B':B,'wavelet_tol':1e-4,'needlet':'chisquare'},
                lengthscale_m = lengthscale_m,
                lengthscale_c = lengthscale_c,
                M = M,
                C = C,
                nside = nside,
                sparse = True,
                pivot = True,
                mu = 0.0,
                sigma = [1.32824171, -2.97102361],
                file_root = f"jmax{jmax}_B{B}_nside{nside}_M{M}_C{C}_l{lengthscale}",
                )
del box

# Run hammer
hammer.optimize(number_of_iterations = 10000)

# Print convergence information
hammer.print_convergence(number_of_lines = 10)
