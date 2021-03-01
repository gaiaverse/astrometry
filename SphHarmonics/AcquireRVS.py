import numpy as np
import h5py
import tqdm
import healpy as hp
from numba import njit

nside = 128
N_pixels = hp.nside2npix(nside)
N_level = hp.nside2order(nside)
healpix_factor = 2**(59-2*N_level)
N_sources = 1811709771
N_chunk = 3*17*6791
N_block = 5231
keys = ['source_id','phot_rp_mean_mag','bp_rp','dr2_radial_velocity']

eps = 1e-10
bins = {'rvs':np.array([-0.5,0.5,1.5]),'healpix':np.arange(-0.5,N_pixels),'phot_rp_mean_mag':np.arange(6.0,16.0+eps,0.1),'bp_rp':np.arange(0.0,3.0+eps,0.1)}
bins_sizes = [bins[k].size-1 for k in ['rvs','healpix','phot_rp_mean_mag','bp_rp']]
counts = np.zeros(bins_sizes,dtype=np.uint64)

@njit
def incrementer(array,i,j,k,l):
    for _i,_j,_k,_l in zip(i,j,k,l):
        array[_i,_j,_k,_l] += 1

with h5py.File('./gaiaedr3.h5', 'r') as f:
    for idx_block in tqdm.tqdm(range(N_block)):

        # Load in data
        box = {k:np.zeros(N_chunk) for k in keys}
        for _k in keys:
            f[_k].read_direct(box[_k],np.s_[idx_block*N_chunk:(idx_block+1)*N_chunk])
            
        # Find sources with radial velocity
        in_interval = np.where((box['phot_rp_mean_mag']>bins['phot_rp_mean_mag'][0])&(box['phot_rp_mean_mag']<bins['phot_rp_mean_mag'][-1])&(box['bp_rp']>bins['bp_rp'][0])&(box['bp_rp']<bins['bp_rp'][-1]))
        
        healpix_idx = np.floor(box['source_id'][in_interval]/healpix_factor).astype(np.int)
        
        phot_rp_mean_mag_idx = np.digitize(box['phot_rp_mean_mag'][in_interval],bins['phot_rp_mean_mag'])-1
        
        bp_rp_idx = np.digitize(box['bp_rp'][in_interval],bins['bp_rp'])-1
        
        rvs_idx = box['dr2_radial_velocity'][in_interval]
        rvs_idx[np.isnan(rvs_idx) == False] = 1
        rvs_idx[np.isnan(rvs_idx) == True] = 0
        rvs_idx = rvs_idx.astype(np.int)
        
        incrementer(counts,rvs_idx,healpix_idx,phot_rp_mean_mag_idx,bp_rp_idx)
    
# Find the most in any bins
max_counts = np.max(counts)
if max_counts < 256:
    counts_dtype = np.uint8
elif max_counts < 65536:
    counts_dtype = np.uint16
elif max_counts < 4294967296:
    counts_dtype = np.uint32
elif max_counts < 18446744073709551616:
    counts_dtype = np.uint64

with h5py.File('./rvs_grid.h5', 'w') as g:
    for k,v in bins.items():
        g.create_dataset(k, data=v, dtype=np.float64)
    g.create_dataset('counts', data=counts, compression="gzip", compression_opts=9, chunks = True, dtype = counts_dtype, fletcher32 = False, shuffle = True, scaleoffset=0)

                

        