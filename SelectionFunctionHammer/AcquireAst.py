import sys, os, numpy as np, healpy as hp
import h5py, sqlutilpy, tqdm
from numba import njit

home = os.path.expanduser("~")
sys.path.append(home+'/Documents/software/')
import getdata

if False:
    nside = 128
    N_pixels = hp.nside2npix(nside)
    N_level = hp.nside2order(nside)
    healpix_factor = 2**(59-2*N_level)
    G_width=0.1
    keys = ['source_id','phot_rp_mean_mag','bp_rp','dr2_radial_velocity']

    eps = 1e-10
    bins = {'ast':np.array([-0.5,0.5,1.5]),'phot_g_mean_mag':np.arange(1.7,23.+eps,0.1),'healpix':np.arange(-0.5,N_pixels)}
    bins_sizes = [bins[k].size-1 for k in ['ast','phot_g_mean_mag','healpix']]
    counts = np.zeros(bins_sizes,dtype=np.uint64)

    # Run query to collect aggregate data - Takes ~3min on 100m sample, ~45min on EDR3 sample
    query = """select floor(source_id/{0}) as hpx,
                      floor(phot_g_mean_mag/{1})*{1} as magbin,
                      count(*) as n,
                      count(*) filter (where parallax is not Null) as k_ast
                    from andy_everall.gaia3_rand100m
                    group by hpx, magbin""".format(healpix_factor, G_width)
    count_box = sqlutilpy.get(query, asDict=True, **getdata.sql_args)

    for i, mag in enumerate(bins['phot_g_mean_mag'][:-1]):
        counts[0,i][count_box['hpx'].astype(int)[count_box['magbin']==mag]] = count_box['n'][count_box['magbin']==mag]
        counts[1,i][count_box['k_ast'].astype(int)[count_box['magbin']==mag]] = count_box['k_ast'][count_box['magbin']==mag]

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

    with h5py.File('./ast_grid.h5', 'w') as g:

        for k,v in bins.items():
            g.create_dataset(k, data=v, dtype=np.float64)

        g.create_dataset('k', data=counts[1].astype(int), compression="gzip", compression_opts=9, chunks = True, dtype = np.uint32, fletcher32 = False, shuffle = True, scaleoffset=0)
        g.create_dataset('n', data=counts[0].astype(int), compression="gzip", compression_opts=9, chunks = True, dtype = np.uint32, fletcher32 = False, shuffle = True, scaleoffset=0)

if True:

    eps=1e-10

    M_bins = np.round(np.arange(5.,22.+eps,.1), decimals=1)
    M = M_bins.shape[0]-1
    C = 1
    nside=128

    nside_original = 2**10
    resize = int((nside_original/nside)**2+0.1)

    # Load in n,k data in magnitude-hpx bins
    box={};
    box['n']=np.zeros((M, C, hp.nside2npix(nside)), dtype=np.uint32)
    box['k']=np.zeros((M, C, hp.nside2npix(nside)), dtype=np.uint32)
    n_arr = np.zeros(hp.nside2npix(2**10), dtype=np.uint32)
    k_arr = np.zeros(hp.nside2npix(2**10), dtype=np.uint32)
    with h5py.File('/data/asfe2/Projects/astrometry/gaia3_astcounts_hpx10.h', 'r') as hf:
        for j in tqdm.tqdm(range(M)):

            i = np.argwhere(hf['magval'][...]==M_bins[j])[0,0]

            n_arr*=0; k_arr*=0
            hpx = hf[str(i)]['hpx'][...]

            n_arr[hpx] = hf[str(i)]['n'][...]
            k_arr[hpx] = hf[str(i)]['k_ast'][...]

            box['n'][j,0] = np.sum(np.reshape(n_arr, (-1,resize)), axis=1)
            box['k'][j,0] = np.sum(np.reshape(k_arr, (-1,resize)), axis=1)

    with h5py.File('/data/asfe2/Projects/astrometry/gaia3_astcounts_arr_hpx%d.h' % nside, 'w') as hf:
        hf.create_dataset('n', data=box['n'], compression="gzip", compression_opts=9, chunks = True, dtype = np.uint32, fletcher32 = False, shuffle = True, scaleoffset=0)
        hf.create_dataset('k', data=box['k'], compression="gzip", compression_opts=9, chunks = True, dtype = np.uint32, fletcher32 = False, shuffle = True, scaleoffset=0)
        hf.create_dataset('magbins', data=M_bins)
