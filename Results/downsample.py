import numpy as np, h5py, healpy as hp

file = 'gaiaedr3_ruwe1p4_kncounts_Gres1Cres1hpx9.h'

eps = 1e-5

level=9; nside=pow(2,level)

M = 1; Mlims = [1,22]; C=1; Clims = [-1,5]; colour = False
sample="astrometry"; file="Gres1hpx10";
# M = 1; Mlims = [2,22]; C=1; Clims = [-1,5]; colour = True
# sample="ruwe1p4"; file=f"Gres1Cres1hpx{level}";
# M = 1; Mlims = [1,18]; C = 1; Clims = [-1,9]; colour = True
# sample="rvs"; file="Gres1Cres1hpx9";

res=(M, C, hp.nside2npix(nside))
_downgrade = lambda A: A.reshape(M, A.shape[0]//M,
                                 C, A.shape[1]//C,
                                 hp.nside2npix(nside), A.shape[2]//hp.nside2npix(nside)).sum(axis=(1,3,5))

mag_res = 1.0;
col_res = 1.0;

data_M=int((Mlims[1]-Mlims[0])/mag_res + eps);
data_C=int((Clims[1]-Clims[0])/col_res + eps);
data_nside = pow(2,10)
data_res=(data_M, data_C, hp.nside2npix(data_nside))
print('data_res: ', data_res)

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
print(np.sum(box['k']), np.sum(box['n']))

box['n'] = _downgrade(box['n'])
box['k'] = _downgrade(box['k'])

with h5py.File(f'/data/asfe2/Projects/astrometry/gaiaedr3_{sample}_kncounts_hpx9.h', 'w') as hf:
    hf.create_dataset('n', data=box['n'], dtype=np.int64, chunks=True, compression="lzf")
    hf.create_dataset('k', data=box['k'], dtype=np.int64, chunks=True, compression="lzf")
