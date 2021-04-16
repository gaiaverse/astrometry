import healpy as hp, h5py, numpy as np, tqdm

colour = False

eps = 1e-5

M = 17; C=1; level=9; nside=pow(2,9)
res=(M, C, hp.nside2npix(nside))
_downgrade = lambda A: A.reshape(M, A.shape[0]//M,
                                 C, A.shape[1]//C,
                                 hp.nside2npix(nside), A.shape[2]//hp.nside2npix(nside)).sum(axis=(1,3,5))
logit = lambda p: np.log(p/(1-p))

Mlims = [5,22]
Clims = [-10,20]

mag_res = 1.0;
col_res = 1.0;

data_M=int((Mlims[1]-Mlims[0])/mag_res + eps);
data_C=int((Clims[1]-Clims[0])/col_res + eps);
data_nside = pow(2,10)
data_res=(M, C, hp.nside2npix(data_nside))
print('data_res: ', data_res)

sample="astrometry"; file="Gres1hpx10"
box={};
with h5py.File(f'/data/asfe2/Projects/astrometry/gaiaedr3_{sample}_kncounts_{file}.h', 'r') as hf:
    box['n'] = np.zeros(data_res, dtype=np.int64)
    box['k'] = np.zeros(data_res, dtype=np.int64)

    Midx = (hf['magnitude'][...]*mag_res - Mlims[0] + eps).astype(int)
    if colour: Cidx = (hf['colour'][...]*col_res - Clims[0] + eps).astype(int)
    else: Cidx = np.zeros(len(Midx), dtype=np.int64)
    in_range = (Midx>-1)&(Midx<data_M)&(Cidx>-1)&(Cidx<data_C)

    Pidx = hf['position'][...]

    for key in ['n','k']:
        box[key][Midx[in_range], Cidx[in_range], Pidx[in_range]] = hf[key][...][in_range]
        box[key] = _downgrade(box[key])
print(box['n'].shape)

lmax=600;
lvals = np.arange(lmax+1)

print('Evaluate spectra')
spectra = np.zeros((M+1, lmax+1))
x = logit((np.sum(box['k'], axis=(0,1))+1)/(np.sum(box['n'], axis=(0,1))+2))
spectra[M] = hp.anafast(x, lmax=lmax)

for i in tqdm.tqdm(range(M), total=M):
    n_sample = box['n'][i,0]
    k_sample = box['k'][i,0]
    x = logit((k_sample+1)/(n_sample+2))
    spectra[i] = hp.anafast(x, lmax=lmax)

with h5py.File(f'/data/asfe2/Projects/astrometry/gaiaedr3_{sample}_kncounts_{file}_spectra.h', 'w') as hf:
    hf.create_dataset('spectra', data=spectra)
    hf.create_dataset('magbins', data=np.linspace(Mlims[0], Mlims[1],M+1))
    if colour: hf.create_dataset('colbins', data=np.linspace(Clims[0], Clims[1],C+1))
    hf.create_dataset('res', data=np.array([M,C,hp.nside2npix(nside)]))
