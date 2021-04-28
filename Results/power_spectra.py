import healpy as hp, h5py, numpy as np, tqdm

eps = 1e-5

level=7; nside=pow(2,level)

# M = 17; Mlims = [5,22]; C=1; Clims = [-50,50]; colour = False
# sample="astrometry"; file="Gres1hpx10"; mag_res = 1.0; col_res = 100.0;
# M = 17; Mlims = [5,22]; C=4; Clims = [-1,7]; colour = True
# sample="ruwe1p4"; file=f"Gres10CRres2hpx7"; mag_res = .1; col_res = .5;
M = 13; Mlims = [5,17.2]; C = 4; Clims = [-1,3]; colour = True
sample="rvs"; file="Gres10CRres2hpx7"; mag_res = .1; col_res = .5;

res=(M, C, hp.nside2npix(nside))
_downgrade = lambda A: A.reshape(M, A.shape[0]//M,
                                 C, A.shape[1]//C,
                                 hp.nside2npix(nside), A.shape[2]//hp.nside2npix(nside)).sum(axis=(1,3,5))
logit = lambda p: np.log(p/(1-p))

data_M=int((Mlims[1]-Mlims[0])/mag_res + eps);
data_C=int((Clims[1]-Clims[0])/col_res + eps);
data_nside = pow(2,7)
data_res=(data_M, data_C, hp.nside2npix(data_nside))
print('data_res: ', data_res)


box={};
with h5py.File(f'/data/asfe2/Projects/astrometry/gaiaedr3_{sample}_kncounts_{file}.h', 'r') as hf:
    box['n'] = np.zeros(data_res, dtype=np.int64)
    box['k'] = np.zeros(data_res, dtype=np.int64)

    Midx = hf['magnitude'][...] - int(Mlims[0]/mag_res + eps)
    try: Cidx = hf['colour'][...] - int(Clims[0]/col_res + eps)
    except KeyError: Cidx = np.zeros(len(Midx), dtype=np.int64)
    Pidx = hf['position'][...]
    in_range = (Midx>-1)&(Midx<data_M)&(Cidx>-1)&(Cidx<data_C)
    for key in ['n','k']:
        box[key][Midx[in_range], Cidx[in_range], Pidx[in_range]] = hf[key][...][in_range]
print(box['n'].shape)
if sample=='rvs':
    box['n'] = np.vstack((box['n'], np.zeros((int((18.-Mlims[1])/mag_res + eps), *box['n'].shape[1:]), dtype=np.int64)))
    box['k'] = np.vstack((box['k'], np.zeros((int((18.-Mlims[1])/mag_res + eps), *box['k'].shape[1:]), dtype=np.int64)))
    print(np.sum(box['n'], axis=(1,2)))
    print(box['n'].shape)
box['n'] = _downgrade(box['n'])
box['k'] = _downgrade(box['k'])
print(box['n'].shape)
print(np.sum(box['n']), np.sum(box['k']))

lmax=400;
lvals = np.arange(lmax+1)

print('Evaluate spectra')
x = logit((np.sum(box['k'], axis=(0,1))+1)/(np.sum(box['n'], axis=(0,1))+2))
spectra_full = hp.anafast(x, lmax=lmax)

spectra = np.zeros((M, C, lmax+1))
for iM in tqdm.tqdm(range(M), total=M):
    for iC in range(C):
        n_sample = box['n'][iM,iC]
        k_sample = box['k'][iM,iC]
        x = logit((k_sample+1)/(n_sample+2))
        spectra[iM,iC] = hp.anafast(x, lmax=lmax)

spectra_file = f'/data/asfe2/Projects/astrometry/gaiaedr3_{sample}_kncounts_M{M}C{C}nside{nside}_spectra.h'
print('savng: ', spectra_file)
with h5py.File(spectra_file, 'w') as hf:
    hf.create_dataset('spectra', data=spectra)
    hf.create_dataset('spectra_full', data=spectra_full)
    hf.create_dataset('magbins', data=np.linspace(Mlims[0], Mlims[1],M+1))
    if colour: hf.create_dataset('colbins', data=np.linspace(Clims[0], Clims[1],C+1))
    hf.create_dataset('res', data=np.array([M,C,hp.nside2npix(nside)]))
