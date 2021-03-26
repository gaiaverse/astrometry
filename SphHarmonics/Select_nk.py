import sys, os
home = os.path.expanduser("~")
sys.path.append(home+'/Documents/software/basecode/')
from importscript import *

import tqdm
no_warnings()

if True:
    level=10

    data={}
    mags = np.arange(8.5,21.,1.)
    for mag in mags:
        with h5py.File('/data/asfe2/Projects/astrometry/gaia3_astcounts_hpx%d.h' % level, 'r') as hf:

            magid = np.argwhere(hf['magval'][...]==mag)[0,0]
            data['%.1f' % mag] = {}
            data['%.1f' % mag]['n']    = hf[str(magid)]['n'][...]
            data['%.1f' % mag]['k_ast']= hf[str(magid)]['k_ast'][...]
            data['%.1f' % mag]['k_rv']= hf[str(magid)]['k_rv'][...]
            data['%.1f' % mag]['hpx']  = hf[str(magid)]['hpx'][...]

    data['full']={}
    for key in ['n','k_ast','k_rv']: data['full'][key]=np.zeros(hp.nside2npix(2**level)).astype(int)
    with h5py.File('/data/asfe2/Projects/astrometry/gaia3_astcounts_hpx%d.h' % level, 'r') as hf:
        for magid in tqdm.tqdm(hf.keys()):
            if magid != 'magval':
                for key in ['n','k_ast','k_rv']:
                    data['full'][key][hf[magid]['hpx'][...]] += hf[str(magid)][key][...]

    with h5py.File('/data/asfe2/Projects/astrometry/gaia3_astcounts_select_hpx{0}.h'.format(level), 'w') as hf:
        hf.create_dataset('magval', data=mags)
        for magid in tqdm.tqdm(data.keys()):
            for key in data[magid].keys():
                hf.create_dataset(os.path.join(magid, key), data=data[magid][key])

if False:

    level=int(sys.argv[1]); nside=pow(2, level); divisor=pow(2,35) * pow(4,12-level)

    full_level=10; full_nside=pow(2, full_level)

    data={}; data['n']={}; data['k_ast']={}; data['k_rvs']={}
    mags = np.arange(8.5,21.,1.)
    with h5py.File('/data/asfe2/Projects/astrometry/gaia3_astcounts_hpx%d.h' % full_level, 'r') as hf:
        for mag in mags:
            magid = np.array([np.argwhere(hf['magval'][...]==mag)[0,0] for mag in mags])
            data[str(magid)]['n'] = np.zeros(hp.nside2npix(full_nside)).astype(int)
            data[str(magid)]['n'][hf[str(magid)]['hpx'][...]] = hf[str(magid)]['n'][...]
            data[str(magid)]['n'] = downsize(data['n'], full_nside, nside)[n2r]
            data[str(magid)]['k_ast'] = np.zeros(hp.nside2npix(full_nside)).astype(int)
            data[str(magid)]['k_ast'][hf[str(magid)]['hpx'][...]] = hf[str(magid)]['k_ast'][...]
            data[str(magid)]['k_ast'] = downsize(data['k'], full_nside, nside)[n2r]
            data[str(magid)]['k_rvs'] = np.zeros(hp.nside2npix(full_nside)).astype(int)
            data[str(magid)]['k_rvs'][hf[str(magid)]['hpx'][...]] = hf[str(magid)]['k_rvs'][...]
            data[str(magid)]['k_rvs'] = downsize(data['k'], full_nside, nside)[n2r]
