import numpy as np, h5py, sys, time
sys.path.append('../SelectionFunctionToolkit/')
from SelectionFunctionGaiaArchive import GaiaArchiveQuery

tstart=time.time()
mag_res=0.1; col_res=0.1; pos_level=7
version="gaiaedr3";
query = GaiaArchiveQuery(magnitude_column = 'phot_g_mean_mag',
                         magnitude_resolution = mag_res,
                         colour_column = 'bp_rp',
                         colour_resolution = col_res,
                         position_column='source_id',
                         position_resolution=pos_level,
                         gaia_version=version,
                         superset_criteria = ['phot_g_mean_mag is not NULL'],
                         subset_criteria=['ruwe < 1.4', 'astrometric_params_solved>3'])

print(query.query)
query.run_query()
print(f'Query time: {time.time()-tstart}')

with h5py.File(f'/data/asfe2/Projects/astrometry/gaiaedr3_ruwe1p4_kncounts.h', 'w') as hf:
    for key in ['magnitude', 'colour', 'position', 'k', 'n']:
        hf.create_dataset(key, data=np.array(query.results[key]))

    hf.create_dataset('magnitude_res', data=np.array([mag_res]))
    hf.create_dataset('colour_res', data=np.array([col_res]))
    hf.create_dataset('position_level', data=np.array([pos_level]))
    hf.attrs["query"] = query.query
