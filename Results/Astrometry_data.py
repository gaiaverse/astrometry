import numpy as np, h5py, sys, time
sys.path.append('../SelectionFunctionToolkit/')
from SelectionFunctionGaiaArchive import GaiaArchiveQuery

if False:
    tstart=time.time()
    mag_res=0.1; col_res=1.; pos_level=7
    version="gaiaedr3";
    query = GaiaArchiveQuery(magnitude_column = 'phot_g_mean_mag',
                             magnitude_resolution = mag_res,
                             colour_column = 'bp_rp',
                             colour_resolution = col_res,
                             position_column='source_id',
                             position_resolution=pos_level,
                             gaia_version=version,
                             superset_criteria = ['phot_g_mean_mag is not NULL'],
                             subset_criteria=['astrometric_params_solved>3'])

    print(query.query)
    query.run_query()
    print(f'Query time: {time.time()-tstart}')

    with h5py.File(f'/data/asfe2/Projects/astrometry/gaiaedr3_astrometry_kncounts.h', 'w') as hf:
        for key in ['magnitude', 'colour', 'position', 'k', 'n']:
            hf.create_dataset(key, data=np.array(query.results[key]))

        hf.create_dataset('magnitude_res', data=np.array([mag_res]))
        hf.create_dataset('colour_res', data=np.array([col_res]))
        hf.create_dataset('position_level', data=np.array([pos_level]))
        hf.attrs["query"] = query.query

if True:
    import subprocess, sqlutilpy
    sys.path.append('/home/asfe2/Documents/software/')
    import getdata

    tstart=time.time()
    mag_res=.1; pos_level=7
    query=f"""select (phot_g_mean_mag/{mag_res})::integer as magnitude,
                   (source_id/{2**35 * 4**(12-pos_level)})::integer as position,
                   count(*) as n,
                   count(*) filter (where astrometric_params_solved>3) as k
                   from gaia_edr3.gaia_source
                   where phot_g_mean_mag is not NULL
                   group by magnitude, position
                   """

    table = f"gaiaedr3_astrometry_kncounts_Gres{1/mag_res:.0f}hpx{pos_level}"
    base = f"""DROP TABLE IF EXISTS andy_everall.{table}; set statement_timeout to 86400000; show statement_timeout; create table {table} as """

    print('Running Query:')
    subprocess.call(f'echo "{base + query}" | psql',shell=True)
    print(f'Time {time.time()-tstart} s')

    print('Downloading data:')
    querygetdata = f"""select * from andy_everall.{table}"""
    result = sqlutilpy.get(querygetdata, asDict=True, **getdata.sql_args)
    print(f'Time {time.time()-tstart} s')

    filename=f'/data/asfe2/Projects/astrometry/{table}.h'
    print('saving: ', filename)
    with h5py.File(filename, 'w') as hf:
        for key in result.keys():
            hf.attrs["query"] = base+query
            hf.create_dataset(key, data=result[key])
