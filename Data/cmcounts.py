import numpy as np, h5py, sys, time
sys.path.append('../SelectionFunctionToolkit/')
from SelectionFunctionGaiaArchive import GaiaArchiveQuery

import subprocess, sqlutilpy
sys.path.append('/home/asfe2/Documents/software/')
import getdata

tstart=time.time()
mag_res=0.1; col_res=0.1;
sample = 'all'

tstart=time.time()
query=f"""select (phot_g_mean_mag/{mag_res})::integer as magnitude,
                 ((phot_g_mean_mag-phot_rp_mean_mag)/{col_res})::integer as colour,
               count(*) as n_full,
               count(*) filter (where astrometric_params_solved=31) as n_ast5,
               count(*) filter (where astrometric_params_solved=95) as n_ast6,
               count(*) filter (where ruwe<1.4) as n_ruwe,
               count(*) filter (where dr2_rv_nb_transits>3) as n_rvs
               from gaia_edr3.gaia_source
               group by magnitude, colour
               """

table = f"gaiaedr3_{sample}_kncounts_Gres{1/mag_res:.0f}CRres{1/col_res:.0f}"
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
        hf.create_dataset(key, data=result[key], dtype=np.int64, chunks=True, compression="lzf")
