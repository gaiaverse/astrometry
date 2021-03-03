import sys, pickle, os
home = os.path.expanduser("~")
sys.path.append(home+'/Documents/software')
from importscript import *

import tqdm
no_warnings()

level=int(sys.argv[1]); nside=pow(2, level); divisor=pow(2,35) * pow(4,12-level)
gwidth=0.1
print(level, nside, divisor)

query = """select source_id/{0} as hpx,
                  floor(phot_g_mean_mag/{1})*{1} as magbin,
                  count(*) as n,
                  count(*) filter (where parallax is not Null) as k_ast,
                  count(*) filter (where dr2_radial_velocity is not Null) as k_rv
                from gaia_edr3.gaia_source
                group by hpx, magbin""".format(divisor,gwidth)
_counts = sqlutilpy.get(query, asDict=True, **getdata.sql_args)

## Save data
magval = np.sort(np.unique(_counts['magbin']))
magval = magval[~np.isnan(magval)]

with h5py.File('/data/asfe2/Projects/astrometry/gaia3_astcounts_hpx{0}.h'.format(level), 'w') as hf:
    hf.create_dataset('magval', data=magval)
    for i, mag in enumerate(magval):
        hf.create_dataset(str(i)+'/n', data=_counts['n'][_counts['magbin']==mag])
        hf.create_dataset(str(i)+'/k_ast', data=_counts['k_ast'][_counts['magbin']==mag])
        hf.create_dataset(str(i)+'/k_rv', data=_counts['k_rv'][_counts['magbin']==mag])
        hf.create_dataset(str(i)+'/hpx', data=_counts['hpx'][_counts['magbin']==mag])
