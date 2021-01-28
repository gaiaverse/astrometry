import sys, os, pickle, numpy as np, h5py, time, pandas as pd

home = os.path.expanduser("~")
sys.path.append(home+'/Documents/software')
import getdata, sqlutilpy, subprocess

#no_warnings()

tstart = time.time(); tprev=time.time()

if False:
    #%% Download data
    print('Download data from wsdb')
    coords = ['ra','dec','parallax','pmra','pmdec']
    errors = [coord+'_error' for coord in coords]
    corrs = [coords[i]+'_'+coords[j]+'_corr'  for j in range(5) for i in range(j)]

    query = """select source_id, {0}, {1}, astrometric_params_solved, phot_g_mean_mag, l, b
                from andy_everall.gaia2_rand100m
    			""".format(','.join(errors), ','.join(corrs))
    _data = sqlutilpy.get(query, asDict=True, **getdata.sql_args)

    cov5 = np.zeros((len(_data['source_id']), 5,5))
    for i in range(5):
        for j in range(5):
            if i==j: cov5[:,i,j] = _data[coords[i]+'_error']**2
            else: cov5[:,i,j] = _data[coords[min(i,j)]+'_'+coords[max(i,j)]+'_corr']*\
                               _data[coords[i]+'_error']*_data[coords[j]+'_error']
    print('Time: %d, Change: %d\n' % (time.time()-tstart, time.time()-tprev)); tprev = time.time()


    #%% Precision matrix
    print('Invert covariance to get 5D Precision matrix')
    prec5 = np.linalg.inv(cov5)

    _output = {}
    for key in ['source_id', 'astrometric_params_solved']:
        _output[key]=_data[key]

    _output['P0_5'] = prec5[:,0,0]+prec5[:,1,1]
    print('Time: %d, Change: %d\n' % (time.time()-tstart, time.time()-tprev)); tprev = time.time()


    #%% Include astrometry prior
    print('Add astrometry prior emulating 2D source astrometry.')
    s0 = 2.187 - 0.2547*_data['phot_g_mean_mag'] + 0.006382*_data['phot_g_mean_mag']**2
    s1 = 0.114 - 0.0579*_data['phot_g_mean_mag'] + 0.01369*_data['phot_g_mean_mag']**2 - 0.000506*_data['phot_g_mean_mag']**3
    s2 = 0.031 - 0.0062*_data['phot_g_mean_mag']
    sigma_pi_f90 = 10**(s0 + s1*np.abs(np.sin(np.deg2rad(_data['b']))) \
                           + s2*       np.cos(np.deg2rad(_data['b']))*np.cos(np.deg2rad(_data['l'])))
    prior_cov = np.zeros((len(_data['source_id']), 5))
    prior_cov[:,[0,1]] = 1000**2
    prior_cov[:,[2,3]] = sigma_pi_f90[:,None]**2
    prior_cov[:,4] = (10*sigma_pi_f90)**2
    prior_prec = 1/prior_cov

    prec5_prior = prec5.copy()
    prec5_prior[:,np.arange(5),np.arange(5)] += prior_prec
    print('Time: %d, Change: %d\n' % (time.time()-tstart, time.time()-tprev)); tprev = time.time()


    #%% Covariance
    print('Emulate 2D covariance.')
    cov5_prior = np.linalg.inv(prec5_prior)[:,:2,:2]

    for i in range(2):
        for j in range(i+1):
            _output['cov_%d%d'%(i,j)] = np.where(_data['astrometric_params_solved']==31,
                                  cov5_prior[:,i,j],
                                  cov5[:,i,j])

    _output['ra_error'] = np.sqrt(_output['cov_00'])
    _output['dec_error'] = np.sqrt(_output['cov_11'])
    _output['ra_dec_corr'] = _output['cov_10']/(_output['ra_error']*_output['dec_error'])

    _output['P0_2'] = (1/_output['ra_error']**2 + 1/_output['dec_error']**2 ) / (1-_output['ra_dec_corr']**2)

    print('Time: %d, Change: %d\n' % (time.time()-tstart, time.time()-tprev)); tprev = time.time()


if True:
    #%% Save
    print('Save results')
    labels = ['source_id', 'ra_error', 'dec_error', 'ra_dec_corr', 'P0_2', 'astrometric_params_solved']
    table = 'p02d_estimated'

    filename = '/data/asfe2/Projects/gaia_psf/%s.h' % table
    if False:
        print('h5py file: %s' % filename)
        with h5py.File(filename, 'w') as hf:
            for key in labels:
                hf.create_dataset(key, data=_output[key])
        print('Time: %d, Change: %d\n' % (time.time()-tstart, time.time()-tprev)); tprev = time.time()

    if False:
        print('Load from hf file: %s' % filename)
        _output = {}
        with h5py.File(filename, 'r') as hf:
            for key in labels:
                _output[key] = hf[key][...]
        print('Time: %d, Change: %d\n' % (time.time()-tstart, time.time()-tprev)); tprev = time.time()

    if False:
        print('online table: %s' % table)
        subprocess.call('echo "DROP TABLE IF EXISTS andy_everall.%s" | psql' % table,shell=True)

        data = [_output[key] for key in labels]
        sqlutilpy.upload('andy_everall.%s' % table, data, labels, **getdata.sql_args)
        print('Time: %d, Change: %d\n' % (time.time()-tstart, time.time()-tprev)); tprev = time.time()

    if False:
        # DR3 total upload time: 19h
        # Clustering/Analysing failed, "idx_source_id" already exists.
        coords = ['ra','dec','parallax','pmra','pmdec', 'pseudocolour']
        columns = {key:'double precision' for key in labels}; columns['source_id']='bigint'; columns['astrometric_params_solved']='int';
        table_headers = ','.join([col+' '+columns[col] for col in columns.keys()])

        upload_table='andy_everall.'+table

        subprocess.call(f'echo "DROP TABLE IF EXISTS {upload_table}" | psql',shell=True)
        subprocess.call(f'echo "CREATE TABLE {upload_table} ( {table_headers} );" | psql',shell=True)

        niter = 10; iter_index = np.linspace(0,len(_output['source_id']), niter+1).astype(int)
        print(iter_index)
        for iter in range(niter):

            temp_file="/data/asfe2/Projects/gaia_edr3/{table}_temp.csv"
            pd.DataFrame(_output)[list(columns.keys())][iter_index[iter]:iter_index[iter+1]].to_csv(temp_file, index=False)
            subprocess.call(f'cat {temp_file} | psql -c "copy {upload_table} from stdin with csv header;"',shell=True)

    if True:
        # reindex="CREATE INDEX idx_sid ON {0} (source_id);".format(table)
        # cluster="CLUSTER {0} USING idx_sid;".format(table)
        reindex="CREATE INDEX sid_idx ON {0} (source_id);".format(table)
        cluster="CLUSTER {0} USING sid_idx;".format(table)
        analyze="ANALYZE {0};".format(table)
        print('Indexing'); subprocess.call('psql -c "{0}"'.format(reindex),shell=True);
        print('Time: %d, Change: %d' % (time.time()-tstart, time.time()-tprev)); tprev = time.time()
        print('Clustering'); subprocess.call('psql -c "{0}"'.format(cluster),shell=True);
        print('Time: %d, Change: %d' % (time.time()-tstart, time.time()-tprev)); tprev = time.time()
        print('Analysing'); subprocess.call('psql -c "{0}"'.format(analyze),shell=True);
        print('Time: %d, Change: %d\n' % (time.time()-tstart, time.time()-tprev)); tprev = time.time()
