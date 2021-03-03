class GaiaArchiveQuery:
    def __init__(self, magnitude_column = None, magnitude_resolution = 1.0, colour_column = None, colour_resolution = 1.0, position_column = None, position_resolution = 0, gaia_version = 'gaiaedr3', superset_criteria = None, subset_criteria = None, limit_results = None, random_results = False):
                 
        assert magnitude_column != None or colour_column != None or position_column != None
        
        self.request = {}
        if magnitude_column is not None: self._build_request('magnitude',magnitude_column,magnitude_resolution)
        if colour_column is not None: self._build_request('colour',colour_column,colour_resolution)
        if position_column is not None: self._build_request('position',position_column,position_resolution)
        
        self.limit_results = '' if limit_results is None else f'top {int(limit_results)}'
        self.random_results = '' if random_results is False else 'order by random_index'
    
        assert gaia_version in ['gaiadr1','gaiadr2','gaiaedr3']
        self.gaia_version = gaia_version
        self.superset_criteria = superset_criteria
        self.subset_criteria = subset_criteria
        
        self._build_query()
        
    def _build_request(self, key, column, resolution):
        assert key in ['magnitude','colour','position']
        
        if key == 'position':
            assert resolution < 13 and resolution >= 0 and int(resolution) == resolution
            assert column == 'source_id'
        else:
            assert resolution > 0.0
            
        self.request[key] = {'column':column,'resolution':resolution}
        
    def _build_query(self):
        subset_line = 'to_integer(' + '*'.join([f'IF_THEN_ELSE({sc}, 1.0, 0.0)' for sc in self.subset_criteria])+ ') as selection'
        superset_line = 'where ' + ' and '.join(self.superset_criteria)
        select_line = ', '.join([k for k in self.request.keys()])
        subquery_line = ', '.join([f"gaia_healpix_index({v['resolution']}, {v['column']}) as {k}" if k == 'position' else f"to_integer(floor({v['column']} * {int(1.0/v['resolution'])})) as {k}" for k,v in self.request.items()])


        self.query = f"""select {select_line}, count(*) as n, sum(selection) as k from ( select {self.limit_results} {subquery_line}, {subset_line} from {self.gaia_version}.gaia_source {superset_line} {self.random_results} ) as subquery group by {select_line}"""
        
    def run_query(self):
        from astroquery.gaia import Gaia
        import time
        
        print('Running query')
        t1 = time.time()
        job = Gaia.launch_job_async(self.query)
        self.results = job.get_results()
        t2 = time.time()
        print(f'Finished query, it took {t2-t1:.1f} seconds')
        
        
