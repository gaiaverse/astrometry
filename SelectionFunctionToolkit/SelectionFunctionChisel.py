import numpy as np
import healpy as hp
import tqdm
import h5py
import os


class Chisel:
    
    
    
    def __init__(self, k, n, jmax, B = 2.0, wavelet_tol = 1e-4, file_root = 'chisel', axes  = ['magnitude','colour','position'],lengthscale_m = 1.0, lengthscale_c = 1.0, M = None, C = None, nside = None, sparse = False, sparse_tol = 1e-4, pivot = False, pivot_tol = 1e-4, nest = True, mu = None, sigma = None, spherical_wavelets_directory='./SphericalWavelets',stan_model_directory='./StanModels',stan_output_directory='./StanOutput'):

        # Utilities
        self.order_to_nside = lambda order: 2**order
        self.nside_to_npix = lambda nside: 12*nside**2
        self.order_to_npix = lambda order: self.nside_to_npix(self.order_to_nside(order))

        self.spherical_wavelets_directory = self._verify_directory(spherical_wavelets_directory)
        self.stan_model_directory = self._verify_directory(stan_model_directory)
        self.stan_output_directory = self._verify_directory(stan_output_directory)

        self.jmax = jmax
        assert B > 1.0
        self.B = B
        assert wavelet_tol > 0.0
        self.wavelet_tol = wavelet_tol
        self.sparse = sparse
        self.sparse_tol = sparse_tol
        self.pivot = pivot
        self.pivot_tol = pivot_tol
        self.nest = nest

        # Reshape k and n to be valid
        self._reshape_k_and_n(k,n,axes)

        # Downgrade the resolution
        self._downgrade_resolution(M,C,nside)

        # These must both be in units of bins!
        self.lengthscale_m = lengthscale_m
        self.lengthscale_c = lengthscale_c
        self.S = 4**(self.jmax + 2) - 3

        # Load spherical wavelets
        self._load_spherical_wavelets()

        # Compute cholesky matrices
        self.M_subspace, self.cholesky_m = self._construct_cholesky_matrix(self.M,self.lengthscale_m)
        self.C_subspace, self.cholesky_c = self._construct_cholesky_matrix(self.C,self.lengthscale_c)

        # Process mu and sigma
        self._process_mu_and_sigma(mu,sigma)

        # Load Stan Model
        self._load_stan_model()

        # Construct Stan Input
        self._construct_stan_input()

        # File root
        self.file_root = file_root

    def optimize(self, number_of_iterations = 1000):

        import time
        print('Running optimisation')
        t1 = time.time()
        _stan_optimum = self.stan_model.optimize(data = self.stan_input, iter = number_of_iterations, output_dir = self.stan_output_directory)
        t2 = time.time()
        print(f'Finished optimisation, it took {t2-t1:.1f} seconds')

        # Extract maxima
        _size_z = self.S*self.M_subspace*self.C_subspace
        _size_x = self.M*self.C*self.P
        _ring_indices = hp.nest2ring(self.nside, np.arange(self.P))
        self.optimum_lnp = _stan_optimum.optimized_params_np[0]
        self.optimum_z = np.transpose(_stan_optimum.optimized_params_np[1:1+_size_z].reshape((self.C_subspace,self.M_subspace,self.S)))
        if self.nest:
            self.optimum_x = self._ring_to_nest(np.transpose(_stan_optimum.optimized_params_np[1+_size_z:].reshape((self.P,self.C,self.M))))
        else:
            self.optimum_x = np.transpose(_stan_optimum.optimized_params_np[1+_size_z:].reshape((self.P,self.C,self.M)))
        self.optimum_b = self.stan_input['mu'][:,None,None] + self.stan_input['sigma'][:,None,None] * (self.cholesky_m @ self.optimum_z @ self.cholesky_c.T)

        # Move convergence information somewhere useful
        import shutil
        self.optimum_convergence_file = self.file_root+'_convergence.txt'
        shutil.move(self.stan_output_directory + str(_stan_optimum).split('/')[-1], self.stan_output_directory + self.optimum_convergence_file)
        print(f'Convergence information stored in {self.stan_output_directory + self.optimum_convergence_file}')

        # Save optimum to h5py
        self.optimum_results_file = self.file_root+'_results.h5'
        with h5py.File(self.stan_output_directory + self.optimum_results_file, 'w') as orf:
            orf.create_dataset('opt_runtime', data = t2-t1)
            orf.create_dataset('lnP', data = self.optimum_lnp)
            orf.create_dataset('z', data = self.optimum_z, dtype = np.float64, compression = 'lzf', chunks = True)
            orf.create_dataset('b', data = self.optimum_b, dtype = np.float64, compression = 'lzf', chunks = True)
            orf.create_dataset('x', data = self.optimum_x, dtype = np.float64, compression = 'lzf', chunks = True)
        print(f'Optimum values stored in {self.stan_output_directory + self.optimum_results_file}')

    def print_convergence(self, number_of_lines = 2):
        for line in self._tail(self.stan_output_directory + self.optimum_convergence_file,number_of_lines):
            print(line)

    def _verify_directory(self,_directory):

        # Check it exists, if not then create
        if not os.path.exists(_directory):
            os.makedirs(_directory)

        # Ensure it ends with '/'
        if _directory[-1] != '/':
            _directory = _directory + '/'

        return _directory

    def _reshape_k_and_n(self,k,n,axes):

        possible_axes = ["magnitude","colour","position"]
        axes_size = len(axes)

        assert 'position' in axes # We do not yet accept magnitude-colour only selection functions
        assert set(axes).issubset(set(possible_axes)) # We can only accept magnitude, colour and position
        assert axes_size == len(set(axes)) # Each axis must be unique
        assert k.shape == n.shape # k and n must have same shape

        new_indices = [possible_axes.index(axis) for axis in axes]
        self.k_original = np.moveaxis(k.copy().reshape(k.shape+(1,)*(3-axes_size)),range(axes_size),new_indices)
        self.n_original = np.moveaxis(n.copy().reshape(n.shape+(1,)*(3-axes_size)),range(axes_size),new_indices)

        self.M_original, self.C_original, self.P_original = self.k_original.shape
        assert hp.isnpixok(self.P_original) # number of pixels must be valid

    def _downgrade_resolution(self,M,C,nside):

        self.M = self.M_original if M == None else M
        self.C = self.C_original if C == None else C
        if nside == None:
            self.nside = hp.npix2nside(self.P_original)
            self.P = self.P_original
        else:
            self.nside = nside
            assert hp.isnsideok(self.nside)
            self.P = hp.nside2npix(self.nside)

        _downgrade = lambda A: A.reshape(self.M, self.M_original//self.M, self.C, self.C_original//self.C, self.P, self.P_original//self.P).sum(axis=(1,3,5))

        if self.nest:
            self.k = self._nest_to_ring(_downgrade(self.k_original))
            self.n = self._nest_to_ring(_downgrade(self.n_original))
        else:
            self.k = self._nest_to_ring(_downgrade(self._ring_to_nest(self.k_original)))
            self.n = self._nest_to_ring(_downgrade(self._ring_to_nest(self.n_original)))

        del self.k_original, self.n_original

    def _load_spherical_wavelets(self):
        """ Loads in the spherical wavelets file corresponding to nside, jmax and B. If they don't exist, then generate them. """

        self.spherical_wavelets_file = f'sphericalwavelets_nside{self.nside}_jmax{self.jmax}_B{self.B}_tol{self.wavelet_tol}.h5'
        if not os.path.isfile(self.spherical_wavelets_directory + self.spherical_wavelets_file):
            print('Spherical wavelets file does not exist, generating... (this may take some time!)')
            self._generate_spherical_wavelets(self.spherical_wavelets_directory + self.spherical_wavelets_file)

        # Load spherical wavelets
        with h5py.File(self.spherical_wavelets_directory + self.spherical_wavelets_file, 'r') as swf:
            self._wavelet_w = swf['wavelet_w'][:]
            self._wavelet_v = swf['wavelet_v'][:]
            self._wavelet_u = swf['wavelet_u'][:]
            self._wavelet_j = swf['wavelet_j'][:]
            self._wavelet_n = self._wavelet_w.size

        print('Spherical wavelets file loaded')

    def _construct_cholesky_matrix(self,N,lengthscale):

        # Create Cholesky matrices
        dx = np.arange(N)
        _covariance = np.exp(-np.square(dx[:,None]-dx[None,:])/(2.0*lengthscale*lengthscale))

        if self.pivot:
            _cholesky = self._pivoted_cholesky(_covariance, M=N, err_tol=self.pivot_tol)
        else:
            _cholesky = np.linalg.cholesky(_covariance+1e-15*np.diag(np.ones(N)))

        _N_subspace = _cholesky.shape[1]
        print(N,_N_subspace)

        return _N_subspace, _cholesky

    def _process_mu_and_sigma(self,mu,sigma):

        # Process mu
        if mu == None:
            self.mu = np.zeros(self.S)
        elif isinstance(mu, np.ndarray):
            assert mu.shape == (self.S,)
            self.mu = mu
        elif callable(mu):
            self.mu = mu(self._wavelets_j)
        else:
            self.mu = mu*np.ones(self.S)

        # Process sigma
        if sigma == None:
            self.sigma = np.ones(self.S)
        elif isinstance(sigma, np.ndarray):
            assert sigma.shape == (self.S,)
            self.sigma = sigma
        elif callable(sigma):
            self.sigma = sigma(self._wavelets_j)
        elif type(sigma) in [list,tuple]:
            assert len(sigma) == 2
            power_spectrum = lambda l: np.sqrt(np.exp(sigma[0])*np.power(1.0+l,sigma[1]))
            
            from SelectionFunctionUtils import littlewoodpaley
            lwp = littlewoodpaley()
            _sigma = np.zeros(self.jmax+1)
            for j in range(self.jmax+1):
                
                nside_needle = self.order_to_nside(j)
                npix_needle = self.nside_to_npix(nside_needle)

                start = int(np.floor(self.B**(j-1)))
                end = int(np.ceil(self.B**(j+1)))
                modes = np.arange(start, end + 1, dtype = 'float')
                window = lwp.window_function(modes / (self.B**j), self.B)**2*power_spectrum(modes)*(2.0*modes+1.0)/npix_needle
                
                _sigma[j] = np.sqrt(window.sum())
                
            self.sigma = np.array([_sigma[j] for j in self._wavelet_j])
        else:
            self.sigma = sigma*np.ones(self.S)



    def _load_stan_model(self):

        _model_file = 'wavelet_magnitude_colour_position'
        _model_file += '_sparse' if self.sparse else ''

        from cmdstanpy import CmdStanModel
        self.stan_model = CmdStanModel(stan_file = self.stan_model_directory+_model_file+'.stan')

    def _construct_stan_input(self):

        self.stan_input = {'k':self.k,
                           'n':self.n,
                           'P':self.P,
                           'M':self.M,
                           'M_subspace':self.M_subspace,
                           'C':self.C,
                           'C_subspace':self.C_subspace,
                           'S':self.S,
                           'wavelet_n':self._wavelet_n,
                           'wavelet_w':self._wavelet_w,
                           'wavelet_v':self._wavelet_v+1,
                           'wavelet_u':self._wavelet_u+1,
                           'mu':self.mu,
                           'sigma':self.sigma}

        if self.sparse:
            self.stan_input['cholesky_n_m'], self.stan_input['cholesky_w_m'], self.stan_input['cholesky_v_m'], self.stan_input['cholesky_u_m'] = self._sparsify(self.cholesky_m)
            self.stan_input['cholesky_n_c'], self.stan_input['cholesky_w_c'], self.stan_input['cholesky_v_c'], self.stan_input['cholesky_u_c'] = self._sparsify(self.cholesky_c)
        else:
            self.stan_input['cholesky_m'] = self.cholesky_m
            self.stan_input['cholesky_c'] = self.cholesky_c

    def _sparsify(self,_matrix):

        # Set any elements in each row that are smaller than self.sparse_tol * max(row) to zero
        _height,_width = _matrix.shape
        _sparse_matrix = _matrix.copy()

        for n in range(_height):
            _row = np.abs(_matrix[n])
            _sparse_matrix[n,_row/max(_row) < self.sparse_tol] = 0

        # Compute the CSR decomposition of the sparse matrix
        from scipy.sparse import csr_matrix
        _csr_matrix = csr_matrix(_sparse_matrix)
        _csr_n = _csr_matrix.data.size
        print(f"{100*(1.0-_csr_n/(_height*_width)):.2f}% sparsity")

        return _csr_n, _csr_matrix.data, _csr_matrix.indices + 1, _csr_matrix.indptr + 1

    def _pivoted_cholesky(self, A, M, err_tol = 1e-6):
        """
        https://dl.acm.org/doi/10.1016/j.apnum.2011.10.001 implemented by https://github.com/NathanWycoff/PivotedCholesky
        A simple python function which computes the Pivoted Cholesky decomposition/approximation of positive semi-definite operator. Only diagonal elements and select rows of that operator's matrix represenation are required.
        get_diag - A function which takes no arguments and returns the diagonal of the matrix when called.
        get_row - A function which takes 1 integer argument and returns the desired row (zero indexed).
        M - The maximum rank of the approximate decomposition; an integer.
        err_tol - The maximum error tolerance, that is difference between the approximate decomposition and true matrix, allowed. Note that this is in the Trace norm, not the spectral or frobenius norm.
        Returns: R, an upper triangular matrix of column dimension equal to the target matrix. It's row dimension will be at most M, but may be less if the termination condition was acceptably low error rather than max iters reached.
        """

        get_diag = lambda: np.diag(A).copy()
        get_row = lambda i: A[i,:]

        d = np.copy(get_diag())
        N = len(d)

        pi = list(range(N))

        R = np.zeros([M,N])

        err = np.sum(np.abs(d))

        m = 0
        while (m < M) and (err > err_tol):

            i = m + np.argmax([d[pi[j]] for j in range(m,N)])

            tmp = pi[m]
            pi[m] = pi[i]
            pi[i] = tmp

            R[m,pi[m]] = np.sqrt(d[pi[m]])
            Apim = get_row(pi[m])
            for i in range(m+1, N):
                if m > 0:
                    ip = np.inner(R[:m,pi[m]], R[:m,pi[i]])
                else:
                    ip = 0
                R[m,pi[i]] = (Apim[pi[i]] - ip) / R[m,pi[m]]
                d[pi[i]] -= pow(R[m,pi[i]],2)

            err = np.sum([d[pi[i]] for i in range(m+1,N)])
            m += 1

        R = R[:m,:]

        return(R.T)

    def _tail(self,filename, lines=1, _buffer=4098):
        """Tail a file and get X lines from the end"""
        # place holder for the lines found
        lines_found = []

        # block counter will be multiplied by buffer
        # to get the block size from the end
        block_counter = -1

        with open(filename,'r') as f:
            # loop until we find X lines
            while len(lines_found) < lines:
                try:
                    f.seek(block_counter * _buffer, os.SEEK_END)
                except IOError:  # either file is too small, or too many lines requested
                    f.seek(0)
                    lines_found = f.readlines()
                    break

                lines_found = f.readlines()

                block_counter -= 1

        return lines_found[-lines:]

    def _nest_to_ring(self,A):
        """ Reorders an array of shape (M,C,P) to ring ordering. """
        _npix = A.shape[2]
        _nside = hp.npix2nside(_npix)
        _reordering = hp.ring2nest(_nside, np.arange(_npix))
        return A[:,:,_reordering]

    def _ring_to_nest(self,A):
        """ Reorders an array of shape (M,C,P) to ring ordering. """
        _npix = A.shape[2]
        _nside = hp.npix2nside(_npix)
        _reordering = hp.nest2ring(_nside, np.arange(_npix))
        return A[:,:,_reordering]

    def _generate_spherical_wavelets(self,gsw_file):
        
        # Import dependencies
        from numba import njit
        from math import sin, cos
        import sys

        nside = self.nside
        jmax = self.jmax
        B = self.B
        needle_sparse_tol = self.wavelet_tol

        # Function to compute needlet across sky
        @njit
        def pixel_space (Y, cos_gamma, window, start, end, legendre):
            '''Return the value of a needlet at gamma radians from the needlet centre.'''

            legendre[0] = 1.0
            legendre[1] = cos_gamma
            for cur_l in range(2, end + 1):
                legendre[cur_l] = ((cos_gamma * (2 * cur_l - 1) * legendre[cur_l - 1] - (cur_l - 1) * legendre[cur_l - 2])) / cur_l

            Y[:] = np.dot(window,legendre[start:end+1])

        # Compute locations of pixels
        npix = self.nside_to_npix(nside)
        colat, lon = np.array(hp.pix2ang(nside=nside,ipix=np.arange(npix),lonlat=False))
        cos_colat, sin_colat = np.cos(colat), np.sin(colat)
        cos_lon, sin_lon = np.cos(lon), np.sin(lon)

        # Instantiate class to compute window function
        from SelectionFunctionUtils import littlewoodpaley
        lwp = littlewoodpaley()

        # Initialise variables
        Y = np.zeros(npix)
        needlet_w = [np.ones(npix)]
        needlet_v = [np.arange(npix)]
        needlet_u = [0]
        legendre = np.zeros((1+int(np.ceil(B**(jmax+1))),npix))
        running_index = npix

        for j in range(jmax+1):

            print(f'Working on order {j} out of {jmax}.')

            nside_needle = self.order_to_nside(j)
            npix_needle = self.nside_to_npix(nside_needle)

            start = int(np.floor(B**(j-1)))
            end = int(np.ceil(B**(j+1)))
            modes = np.arange(start, end + 1, dtype = 'float')
            window = lwp.window_function(modes / (B**j), B)*(2.0*modes+1.0)/np.sqrt(4.0*np.pi*npix_needle)

            for ipix_needle in tqdm.tqdm(range(npix_needle),file=sys.stdout):

                colat_needle, lon_needle = hp.pix2ang(nside=nside_needle,ipix=ipix_needle,lonlat=False)

                cos_gamma = cos(colat_needle) * cos_colat + sin(colat_needle) * sin_colat * (cos(lon_needle) * cos_lon + sin(lon_needle) * sin_lon)

                pixel_space (Y, cos_gamma = cos_gamma, window = window, start = start, end = end, legendre = legendre)

                _significant = np.where(np.abs(Y) > Y.max()*needle_sparse_tol)[0]
                needlet_w.append(Y[_significant])
                needlet_v.append(_significant)
                needlet_u.append(running_index)
                running_index += _significant.size
        
        # Add the ending index to u
        needlet_u.append(running_index)

        # Concatenate the lists
        needlet_w = np.concatenate(needlet_w)
        needlet_v = np.concatenate(needlet_v)
        needlet_u = np.array(needlet_u)
        
        # Flip them round
        from scipy import sparse
        Y = sparse.csr_matrix((needlet_w,needlet_v,needlet_u)).transpose().tocsr()
        wavelet_w, wavelet_v, wavelet_u = Y.data, Y.indices, Y.indptr
        wavelet_j = np.concatenate([np.zeros(1)]+[j*np.ones(self.order_to_npix(j)) for j in range(jmax+1)]).astype(int)

        # Save file
        save_kwargs = {'compression':"lzf", 'chunks':True, 'fletcher32':False, 'shuffle':True}
        with h5py.File(gsw_file, 'w') as f:
            f.create_dataset('wavelet_w', data = wavelet_w, dtype = np.float64, **save_kwargs)
            f.create_dataset('wavelet_v', data = wavelet_v, dtype = np.uint64, scaleoffset=0, **save_kwargs)
            f.create_dataset('wavelet_u', data = wavelet_u, dtype = np.uint64, scaleoffset=0, **save_kwargs)
            f.create_dataset('wavelet_j', data = wavelet_j, dtype = np.uint64, scaleoffset=0, **save_kwargs)
