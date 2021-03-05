import numpy as np
import healpy as hp
import tqdm
import h5py
import os


class Hammer:
    def __init__(self, k, n, lmax, file_root = 'hammer', axes  = ['magnitude','colour','position'],lengthscale_m = 1.0, lengthscale_c = 1.0, M = None, C = None, nside = None, sparse = False, sparse_tol = 1e-4, pivot = False, pivot_tol = 1e-4, nest = True, mu = None, sigma = None, spherical_harmonics_directory='./SphericalHarmonics',stan_model_directory='./StanModels',stan_output_directory='./StanOutput'):


        self.spherical_harmonics_directory = self._verify_directory(spherical_harmonics_directory)
        self.stan_model_directory = self._verify_directory(stan_model_directory)
        self.stan_output_directory = self._verify_directory(stan_output_directory)

        self.lmax = lmax
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
        self.L, self.H, self.R = 2 * self.lmax + 1, (self.lmax + 1) ** 2, 4 * self.nside - 1

        # Load spherical harmonics
        self._load_spherical_harmonics()

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
        _size_z = self.H*self.M_subspace*self.C_subspace
        _size_x = self.M*self.C*self.P
        _ring_indices = hp.nest2ring(self.nside, np.arange(self.P))
        self.optimum_lnp = _stan_optimum.optimized_params_np[0]
        self.optimum_z = np.transpose(_stan_optimum.optimized_params_np[1:1+_size_z].reshape((self.C_subspace,self.M_subspace,self.H)))
        if self.nest:
            self.optimum_x = self._ring_to_nest(np.transpose(_stan_optimum.optimized_params_np[1+_size_z:].reshape((self.P,self.C,self.M))))
        else:
            self.optimum_x = np.transpose(_stan_optimum.optimized_params_np[1+_size_z:].reshape((self.P,self.C,self.M)))
        self.optimum_a = self.stan_input['mu'][:,None,None] + self.stan_input['sigma'][:,None,None] * (self.cholesky_m @ self.optimum_z @ self.cholesky_c.T)

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
            orf.create_dataset('a', data = self.optimum_a, dtype = np.float64, compression = 'lzf', chunks = True)
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

    def _load_spherical_harmonics(self):
        """ Loads in the spherical harmonics file corresponding to nside and lmax. If they don't exist, then generate them. """

        self.spherical_harmonics_file = f'sphericalharmonics_nside{self.nside}_lmax{self.lmax}.h5'
        if not os.path.isfile(self.spherical_harmonics_directory + self.spherical_harmonics_file):
            print('Spherical harmonic file does not exist, generating...')
            self._generate_spherical_harmonics(self.spherical_harmonics_directory + self.spherical_harmonics_file)

        # Load spherical harmonics
        with h5py.File(self.spherical_harmonics_directory + self.spherical_harmonics_file, 'r') as shf:
            self._lambda = shf['lambda'][:].T
            self._azimuth = shf['azimuth'][:]
            self._pixel_to_ring = shf['pixel_to_ring'][:].astype(int)
            self._lower = shf['lower'][:].astype(int)
            self._upper = shf['upper'][:].astype(int)
            self._l = shf['l'][:]
            self._m = shf['m'][:]

        print('Spherical harmonic file loaded')

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
            self.mu = np.zeros(self.H)
        elif isinstance(mu, np.ndarray):
            assert mu.shape == (self.H,)
            self.mu = mu
        elif callable(mu):
            self.mu = mu(self._l,self._m)
        else:
            self.mu = mu*np.ones(self.H)

        # Process sigma
        if sigma == None:
            self.sigma = np.ones(self.H)
        elif isinstance(sigma, np.ndarray):
            assert sigma.shape == (self.H,)
            self.sigma = sigma
        elif callable(sigma):
            self.sigma = sigma(self._l,self._m)
        elif type(sigma) in [list,tuple]:
            assert len(sigma) == 2
            self.sigma = np.sqrt(np.exp(sigma[0])*np.power(1.0+self._l,sigma[1]))
        else:
            self.sigma = sigma*np.ones(self.H)



    def _load_stan_model(self):

        _model_file = 'magnitude_colour_position'
        _model_file += '_sparse' if self.sparse else ''

        from cmdstanpy import CmdStanModel
        self.stan_model = CmdStanModel(stan_file = self.stan_model_directory+_model_file+'.stan')

    def _construct_stan_input(self):

        # Construct Y
        harmonic_to_pixel = self._azimuth[self._m,:].T * self._lambda[self._pixel_to_ring,:]


        self.stan_input = {'k':self.k,
                           'n':self.n,
                           'P':self.P,
                           'M':self.M,
                           'M_subspace':self.M_subspace,
                           'C':self.C,
                           'C_subspace':self.C_subspace,
                           'H':self.H,
                           'harmonic_to_pixel':self._harmonic_to_pixel,
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

    def _generate_spherical_harmonics(self,gsh_file):

        nside = self.nside
        lmax = self.lmax
        Npix = self.P

        # Form the l's and m's
        Nmodes = int((lmax+1)**2)
        Nmodes_hp = int((lmax+1)*(lmax+2)/2)
        l_hp,m_hp = hp.sphtfunc.Alm.getlm(lmax=lmax)
        assert Nmodes_hp == l_hp.size

        l, m = np.zeros(Nmodes,dtype=int), np.zeros(Nmodes,dtype=int)
        l[:Nmodes_hp],m[:Nmodes_hp] = l_hp,m_hp
        l[Nmodes_hp:],m[Nmodes_hp:] = l_hp[lmax+1:],-m_hp[lmax+1:]

        # Ring idxs of pixels with phi=0
        theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
        theta_ring, unique_idx, jpix = np.unique(theta, return_index=True, return_inverse=True)

        # Generate lambda
        _lambda = np.zeros((Nmodes, 4*nside-1))
        if False: # From scipy
            # For |m|>0 this comes out a factor of 2 smaller than the healpy version
            # For m<0 there's also a factor of (-1)^m difference
            for i,(_l,_m) in enumerate(zip(tqdm.tqdm(l),m)):
                _lambda[i] = (-1)**np.abs(_m) * np.real( scipy.special.sph_harm(np.abs(_m), _l, theta_ring*0., theta_ring) )
        else: # From healpy
            alm_hp = np.zeros(Nmodes_hp)
            for i,(_l,_m) in enumerate(zip(tqdm.tqdm(l),m)):
                i_hp = hp.sphtfunc.Alm.getidx(lmax, _l, np.abs(_m))
                alm_hp = np.zeros(Nmodes_hp)*(0.+0.j)
                # Get real component
                alm_hp[i_hp] = 1.+0.j
                map_hp = (1.+0.j)*hp.sphtfunc.alm2map(alm_hp,nside=nside, verbose=False)
                # Add imaginary component
                alm_hp[i_hp] = 0.+1.j
                map_hp += (0.-1.j)*hp.sphtfunc.alm2map(alm_hp,nside=nside, verbose=False)
                alm_hp[i_hp] = 0.+0.j
                map_hp /= np.exp(1.j*np.abs(_m)*phi)
                # Select unique latitude indices
                _lambda[i] = (-1)**np.abs(_m) * np.real(map_hp)[unique_idx]

                # Divide by 2
                if _m != 0:
                    _lambda[i] /= 2.0

        # Generate Exponential
        azimuth = np.ones((2*lmax+1,Npix))
        for _m in range(-lmax, lmax+1):
            if _m<0:   azimuth[_m+lmax] = np.sqrt(2) * np.sin(-_m*phi)
            elif _m>0: azimuth[_m+lmax] = np.sqrt(2) * np.cos(_m*phi)
            else: pass

        # Generate indices mapping m to alm
        lower, upper = np.zeros(2*lmax+1),np.zeros(2*lmax+1)
        for i, _m in enumerate(range(-lmax,lmax+1)):
            match = np.where(m==_m)[0]
            lower[i] = match[0]
            upper[i] = match[-1]

        save_kwargs = {'compression':"lzf", 'chunks':True, 'fletcher32':False, 'shuffle':True}
        with h5py.File(gsh_file, 'w') as f:
            # Create datasets
            f.create_dataset('lambda', data = _lambda, shape = (Nmodes, 4*nside-1,), dtype = np.float64, **save_kwargs)
            f.create_dataset('azimuth',data = azimuth, shape = (2*lmax+1, Npix, ),   dtype = np.float64, **save_kwargs)
            f.create_dataset('l',      data = l,       shape = (Nmodes,), dtype = np.uint32, scaleoffset=0, **save_kwargs)
            f.create_dataset('m',      data = m,       shape = (Nmodes,), dtype = np.int32, scaleoffset=0, **save_kwargs)
            f.create_dataset('pixel_to_ring',   data = jpix,    shape = (Npix,),   dtype = np.uint32, scaleoffset=0, **save_kwargs)
            f.create_dataset('lower',   data = lower,    shape = (2*lmax+1,),   dtype = np.uint32, scaleoffset=0, **save_kwargs)
            f.create_dataset('upper',   data = upper,    shape = (2*lmax+1,),   dtype = np.uint32, scaleoffset=0, **save_kwargs)
