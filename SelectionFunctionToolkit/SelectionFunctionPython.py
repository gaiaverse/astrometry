import numpy as np
import healpy as hp
import tqdm, h5py, time
import scipy.optimize
import scipy.sparse

from SelectionFunctionBase import Base
from SelectionFunctionChisel import Chisel

from PythonModels.wavelet_magnitude_colour_position import wavelet_magnitude_colour_position, wavelet_magnitude_colour_position_sparse, get_wavelet_x



class pyChisel(Chisel):

    basis_keyword = 'wavelet'

    def evaluate_likelihood(self, z):

        def likelihood(z, S, M, C, P):
            lnL, grad = wavelet_magnitude_colour_position(z.reshape((S, M, C)), M, C, P, *self.wavelet_args)
            return -lnL, -grad.flatten()

        return likelihood(z.flatten(), self.S, self.M, self.C, self.P)

    def minimize(self, z0, bounds=None, method=None, **scipy_kwargs):

        tstart = time.time()
        self._generate_args(sparse=True)

        if method is None:
            if bounds is None: method='BFGS'
            else: method='L-BFGS-B'

        def likelihood(z):
            #lnL, grad = wavelet_magnitude_colour_position(z.reshape((self.S, self.M, self.C)), self.M, self.C, self.P, *self.wavelet_args)
            lnL, grad = wavelet_magnitude_colour_position_sparse(z.reshape((self.S, self.M, self.C)), self.M, self.C, self.P, *self.wavelet_args)
            return -lnL, -grad.flatten()

        res = scipy.optimize.minimize(likelihood, z0.flatten(), method='L-BFGS-B', jac=True, bounds=bounds, **scipy_kwargs)

        self.optimum_z = res['x'].reshape((self.S, self.M, self.C))

        #self.optimum_b = self.stan_input['mu'][:,None,None] + self.stan_input['sigma'][:,None,None] * (self.cholesky_m @ self.optimum_z @ self.cholesky_c.T)

        self.optimum_b, self.optimum_x = self._get_bx(self.optimum_z)

        if self.nest: self.optimum_x = self._ring_to_nest(np.moveaxis(self.optimum_x, 0, -1))

        # Save optimum to h5py
        self.optimum_results_file = self.file_root+'_scipy_results.h5'
        with h5py.File(self.stan_output_directory + self.optimum_results_file, 'w') as orf:
            orf.create_dataset('opt_runtime', data = time.time()-tstart)
            orf.create_dataset('lnP', data = -res['fun'])
            orf.create_dataset('z', data = self.optimum_z, dtype = np.float64, compression = 'lzf', chunks = True)
            orf.create_dataset('b', data = self.optimum_b, dtype = np.float64, compression = 'lzf', chunks = True)
            orf.create_dataset('x', data = self.optimum_x, dtype = np.float64, compression = 'lzf', chunks = True)
        print(f'Optimum values stored in {self.stan_output_directory + self.optimum_results_file}')

        return res

    def _get_bx(self, z):

        Y = scipy.sparse.csr_matrix((self.stan_input['wavelet_w'], self.stan_input['wavelet_v']-1, self.stan_input['wavelet_u']-1))
        Cm = scipy.sparse.csr_matrix((self.stan_input['cholesky_w_m'], self.stan_input['cholesky_v_m']-1, self.stan_input['cholesky_u_m']-1))
        Cc = scipy.sparse.csr_matrix((self.stan_input['cholesky_w_c'], self.stan_input['cholesky_v_c']-1, self.stan_input['cholesky_u_c']-1))

        b = self.stan_input['mu'][:,None,None] + self.stan_input['sigma'][:,None,None] * (self.cholesky_m @ z @ self.cholesky_c.T)

        x = np.moveaxis(np.array([Y @ b[:,:,iC] for iC in range(self.C)]), 0,2)

        return b, x


    def _generate_args(self, sparse=False):

        if sparse:
            cholesky_args = [self.stan_input['cholesky_u_m']-1,
                             self.stan_input['cholesky_v_m']-1,
                             self.stan_input['cholesky_w_m'],
                             self.stan_input['cholesky_u_c']-1,
                             self.stan_input['cholesky_v_c']-1,
                             self.stan_input['cholesky_w_c']]
            self.wavelet_model = wavelet_magnitude_colour_position_sparse
        elif not sparse:
            self.cholesky_m = scipy.sparse.csr_matrix((self.stan_input['cholesky_w_m'],
                                                self.stan_input['cholesky_v_m']-1,
                                                self.stan_input['cholesky_u_m']-1), shape=(self.M,self.M)).toarray()
            self.cholesky_c = scipy.sparse.csr_matrix((self.stan_input['cholesky_w_c'],
                                            self.stan_input['cholesky_v_c']-1,
                                            self.stan_input['cholesky_u_c']-1), shape=(self.C,self.C)).toarray()
            cholesky_args = [self.cholesky_m, self.cholesky_c]
            self.wavelet_model = wavelet_magnitude_colour_position

        lnL_grad = np.zeros((self.S, self.M, self.C))
        x = np.zeros((self.M, self.C))

        self.wavelet_args = [np.moveaxis(self.k, -1,0).astype(np.int64).copy(),np.moveaxis(self.n, -1,0).astype(np.int64).copy()] \
                          + [self.stan_input[arg].copy() for arg in ['mu', 'sigma', 'wavelet_u', 'wavelet_v', 'wavelet_w']]\
                          + cholesky_args + [lnL_grad,x]
        self.wavelet_args[4]-=1
        self.wavelet_args[5]-=1
