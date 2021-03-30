import numpy as np
import healpy as hp
import tqdm, h5py, time
import ray, multiprocessing
import scipy.optimize
import scipy.sparse

from SelectionFunctionBase import Base
from SelectionFunctionChisel import Chisel

from PythonModels.wavelet_magnitude_colour_position import wavelet_magnitude_colour_position, wavelet_magnitude_colour_position_sparse, get_wavelet_x

global lnlike_iter
global gnorm_iter
global tinit

def fcall(X):
    global tinit
    global lnlike_iter
    global gnorm_iter
    print(f't={int(time.time()-tinit):05d}, lnL={lnlike_iter:.0f}, gnorm={gnorm_iter:.0f}')

@ray.remote
class evaluate():

    def __init__(self, S, M, C, P, wavelet_args):

        self.S=S
        self.M=M
        self.C=C
        self.P=P
        self.wavelet_args=wavelet_args

    def evaluate_likelihood(self, z):

        return wavelet_magnitude_colour_position_sparse(z.reshape((self.S, self.M, self.C)), self.M, self.C, self.P, *self.wavelet_args[:-2])

@ray.remote
class combine():

    def __init__(self, S, M, C):

        self.S=S
        self.M=M
        self.C=C

        self.tinit = time.time()
        self.lnlike_iter = 0.
        self.gnorm_iter = 0.

    def merge_likelihoods(self, evaluations):

        lnL=0.
        grad = np.zeros((self.S, self.M, self.C))

        for e in evaluations:
            lnL += e[0]
            grad += e[1]

        self.lnlike_iter = lnL
        self.gnorm_iter = np.sum(np.abs(grad))

        return -lnL, -grad.flatten()

    def fcall(self, X):
        print(f't={int(time.time()-self.tinit):05d}, lnL={self.lnlike_iter:.0f}, gnorm={self.gnorm_iter:.0f}')

class pyChisel(Chisel):

    basis_keyword = 'wavelet'

    def evaluate_likelihood(self, z):

        def likelihood(z, S, M, C, P):
            lnL, grad = wavelet_magnitude_colour_position_sparse(z.reshape((S, M, C)), M, C, P, *self.wavelet_args)
            return -lnL, -grad.flatten()

        return likelihood(z.flatten(), self.S, self.M, self.C, self.P)

    def minimize_ray(self, z0, ncores=2, bounds=None, method='BFGS', **scipy_kwargs):

        tstart = time.time()

        print('Initialising arguments.')
        self._generate_args_ray(nsets=ncores-1, sparse=True)

        if False:
            print('Initialising ray processes.')
            ray.init()
            evaluators = [evaluate.remote(self.S, self.M, self.C, self.P_ray[i], self.wavelet_args_ray[i]) for i in range(ncores-1)]
            combiner = combine.remote(self.S, self.M, self.C)
            def likelihood(z):
                evaluations = [e.evaluate_likelihood.remote(z) for e in evaluators]
                combination = combiner.merge_likelihoods.remote(ray.get(evaluations))
                return ray.get(combination)
            callback=combiner.fcall.remote
        if True:
            print('Initialising processes.')
            evaluators = [evaluate(self.S, self.M, self.C, self.P_ray[i], self.wavelet_args_ray[i]) for i in range(ncores-1)]
            combiner = combine(self.S, self.M, self.C)
            def likelihood(z):
                evaluations = [e.evaluate_likelihood(z) for e in evaluators]
                combination = combiner.merge_likelihoods(evaluations)
                return combination
            callback=combiner.fcall

        global tinit; tinit = time.time()

        print('Running optimizer.')
        res = scipy.optimize.minimize(likelihood, z0.flatten(), method=method, jac=True, bounds=bounds, callback=callback, **scipy_kwargs)

        print('Processing results.')
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

    def minimize(self, z0, bounds=None, method='BFGS', **scipy_kwargs):

        tstart = time.time()
        self._generate_args(sparse=True)

        def likelihood(z):
            lnL, grad = wavelet_magnitude_colour_position_sparse(z.reshape((self.S, self.M, self.C)), self.M, self.C, self.P, *self.wavelet_args)
            global lnlike_iter; lnlike_iter = lnL
            global gnorm_iter; gnorm_iter = np.sum(np.abs(grad))
            return -lnL, -grad.flatten()

        global tinit
        tinit = time.time()

        res = scipy.optimize.minimize(likelihood, z0.flatten(), method=method, jac=True, bounds=bounds, callback=fcall, **scipy_kwargs)

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

    def _cholesky_args(self, sparse=False):

        if sparse:
            cholesky_args = [self.stan_input['cholesky_u_m']-1,
                             self.stan_input['cholesky_v_m']-1,
                             self.stan_input['cholesky_w_m'],
                             self.stan_input['cholesky_u_c']-1,
                             self.stan_input['cholesky_v_c']-1,
                             self.stan_input['cholesky_w_c']]
            self.cholesky_m = scipy.sparse.csr_matrix((self.stan_input['cholesky_w_m'],
                                                self.stan_input['cholesky_v_m']-1,
                                                self.stan_input['cholesky_u_m']-1), shape=(self.M,self.M)).toarray()
            self.cholesky_c = scipy.sparse.csr_matrix((self.stan_input['cholesky_w_c'],
                                            self.stan_input['cholesky_v_c']-1,
                                            self.stan_input['cholesky_u_c']-1), shape=(self.C,self.C)).toarray()
            self.wavelet_model = wavelet_magnitude_colour_position_sparse
        elif not sparse:
            self.cholesky_m = scipy.sparse.csr_matrix((self.stan_input['cholesky_w_m'],
                                                self.stan_input['cholesky_v_m']-1,
                                                self.stan_input['cholesky_u_m']-1), shape=(self.M,self.M)).toarray()
            self.cholesky_c = scipy.sparse.csr_matrix((self.stan_input['cholesky_w_c'],
                                            self.stan_input['cholesky_v_c']-1,
                                            self.stan_input['cholesky_u_c']-1), shape=(self.C,self.C)).toarray()
            cholesky_args = [self.cholesky_m, self.cholesky_c]

        return cholesky_args

    def _generate_args(self, sparse=False):

        cholesky_args = self._cholesky_args(sparse=sparse)

        self.wavelet_model = wavelet_magnitude_colour_position

        lnL_grad = np.zeros((self.S, self.M, self.C))
        x = np.zeros((self.M, self.C))

        self.wavelet_args = [np.moveaxis(self.k, -1,0).astype(np.int64).copy(),np.moveaxis(self.n, -1,0).astype(np.int64).copy()] \
                          + [self.stan_input[arg].copy() for arg in ['mu', 'sigma', 'wavelet_u', 'wavelet_v', 'wavelet_w']]\
                          + cholesky_args + [lnL_grad,x]
        self.wavelet_args[4]-=1
        self.wavelet_args[5]-=1

    def _generate_args_ray(self, nsets=1, sparse=False):

        cholesky_args = self._cholesky_args(sparse=sparse)

        P_ray = np.zeros(nsets, dtype=int) + self.P//nsets
        P_ray[:self.P - np.sum(P_ray)] += 1
        print('P sets: ', P_ray, np.sum(P_ray))

        self.wavelet_args_ray = []
        iP = 0
        for iset in range(nsets):

            lnL_grad = np.zeros((self.S, self.M, self.C))
            x = np.zeros((self.M, self.C))

            #print(self.stan_input['wavelet_u'][iP])

            wavelet_args_set  = [np.moveaxis(self.k, -1,0).astype(np.int64)[iP:iP+P_ray[iset]].copy(),
                                 np.moveaxis(self.n, -1,0).astype(np.int64)[iP:iP+P_ray[iset]].copy()] \
                              + [self.stan_input[arg].copy() for arg in ['mu', 'sigma']] \
                              + [self.stan_input['wavelet_u'][iP:iP+P_ray[iset]+1].copy() - self.stan_input['wavelet_u'][iP],] \
                              + [self.stan_input[arg][int(self.stan_input['wavelet_u'][iP]-1):int(self.stan_input['wavelet_u'][iP+P_ray[iset]]-1)] \
                                                                        for arg in ['wavelet_v', 'wavelet_w']] \
                              + cholesky_args + [lnL_grad,x]
            wavelet_args_set[5] -= 1

            self.wavelet_args_ray.append(wavelet_args_set)
            iP += P_ray[iset]

        self.P_ray = P_ray
