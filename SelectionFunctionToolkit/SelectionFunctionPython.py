import sys, os
import numpy as np
import healpy as hp
import tqdm, h5py, time
import ray, multiprocessing, numba
import scipy.optimize
import scipy.sparse

from SelectionFunctionBase import Base
from SelectionFunctionChisel import Chisel

from PythonModels.wavelet_magnitude_colour_position import wavelet_magnitude_colour_position, wavelet_magnitude_colour_position_sparse, wavelet_x_sparse, wavelet_b_sparse

global lnlike_iter
global gnorm_iter
global tinit

def fcall(X):
    global tinit
    global lnlike_iter
    global gnorm_iter
    print(f't={int(time.time()-tinit):05d}, lnL={lnlike_iter:.0f}, gnorm={gnorm_iter:.0f}')

def print_log(message, logfile="data/vault/asfe2/logs/default_log.txt"):

    if os.path.exists(logfile): mode='a'
    else: mode='w'

    with open(logfile, mode) as f:
        f.write(message)
    print(message)


@ray.remote
class evaluate():

    def __init__(self, P, S, M, C, M_subspace, C_subspace, wavelet_args,
                       logfile='/data/asfe2/Projects/astrometry/PyOutput/log.txt',
                       savefile='/data/asfe2/Projects/astrometry/PyOutput/progress.h'):

        self.P=P
        self.S=S
        self.M=M
        self.C=C
        self.M_subspace=M_subspace
        self.C_subspace=C_subspace
        self.wavelet_args=wavelet_args

        self.logfile=logfile
        self.savefile=savefile

        self.x = np.zeros((self.P, self.M, self.C))
        self.lnL_grad = np.zeros((self.S, self.M_subspace, self.C_subspace))
        #self.MC = np.zeros((self.M, self.C))

        self.tinit = time.time()
        self.lnlike_iter = 0.
        self.gnorm_iter = 0.
        self.nfev = 0

    def evaluate_likelihood(self, z):

        x = np.zeros((self.P, self.M, self.C))
        lnL_grad = np.zeros((self.S, self.M_subspace, self.C_subspace))

        lnL, lnL_grad = wavelet_magnitude_colour_position_sparse(z.reshape((self.S, self.M_subspace, self.C_subspace)), \
                                                        self.M, self.C, self.P, *self.wavelet_args, x, lnL_grad)

        return lnL, lnL_grad

    def merge_likelihoods(self, evaluations):

        lnL=0.
        grad = np.zeros((self.S, self.M_subspace, self.C_subspace))

        for e in evaluations:
            lnL += e[0]
            grad += e[1]

        self.lnlike_iter = lnL
        self.gnorm_iter = np.sqrt(np.sum(grad**2))/(self.S * self.M_subspace * self.C_subspace)
        self.nfev += 1

        return -lnL, -grad.flatten()

    def save_progress(self, X):

        if os.path.exists(self.savefile): mode='a'
        else: mode='w'

        with h5py.File(self.savefile, mode) as hf:
            hf.create_dataset(str(self.nfev), data=X)

    def fcall(self, X):

        self.save_progress(X)
        print_log(f't={int(time.time()-self.tinit):03d}, n={self.nfev:02d}, lnL={self.lnlike_iter:.0f}, gnorm={self.gnorm_iter:.5f}\n', logfile=self.logfile)


class pyChisel(Chisel):

    basis_keyword = 'wavelet'

    def evaluate_likelihood(self, z):

        def likelihood(z, S, M, C, P):
            lnL, grad = wavelet_magnitude_colour_position_sparse(z.reshape((S, M, C)), M, C, P, *self.wavelet_args)
            return -lnL, -grad.flatten()

        return likelihood(z.flatten(), self.S, self.M, self.C, self.P)

    def minimize_ray(self, z0, ncores=2, bounds=None, method='BFGS', **scipy_kwargs):

        tstart = time.time()

        numba.set_num_threads(1)
        logfile = '/data/asfe2/Projects/astrometry/PyOutput/'+self.file_root+'_log.txt'
        savefile = '/data/asfe2/Projects/astrometry/PyOutput/'+self.file_root+'_progress.h'
        if os.path.exists(savefile):
            raise OSError(f"File {savefile} already exists, won't overwrite.")

        print('Initialising arguments.')
        self._generate_args(sparse=True)
        self._generate_args_ray(nsets=ncores-1, sparse=True)

        if True:
            print('Initialising ray processes.')
            ray.shutdown()
            ray.init()
            evaluators = [evaluate.remote(self.P_ray[i], self.S, self.M, self.C, self.M_subspace, self.C_subspace, self.wavelet_args_ray[i], logfile=logfile, savefile=savefile) for i in range(ncores-1)]
            def likelihood(z):
                evaluations = [e.evaluate_likelihood.remote(z) for e in evaluators]
                combination = evaluators[0].merge_likelihoods.remote(ray.get(evaluations))
                return ray.get(combination)
            callback=evaluators[0].fcall.remote
        if False:
            evaluators = [evaluate(self.P_ray[i], self.S, self.M, self.C, self.M_subspace, self.C_subspace, self.wavelet_args_ray[i]) for i in range(ncores-1)]
            def likelihood(z):
                evaluations = [e.evaluate_likelihood(z) for e in evaluators]
                return evaluators[0].merge_likelihoods(evaluations)
            callback=evaluators[0].fcall

        global tinit; tinit = time.time()

        print('Running optimizer.')
        res = scipy.optimize.minimize(likelihood, z0.flatten(), method=method, jac=True, bounds=bounds, callback=callback, **scipy_kwargs)
        ray.shutdown()

        print('Processing results.')
        self.optimum_z = res['x'].reshape((self.S, self.M_subspace, self.C_subspace))
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

    def minimize(self, z0, ncores=2, bounds=None, method='BFGS', **scipy_kwargs):

        tstart = time.time()
        self._generate_args(sparse=True)

        print('Number of threads: ', numba.get_num_threads())
        numba.set_num_threads(ncores)
        print('Number of threads: ', numba.get_num_threads())

        def likelihood(z):
            x = np.zeros((self.P, self.M, self.C))
            lnL_grad = np.zeros((self.S, self.M_subspace, self.C_subspace))
            lnL, grad = wavelet_magnitude_colour_position_sparse(z.reshape((self.S, self.M_subspace, self.C_subspace)), self.M, self.C, self.P, *self.wavelet_args, x, lnL_grad)
            global lnlike_iter; lnlike_iter = lnL
            global gnorm_iter; gnorm_iter = np.sum(np.abs(grad))
            return -lnL, -grad.flatten()

        global tinit
        tinit = time.time()

        res = scipy.optimize.minimize(likelihood, z0.flatten(), method=method, jac=True, bounds=bounds, callback=fcall, **scipy_kwargs)

        self.optimum_z = res['x'].reshape((self.S, self.M_subspace, self.C_subspace))
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

    def _evaluate_likelihood(self, z, ncores=1, generate=True):

        if generate:
            numba.set_num_threads(ncores)
            self._generate_args(sparse=True)

        x = np.zeros((self.P, self.M, self.C))
        lnL_grad = np.zeros((self.S, self.M_subspace, self.C_subspace))
        lnL, grad = wavelet_magnitude_colour_position_sparse(z.reshape((self.S, self.M_subspace, self.C_subspace)), self.M, self.C, self.P, *self.wavelet_args, x, lnL_grad)
        return lnL, grad

    def _get_bx(self, z):

        MC = np.zeros((self.M, self.C))

        b = np.zeros((self.S, self.M, self.C))
        b = wavelet_b_sparse(z.reshape((self.S, self.M_subspace, self.C_subspace)), self.M, self.C, self.S, *self.wavelet_args[2:], b, MC)

        x = np.zeros((self.P, self.M, self.C))
        x = wavelet_x_sparse(z.reshape((self.S, self.M_subspace, self.C_subspace)), self.M, self.C, self.P, *self.wavelet_args[2:], x, MC)

        return b, x

    def _cholesky_args(self, sparse=False):

        if sparse:
            self.cholesky_m = scipy.sparse.csr_matrix((self.stan_input['cholesky_w_m'],
                                                self.stan_input['cholesky_v_m']-1,
                                                self.stan_input['cholesky_u_m']-1)).toarray()
            self.cholesky_c = scipy.sparse.csr_matrix((self.stan_input['cholesky_w_c'],
                                            self.stan_input['cholesky_v_c']-1,
                                            self.stan_input['cholesky_u_c']-1)).toarray()

            cholesky_u_m = np.zeros(len(self.stan_input['cholesky_v_m']), dtype=int)
            for iS, iY in enumerate(self.stan_input['cholesky_u_m'][1:]-1):
                cholesky_u_m[iY:] += 1
            cholesky_u_c = np.zeros(len(self.stan_input['cholesky_v_c']), dtype=int)
            for iS, iY in enumerate(self.stan_input['cholesky_u_c'][1:]-1):
                cholesky_u_c[iY:] += 1

            cholesky_args = [cholesky_u_m,
                             self.stan_input['cholesky_v_m']-1,
                             self.stan_input['cholesky_w_m'],
                             cholesky_u_c,
                             self.stan_input['cholesky_v_c']-1,
                             self.stan_input['cholesky_w_c']]

            self.wavelet_model = wavelet_magnitude_colour_position_sparse
        elif not sparse:
            self.cholesky_m = scipy.sparse.csr_matrix((self.stan_input['cholesky_w_m'],
                                                self.stan_input['cholesky_v_m']-1,
                                                self.stan_input['cholesky_u_m']-1)).toarray()
            self.cholesky_c = scipy.sparse.csr_matrix((self.stan_input['cholesky_w_c'],
                                            self.stan_input['cholesky_v_c']-1,
                                            self.stan_input['cholesky_u_c']-1)).toarray()
            cholesky_args = [self.cholesky_m, self.cholesky_c]

        return cholesky_args

    def _generate_args(self, sparse=False):

        cholesky_args = self._cholesky_args(sparse=sparse)

        self.wavelet_model = wavelet_magnitude_colour_position

        lnL_grad = np.zeros((self.S, self.M, self.C))
        x = np.zeros((self.M, self.C))

        self.wavelet_args = [np.moveaxis(self.k, -1,0).astype(np.int64).copy(),np.moveaxis(self.n, -1,0).astype(np.int64).copy()] \
                          + [self.stan_input[arg].copy() for arg in ['mu', 'sigma']]\
                          + [self.stan_input['wavelet_un'].copy()-1,
                             self.stan_input['wavelet_v'].copy()-1,
                             self.stan_input['wavelet_w'].copy()]\
                          + cholesky_args

    def _generate_args_ray(self, nsets=1, sparse=False):

        cholesky_args = self._cholesky_args(sparse=sparse)

        P_ray = np.zeros(nsets, dtype=int) + self.P//nsets
        P_ray[:self.P - np.sum(P_ray)] += 1
        print('P sets: ', P_ray, np.sum(P_ray))

        wavelet_u = self.stan_input['wavelet_un'].copy()-1
        wavelet_v = self.stan_input['wavelet_v'].copy()-1
        wavelet_w = self.stan_input['wavelet_w'].copy()

        self.wavelet_args_ray = []
        iP = 0
        for iset in range(nsets):

            lnL_grad = np.zeros((self.S, self.M, self.C))
            x = np.zeros((self.M, self.C))

            wavelet_args_set  = [np.moveaxis(self.k, -1,0).astype(np.int64)[iP:iP+P_ray[iset]].copy(),
                                 np.moveaxis(self.n, -1,0).astype(np.int64)[iP:iP+P_ray[iset]].copy()] \
                              + [self.stan_input[arg].copy() for arg in ['mu', 'sigma']] \
                              + [arg[int(self.stan_input['wavelet_u'][iP]-1):int(self.stan_input['wavelet_u'][iP+P_ray[iset]]-1)].copy() \
                                                                        for arg in [wavelet_u,wavelet_v,wavelet_w]] \
                              + cholesky_args

            self.wavelet_args_ray.append(wavelet_args_set)
            iP += P_ray[iset]

        self.P_ray = P_ray
