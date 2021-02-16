import sys, os, pickle

import numpy as np, healpy as hp
import emcee, tqdm, corner, pystan
from scipy import stats, special, linalg
import scipy.optimize

def expit(x):
    return np.exp(x)/(1+np.exp(x))

def save(obj, filename):
    """Save compiled models for reuse."""
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load(filename):
    """Reload compiled models for reuse."""
    import pickle
    return pickle.load(open(filename, 'rb'))

lmax=5
nlm = int(lmax*(lmax+1)/2+0.01);
nside=16

if True:# Build stan model
    binomial_code = """
    functions{
    }
    data {
      int<lower=0> J;            // number of pixels
      int<lower=0> H;            // number of harmonics
      vector[J] k;               // astrometry sample counts
      vector[J] n;               // full sample counts
      matrix[J,H] Ylm;             // spherical harmonics
    }
    parameters {
      vector[H] alm;               // logit selection probability
    }
    transformed parameters {
      vector[J] x = Ylm * alm;
    }
    model {
      target += normal_lpdf(alm | 0,1);        // log-prior
      target += k .*x - n .*log(1+exp(x));   // log-likelihood
    }
    """
    bn = pystan.StanModel(model_code=binomial_code)
    save(bn,'binomial_sphharm.pic')
else:# Load stan model
    bn = load('binomial_sphharm.pic')


if False:# Generate mock data
    sample = {}
    sample['hpx'] = np.random.choice(np.arange(hp.nside2npix(nside)), size=int(1e6), replace=True)
    sample['p'] = expit(_map[sample['hpx']])
    sample['selected'] = (np.random.rand(len(sample['hpx'])) < sample['p']).astype(int)


if False: # Load spherical harmonic processes
    # generate sph harm grid
    Ylm_grid = np.zeros((2*nlm, hp.nside2npix(nside)))
    for ii in tqdm.tqdm_notebook(range(nlm)):
        alm_grid = np.zeros(nlm)
        alm_grid[ii]=1
        Ylm_grid[ii] = hp.sphtfunc.alm2map(alm_grid+0.j*alm_grid,nside=nside,verbose=False)
        Ylm_grid[nlm+ii] = hp.sphtfunc.alm2map(0.*alm_grid+1.j*alm_grid,nside=nside,verbose=False)

if False:

    binomial_data = {"J":hp.nside2npix(nside),
                     "H":2*nlm,
                     "k": k,
                     "n": n,
                     "Ylm": Ylm_grid.T}

    fit = bn.sampling(data=norm_dat, iter=100, chains=1, n_jobs=1)
