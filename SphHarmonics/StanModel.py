import sys, os, pickle

import numpy as np, healpy as hp,h5py
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

#lmax=5, nside=16
nside=128; lmax=85
nlm = int(lmax*(lmax+1)/2+0.01);

if True:# Build stan model
    binomial_code = """
    functions{
            vector sphharm_sum (vector alm, matrix lambda, matrix azimuth, int[] m, int lmax, int[] jpix, int Nring, int Npix, int H) {
                vector[Npix] result = rep_vector(0.,Npix);
                matrix[2*lmax+1,Nring] F;
                for (nu in 1:H) {
                    for (j in 1:Nring) {
                        F[m[nu]+lmax+1, j] += lambda[nu,j]*alm[nu];
                    }
                }
                for (i in 1:Npix){
                    for (m_idx in -lmax:lmax){
                        result[i] += F[m_idx+lmax+1, jpix[i]+1]*azimuth[m_idx+lmax+1, i];
                    }
                }
                return result;
            }
    }
    data {
      int<lower=0> J;            // number of pixels
      int<lower=0> Nring;        // number of HEALPix isolatitude rings
      int<lower=0> H;            // number of harmonics
      int<lower=0> lmax;         // max l of hamonics
      vector[J] k;               // astrometry sample counts
      vector[J] n;               // full sample counts
      // matrix[J,H] Ylm;           // spherical harmonics
      matrix[H,Nring] lambda;    // spherical harmonics decomposed
      matrix[2*lmax+1,J] azimuth;// spherical harmonics decomposed
      int m[H];
      int jpix[J];
    }
    parameters {
      vector[H] alm;               // logit selection probability
    }
    transformed parameters {
      // vector[J] x = Ylm * alm;
      vector[J] x = sphharm_sum(alm, lambda, azimuth, m, lmax, jpix, Nring, J, H);
    }
    model {
      target += normal_lpdf(alm | 0,1);        // log-prior
      target += k .*x - n .*log(1+exp(x));     // log-likelihood
    }
    """
    bn = pystan.StanModel(model_code=binomial_code)
    save(bn,'binomial_sphharm.pic')
else:# Load stan model
    bn = load('binomial_sphharm.pic')


if True:# Generate mock data
    l = hp.sphtfunc.Alm.getlm(lmax=lmax-1)[0]
    scale = 10.0/(1.0+l)**2
    _alm = np.random.normal(0,scale,int(lmax*(lmax+1)/2)) + 1j*np.random.normal(0,scale,int(lmax*(lmax+1)/2))
    _map = hp.sphtfunc.alm2map(_alm,nside=nside,verbose=False)

    sample = {}
    sample['hpx'] = np.random.choice(np.arange(hp.nside2npix(nside)), size=int(1e6), replace=True)
    sample['p'] = expit(_map[sample['hpx']])
    sample['selected'] = (np.random.rand(len(sample['hpx'])) < sample['p']).astype(int)

    k = stats.binned_statistic(sample['hpx'], sample['selected'],
                           bins=np.arange(hp.nside2npix(nside)+1)-0.5, statistic='sum').statistic
    n = stats.binned_statistic(sample['hpx'], sample['selected'],
                           bins=np.arange(hp.nside2npix(nside)+1)-0.5, statistic='count').statistic

if False: # Load spherical harmonic processes
    # generate sph harm grid
    Ylm_grid = np.zeros((2*nlm, hp.nside2npix(nside)))
    for ii in tqdm.tqdm_notebook(range(nlm)):
        alm_grid = np.zeros(nlm)
        alm_grid[ii]=1
        Ylm_grid[ii] = hp.sphtfunc.alm2map(alm_grid+0.j*alm_grid,nside=nside,verbose=False)
        Ylm_grid[nlm+ii] = hp.sphtfunc.alm2map(0.*alm_grid+1.j*alm_grid,nside=nside,verbose=False)

    binomial_data = {"J":hp.nside2npix(nside),
                     "H":2*nlm,
                     "k": k,
                     "n": n,
                     "Ylm": Ylm_grid.T}

if True: # Load spherical harmonic decomposition
    data = {}
    with h5py.File('/data/asfe2/Projects/gaia_edr3/sphericalharmonics_decomposed_nside{0}_lmax{1}.h5'.format(nside,lmax), 'r') as hf:
        for key in hf.keys():
            data[key] = hf[key][...]

    binomial_data = {"J":hp.nside2npix(nside),
                     "Nring":data['lambda'].shape[1],
                     "H":data['m'].shape[0],
                     "lmax":lmax,
                     "k": k,
                     "n": n,
                     "lambda":data['lambda'],
                     "azimuth":data['azimuth'],
                     "m":data['m'],
                     "jpix":data['jpix']}

    fit = bn.sampling(data=binomial_data, iter=100, chains=1, n_jobs=1)
