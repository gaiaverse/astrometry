""" Compute the real-valued spherical harmonics for nested Healpix maps. """

import sys
import numpy as np, healpy as hp, scipy.stats
import h5py, tqdm
from numba import njit


@njit
def sphharm_sum_quick(alm, _lambda, azimuth, m, lmax, jpix, Nring, F):
    """
    alm - ndarray - (Nmode)
    _lambda - ndarray - (Nmode, Nring)
    azimuth - ndarray - (2*lmax+1, Npix)
    mmode - ndarray - (Nmode)
    lmax - int
    jpix - ndarray - (Npix)
    Nring - int
    """
    nu = 0
    for _m in m:
        for j in range(Nring):
            F[_m+lmax, j] += _lambda[nu,j]*alm[nu]
        nu += 1

    result = 0.
    i = 0
    for j in jpix:
        for _m in range(-lmax, lmax+1):
            result += F[_m+lmax, j]*azimuth[_m+lmax, i]
        i += 1

    return result

def sphharm_sum_truth(alm, nside, lmax):

    Npix = hp.nside2npix(nside)

    # Form the l's and m's
    l = np.concatenate([np.repeat(_l,2*_l+1) for _l in range(lmax+1)]).astype(np.int)
    m = np.concatenate([np.arange(-_l,_l+1e-9) for _l in range(lmax+1)]).astype(np.int)
    l_hp,m_hp = hp.sphtfunc.Alm.getlm(lmax=lmax)
    Nmodes = int((lmax+1)**2)
    Nmodes_healpy = int((lmax+1)*(lmax+2)/2)

    # Ring idxs of pixels with phi=0
    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))

    # Generate Exponential
    result = np.zeros(Npix)
    for i,(_l,_m) in enumerate(zip(tqdm.tqdm(l),m)):
        if _m<0:   result += alm[i]*(-1)**(-_m) * np.sqrt(2) * np.real( scipy.special.sph_harm(np.abs(_m), _l, phi*0., theta) ) * np.sin(-_m*phi)
        elif _m>0: result += alm[i]*(-1)**_m    * np.sqrt(2) * np.real( scipy.special.sph_harm(np.abs(_m), _l, phi*0., theta) ) * np.cos( _m*phi)
        else:      result += alm[i]* np.real( scipy.special.sph_harm(np.abs(_m), _l, phi*0., theta) )

    return np.sum(result)

@njit
def real_harmonics(_lambda, azimuth, m, lmax, jpix):
    """
    Get real harmonic coefficients from _lambda, azimuth
    -----
    _lambda - ndarray - (Nmode, Nring)
    azimuth - ndarray - (2*lmax+1, Npix)
    m - ndarray - (Nmode)
    lmax - int
    jpix - ndarray - (Npix)
    """
    Y = np.zeros((m.shape[0], jpix.shape[0]))
    for i,j in enumerate(jpix):
        for nu, _m in enumerate(m):
            Y[nu,i] = _lambda[nu,j]*azimuth[_m+lmax,i]
    return Y

if __name__=='__main__':

    nside = int(sys.argv[1])
    lmax = int(sys.argv[2])
    Npix = hp.nside2npix(nside)

    # Form the l's and m's
    l = np.concatenate([np.repeat(_l,2*_l+1) for _l in range(lmax+1)]).astype(np.int)
    m = np.concatenate([np.arange(-_l,_l+1e-9) for _l in range(lmax+1)]).astype(np.int)
    l_hp,m_hp = hp.sphtfunc.Alm.getlm(lmax=lmax)
    Nmodes = int((lmax+1)**2)
    Nmodes_healpy = int((lmax+1)*(lmax+2)/2)

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
    if True: # From healpy
        alm_hp = np.zeros(Nmodes_healpy)
        for i,(_l,_m) in enumerate(zip(tqdm.tqdm(l),m)):
            i_hp = hp.sphtfunc.Alm.getidx(lmax, _l, np.abs(_m))
            alm_hp = np.zeros(Nmodes_healpy)*(0.+0.j)
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

    # Generate Exponential
    azimuth = np.ones((2*lmax+1,Npix))
    for _m in range(-lmax, lmax+1):
        if _m<0:   azimuth[_m+lmax] = np.sqrt(2) * np.sin(-_m*phi)
        elif _m>0: azimuth[_m+lmax] = np.sqrt(2) * np.cos(_m*phi)
        else: pass

    save_kwargs = {'compression':"lzf", 'chunks':True, 'fletcher32':False, 'shuffle':True}
    with h5py.File('/data/asfe2/Projects/gaia_edr3/sphericalharmonics_decomposed_nside{0}_lmax{1}.h5'.format(nside,lmax), 'w') as f:
        # Create datasets
        f.create_dataset('lambda', data = _lambda, shape = (Nmodes, 4*nside-1,), dtype = np.float64, **save_kwargs)
        f.create_dataset('azimuth',data = azimuth, shape = (2*lmax+1, Npix, ),   dtype = np.float64, **save_kwargs)
        f.create_dataset('l',      data = l,       shape = (Nmodes,), dtype = np.uint32, scaleoffset=0, **save_kwargs)
        f.create_dataset('m',      data = m,       shape = (Nmodes,), dtype = np.int32, scaleoffset=0, **save_kwargs)
        f.create_dataset('jpix',   data = jpix,    shape = (Npix,),   dtype = np.uint32, scaleoffset=0, **save_kwargs)
