""" Compute the real-valued spherical harmonics for nested Healpix maps. """

import sys
import numpy as np, healpy as hp, scipy.stats
import h5py, tqdm
from numba import njit

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
    with h5py.File('./SphericalHarmonics/sphericalharmonics_decomposed_quick_nside{0}_lmax{1}.h5'.format(nside,lmax), 'w') as f:
        # Create datasets
        f.create_dataset('lambda', data = _lambda, shape = (Nmodes, 4*nside-1,), dtype = np.float64, **save_kwargs)
        f.create_dataset('azimuth',data = azimuth, shape = (2*lmax+1, Npix, ),   dtype = np.float64, **save_kwargs)
        f.create_dataset('l',      data = l,       shape = (Nmodes,), dtype = np.uint32, scaleoffset=0, **save_kwargs)
        f.create_dataset('m',      data = m,       shape = (Nmodes,), dtype = np.int32, scaleoffset=0, **save_kwargs)
        f.create_dataset('jpix',   data = jpix,    shape = (Npix,),   dtype = np.uint32, scaleoffset=0, **save_kwargs)
        f.create_dataset('lower',   data = lower,    shape = (2*lmax+1,),   dtype = np.uint32, scaleoffset=0, **save_kwargs)
        f.create_dataset('upper',   data = upper,    shape = (2*lmax+1,),   dtype = np.uint32, scaleoffset=0, **save_kwargs)
