import numpy as np
import healpy as hp
import tqdm
import h5py
from SelectionFunctionBase import Base

class Chisel(Base):
    
    basis_keyword = 'wavelet'
        
    def _process_basis_options(self,basis_options):
        self.jmax = basis_options['jmax'] if 'jmax' in basis_options.keys() else 0
        self.B = basis_options['B'] if 'B' in basis_options.keys() else 2.0
        self.wavelet_tol = basis_options['wavelet_tol'] if 'wavelet_tol' in basis_options.keys() else 0.0
        self.spherical_basis_file = f'{self.basis_keyword}_nside{self.nside}_jmax{self.jmax}_B{self.B}_tol{self.wavelet_tol}.h5'
        
        assert self.B > 1.0
        assert self.wavelet_tol >= 0.0
        self.S = 4**(self.jmax + 2) - 3
        
            
    def _process_sigma_basis_specific(self,sigma):
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
            
        return np.array([_sigma[j] for j in self.basis['modes']])

    def _generate_spherical_basis(self,gsb_file):
        
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
        wavelet_n = wavelet_w.size

        # Save file
        save_kwargs = {'compression':"lzf", 'chunks':True, 'fletcher32':False, 'shuffle':True}
        with h5py.File(gsb_file, 'w') as f:
            f.create_dataset('wavelet_w', data = wavelet_w, dtype = np.float64, **save_kwargs)
            f.create_dataset('wavelet_v', data = wavelet_v, dtype = np.uint64, scaleoffset=0, **save_kwargs)
            f.create_dataset('wavelet_u', data = wavelet_u, dtype = np.uint64, scaleoffset=0, **save_kwargs)
            f.create_dataset('wavelet_n', data = wavelet_n)
            f.create_dataset('modes', data = wavelet_j, dtype = np.uint64, scaleoffset=0, **save_kwargs)
            