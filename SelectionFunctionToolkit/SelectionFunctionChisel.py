import numpy as np
import healpy as hp
import tqdm
import h5py
from SelectionFunctionBase import Base

class Chisel(Base):

    basis_keyword = 'wavelet'

    def _process_basis_options(self, needlet = 'littlewoodpaley', j=[0], B = 2.0, p = 1.0, wavelet_tol = 1e-10):

        if type(j) in [list,tuple,np.ndarray]:
            self.j = sorted([int(_j) for _j in j])
        else:
<<<<<<< HEAD
            self.j = [_j for _j in range(j+1)]

=======
            self.j = [_j for _j in range(-1,j+1)]
        
>>>>>>> 3c2a97447ae2b93f05902469d5b7f0abb29ec6c7
        self.needlet, self.B, self.p, self.wavelet_tol = needlet, B, p, wavelet_tol

        self.spherical_basis_file = f"{self.basis_keyword}_{self.needlet}_nside{self.nside}_B{self.B}_"+ (f"p{self.p}_" if self.needlet == 'chisquare' else '') + f"tol{self.wavelet_tol}_j[{','.join([str(_i) for _i in self.j])}].h5"

        assert self.B > 1.0
        assert self.wavelet_tol >= 0.0
        assert self.needlet in ['littlewoodpaley','chisquare']
        self.S = sum([self.order_to_npix(_j) if _j >= 0 else 1 for _j in self.j])

        if self.needlet == 'chisquare':
            from SelectionFunctionUtils import chisquare
            self.weighting = chisquare(self.j, p = self.p, B = self.B)
        else:
            from SelectionFunctionUtils import littlewoodpaley
            self.weighting = littlewoodpaley(B = self.B)


    def _process_sigma_basis_specific(self,sigma):
        assert len(sigma) == 2
        power_spectrum = lambda l: np.sqrt(np.exp(sigma[0])*np.power(1.0+l,sigma[1]))

        _sigma = np.zeros(self.S)
        running_index = 0
        for j in self.j:

            if j == -1:
                _sigma[running_index] = 1.0
                running_index += 1
                continue

            npix_needle = self.order_to_npix(j)

            start = self.weighting.start(j)
            end = self.weighting.end(j)
            modes = np.arange(start, end + 1, dtype = 'float')
            window = self.weighting.window_function(modes,j)**2*power_spectrum(modes)*(2.0*modes+1.0)/npix_needle

            _sigma[running_index:running_index+npix_needle] = np.sqrt(window.sum())
            running_index += npix_needle

        return _sigma

    def _generate_spherical_basis(self,gsb_file):

        # Import dependencies
        from numba import njit
        from math import sin, cos
        import sys

        nside = self.nside
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

        # Initialise variables
        running_index = 0
        needlet_w, needlet_v, needlet_u, needlet_j = [], [], [], []
        Y = np.zeros(npix)
        legendre = np.zeros((1+self.weighting.end(max(self.j)),npix))

        for j in self.j:

            print(f'Working on order {j}.')

            if j == -1:
                needlet_w.append(np.ones(npix))
                needlet_v.append(np.arange(npix))
                needlet_u.append(0)
                needlet_j.append(np.zeros(1))
                running_index += npix
                continue

            nside_needle = self.order_to_nside(j)
            npix_needle = self.nside_to_npix(nside_needle)

            start = self.weighting.start(j)
            end = self.weighting.end(j)
            modes = np.arange(start, end + 1, dtype = 'float')
            window = self.weighting.window_function(modes,j)*(2.0*modes+1.0)/np.sqrt(4.0*np.pi*npix_needle)

            for ipix_needle in tqdm.tqdm(range(npix_needle),file=sys.stdout):

                colat_needle, lon_needle = hp.pix2ang(nside=nside_needle,ipix=ipix_needle,lonlat=False)

                cos_gamma = cos(colat_needle) * cos_colat + sin(colat_needle) * sin_colat * (cos(lon_needle) * cos_lon + sin(lon_needle) * sin_lon)

                pixel_space(Y, cos_gamma = cos_gamma, window = window, start = start, end = end, legendre = legendre)

                _significant = np.where(np.abs(Y) > Y.max()*needle_sparse_tol)[0]
                needlet_w.append(Y[_significant])
                needlet_v.append(_significant)
                needlet_u.append(running_index)
                needlet_j.append(j*np.ones(self.order_to_npix(j)))
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
        wavelet_j = np.concatenate(needlet_j).astype(int)
        wavelet_n = wavelet_w.size

        # Save file
        save_kwargs = {'compression':"lzf", 'chunks':True, 'fletcher32':False, 'shuffle':True}
        with h5py.File(gsb_file, 'w') as f:
            f.create_dataset('wavelet_w', data = wavelet_w, dtype = np.float64, **save_kwargs)
            f.create_dataset('wavelet_v', data = wavelet_v, dtype = np.uint64, scaleoffset=0, **save_kwargs)
            f.create_dataset('wavelet_u', data = wavelet_u, dtype = np.uint64, scaleoffset=0, **save_kwargs)
            f.create_dataset('wavelet_n', data = wavelet_n)
            f.create_dataset('modes', data = wavelet_j, dtype = np.uint64, scaleoffset=0, **save_kwargs)
