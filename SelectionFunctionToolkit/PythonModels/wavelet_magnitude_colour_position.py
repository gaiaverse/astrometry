import numpy as np
from numba import njit

eps=1e-10


#@njit
def get_wavelet_x(b, P, M, C, wavelet_u, wavelet_v, wavelet_w):

    x = np.zeros((P,M,C))
    print(x.shape, b.shape, wavelet_w.shape)

    # Iterate over pixels
    for ipix in range(P):

        # Iterate over modes which are not sparsified in Y
        imin = wavelet_u[ipix]
        imax = wavelet_u[ipix+1]
        iY = 0
        for iS in wavelet_v[imin:imax]:

            # Evaluate x from b
            x[ipix] += b[iS] * wavelet_w[int(wavelet_u[ipix]+iY)]

            iY += 1

    return x

@njit
def get_wavelet_x1(x, z, M, C, P, k, n, mu, sigma,
                wavelet_u, wavelet_v, wavelet_w,
                cholesky_m, cholesky_c):

    # Iterate over pixels
    iY = 0
    for ipix in range(P):
        # Iterate over modes which are not sparsified in Y
        iSmin = wavelet_u[ipix]
        iSmax = wavelet_u[ipix+1]
        for iS in wavelet_v[iSmin:iSmax]:

            # Evaluate b from z at iS mode
            b = mu[iS] + sigma[iS] * (cholesky_m @ z[iS] @ cholesky_c.T);

            # Evaluate x from b
            x[ipix] += b * wavelet_w[iY]

            iY += 1

    return x

@njit
def get_wavelet_x2(x, z, M, C, P, k, n, mu, sigma,
                wavelet_u, wavelet_v, wavelet_w,
                cholesky_u_m, cholesky_v_m, cholesky_w_m,
                cholesky_u_c, cholesky_v_c, cholesky_w_c):

    b = np.zeros((M, C))

    # Iterate over pixels
    iY = 0
    for ipix in range(P):
        # Iterate over modes which are not sparsified in Y
        iSmin = wavelet_u[ipix]
        iSmax = wavelet_u[ipix+1]
        for iS in wavelet_v[iSmin:iSmax]:
            # Evaluate b from z at iS mode
            b *= 0.
            iYmag = 0
            for imag in range(M):
                iMmin = cholesky_u_m[imag]
                iMmax = cholesky_u_m[imag+1]
                for iM in cholesky_v_m[iMmin:iMmax]:
                    iYcol = 0
                    for icol in range(C):
                        iCmin = cholesky_u_c[icol]
                        iCmax = cholesky_u_c[icol+1]
                        for iC in cholesky_v_c[iCmin:iCmax]:
                            b[imag, iC] += (cholesky_w_m[iYmag] *\
                                          z[iS,iM,icol] *\
                                          cholesky_w_c[iYcol]);

                            iYcol+=1
                    iYmag += 1
            # Evaluate x from b
            x[ipix] += (mu[iS] + sigma[iS] * b) * wavelet_w[iY]
            iY += 1
    return x

#@njit
def wavelet_magnitude_colour_position(z, M, C, P, k, n, mu, sigma,
                                     wavelet_u, wavelet_v, wavelet_w, cholesky_m, cholesky_c,
                                     lnL_grad, x):

    lnL = 0.
    lnL_grad *= 0.
    lnL_grad_local = np.zeros(lnL_grad.shape)

    b = np.zeros(z.shape[1:])

    # Iterate over pixels
    for ipix in range(P):
        x *= 0.
        lnL_grad_local *= 0.

        # Iterate over modes which are not sparsified in Y
        imin = wavelet_u[ipix]
        imax = wavelet_u[ipix+1]
        iY = 0
        for iS in wavelet_v[imin:imax]:

            # Evaluate b from z at iS mode
            b = mu[iS] + sigma[iS] * (cholesky_m @ z[iS] @ cholesky_c.T);

            # Evaluate x from b
            x += b * wavelet_w[int(wavelet_u[ipix]+iY)]
            #x += b * wavelet_w[wavelet_u[ipix]+iY]

            # Likelihood gradient - dx/db
            lnL_grad_local[iS] += wavelet_w[int(wavelet_u[ipix]+iY)]

            iY += 1

        exp_x = np.exp(x)

        # Likelihood gradient db/dz * dx/db * dlnL/dx
        for iS in wavelet_v[imin:imax]:
            lnL_grad[iS] += \
            sigma[iS] * cholesky_m.T @ ( lnL_grad_local[iS] * (k[ipix] - n[ipix]/(1+1/exp_x)) ) @ cholesky_c
            #lnL_grad[iS] += lnL_grad_local[iS] * (k[ipix] - n[ipix]/(1+1/exp_x))

        #lnL += np.sum( -k[ipix]*np.log(1+1/exp_x) - (n[ipix]-k[ipix])*np.log(1+exp_x) )
        lnL += np.sum( k[ipix]*x - n[ipix]*np.log1p(exp_x) )

        #(k*x - (n-k))*(x/2)*log(e^x/2 + e^-x/2)

    # Add on Gaussian prior
    return lnL + np.sum(-0.5*z**2), lnL_grad - z

@njit
def wavelet_magnitude_colour_position_sparse(z, M, C, P, k, n, mu, sigma,
                                                wavelet_u, wavelet_v, wavelet_w,
                                                cholesky_u_m, cholesky_v_m, cholesky_w_m,
                                                cholesky_u_c, cholesky_v_c, cholesky_w_c):
                                                #lnL_grad, x):

    lnL = 0.
    #lnL_grad *= 0.
    x = np.zeros((M, C))
    lnL_grad = np.zeros(z.shape)
    lnL_grad_local = np.zeros(z.shape)

    # Iterate over pixels
    iY = 0
    for ipix in range(P):
        x *= 0.
        lnL_grad_local *= 0.

        # Iterate over modes which are not sparsified in Y
        iSmin = wavelet_u[ipix]
        iSmax = wavelet_u[ipix+1]
        for iS in wavelet_v[iSmin:iSmax]:

            # Evaluate b from z at iS mode
            iYmag = 0
            for imag in range(M):
                iMmin = cholesky_u_m[imag]
                iMmax = cholesky_u_m[imag+1]
                for iM in cholesky_v_m[iMmin:iMmax]:
                    iYcol = 0
                    for icol in range(C):
                        iCmin = cholesky_u_c[icol]
                        iCmax = cholesky_u_c[icol+1]
                        for iC in cholesky_v_c[iCmin:iCmax]:
                            b = mu[iS] + sigma[iS] * (cholesky_w_m[iYmag] *\
                                                      z[iS,iM,icol] *\
                                                      cholesky_w_c[iYcol]);

                            # Evaluate x from b
                            x[imag,iC] += b * wavelet_w[iY]

                            iYcol+=1
                    iYmag += 1

            # Likelihood gradient - dx/db
            lnL_grad_local[iS] += wavelet_w[iY]
            iY += 1

        exp_x = np.exp(x)
        d = 1 + np.exp(-np.abs(x))

        # Likelihood gradient db/dz * dx/db * dlnL/dx
        for iS in wavelet_v[iSmin:iSmax]:
            # Evaluate b from z at iS mode
            iYmag = 0
            for imag in range(M):
                iMmin = cholesky_u_m[imag]
                iMmax = cholesky_u_m[imag+1]
                for iM in cholesky_v_m[iMmin:iMmax]:
                    iYcol = 0
                    for icol in range(C):
                        iCmin = cholesky_u_c[icol]
                        iCmax = cholesky_u_c[icol+1]
                        for iC in cholesky_v_c[iCmin:iCmax]:

                            # lnL_grad[iS,iM,icol] += sigma[iS] * cholesky_w_m[iYmag] * \
                            #                                     ( lnL_grad_local[iS,imag,iC] * (k[ipix,imag,iC] - n[ipix,imag,iC]/(1+1/exp_x[imag,iC])) ) * \
                            #                                     cholesky_w_c[iYcol]
                            lnL_grad[iS,iM,icol] += sigma[iS] * cholesky_w_m[iYmag] * \
                                        ( lnL_grad_local[iS,imag,iC] * (k[ipix,imag,iC] - n[ipix,imag,iC]*(0.5 + np.sign(x[imag,iC])*(0.5+(1-d[imag,iC])/d[imag,iC])) ) ) * \
                                                                cholesky_w_c[iYcol]
                            iYcol+=1
                    iYmag+=1

        #lnL += np.sum( -k[ipix]*np.log(1+1/exp_x) - (n[ipix]-k[ipix])*np.log(1+exp_x) )
        #lnL += np.sum( k[ipix]*x - n[ipix]*np.log1p(exp_x) )
        lnL += np.sum( k[ipix]*x - n[ipix]*(x/2 + np.abs(x)/2 + np.log(d) ) )
        #lnL += np.sum( k[ipix]*x - n[ipix]*(x/2 + np.log(2*np.cosh(x/2)) ) )

    # Add on Gaussian prior
    return lnL - 0.5*np.sum(z**2), lnL_grad - z

# @njit
# def wavelet_magnitude_colour_position_sparse(z, M, C, P, k, n, mu, sigma,
#                                                 wavelet_u, wavelet_v, wavelet_w,
#                                                 cholesky_u_m, cholesky_v_m, cholesky_w_m,
#                                                 cholesky_u_c, cholesky_v_c, cholesky_w_c,
#                                                 lnL_grad, x):
#
#     lnL = 0.
#     lnL_grad *= 0.
#     lnL_grad_local = np.zeros(lnL_grad.shape)
#
#     # Iterate over magnitudes
#     iYmag = -1
#     for imag in range(M):
#         iMmin = cholesky_u_m[imag]
#         iMmax = cholesky_u_m[imag+1]
#         for iM in cholesky_v_m[iMmin:iMmax]:
#             iYmag += 1
#
#             # Iterate over colours
#             iYcol = -1
#             for icol in range(C):
#                 iCmin = cholesky_u_c[icol]
#                 iCmax = cholesky_u_c[icol+1]
#                 for iC in cholesky_v_c[iCmin:iCmax]:
#                     iYcol+=1
#
#                     # Iterate over pixels
#                     iY = -1
#                     for ipix in range(P):
#                         x = 0.
#                         lnL_grad_local *= 0.
#
#                         # Iterate over modes which are not sparsified in Y
#                         iSmin = wavelet_u[ipix]
#                         iSmax = wavelet_u[ipix+1]
#                         for iS in wavelet_v[iSmin:iSmax]:
#                             iY += 1
#                             b = mu[iS] + sigma[iS] * (cholesky_w_m[iYmag] *\
#                                                       z[iS,iM,icol] *\
#                                                       cholesky_w_c[iYcol]);
#
#                             # Evaluate x from b
#                             x += b * wavelet_w[iY]
#
#                             # Likelihood gradient - dx/db
#                             lnL_grad_local[iS] += wavelet_w[iY]
#
#                         exp_x = np.exp(x)
#
#                         # Likelihood gradient db/dz * dx/db * dlnL/dx
#                         for iS in wavelet_v[iSmin:iSmax]:
#
#                             lnL_grad[iS,iM,icol] += sigma[iS] * cholesky_w_m[iYmag] * \
#                                                                 ( lnL_grad_local[iS,imag,iC] * (k[ipix,imag,iC] - n[ipix,imag,iC]/(1+1/exp_x)) ) * \
#                                                                 cholesky_w_c[iYcol]
#
#                         #lnL += np.sum( -k[ipix]*np.log(1+1/exp_x) - (n[ipix]-k[ipix])*np.log(1+exp_x) )
#                         lnL += k[ipix,imag,iC]*x - n[ipix,imag,iC]*np.log1p(exp_x)
#                         #lnL += np.sum( k[ipix]*x - n[ipix]*(x/2 + np.log(2*np.cosh(x/2)) ) )
#
#     # Add on Gaussian prior
#     return lnL - 0.5*np.sum(z**2), lnL_grad - z
