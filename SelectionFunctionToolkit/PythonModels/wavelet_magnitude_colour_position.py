import numpy as np
from numba import njit

eps=1e-10

@njit
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
            b = mu[iS] + sigma[iS] * (cholesky_m @ z[iS] @ cholesky_c);
            #b = z[iS]

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
            sigma[iS] * cholesky_m.T @ ( lnL_grad_local[iS] * (k[ipix] - n[ipix]/(1+1/exp_x)) ) @ cholesky_c.T
            #lnL_grad[iS] += lnL_grad_local[iS] * (k[ipix] - n[ipix]/(1+1/exp_x))

        lnL += np.sum( -k[ipix]*np.log(1+1/exp_x) - (n[ipix]-k[ipix])*np.log(1+exp_x) )

    # Add on Gaussian prior
    return lnL + np.sum(-0.5*z**2), lnL_grad - z

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
def wavelet_magnitude_colour_position_sparse(z, M, C, P, k, n, mu, sigma,
                                     wavelet_u, wavelet_v, wavelet_w,
                                     cholesky_u_m, cholesky_v_m, cholesky_w_m,
                                     cholesky_u_c, cholesky_v_c, cholesky_w_c,
                                     lnL_grad, x):

    lnL = 0.
    lnL_grad *= 0.
    lnL_grad_local = np.zeros(lnL_grad.shape)

    #lnprior = np.sum(-0.5*z**2)
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
            b *= 0.
            for m in range(M):
                for c in range(C):
                    for ms in range(cholesky_u_m[m],cholesky_u_m[m+1]):
                        for cs in range(cholesky_u_c[c],cholesky_u_c[c+1]):
                            #print(ms, cs, b.shape)
                            b[cholesky_v_m[ms], \
                              cholesky_v_c[cs]] += \
                                    mu[iS] + sigma[iS] * (cholesky_w_m[ms] *\
                                                            z[iS,cholesky_v_m[ms], \
                                                                 cholesky_v_c[cs]] *\
                                                            cholesky_w_c[cs]);

            # Evaluate x from b
            #x += b * wavelet_w[int(wavelet_u[ipix]+iY)]
            x += b * wavelet_w[wavelet_u[ipix]+iY]

            # Likelihood gradient - dx/db
            #lnL_grad_local[iS] += wavelet_w[int(wavelet_u[ipix]+iY+eps)]
            lnL_grad_local[iS] += wavelet_w[wavelet_u[ipix]+iY]

            iY += 1

        exp_x = np.exp(x)

        # Likelihood gradient db/dz * dx/db * dlnL/dx
        for m in range(M):
            for c in range(C):
                for ms in range(cholesky_u_m[m],cholesky_u_m[m+1]):
                    for cs in range(cholesky_u_c[c],cholesky_u_c[c+1]):
                        lnL_grad[:,cholesky_v_m[ms], \
                                   cholesky_v_c[cs]] += \
                                    sigma * cholesky_w_m[ms] * cholesky_w_c[cs] * \
                                    lnL_grad_local[:,m,c] * (k[ipix,m,c] - n[ipix,m,c]/(1+1/exp_x[m,c]))

        lnL += np.sum( -k[ipix]*np.log(1+1/exp_x) - (n[ipix]-k[ipix])*np.log(1+exp_x) )

    return lnL, lnL_grad
