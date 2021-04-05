import numpy as np
from numba import njit, jit, prange, config

config.THREADING_LAYER = 'threadsafe'
parallel = False

eps=1e-10


@njit(parallel=parallel)
def wavelet_x_sparse(z, M, C, P, mu, sigma,
            wavelet_u, wavelet_v, wavelet_w,
            cholesky_u_m, cholesky_v_m, cholesky_w_m,
            cholesky_u_c, cholesky_v_c, cholesky_w_c,
            x, MC):

    #x = np.zeros((P, M, C))
    #MC = np.zeros((M, C))
    x *= 0.
    MC *= 0.

    # Iterate over pixels
    iY = 0
    for ipix in prange(P):

        # Iterate over modes which are not sparsified in Y
        iSmin = wavelet_u[ipix]
        iSmax = wavelet_u[ipix+1]
        for iS in wavelet_v[iSmin:iSmax]:
            MC *= 0.
            iYmag = 0
            for iM in prange(M):
                iMmin = cholesky_u_m[iM]
                iMmax = cholesky_u_m[iM+1]
                for iMsub in cholesky_v_m[iMmin:iMmax]:
                    iYcol = 0
                    for iC in prange(C):
                        iCmin = cholesky_u_c[iC]
                        iCmax = cholesky_u_c[iC+1]
                        for iCsub in cholesky_v_c[iCmin:iCmax]:
                            MC[iM, iC] += (cholesky_w_m[iYmag] * z[iS,iMsub,iCsub] * cholesky_w_c[iYcol]);

                            iYcol+=1
                    iYmag += 1

            x[ipix] += (mu[iS] + sigma[iS]*MC)* wavelet_w[iY]

            iY += 1

    return x

@njit
def wavelet_b_sparse(z, M, C, S, mu, sigma,
            wavelet_u, wavelet_v, wavelet_w,
            cholesky_u_m, cholesky_v_m, cholesky_w_m,
            cholesky_u_c, cholesky_v_c, cholesky_w_c,
            b, MC):

    #x = np.zeros((P, M, C))
    #MC = np.zeros((M, C))
    b *= 0.
    MC *= 0.

    # Iterate over modes which are not sparsified in Y
    for iS in range(S):
        MC *= 0.
        iYmag = 0
        for iM in range(M):
            iMmin = cholesky_u_m[iM]
            iMmax = cholesky_u_m[iM+1]
            for iMsub in cholesky_v_m[iMmin:iMmax]:
                iYcol = 0
                for iC in range(C):
                    iCmin = cholesky_u_c[iC]
                    iCmax = cholesky_u_c[iC+1]
                    for iCsub in cholesky_v_c[iCmin:iCmax]:
                        MC[iM, iC] += (cholesky_w_m[iYmag] * z[iS,iMsub,iCsub] * cholesky_w_c[iYcol]);

                        iYcol+=1
                iYmag += 1

        b[iS] = (mu[iS] + sigma[iS]*MC)

    return b

@njit(parallel=parallel)
def wavelet_magnitude_colour_position_sparse(z, M, C, P, k, n, mu, sigma,
                                                wavelet_u, wavelet_v, wavelet_w,
                                                cholesky_u_m, cholesky_v_m, cholesky_w_m,
                                                cholesky_u_c, cholesky_v_c, cholesky_w_c,
                                                x, lnL_grad, MC):

    # x = np.zeros((M, C))
    # lnL_grad = np.zeros((S, M_subspace, C_subspace))

    x = wavelet_x_sparse(z, M, C, P, mu, sigma,
                wavelet_u, wavelet_v, wavelet_w,
                cholesky_u_m, cholesky_v_m, cholesky_w_m,
                cholesky_u_c, cholesky_v_c, cholesky_w_c,
                x, MC)

    lnL = 0.
    lnL_grad *= 0.

    # Iterate over pixels
    iY = 0
    for ipix in prange(P):

        d = 1 + np.exp(-np.abs(x[ipix]))

        # Iterate over modes which are not sparsified in Y
        iSmin = wavelet_u[ipix]
        iSmax = wavelet_u[ipix+1]
        # Likelihood gradient db/dz * dx/db * dlnL/dx
        for iS in wavelet_v[iSmin:iSmax]:
            # Evaluate b from z at iS mode
            iYmag = 0
            for iM in prange(M):
                iMmin = cholesky_u_m[iM]
                iMmax = cholesky_u_m[iM+1]
                for iMsub in cholesky_v_m[iMmin:iMmax]:
                    iYcol = 0
                    for iC in prange(C):

                        iCmin = cholesky_u_c[iC]
                        iCmax = cholesky_u_c[iC+1]
                        for iCsub in cholesky_v_c[iCmin:iCmax]:

                            lnL_grad[iS,iMsub,iCsub] += sigma[iS] * cholesky_w_m[iYmag] * cholesky_w_c[iYcol] * wavelet_w[iY] \
                                          * (k[ipix,iM,iC] - n[ipix,iM,iC]*(0.5 + np.sign(x[ipix,iM,iC])*(0.5+(1-d[iM,iC])/d[iM,iC])) )

                            iYcol+=1
                    iYmag+=1


            iY += 1

        #lnL += np.sum( -k[ipix]*np.log(1+1/exp_x) - (n[ipix]-k[ipix])*np.log(1+exp_x) )
        #lnL += np.sum( k[ipix]*x - n[ipix]*np.log1p(exp_x) )
        #lnL += np.sum( k[ipix]*x - n[ipix]*(x/2 + np.log(2*np.cosh(x/2)) ) )
        lnL += np.sum( k[ipix]*x[ipix] - n[ipix]*(x[ipix]/2 + np.abs(x[ipix])/2 + np.log(d) ) )

    # Add on Gaussian prior
    return lnL - 0.5*np.sum(z**2), lnL_grad - z

@njit(parallel=False)
def wavelet_magnitude_colour_position_sparse3(z, M, C, P, k, n, mu, sigma,
                                                wavelet_u, wavelet_v, wavelet_w,
                                                cholesky_u_m, cholesky_v_m, cholesky_w_m,
                                                cholesky_u_c, cholesky_v_c, cholesky_w_c,
                                                x, lnL_grad, MC):

    # x = np.zeros((M, C))
    # lnL_grad = np.zeros((S, M_subspace, C_subspace))

    lnL = 0.
    lnL_grad *= 0.
    x *= 0.
    MC *= 0.

    # Iterate over pixels
    iY = 0
    for ipix in prange(P):

        x = wavelet_x_sparse(z, M, C, 1, mu, sigma,
                    wavelet_u[ipix:ipix+2], wavelet_v, wavelet_w,
                    cholesky_u_m, cholesky_v_m, cholesky_w_m,
                    cholesky_u_c, cholesky_v_c, cholesky_w_c,
                    x, MC)
        d = 1 + np.exp(-np.abs(x[0]))

        # Iterate over modes which are not sparsified in Y
        iSmin = wavelet_u[ipix]
        iSmax = wavelet_u[ipix+1]
        # Likelihood gradient db/dz * dx/db * dlnL/dx
        for iS in wavelet_v[iSmin:iSmax]:

            # Evaluate b from z at iS mode
            iYmag = 0
            for iM in range(M):
                iMmin = cholesky_u_m[iM]
                iMmax = cholesky_u_m[iM+1]
                for iMsub in cholesky_v_m[iMmin:iMmax]:
                    iYcol = 0
                    for iC in range(C):

                        iCmin = cholesky_u_c[iC]
                        iCmax = cholesky_u_c[iC+1]
                        for iCsub in cholesky_v_c[iCmin:iCmax]:

                            lnL_grad[iS,iMsub,iCsub] += sigma[iS] * cholesky_w_m[iYmag] * cholesky_w_c[iYcol] * wavelet_w[iY] \
                                          * (k[ipix,iM,iC] - n[ipix,iM,iC]*(0.5 + np.sign(x[0,iM,iC])*(0.5+(1-d[iM,iC])/d[iM,iC])) )

                            iYcol+=1
                    iYmag+=1
                #lnL += np.sum(k[ipix,iM]*x[iM] - n[ipix,iM]*(x[iM]/2 + np.abs(x[iM])/2 + np.log(d[iM]) ))
            iY += 1

        #lnL += np.sum( -k[ipix]*np.log(1+1/exp_x) - (n[ipix]-k[ipix])*np.log(1+exp_x) )
        #lnL += np.sum( k[ipix]*x - n[ipix]*np.log1p(exp_x) )
        #lnL += np.sum( k[ipix]*x - n[ipix]*(x/2 + np.log(2*np.cosh(x/2)) ) )
        lnL += np.sum( k[ipix]*x - n[ipix]*(x[0]/2 + np.abs(x[0])/2 + np.log(d) ) )

    # Add on Gaussian prior
    return lnL - 0.5*np.sum(z**2), lnL_grad - z

@njit(parallel=parallel)
def wavelet_magnitude_colour_position_sparse2(z, M, C, P, k, n, mu, sigma,
                                                wavelet_u, wavelet_v, wavelet_w,
                                                cholesky_u_m, cholesky_v_m, cholesky_w_m,
                                                cholesky_u_c, cholesky_v_c, cholesky_w_c,
                                                x, lnL_grad, MC):

    # x = np.zeros((M, C))
    # lnL_grad = np.zeros((S, M_subspace, C_subspace))

    lnL = 0.
    lnL_grad *= 0.
    x *= 0.
    MC *= 0.

    # Iterate over pixels
    iY = 0
    for ipix in prange(P):

        # Iterate over modes which are not sparsified in Y
        iSmin = wavelet_u[ipix]
        iSmax = wavelet_u[ipix+1]
        for iS in wavelet_v[iSmin:iSmax]:
            MC *= 0.
            iYmag = 0
            for iM in range(M):
                iMmin = cholesky_u_m[iM]
                iMmax = cholesky_u_m[iM+1]
                for iMsub in cholesky_v_m[iMmin:iMmax]:
                    iYcol = 0
                    for iC in range(C):
                        iCmin = cholesky_u_c[iC]
                        iCmax = cholesky_u_c[iC+1]
                        for iCsub in cholesky_v_c[iCmin:iCmax]:
                            MC[iM, iC] += (cholesky_w_m[iYmag] * z[iS,iMsub,iCsub] * cholesky_w_c[iYcol]);

                            iYcol+=1
                    iYmag += 1

            x += (mu[iS] + sigma[iS]*MC)* wavelet_w[iY]

        d = 1 + np.exp(-np.abs(x))
        for iS in wavelet_v[iSmin:iSmax]:

            # Evaluate b from z at iS mode
            iYmag = 0
            for iM in range(M):
                iMmin = cholesky_u_m[iM]
                iMmax = cholesky_u_m[iM+1]
                for iMsub in cholesky_v_m[iMmin:iMmax]:
                    iYcol = 0
                    for iC in range(C):

                        iCmin = cholesky_u_c[iC]
                        iCmax = cholesky_u_c[iC+1]
                        for iCsub in cholesky_v_c[iCmin:iCmax]:

                            lnL_grad[iS,iMsub,iCsub] += sigma[iS] * cholesky_w_m[iYmag] * cholesky_w_c[iYcol] * wavelet_w[iY] \
                                          * (k[ipix,iM,iC] - n[ipix,iM,iC]*(0.5 + np.sign(x[iM,iC])*(0.5+(1-d[iM,iC])/d[iM,iC])) )

                            iYcol+=1
                    iYmag+=1
                #lnL += np.sum(k[ipix,iM]*x[iM] - n[ipix,iM]*(x[iM]/2 + np.abs(x[iM])/2 + np.log(d[iM]) ))
            iY += 1

        #lnL += np.sum( -k[ipix]*np.log(1+1/exp_x) - (n[ipix]-k[ipix])*np.log(1+exp_x) )
        #lnL += np.sum( k[ipix]*x - n[ipix]*np.log1p(exp_x) )
        #lnL += np.sum( k[ipix]*x - n[ipix]*(x/2 + np.log(2*np.cosh(x/2)) ) )
        lnL += np.sum( k[ipix]*x - n[ipix]*(x/2 + np.abs(x)/2 + np.log(d) ) )

    # Add on Gaussian prior
    return lnL - 0.5*np.sum(z**2), lnL_grad - z

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
