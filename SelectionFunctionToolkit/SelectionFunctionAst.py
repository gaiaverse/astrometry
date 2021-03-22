import sys, h5py, numpy as np, scipy.stats, healpy as hp, tqdm

eps=1e-10

M = 17
C = 1
nside=32

box={};
with h5py.File('/data/asfe2/Projects/astrometry/gaia3_astcounts_arr_hpx128.h', 'r') as hf:
    box['n'] = hf['n'][...]
    box['k'] = hf['k'][...]
    M_bins = hf['magbins'][...]
print("Mag bins:", np.linspace(M_bins[0], M_bins[-1], M+1))

lengthscale = 0.3
lengthscale_m = lengthscale/(M_bins[1]-M_bins[0])
lengthscale_c = 1.


if True:
    # Scales linearly with (2^(2*jmax)), M,

    # jmax:3, nside:16,  M:17, tol-1e-4  - ~ ? Gb, ? s
    # jmax:3, nside:16,  M:17, tol-1e-2  - ~ ? Gb, ? s
    # jmax:3, nside:16,  M:85, tol-1e-4  - ~ ? Gb, ? s
    # jmax:3, nside:16,  M:85, tol-1e-2  - ~ ? Gb, ? s
    # jmax:4, nside:16,  M:17, tol-1e-2  - ~ ? Gb, ? s
    # jmax:3, nside:32,  M:17, tol-1e-2  - ~ ? Gb, ? s
    # jmax:3, nside:32,  M:85, tol-1e-4  - ~ ? Gb, ? s
    # jmax:4, nside:16,  M:85, tol-1e-4  - ~ ? Gb, ? s
    # jmax:5, nside:16,  M:17, tol-1e-4  - ~ ? Gb, ? s
    # jmax:5, nside:16,  M:85, tol-1e-4  - ~ ? Gb, ? s
    # jmax:5, nside:32,  M:85, tol-1e-4  - ~ ? Gb, ? s
    # jmax:5, nside:64,  M:85, tol-1e-4  - ~ ? Gb, ? s
    # jmax:6, nside:128, M:85, tol-1e-4  - ~ ? Gb, ? s
    # jmax:6, nside:32,  M:17, tol-1e-2  - ~ ? Gb, ? s

    jmax=4
    file_root = f"chisquare_jmax{jmax}_nside{nside}_M{M}_C{C}_l{lengthscale}"
    print(file_root)
    basis_options = {'needlet':'chisquare', 'j':jmax, 'B':2.0, 'p':1.0, 'wavelet_tol':1e-10}
    # Import chisel
    from SelectionFunctionChisel import Chisel
    chisel = Chisel(box['k'], box['n'],
                    basis_options,file_root,
                    axes = ['magnitude','colour','position'],
                    nest = True,
                    lengthscale_m = lengthscale_m,
                    lengthscale_c = lengthscale_c,
                    M = M,
                    C = C,
                    nside = nside,
                    sparse = True,
                    pivot = True,
                    mu = 0.0,
                    sigma = [-0.81489922, -2.55429039],
                    Mlim = [M_bins[0], M_bins[-1]],
                    Clim = [-100,100],
                    spherical_wavelets_directory='/data/asfe2/Projects/astrometry/SphericalWavelets/',
                    stan_output_directory='/data/asfe2/Projects/astrometry/StanOutput/'
                    )

    # Run hammer
    chisel.optimize(number_of_iterations = 10000)

    # Print convergence information
    chisel.print_convergence(number_of_lines = 10)

if False:
    # lmax:100, nside:32, M:0.5 --> ~2.5Gb, 54s
    # lmax:100, nside:64, M:0.5 - ~12Gb, 800s

    # lmax:100, nside:16, M:17 - ~?Gb, ?s
    # lmax:100, nside:16, M:85 - ~3.9Gb, 6167s - 1h55m
    # lmax:100, nside:32, M:85 - ~9.4Gb, 13933s - 4h
    # lmax:100, nside:64, M:85 - ~30.3Gb, 34212s - 10.5h
    # lmax:100, nside:128, M:85 - ~?Gb, ?s
    lmax = 100
    file_root = f"lmax{lmax}_nside{nside}_M{M}_C{C}_l{lengthscale}"
    print(file_root)
    # Import hammer
    from SelectionFunctionHammer import Hammer
    hammer = Hammer(k = box['k'],
                    n = box['n'],
                    axes = ['magnitude','colour','position'],
                    nest = True,
                    lmax = lmax,
                    lengthscale_m = lengthscale_m,
                    lengthscale_c = lengthscale_c,
                    M = M,
                    C = C,
                    nside = nside,
                    sparse = True,
                    pivot = True,
                    mu = 0.0,
                    sigma = [-0.81489922, -2.55429039],
                    Mlim = [M_bins[0], M_bins[-1]],
                    Clim = [-100,100],
                    file_root = file_root,
                    spherical_harmonics_directory='/data/asfe2/Projects/astrometry/SphericalHarmonics/',
                    stan_output_directory='/data/asfe2/Projects/astrometry/StanOutput/'
                    )

    # Run hammer
    hammer.optimize(number_of_iterations = 10000)

    # Print convergence information
    hammer.print_convergence(number_of_lines = 10)

if False:
    # Scales linearly with (2^(2*jmax)), M,

    # jmax:3, nside:16,  M:17, tol-1e-4  - ~1.7 Gb, 1654s
    # jmax:3, nside:16,  M:17, tol-1e-2  - ~0.75Gb, 946s
    # jmax:3, nside:16,  M:85, tol-1e-4  - ~8.3 Gb, 11464s
    # jmax:3, nside:16,  M:85, tol-1e-2  - ~3.6 Gb, 5656s
    # jmax:4, nside:16,  M:17, tol-1e-2  - ~1.1 Gb, 1434s
    # jmax:3, nside:32,  M:17, tol-1e-2  - ~2.8 Gb, 4263s
    # jmax:3, nside:32,  M:85, tol-1e-4  - ~32  Gb, ?s
    # jmax:4, nside:16,  M:85, tol-1e-4  - ~28.8Gb, ?s
    # jmax:5, nside:16,  M:17, tol-1e-4  - ~16  Gb, 27265s
    # jmax:5, nside:16,  M:85, tol-1e-4  - ~?   Gb, ?s
    # jmax:5, nside:32,  M:85, tol-1e-4  - ~?   Gb, ?s
    # jmax:5, nside:64,  M:85, tol-1e-4  - ~46.5Gb, ?s - CRASHED
    # jmax:6, nside:128, M:85, tol-1e-4  - ~?   Gb, ?s
    # jmax:6, nside:32,  M:17, tol-1e-2  - ~7.6 Gb, 15942s

    jmax=6
    file_root = f"jmax{jmax}_nside{nside}_M{M}_C{C}_l{lengthscale}"
    print(file_root)
    # Import chisel
    from SelectionFunctionChisel import Chisel
    chisel = Chisel(k = box['k'],
                    n = box['n'],
                    jmax = jmax,
                    axes = ['magnitude','colour','position'],
                    nest = True,
                    lengthscale_m = lengthscale_m,
                    lengthscale_c = lengthscale_c,
                    wavelet_tol = 1e-2,
                    M = M,
                    C = C,
                    nside = nside,
                    sparse = True,
                    pivot = True,
                    mu = 0.0,
                    sigma = [-0.81489922, -2.55429039],
                    Mlim = [M_bins[0], M_bins[-1]],
                    Clim = [-100,100],
                    file_root = file_root,
                    spherical_wavelets_directory='/data/asfe2/Projects/astrometry/SphericalWavelets/',
                    stan_output_directory='/data/asfe2/Projects/astrometry/StanOutput/'
                    )

    # Run hammer
    chisel.optimize(number_of_iterations = 10000)

    # Print convergence information
    chisel.print_convergence(number_of_lines = 10)
