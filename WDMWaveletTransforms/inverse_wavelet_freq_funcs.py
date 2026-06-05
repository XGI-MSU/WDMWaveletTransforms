"""functions for computing the inverse wavelet transforms"""

import numpy as np
from numba import njit
from numpy.typing import NDArray

import WDMWaveletTransforms.fft_funcs as fft

# @njit()
# def unpack_wave_inverse(
#    m: int,
#    Nt: int,
#    Nf: int,
#    mult_f: int,
#    phif: NDArray[np.floating],
#    fft_prefactor2s: NDArray[np.complexfloating],
#    res: NDArray[np.complexfloating],
# ) -> None:
#    """Helper for unpacking results of frequency domain inverse transform"""
#    ND = Nf * Nt
#    K = mult_f * Nt
#
#    assert 0 <= m <= Nf
#    assert Nt % 2 == 0
#    assert Nf % 2 == 0
#
#    assert Nf > 0
#    assert Nt > 0
#    assert mult_f > 0
#
#    f_size = int(ND//2 + 1)
#    half_K = int(K //2)
#    half_Nt = int(Nt // 2)
#
#    assert res.shape == (f_size,)
#    assert fft_prefactor2s.shape == (K,)
#    assert phif.shape == (half_K+1,)
#
#    i_midpoint = m * half_Nt
#
#    if m in (0, Nf):
#        for i_ind in range(half_K):
#            i = abs(i_midpoint - i_ind)
#            ind3 = (2 * i) % K
#            #for r in range(mult_f):
#            res[i] += fft_prefactor2s[ind3] * phif[i_ind]
#        if m == Nf:
#            i_ind = half_K
#            i = abs(i_midpoint - i_ind)
#            ind3 = 0 * mult_f
#            #for r in range(mult_f):
#            res[i] += fft_prefactor2s[ind3] * phif[i_ind]
#    else:
#        ind31 = i_midpoint % K
#        ind32 = i_midpoint % K
#        for i_ind in range(half_K):
#            i1 = i_midpoint - i_ind
#            i2 = i_midpoint + i_ind
#
#            if i1 >= 0:
#                #for r in range(mult_f):
#                res[i1] += fft_prefactor2s[ind31] * phif[i_ind]
#            if i2 < f_size:
#                #for r in range(mult_f):
#                res[i2] += fft_prefactor2s[ind32] * phif[i_ind]
#            ind31 -= 1
#            ind32 += 1
#            if ind31 < 0:
#                ind31 = K - 1
#            if ind32 == K:
#                ind32 = 0
#        #for r in range(mult_f):
#        res[i_midpoint] = fft_prefactor2s[(mult_f * i_midpoint) % K] * phif[0] #* np.sqrt(mult_f)


@njit()
def unpack_wave_inverse(
    m: int,
    Nt: int,
    Nf: int,
    mult_f: int,
    phif: NDArray[np.floating],
    fft_prefactor2s: NDArray[np.complexfloating],
    res: NDArray[np.complexfloating],
) -> None:
    """Helper for unpacking results of frequency domain inverse transform"""
    ND = Nf * Nt
    K = mult_f * Nt

    assert 0 <= m <= Nf
    assert Nt % 2 == 0
    assert Nf % 2 == 0

    assert Nf > 0
    assert Nt > 0
    assert mult_f > 0

    f_size = int(ND // 2 + 1)
    half_K = int(K // 2)
    half_Nt = int(Nt // 2)

    assert res.shape == (f_size,)
    assert fft_prefactor2s.shape == (K,)
    assert phif.shape == (half_K + 1,)

    i_midpoint = m * half_Nt

    if m in (0, Nf):
        for i_ind in range(half_K):
            i = abs(i_midpoint - i_ind)
            ind3 = (2 * i) % K
            # for r in range(mult_f):
            res[i] += fft_prefactor2s[ind3] * phif[i_ind]
        if m == Nf:
            i_ind = half_K
            i = abs(i_midpoint - i_ind)
            ind3 = 0 * mult_f
            # for r in range(mult_f):
            res[i] += fft_prefactor2s[ind3] * phif[i_ind]
    else:
        ind31 = (i_midpoint - 1) % K
        for i_ind in range(1, half_K):
            i1 = i_midpoint - i_ind

            if i1 >= 0:  # and i1%half_Nt >0:
                # for r in range(mult_f):
                res[i1] += fft_prefactor2s[ind31] * phif[i_ind]
            ind31 -= 1
            if ind31 < 0:
                ind31 = K - 1

        ind32 = (i_midpoint + 1) % K
        for i_ind in range(1, half_K):
            i2 = i_midpoint + i_ind
            if i2 < f_size:  # and i2%half_Nt >0:
                # for r in range(mult_f):
                res[i2] += fft_prefactor2s[ind32] * phif[i_ind]
            ind32 += 1
            if ind32 == K:
                ind32 = 0
        # for r in range(mult_f):
        # TODO why is only this one conjugated? maybe some other indices should be but the effect is smaller?
        if mult_f % 2 or m % 2 == 0:
            res[i_midpoint] += fft_prefactor2s[(mult_f * i_midpoint) % K] * phif[0]  # * np.sqrt(mult_f)
        else:
            res[i_midpoint] += -np.conjugate(fft_prefactor2s[(mult_f * i_midpoint) % K]) * phif[0]  # * np.sqrt(mult_f)


# @njit()
# def unpack_wave_inverse(m, Nt, Nf, phif, fft_prefactor2s, res):
#    """helper for unpacking results of frequency domain inverse transform"""
#    ND = Nt*Nf
#    i_min2 = min(max(Nt//2*(m-1), 0), ND//2+1)
#    i_max2 = min(max(Nt//2*(m+1), 0), ND//2+1)
#    for i in range(i_min2, i_max2):
#        i_ind = np.abs(i-Nt//2*m)
#        if i_ind > Nt//2:
#            continue
#        if m == 0:
#            res[i] += fft_prefactor2s[(2*i) % Nt]*phif[i_ind]
#        elif m == Nf:
#            res[i] += fft_prefactor2s[(2*i) % Nt]*phif[i_ind]
#        else:
#            res[i] += fft_prefactor2s[i % Nt]*phif[i_ind]


@njit()
def pack_wave_inverse(
    m: int,
    Nt: int,
    Nf: int,
    mult_f: int,
    prefactor2s: NDArray[np.complexfloating],
    wave_in: NDArray[np.floating],
) -> None:
    """Helper for fast frequency domain inverse transform to prepare for fourier transform"""
    if m == 0:
        for n in range(Nt):
            prefactor2s[n * mult_f] = 1 / np.sqrt(2) * wave_in[(2 * n) % Nt, 0]
    elif m == Nf:
        for n in range(Nt):
            prefactor2s[n * mult_f] = 1 / np.sqrt(2) * wave_in[(2 * n) % Nt + 1, 0]
    else:
        for n in range(Nt):
            val = float(wave_in[n, m])
            if (n + m) % 2:
                mult2 = -1j
            else:
                mult2 = 1

            prefactor2s[n * mult_f] = mult2 * val


# @njit()
def inverse_wavelet_freq_helper_fast(
    wave_in: NDArray[np.floating],
    phif: NDArray[np.floating],
    Nf: int,
    Nt: int,
    mult_f: int,
) -> NDArray[np.complexfloating]:
    """Jit compatible loop for inverse_wavelet_freq"""
    ND = Nf * Nt

    prefactor2s = np.zeros(Nt * mult_f, dtype=complex)
    res = np.zeros(ND // 2 + 1, dtype=complex)

    phif_alt = phif[: Nt // 2 + 1]
    prefactor2s_alt = np.zeros(Nt, dtype=complex)
    res_alt = np.zeros(ND // 2 + 1, dtype=complex)

    for m in range(Nf + 1):
        prefactor2s[:] = 0.0
        pack_wave_inverse(m, Nt, Nf, mult_f, prefactor2s, wave_in)
        fft_prefactor2s = fft.fft(prefactor2s)
        unpack_wave_inverse(m, Nt, Nf, mult_f, phif, fft_prefactor2s, res)

    return res
