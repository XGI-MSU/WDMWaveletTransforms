"""functions for computing the inverse wavelet transforms"""

import numpy as np
from numba import njit
from numpy.typing import NDArray

import WDMWaveletTransforms.fft_funcs as fft


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
    """Helper for unpacking results of frequency domain inverse transform.

    fft_prefactor2s is the length-Nt fft of the packed pixel coefficients; window
    tap delta contributes fft_prefactor2s[jj mod Nt] * phif[|delta|] at spectrum
    bin jj = m*Nt/2 + delta, over the open support delta in (-K/2, K/2).

    For interior bands, contributions at bins outside the rfft range [0, ND/2]
    are the mirror lobe of the real wavelet and fold back conjugated; the
    self-conjugate bins 0 and ND/2 receive both lobes at once (2 Re).  The
    self-conjugate bands m = 0 and m = Nf scatter only their own half-plane,
    which is already the complete synthesis for reflection-symmetric atoms.
    """
    ND = Nf * Nt
    K = mult_f * Nt

    assert 0 <= m <= Nf
    assert Nt % 2 == 0
    assert Nf % 2 == 0

    assert Nf > 0
    assert Nt > 0
    assert 0 < mult_f <= Nf

    f_size = int(ND // 2 + 1)
    half_K = int(K // 2)
    half_Nt = int(Nt // 2)
    half_ND = int(ND // 2)

    assert res.shape == (f_size,)
    assert fft_prefactor2s.shape == (Nt,)
    assert phif.shape == (half_K + 1,)

    i_midpoint = m * half_Nt

    if m in (0, Nf):
        # taps at delta = 0..K/2-1 for m = 0, delta = 1-K/2..0 for m = Nf; both
        # stay inside [0, ND/2] since mult_f <= Nf
        for i_ind in range(half_K):
            i = abs(i_midpoint - i_ind)
            ind3 = (2 * i) % Nt
            res[i] += fft_prefactor2s[ind3] * phif[i_ind]
    else:
        for i_ind in range(1, half_K):
            i1 = i_midpoint - i_ind
            val = fft_prefactor2s[i1 % Nt] * phif[i_ind]
            if i1 > 0:
                res[i1] += val
            elif i1 == 0:
                # self-conjugate bin: both lobes land here
                res[0] += 2.0 * val.real
            else:
                # mirror lobe of the real atom folds back conjugated
                res[-i1] += np.conj(val)

            i2 = i_midpoint + i_ind
            val = fft_prefactor2s[i2 % Nt] * phif[i_ind]
            if i2 < half_ND:
                res[i2] += val
            elif i2 == half_ND:
                res[half_ND] += 2.0 * val.real
            else:
                res[ND - i2] += np.conj(val)

        # center tap; i_midpoint is strictly inside (0, ND/2) for interior m
        res[i_midpoint] += fft_prefactor2s[i_midpoint % Nt] * phif[0]


@njit()
def pack_wave_inverse(
    m: int,
    Nt: int,
    Nf: int,
    prefactor2s: NDArray[np.complexfloating],
    wave_in: NDArray[np.floating],
) -> None:
    """Helper for fast frequency domain inverse transform to prepare for fourier transform"""
    if m == 0:
        for n in range(Nt):
            prefactor2s[n] = 1 / np.sqrt(2) * wave_in[(2 * n) % Nt, 0]
    elif m == Nf:
        for n in range(Nt):
            prefactor2s[n] = 1 / np.sqrt(2) * wave_in[(2 * n) % Nt + 1, 0]
    else:
        for n in range(Nt):
            val = float(wave_in[n, m])
            if (n + m) % 2:
                mult2 = -1j
            else:
                mult2 = 1

            prefactor2s[n] = mult2 * val


def inverse_wavelet_freq_helper_fast(
    wave_in: NDArray[np.floating],
    phif: NDArray[np.floating],
    Nf: int,
    Nt: int,
    mult_f: int,
) -> NDArray[np.complexfloating]:
    """Jit compatible loop for inverse_wavelet_freq"""
    ND = Nf * Nt

    prefactor2s = np.zeros(Nt, dtype=complex)
    res = np.zeros(ND // 2 + 1, dtype=complex)

    for m in range(Nf + 1):
        pack_wave_inverse(m, Nt, Nf, prefactor2s, wave_in)
        fft_prefactor2s = fft.fft(prefactor2s)
        unpack_wave_inverse(m, Nt, Nf, mult_f, phif, fft_prefactor2s, res)

    return res
