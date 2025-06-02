"""functions for computing the inverse wavelet transforms"""
import numpy as np
from numba import njit
from numpy.typing import NDArray

import WDMWaveletTransforms.fft_funcs as fft


@njit()
def unpack_time_wave_helper_compact(n: int, Nf: int, Nt: int, K: int, phis: NDArray[np.floating], fft_fin: NDArray[np.complexfloating], res: NDArray[np.floating]) -> None:
    """Helper for time domain wavelet transform to unpack wavelet domain coefficients
    in compact representation where cosine and sine parts are real and imaginary parts
    """
    ND = Nf*Nt
    fft_fin_real = np.zeros(4*Nf)
    fft_fin_imag = np.zeros(4*Nf)
    for itrf in range(2*Nf):
        fft_fin_real[itrf] = np.real(fft_fin[itrf])
        fft_fin_real[itrf+2*Nf] = fft_fin_real[itrf]
        fft_fin_imag[itrf] = np.imag(fft_fin[(itrf+Nf) % (2*Nf)])
        fft_fin_imag[itrf+2*Nf] = fft_fin_imag[itrf]

    idxf1_base = (-K//2+n*Nf+ND) % (2*Nf)
    k1_base = (-K//2+n*Nf) % ND
    for k_ind in range(0, K, 2*Nf):
        for idxf1_add in range(2*Nf):
            idxf1 = idxf1_base+idxf1_add  # k_ind % (2*Nf)
            k_ind_loc = k_ind+idxf1_add
            k1 = k1_base+k_ind_loc

            res[k1] += phis[k_ind_loc]*fft_fin_real[idxf1]
            res[k1+Nf] += phis[k_ind_loc]*fft_fin_imag[idxf1]


@njit()
def pack_wave_time_helper_compact(n: int, Nf: int, Nt: int, wave_in: NDArray[np.floating], afins: NDArray[np.complexfloating]) -> None:
    """Helper for time domain transform to pack wavelet domain coefficients
    in packed representation with odd and even coefficients in real and imaginary pars
    """
    afins[0] = np.sqrt(2)*wave_in[n, 0]
    if n+1 < Nt:
        afins[Nf] = np.sqrt(2)*wave_in[n+1, 0]

    for idxm in range(0, Nf-2, 2):
        afins[idxm+2] = wave_in[n, idxm+2]-wave_in[n+1, idxm+2]
        afins[2*Nf-idxm-2] = wave_in[n, idxm+2]+wave_in[n+1, idxm+2]

        afins[idxm+1] = 1j*(wave_in[n, idxm+1]-wave_in[n+1, idxm+1])
        afins[2*Nf-idxm-1] = -1j*(wave_in[n, idxm+1]+wave_in[n+1, idxm+1])

    afins[Nf-1] = 1j*(wave_in[n, Nf-1]-wave_in[n+1, Nf-1])
    afins[Nf+1] = -1j*(wave_in[n, Nf-1]+wave_in[n+1, Nf-1])


def inverse_wavelet_time_helper_fast(wave_in: NDArray[np.floating], phi: NDArray[np.floating], Nf: int, Nt: int, mult: int) -> NDArray[np.floating]:
    """Helper loop for fast inverse wavelet transform"""
    ND = Nf*Nt
    K = mult*2*Nf
    # res = np.zeros(ND)

    # extend this array, we can use wrapping boundary conditions at end
    res: NDArray[np.float64] = np.zeros(ND+K+Nf, dtype=np.float64)

    afins = np.zeros(2*Nf, dtype=np.complex128)

    for n in range(Nt):
        # we can pack both the sin and cos parts into the real and imaginary parts
        # of the same transform so we only need to do every other one
        # this currently assumes Nt is even
        if n % 2 == 0:
            pack_wave_time_helper_compact(n, Nf, Nt, wave_in, afins)
            ffts_fin = fft.fft(afins)
            unpack_time_wave_helper_compact(n, Nf, Nt, K, phi, ffts_fin, res)

    # wrap boundary conditions
    res[:min(K+Nf, ND)] += res[ND:min(ND+K+Nf, 2*ND)]
    if K+Nf > ND:
        res[:K+Nf-ND] += res[2*ND:ND+K*Nf]

    return np.asarray(res[:ND], dtype=np.float64)
