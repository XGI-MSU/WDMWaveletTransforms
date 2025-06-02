"""helper functions for transform_time.py"""

import numpy as np
from numpy.typing import NDArray

import WDMWaveletTransforms.fft_funcs as fft
from WDMWaveletTransforms.inverse_wavelet_freq_funcs import inverse_wavelet_freq_helper_fast
from WDMWaveletTransforms.inverse_wavelet_time_funcs import inverse_wavelet_time_helper_fast
from WDMWaveletTransforms.transform_freq_funcs import phitilde_vec_norm, transform_wavelet_freq_helper
from WDMWaveletTransforms.transform_time_funcs import phi_vec, transform_wavelet_time_helper

__all__ = [
    'inverse_wavelet_freq',
    'inverse_wavelet_freq_time',
    'inverse_wavelet_time',
    'transform_wavelet_freq',
    'transform_wavelet_freq_time',
    'transform_wavelet_time',
]



def inverse_wavelet_time(wave_in: NDArray[np.floating], Nf: int, Nt: int, nx: float=4., mult: int=32) -> NDArray[np.floating]:
    """Fast inverse wavelet transform to time domain"""
    mult = int(min(mult, int(Nt//2)))  # make sure K isn't bigger than ND
    phi: NDArray[np.floating] = phi_vec(Nf, nx=nx, mult=mult)/2

    return inverse_wavelet_time_helper_fast(wave_in, phi, Nf, Nt, mult)


def inverse_wavelet_freq(wave_in: NDArray[np.floating], Nf: int, Nt: int, nx: float=4.) -> NDArray[np.complexfloating]:
    """Inverse wavelet transform to freq domain signal"""
    phif: NDArray[np.floating] = phitilde_vec_norm(Nf, Nt, nx)
    return inverse_wavelet_freq_helper_fast(wave_in, phif, Nf, Nt)


def inverse_wavelet_freq_time(wave_in: NDArray[np.floating], Nf: int, Nt: int, nx: float=4.) -> NDArray[np.floating]:
    """Inverse wavlet transform to time domain via fourier transform of frequency domain"""
    res_f: NDArray[np.complexfloating] = inverse_wavelet_freq(wave_in, Nf, Nt, nx)
    return fft.irfft(res_f)


def transform_wavelet_time(data: NDArray[np.floating], Nf: int, Nt: int, nx: float=4., mult: int=32) -> NDArray[np.floating]:
    """Do the wavelet transform in the time domain,
    note there can be significant leakage if mult is too small and the
    transform is only approximately exact if mult=Nt/2
    """
    mult = int(min(mult, int(Nt//2)))  # make sure K isn't bigger than ND
    phi: NDArray[np.floating] = phi_vec(Nf, nx, mult)
    return transform_wavelet_time_helper(data, Nf, Nt, phi, mult)


def transform_wavelet_freq(data: NDArray[np.complexfloating], Nf: int, Nt: int, nx: float=4.) -> NDArray[np.floating]:
    """Do the wavelet transform using the fast wavelet domain transform"""
    phif: NDArray[np.floating] = 2/Nf*phitilde_vec_norm(Nf, Nt, nx)
    return transform_wavelet_freq_helper(data, Nf, Nt, phif)


def transform_wavelet_freq_time(data: NDArray[np.floating], Nf: int, Nt: int, nx: float=4.) -> NDArray[np.floating]:
    """Transform time domain data into wavelet domain via fft and then frequency transform"""
    data_fft: NDArray[np.complexfloating] = fft.rfft(data)

    return transform_wavelet_freq(data_fft, Nf, Nt, nx)
