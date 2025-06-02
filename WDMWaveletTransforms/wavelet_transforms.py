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



def inverse_wavelet_time(wave_in: NDArray[np.float64], Nf: int, Nt: int, nx: float=4., mult: int=32) -> NDArray[np.float64]:
    """Fast inverse wavelet transform to time domain"""
    assert len(wave_in.shape) == 2, 'Only 2D Arrays supported currently'
    mult = int(min(mult, int(Nt//2)))  # make sure K isn't bigger than ND
    phi: NDArray[np.float64] = phi_vec(Nf, nx=nx, mult=mult)/2

    return inverse_wavelet_time_helper_fast(wave_in, phi, Nf, Nt, mult)


def inverse_wavelet_freq(wave_in: NDArray[np.float64], Nf: int, Nt: int, nx: float=4.) -> NDArray[np.complex128]:
    """Inverse wavelet transform to freq domain signal"""
    assert len(wave_in.shape) == 2, 'Only 2D Arrays supported currently'
    phif: NDArray[np.float64] = phitilde_vec_norm(Nf, Nt, nx)
    return inverse_wavelet_freq_helper_fast(wave_in, phif, Nf, Nt)


def inverse_wavelet_freq_time(wave_in: NDArray[np.float64], Nf: int, Nt: int, nx: float=4.) -> NDArray[np.float64]:
    """Inverse wavlet transform to time domain via fourier transform of frequency domain"""
    assert len(wave_in.shape) == 2, 'Only 2D Arrays supported currently'
    res_f: NDArray[np.complex128] = inverse_wavelet_freq(wave_in, Nf, Nt, nx)
    return fft.irfft(res_f)


def transform_wavelet_time(data: NDArray[np.float64], Nf: int, Nt: int, nx: float=4., mult: int=32) -> NDArray[np.float64]:
    """Do the wavelet transform in the time domain,
    note there can be significant leakage if mult is too small and the
    transform is only approximately exact if mult=Nt/2
    """
    assert len(data.shape) == 1, 'Only 1D Arrays supported currently'
    mult = int(min(mult, int(Nt//2)))  # make sure K isn't bigger than ND
    phi: NDArray[np.float64] = phi_vec(Nf, nx, mult)
    return transform_wavelet_time_helper(data, Nf, Nt, phi, mult)


def transform_wavelet_freq(data: NDArray[np.complex128], Nf: int, Nt: int, nx: float=4.) -> NDArray[np.float64]:
    """Do the wavelet transform using the fast wavelet domain transform"""
    assert len(data.shape) == 1, 'Only 1D Arrays supported currently'
    phif: NDArray[np.float64] = 2/Nf*phitilde_vec_norm(Nf, Nt, nx)
    return transform_wavelet_freq_helper(data, Nf, Nt, phif)


def transform_wavelet_freq_time(data: NDArray[np.float64], Nf: int, Nt: int, nx: float=4.) -> NDArray[np.float64]:
    """Transform time domain data into wavelet domain via fft and then frequency transform"""
    assert len(data.shape) == 1, 'Only 1D Arrays supported currently'
    data_fft: NDArray[np.complex128] = fft.rfft(data)

    return transform_wavelet_freq(data_fft, Nf, Nt, nx)
