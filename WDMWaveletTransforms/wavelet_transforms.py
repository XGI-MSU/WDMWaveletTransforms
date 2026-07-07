"""helper functions for transform_time.py"""

import numpy as np
from numpy.typing import NDArray

import WDMWaveletTransforms.fft_funcs as fft
import WDMWaveletTransforms.modified_gaussian as mg
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

WAVELET_FAMILIES = ('meyer', 'modified_gaussian')


def _get_phif(Nf: int, Nt: int, nx: float, mult_f: int, family: str) -> NDArray[np.float64]:
    """Normalized frequency-domain window for the requested wavelet family.

    nx (filter steepness) applies to the Meyer family only.
    """
    if family == 'meyer':
        return phitilde_vec_norm(Nf, Nt, nx, mult_f)
    if family == 'modified_gaussian':
        return np.asarray(mg.phitilde_vec_norm(Nf, Nt, mult_f), dtype=np.float64)
    msg = f'unrecognized wavelet family: {family!r}, expected one of {WAVELET_FAMILIES}'
    raise ValueError(msg)


def _get_phi(Nf: int, nx: float, mult: int, family: str) -> NDArray[np.float64]:
    """Time-domain window of length K = 2*mult*Nf for the requested wavelet family."""
    if family == 'meyer':
        return phi_vec(Nf, nx=nx, mult=mult)
    if family == 'modified_gaussian':
        return mg.phi_vec(Nf, mult_t=mult)
    msg = f'unrecognized wavelet family: {family!r}, expected one of {WAVELET_FAMILIES}'
    raise ValueError(msg)


def inverse_wavelet_time(
    wave_in: NDArray[np.float64],
    Nf: int,
    Nt: int,
    nx: float = 4.0,
    mult: int = 32,
    family: str = 'meyer',
) -> NDArray[np.float64]:
    """Fast inverse wavelet transform to time domain"""
    assert len(wave_in.shape) == 2, 'Only 2D Arrays supported currently'
    mult = int(min(mult, int(Nt // 2)))  # make sure K isn't bigger than ND
    phi: NDArray[np.float64] = _get_phi(Nf, nx, mult, family) / 2

    return inverse_wavelet_time_helper_fast(wave_in, phi, Nf, Nt, mult)


def inverse_wavelet_freq(
    wave_in: NDArray[np.float64],
    Nf: int,
    Nt: int,
    nx: float = 4.0,
    mult_f: int = 1,
    family: str = 'meyer',
) -> NDArray[np.complex128]:
    """Inverse wavelet transform to freq domain signal"""
    assert len(wave_in.shape) == 2, 'Only 2D Arrays supported currently'
    phif: NDArray[np.float64] = _get_phif(Nf, Nt, nx, mult_f, family)
    return inverse_wavelet_freq_helper_fast(wave_in, phif, Nf, Nt, mult_f)


def inverse_wavelet_freq_time(
    wave_in: NDArray[np.float64],
    Nf: int,
    Nt: int,
    nx: float = 4.0,
    mult_f: int = 1,
    family: str = 'meyer',
) -> NDArray[np.float64]:
    """Inverse wavlet transform to time domain via fourier transform of frequency domain"""
    assert len(wave_in.shape) == 2, 'Only 2D Arrays supported currently'
    res_f: NDArray[np.complex128] = inverse_wavelet_freq(wave_in, Nf, Nt, nx, mult_f, family)
    return fft.irfft(res_f)


def transform_wavelet_time(
    data: NDArray[np.float64],
    Nf: int,
    Nt: int,
    nx: float = 4.0,
    mult: int = 32,
    family: str = 'meyer',
) -> NDArray[np.float64]:
    """Do the wavelet transform in the time domain,
    note there can be significant leakage if mult is too small and the
    transform is only approximately exact if mult=Nt/2
    """
    assert len(data.shape) == 1, 'Only 1D Arrays supported currently'
    mult = int(min(mult, int(Nt // 2)))  # make sure K isn't bigger than ND
    phi: NDArray[np.float64] = _get_phi(Nf, nx, mult, family)
    return transform_wavelet_time_helper(data, Nf, Nt, phi, mult)


def transform_wavelet_freq(
    data: NDArray[np.complex128],
    Nf: int,
    Nt: int,
    nx: float = 4.0,
    mult_f: int = 1,
    family: str = 'meyer',
) -> NDArray[np.float64]:
    """Do the wavelet transform using the fast wavelet domain transform"""
    assert len(data.shape) == 1, 'Only 1D Arrays supported currently'
    phif: NDArray[np.float64] = 2 / Nf * _get_phif(Nf, Nt, nx, mult_f, family)
    return transform_wavelet_freq_helper(data, Nf, Nt, mult_f, phif)


def transform_wavelet_freq_time(
    data: NDArray[np.float64],
    Nf: int,
    Nt: int,
    nx: float = 4.0,
    mult_f: int = 1,
    family: str = 'meyer',
) -> NDArray[np.float64]:
    """Transform time domain data into wavelet domain via fft and then frequency transform"""
    assert len(data.shape) == 1, 'Only 1D Arrays supported currently'
    data_fft: NDArray[np.complex128] = fft.rfft(data)

    return transform_wavelet_freq(data_fft, Nf, Nt, nx=nx, mult_f=mult_f, family=family)
