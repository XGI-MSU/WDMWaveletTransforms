from typing import Optional, Tuple, Union

import bilby
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


def compute_wavelet_snr(h: "Wavelet", PSD: "Wavelet") -> float:
    """Compute the SNR of a model h[ti,fi] given data d[ti,fi] and PSD[ti,fi].

    SNR(h) = Sum_{ti,fi} [ h_hat[ti,fi] d[ti,fi] / PSD[ti,fi]

    Parameters
    ----------
    h : np.ndarray
        The model in the wavelet domain (binned in [ti,fi]).
    d : np.ndarray
        The data in the wavelet domain (binned in [ti,fi]).
    PSD : np.ndarray
        The PSD in the wavelet domain (binned in [ti,fi]).

    Returns
    -------
    float
        The SNR of the model h given data d and PSD.

    """
    snr_sqrd = np.nansum((h * h) / PSD)
    return np.sqrt(snr_sqrd)


def compute_frequency_optimal_snr(h_freq, psd, duration) -> float:
    """
    A18 from Veitch et al. 2009
    https://arxiv.org/abs/0911.3820
    """
    snr_sqrd = __noise_weighted_inner_product(
        aa=h_freq, bb=h_freq, power_spectral_density=psd, duration=duration
    ).real
    return np.sqrt(snr_sqrd)


def __noise_weighted_inner_product(aa, bb, power_spectral_density, duration):
    integrand = np.conj(aa) * bb / power_spectral_density
    return (4 / duration) * np.sum(integrand)


def evolutionary_psd_from_stationary_psd(
    psd: np.ndarray,
    psd_f: np.ndarray,
    f_grid,
    t_grid,
) -> "Wavelet":
    """
    PSD[ti,fi] = PSD[fi] * delta_f
    """

    Nt = len(t_grid)
    delta_F = f_grid[1] - f_grid[0]
    delta_T = t_grid[1] - t_grid[0]

    freq_data = psd
    nan_val = np.max(freq_data)
    psd_grid = (
        interp1d(
            psd_f,
            freq_data,
            kind="nearest",
            fill_value=nan_val,
            bounds_error=False,
        )(f_grid)
        # * delta_F
    )

    # repeat the PSD for each time bin
    psd_grid = np.repeat(psd_grid[None, :], Nt, axis=0)
    return psd_grid


def get_wavelet_bins(duration, data_len, Nf, Nt):
    """Get the bins for the wavelet transform
    Eq 4-6 in Wavelets paper
    """
    T = duration
    N = data_len
    fs = N / T
    fmax = fs / 2

    delta_t = T / Nt
    delta_f = 1 / (2 * delta_t)

    # assert delta_f == fmax / Nf, f"delta_f={delta_f} != fmax/Nf={fmax/Nf}"

    f_bins = np.arange(0, Nf) * delta_f
    t_bins = np.arange(0, Nt) * delta_t

    return t_bins, f_bins
