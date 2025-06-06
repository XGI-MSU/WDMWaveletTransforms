"""helper functions for transform_freq"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.special
from numba import njit

import WDMWaveletTransforms.fft_funcs as fft

if TYPE_CHECKING:
    from numpy.typing import NDArray


def phitilde_vec(om: NDArray[np.float64], Nf: int, nx: float = 4.0) -> NDArray[np.float64]:
    """Compute phitilde, om i array, nx is filter steepness, defaults to 4."""
    OM: float = np.pi  # Nyquist angular frequency
    DOM: float = float(OM / Nf)  # 2 pi times DF
    insDOM: float = float(1.0 / np.sqrt(DOM))
    B = OM / (2 * Nf)
    A = (DOM - B) / 2
    z = np.zeros(om.size, dtype=np.float64)

    mask = (np.abs(om) >= A) & (np.abs(om) < A + B)

    x = (np.abs(om[mask]) - A) / B
    y = scipy.special.betainc(nx, nx, x)
    z[mask] = insDOM * np.cos(np.pi / 2.0 * y)

    z[np.abs(om) < A] = insDOM
    return z


def phitilde_vec_norm(Nf: int, Nt: int, nx: float) -> NDArray[np.float64]:
    """Normalize phitilde as needed for inverse frequency domain transform"""
    ND: int = Nf * Nt
    oms: NDArray[np.float64] = np.asarray(2 * np.pi / ND * np.arange(0, Nt // 2 + 1), dtype=np.float64)
    phif: NDArray[np.float64] = phitilde_vec(oms, Nf, nx)
    # nrm should be 1
    nrm: float = float(
        np.sqrt((2 * np.sum(phif[1:] ** 2) + phif[0] ** 2) * 2 * np.pi / ND) / (np.pi ** (3 / 2) / np.pi),
    )
    return phif / nrm


@njit()
def tukey(data: NDArray[np.float64 | np.complex128], alpha: float, N: int) -> None:
    """Apply tukey window function to data"""
    imin: int = int(alpha * (N - 1) / 2)
    imax: int = int((N - 1) * (1 - alpha / 2))
    Nwin: int = N - imax

    for i in range(N):
        f_mult: float = 1.0
        if i < imin:
            f_mult = float(0.5 * (1.0 + np.cos(np.pi * (i / imin - 1.0))))
        if i > imax:
            f_mult = float(0.5 * (1.0 + np.cos(np.pi / Nwin * (i - imax))))
        data[i] *= f_mult


@njit()
def DX_assign_loop(
    m: int,
    Nt: int,
    Nf: int,
    DX: NDArray[np.complex128],
    data: NDArray[np.complex128],
    phif: NDArray[np.float64],
) -> None:
    """Helper for assigning DX in the main loop"""
    assert len(DX.shape) == 1, 'Storage array must be 1D'
    assert len(data.shape) == 1, 'Data must be 1D'
    assert len(phif.shape) == 1, 'Phi array must be 1D'

    i_base: int = int(Nt // 2)
    jj_base: int = int(m * Nt // 2)

    if m in (0, Nf):
        # NOTE this term appears to be needed to recover correct constant (at least for m=0) but was previously missing
        DX[Nt // 2] = phif[0] * data[int(m * Nt // 2)] / 2.0
    else:
        DX[Nt // 2] = phif[0] * data[int(m * Nt // 2)]

    for jj in range(jj_base + 1 - int(Nt // 2), jj_base + int(Nt // 2)):
        j: int = int(np.abs(jj - jj_base))
        i: int = i_base - jj_base + jj
        if (m == Nf and jj > jj_base) or (m == 0 and jj < jj_base):
            DX[i] = 0.0
        elif j == 0:
            continue
        else:
            DX[i] = phif[j] * data[jj]


@njit()
def DX_unpack_loop(m: int, Nt: int, Nf: int, DX_trans: NDArray[np.complex128], wave: NDArray[np.float64]) -> None:
    """Helper for unpacking fftd DX in main loop"""
    assert len(DX_trans.shape) == 1, 'Data array must be 1D'
    assert len(wave.shape) == 2, 'Output array must be 2D'
    if m == 0:
        # half of lowest and highest frequency bin pixels are redundant
        # so store them in even and odd components of m=0 respectively
        for n in range(0, Nt, 2):
            wave[n, 0] = DX_trans[n].real * np.sqrt(2.0)
    elif m == Nf:
        for n in range(0, Nt, 2):
            wave[n + 1, 0] = DX_trans[n].real * np.sqrt(2.0)
    else:
        for n in range(Nt):
            if m % 2:
                if (n + m) % 2:
                    wave[n, m] = -DX_trans[n].imag
                else:
                    wave[n, m] = DX_trans[n].real
            elif (n + m) % 2:
                wave[n, m] = DX_trans[n].imag
            else:
                wave[n, m] = DX_trans[n].real


def transform_wavelet_freq_helper(
    data: NDArray[np.complex128],
    Nf: int,
    Nt: int,
    phif: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Helper to do the wavelet transform using the fast wavelet domain transform"""
    assert len(data.shape) == 1, 'Only support 1D Arrays currently'
    assert len(phif.shape) == 1, 'phif must be 1D'
    wave = np.zeros((Nt, Nf))  # wavelet wavepacket transform of the signal

    DX = np.zeros(Nt, dtype=np.complex128)
    for m in range(Nf + 1):
        DX_assign_loop(m, Nt, Nf, DX, data, phif)
        DX_trans = fft.ifft(DX, Nt)
        DX_unpack_loop(m, Nt, Nf, DX_trans, wave)
    return wave
