"""helper functions for transform_freq"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.special
from numba import njit

import WDMWaveletTransforms.fft_funcs as fft

if TYPE_CHECKING:
    from numpy.typing import NDArray


def phitilde_vec(om: NDArray[np.floating], Nf: int, nx: float = 4.0) -> NDArray[np.floating]:
    """Compute phitilde, om i array, nx is filter steepness, defaults to 4."""
    OM: float = np.pi  # Nyquist angular frequency
    DOM: float = float(OM / Nf)  # 2 pi times DF
    insDOM: float = float(1.0 / np.sqrt(DOM))
    B = OM / (2 * Nf)
    A = (DOM - B) / 2
    z = np.zeros(om.size, dtype=float)

    mask = (np.abs(om) >= A) & (np.abs(om) < A + B)

    x = (np.abs(om[mask]) - A) / B
    # guarantee restriction to range of betainc
    x[x > 1.0] = 1.0
    x[x < 0.0] = 0.0
    y = scipy.special.betainc(nx, nx, x)
    z[mask] = insDOM * np.cos(np.pi / 2.0 * y)

    z[np.abs(om) < A] = insDOM
    return z


def phitilde_vec_norm(Nf: int, Nt: int, nx: float, mult_f: int) -> NDArray[np.floating]:
    """Normalize phitilde as needed for inverse frequency domain transform"""
    ND: int = Nf * Nt
    oms: NDArray[np.floating] = np.asarray(2 * np.pi / ND * np.arange(0, mult_f * Nt // 2 + 1), dtype=float)
    phif: NDArray[np.floating] = phitilde_vec(oms, Nf, nx)
    # nrm should be 1
    nrm: float = float(
        np.sqrt((2 * np.sum(phif[1:] ** 2) + phif[0] ** 2) * 2 * np.pi / ND) / (np.pi ** (3 / 2) / np.pi),
    )
    return phif / nrm


@njit()
def tukey(data: NDArray[np.floating | np.complexfloating], alpha: float, N: int) -> None:
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
    mult_f: int,
    DX: NDArray[np.complexfloating],
    data: NDArray[np.complexfloating],
    phif: NDArray[np.floating],
) -> None:
    """Helper for assigning DX in the main loop.

    Tap delta = i - K/2 multiplies the spectrum at bin jj = m*Nt/2 + delta.  For
    interior bands, bins outside the rfft range [0, ND/2] are read from the full
    conjugate-symmetric spectrum of the real signal: X[-jj] = conj(X[jj]) and
    X[ND - jj] = conj(X[jj]) ("conjugate fold-back"), which makes the pixel the
    exact inner product with the real wavelet even when the window crosses the
    frequency extremes.  The self-conjugate bands m = 0 and m = Nf instead keep
    only their own half-plane with the center tap halved: the reflection maps
    the window onto itself there, so Re[] of the half sum is already the exact
    fold.

    Taps are alias-folded onto one Nt-period (the transform only ever needs the
    length-K ifft at stride mult_f, which equals the length-Nt ifft of the
    aliased taps), so DX has length Nt.  The unpaired tap at delta = -K/2 is
    excluded: effective support is the open range delta in (-K/2, K/2).
    """
    assert len(DX.shape) == 1, 'Storage array must be 1D'
    assert len(data.shape) == 1, 'Data must be 1D'
    assert len(phif.shape) == 1, 'Phi array must be 1D'

    assert 0 <= m <= Nf
    assert Nf % 2 == 0
    assert Nt % 2 == 0
    assert Nf > 0
    assert Nt > 0
    assert 0 < mult_f <= Nf

    K = mult_f * Nt
    half_K = int(K // 2)
    half_Nt = int(Nt // 2)

    ND = Nf * Nt
    half_ND = int(ND // 2)

    assert phif.shape == (half_K + 1,)
    assert DX.shape == (Nt,)
    assert data.shape == (half_ND + 1,)

    DX[:] = 0.0

    for i in range(1, K):
        j = abs(i - half_K)
        jj = m * half_Nt + i - half_K
        if j == 0:
            if m in (0, Nf):
                # halve the self-conjugate center tap
                DX[i % Nt] += phif[j] * data[jj] / 2.0
            else:
                DX[i % Nt] += phif[j] * data[jj]
        elif jj < 0 or jj > half_ND:
            if m in (0, Nf):
                # self-conjugate bands keep only their own half-plane
                pass
            elif jj < 0:
                DX[i % Nt] += phif[j] * np.conj(data[-jj])
            else:
                DX[i % Nt] += phif[j] * np.conj(data[ND - jj])
        else:
            DX[i % Nt] += phif[j] * data[jj]


@njit()
def DX_unpack_loop(
    m: int, Nt: int, Nf: int, mult_f: int, DX_trans: NDArray[np.complexfloating], wave: NDArray[np.floating]
) -> None:
    """Helper for unpacking fftd DX in main loop.

    DX_trans is the length-Nt ifft of the alias-folded taps; the tap positions
    carry an offset of mult_f*Nt/2, so relative to the (-1)^n convention of the
    analytic coefficient an extra factor (-1)^((mult_f+1)*n) appears, giving the
    sign flips conditioned on the parity of mult_f below (n's parity is fixed by
    the parities of m and n+m).
    """
    assert len(DX_trans.shape) == 1, 'Data array must be 1D'
    assert len(wave.shape) == 2, 'Output array must be 2D'

    assert 0 <= m <= Nf
    assert Nf % 2 == 0
    assert Nt % 2 == 0
    assert Nf > 0
    assert Nt > 0
    assert mult_f > 0

    assert DX_trans.shape == (Nt,)
    assert wave.shape == (Nt, Nf)

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
                    if mult_f % 2:
                        wave[n, m] = DX_trans[n].real
                    else:
                        wave[n, m] = -DX_trans[n].real
            elif (n + m) % 2:
                if mult_f % 2:
                    wave[n, m] = DX_trans[n].imag
                else:
                    wave[n, m] = -DX_trans[n].imag
            else:
                wave[n, m] = DX_trans[n].real


def transform_wavelet_freq_helper(
    data: NDArray[np.complexfloating],
    Nf: int,
    Nt: int,
    mult_f: int,
    phif: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Helper to do the wavelet transform using the fast wavelet domain transform"""
    assert len(data.shape) == 1, 'Only support 1D Arrays currently'
    assert len(phif.shape) == 1, 'phif must be 1D'

    assert Nf % 2 == 0
    assert Nt % 2 == 0
    assert Nf > 0
    assert Nt > 0
    assert 0 < mult_f <= Nf, 'window must not wrap around the full spectrum'

    K = mult_f * Nt

    assert data.shape == (Nf * Nt // 2 + 1,)
    assert phif.shape == (K // 2 + 1,)

    wave = np.zeros((Nt, Nf))  # wavelet wavepacket transform of the signal

    DX = np.zeros(Nt, dtype=complex)
    for m in range(Nf + 1):
        DX_assign_loop(m, Nt, Nf, mult_f, DX, data, phif)
        DX_trans = fft.ifft(DX, Nt)
        DX_unpack_loop(m, Nt, Nf, mult_f, DX_trans, wave)
    return wave
