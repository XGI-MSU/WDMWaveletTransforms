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
def DX_assign_loop_old(
    m: int,
    Nt: int,
    Nf: int,
    mult_f: int,
    DX: NDArray[np.complexfloating],
    data: NDArray[np.complexfloating],
    phif: NDArray[np.floating],
) -> None:
    """Helper for assigning DX in the main loop"""
    assert len(DX.shape) == 1, 'Storage array must be 1D'
    assert len(data.shape) == 1, 'Data must be 1D'
    assert len(phif.shape) == 1, 'Phi array must be 1D'

    assert 0 <= m <= Nf
    assert Nf % 2 == 0
    assert Nt % 2 == 0
    assert Nf > 0
    assert Nt > 0
    assert mult_f > 0

    K = mult_f * Nt
    half_K = int(K // 2)
    half_Nt = int(Nt // 2)

    ND = Nf * Nt
    half_ND = int(ND // 2)

    assert phif.shape == (half_K + 1,)
    assert DX.shape == (K,)
    assert data.shape == (half_ND + 1,)

    DX[:] = 0.0

    i_base: int = mult_f * half_Nt
    jj_base: int = m * half_Nt
    if m in (0, Nf):
        # NOTE this term appears to be needed to recover correct constant (at least for m=0) but was previously missing
        DX[half_K] = phif[0] * data[m * half_Nt] / 2.0
    else:
        DX[half_K] = phif[0] * data[m * half_Nt]

    # should never be set anywhere, but explicitly ensure it is 0
    DX[0] = 0.0

    for jj in range(jj_base + 1 - half_K, jj_base + half_K, mult_f):
        j: int = int(np.abs(jj - jj_base))
        i: int = i_base - jj_base + jj
        if jj < 0 or jj > half_ND or (m == Nf and jj > jj_base) or (m == 0 and jj < jj_base):
            DX[i] = 0.0
        elif j == 0:
            # happens when i == half_K, handled as special case above
            continue
        else:
            DX[i] = phif[j] * data[jj]


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
    """Helper for assigning DX in the main loop"""
    assert len(DX.shape) == 1, 'Storage array must be 1D'
    assert len(data.shape) == 1, 'Data must be 1D'
    assert len(phif.shape) == 1, 'Phi array must be 1D'

    assert 0 <= m <= Nf
    assert Nf % 2 == 0
    assert Nt % 2 == 0
    assert Nf > 0
    assert Nt > 0
    assert mult_f > 0

    K = mult_f * Nt
    half_K = int(K // 2)
    half_Nt = int(Nt // 2)

    ND = Nf * Nt
    half_ND = int(ND // 2)

    assert phif.shape == (half_K + 1,)
    assert DX.shape == (K,)
    assert data.shape == (half_ND + 1,)

    DX[:] = 0.0

    i_base: int = mult_f * half_Nt
    jj_base: int = m * half_Nt
    for i in range(K):
        j = abs(i - half_K)
        jj = m * half_Nt + i - half_K
        if j == 0:
            if m in (0, Nf):
                # NOTE this term appears to be needed to recover correct constant (at least for m=0) but was previously missing
                DX[i] = phif[j] * data[jj] / 2.0
            else:
                DX[i] = phif[j] * data[jj]
        elif jj < 0 or jj > half_ND:
            DX[i] = 0.0
        else:
            DX[i] = phif[j] * data[jj]
    DX[0] = 0.0


@njit()
def DX_unpack_loop(
    m: int, Nt: int, Nf: int, mult_f: int, DX_trans: NDArray[np.complexfloating], wave: NDArray[np.floating]
) -> None:
    """Helper for unpacking fftd DX in main loop"""
    assert len(DX_trans.shape) == 1, 'Data array must be 1D'
    assert len(wave.shape) == 2, 'Output array must be 2D'

    assert 0 <= m <= Nf
    assert Nf % 2 == 0
    assert Nt % 2 == 0
    assert Nf > 0
    assert Nt > 0
    assert mult_f > 0

    K = mult_f * Nt

    assert DX_trans.shape == (K,)
    assert wave.shape == (Nt, Nf)

    if m == 0:
        # half of lowest and highest frequency bin pixels are redundant
        # so store them in even and odd components of m=0 respectively
        for n in range(0, Nt, 2):
            wave[n, 0] = mult_f * DX_trans[n * mult_f].real * np.sqrt(2.0)
    elif m == Nf:
        for n in range(0, Nt, 2):
            wave[n + 1, 0] = mult_f * DX_trans[n * mult_f].real * np.sqrt(2.0)
    else:
        for n in range(Nt):
            if m % 2:
                if (n + m) % 2:
                    wave[n, m] = -mult_f * DX_trans[n * mult_f].imag
                else:
                    if mult_f % 2:
                        wave[n, m] = mult_f * DX_trans[n * mult_f].real
                    else:
                        wave[n, m] = -mult_f * DX_trans[n * mult_f].real
            elif (n + m) % 2:
                if mult_f % 2:
                    wave[n, m] = mult_f * DX_trans[n * mult_f].imag
                else:
                    wave[n, m] = -mult_f * DX_trans[n * mult_f].imag
            else:
                wave[n, m] = mult_f * DX_trans[n * mult_f].real


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
    assert mult_f > 0

    K = mult_f * Nt

    assert data.shape == (Nf * Nt // 2 + 1,)
    assert phif.shape == (K // 2 + 1,)

    wave = np.zeros((Nt, Nf))  # wavelet wavepacket transform of the signal
    wave_alt = np.zeros((Nt, Nf))  # wavelet wavepacket transform of the signal

    DX = np.zeros(K, dtype=complex)
    # DX_alt = np.zeros(Nt, dtype=complex)
    for m in range(Nf + 1):
        DX_assign_loop(m, Nt, Nf, mult_f, DX, data, phif)
        # DX_assign_loop_old(m, Nt, Nf, 1, DX_alt, data, phif[:Nt//2+1])
        # import matplotlib.pyplot as plt
        # plt.plot(np.arange(-K//2, K//2), np.abs(DX))
        # plt.plot(np.arange(-Nt//2, Nt//2), np.abs(DX_alt))
        # plt.show()
        # assert_allclose(DX[K//2:K//2+Nt//2], DX_alt[Nt//2:Nt//2+Nt//2], atol=1.e-100, rtol=1.e-10)
        # assert_allclose(DX[K//2-Nt//2+1:K//2], DX_alt[1:Nt//2], atol=1.e-100, rtol=1.e-10)
        DX_trans = fft.ifft(DX, K)
        # DX_trans_alt = fft.ifft(DX_alt, Nt)
        # plt.plot(np.linspace(-1., 1., K)[::mult_f], np.imag(DX_trans)[::mult_f]*mult_f)
        # plt.plot(np.linspace(-1., 1., Nt), np.imag(DX_trans_alt))
        # plt.show()
        DX_unpack_loop(m, Nt, Nf, mult_f, DX_trans, wave)
        # DX_unpack_loop(m, Nt, Nf, 1, DX_trans_alt, wave_alt)
        # assert_allclose(wave[:,m]*mult_f, wave_alt[:,m], atol=1.e-2, rtol=1.e-2)
        # if m == Nf:
        #    plt.plot(wave[:,0]*mult_f)
        #    plt.plot(wave_alt[:,0])
        #    plt.show()
    return wave
