"""helper functions for transform_time.py"""

import numpy as np
from numba import njit
from numpy.typing import NDArray

import WDMWaveletTransforms.fft_funcs as fft
from WDMWaveletTransforms.transform_freq_funcs import phitilde_vec


@njit()
def assign_wdata(
    i: int,
    K: int,
    ND: int,
    Nf: int,
    wdata: NDArray[np.float64],
    data_pad: NDArray[np.float64],
    phi: NDArray[np.float64],
) -> None:
    """Assign wdata to be fftd in loop, data_pad needs K extra values on the right to loop"""
    assert len(wdata.shape) == 1, 'Input data array must be 1D'
    assert len(data_pad.shape) == 1, 'Input data array must be 1D'
    assert len(data_pad.shape) == 1, 'Input phi array must be 1D'

    # half_K = np.int64(K/2)
    jj = i * Nf - K // 2
    if jj < 0:
        jj += ND  # periodically wrap the data
    if jj >= ND:
        jj -= ND  # periodically wrap the data
    for j in range(K):
        # jj = i*Nf-half_K+j
        wdata[j] = data_pad[jj] * phi[j]  # apply the window
        jj += 1
        # if jj == ND:
        #    jj -= ND # periodically wrap the data


@njit()
def pack_wave(i: int, mult: int, Nf: int, wdata_trans: NDArray[np.complex128], wave: NDArray[np.float64]) -> None:
    """Pack fftd wdata into wave array"""
    assert len(wdata_trans.shape) == 1, 'Input data array must be 1D'
    assert wdata_trans.size == Nf * mult + 1, 'Input array must have size Nf*mult+1'
    assert len(wave.shape) == 2, 'Wave array must be 2D'

    if i % 2 == 0 and i < wave.shape[0] - 1:
        # m=0 value at even Nt and
        wave[i, 0] = wdata_trans[0].real / np.sqrt(2)
        wave[i + 1, 0] = wdata_trans[Nf * mult].real / np.sqrt(2)

    for j in range(1, Nf):
        if (i + j) % 2:
            wave[i, j] = -wdata_trans[j * mult].imag
        else:
            wave[i, j] = wdata_trans[j * mult].real


def transform_wavelet_time_helper(
    data: NDArray[np.float64],
    Nf: int,
    Nt: int,
    phi: NDArray[np.float64],
    mult: int,
) -> NDArray[np.float64]:
    """Helper function do do the wavelet transform in the time domain"""
    # the time domain data stream
    ND = Nf * Nt

    # mult, can cause bad leakage if it is too small but may be possible to mitigate
    # Filter is mult times pixel with in time

    K = mult * 2 * Nf

    # windowed data packets
    wdata = np.zeros(K)

    wave = np.zeros((Nt, Nf))  # wavelet wavepacket transform of the signal
    data_pad = np.zeros(ND + K)
    data_pad[:ND] = data
    data_pad[ND : ND + K] = data[:K]

    for i in range(Nt):
        assign_wdata(i, K, ND, Nf, wdata, data_pad, phi)
        wdata_trans = fft.rfft(wdata, K)
        pack_wave(i, mult, Nf, wdata_trans, wave)

    return wave


def phi_vec(Nf: int, nx: float = 4.0, mult: int = 16) -> NDArray[np.float64]:
    """Get time domain phi as fourier transform of phitilde_vec"""
    # TODO fix mult

    OM: float = np.pi
    DOM = float(OM / Nf)
    insDOM: float = float(1.0 / np.sqrt(DOM))
    K: int = int(mult * 2 * Nf)
    half_K: int = int(mult * Nf)  # np.int64(K/2)

    dom: np.float64 = np.float64(2 * np.pi / K)  # max frequency is K/2*dom = pi/dt = OM

    phitilde_loc = np.zeros(K, dtype=np.complex128)

    # zero frequency
    phitilde_loc[0] = insDOM

    # postive frequencies
    phitilde_loc[1 : half_K + 1] = phitilde_vec(dom * np.arange(1, half_K + 1), Nf, nx)
    # negative frequencies
    phitilde_loc[half_K + 1 :] = phitilde_vec(-dom * np.arange(half_K - 1, 0, -1), Nf, nx)
    phi_loc = K * fft.ifft(phitilde_loc, K)

    del phitilde_loc

    phi = np.zeros(K, dtype=np.float64)
    phi[0:half_K] = np.real(phi_loc[half_K:K])
    phi[half_K:] = np.real(phi_loc[0:half_K])

    nrm: float = float(np.sqrt(K / dom))  # *np.linalg.norm(phi)

    fac: float = float(float(np.sqrt(2.0)) / nrm)
    return phi * fac
