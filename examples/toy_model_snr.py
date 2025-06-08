"""

Toy model of sine-wave signal and a constant PSD

"""
import matplotlib.pyplot as plt
import numpy as np
from WDMWaveletTransforms.wavelet_transforms import transform_wavelet_time, inverse_wavelet_time
import pytest
import scipy

def flat_psd_func(f, psd_amp):
    return psd_amp * np.ones(len(f))


def colored_psd_func(f, psd_amp):
    return psd_amp * np.ones(len(f)) * (1 + (f / 10) ** 2)


def colored_psd_func2(f, psd_amp):
    return psd_amp * np.ones(len(f)) * (1 + 1 / (f / 10) ** 2)


# pytest parameterize decorator
@pytest.mark.parametrize(
    "f0, T, A, PSD_AMP, Nf, psd_func",
    [
        (20, 1000, 1e-3, 1e-2, 16, flat_psd_func),
        (10, 1000, 1e-3, 1e-2, 32, colored_psd_func2),
        (20, 1000, 1e-3, 1e-2, 16, colored_psd_func),
    ])
def test_wavelet_timedomain_snr(f0, T, A, PSD_AMP, Nf, psd_func):
    ########################################
    # Part1: Analytical SNR calculation
    ########################################
    dt = 0.5 / (2 * f0)  # Shannon's sampling theorem, set dt < 1/2*highest_freq
    t = np.arange(0, T, dt)  # Time array
    # round len(t) to the nearest power of 2
    t = t[:2 ** int(np.log2(len(t)))]
    T = len(t) * dt

    y = A * np.sin(2 * np.pi * f0 * t)  # Signal waveform we wish to test

    freq = np.fft.fftshift(np.fft.fftfreq(len(y), dt))  # Frequencies
    df = abs(freq[1] - freq[0])  # Sample spacing in frequency

    y_fft = dt * np.fft.fftshift(np.fft.fft(y))  # continuous time fourier transform [seconds]
    N_f = len(y_fft)
    N_t = len(y)

    PSD = psd_func(freq, PSD_AMP)  # PSD of the noise

    # Compute the SNRs
    SNR2_f = 2 * np.sum(abs(y_fft) ** 2 / PSD) * df
    SNR2_t = 2 * dt * np.sum(abs(y) ** 2 / PSD)
    SNR2_t_analytical = (A ** 2) * T / PSD[0]

    ########################################
    # Part2: Wavelet domain
    ########################################

    ND = len(y)
    Nt = ND // Nf
    ND = Nf * Nt

    signal_wavelet = transform_wavelet_time(y, Nf=Nf, Nt=Nt) * np.sqrt(2) * dt

    delta_t = T / Nt
    delta_f = 1 / (2 * delta_t)
    freq_grid = np.arange(0, Nf) * delta_f
    time_grid = np.arange(0, Nt) * delta_t
    psd = psd_func(freq_grid, PSD_AMP)
    psd_wavelet = np.repeat(psd[None, :], Nt, axis=0) * dt

    wavelet_snr2 = np.sum((signal_wavelet * signal_wavelet / psd_wavelet))
    mse = np.mean((y - inverse_wavelet_time(signal_wavelet, Nf=Nf, Nt=Nt)) ** 2)
    print('---------')
    print(f"SNR squared in the frequency domain is = {SNR2_f:.2f}")
    print(f"SNR squared in the time domain (Parseval's theorem) is = {SNR2_t:.2f}", )
    print(f"(pen and paper) Analytical result would predict SNR squared = {SNR2_t_analytical:.2f}")
    print(f"In the wavelet domain, SNR_sqr = {wavelet_snr2:.2f}")
    print(f"Mean squared error in the wavelet domain = {mse:.2f}")
    print('---------')
    assert np.isclose(SNR2_f, wavelet_snr2, atol=1e-2), "SNR in time domain and wavelet domain should be the same"



def test_chirp_signal():
    T = 1000
    f0 = 1e-3
    f1 = 20
    dt = 0.5 / (2 * f1)  # Shannon's sampling theorem, set dt < 1/2*highest_freq
    t = np.arange(0, T, dt)  # Time array
    # round len(t) to the nearest power of 2
    t = t[:2 ** int(np.log2(len(t)))]
    T = len(t) * dt
    y = scipy.signal.chirp(t, f0=f0, f1=f1, t1=T, method='quadratic')
    # plot spectogram
    plt.specgram(y, Fs=1/dt, NFFT=128, noverlap=64, cmap='viridis')
    plt.show()


    freq = np.fft.fftshift(np.fft.fftfreq(len(y), dt))  # Frequencies
    df = abs(freq[1] - freq[0])  # Sample spacing in frequency

    y_fft = dt * np.fft.fftshift(np.fft.fft(y))  # continuous time fourier transform [seconds]
    N_f = len(y_fft)
    N_t = len(y)

    psd_func = flat_psd_func
    PSD_AMP = 1e-2
    PSD = psd_func(freq, PSD_AMP)  # PSD of the noise

    # Compute the SNRs
    SNR2_f = 2 * np.sum(abs(y_fft) ** 2 / PSD) * df

    ########################################
    # Part2: Wavelet domain
    ########################################
    Nf = 16
    ND = len(y)
    Nt = ND // Nf
    ND = Nf * Nt

    signal_wavelet = transform_wavelet_time(y, Nf=Nf, Nt=Nt) * np.sqrt(2) * dt

    delta_t = T / Nt
    delta_f = 1 / (2 * delta_t)
    freq_grid = np.arange(0, Nf) * delta_f
    time_grid = np.arange(0, Nt) * delta_t
    psd = psd_func(freq_grid, PSD_AMP)
    psd_wavelet = np.repeat(psd[None, :], Nt, axis=0) * dt

    wavelet_snr2 = np.sum((signal_wavelet * signal_wavelet / psd_wavelet))
    mse = np.mean((y - inverse_wavelet_time(signal_wavelet, Nf=Nf, Nt=Nt)) ** 2)
    print('---------')

    assert np.isclose(SNR2_f, wavelet_snr2, atol=1e-2), "SNR in time domain and wavelet domain should be the same"






if __name__ == '__main__':
    test_chirp_signal()
