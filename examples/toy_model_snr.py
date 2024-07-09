"""

Toy model of sine-wave signal and a constant PSD

"""

import numpy as np
from matplotlib import colors

import matplotlib.pyplot as plt
from common import compute_frequency_optimal_snr, evolutionary_psd_from_stationary_psd, compute_wavelet_snr, get_wavelet_bins
from WDMWaveletTransforms.wavelet_transforms import transform_wavelet_time

from collections import namedtuple

TIMESERIES = namedtuple("TimeSeries", ["data", "time"])

true_f = 10
fs = 20 * 3
dt = 1 / fs
ND = 4096
time = np.linspace(0, ND * dt, ND)
duration = ND * dt
signal_t = np.sin(2 * np.pi * true_f * time)
signal_f = np.fft.fft(signal_t)
freq = np.fft.fftfreq(ND, dt)

psd_f = np.ones(len(signal_f))
snr = compute_frequency_optimal_snr(
    signal_f[: ND // 2], psd_f[: ND // 2], duration=duration
)
signal_t = TIMESERIES(
    data=signal_t,
    time=time,
)

# analytical SNR for flat white noise in time domain:
# sin**2 ~ 0.5
sigma_sqr = 0.5 * dt
analytical_snr = np.sqrt(np.sum(signal_t.data**2 / sigma_sqr))

assert np.isclose(
    analytical_snr - snr, 0, atol=1e-2
), f"{analytical_snr} != {snr}"


Nf = 64
Nt = ND // Nf

signal_wavelet = transform_wavelet_time(signal_t.data, Nf=Nf, Nt=Nt)
freq_grid, time_grid = get_wavelet_bins(duration, ND, Nf, Nt)



psd_wavelet = evolutionary_psd_from_stationary_psd(
    psd=psd_f,
    psd_f=np.arange(len(psd_f)),
    f_grid=freq_grid,
    t_grid=time_grid,
)


wavelet_snr = compute_wavelet_snr(signal_wavelet, psd_wavelet)




fig, axes = plt.subplots(2,1, figsize=(5, 8))
axes[0].loglog(freq, np.abs(signal_f), label='Signal')
axes[0].axvline(true_f, color="red", linestyle="--", label="True frequency")
axes[0].plot(freq, psd_f, label="PSD")
axes[0].text(0.1, 0.85, f"Freq SNR: {snr:.2f}", transform=axes[0].transAxes, fontsize='x-large')
axes[0].text(0.1, 0.8, f"timeseries SNR: {analytical_snr:.2f}", transform=axes[0].transAxes, fontsize='x-large')
axes[0].legend()
axes[0].set_xlim(0, 20)
axes[0].set_title("Signal and PSD")
axes[0].set_xlabel("Frequency [Hz]")
axes[0].set_ylabel("PSD")
axes[1].pcolor(time_grid, freq_grid, signal_wavelet.T, cmap="RdBu", norm=colors.TwoSlopeNorm( vcenter=0))
axes[1].text(0.1, 0.85, f"Wavelet SNR: {wavelet_snr:.2f}", transform=axes[1].transAxes, fontsize='x-large')
axes[1].set_xlabel("Time bins")
axes[1].set_ylabel("Frequency bins")
fig.savefig('toy_model_snr.pdf', dpi=300)


assert np.isclose(
    snr, wavelet_snr, atol=10
), f"{snr} != {wavelet_snr}, wavelet/freq snr = {snr / wavelet_snr:.2f}"