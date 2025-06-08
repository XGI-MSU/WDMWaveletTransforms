import numpy as np
from scipy.signal.windows import tukey
import matplotlib.pyplot as plt
from scipy.signal import chirp, spectrogram

from WDMWaveletTransforms.wavelet_transforms import transform_wavelet_time
from WDMWaveletTransforms.wavelet_transforms import inverse_wavelet_time

FREQ_RANGE = (10, 500)


def waveform_fft(
    t,
    waveform,
):
    N = len(waveform)
    taper = tukey(N, 0.1)
    waveform_w_pad = zero_pad(waveform * taper)
    waveform_f = np.fft.rfft(waveform_w_pad)[1:]

    n_t = len(zero_pad(t))
    delta_t = t[1] - t[0]
    freq = np.fft.rfftfreq(n_t, delta_t)[1:]
    return freq, waveform_f


def zero_pad(data):
    N = len(data)
    pow_2 = np.ceil(np.log2(N))
    return np.pad(data, (0, int((2**pow_2) - N)), "constant")


# plot signal
def plot_time_domain_signal(t, h):
    T = max(t)
    fs = 1 / (t[1] - t[0])
    ff, tt, Sxx = spectrogram(h, fs=fs, nperseg=256, nfft=576)
    freq, h_freq = waveform_fft(t, h)

    fig, axes = plt.subplots(3, 1, figsize=(4, 6))
    axes[0].plot(t, h, lw=0.1)
    axes[0].set_ylabel("h(t)")
    axes[0].set_xlim(0, T)
    axes[1].plot(freq, np.abs(h_freq))
    axes[1].set_ylabel("|h(f)|")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_xlim(*FREQ_RANGE)
    axes[2].set_xlim(0, T)
    axes[2].pcolormesh(tt, ff, Sxx)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Frequency (Hz)")
    axes[2].set_xlim(0, T)
    axes[2].set_ylim(*FREQ_RANGE)
    # add colorbar to the last axis
    cbar = plt.colorbar(axes[2].collections[0], ax=axes[2])
    cbar.set_label("Amplitude")
    plt.tight_layout()
    return fig


def plot_wavelet_domain_signal(wavelet_data, time_grid, freq_grid):
    fig = plt.figure()
    plt.imshow(
        np.abs(np.rot90(wavelet_data)),
        aspect="auto",
        extent=[time_grid[0], time_grid[-1], freq_grid[0], freq_grid[-1]],
    )
    cbar = plt.colorbar()
    cbar.set_label("Wavelet Amplitude")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.ylim(*FREQ_RANGE)
    plt.tight_layout()
    return fig


def generate_time_domain_signal(t):
    return chirp(t, f0=FREQ_RANGE[0], f1=FREQ_RANGE[1], t1=t[-1], method="quadratic")


def main():
    dt = 1 / 4096
    Nt = 2**6
    Nf = 2**7
    mult = 16

    ND = Nt * Nf
    Tobs = dt * ND

    # time and frequency grids
    ts = np.arange(0, ND) * dt
    fs = np.arange(0, ND // 2 + 1) * 1 / (Tobs)

    # generate signal
    h = generate_time_domain_signal(ts)

    # plot original signal
    fig = plot_time_domain_signal(ts, h)
    fig.savefig("original_signal.png", dpi=300)

    # transform to wavelet domain
    h_wavelet = transform_wavelet_time(h, Nf=Nf, Nt=Nt, mult=mult)
    fig = plot_wavelet_domain_signal(h_wavelet, ts, fs)
    fig.savefig("wavelet_domain.png", dpi=300)

    # transform back to time domain
    h_reconstructed = inverse_wavelet_time(h_wavelet, Nf=Nf, Nt=Nt)
    fig = plot_time_domain_signal(ts, h_reconstructed)
    fig.savefig("reconstructed_signal.png", dpi=300)

    # check that the reconstructed signal is the same as the original
    residuals = h - h_reconstructed
    fig = plt.figure()
    plt.hist(residuals, bins=100)
    plt.xlabel("Residuals")
    plt.ylabel("Count")
    fig.savefig("residuals.png", dpi=300)
    assert np.allclose(h, h_reconstructed, atol=1e-3)


if __name__ == "__main__":
    main()
