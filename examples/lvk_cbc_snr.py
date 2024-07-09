""" Example script to
1. Generate a CBC signal with a given SNR in LVK 04 noise
2. Perform a wavelet transform on the signal
3. Compute the SNR of the wavelet transform
4. Plot the wavelet transform + time domain signal


REQUIRES:
- WDMWaveletTransforms, bilby[gw], numba
"""

from WDMWaveletTransforms.wavelet_transforms import transform_wavelet_time
import bilby
from typing import Tuple
import numpy as np
from common import compute_wavelet_snr, evolutionary_psd_from_stationary_psd, get_wavelet_bins

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from collections import namedtuple
TIMESERIES = namedtuple("TimeSeries", ["data", "time"])

from scipy.interpolate import interp1d



DURATION = 8
SAMPLING_FREQUENCY = 4096
DT = 1 / SAMPLING_FREQUENCY
MINIMUM_FREQUENCY = 20
MAXIMUM_FREQUENCY = 256

CBC_GENERATOR = bilby.gw.WaveformGenerator(
    duration=DURATION,
    sampling_frequency=SAMPLING_FREQUENCY,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=dict(
        waveform_approximant="IMRPhenomD",
        reference_frequency=20.0,
        minimum_frequency=MINIMUM_FREQUENCY,
        maximum_frequency=MAXIMUM_FREQUENCY,
    ),
)

GW_PARMS = dict(
    mass_1=30,
    mass_2=30,  # 2 mass parameters
    a_1=0.1,
    a_2=0.1,
    tilt_1=0.0,
    tilt_2=0.0,
    phi_12=0.0,
    phi_jl=0.0,  # 6 spin parameters
    ra=1.375,
    dec=-1.2108,
    luminosity_distance=2000.0,
    theta_jn=0.0,  # 7 extrinsic parameters
    psi=2.659,
    phase=1.3,
    geocent_time=0,
)


def _get_ifo(t0=0.0, noise=True):
    ifos = bilby.gw.detector.InterferometerList(["H1"])  # design sensitivity
    if noise:
        ifos.set_strain_data_from_power_spectral_densities(
            sampling_frequency=SAMPLING_FREQUENCY,
            duration=DURATION,
            start_time=t0,
        )
    else:
        ifos.set_strain_data_from_zero_noise(
            sampling_frequency=SAMPLING_FREQUENCY,
            duration=DURATION,
            start_time=t0,
        )
    return ifos


def inject_signal_in_noise(
        mc, q=1, distance=1000.0, noise=True
) -> Tuple[TIMESERIES, float]:
    injection_parameters = GW_PARMS.copy()
    (
        injection_parameters["mass_1"],
        injection_parameters["mass_2"],
    ) = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(
        mc, q
    )
    injection_parameters["luminosity_distance"] = distance

    ifos = _get_ifo(injection_parameters["geocent_time"] + 1.5, noise=noise)
    ifos.inject_signal(
        waveform_generator=CBC_GENERATOR, parameters=injection_parameters
    )
    ifo: bilby.gw.detector.Interferometer = ifos[0]

    snr = ifo.meta_data["matched_filter_SNR"]

    ifo.time_array = np.linspace(0, DURATION, int(DURATION * SAMPLING_FREQUENCY))
    data = TIMESERIES(ifo.strain_data.time_domain_strain, ifo.time_array)
    return data, np.abs(snr)


def get_lvk_psd():
    ifo: bilby.gw.detector.Interferometer = _get_ifo()[0]
    psd = ifo.power_spectral_density.psd_array
    psd_f = ifo.power_spectral_density.frequency_array
    return psd, psd_f


def get_lvk_psd_function():
    psd, psd_f = get_lvk_psd()
    return interp1d(psd_f, psd, bounds_error=False, fill_value=max(psd))



def plot(signal_t, optimal_snr_ht, psd_f, psd, signal_wavelet, time_grid, freq_grid, psd_wavelet, snr_wavelet, DURATION, MINIMUM_FREQUENCY):
    # plot Timeseries, wavelet signal and PSD
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    ax[0, 0].plot(signal_t.time, signal_t.data)
    ax[0, 0].set_xlim(0,DURATION)
    ax[0, 0].set_xlabel("Time [s]")
    ax[0, 0].set_ylabel("Strain")
    ax[0, 0].set_title("Time domain signal")
    # ANNOTATE SNR
    ax[0, 0].text(0.1, 0.85, f"SNR: {optimal_snr_ht:.2f}", transform=ax[0, 0].transAxes, fontsize='x-large')
    ax[0, 1].loglog(psd_f, psd)
    ax[0, 1].set_title("PSD in frequency domain")

    ax[1, 0].pcolor(time_grid, freq_grid, signal_wavelet.T, cmap="RdBu", norm=colors.TwoSlopeNorm( vcenter=0))
    ax[1, 0].set_ylim(MINIMUM_FREQUENCY, 512)
    ax[1, 0].set_xlabel("Time [s]")
    ax[1, 0].set_ylabel("Frequency [Hz]")
    ax[1, 0].set_title("Wavelet transform")
    ax[1, 0].text(0.1, 0.85, f"SNR: {snr_wavelet:.2f}", transform=ax[1, 0].transAxes, fontsize='x-large')
    ax[1,1].pcolor(time_grid, freq_grid, psd_wavelet.T, cmap="viridis", norm='log')
    ax[1,1].set_yscale('log')
    ax[1,1].set_ylim(MINIMUM_FREQUENCY, 512)
    ax[1,1].set_title("PSD in wavelet domain")
    ax[1,1].set_xlabel("Time [s]")
    ax[1,1].set_ylabel("Frequency [Hz]")
    return fig

def main():
    signal_t, optimal_snr_ht = inject_signal_in_noise(mc=30, noise=False)
    psd, psd_f = get_lvk_psd()
    ND = len(signal_t.time)
    Nt = 128
    Nf = ND // Nt
    signal_wavelet = transform_wavelet_time(signal_t.data, Nf, Nt)
    time_grid, freq_grid = get_wavelet_bins(DURATION, ND, Nf, Nt)
    psd_wavelet = evolutionary_psd_from_stationary_psd(psd, psd_f, freq_grid, time_grid)
    snr_wavelet = compute_wavelet_snr(signal_wavelet, psd_wavelet)
    print("SNR in time domain:", optimal_snr_ht)
    print(f"ND: {ND}, Nt: {Nt}, Nf: {Nf}")
    print("SNR in wavelet domain:", snr_wavelet)
    plot(
        signal_t, optimal_snr_ht,
        psd_f, psd,
        signal_wavelet, time_grid, freq_grid,
        psd_wavelet, snr_wavelet,
        DURATION, MINIMUM_FREQUENCY
    ).savefig('lvk_cbc_snr.pdf', dpi=300)


if __name__ == "__main__":
    main()
