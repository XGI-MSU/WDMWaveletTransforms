import numpy as np
import scipy.stats
from numpy.testing import assert_allclose, assert_array_less
from numpy.typing import NDArray
from scipy.signal import correlate

from WDMWaveletTransforms.wavelet_transforms import inverse_wavelet_time, transform_wavelet_time


# np.seterr(all='raise')
def unit_normal_battery(
    signal: NDArray[np.floating],
    *,
    mult: float = 1.0,
    sig_thresh: float = 5.0,
    a2_cut: float = 2.28,
    do_assert: bool = True,
    verbose: bool = False,
) -> tuple[bool, float, float, float]:
    """
    Test if a signal is consistent with unit normal white noise.

    This function applies several statistical tests to determine if the input signal
    behaves like unit normal (mean 0, variance 1) white noise. It uses the Anderson-Darling
    test for normality, and checks for zero mean and unit variance. The default Anderson-Darling
    cutoff of 2.26 is hand selected give ~1 in 1e5 empirical probablity of false positive for n=64.
    The calibration is about the same for other n tested, such as n=32.

    Parameters
    ----------
    signal : NDArray[np.floating]
        The input signal array to be tested.
    mult : float
        Scaling factor applied to the signal before testing (default is 1.0).
    sig_thresh : float
        Threshold for the mean and standard deviation tests (default is 5.0).
    a2_cut : float
        Anderson-Darling test cutoff value (default is 2.28).
    do_assert : bool
        If True, assertions are raised if any test fails (default is True).
    verbose : bool
        If True, prints the Anderson-Darling statistic and cutoff (default is False).

    Returns
    -------
    test_combo : bool
        True if all tests are passed, False otherwise.
    a2_star : float
        Anderson-Darling test statistic (bias-corrected).
    mean_stat : float
        Normalized mean test statistic.
    std_stat : float
        Normalized standard deviation test statistic.
    """
    n_sig = signal.size
    if n_sig == 0:
        return True, 0.0, 0.0, 0.0

    sig_adjust = signal / mult
    mean_wave = np.mean(sig_adjust)
    std_wave = np.std(sig_adjust)
    std_std_wave: float = float(np.std(sig_adjust) * np.sqrt(2 / n_sig))

    # anderson darling test statistic assuming true mean and variance are unknown
    sig_sort = np.sort((sig_adjust - mean_wave) / std_wave)
    phis = scipy.stats.norm.cdf(sig_sort)
    xs = np.arange(1, n_sig + 1)
    a2: float = -n_sig - 1 / n_sig * np.sum((2 * xs - 1) * np.log(phis) + (2 * (n_sig - xs) + 1) * np.log(1 - phis))
    a2_star: float = a2 * (1 + 4 / n_sig - 25 / n_sig**2)
    if verbose:
        print(a2_star, a2_cut)

    mean_stat: float = float(np.abs(mean_wave) / std_wave * np.sqrt(n_sig))
    std_stat: float = float(np.abs(std_wave - 1.0) / std_std_wave)
    test1: bool = mean_stat < sig_thresh
    test2: bool = std_stat < sig_thresh
    test3: bool = bool(a2_star < a2_cut)  # should be less than cutoff value

    test_combo: bool = test1 and test2 and test3

    # check mean and variance
    if do_assert:
        assert_array_less(mean_stat, sig_thresh)
        assert_array_less(std_stat, sig_thresh)
        assert_array_less(a2_star, a2_cut)

    return test_combo, a2_star, mean_stat, std_stat


if __name__ == '__main__':
    Nf = 1024
    Nt = 1024
    mult = 64
    gen = np.random.default_rng(314159)

    data_time = gen.normal(0.0, 1.0, Nf * Nt)

    data_wavelet = transform_wavelet_time(data_time, Nf, Nt, mult=mult, family='modified_gaussian')
    data_time_rec = inverse_wavelet_time(data_wavelet, Nf, Nt, mult=mult, family='modified_gaussian')
    # check correlation of streams
    assert_allclose(1.0 - np.corrcoef(data_time, data_time_rec)[0, 1], 0.0, atol=1.0e-14)
    # check variance preserved for parseval's theorem
    assert_allclose(np.var(data_wavelet), np.var(data_time), atol=1.0e-100, rtol=3.0e-6)
    assert_allclose(np.var(data_wavelet), np.var(data_time_rec), atol=1.0e-100, rtol=3.0e-6)
    assert_allclose(np.var(data_time), np.var(data_time_rec), atol=1.0e-100, rtol=1.0e-14)
    # check mean preserved
    assert_allclose(np.mean(data_time), np.mean(data_time_rec), atol=1.0e-100, rtol=2.0e-12)

    # check known variance
    assert_allclose(np.var(data_time), 1.0, atol=1.0e-100, rtol=2.0e-3)
    assert_allclose(np.var(data_time_rec), 1.0, atol=1.0e-100, rtol=2.0e-3)
    assert_allclose(np.var(data_wavelet), 1.0, atol=1.0e-100, rtol=2.0e-3)

    print(np.var(data_wavelet) / np.var(data_time))
    print(np.mean(data_time_rec / data_time), np.var(data_time_rec) / np.var(data_time))

    unit_normal_battery(data_time)
    unit_normal_battery(data_wavelet.flatten())
    unit_normal_battery(data_time_rec)

    # check no unexpected correlations
    corr_wave = correlate(data_wavelet, data_wavelet, mode='same')
    corr_center = correlate(data_wavelet, data_wavelet, mode='same')[Nt // 2, Nf // 2]
    corr_wave /= corr_center
    corr_wave[Nt // 2, Nf // 2] = 0.0
    assert_allclose(corr_wave, 0.0, atol=4.0e-3)
    assert_allclose(np.mean(corr_wave), 0.0, atol=2.0e-6)
    assert_allclose(np.std(corr_wave) * 4 / 3 * np.sqrt(Nt * Nf), 1.0, atol=6.0e-4)
    assert_allclose(np.mean(corr_wave, axis=0), 0.0, atol=1.0e-4)
    assert_allclose(np.mean(corr_wave, axis=1), 0.0, atol=1.0e-4)

    # import matplotlib.pyplot as plt
    # plt.plot(data_time)
    # plt.plot(data_time_rec)
    # plt.show()
