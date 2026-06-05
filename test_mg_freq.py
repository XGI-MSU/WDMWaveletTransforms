import numpy as np
from numpy.testing import assert_allclose
from numpy.typing import NDArray
from scipy.signal import correlate

from test_mg_time import unit_normal_battery
from WDMWaveletTransforms.wavelet_transforms import inverse_wavelet_freq, transform_wavelet_freq

# np.seterr(all='raise')


def parseval_rfft(sig_freq: NDArray[np.floating] | NDArray[np.complexfloating], ND: int) -> float:
    """Sum for parseval theorem in freq domain"""
    if ND % 2:
        pars2 = 1 / ND * (np.abs(sig_freq[0]) ** 2 + 2 * np.sum(np.abs(sig_freq[1 : ND // 2 + 1]) ** 2))
    else:
        pars2 = (
            1
            / ND
            * (
                np.abs(sig_freq[0]) ** 2
                + np.abs(sig_freq[ND // 2]) ** 2
                + 2 * np.sum(np.abs(sig_freq[1 : ND // 2]) ** 2)
            )
        )
    return pars2


if __name__ == '__main__':
    Nf = 2048
    Nt = 1024
    gen = np.random.default_rng(314159)

    mult_f = 7

    scale = np.sqrt((Nf * Nt) // 2)
    data_freq = scale * (gen.normal(0.0, 1.0, (Nf * Nt) // 2 + 1) + 1j * gen.normal(0.0, 1.0, (Nf * Nt) // 2 + 1))
    # highest and lowest frequency components are real
    data_freq[0] = np.real(data_freq[0])
    data_freq[-1] = np.real(data_freq[-1])

    data_wavelet = transform_wavelet_freq(data_freq, Nf, Nt, mult_f=mult_f)
    # TODO temporary fudge until normalization issue handled
    # data_wavelet = data_wavelet / (np.std(data_wavelet))
    data_freq_rec = inverse_wavelet_freq(data_wavelet, Nf, Nt, mult_f=mult_f)

    print(scale * np.var(data_wavelet) / np.var(data_freq))
    print(np.mean(data_freq_rec / data_freq), np.var(data_freq_rec) / np.var(data_freq))

    import matplotlib.pyplot as plt

    plt.plot(np.arange(0, Nf * Nt // 2 + 1) / (Nt // 2), np.abs(data_freq) - np.abs(data_freq_rec))
    plt.show()

    plt.plot(np.abs((data_freq - data_freq_rec)[:-1].reshape(Nf, Nt // 2)))
    plt.show()
    # plt.plot((np.abs(data_freq)-np.abs(data_freq_rec))[::Nt//2])
    plt.plot((np.abs(data_freq - data_freq_rec))[: Nt // 2])
    plt.plot((np.abs(data_freq) - np.abs(data_freq_rec))[: Nt // 2])
    plt.show()

    plt.plot((np.real(data_freq))[: Nt // 2])
    plt.plot((np.real(data_freq_rec))[: Nt // 2])
    plt.show()

    # check correlation of streams
    assert_allclose(1.0 - np.corrcoef(np.real(data_freq), np.real(data_freq_rec))[0, 1], 0.0, atol=1.0e-5)
    assert_allclose(1.0 - np.corrcoef(np.imag(data_freq), np.imag(data_freq_rec))[0, 1], 0.0, atol=1.0e-4)
    assert_allclose(1.0 - np.corrcoef(np.abs(data_freq), np.abs(data_freq_rec))[0, 1], 0.0, atol=1.0e-4)
    assert_allclose(1.0 - np.corrcoef(np.angle(data_freq), np.angle(data_freq_rec))[0, 1], 0.0, atol=1.0e-14)

    # check variance preserved for parseval's theorem
    assert_allclose(np.sum(data_wavelet**2), parseval_rfft(data_freq, Nf * Nt), atol=1.0e-100, rtol=1.0e-6)
    assert_allclose(np.sum(data_wavelet**2), parseval_rfft(data_freq_rec, Nf * Nt), atol=1.0e-100, rtol=2.0e-5)
    assert_allclose(
        parseval_rfft(data_freq, Nf * Nt), parseval_rfft(data_freq_rec, Nf * Nt), atol=1.0e-100, rtol=2.0e-5
    )

    # check known variance
    assert_allclose(parseval_rfft(data_freq, Nf * Nt), 2 * scale**2, atol=1.0e-100, rtol=1.0e-3)
    assert_allclose(parseval_rfft(data_freq_rec, Nf * Nt), 2 * scale**2, atol=1.0e-100, rtol=1.0e-3)
    assert_allclose(np.var(data_wavelet), 1.0, atol=1.0e-100, rtol=2.0e-3)

    # check mean preserved
    assert_allclose(
        np.mean(np.real(data_freq)) / scale, np.mean(np.real(data_freq_rec)) / scale, atol=1.0e-100, rtol=2.0e-3
    )
    assert_allclose(
        np.mean(np.imag(data_freq)) / scale, np.mean(np.imag(data_freq_rec)) / scale, atol=1.0e-100, rtol=4.0e-3
    )
    assert_allclose(
        np.mean(np.abs(data_freq)) / scale, np.mean(np.abs(data_freq_rec)) / scale, atol=1.0e-100, rtol=1.0e-7
    )
    assert_allclose(
        np.mean(np.angle(data_freq)) / scale, np.mean(np.angle(data_freq_rec)) / scale, atol=1.0e-13, rtol=1.0e-7
    )

    unit_normal_battery(data_wavelet.flatten())
    unit_normal_battery(np.real(data_freq), mult=scale)
    unit_normal_battery(np.imag(data_freq[1:-1]), mult=scale)
    unit_normal_battery(np.real(data_freq_rec), mult=scale)
    unit_normal_battery(np.imag(data_freq_rec[1:-1]), mult=scale)

    # check no unexpected correlations
    corr_wave = correlate(data_wavelet, data_wavelet, mode='same')
    corr_center = correlate(data_wavelet, data_wavelet, mode='same')[Nt // 2, Nf // 2]
    corr_wave /= corr_center
    corr_wave[Nt // 2, Nf // 2] = 0.0
    assert_allclose(corr_wave, 0.0, atol=4.0e-3)
    assert_allclose(np.mean(corr_wave), 0.0, atol=3.0e-7)
    assert_allclose(np.std(corr_wave) * 4 / 3 * np.sqrt(Nt * Nf), 1.0, atol=3.0e-5)
    assert_allclose(np.mean(corr_wave, axis=0), 0.0, atol=1.0e-4)
    assert_allclose(np.mean(corr_wave, axis=1), 0.0, atol=1.0e-4)

    # plt.plot(data_freq_rec)
    # plt.show()
