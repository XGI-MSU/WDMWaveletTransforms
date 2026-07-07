import numpy as np
from numpy.testing import assert_allclose
from scipy.signal import correlate

from test_mg_time import unit_normal_battery
from WDMWaveletTransforms.wavelet_transforms import (
    inverse_wavelet_freq_time,
    inverse_wavelet_time,
    transform_wavelet_freq_time,
    transform_wavelet_time,
)

# np.seterr(all='raise')

if __name__ == '__main__':
    Nf = 1024
    Nt = 2048
    mult = 64
    mult_f = 4
    gen = np.random.default_rng(314159)

    data_time = gen.normal(0.0, 1.0, Nf * Nt)

    data_wavelet1 = transform_wavelet_time(data_time, Nf, Nt, mult=mult, family='modified_gaussian')
    data_wavelet2 = transform_wavelet_freq_time(data_time, Nf, Nt, mult_f=mult_f, family='modified_gaussian')

    assert_allclose(np.var(data_wavelet1), np.var(data_wavelet2), atol=1.0e-10, rtol=1.0e-3)
    # assert_allclose(np.mean(data_wavelet1), np.mean(data_wavelet2), atol=1.e-10, rtol=1.e-4)

    data_time_rec1 = inverse_wavelet_time(data_wavelet1, Nf, Nt, mult=mult, family='modified_gaussian')
    data_time_rec1_2 = inverse_wavelet_freq_time(data_wavelet1, Nf, Nt, mult_f=mult_f, family='modified_gaussian')
    data_time_rec2 = inverse_wavelet_time(data_wavelet2, Nf, Nt, mult=mult, family='modified_gaussian')
    data_time_rec2_2 = inverse_wavelet_freq_time(data_wavelet2, Nf, Nt, mult_f=mult_f, family='modified_gaussian')
    data_wavelet3 = transform_wavelet_time(data_time_rec1_2, Nf, Nt, mult=mult, family='modified_gaussian')

    # check correlation of streams
    assert_allclose(1.0 - np.corrcoef(data_time, data_time_rec1)[0, 1], 0.0, atol=1.0e-14)
    assert_allclose(1.0 - np.corrcoef(data_time, data_time_rec1_2)[0, 1], 0.0, atol=1.0e-2)
    assert_allclose(1.0 - np.corrcoef(data_time_rec1, data_time_rec1_2)[0, 1], 0.0, atol=1.0e-2)

    assert_allclose(1.0 - np.corrcoef(data_time, data_time_rec2)[0, 1], 0.0, atol=1.0e-2)
    assert_allclose(1.0 - np.corrcoef(data_time, data_time_rec2_2)[0, 1], 0.0, atol=1.0e-4)
    assert_allclose(1.0 - np.corrcoef(data_time_rec2, data_time_rec2_2)[0, 1], 0.0, atol=1.0e-2)

    assert_allclose(1.0 - np.corrcoef(data_time_rec1, data_time_rec2)[0, 1], 0.0, atol=1.0e-2)
    assert_allclose(1.0 - np.corrcoef(data_time_rec1, data_time_rec2_2)[0, 1], 0.0, atol=1.0e-4)
    assert_allclose(1.0 - np.corrcoef(data_time_rec1_2, data_time_rec2_2)[0, 1], 0.0, atol=1.0e-2)
    assert_allclose(1.0 - np.corrcoef(data_time_rec2, data_time_rec1_2)[0, 1], 0.0, atol=3.0e-2)

    # check variance preserved for parseval's theorem
    assert_allclose(np.var(data_wavelet1), np.var(data_time), atol=1.0e-100, rtol=3.0e-6)
    assert_allclose(np.var(data_wavelet1), np.var(data_time_rec1), atol=1.0e-100, rtol=3.0e-6)
    assert_allclose(np.var(data_time), np.var(data_time_rec1), atol=1.0e-100, rtol=1.0e-14)

    assert_allclose(np.var(data_wavelet2), np.var(data_time), atol=1.0e-100, rtol=1.0e-4)
    assert_allclose(np.var(data_wavelet2), np.var(data_time_rec2), atol=1.0e-100, rtol=1.0e-6)
    assert_allclose(np.var(data_time), np.var(data_time_rec2), atol=1.0e-100, rtol=1.0e-4)

    # check mean preserved
    assert_allclose(np.mean(data_time), np.mean(data_time_rec1), atol=1.0e-100, rtol=1.0e-12)
    # assert_allclose(np.mean(data_time), np.mean(data_time_rec1_2), atol=1.e-100, rtol=1.e-13)
    # assert_allclose(np.mean(data_time), np.mean(data_time_rec2_2), atol=1.e-100, rtol=1.e-13)
    # assert_allclose(np.mean(data_time), np.mean(data_time_rec2), atol=1.e-100, rtol=1.e-12)
    # assert_allclose(np.mean(data_time_rec2_2), np.mean(data_time_rec2), atol=1.e-100, rtol=1.e-13)

    unit_normal_battery(data_time)
    unit_normal_battery(data_wavelet1.flatten())
    unit_normal_battery(data_wavelet2.flatten())
    unit_normal_battery(data_time_rec1)
    unit_normal_battery(data_time_rec2)
    unit_normal_battery(data_time_rec2_2)
    unit_normal_battery(data_time_rec1_2)

    # check known variance
    assert_allclose(np.var(data_time), 1.0, atol=1.0e-100, rtol=2.0e-3)
    assert_allclose(np.var(data_time_rec1), 1.0, atol=1.0e-100, rtol=2.0e-3)
    assert_allclose(np.var(data_wavelet1), 1.0, atol=1.0e-100, rtol=2.0e-3)
    assert_allclose(np.var(data_wavelet2), 1.0, atol=1.0e-100, rtol=2.0e-3)
    assert_allclose(np.var(data_time_rec2), 1.0, atol=1.0e-100, rtol=2.0e-3)
    assert_allclose(np.var(data_time_rec2_2), 1.0, atol=1.0e-100, rtol=2.0e-3)
    assert_allclose(np.var(data_time_rec1_2), 1.0, atol=1.0e-100, rtol=2.0e-3)

    print(np.var(data_wavelet1) / np.var(data_time))
    print(np.mean(data_time_rec1 / data_time), np.var(data_time_rec1) / np.var(data_time))

    # check for correlation structure
    corr_wave11 = correlate(data_wavelet1, data_wavelet1, mode='same')
    corr_wave22 = correlate(data_wavelet2, data_wavelet2, mode='same')
    corr_wave12 = correlate(data_wavelet1, data_wavelet2, mode='same')
    corr_center11 = correlate(data_wavelet1, data_wavelet1, mode='same')[Nt // 2, Nf // 2]
    corr_center22 = correlate(data_wavelet2, data_wavelet2, mode='same')[Nt // 2, Nf // 2]
    corr_center12 = correlate(data_wavelet1, data_wavelet2, mode='same')[Nt // 2, Nf // 2]

    assert_allclose(corr_center11, corr_center12, rtol=1.0e-2)
    assert_allclose(corr_center12 / np.sqrt(corr_center11 * corr_center12), 1.0, atol=1.0e-2)

    corr_wave11 /= corr_center11
    corr_wave22 /= corr_center22
    corr_wave12 /= corr_center12

    corr_wave11[Nt // 2, Nf // 2] = 0.0
    corr_wave22[Nt // 2, Nf // 2] = 0.0
    corr_wave12[Nt // 2, Nf // 2] = 0.0

    assert_allclose(corr_wave11, 0.0, atol=1.0e-2)
    assert_allclose(corr_wave22, 0.0, atol=1.0e-2)
    assert_allclose(corr_wave12, 0.0, atol=1.0e-2)

    assert_allclose(np.mean(corr_wave11), 0.0, atol=1.0e-6)
    assert_allclose(np.mean(corr_wave22), 0.0, atol=1.0e-6)
    assert_allclose(np.mean(corr_wave12), 0.0, atol=1.0e-6)

    assert_allclose(np.std(corr_wave11) * 4 / 3 * np.sqrt(Nt * Nf), 1.0, atol=2.0e-3)
    assert_allclose(np.std(corr_wave22) * 4 / 3 * np.sqrt(Nt * Nf), 1.0, atol=2.0e-3)

    assert_allclose(np.mean(corr_wave11, axis=0), 0.0, atol=1.0e-4)
    assert_allclose(np.mean(corr_wave11, axis=1), 0.0, atol=1.0e-4)

    assert_allclose(np.mean(corr_wave22, axis=0), 0.0, atol=1.0e-4)
    assert_allclose(np.mean(corr_wave22, axis=1), 0.0, atol=1.0e-4)

    assert_allclose(corr_wave11, corr_wave22, atol=1.0e-2, rtol=1.0e-10)

# import matplotlib.pyplot as plt
# plt.imshow(np.rot90((data_wavelet1-data_wavelet2)**2))
# plt.show()
# plt.plot(data_time)
# plt.plot(data_time_rec)
# plt.show()
