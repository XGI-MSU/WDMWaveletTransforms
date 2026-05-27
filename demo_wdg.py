from time import perf_counter

import numpy as np

import WDMWaveletTransforms.fft_funcs as fft
import WDMWaveletTransforms.modified_gaussian as mg
from checks2 import phi_eval, phihat_eval
from WDG import meyer_fd, tw_freq, wilson_fd
from WDMWaveletTransforms.transform_freq_funcs import phitilde_vec_norm
from WDMWaveletTransforms.transform_time_funcs import phi_vec
from WDMWaveletTransforms.wavelet_transforms import transform_wavelet_freq_time

# np.seterr(all='raise')

if __name__ == '__main__':
    Nf = 512
    Nt = 1024
    nx = 4.0
    dt = 1.0
    mult = 128
    K = mult * 2 * Nf
    Tobs = Nf * Nt * dt
    DF = 1 / (2 * dt * Nf)
    domega = 2 * np.pi / Tobs
    np.random.seed(314159)
    x = np.random.normal(0.0, 1.0, Nf * Nt)

    phitilde_meyer1 = phitilde_vec_norm(Nf, Nt, nx)
    phitilde_wilson1 = mg.phitilde_vec_norm(Nf, Nt)

    phitilde_meyer2 = np.zeros(Nt // 2 + 1)
    ws = np.arange(0, Nt // 2 + 1) * domega
    for i in range(Nt // 2 + 1):
        phitilde_meyer2[i] = np.sqrt(np.pi / dt) * meyer_fd(domega * i, DF)

    nu = 1.0 / 4.0
    fn = np.linspace(0, 0.5, Nt // 2 + 1, endpoint=False)
    phitilde_wilson2 = np.sqrt(np.pi) * np.sqrt(Nf) / 2 * np.real(wilson_fd(fn / Nt, DF, nu))

    # phitilde_wilson3 = np.sqrt(np.pi/dt)*np.sqrt(dt)/np.sqrt(Nf)*Nf/2*np.real(phihat_eval(2/np.pi*Nf/4*ws*dt))
    phitilde_wilson3 = np.sqrt(np.pi) * np.sqrt(Nf) / 2 * np.real(phihat_eval(fn))
    phitilde_wilson5 = np.sqrt(np.pi) * np.sqrt(Nf) / 2 * np.real(mg.phihat_eval(fn))

    # assert_allclose(phitilde_wilson3, phitilde_wilson5,atol=1.e-20, rtol=1.e-13)

    phi_meyer1 = phi_vec(Nf, nx, mult=mult)
    phi_meyer1_rearrange = np.zeros(K)
    phi_meyer1_rearrange[K // 2 : K] = phi_meyer1[0 : K // 2]
    phi_meyer1_rearrange[0 : K // 2] = phi_meyer1[K // 2 :]

    ts = np.arange(-K // 2, K // 2) / (Nf)

    phi_wilson3 = np.sqrt(np.pi / dt) * np.sqrt(dt) / np.sqrt(Nf) * np.real(phi_eval(ts))

    phi_wilson3_rearrange = np.zeros(K)
    phi_wilson3_rearrange[K // 2 : K] = phi_wilson3[0 : K // 2]
    phi_wilson3_rearrange[0 : K // 2] = phi_wilson3[K // 2 :]

    phi_wilson4 = mg.phi_vec(Nf, mult=mult)
    phi_wilson6 = mg.phi_vec_transform(Nf, mult=mult)

    phi_wilson4_rearrange = np.zeros(K)
    phi_wilson4_rearrange[K // 2 : K] = phi_wilson4[0 : K // 2]
    phi_wilson4_rearrange[0 : K // 2] = phi_wilson4[K // 2 :]

    phi_wilson5 = np.sqrt(2.0 / Nf) * np.real(mg.phi_eval(ts, M=20, N=20, nu=0.5))
    res2 = np.real(phi_eval(ts))
    # assert_allclose(phi_wilson5, phi_wilson4, atol=1.e-14, rtol=1.e-14)
    # assert_allclose(phi_wilson5, res2, atol=1.e-14, rtol=1.e-14)

    print(np.sum(phi_wilson3), np.sum(phi_meyer1), 2 * np.sqrt(Nf), np.sum(phi_wilson3) / np.sum(phi_meyer1))
    print(
        np.sum(phitilde_wilson3) / Nt / np.sqrt(Nf),
        np.sum(phitilde_meyer1) / Nt / np.sqrt(Nf),
        np.sum(phitilde_wilson3) / np.sum(phitilde_meyer1),
    )

    dom = 2 * np.pi / K
    nrm = np.sqrt(K / dom)
    fac = np.sqrt(2.0) / nrm
    phitilde_meyer5_rearrange = np.real(np.sqrt(np.pi) * (1.0 / fac) * 1.0 / K * fft.fft(phi_meyer1_rearrange, K))
    phitilde_meyer5 = np.zeros(K)
    phitilde_meyer5[K // 2 :] = phitilde_meyer5_rearrange[1 : K // 2 + 1]
    phitilde_meyer5[: K // 2 - 1] = phitilde_meyer5_rearrange[K // 2 + 1 :]
    phitilde_meyer5[K // 2 - 1] = phitilde_meyer5_rearrange[0]

    phitilde_wilson4_rearrange = np.real(np.sqrt(np.pi) * (1.0 / fac) * 1.0 / K * fft.fft(phi_wilson3_rearrange, K))
    phitilde_wilson4 = np.zeros(K)
    phitilde_wilson4[K // 2 :] = phitilde_wilson4_rearrange[1 : K // 2 + 1]
    phitilde_wilson4[: K // 2 - 1] = phitilde_wilson4_rearrange[K // 2 + 1 :]
    phitilde_wilson4[K // 2 - 1] = phitilde_wilson4_rearrange[0]

    phitilde_wilson6_rearrange = np.real(np.sqrt(np.pi) * (1.0 / fac) * 1.0 / K * fft.fft(phi_wilson4_rearrange, K))
    phitilde_wilson6 = np.zeros(K)
    phitilde_wilson6[K // 2 :] = phitilde_wilson6_rearrange[1 : K // 2 + 1]
    phitilde_wilson6[: K // 2 - 1] = phitilde_wilson6_rearrange[K // 2 + 1 :]
    phitilde_wilson6[K // 2 - 1] = phitilde_wilson6_rearrange[0]

    wave1_meyer = transform_wavelet_freq_time(x, Nf, Nt, nx=4.0)
    wave2_meyer = tw_freq(x, Nf, Nt, 1.0 / dt, 1.0, window_choice='meyer')

    wave2_wilson = tw_freq(x, Nf, Nt, 1.0 / dt, 1.0, window_choice='wilson')
    print(np.var(wave2_wilson))
    # only the zeroth row is meaningfully different
    # assert_allclose(wave1_meyer[:, 1:], (wave2_meyer.T)[:, 1:], atol=1.e-14, rtol=1.e-14)

    mg._coefficient_recursion_helper(Kmax=40, M=20, N=20, nu=0.5)
    n_run = 50
    t0 = perf_counter()
    for _itrm in range(n_run):
        mg._coefficient_recursion_helper(Kmax=40, M=20, N=20, nu=0.5)
    tf = perf_counter()

    print('coefficient recursion took %8.7f s' % ((tf - t0) / n_run))

    mg.phi_eval(ts, M=20, N=20, nu=0.5)

    n_run = 1
    t0 = perf_counter()
    for _itrm in range(n_run):
        mg.phi_eval(ts, M=20, N=20, nu=0.5)
    tf = perf_counter()

    print('phi eval took %8.7f s' % ((tf - t0) / n_run))

    mg.phihat_eval(fn, M=20, N=20, nu=0.5)
    n_run = 5
    t0 = perf_counter()
    for _itrm in range(n_run):
        mg.phihat_eval(fn, M=20, N=20, nu=0.5)
    tf = perf_counter()

    print('phihat eval took %8.7f s' % ((tf - t0) / n_run))

    import matplotlib.pyplot as plt

    # plt.plot(phi_meyer1)
    # plt.plot(phi_wilson3)
    plt.plot(phi_wilson4)
    plt.plot(phi_wilson6)
    plt.show()

    plt.plot(phitilde_wilson1)
    plt.plot(np.arange(-K // 2, K // 2) / mult * Nt / 2, phitilde_wilson6)
    plt.xlim(0, Nt)
    plt.show()

    plt.plot(phitilde_meyer1)
    plt.plot(phitilde_meyer2)
    # plt.plot(phitilde_wilson2)
    plt.plot(phitilde_wilson1)
    plt.plot(phitilde_wilson3)
    plt.plot(phitilde_wilson5)
    plt.plot(np.arange(-K // 2, K // 2) / mult * Nt / 2, phitilde_meyer5)
    plt.plot(np.arange(-K // 2, K // 2) / mult * Nt / 2, phitilde_wilson4)
    plt.plot(np.arange(-K // 2, K // 2) / mult * Nt / 2, phitilde_wilson6)
    # plt.plot(np.real(phitilde_wilson2))
    # plt.plot(10*np.real(phitilde_wilson3))
    plt.xlim(0, Nt)
    plt.show()

# plt.plot(np.sqrt(2)*np.arange(0, phitilde_wilson3.size), phitilde_wilson3)
# plt.plot(np.pi/2.*np.arange(0, phitilde_wilson3.size), phitilde_wilson3)
# plt.plot(np.arange(-K//2, K//2)/mult*Nt/2, phitilde_wilson4)
# plt.xlim(0, Nt)
# plt.show()

# plt.plot(phitilde_meyer1/phitilde_meyer2)
# plt.show()
