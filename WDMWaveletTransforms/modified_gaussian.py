import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.fft import fftn, ifftn, next_fast_len

import WDMWaveletTransforms.fft_funcs as fft

# ============================================================
# Gaussian and atoms
# ============================================================


@njit()
def g_nu(x, nu):
    return (2 * nu) ** 0.25 * np.exp(-np.pi * nu * x**2)


@njit()
def zak_g(t, s, nu, L=20):
    z = np.zeros_like(s, dtype=np.complex128)
    for k in range(-L, L + 1):
        z += g_nu(t + k, nu) * np.exp(2j * np.pi * k * s)
    return z


@njit()
def zak_g_2filter(t, s, nu, L=20):
    z0 = np.zeros_like(s, dtype=np.complex128)
    z1 = np.zeros_like(s, dtype=np.complex128)
    for k in range(-L, L + 1):
        prod0 = g_nu(t + k, nu) * np.exp(2j * np.pi * k * s)
        prod1 = ((-1) ** (k)) * prod0
        z0 += prod0
        z1 += prod1
    return (z0, z1)


# @njit()
# def zak_g_abs(t, s, nu, L=20):
#    z0 = np.zeros_like(s, dtype=np.complex128)
#    z1 = np.zeros_like(s, dtype=np.complex128)
#    for k in range(-L, L + 1):
#        prod0 = g_nu(t + k, nu) * np.exp(2j * np.pi * k * s)
#        prod1 = ((-1) ** (k)) * prod0
#        z0 += prod0
#        z1 += prod1
#    return np.abs(z0)**2 + np.abs(z1)**2


@njit()
def zak_g_abs2(nu, Nt=100, Ns=100, L=20):
    f_min = np.inf
    f_max = -np.inf
    for itrt in range(Nt):
        t = itrt / Nt
        for itrs in range(Ns):
            s = itrs / Ns
            z0 = 0.0j
            z1 = 0.0j
            for k in range(-L, L + 1):
                prod0_loc = g_nu(t + k, nu) * np.exp(2j * np.pi * k * s)
                prod1_loc = ((-1) ** (k)) * prod0_loc
                # prod1_loc = g_nu(t + k, nu) * np.exp(2j * np.pi * k * (s+0.5))

                z0 += prod0_loc
                z1 += prod1_loc
            f = np.abs(z0) ** 2 + np.abs(z1) ** 2
            if f < f_min:
                f_min = f
            if f > f_max:
                f_max = f
    return f_min, f_max


@njit()
def _zak_load_array_helper(nu, Nt, Ns, L):
    t_vals = np.arange(Nt) / Nt

    C0 = np.zeros((Nt, Ns), dtype=np.complex128)

    for itrt in range(Nt):
        t = itrt / Nt
        for k in range(-L, L + 1):
            km = np.mod(k, Ns)
            C0[itrt, km] += g_nu(t + k, nu)
    return C0


def zak_g_abs2_fft(nu, Nt=100, Ns=100, L=20):
    if Ns % 2 != 0:
        msg = 'Ns must be even'
        raise ValueError(msg)

    C0 = _zak_load_array_helper(nu, Nt, Ns, L)

    z0 = Ns * np.fft.ifft(C0, axis=1)
    z1 = np.roll(z0, shift=-Ns // 2, axis=1)

    f = np.abs(z0) ** 2 + np.abs(z1) ** 2

    return f.min(), f.max()


@njit()
def frame_bounds(nu, Nt=100, Ns=100, L=20):
    # ts = np.linspace(0, 1, Nt+1)[:Nt]
    ss = np.linspace(0, 1, Ns + 1)[:Ns]

    A = np.inf
    B = -np.inf

    t_min = -np.inf
    s_min = -np.inf
    t_max = -np.inf
    s_max = -np.inf

    for itrt in range(Nt):
        t = itrt / Nt

        # Z0 = zak_g(t, ss, nu, L=L)
        # Z1 = zak_g(t, ss+0.5, nu, L=L)
        # F = np.abs(Z0)**2 + np.abs(Z1)**2

        Z0, Z1 = zak_g_2filter(t, ss, nu, L=L)
        F = np.abs(Z0) ** 2 + np.abs(Z1) ** 2

        # Z = zak_g(t, np.array([ss, ss+0.5]), nu, L=L)
        # Z0 = Z[0]
        # Z1 = Z[1]
        # F = np.abs(Z0)**2 + np.abs(Z1)**2

        # F = zak_g_abs(t, ss, nu, L=L)

        j_min = np.argmin(F)
        j_max = np.argmax(F)
        f_min = F[j_min]
        f_max = F[j_max]

        # f_min, f_max, j_min, j_max = zak_g_abs2(t, Ns, nu, L=L)

        if f_min < A:
            A = f_min
            t_min = t
            # s_min = ss[j_min]
            s_min = j_min / Ns

        if f_max > B:
            B = f_max
            t_max = t
            # s_max = ss[j_max]
            s_max = j_max / Ns

    # print("A_nu =", A)
    # print("  attained near t =", t_min)
    # print("  attained near s =", s_min)

    # print("B_nu =", B)
    # print("  attained near t =", t_max)
    # print("  attained near s =", s_max)

    # print("(B_nu - A_nu)/(B_nu + A_nu) =", (B - A) / (B + A))

    return A, B, t_min, s_min, t_max, s_max


# @njit()
def frame_bounds_short(nu, Nt=100, Ns=100, L=20):
    return zak_g_abs2_fft(nu, Nt=Nt, Ns=Ns, L=L)


@njit()
def _omega_array_loop(M, N, nu):
    n_pair = (2 * M + 1) * (2 * N + 1)
    Omega = np.zeros((n_pair, n_pair), dtype=np.complex128)
    itrp1 = 0
    n_scales = np.zeros(2 * N + 1)
    m_scales = np.zeros(2 * M + 1)
    # calculate the scaling parts for difference in m and n
    for dn in range(2 * N + 1):
        n_scales[dn] = np.exp(-np.pi * nu * dn**2 / 2)
    for dm in range(2 * M + 1):
        m_scales[dm] = np.exp(-np.pi * dm**2 / (8 * nu))

    # moduli = np.outer(n_scales, m_scales)

    for m1 in range(-M, M + 1):
        for n1 in range(-N, N + 1):
            itrp2 = 0
            for m2 in range(-M, M + 1):
                dm = abs(m1 - m2)
                for n2 in range(-N, N + 1):
                    dn = abs(n1 - n2)
                    if itrp2 >= itrp1:
                        # real_part_arg = - np.pi * nu * (n1 - n2)**2 / 2 - np.pi * (m1 - m2)**2 / (8 * nu)
                        # modulus = np.exp(real_part_arg)
                        modulus = n_scales[dn] * m_scales[dm]
                        # modulus = moduli[abs(n1-n2), abs(m1-m2)]
                        # get the complex part of the array
                        imag_part_arg = 1j ** np.mod(dm * (n1 + n2), 4)
                        Omega[itrp1, itrp2] = modulus * imag_part_arg
                        if itrp2 > itrp1:
                            # take advantage of the fact the array is hermitian
                            Omega[itrp2, itrp1] = modulus * np.conjugate(imag_part_arg)
                    itrp2 += 1
            itrp1 += 1
    return Omega


def _omega_array_loop_wrapper(M, N, nu):
    return _omega_array_loop(M, N, nu)


# @njit()
# def _om_dot_helper(M, N, b, m_scales, n_scales):
#    n_pair = (2*M+1)*(2*N+1)
#    res = np.zeros(n_pair, dtype=np.complex128)
#
#    itrp1 = 0
#    for m1 in range(-M, M+1):
#        for n1 in range(-N, N+1):
#            itrp2 = 0
#            for m2 in range(-M, M+1):
#                dm = abs(m1 - m2)
#                for n2 in range(-N, N+1):
#                    dn = abs(n1 - n2)
#                    if itrp2 >= itrp1:
#                        modulus = n_scales[dn] * m_scales[dm]
#                        imag_part_arg = 1j ** np.mod(dm * (n1 + n2), 4)
#                        res[itrp1] += modulus * imag_part_arg * b[itrp2]
#                        if itrp2 > itrp1:
#                            # take advantage of the fact the array is hermitian
#                            res[itrp2] += modulus * np.conjugate(imag_part_arg) * b[itrp1]
#
#                        #assert_allclose(modulus, np.abs(Omega[itrp1, itrp2]), atol=1.e-100, rtol=1.e-14)
#                    itrp2 += 1
#            itrp1 += 1
#    return res


def _next_fast_even(n):
    n = int(n)
    m = next_fast_len(n)
    while m % 2:
        m = next_fast_len(m + 1)
    return m


def _om_dot_helper(M, N, b, F_even, F_odd, fft_shape):
    crop_m = 2 * M
    crop_n = 2 * N

    n_parity_sign = 1.0 if (N % 2 == 0) else -1.0

    B = b.reshape(2 * M + 1, 2 * N + 1)

    FB = fftn(B, fft_shape)

    F0 = FB * F_even
    F1 = FB * F_odd

    F_selected = 0.5 * (F0 + F1) + 0.5 * n_parity_sign * np.roll(F0 - F1, fft_shape[1] // 2, axis=1)

    convolved = ifftn(F_selected, fft_shape)
    Y = convolved[crop_m : crop_m + 2 * M + 1, crop_n : crop_n + 2 * N + 1]

    # Y_even = fftconvolve(B, kernel_even, mode="same")
    # Y_odd = fftconvolve(B, kernel_odd, mode="same")

    # Y = Y_even.copy()
    # if N % 2 == 0:
    #    Y[:, 1::2] = Y_odd[:, 1::2]
    # else:
    #    Y[:, ::2] = Y_odd[:, ::2]
    # import matplotlib.pyplot as plt
    # plt.plot(np.real(Y_alt))
    # plt.plot(np.real(Y))
    # plt.show()
    # assert_allclose(Y_alt, Y, atol=1.e-100, rtol=1.e-14)

    return Y.reshape(-1)


@njit()
def _kernel_helper(M, N, nu):
    n_scales = np.zeros(2 * N + 1)
    m_scales = np.zeros(2 * M + 1)

    # calculate the scaling parts for difference in m and n
    for dn in range(2 * N + 1):
        n_scales[dn] = np.exp(-np.pi * nu * dn**2 / 2)
    for dm in range(2 * M + 1):
        m_scales[dm] = np.exp(-np.pi * dm**2 / (8 * nu))

    kernel_even = np.zeros((4 * M + 1, 4 * N + 1), dtype=np.complex128)
    # kernel_odd = np.zeros((4*M+1, 4*N+1), dtype=np.complex128)
    for dm in range(-2 * M, 2 * M + 1):
        for dn in range(-2 * N, 2 * N + 1):
            modulus_loc = m_scales[np.abs(dm)] * n_scales[np.abs(dn)]
            phase_even_loc = 1j ** np.mod(dm * dn, 4)
            # phase_odd_loc = (-1)**(dm+2*M) * phase_even_loc
            kernel_even[dm + 2 * M, dn + 2 * N] = modulus_loc * phase_even_loc
            # kernel_odd[dm+2*M, dn+2*N]= modulus_loc * phase_odd_loc
    return kernel_even  # , kernel_odd


# @njit()
def _recursive_loop(Omega, Kmax, center_index, alpha, M, N, nu):
    # ============================================================
    # Coefficient recursion from equations (6.2)--(6.3)
    # ============================================================
    # n_pair = Omega.shape[0]
    n_pair = (2 * M + 1) * (2 * N + 1)

    b = np.zeros(n_pair, dtype=np.complex128)
    b[center_index] = 1.0

    a = np.zeros(n_pair, dtype=np.complex128)
    comp = np.zeros(n_pair, dtype=np.complex128)

    kernel_even = _kernel_helper(M, N, nu)

    fft_shape = (
        _next_fast_even(2 * M + 1 + kernel_even.shape[0] - 1),
        _next_fast_even(2 * N + 1 + kernel_even.shape[1] - 1),
    )
    F_even = fftn(kernel_even, fft_shape)
    F_odd = np.roll(F_even, fft_shape[0] // 2, axis=0)

    c_k = 1.0

    for k in range(Kmax + 1):
        # a += c_k * b
        # accumulate a by Kahan compensated summation algorithm to reduce loss of numerical precision
        term = c_k * b
        y = term - comp
        t = a + y
        comp = (t - a) - y
        a = t

        # Equivalent to applying [I - 2P/(A+B)]
        # mat_prod_alt = np.dot(Omega, b)
        mat_prod = _om_dot_helper(M, N, b, F_even, F_odd, fft_shape)
        # assert_allclose(mat_prod, mat_prod_alt, atol=1.e-15, rtol=1.e-14)
        b = b - alpha * mat_prod
        c_k *= (2 * k + 1) / (2 * k + 2)
    return a


def _recursive_loop_wrapper(Omega, Kmax, center_index, alpha, M, N, nu):
    return _recursive_loop(Omega, Kmax, center_index, alpha, M, N, nu)


def _coefficient_recursion_helper(Kmax, M, N, nu, Nt=100, Ns=100, L=20):
    # This computation is slow, so use pre-computed values
    A, B = frame_bounds_short(nu, Nt=Nt, Ns=Ns, L=L)

    A_nu = A
    B_nu = B

    alpha = 2.0 / (A_nu + B_nu)

    # ============================================================
    # Index lattice
    # ============================================================
    prefactor = 2.0 * np.sqrt(1.0 / (A_nu + B_nu))

    n_pair = (2 * M + 1) * (2 * N + 1)

    m_arr = np.repeat(np.arange(-M, M + 1), 2 * N + 1)
    n_arr = np.tile(np.arange(-N, N + 1), 2 * M + 1)
    center_index = M * (2 * N + 1) + N

    assert m_arr.shape == n_arr.shape
    assert m_arr.shape == (n_pair,)
    assert m_arr[center_index] == 0
    assert n_arr[center_index] == 0

    # ============================================================
    # Overlap matrix
    # ============================================================
    # m = m_arr[:, None]
    # n = n_arr[:, None]
    # Omega = _omega_array_loop_wrapper(M, N, nu)
    Omega = 0.0
    # Omega = np.exp(
    #    1j * np.pi * (m.T - m) * (n + n.T) / 2
    #    - np.pi * nu * (n - n.T)**2 / 2
    #    - np.pi * (m - m.T)**2 / (8 * nu)
    # )
    # assert_allclose(Omega, Omega_alt, atol=1.e-100, rtol=1.e-13)

    a = _recursive_loop_wrapper(Omega, Kmax, center_index, alpha, M, N, nu)

    return a, m_arr, n_arr, prefactor


nu_init = 0.5
M_init = 20
N_init = 80
L_init = 20
Ns_init = 200
Nt_init = 200
a, m_arr, n_arr, prefactor = _coefficient_recursion_helper(
    Kmax=240, M=M_init, N=N_init, nu=nu_init, L=L_init, Ns=Ns_init, Nt=Nt_init
)


# ============================================================
# Evaluation functions
# ============================================================

# TODO put back prefactor


@njit()
def g_mn_x(x, m, n, nu):
    return np.exp(1j * np.pi * m * x) * g_nu(x - n, nu)


@njit()
def phi_vec(Nf, mult, M=M_init, N=N_init, nu=nu_init):
    assert mult % 2 == 0
    assert Nf % 2 == 0
    K = mult * 2 * Nf
    ts = np.arange(-K // 2, K // 2) / Nf
    out = np.zeros(K, dtype=np.complex128)

    itrp = 0
    for m in range(-M, M + 1):
        for n in range(-N, N + 1):
            coeff = a[itrp]
            out += coeff * g_mn_x(ts, m, n, nu_init)
            itrp += 1

    return prefactor * np.sqrt(2.0 / Nf) * np.real(out)


# @njit()
# def phihat_eval(yvals):
#    yvals = np.asarray(yvals)
#    out = np.zeros_like(yvals, dtype=np.complex128)
#
#    for coeff, m, n in zip(a, m_arr, n_arr):
#        out += coeff * g_mn_hat(yvals, m, n, nu_init)
#
#    return prefactor * out


@njit()
def g_mn_hat(y, m, n, nu):
    sign = 1.0 if (m * n) % 2 == 0 else -1.0
    return sign * np.exp(2j * np.pi * y * n) * g_nu(y + m / 2, 1 / nu)


@njit()
def phihat_eval(yvals, M=M_init, N=N_init, nu=nu_init):
    yvals = np.asarray(yvals)
    out = np.zeros_like(yvals, dtype=np.complex128)

    itrp = 0
    for m in range(-M, M + 1):
        for n in range(-N, N + 1):
            coeff = a[itrp]
            out += coeff * g_mn_hat(yvals, m, n, nu_init)
            itrp += 1

    return prefactor * np.real(out)


def phi_vec_transform(Nf: int, mult: int = 16) -> NDArray[np.floating]:
    """Get time domain phi as fourier transform of phitilde_vec"""
    # TODO fix mult

    OM: float = np.pi
    DOM = float(OM / Nf)
    insDOM: float = float(1.0 / np.sqrt(DOM))
    K: int = int(mult * 2 * Nf)
    half_K: int = int(mult * Nf)  # np.int64(K/2)

    dom: float = 2 * np.pi / K  # max frequency is K/2*dom = pi/dt = OM

    phitilde_loc = np.zeros(K, dtype=complex)

    # zero frequency
    phitilde_loc[0] = phihat_eval(0.0)

    # postive frequencies
    phitilde_loc[1 : half_K + 1] = phihat_eval(np.arange(1, half_K + 1) / half_K * Nf / 2)
    # negative frequencies
    phitilde_loc[half_K + 1 :] = phihat_eval(-np.arange(half_K - 1, 0, -1) / half_K * Nf / 2)
    phi_loc = K * fft.ifft(phitilde_loc, K)

    del phitilde_loc

    phi = np.zeros(K, dtype=float)
    phi[0:half_K] = np.real(phi_loc[half_K:K])
    phi[half_K:] = np.real(phi_loc[0:half_K])

    nrm: float = float(np.sqrt(K / dom))  # *np.linalg.norm(phi)

    fac: float = float(float(np.sqrt(2.0)) / nrm)
    return phi / np.sqrt(2.0 / Nf) / np.sqrt(np.pi) * fac


def phitilde_vec_norm(Nf: int, Nt: int) -> NDArray[np.floating]:
    """Normalize phitilde as needed for inverse frequency domain transform"""
    ND: int = Nf * Nt
    # oms: NDArray[np.floating] = np.asarray(2 * np.pi / ND * np.arange(0, Nt // 2 + 1), dtype=float)
    fn = np.linspace(0, 0.5, Nt // 2 + 1, endpoint=False)
    phif: NDArray[np.floating] = np.sqrt(np.pi) * np.sqrt(Nf) / 2 * phihat_eval(fn)
    # nrm should be 1
    nrm: float = float(
        np.sqrt((2 * np.sum(phif[1:] ** 2) + phif[0] ** 2) * 2 * np.pi / ND) / (np.pi ** (3 / 2) / np.pi),
    )
    return phif / nrm


@njit()
def phi_eval(xvals, M=M_init, N=N_init, nu=nu_init):
    xvals = np.asarray(xvals)
    out = np.zeros_like(xvals, dtype=np.complex128)

    itrp = 0
    for m in range(-M, M + 1):
        for n in range(-N, N + 1):
            coeff = a[itrp]
            out += coeff * g_mn_x(xvals, m, n, nu_init)
            itrp += 1

    return prefactor * out
