import numpy as np
import scipy.special
from numba import njit
from numpy.typing import NDArray
from scipy.fft import fftn, ifftn, next_fast_len
from scipy.signal import fftconvolve

import WDMWaveletTransforms.fft_funcs as fft

# ============================================================
# Gaussian and atoms
# ============================================================


@njit()
def g_nu(x, nu: float):
    return np.sqrt(np.sqrt(2 * nu)) * np.exp(-np.pi * nu * x**2)


@njit()
def zak_g(t, s, nu: float, L: int = 20):
    z = np.zeros_like(s, dtype=np.complex128)
    for k in range(-L, L + 1):
        z += g_nu(t + k, nu) * np.exp(2j * np.pi * k * s)
    return z


@njit()
def zak_g_2filter(t, s, nu: float, L: int = 20):
    z0 = np.zeros_like(s, dtype=np.complex128)
    z1 = np.zeros_like(s, dtype=np.complex128)
    for k in range(-L, L + 1):
        prod0 = g_nu(t + k, nu) * np.exp(2j * np.pi * k * s)
        prod1 = ((-1) ** (k)) * prod0
        z0 += prod0
        z1 += prod1
    return (z0, z1)


@njit()
def zak_g_abs2(nu: float, Nt: int = 100, Ns: int = 100, L: int = 20) -> tuple[float, float]:
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
def _zak_load_array_helper(nu: float, Nt: int, Ns: int, L: int) -> NDArray[np.complex128]:
    t_vals = np.arange(Nt) / Nt

    C0 = np.zeros((Nt, Ns), dtype=np.complex128)

    for itrt in range(Nt):
        t = itrt / Nt
        for k in range(-L, L + 1):
            km = np.mod(k, Ns)
            C0[itrt, km] += g_nu(t + k, nu)
    return C0


def zak_g_abs2_fft(nu: float, Nt: int = 100, Ns: int = 100, L: int = 20) -> tuple[float, float]:
    if Ns % 2 != 0:
        msg = 'Ns must be even'
        raise ValueError(msg)

    C0 = _zak_load_array_helper(nu, Nt, Ns, L)

    z0 = Ns * np.fft.ifft(C0, axis=1)
    z1 = np.roll(z0, shift=-Ns // 2, axis=1)

    f = np.abs(z0) ** 2 + np.abs(z1) ** 2

    return f.min(), f.max()


@njit()
def frame_bounds(nu: float, Nt: int = 100, Ns: int = 100, L: int = 20):
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
def frame_bounds_short(nu: float, Nt: int = 100, Ns: int = 100, L: int = 20, fft_mode: int = 0) -> tuple[float, float]:
    if fft_mode == 0:
        return zak_g_abs2(nu, Nt=Nt, Ns=Ns, L=L)
    if fft_mode == 1:
        return zak_g_abs2_fft(nu, Nt=Nt, Ns=Ns, L=L)
    msg = 'fft_mode must be 0 or 1'
    raise ValueError(msg)


@njit()
def _omega_array_loop(M: int, N: int, nu: float) -> NDArray[np.complex128]:
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


def _omega_array_loop_wrapper(M: int, N: int, nu: float) -> NDArray[np.complex128]:
    return _omega_array_loop(M, N, nu)


@njit()
def _om_dot_helper_old(
    M: int, N: int, b, m_scales: NDArray[np.float64], n_scales: NDArray[np.float64]
) -> NDArray[np.complex128]:
    n_pair = (2 * M + 1) * (2 * N + 1)
    res = np.zeros(n_pair, dtype=np.complex128)

    itrp1 = 0
    for m1 in range(-M, M + 1):
        for n1 in range(-N, N + 1):
            itrp2 = 0
            for m2 in range(-M, M + 1):
                dm = abs(m1 - m2)
                if m_scales[dm] == 0.0:
                    # short circuit cases where the scaling is zero
                    itrp2 += 2 * N + 1
                    continue
                for n2 in range(-N, N + 1):
                    dn = abs(n1 - n2)
                    if n_scales[dn] == 0.0:
                        # short circuit cases where the scaling is zero
                        pass
                    elif itrp2 >= itrp1:
                        modulus = n_scales[dn] * m_scales[dm]
                        imag_part_arg = 1j ** np.mod(dm * (n1 + n2), 4)
                        res[itrp1] += modulus * imag_part_arg * b[itrp2]
                        if itrp2 > itrp1:
                            # take advantage of the fact the array is hermitian
                            res[itrp2] += modulus * np.conjugate(imag_part_arg) * b[itrp1]

                        # assert_allclose(modulus, np.abs(Omega[itrp1, itrp2]), atol=1.e-100, rtol=1.e-14)
                    itrp2 += 1
            itrp1 += 1
    return res


def _next_fast_even(n: int) -> int:
    n = int(n)
    n_new = next_fast_len(n)
    while n_new % 2:
        n_new = next_fast_len(n_new + 1)
    return n_new


def _om_dot_helper(
    M: int, N: int, b, F_even: NDArray[np.complex128], F_odd: NDArray[np.complex128], fft_shape: tuple[int, int]
) -> NDArray[np.complex128]:
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

    return Y.reshape(-1)


def _om_dot_helper_convolve(M: int, N: int, b, kernel_even, kernel_odd) -> NDArray[np.complex128]:
    B = b.reshape(2 * M + 1, 2 * N + 1)

    Y_even = fftconvolve(B, kernel_even, mode='same')
    Y_odd = fftconvolve(B, kernel_odd, mode='same')

    Y = Y_even.copy()
    if N % 2 == 0:
        Y[:, 1::2] = Y_odd[:, 1::2]
    else:
        Y[:, ::2] = Y_odd[:, ::2]

    return Y.reshape(-1)


@njit()
def _om_dot_helper_explicit(M: int, N: int, b, kernel_even) -> NDArray[np.complex128]:
    B = b.reshape(2 * M + 1, 2 * N + 1)

    Y = np.zeros_like(B, dtype=np.complex128)

    # Explicit 2D convolution

    for m1 in range(-M, M + 1):
        for n1 in range(-N, N + 1):
            sum_loc = 0.0j
            for m2 in range(-M, M + 1):
                k_dm = 2 * M + m1 - m2
                if kernel_even[k_dm, 2 * N] == 0.0:
                    # short circuit if this entire component is going to be zero (kernel always maximized at dn=0 by construction)
                    continue
                for n2 in range(-N, N + 1):
                    k_dn = 2 * N + n1 - n2
                    x = B[m2 + M, n2 + N]
                    if (n1 % 2 == 0) or ((m1 - m2) % 2 == 0):
                        sum_loc += x * kernel_even[k_dm, k_dn]
                    else:
                        # use known symmetry to avoid needing kernel_odd
                        sum_loc += -x * kernel_even[k_dm, k_dn]

            Y[m1 + M, n1 + N] = sum_loc

    return Y.reshape(-1)


@njit()
def _kernel_helper(M: int, N: int, nu: float):
    n_scales = np.zeros(2 * N + 1)
    m_scales = np.zeros(2 * M + 1)

    # calculate the scaling parts for difference in m and n
    for dn in range(2 * N + 1):
        n_scales[dn] = np.exp(-np.pi * nu * dn**2 / 2)
    for dm in range(2 * M + 1):
        m_scales[dm] = np.exp(-np.pi * dm**2 / (8 * nu))

    kernel_even = np.zeros((4 * M + 1, 4 * N + 1), dtype=np.complex128)
    kernel_odd = np.zeros((4 * M + 1, 4 * N + 1), dtype=np.complex128)
    for dm in range(-2 * M, 2 * M + 1):
        for dn in range(-2 * N, 2 * N + 1):
            modulus_loc = m_scales[np.abs(dm)] * n_scales[np.abs(dn)]
            phase_even_loc = 1j ** np.mod(dm * dn, 4)
            phase_odd_loc = (-1) ** dm * phase_even_loc
            kernel_even[dm + 2 * M, dn + 2 * N] = modulus_loc * phase_even_loc
            kernel_odd[dm + 2 * M, dn + 2 * N] = modulus_loc * phase_odd_loc
    return kernel_even, kernel_odd, m_scales, n_scales


# @njit()
def _recursive_loop(
    Omega,
    Kmax: int,
    center_index: int,
    alpha: float,
    M: int,
    N: int,
    nu: float,
    fft_mode: int,
    compensate_mode: int,
    null_mode: int,
):
    # ============================================================
    # Coefficient recursion from equations (6.2)--(6.3)
    # ============================================================
    assert compensate_mode in (0, 1)
    assert fft_mode in (0, 1, 2, 3)
    assert null_mode in (0, 1, 2)
    n_pair = (2 * M + 1) * (2 * N + 1)

    b = np.zeros(n_pair, dtype=np.complex128)
    b[center_index] = 1.0
    b_null = b.copy()

    a = np.zeros(n_pair, dtype=np.complex128)
    comp = np.zeros(n_pair, dtype=np.complex128)

    kernel_even, kernel_odd, m_scales, n_scales = _kernel_helper(M, N, nu)

    fft_shape = (
        _next_fast_even(2 * M + 1 + kernel_even.shape[0] - 1),
        _next_fast_even(2 * N + 1 + kernel_even.shape[1] - 1),
    )
    F_even = fftn(kernel_even, fft_shape)
    F_odd = np.roll(F_even, fft_shape[0] // 2, axis=0)

    c_k = 1.0
    # Omega = _omega_array_loop_wrapper(M, N, nu)
    # eig_omega = np.linalg.eigh(Omega)

    if null_mode in (0, 1):
        # get the component of b pointing into a (nearly) null eigenspace so that we can remove it from the actual iterative accumulation
        for k in range(Kmax + 1):
            # Equivalent to applying [I - 2P/(A+B)]
            # mat_prod_alt = np.dot(Omega, b)
            if fft_mode == 0:
                mat_prod = _om_dot_helper(M, N, b_null, F_even, F_odd, fft_shape)
            elif fft_mode == 1:
                mat_prod = _om_dot_helper_convolve(M, N, b_null, kernel_even, kernel_odd)
            elif fft_mode == 2:
                mat_prod = _om_dot_helper_explicit(M, N, b_null, kernel_even)
            elif fft_mode == 3:
                mat_prod = _om_dot_helper_old(M, N, b_null, m_scales, n_scales)
            else:
                msg = 'Unrecogized option for fft_mode'
                raise ValueError(msg)
            b_null = b_null - alpha * mat_prod

        # remove the component of b in the nearly null eigenspace
        if null_mode == 0:
            b = b - b_null
            b_null[:] = 0.0
        elif null_mode == 1:
            # subtract off the null part in the iteration, also works but maybe not as accurate
            pass
    elif null_mode == 2:
        # no null handling
        b_null[:] = 0.0
    else:
        msg = 'Unrecgonized option for null_mode'
        raise ValueError(msg)

    for k in range(Kmax + 1):
        if compensate_mode == 0:
            a += c_k * (b - b_null)
        elif compensate_mode == 1:
            # accumulate a by Kahan compensated summation algorithm to reduce loss of numerical precision
            term = c_k * (b - b_null)
            # term = c_k * b
            y = term - comp
            t = a + y
            comp = (t - a) - y
            a = t
        else:
            msg = 'Unrecognized option for compensate_mode'
            raise ValueError(msg)

        # Equivalent to applying [I - 2P/(A+B)]
        # mat_prod_alt = np.dot(Omega, b)
        if fft_mode == 0:
            mat_prod = _om_dot_helper(M, N, b, F_even, F_odd, fft_shape)
        elif fft_mode == 1:
            mat_prod = _om_dot_helper_convolve(M, N, b, kernel_even, kernel_odd)
        elif fft_mode == 2:
            mat_prod = _om_dot_helper_explicit(M, N, b, kernel_even)
        elif fft_mode == 3:
            mat_prod = _om_dot_helper_old(M, N, b, m_scales, n_scales)
        else:
            msg = 'Unrecogized option for fft_mode'
            raise ValueError(msg)
        b = b - alpha * mat_prod
        # c_k *= (2 * k + 1) / (2 * k + 2)
        # use exact form of c_k to try to reduce accumulated numerical error
        c_k = scipy.special.beta(1.5 + k, 0.5) / np.pi
    return a


def _recursive_loop_wrapper(
    Omega,
    Kmax: int,
    center_index: int,
    alpha: float,
    M: int,
    N: int,
    nu: float,
    fft_mode: int,
    compensate_mode: int,
    null_mode: int,
):
    return _recursive_loop(
        Omega,
        Kmax,
        center_index,
        alpha,
        M,
        N,
        nu,
        fft_mode=fft_mode,
        compensate_mode=compensate_mode,
        null_mode=null_mode,
    )


def _coefficient_recursion_helper(
    Kmax: int,
    M: int,
    N: int,
    nu: float,
    Nt: int = 100,
    Ns: int = 100,
    L: int = 20,
    fft_mode_frame: int = 0,
    fft_mode_recurse: int = 0,
    compensate_mode: int = 1,
    null_mode=0,
):
    A, B = frame_bounds_short(nu, Nt=Nt, Ns=Ns, L=L, fft_mode=fft_mode_frame)

    A_nu = A
    B_nu = B

    alpha = 2.0 / (A_nu + B_nu)

    # ============================================================
    # Index lattice
    # ============================================================
    prefactor = 2.0 * np.sqrt(1.0 / (A_nu + B_nu))

    n_pair = (2 * M + 1) * (2 * N + 1)

    center_index = M * (2 * N + 1) + N

    # ============================================================
    # Overlap matrix
    # ============================================================
    Omega = 0.0
    # Omega = np.exp(
    #    1j * np.pi * (m.T - m) * (n + n.T) / 2
    #    - np.pi * nu * (n - n.T)**2 / 2
    #    - np.pi * (m - m.T)**2 / (8 * nu)
    # )
    # assert_allclose(Omega, Omega_alt, atol=1.e-100, rtol=1.e-13)

    a = _recursive_loop_wrapper(
        Omega,
        Kmax,
        center_index,
        alpha,
        M,
        N,
        nu,
        fft_mode=fft_mode_recurse,
        compensate_mode=compensate_mode,
        null_mode=null_mode,
    )

    return a, prefactor


# nu_init = 0.5
# M_init = 20
# N_init = 80
# L_init = 20
# Ns_init = 200
# Nt_init = 200
# Kmax_init = 240

nu_init = 0.5
M_init = 20
N_init = 80
L_init = 6  # at 64 bit resolution, constant starting with L=5 for nu=0.5
Ns_init = 8  # at nu=0.5, constant for positive Ns divisible by 4
Nt_init = 8  # with nu=0.5, constant for even positive Nt
Kmax_init = 50
fft_mode_frame_init = 0
fft_mode_recurse_init = 2
compensate_mode_init = 1
a, prefactor = _coefficient_recursion_helper(
    Kmax=Kmax_init,
    M=M_init,
    N=N_init,
    nu=nu_init,
    L=L_init,
    Ns=Ns_init,
    Nt=Nt_init,
    fft_mode_frame=fft_mode_frame_init,
    fft_mode_recurse=fft_mode_recurse_init,
    compensate_mode=compensate_mode_init,
)

a_init = a.reshape(((2 * M_init + 1), (2 * N_init + 1)))
# a_init = None


# ============================================================
# Evaluation functions
# ============================================================


@njit()
def g_mn_x(x: NDArray[np.float64], m: int, n: int, nu: float) -> NDArray[np.complex128]:
    return np.exp(1j * np.pi * m * x) * g_nu(x - n, nu)


@njit()
def phi_vec(
    Nf: int, mult_t: int, M: int = M_init, N: int = N_init, nu: float = nu_init, a_in=a_init
) -> NDArray[np.float64]:
    assert mult_t % 2 == 0
    assert Nf % 2 == 0
    assert a_in.shape == (2 * M + 1, 2 * N + 1)
    K = mult_t * 2 * Nf
    ts = np.arange(-K // 2, K // 2) / Nf
    out = np.zeros(K, dtype=np.complex128)

    itrp = 0
    for m in range(-M, M + 1):
        for n in range(-N, N + 1):
            coeff = a_in[m + M, n + N]
            out += coeff * g_mn_x(ts, m, n, nu_init)
            itrp += 1

    return prefactor * np.sqrt(2.0 / Nf) * np.real(out)


@njit()
def g_mn_hat(y: NDArray[np.float64], m: int, n: int, nu: float) -> NDArray[np.complex128]:
    sign = 1.0 if (m * n) % 2 == 0 else -1.0
    return sign * np.exp(2j * np.pi * y * n) * g_nu(y + m / 2, 1 / nu)


@njit()
def phihat_eval(yvals: NDArray[np.float64], M: int = M_init, N: int = N_init, nu: float = nu_init, a_in=a_init):
    assert a_in.shape == (2 * M + 1, 2 * N + 1)

    yvals = np.asarray(yvals)
    out = np.zeros_like(yvals, dtype=np.complex128)

    itrp = 0
    for m in range(-M, M + 1):
        for n in range(-N, N + 1):
            coeff = a_in[m + M, n + N]
            out += coeff * g_mn_hat(yvals, m, n, nu_init)
            itrp += 1

    return prefactor * np.real(out)


def phi_vec_transform(
    Nf: int, mult_t: int = 16, M: int = M_init, N: int = N_init, nu: float = nu_init, a_in=a_init
) -> NDArray[np.floating]:
    """Get time domain phi as fourier transform of phitilde_vec"""
    # TODO fix mult

    OM: float = np.pi
    DOM = float(OM / Nf)
    insDOM: float = float(1.0 / np.sqrt(DOM))
    K: int = int(mult_t * 2 * Nf)
    half_K: int = int(mult_t * Nf)  # np.int64(K/2)

    dom: float = 2 * np.pi / K  # max frequency is K/2*dom = pi/dt = OM

    phitilde_loc = np.zeros(K, dtype=complex)

    # zero frequency
    phitilde_loc[0] = phihat_eval(0.0, M=M, N=N, nu=nu, a_in=a_in)

    # postive frequencies
    phitilde_loc[1 : half_K + 1] = phihat_eval(np.arange(1, half_K + 1) / half_K * Nf / 2, M=M, N=N, nu=nu, a_in=a_in)
    # negative frequencies
    phitilde_loc[half_K + 1 :] = phihat_eval(
        -np.arange(half_K - 1, 0, -1) / half_K * Nf / 2, M=M, N=N, nu=nu, a_in=a_in
    )
    phi_loc = K * fft.ifft(phitilde_loc, K)

    del phitilde_loc

    phi = np.zeros(K, dtype=float)
    phi[0:half_K] = np.real(phi_loc[half_K:K])
    phi[half_K:] = np.real(phi_loc[0:half_K])

    nrm: float = float(np.sqrt(K / dom))  # *np.linalg.norm(phi)

    fac: float = float(float(np.sqrt(2.0)) / nrm)
    return phi / np.sqrt(2.0 / Nf) / np.sqrt(np.pi) * fac


def phitilde_vec_norm(
    Nf: int, Nt: int, mult_f: int = 1, M: int = M_init, N: int = N_init, nu: float = nu_init, a_in=a_init
) -> NDArray[np.floating]:
    """Normalize phitilde as needed for inverse frequency domain transform"""
    ND: int = Nf * Nt
    # oms: NDArray[np.floating] = np.asarray(2 * np.pi / ND * np.arange(0, Nt // 2 + 1), dtype=float)
    fn = np.linspace(0, 0.5 * mult_f, mult_f * Nt // 2 + 1, endpoint=False)
    phif: NDArray[np.floating] = np.sqrt(np.pi) * np.sqrt(Nf) / 2 * phihat_eval(fn, M=M, N=N, nu=nu, a_in=a_in)
    # nrm should be 1
    nrm: float = float(
        np.sqrt((2 * np.sum(phif[1:] ** 2) + phif[0] ** 2) * 2 * np.pi / ND) / (np.pi ** (3 / 2) / np.pi),
    )
    return phif / nrm


@njit()
def phi_eval(xvals: NDArray[np.float64], M=M_init, N=N_init, nu=nu_init, a_in=a_init) -> NDArray[np.complex128]:
    xvals = np.asarray(xvals)
    out = np.zeros_like(xvals, dtype=np.complex128)

    itrp = 0
    for m in range(-M, M + 1):
        for n in range(-N, N + 1):
            coeff = a_in[m + M, n + N]
            out += coeff * g_mn_x(xvals, m, n, nu_init)

    return prefactor * out
