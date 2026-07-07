import numpy as np
import scipy.special
from numba import njit
from numpy.typing import NDArray
from scipy.fft import fftn, ifftn, next_fast_len

# ============================================================
# Gaussian and atoms
# ============================================================


@njit()
def g_nu(x, nu: float):
    return np.sqrt(np.sqrt(2 * nu)) * np.exp(-np.pi * nu * x**2)


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


def frame_bounds_short(nu: float, Nt: int = 100, Ns: int = 100, L: int = 20, fft_mode: int = 0) -> tuple[float, float]:
    if fft_mode == 0:
        return zak_g_abs2(nu, Nt=Nt, Ns=Ns, L=L)
    if fft_mode == 1:
        return zak_g_abs2_fft(nu, Nt=Nt, Ns=Ns, L=L)
    msg = 'fft_mode must be 0 or 1'
    raise ValueError(msg)


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


def _recursive_loop(
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
    assert fft_mode in (0, 2), 'fft_mode must be 0 (fft) or 2 (explicit loop)'
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

    def om_dot(vec):
        # apply the overlap matrix Omega, equivalent to np.dot(Omega, vec)
        if fft_mode == 0:
            return _om_dot_helper(M, N, vec, F_even, F_odd, fft_shape)
        return _om_dot_helper_explicit(M, N, vec, kernel_even)

    c_k = 1.0

    if null_mode in (0, 1):
        # get the component of b pointing into a (nearly) null eigenspace so that we can remove it from the actual iterative accumulation
        for k in range(Kmax + 1):
            # Equivalent to applying [I - 2P/(A+B)]
            b_null = b_null - alpha * om_dot(b_null)

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
        else:
            # accumulate a by Kahan compensated summation algorithm to reduce loss of numerical precision
            term = c_k * (b - b_null)
            y = term - comp
            t = a + y
            comp = (t - a) - y
            a = t

        # Equivalent to applying [I - 2P/(A+B)]
        b = b - alpha * om_dot(b)
        # c_k *= (2 * k + 1) / (2 * k + 2)
        # use exact form of c_k to try to reduce accumulated numerical error
        c_k = scipy.special.beta(1.5 + k, 0.5) / np.pi
    return a


def _recursive_loop_wrapper(
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

    prefactor = 2.0 * np.sqrt(1.0 / (A_nu + B_nu))

    center_index = M * (2 * N + 1) + N

    a = _recursive_loop_wrapper(
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

# lazy cache for the dual-window expansion coefficients: computing them runs the
# full coefficient recursion, so it is done on first use instead of at import
_MG_EXPANSION_CACHE: dict = {}


def get_mg_expansion(
    Kmax: int = Kmax_init,
    M: int = M_init,
    N: int = N_init,
    nu: float = nu_init,
    L: int = L_init,
    Ns: int = Ns_init,
    Nt_frame: int = Nt_init,
    fft_mode_frame: int = fft_mode_frame_init,
    fft_mode_recurse: int = fft_mode_recurse_init,
    compensate_mode: int = compensate_mode_init,
    null_mode: int = 0,
) -> tuple[NDArray[np.complex128], float]:
    """Get the (2M+1, 2N+1) expansion coefficients a_mn and overall prefactor, cached."""
    key = (Kmax, M, N, nu, L, Ns, Nt_frame, fft_mode_frame, fft_mode_recurse, compensate_mode, null_mode)
    if key not in _MG_EXPANSION_CACHE:
        a_flat, prefactor = _coefficient_recursion_helper(
            Kmax=Kmax,
            M=M,
            N=N,
            nu=nu,
            L=L,
            Ns=Ns,
            Nt=Nt_frame,
            fft_mode_frame=fft_mode_frame,
            fft_mode_recurse=fft_mode_recurse,
            compensate_mode=compensate_mode,
            null_mode=null_mode,
        )
        _MG_EXPANSION_CACHE[key] = (a_flat.reshape(((2 * M + 1), (2 * N + 1))), prefactor)
    return _MG_EXPANSION_CACHE[key]


# ============================================================
# Evaluation functions
# ============================================================


@njit()
def g_mn_x(x: NDArray[np.float64], m: int, n: int, nu: float) -> NDArray[np.complex128]:
    return np.exp(1j * np.pi * m * x) * g_nu(x - n, nu)


@njit()
def _phi_vec_core(
    Nf: int, mult_t: int, M: int, N: int, nu: float, a_in: NDArray[np.complex128], prefactor: float
) -> NDArray[np.float64]:
    assert Nf % 2 == 0
    assert a_in.shape == (2 * M + 1, 2 * N + 1)
    K = mult_t * 2 * Nf
    ts = np.arange(-K // 2, K // 2) / Nf
    out = np.zeros(K, dtype=np.complex128)

    for m in range(-M, M + 1):
        for n in range(-N, N + 1):
            coeff = a_in[m + M, n + N]
            out += coeff * g_mn_x(ts, m, n, nu)

    return prefactor * np.sqrt(2.0 / Nf) * np.real(out)


def phi_vec(Nf: int, mult_t: int, M: int = M_init, N: int = N_init, nu: float = nu_init) -> NDArray[np.float64]:
    """Time-domain window on the grid of K = 2*mult_t*Nf samples around the pixel center."""
    a_in, prefactor = get_mg_expansion(M=M, N=N, nu=nu)
    return _phi_vec_core(Nf, mult_t, M, N, nu, a_in, prefactor)


@njit()
def g_mn_hat(y: NDArray[np.float64], m: int, n: int, nu: float) -> NDArray[np.complex128]:
    sign = 1.0 if (m * n) % 2 == 0 else -1.0
    return sign * np.exp(2j * np.pi * y * n) * g_nu(y + m / 2, 1 / nu)


@njit()
def _phihat_eval_core(
    yvals: NDArray[np.float64], M: int, N: int, nu: float, a_in: NDArray[np.complex128], prefactor: float
) -> NDArray[np.float64]:
    assert a_in.shape == (2 * M + 1, 2 * N + 1)

    out = np.zeros_like(yvals, dtype=np.complex128)

    for m in range(-M, M + 1):
        for n in range(-N, N + 1):
            coeff = a_in[m + M, n + N]
            out += coeff * g_mn_hat(yvals, m, n, nu)

    return prefactor * np.real(out)


def phihat_eval(
    yvals: NDArray[np.float64], M: int = M_init, N: int = N_init, nu: float = nu_init
) -> NDArray[np.float64]:
    """Frequency-domain window; y in units of the band spacing 1/(2 dt Nf), band edge at y = 1/2."""
    a_in, prefactor = get_mg_expansion(M=M, N=N, nu=nu)
    return _phihat_eval_core(np.asarray(yvals, dtype=np.float64), M, N, nu, a_in, prefactor)


def phitilde_vec_norm(
    Nf: int, Nt: int, mult_f: int = 1, M: int = M_init, N: int = N_init, nu: float = nu_init
) -> NDArray[np.floating]:
    """Normalize phitilde as needed for inverse frequency domain transform.

    The window is sampled on the exact frequency-bin grid delta/Nt, delta = 0..mult_f*Nt/2,
    so that tap delta multiplies rfft bin m*Nt/2 + delta (band edge y = 1/2 at delta = Nt/2).
    """
    ND: int = Nf * Nt
    fn = np.arange(0, mult_f * Nt // 2 + 1) / Nt
    phif: NDArray[np.floating] = np.sqrt(Nf / 2) * phihat_eval(fn, M=M, N=N, nu=nu)
    # nrm should be 1
    nrm: float = float(
        np.sqrt((2 * np.sum(phif[1:] ** 2) + phif[0] ** 2) * 2 * np.pi / ND) / np.sqrt(np.pi),
    )
    return phif / nrm


@njit()
def _phi_eval_core(
    xvals: NDArray[np.float64], M: int, N: int, nu: float, a_in: NDArray[np.complex128], prefactor: float
) -> NDArray[np.complex128]:
    assert a_in.shape == (2 * M + 1, 2 * N + 1)

    out = np.zeros_like(xvals, dtype=np.complex128)

    for m in range(-M, M + 1):
        for n in range(-N, N + 1):
            coeff = a_in[m + M, n + N]
            out += coeff * g_mn_x(xvals, m, n, nu)

    return prefactor * out


def phi_eval(
    xvals: NDArray[np.float64], M: int = M_init, N: int = N_init, nu: float = nu_init
) -> NDArray[np.complex128]:
    """Time-domain window evaluated at arbitrary points, x in units of the pixel width."""
    a_in, prefactor = get_mg_expansion(M=M, N=N, nu=nu)
    return _phi_eval_core(np.asarray(xvals, dtype=np.float64), M, N, nu, a_in, prefactor)
