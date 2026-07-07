"""Unit tests for the WDM transform conventions, edge cases, and regression guards.

These tests pin down the sign/parity conventions, the m = 0 / m = Nf packing, the
mult_f oversampling phases, and the conjugate fold-back at the frequency-domain
extremes, for both the Meyer and modified-Gaussian wavelet families.  The jitted
production helpers are checked against the plain-numpy reference implementations
in WDMWaveletTransforms.wavelet_reference and against closed-form time-domain
atoms constructed directly from the window.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.fft import fftn

import WDMWaveletTransforms.modified_gaussian as mg
from WDMWaveletTransforms.inverse_wavelet_freq_funcs import inverse_wavelet_freq_helper_fast
from WDMWaveletTransforms.transform_freq_funcs import phitilde_vec_norm, transform_wavelet_freq_helper
from WDMWaveletTransforms.wavelet_reference import (
    inverse_wavelet_freq_reference,
    transform_wavelet_freq_reference,
)
from WDMWaveletTransforms.wavelet_transforms import (
    inverse_wavelet_freq,
    inverse_wavelet_freq_time,
    inverse_wavelet_time,
    transform_wavelet_freq,
    transform_wavelet_freq_time,
    transform_wavelet_time,
)


def _rand_rfft(ND: int, rng: np.random.Generator) -> np.ndarray:
    data = rng.normal(size=ND // 2 + 1) + 1j * rng.normal(size=ND // 2 + 1)
    data[0] = data[0].real
    data[-1] = data[-1].real
    return data


def _get_phin(family: str, Nf: int, Nt: int, mult_f: int) -> np.ndarray:
    if family == 'meyer':
        return phitilde_vec_norm(Nf, Nt, 4.0, mult_f)
    return np.asarray(mg.phitilde_vec_norm(Nf, Nt, mult_f), dtype=np.float64)


def _phi_K_kernel(phif: np.ndarray, ND: int, K: int) -> np.ndarray:
    """Closed-form truncated window kernel Phi_K(tau) for tau = 0..ND-1."""
    taus = np.arange(ND)
    deltas = np.arange(1 - K // 2, K // 2)
    return np.real(
        np.sum(phif[np.abs(deltas)][None, :] * np.exp(2j * np.pi * deltas[None, :] * taus[:, None] / ND), axis=1)
    )


def _atom(n: int, m: int, Nf: int, Nt: int, PhiK: np.ndarray) -> np.ndarray:
    """Closed-form time-domain analysis atom for pixel (n, m), forward-scaled window."""
    ND = Nf * Nt
    ks = np.arange(ND)
    kappa_mod = np.mod(ks - n * Nf, ND)
    Pk = PhiK[kappa_mod]
    kap = ks - n * Nf
    if m == 0:
        return Pk / (np.sqrt(2.0) * Nt)
    if m == Nf:
        return Pk * np.cos(np.pi * kap) / (np.sqrt(2.0) * Nt)
    if (n + m) % 2 == 0:
        return Pk * np.cos(np.pi * m * kap / Nf) / Nt
    return Pk * np.sin(np.pi * m * kap / Nf) / Nt


# ---------------------------------------------------------------------------
# jitted implementation must match the reference implementation exactly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('family', ['meyer', 'modified_gaussian'])
@pytest.mark.parametrize(('Nf', 'Nt'), [(8, 16), (16, 8), (32, 64)])
@pytest.mark.parametrize('mult_f', [1, 2, 3, 4])
def test_jit_matches_reference(family: str, Nf: int, Nt: int, mult_f: int) -> None:
    """Forward and inverse jitted helpers agree with the numpy reference to fp precision,
    covering both parities of mult_f (the parity enters the unpacking phases).
    """
    rng = np.random.default_rng(101)
    ND = Nf * Nt
    phin = _get_phin(family, Nf, Nt, mult_f)
    phif_fwd = 2 / Nf * phin
    data = _rand_rfft(ND, rng)

    w_ref = transform_wavelet_freq_reference(data, Nf, Nt, mult_f, phif_fwd)
    w_jit = transform_wavelet_freq_helper(data, Nf, Nt, mult_f, phif_fwd)
    assert_allclose(w_jit, w_ref, atol=1.0e-13 * np.max(np.abs(w_ref)), rtol=0.0)

    r_ref = inverse_wavelet_freq_reference(w_ref, Nf, Nt, mult_f, phin)
    r_jit = inverse_wavelet_freq_helper_fast(w_ref, phin, Nf, Nt, mult_f)
    assert_allclose(r_jit, r_ref, atol=1.0e-13 * np.max(np.abs(r_ref)), rtol=0.0)


# ---------------------------------------------------------------------------
# closed-form time-domain atoms pin the analysis conventions without any ffts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('family', ['meyer', 'modified_gaussian'])
@pytest.mark.parametrize('mult_f', [1, 2, 3])
def test_forward_matches_closed_form_atoms(family: str, mult_f: int) -> None:
    """wave[n, m] equals the inner product of the signal with the closed-form real atom:
    Phi_K(k - n*Nf)/Nt times cos (n+m even) or sin (n+m odd) of pi*m*(k - n*Nf)/Nf,
    with the m = 0 / m = Nf atoms at 1/sqrt(2) amplitude in the even/odd rows of column 0.
    """
    Nf, Nt = 8, 16
    ND = Nf * Nt
    K = mult_f * Nt
    rng = np.random.default_rng(202)
    phin = _get_phin(family, Nf, Nt, mult_f)
    phif_fwd = 2 / Nf * phin
    PhiK = _phi_K_kernel(phif_fwd, ND, K)

    data = _rand_rfft(ND, rng)
    x_time = np.fft.irfft(data)
    w_jit = transform_wavelet_freq_helper(data, Nf, Nt, mult_f, phif_fwd)

    wave_bf = np.zeros((Nt, Nf))
    for n in range(Nt):
        for m in range(Nf + 1):
            val = float(np.dot(x_time, _atom(n, m, Nf, Nt, PhiK)))
            if m == 0:
                if n % 2 == 0:
                    wave_bf[n, 0] = val
            elif m == Nf:
                if n % 2 == 0:
                    wave_bf[n + 1, 0] = val
            else:
                wave_bf[n, m] = val

    assert_allclose(w_jit, wave_bf, atol=1.0e-12 * np.max(np.abs(wave_bf)), rtol=0.0)


# ---------------------------------------------------------------------------
# round-trip identities and their convergence with mult_f
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('mult_f', [1, 2, 3])
def test_roundtrip_meyer_exact(mult_f: int) -> None:
    """The Meyer window has compact support, so the round trip is exact at any mult_f."""
    Nf, Nt = 16, 32
    ND = Nf * Nt
    rng = np.random.default_rng(303)
    data = _rand_rfft(ND, rng)
    w = transform_wavelet_freq(data, Nf, Nt, mult_f=mult_f)
    rec = inverse_wavelet_freq(w, Nf, Nt, mult_f=mult_f)
    assert_allclose(rec, data, atol=1.0e-13 * np.max(np.abs(data)), rtol=0.0)


@pytest.mark.parametrize(('mult_f', 'tol'), [(4, 5.0e-3), (8, 1.0e-5), (16, 1.0e-10), (24, 5.0e-13)])
def test_roundtrip_mg_convergence(mult_f: int, tol: float) -> None:
    """The modified-Gaussian round-trip error is set by the window tail truncated at
    |delta| = mult_f*Nt/2 and must fall with mult_f down to machine precision.
    Guards the window grid, the fold-back, and all packing conventions at once.
    """
    Nf, Nt = 32, 64
    ND = Nf * Nt
    rng = np.random.default_rng(404)
    data = _rand_rfft(ND, rng)
    w = transform_wavelet_freq(data, Nf, Nt, mult_f=mult_f, family='modified_gaussian')
    rec = inverse_wavelet_freq(w, Nf, Nt, mult_f=mult_f, family='modified_gaussian')
    err = np.max(np.abs(rec - data)) / np.max(np.abs(data))
    assert err < tol


def test_roundtrip_wave_side_identity() -> None:
    """The transform is a square map, so forward(inverse(wave)) = wave as well."""
    Nf, Nt = 16, 32
    rng = np.random.default_rng(505)
    wave = rng.normal(size=(Nt, Nf))
    mult_f = 16
    data = inverse_wavelet_freq(wave, Nf, Nt, mult_f=mult_f, family='modified_gaussian')
    wave_rec = transform_wavelet_freq(data, Nf, Nt, mult_f=mult_f, family='modified_gaussian')
    assert_allclose(wave_rec, wave, atol=1.0e-10 * np.max(np.abs(wave)), rtol=0.0)


# ---------------------------------------------------------------------------
# boundary and edge-case behavior at the frequency extremes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('jbin', [0, 1, 8, 16, 511, 512])
def test_delta_function_bins_roundtrip(jbin: int) -> None:
    """Spectra concentrated at the boundary bins (DC, Nyquist, and neighbors) survive
    the round trip: these bins are exactly where the fold-back conventions act.
    """
    Nf, Nt = 32, 32
    ND = Nf * Nt
    mult_f = 16
    data = np.zeros(ND // 2 + 1, dtype=np.complex128)
    data[jbin] = 1.0 if jbin in (0, ND // 2) else 1.0 + 0.5j
    w = transform_wavelet_freq(data, Nf, Nt, mult_f=mult_f, family='modified_gaussian')
    rec = inverse_wavelet_freq(w, Nf, Nt, mult_f=mult_f, family='modified_gaussian')
    assert_allclose(rec, data, atol=2.0e-11, rtol=0.0)


def test_dc_and_nyquist_bins_stay_real() -> None:
    """Reconstruction keeps the self-conjugate bins real (2 Re fold at j = 0, ND/2)."""
    Nf, Nt = 16, 32
    ND = Nf * Nt
    rng = np.random.default_rng(606)
    data = _rand_rfft(ND, rng)
    for mult_f in (3, 4):
        w = transform_wavelet_freq(data, Nf, Nt, mult_f=mult_f, family='modified_gaussian')
        rec = inverse_wavelet_freq(w, Nf, Nt, mult_f=mult_f, family='modified_gaussian')
        assert abs(rec[0].imag) < 1.0e-13 * np.max(np.abs(data))
        assert abs(rec[ND // 2].imag) < 1.0e-13 * np.max(np.abs(data))


def test_m0_nf_column_packing() -> None:
    """Half of the lowest and highest frequency bands are redundant; the lowest band is
    stored in the even rows of column 0 and the highest band in the odd rows.
    """
    Nf, Nt = 16, 32
    ND = Nf * Nt
    mult_f = 8
    rng = np.random.default_rng(707)

    # spectrum supported strictly inside the lowest band only
    data_low = np.zeros(ND // 2 + 1, dtype=np.complex128)
    data_low[: Nt // 4] = rng.normal(size=Nt // 4) + 1j * rng.normal(size=Nt // 4)
    data_low[0] = data_low[0].real
    w = transform_wavelet_freq(data_low, Nf, Nt, mult_f=mult_f, family='modified_gaussian')
    frac_odd = np.max(np.abs(w[1::2, 0])) / np.max(np.abs(w[::2, 0]))
    assert frac_odd < 1.0e-10  # no leakage into the m = Nf storage rows

    # spectrum supported strictly inside the highest band only
    data_high = np.zeros(ND // 2 + 1, dtype=np.complex128)
    data_high[-Nt // 4 :] = rng.normal(size=Nt // 4) + 1j * rng.normal(size=Nt // 4)
    data_high[-1] = data_high[-1].real
    w = transform_wavelet_freq(data_high, Nf, Nt, mult_f=mult_f, family='modified_gaussian')
    frac_even = np.max(np.abs(w[::2, 0])) / np.max(np.abs(w[1::2, 0]))
    assert frac_even < 1.0e-10  # no leakage into the m = 0 storage rows


def test_single_pixel_boundary_atoms_roundtrip() -> None:
    """Single-pixel wavelet arrays at the corner/boundary pixels reproduce themselves,
    including the special m = 0 / m = Nf rows and both n parities.
    """
    Nf, Nt = 16, 16
    mult_f = 16
    for n, m_col in [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (Nt - 1, Nf - 1), (0, Nf - 1), (5, 8)]:
        wave = np.zeros((Nt, Nf))
        wave[n, m_col] = 1.0
        data = inverse_wavelet_freq(wave, Nf, Nt, mult_f=mult_f, family='modified_gaussian')
        wave_rec = transform_wavelet_freq(data, Nf, Nt, mult_f=mult_f, family='modified_gaussian')
        assert_allclose(wave_rec, wave, atol=5.0e-11, rtol=0.0, err_msg=f'pixel ({n}, {m_col})')


def test_mult_f_must_not_wrap() -> None:
    """mult_f > Nf would wrap the window around the full spectrum and is rejected."""
    Nf, Nt = 4, 8
    data = np.zeros(Nf * Nt // 2 + 1, dtype=np.complex128)
    with pytest.raises(AssertionError):
        transform_wavelet_freq(data, Nf, Nt, mult_f=Nf + 1, family='modified_gaussian')


# ---------------------------------------------------------------------------
# normalization and window construction guards
# ---------------------------------------------------------------------------


def test_parseval_mg() -> None:
    """Total wavelet-domain power equals the Parseval sum of the input spectrum."""
    Nf, Nt = 32, 64
    ND = Nf * Nt
    mult_f = 8
    rng = np.random.default_rng(808)
    data = _rand_rfft(ND, rng)
    w = transform_wavelet_freq(data, Nf, Nt, mult_f=mult_f, family='modified_gaussian')
    pars = 1 / ND * (np.abs(data[0]) ** 2 + np.abs(data[-1]) ** 2 + 2 * np.sum(np.abs(data[1:-1]) ** 2))
    assert_allclose(np.sum(w**2), pars, rtol=1.0e-6)


def test_phitilde_mg_grid() -> None:
    """The window taps must sit on the exact bin grid delta/Nt: tap delta of
    phitilde_vec_norm is proportional to phihat_eval(delta/Nt) with one global constant.
    Guards against off-by-one/linspace-endpoint grid regressions.
    """
    Nf, Nt, mult_f = 16, 32, 4
    phin = np.asarray(mg.phitilde_vec_norm(Nf, Nt, mult_f), dtype=np.float64)
    direct = np.asarray(mg.phihat_eval(np.arange(0, mult_f * Nt // 2 + 1) / Nt), dtype=np.float64)
    mask = np.abs(direct) > 1.0e-8 * np.max(np.abs(direct))
    ratios = phin[mask] / direct[mask]
    assert_allclose(ratios, ratios[0], rtol=1.0e-12)


def test_phitilde_mg_nrm_near_one() -> None:
    """With the sqrt(Nf/2) prefactor the normalization constant is 1 to window accuracy."""
    Nf, Nt, mult_f = 32, 64, 8
    ND = Nf * Nt
    phif = np.sqrt(Nf / 2) * np.asarray(mg.phihat_eval(np.arange(0, mult_f * Nt // 2 + 1) / Nt))
    nrm = np.sqrt((2 * np.sum(phif[1:] ** 2) + phif[0] ** 2) * 2 * np.pi / ND) / np.sqrt(np.pi)
    assert_allclose(nrm, 1.0, atol=1.0e-5)


def test_om_dot_fft_matches_explicit() -> None:
    """The fft-based Omega matrix product agrees with the explicit-loop implementation."""
    M, N, nu = 6, 10, 0.5
    rng = np.random.default_rng(909)
    n_pair = (2 * M + 1) * (2 * N + 1)
    b = rng.normal(size=n_pair) + 1j * rng.normal(size=n_pair)

    kernel_even, _kernel_odd, _m_scales, _n_scales = mg._kernel_helper(M, N, nu)  # noqa: SLF001
    fft_shape = (
        mg._next_fast_even(2 * M + 1 + kernel_even.shape[0] - 1),  # noqa: SLF001
        mg._next_fast_even(2 * N + 1 + kernel_even.shape[1] - 1),  # noqa: SLF001
    )
    F_even = fftn(kernel_even, fft_shape)
    F_odd = np.roll(F_even, fft_shape[0] // 2, axis=0)

    out_fft = mg._om_dot_helper(M, N, b, F_even, F_odd, fft_shape)  # noqa: SLF001
    out_explicit = mg._om_dot_helper_explicit(M, N, b, kernel_even)  # noqa: SLF001
    assert_allclose(out_fft, out_explicit, atol=1.0e-12 * np.max(np.abs(out_explicit)), rtol=0.0)


# ---------------------------------------------------------------------------
# time-domain path and cross-domain consistency for the modified Gaussian
# ---------------------------------------------------------------------------


def test_time_domain_mg_roundtrip() -> None:
    """The time-domain mg path (which crashed before the phi_vec signature fix) is
    self-consistent to machine precision at large mult.
    """
    Nf, Nt = 16, 32
    rng = np.random.default_rng(1010)
    x = rng.normal(size=Nf * Nt)
    w = transform_wavelet_time(x, Nf, Nt, mult=16, family='modified_gaussian')
    x_rec = inverse_wavelet_time(w, Nf, Nt, mult=16, family='modified_gaussian')
    # accuracy limited by the time tail of the orthogonalized window truncated at
    # mult = Nt/2 pixels, not by machine precision
    assert_allclose(x_rec, x, atol=1.0e-10 * np.max(np.abs(x)), rtol=0.0)


def test_time_freq_domain_agreement_mg() -> None:
    """Forward transforms computed in the time and frequency domains agree to the
    frequency-window truncation error.
    """
    Nf, Nt = 16, 32
    rng = np.random.default_rng(1111)
    x = rng.normal(size=Nf * Nt)
    w_time = transform_wavelet_time(x, Nf, Nt, mult=16, family='modified_gaussian')
    w_freq = transform_wavelet_freq_time(x, Nf, Nt, mult_f=16, family='modified_gaussian')
    assert_allclose(w_time, w_freq, atol=1.0e-9 * np.max(np.abs(w_freq)), rtol=0.0)
    x_rec = inverse_wavelet_freq_time(w_freq, Nf, Nt, mult_f=16, family='modified_gaussian')
    assert_allclose(x_rec, x, atol=1.0e-9 * np.max(np.abs(x)), rtol=0.0)


# ---------------------------------------------------------------------------
# public API guards
# ---------------------------------------------------------------------------


def test_meyer_is_default_family() -> None:
    """Backward compatibility: the default family is the original Meyer wavelet."""
    Nf, Nt = 8, 16
    rng = np.random.default_rng(1212)
    data = _rand_rfft(Nf * Nt, rng)
    w_default = transform_wavelet_freq(data, Nf, Nt)
    w_meyer = transform_wavelet_freq(data, Nf, Nt, family='meyer')
    assert np.array_equal(w_default, w_meyer)


def test_unknown_family_raises() -> None:
    Nf, Nt = 8, 16
    data = np.zeros(Nf * Nt // 2 + 1, dtype=np.complex128)
    with pytest.raises(ValueError, match='family'):
        transform_wavelet_freq(data, Nf, Nt, family='haar')
