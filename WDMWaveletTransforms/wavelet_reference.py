"""Vectorized numpy reference implementations of the WDM frequency-domain transforms.

This module is the executable specification of the transform conventions.  It is
deliberately written with plain numpy array operations (no numba) so that every
sign, parity, and boundary rule is visible in one place.  The production jitted
helpers in transform_freq_funcs.py / inverse_wavelet_freq_funcs.py must agree with
these functions to floating-point precision; the unit tests enforce that.

Conventions implemented here (see the audit artifact for derivations):

Forward transform of rfft data X[j], j = 0..ND/2 (ND = Nf*Nt, K = mult_f*Nt):

  For each frequency band m the windowed analytic coefficient is

      c_m[n] = ((-1)^n / Nt) * sum_{delta in (-K/2, K/2)}
               phif[|delta|] * X_full[m*Nt/2 + delta] * exp(2j*pi*delta*n/Nt)

  where X_full is the full conjugate-symmetric periodic spectrum of the real
  signal, i.e. X_full[j] = X[j mod ND] with X[-j] = conj(X[j]).  Reading X_full
  instead of zero-padding implements the exact inner product with the real
  wavelet basis functions when the window crosses j = 0 or j = ND/2
  ("conjugate fold-back"); for mult_f = 1 interior bands it reduces to the
  original convention because the window never crosses.

  Interior bands 0 < m < Nf store real coefficients with the parity table
  (identical to the original mult_f = 1 implementation):

      m odd,  n even: wave[n, m] = -Im c_m[n]
      m odd,  n odd:  wave[n, m] = +Re c_m[n]
      m even, n odd:  wave[n, m] = +Im c_m[n]
      m even, n even: wave[n, m] = +Re c_m[n]

  The m = 0 and m = Nf bands are self-conjugate: the reflection at j = 0
  (resp. j = ND/2) maps the window onto itself, so the exact inner product is
  implemented by keeping only delta >= 0 (resp. delta <= 0), halving the
  center tap, and storing sqrt(2) * Re c in the even (resp. odd) rows of
  column 0.

Inverse transform (synthesis, the exact dual of the forward given a window
satisfying the WDM orthonormality conditions):

  For each interior band, F_m[l] = sum_n mult2_{nm} * wave[n, m] * exp(-2j*pi*n*l/Nt)
  with mult2 = -1j if (n+m) odd else 1, and each window tap contributes

      c = F_m[(m*Nt/2 + delta) mod Nt] * phif[|delta|]

  at ring position jj = m*Nt/2 + delta.  Contributions landing outside the
  rfft range fold back conjugated (the mirror lobe of the real atom):

      target = jj mod ND
      target == 0 or ND/2 : res[target] += 2*Re(c)
      target <  ND/2      : res[target] += c
      target >  ND/2      : res[ND - target] += conj(c)

  m = 0 and m = Nf use the doubled-frequency packing of the original
  implementation and scatter only their own half-plane, which is already the
  complete synthesis for the self-conjugate atoms.

  The tap at delta = -K/2 (the unpaired edge of the length-K window array) is
  excluded everywhere, in both directions: effective support is the open range
  delta in (-K/2, K/2).  (The original mult_f = 1 code added one such tap on
  the inverse side for m = Nf only; it is invisible for the Meyer window,
  which vanishes identically at |delta| = Nt/2, but would be inconsistent for
  windows with unbounded support.)
"""

import numpy as np
from numpy.typing import NDArray


def _check_args(Nf: int, Nt: int, mult_f: int) -> None:
    assert Nf > 0
    assert Nt > 0
    assert mult_f > 0
    assert Nf % 2 == 0
    assert Nt % 2 == 0
    assert mult_f <= Nf, 'window must not wrap around the full spectrum'


def _gather_spectrum_folded(
    data: NDArray[np.complexfloating],
    jj: NDArray[np.integer],
    ND: int,
) -> NDArray[np.complex128]:
    """Read the full conjugate-symmetric periodic spectrum at (possibly out of range) bins jj.

    data holds the rfft bins X[0..ND/2]; X_full[jj] = X[jj mod ND] with the
    negative-frequency half given by conjugate symmetry.
    """
    jj_mod = np.mod(jj, ND)
    fold = jj_mod > ND // 2
    idx = np.where(fold, ND - jj_mod, jj_mod)
    vals = np.asarray(data[idx], dtype=np.complex128)
    vals[fold] = np.conj(vals[fold])
    return vals


def transform_wavelet_freq_reference(
    data: NDArray[np.complexfloating],
    Nf: int,
    Nt: int,
    mult_f: int,
    phif: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Reference forward transform from rfft data to wavelet pixels.

    phif must already carry the overall normalization used by the production
    helper (i.e. the 2/Nf-scaled normalized window).
    """
    _check_args(Nf, Nt, mult_f)
    ND = Nf * Nt
    K = mult_f * Nt
    half_K = K // 2
    half_Nt = Nt // 2

    assert data.shape == (ND // 2 + 1,)
    assert phif.shape == (half_K + 1,)

    wave = np.zeros((Nt, Nf))

    ns = np.arange(Nt)
    sign_n = np.where(ns % 2 == 0, 1.0, -1.0)  # (-1)^n

    # effective window support: open range delta in (-K/2, K/2)
    deltas_int = np.arange(1 - half_K, half_K)  # interior bands
    deltas_low = np.arange(0, half_K)  # m = 0 keeps delta >= 0
    deltas_high = np.arange(1 - half_K, 1)  # m = Nf keeps delta <= 0

    for m in range(Nf + 1):
        if m == 0:
            deltas = deltas_low
        elif m == Nf:
            deltas = deltas_high
        else:
            deltas = deltas_int

        jj = m * half_Nt + deltas
        W = phif[np.abs(deltas)] * _gather_spectrum_folded(data, jj, ND)
        if m in (0, Nf):
            # halve the self-conjugate center tap; Re[] then implements the
            # exact fold of the reflection-symmetric atom
            W[deltas == 0] *= 0.5

        # alias the taps onto one Nt-period: exp(2j*pi*delta*n/Nt) depends
        # only on delta mod Nt
        w_alias = np.zeros(Nt, dtype=np.complex128)
        np.add.at(w_alias, np.mod(deltas, Nt), W)

        # c_m[n] = ((-1)^n / Nt) * sum_r w_alias[r] exp(2j*pi*r*n/Nt)
        c_m = sign_n * np.fft.ifft(w_alias)

        if m == 0:
            wave[::2, 0] = np.sqrt(2.0) * c_m[::2].real
        elif m == Nf:
            wave[1::2, 0] = np.sqrt(2.0) * c_m[::2].real
        elif m % 2:
            wave[::2, m] = -c_m[::2].imag  # n even
            wave[1::2, m] = c_m[1::2].real  # n odd
        else:
            wave[1::2, m] = c_m[1::2].imag  # n odd
            wave[::2, m] = c_m[::2].real  # n even

    return wave


def _scatter_folded(
    res: NDArray[np.complexfloating],
    jj: NDArray[np.integer],
    contrib: NDArray[np.complexfloating],
    ND: int,
) -> None:
    """Scatter-add synthesis contributions at ring bins jj into the rfft array res.

    Contributions at bins outside [0, ND/2] are the mirror lobe of the real
    atom and fold back conjugated; the self-conjugate bins 0 and ND/2 receive
    both lobes at once (2 Re).
    """
    target = np.mod(jj, ND)
    self_conj = (target == 0) | (target == ND // 2)
    upper = target > ND // 2

    np.add.at(res, target[self_conj], 2.0 * contrib[self_conj].real)
    direct = ~self_conj & ~upper
    np.add.at(res, target[direct], contrib[direct])
    np.add.at(res, ND - target[upper], np.conj(contrib[upper]))


def inverse_wavelet_freq_reference(
    wave_in: NDArray[np.floating],
    Nf: int,
    Nt: int,
    mult_f: int,
    phif: NDArray[np.floating],
) -> NDArray[np.complex128]:
    """Reference inverse transform from wavelet pixels to rfft data.

    phif is the normalized window without the 2/Nf forward scaling.
    """
    _check_args(Nf, Nt, mult_f)
    ND = Nf * Nt
    K = mult_f * Nt
    half_K = K // 2
    half_Nt = Nt // 2

    assert wave_in.shape == (Nt, Nf)
    assert phif.shape == (half_K + 1,)

    res = np.zeros(ND // 2 + 1, dtype=np.complex128)

    ns = np.arange(Nt)

    deltas_int = np.arange(1 - half_K, half_K)
    deltas_low = np.arange(0, half_K)
    deltas_high = np.arange(1 - half_K, 1)

    for m in range(Nf + 1):
        if m in (0, Nf):
            # doubled-frequency packing of the self-conjugate bands
            col = wave_in[np.mod(2 * ns, Nt), 0] if m == 0 else wave_in[np.mod(2 * ns, Nt) + 1, 0]
            a = col / np.sqrt(2.0)
            F_m = np.fft.fft(a)
            deltas = deltas_low if m == 0 else deltas_high
            jj = m * half_Nt + deltas
            contrib = F_m[np.mod(2 * jj, Nt)] * phif[np.abs(deltas)]
            # single-lobe scatter: the self-conjugate atoms stay inside [0, ND/2]
            res[jj] += contrib
            continue

        mult2 = np.where((ns + m) % 2 == 1, -1j, 1.0 + 0j)
        a = mult2 * wave_in[:, m]
        F_m = np.fft.fft(a)

        jj = m * half_Nt + deltas_int
        contrib = F_m[np.mod(jj, Nt)] * phif[np.abs(deltas_int)]
        _scatter_folded(res, jj, contrib, ND)

    return res
