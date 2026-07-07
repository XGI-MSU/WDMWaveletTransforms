"""Speed comparison of the numba-jitted frequency-domain transforms against the
plain-numpy reference implementations in WDMWaveletTransforms.wavelet_reference.

Run directly: python benchmark_transforms.py
"""
# ruff: noqa: T201

from functools import partial
from time import perf_counter
from typing import Callable

import numpy as np

import WDMWaveletTransforms.modified_gaussian as mg
from WDMWaveletTransforms.inverse_wavelet_freq_funcs import inverse_wavelet_freq_helper_fast
from WDMWaveletTransforms.transform_freq_funcs import transform_wavelet_freq_helper
from WDMWaveletTransforms.wavelet_reference import (
    inverse_wavelet_freq_reference,
    transform_wavelet_freq_reference,
)


def _time(fn: Callable[[], object], n_rep: int = 3) -> float:
    best = np.inf
    for _ in range(n_rep):
        t0 = perf_counter()
        fn()
        best = min(best, perf_counter() - t0)
    return best


if __name__ == '__main__':
    rng = np.random.default_rng(31415)

    for Nf, Nt, mult_f in [(256, 256, 4), (512, 512, 8), (2048, 1024, 8)]:
        ND = Nf * Nt
        data = rng.normal(size=ND // 2 + 1) + 1j * rng.normal(size=ND // 2 + 1)
        data[0] = data[0].real
        data[-1] = data[-1].real

        phin = np.asarray(mg.phitilde_vec_norm(Nf, Nt, mult_f), dtype=np.float64)
        phif_fwd = 2 / Nf * phin

        # warm up jit compilation before timing
        wave = transform_wavelet_freq_helper(data, Nf, Nt, mult_f, phif_fwd)
        inverse_wavelet_freq_helper_fast(wave, phin, Nf, Nt, mult_f)

        t_fwd_jit = _time(partial(transform_wavelet_freq_helper, data, Nf, Nt, mult_f, phif_fwd))
        t_fwd_ref = _time(partial(transform_wavelet_freq_reference, data, Nf, Nt, mult_f, phif_fwd))
        t_inv_jit = _time(partial(inverse_wavelet_freq_helper_fast, wave, phin, Nf, Nt, mult_f))
        t_inv_ref = _time(partial(inverse_wavelet_freq_reference, wave, Nf, Nt, mult_f, phin))

        print(f'Nf={Nf} Nt={Nt} mult_f={mult_f}:')
        print(
            f'  forward  jitted {t_fwd_jit * 1e3:8.1f} ms   reference {t_fwd_ref * 1e3:8.1f} ms   '
            f'ratio {t_fwd_ref / t_fwd_jit:5.2f}x'
        )
        print(
            f'  inverse  jitted {t_inv_jit * 1e3:8.1f} ms   reference {t_inv_ref * 1e3:8.1f} ms   '
            f'ratio {t_inv_ref / t_inv_jit:5.2f}x'
        )
