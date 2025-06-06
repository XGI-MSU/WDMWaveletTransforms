"""Harness for computing inverse wavelet transform using time transform.

Take input .dat file in wavelet domain (Nt rows by Nf columns)
and write to .dat file in time domain (columns frequency, h(t)).
"""

import sys
from time import perf_counter

import numpy as np

from WDMWaveletTransforms.wavelet_transforms import inverse_wavelet_time


def main() -> None:
    # assume input .dat file is Nt rows by Nf columns
    if len(sys.argv) != 5:
        print('inverse_wavelet_time_harness.py filename_wavelet_in filename_time_out dt mult')
        sys.exit(1)

    file_in = sys.argv[1]
    file_out = sys.argv[2]
    dt = np.float64(sys.argv[3])
    mult = int(sys.argv[4])

    print('begin loading data file')
    t0 = perf_counter()
    wave_in = np.loadtxt(file_in)
    t1 = perf_counter()
    print('loaded input file in %5.3fs' % (t1 - t0))

    Nt = wave_in.shape[0]
    Nf = wave_in.shape[1]

    ND = Nt * Nf

    # time and frequency grids
    ts = np.arange(0, ND) * dt

    t0 = perf_counter()
    signal_time = inverse_wavelet_time(wave_in, Nf, Nt, mult=mult)
    t1 = perf_counter()

    print('got time domain transform in %5.3fs' % (t1 - t0))

    t4 = perf_counter()
    np.savetxt(file_out, np.vstack([ts, signal_time]).T)
    t5 = perf_counter()
    print('saved file in %5.3fs' % (t5 - t4))


if __name__ == '__main__':
    main()
