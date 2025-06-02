""""harness for computing forward time domain wavelet transform via fft, take input .dat file in time domain (columns ftime, h(t)"
write to .dat file in wavelet domain (Nt rows by Nf columns)
"""

import sys
from time import perf_counter

import numpy as np

from WDMWaveletTransforms.wavelet_transforms import transform_wavelet_freq_time


def main() -> None:
    if len(sys.argv)!=6:
        print('forward_wavelet_freq_time_harness.py filename_time_in filename_wavelet_out dt Nt Nf')
        sys.exit(1)

    #transform parameters
    file_in = sys.argv[1]
    file_out = sys.argv[2]

    dt = np.float64(sys.argv[3])
    Nt = int(sys.argv[4])
    Nf = int(sys.argv[5])

    print('begin loading data file')
    t0 = perf_counter()
    #the frequency domain representation
    ts_in,signal_time = np.loadtxt(file_in).T
    t1 = perf_counter()
    print('loaded input file in %5.3fs'%(t1-t0))

    ND = Nt*Nf

    #time and frequency grids
    ts = dt*np.arange(0,ND)
    assert np.all(ts==ts_in)

    t0 = perf_counter()
    wave_time = transform_wavelet_freq_time(signal_time,Nf,Nt)
    t1 = perf_counter()

    print('got time domain transform in %5.3fs'%(t1-t0))

    t4 = perf_counter()
    np.savetxt(file_out,wave_time)
    t5 = perf_counter()
    print('saved file in %5.3fs'%(t5-t4))


if __name__=='__main__':
    main()
