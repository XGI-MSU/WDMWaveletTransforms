""""harness for computing forward frequency domain wavelet transform, take input .dat file in frequency domain (columns frequency, real part(h(f)), imag part(h(f))"
write to .dat file in wavelet domain (Nt rows by Nf columns)
"""

import sys
from time import perf_counter

import numpy as np

from WDMWaveletTransforms.wavelet_transforms import transform_wavelet_freq


def main():
    if len(sys.argv)!=6:
        print('forward_wavelet_freq_harness.py filename_freq_in filename_wavelet_out dt Nt Nf')
        sys.exit(1)

    #transform parameters
    file_in = sys.argv[1]
    file_out = sys.argv[2]

    dt = np.float64(sys.argv[3])
    Nt = np.int64(sys.argv[4])
    Nf = np.int64(sys.argv[5])

    print('begin loading data file')
    t0 = perf_counter()
    #the frequency domain representation
    fs_in,signal_freq_real,signal_freq_im = np.loadtxt(file_in).T
    signal_freq = signal_freq_real+1j*signal_freq_im
    t1 = perf_counter()
    print('loaded input file in %5.3fs'%(t1-t0))

    ND = Nt*Nf
    Tobs = dt*ND

    #time and frequency grids
    fs = np.arange(0,ND//2+1)*1/(Tobs)
    assert np.all(fs==fs_in)

    t0 = perf_counter()
    wave_freq = transform_wavelet_freq(signal_freq,Nf,Nt)
    t1 = perf_counter()

    print('got frequency domain transform in %5.3fs'%(t1-t0))

    t4 = perf_counter()
    np.savetxt(file_out,wave_freq)
    t5 = perf_counter()
    print('saved file in %5.3fs'%(t5-t4))

if __name__=='__main__':
    main()
