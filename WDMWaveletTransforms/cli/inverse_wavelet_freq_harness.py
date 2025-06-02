""""harness for computing inverse frequency domain wavelet transform, take input .dat file in wavelet domain (Nt rows by Nf columns)
and write to .dat file in frequency domain (columns frequency, real part(h(f)), imag part(h(f))
"""
import sys
from time import perf_counter

import numpy as np

from WDMWaveletTransforms.wavelet_transforms import inverse_wavelet_freq


def main() -> None:
    #assume input .dat file is Nt rows by Nf columns
    if len(sys.argv)!=4:
        print('inverse_wavelet_freq_harness.py filename_wavelet_in filename_freq_out dt')
        sys.exit(1)

    #transform parameters
    file_in = sys.argv[1]
    file_out = sys.argv[2]

    dt = np.float64(sys.argv[3])

    #get a wavelet representation of a signal
    print('begin loading data file')
    t0 = perf_counter()
    wave_in = np.loadtxt(file_in)
    t1 = perf_counter()
    print('loaded input file in %5.3fs'%(t1-t0))

    Nt = wave_in.shape[0]
    Nf = wave_in.shape[1]

    ND = Nt*Nf
    Tobs = dt*ND

    #time and frequency grids
    fs = np.arange(0,ND//2+1)*1/(Tobs)


    t0 = perf_counter()
    signal_freq = inverse_wavelet_freq(wave_in,Nf,Nt)
    t1 = perf_counter()

    print('got frequency domain transform in %5.3fs'%(t1-t0))

    t4 = perf_counter()
    np.savetxt(file_out,np.vstack([fs,np.real(signal_freq),np.imag(signal_freq)]).T)
    t5 = perf_counter()
    print('saved file in %5.3fs'%(t5-t4))


if __name__=='__main__':
    main()
