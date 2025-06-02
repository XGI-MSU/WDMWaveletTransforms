""""test that both inverse functions perform as specified in stored dat files"""
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest

import WDMWaveletTransforms.fft_funcs as fft
from WDMWaveletTransforms.wavelet_transforms import transform_wavelet_freq, transform_wavelet_freq_time, transform_wavelet_time

EXACT_MATCH = False

def test_inverse_wavelets() -> None:
    """Test that forward wavelet transforms perform precisely as recorded in the input dat files
    for random input data
    """
    file_freq = Path(__file__).parent / 'data' / 'rand_wave_freq.dat'
    file_time = Path(__file__).parent / 'data' / 'rand_wave_time.dat'

    file_wave = Path(__file__).parent / 'data' / 'rand_wavelet.dat'
    file_wave_freq = Path(__file__).parent / 'data' / 'rand_wavelet_freq.dat'
    file_wave_time = Path(__file__).parent / 'data' / 'rand_wavelet_time.dat'

    dt = 30.

    #get a wavelet representation of a signal
    print('begin loading data files')
    t0 = perf_counter()
    #the original data (wave_in)
    wave_in = np.loadtxt(file_wave)

    #the forward wavelet transform of wave_in inverse wavelet transformed using frequency domain transforms both ways
    wave_freq_in = np.loadtxt(file_wave_freq)

    #the forward wavelet transform of wave_in inverse wavelet transformed time frequency domain transforms both ways
    wave_time_in = np.loadtxt(file_wave_time)

    #frequency domain forward and inverse transform is almost lossless so these should be nearly identical
    assert np.allclose(wave_freq_in,wave_in,atol=1.e-14,rtol=1.e-15)

    #time domain forward and inverse transforms are not quite as accurate so these have a little more tolerance
    assert np.allclose(wave_in,wave_time_in,atol=1.e-6,rtol=1.e-6)
    assert np.allclose(wave_freq_in,wave_time_in,atol=1.e-6,rtol=1.e-6)

    #the frequency domain inverse wavelet transform of wave_in
    fs_in,signal_freq_real_in,signal_freq_im_in = np.loadtxt(file_freq).T

    #the time domain inverse wvaelet transform of wave_in
    signal_freq_in = signal_freq_real_in+1j*signal_freq_im_in
    ts_in,signal_time_in = np.loadtxt(file_time).T


    t1 = perf_counter()
    print('loaded input files in %5.3fs'%(t1-t0))

    Nt = wave_in.shape[0]
    Nf = wave_in.shape[1]

    ND = Nt*Nf
    Tobs = dt*ND

    #time and frequency grids
    ts = np.arange(0,ND)*dt
    fs = np.arange(0,ND//2+1)*1/(Tobs)

    assert np.all(ts_in==ts)
    assert np.all(fs_in==fs)

    wave_freq_got = transform_wavelet_freq(signal_freq_in,Nf,Nt)

    t0 = perf_counter()
    wave_freq_got = transform_wavelet_freq(signal_freq_in,Nf,Nt)
    t1 = perf_counter()

    print('got frequency domain transform in %5.3fs'%(t1-t0))


    wave_time_got = transform_wavelet_time(signal_time_in,Nf,Nt,mult=32)

    t0 = perf_counter()
    wave_time_got = transform_wavelet_time(signal_time_in,Nf,Nt,mult=32)
    t1 = perf_counter()
    print('got time domain forward transform in %5.3fs'%(t1-t0))

    wave_time_got2 = transform_wavelet_freq_time(signal_time_in,Nf,Nt)

    t0 = perf_counter()
    wave_time_got2 = transform_wavelet_freq_time(signal_time_in,Nf,Nt)
    t1 = perf_counter()

    print('got from time domain to wavelet domain via fft in %5.3fs'%(t1-t0))

    #needed for internal consistency check of wave_time_got2
    wave_time_got3 = transform_wavelet_freq(fft.rfft(signal_time_in),Nf,Nt)

    if EXACT_MATCH:
        assert np.all(wave_freq_got==wave_freq_in)
        print('forward frequency domain transform matches expectation exactly')
        print(wave_time_got[0,0])
        print(wave_time_in[0,0])
        print(wave_freq_got[0,0])
        print(wave_time_got==wave_time_in)
        print(np.sum(wave_time_got==wave_time_in))
        assert np.all(wave_time_got==wave_time_in)
        print('forward time domain transform matches expectation exactly')
    else:
        #on different architecture than originally generated the files (i.e. different fft implementations)
        #match may not be exact but should still be close
        assert np.allclose(wave_freq_in,wave_freq_got,atol=1.e-14,rtol=1.e-15)
        print('forward frequency domain transform matches expectation closely')

        assert np.allclose(wave_time_in,wave_time_got,atol=1.e-14,rtol=1.e-15)
        print('forward time domain transform matches expectation closely')

    #still expect good match within the given architecture
    assert np.allclose(wave_freq_got,wave_time_got,atol=1.e-6,rtol=1.e-6)
    print('transforms match as closely as expected')

    #internal consistency of helper function
    assert np.all(wave_time_got3==wave_time_got2)

    print('all tests passed')

if __name__=='__main__':
    pytest.cmdline.main(['forward_wavelet_test.py'])
