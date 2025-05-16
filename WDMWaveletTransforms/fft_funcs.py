"""helper to make sure available fft functions are consistent across modules depending on install
mkl-fft is faster so it is the default, but numpy fft is probably more commonly installed to it is the fallback
"""
try:
    import mkl_fft
    rfft = mkl_fft.rfft_numpy
    irfft = mkl_fft.irfft_numpy
    fft = mkl_fft.fft
    ifft = mkl_fft.ifft
    print('WDMWaveletTransforms: using mkl fft')
except ImportError:
    import numpy as np
    rfft = np.fft.rfft
    irfft = np.fft.irfft
    fft = np.fft.fft
    ifft = np.fft.ifft
    print('WDMWaveletTransforms: using numpy fft')
