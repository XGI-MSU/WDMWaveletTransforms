"""helper functions for transform_time.py"""
import numpy as np
from transform_freq_funcs import phitilde_vec_norm,DX_assign_loop,DX_unpack_loop
from inverse_wavelet_freq_funcs import inverse_wavelet_freq_helper_fast
from inverse_wavelet_time_funcs import inverse_wavelet_time_helper_fast
import fft_funcs as fft
from transform_time_funcs import assign_wdata,pack_wave,phi_vec

def inverse_wavelet_time(wave_in,Nf,Nt,dt,nx=4.,mult=32):
    """fast inverse wavelet transform to time domain"""
    ND = Nf*Nt

    mult = min(mult,ND//(2*Nf)) #make sure K isn't bigger than ND
    K = mult*2*Nf

    phi = phi_vec(Nf,dt,nx,mult)

    return inverse_wavelet_time_helper_fast(wave_in,phi,Nf,Nt,K)

def inverse_wavelet_freq_time(wave_in,Nf,Nt,dt,nx=4.):
    """inverse wavlet transform to time domain via fourier transform of frequency domain"""
    res_f = inverse_wavelet_freq(wave_in,Nf,Nt,dt,nx)
    return fft.irfft(res_f)

def inverse_wavelet_freq(wave_in,Nf,Nt,dt,nx=4.):
    """inverse wavelet transform to freq domain signal"""
    phif = phitilde_vec_norm(Nf,Nt,dt,nx)
    return inverse_wavelet_freq_helper_fast(wave_in,phif,Nf,Nt)

def transform_wavelet_time(data,Nf,Nt,dt,nx=4.,mult=32):
    """do the wavelet transform in the time domain"""
    # the time domain data stream
    ND = Nf*Nt

    #mult, can cause bad leakage if it is too small but may be possible to mitigate
    #mult = 16 # Filter is mult times pixel with in time

    K = mult*2*Nf

    # windowed data packets
    wdata = np.zeros(K)

    wave = np.zeros((Nt,Nf))  # wavelet wavepacket transform of the signal
    phi = phi_vec(Nf,dt,nx,mult)

    for i in range(0,Nt):
        assign_wdata(i,K,ND,Nf,wdata,data,phi)
        wdata_trans = fft.rfft(wdata,K)
        pack_wave(i,mult,Nf,wdata_trans,wave)

    return wave

def transform_wavelet_freq_time(data,Nf,Nt,dt,nx=4.):
    """transform time domain data into wavelet domain via fft and then frequency transform"""
    data_fft = fft.rfft(data)

    return transform_wavelet_freq(data_fft,Nf,Nt,dt,nx)

def transform_wavelet_freq(data,Nf,Nt,dt,nx=4.):
    """do the wavelet transform using the fast wavelet domain transform"""
    ND = Nf*Nt
#    Tobs = dt*ND
#
#    dom = 2*np.pi/Tobs  # max frequency is K/2*dom = pi/dt = OM
#
#    half_Nt = np.int64(Nt/2)
#    phif = phitilde_vec(dom*np.arange(0,half_Nt+1),Nf,dt,nx)
#    if phif[-1]!=0.:
#        raise ValueError('filter is likely not large enough to normalize correctly')
#
#
#    nrm = 0.0
#    for l in range(-half_Nt,half_Nt+1):
#        nrm += phif[abs(l)]**2
#
#    nrm = np.sqrt(nrm/2.0)
#    nrm *= np.sqrt(Nt*Nf)
#
#    phif /= nrm
#    phif *= Nt

    phif = 2/Nf*phitilde_vec_norm(Nf,Nt,dt,nx)

    wave = np.zeros((Nt,Nf)) # wavelet wavepacket transform of the signal

    DX = np.zeros(Nt,dtype=np.complex128)
    for m in range(0,Nf+1):
        DX_assign_loop(m,Nt,ND,DX,data,phif)
        DX_trans = fft.ifft(DX,Nt)
        DX_unpack_loop(m,Nt,Nf,DX_trans,wave)
    return wave
