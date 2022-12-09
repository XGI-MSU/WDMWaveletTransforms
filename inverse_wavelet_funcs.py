"""functions for computing the inverse wavelet transforms"""
from numba import njit
import numpy as np
import fft_funcs as fft

from transform_freq_funcs import phitilde_vec

def inverse_wavelet_freq_time(wave_in,Nf,Nt,dt,nx=4.):
    """inverse wavlet transform to time domain via fourier transform of frequency domain"""
    res_f = inverse_wavelet_freq(wave_in,Nf,Nt,dt,nx)
    return fft.irfft(res_f)

def inverse_wavelet_freq(wave_in,Nf,Nt,dt,nx=4.):
    """inverse wavelet transform to freq domain signal"""
    ND = Nf*Nt
    Tobs = ND*dt
    oms = 2*np.pi/Tobs*np.arange(0,Nt//2+1)
    phif = phitilde_vec(oms,Nf,dt,nx)
    #nrm should be 1
    nrm = np.sqrt((2*np.sum(phif[1:]**2)+phif[0]**2)*2*np.pi/Tobs)#np.linalg.n
    nrm /= np.pi**(3/2)/np.pi/np.sqrt(dt) #normalization is ad hoc but appears correct
    phif /= nrm
    return inverse_wavelet_freq_helper_fast(wave_in,phif,Nf,Nt)

#@njit()
def inverse_wavelet_freq_helper_fast(wave_in,phif,Nf,Nt):
    """jit compatible loop for inverse_wavelet_freq"""
    ND=Nf*Nt

    prefactor2s = np.zeros(Nt,np.complex128)
    res = np.zeros(ND//2+1,dtype=np.complex128)

    for m in range(0,Nf+1):
        pack_wave_inverse(m,Nt,Nf,prefactor2s,wave_in)
        #with numba.objmode(fft_prefactor2s="complex128[:]"):
        fft_prefactor2s = fft.fft(prefactor2s)
        unpack_wave_inverse(m,Nt,Nf,phif,fft_prefactor2s,res)

    return res

@njit()
def unpack_wave_inverse(m,Nt,Nf,phif,fft_prefactor2s,res):
    """helper for unpacking results of frequency domain inverse transform"""
    ND = Nt*Nf
    i_min2 = min(max(Nt//2*(m-1),0),ND//2+1)
    i_max2 = min(max(Nt//2*(m+1),0),ND//2+1)
    for i in range(i_min2,i_max2):
        i_ind = np.abs(i-Nt//2*m)
        if i_ind>Nt//2:
            continue
        if m==0:
            res[i] += fft_prefactor2s[(2*i)%Nt]*phif[i_ind]
        elif m==Nf:
            res[i] += fft_prefactor2s[(2*i)%Nt]*phif[i_ind]
        else:
            res[i] += fft_prefactor2s[i%Nt]*phif[i_ind]

@njit()
def pack_wave_inverse(m,Nt,Nf,prefactor2s,wave_in):
    """helper for fast frequency domain inverse transform to prepare for fourier transform"""
    if m==0:
        for n in range(0,Nt):
            prefactor2s[n] = 1/np.sqrt(2)*wave_in[(2*n)%Nt,0]
    elif m==Nf:
        for n in range(0,Nt):
            prefactor2s[n] = 1/np.sqrt(2)*wave_in[(2*n)%Nt+1,0]
    else:
        for n in range(0,Nt):
            val = wave_in[n,m]
            if (n+m)%2:
                mult2 = -1j
            else:
                mult2 = 1

            prefactor2s[n] = mult2*val
