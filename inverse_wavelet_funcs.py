"""functions for computing the inverse wavelet transforms"""
from numba import njit
import numpy as np
import numba
import fft_funcs as fft

from transform_freq_funcs import phitilde_vec

def inverse_wavelet_freq_time(wave_in,Nf,Nt,dt,nx=4.):
    """inverse wavlet transform to time domain via fourier transform of frequency domain"""
    res_f = inverse_wavelet_freq(wave_in,Nf,Nt,dt,nx)
    #TODO alias so not numpy_rfft
    return fft.irfft(res_f)

def inverse_wavelet_freq(wave_in,Nf,Nt,dt,nx=4.):
    """inverse wavelet transform to freq domain signal, current implementation is naive and slow but works"""
    ND = Nf*Nt
    Tobs = ND*dt
    oms = 2*np.pi/Tobs*np.arange(0,Nt//2+1)
    phif = phitilde_vec(oms,Nf,dt,nx)
    #nrm should be 1
    nrm = np.sqrt((2*np.sum(phif[1:]**2)+phif[0]**2)*2*np.pi/Tobs)#np.linalg.n
    nrm /= np.pi**(3/2)/np.pi/np.sqrt(dt) #normalization is ad hoc but appears correct
    #nrm = np.sqrt((2*np.sum(phif[1:]**2)+phif[0]**2)/(dt*Tobs)) #normalization is ad hoc but appears to be correct
    phif /= nrm
    #import matplotlib.pyplot as plt
    #plt.plot(phif)
    #plt.show()
    return inverse_wavelet_freq_helper_fast(wave_in,phif,Nf,Nt)

@njit(fastmath=True)
def inverse_wavelet_freq_helper_naive(wave_in,phif,Nf,Nt,dt):
    """jit compatible loop for inverse_wavelet_freq"""
    ND=Nf*Nt
    DT = dt*Nf
    Tobs = dt*ND
    res = np.zeros(ND//2+1,dtype=np.complex128)
    mult = np.zeros(ND//2+1,dtype=np.complex128)
    #TODO unsure if this m=0 part is correct because I don't have a forward transform to compare it to
    #TODO check handling of highest frequency bin

    #m=0
    for n in range(0,Nt):
        for i in range(0,ND//2+1):
            res[i] += np.sqrt(2)*wave_in[n,0]*np.exp(-1j*i*n*DT*4*np.pi/Tobs)*phif[i]
    #m=Nf #case not handled because we do not store it
    #m = Nf
    #if m%2:
    #    q=1
    #else:
    #    q=0

    #i_min1 = min(max(0,-Nt//2*m),ND//2+1)
    #i_max1 = min(max(0,Nt//2-Nt//2*m),ND//2+1)
    #i_min2 = min(max(Nt//2*(m-1),0),ND//2+1)
    #i_max2 = min(max(Nt//2*(m+1),0),ND//2+1)
    #print(i_min1,i_max1,i_min2,i_max2)
    #    prefactor = wave_in[n,m]
    #
    #    for i in range(i_min1,i_max1):
    #        mult_loc = np.exp(-1j*i*(2*n+q)*DT*2*np.pi/Tobs)
    #        res[i] += prefactor*mult_loc*phif[np.abs(i+Nt//2*m)]
    #    for i in range(i_min2,i_max2):
    #        mult_loc = np.exp(-1j*i*(2*n+q)*DT*2*np.pi/Tobs)
    #        res[i] += prefactor*mult_loc*phif[np.abs(i-Nt//2*m)]
    #for 0<m<Nf
    for n in range(0,Nt):
        for i in range(0,ND//2+1):
            mult[i] = np.exp(-1j*i*n*DT*2*np.pi/Tobs)
        for m in range(1,Nf-1):
            val = wave_in[n,m]
            if val!=0.:
                if(n+m)%2:
                    prefactor1 = 1j*val
                    prefactor2 = -1j*val
                else:
                    prefactor1 = 1*val
                    prefactor2 = 1*val
                i_min1 = min(max(0,-Nt//2*m),ND//2+1)
                i_max1 = min(max(0,Nt//2-Nt//2*m),ND//2+1)
                i_min2 = min(max(Nt//2*(m-1),0),ND//2+1)
                i_max2 = min(max(Nt//2*(m+1),0),ND//2+1)

                for i in range(i_min1,i_max1):
                    res[i] += prefactor1*phif[np.abs(i+Nt//2*m)]*mult[i]
                for i in range(i_min2,i_max2):
                    res[i] += prefactor2*phif[np.abs(i-Nt//2*m)]*mult[i]

    return res

#@njit()
def inverse_wavelet_freq_helper_fast(wave_in,phif,Nf,Nt):
    """jit compatible loop for inverse_wavelet_freq"""
    ND=Nf*Nt

    prefactor2s = np.zeros(Nt,np.complex128)
    res = np.zeros(ND//2+1,dtype=np.complex128)

    #m=Nf #case not handled because we do not store it
    #assert Nf%2==0 #don't handle odd number of pixels currently
    #if Nf%2:
    #    q=1
    #else:
    #    q=0

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
    #lower range is redundant because it is always from 0 to 0
    #i_min1 = min(max(0,-Nt//2*m),ND//2+1)
    #i_max1 = min(max(0,Nt//2-Nt//2*m),ND//2+1)
    i_min2 = min(max(Nt//2*(m-1),0),ND//2+1)
    i_max2 = min(max(Nt//2*(m+1),0),ND//2+1)
    for i in range(i_min2,i_max2):
        i_ind = np.abs(i-Nt//2*m)
        if i_ind>Nt//2:
            continue
        if m==0:
            res[i] += fft_prefactor2s[(2*i)%Nt]*phif[i_ind]
        elif m==Nf:
            res[i] += fft_prefactor2s[(2*i)%Nt]*phif[i_ind]#print(res[i],fft_prefactor[i%Nt]*phif[np.abs(i-Nt//2*m)])
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
                #prefactor1 = 1j*val
                mult2 = -1j
            else:
                #prefactor1 = 1*val
                mult2 = 1

            #prefactor1s[n] = prefactor1
            prefactor2s[n] = mult2*val
