"""functions for computing the inverse wavelet transforms"""
from numba import njit
import numpy as np
import WDMWaveletTransforms.fft_funcs as fft

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

    if m==0 or m==Nf:
        for i_ind in range(0,Nt//2):
            i = np.abs(m*Nt//2-i_ind)#i_off+i_min2
            ind3 = (2*i)%Nt
            res[i] += fft_prefactor2s[ind3]*phif[i_ind]
        if m==Nf:
            i_ind = Nt//2
            i = np.abs(m*Nt//2-i_ind)#i_off+i_min2
            ind3 = 0
            res[i] += fft_prefactor2s[ind3]*phif[i_ind]
    else:
        ind31 = (Nt//2*m)%Nt
        ind32 = (Nt//2*m)%Nt
        for i_ind in range(0,Nt//2):
            i1 = Nt//2*m-i_ind
            i2 = Nt//2*m+i_ind
            #assert ind31 == i1%Nt
            #assert ind32 == i2%Nt
            res[i1] += fft_prefactor2s[ind31]*phif[i_ind]
            res[i2] += fft_prefactor2s[ind32]*phif[i_ind]
            ind31 -= 1
            ind32 += 1
            if ind31<0:
                ind31 = Nt-1
            if ind32==Nt:
                ind32 = 0

        res[Nt//2*m] = fft_prefactor2s[(Nt//2*m)%Nt]*phif[0]

#@njit()
#def unpack_wave_inverse(m,Nt,Nf,phif,fft_prefactor2s,res):
#    """helper for unpacking results of frequency domain inverse transform"""
#    ND = Nt*Nf
#    i_min2 = min(max(Nt//2*(m-1),0),ND//2+1)
#    i_max2 = min(max(Nt//2*(m+1),0),ND//2+1)
#    for i in range(i_min2,i_max2):
#        i_ind = np.abs(i-Nt//2*m)
#        if i_ind>Nt//2:
#            continue
#        if m==0:
#            res[i] += fft_prefactor2s[(2*i)%Nt]*phif[i_ind]
#        elif m==Nf:
#            res[i] += fft_prefactor2s[(2*i)%Nt]*phif[i_ind]
#        else:
#            res[i] += fft_prefactor2s[i%Nt]*phif[i_ind]

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
