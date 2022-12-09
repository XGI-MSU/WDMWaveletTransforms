"""functions for computing the inverse wavelet transforms"""
from numba import njit
import numpy as np

import fft_funcs as fft

from transform_freq_funcs import phitilde_vec


def inverse_wavelet_time(wave_in,Nf,Nt,dt,nx=4.,mult=32):
    """fast inverse wavelet transform to time domain"""
    ND = Nf*Nt

    mult = min(mult,ND//(2*Nf)) #make sure K isn't bigger than ND

    OM = np.pi/dt
    M = Nf
    DOM = OM/M
    insDOM = 1./np.sqrt(DOM)
    K = mult*2*M
    half_K = np.int64(K/2)

    Tw = dt*K

    dom = 2*np.pi/Tw  # max frequency is K/2*dom = pi/dt = OM

    #zero frequency
    DX = np.zeros(K,dtype=np.complex128)
    DX[0] =  insDOM

    DX = DX.copy()
    # postive frequencies
    DX[1:half_K+1] = phitilde_vec(dom*np.arange(1,half_K+1),Nf,dt,nx)
    # negative frequencies
    DX[half_K+1:] = phitilde_vec(-dom*np.arange(half_K-1,0,-1),Nf,dt,nx)

    DX = K*fft.ifft(DX,K)

    phi = np.zeros(K)
    phi[0:half_K] = np.real(DX[half_K:K])
    phi[half_K:] = np.real(DX[0:half_K])

    nrm = np.sqrt(K/dom)

    # windowed data packets
    fac = np.sqrt(2.0)/nrm
    phi *= fac

    return inverse_wavelet_time_helper_fast(wave_in,phi,Nf,Nt,K)


def inverse_wavelet_time_helper_fast(wave_in,phi,Nf,Nt,K):
    """helper loop for fast inverse wavelet transform"""
    ND=Nf*Nt
    res = np.zeros(ND)

    afins = np.zeros(2*Nf,dtype=np.complex128)

    for n in range(0,Nt):
        pack_wave_time_helper(n,Nf,Nt,wave_in,afins)
        ffts_fin_real = np.real(fft.fft(afins))
        unpack_time_wave_helper(n,Nf,Nt,K,phi,ffts_fin_real,res)

    return res

@njit()
def unpack_time_wave_helper(n,Nf,Nt,K,phis,fft_fin_real,res):
    """helper for time domain wavelet transform to unpack wavelet domain coefficients"""
    ND = Nf*Nt

    idxf = (-K//2+n*Nf+ND)%(2*Nf)
    k = (-K//2+n*Nf)%ND

    for k_ind in range(0,K):
        res_loc = fft_fin_real[idxf]
        res[k] += phis[k_ind]*res_loc
        idxf+=1
        k += 1

        if idxf==2*Nf:
            idxf = 0
        if k==ND:
            k = 0

@njit()
def pack_wave_time_helper(n,Nf,Nt,wave_in,afins):
    """helper for time domain transform to pack wavelet domain coefficients"""
    if n%2==0:
        #assign highest and lowest bin correctly
        afins[0] = 1/np.sqrt(2)*wave_in[n,0]
        if n+1<Nt:
            afins[Nf] = 1/np.sqrt(2)*wave_in[n+1,0]
    else:
        afins[0] = 0.
        afins[Nf] = 0.

    for idxm in range(0,Nf//2-1):
        if n%2:
            afins[2*idxm+2] = 1j*wave_in[n,2*idxm+2]
        else:
            afins[2*idxm+2] = wave_in[n,2*idxm+2]

    for idxm in range(0,Nf//2):
        if n%2:
            afins[2*idxm+1] = -wave_in[n,2*idxm+1]
        else:
            afins[2*idxm+1] = 1j*wave_in[n,2*idxm+1]
