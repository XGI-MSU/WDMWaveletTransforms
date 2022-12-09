"""helper functions for transform_freq"""
import numpy as np
from numba import njit
import scipy.special
import fft_funcs as fft

def phitilde_vec(om,Nf,dt,nx=4.):
    """compute phitilde, om i array, nx is filter steepness, defaults to 4."""
    OM = np.pi/dt  #Nyquist angular frequency
    DOM = OM/Nf #2 pi times DF
    insDOM = 1./np.sqrt(DOM)
    B = OM/(2*Nf)
    A = (DOM-B)/2
    z = np.zeros(om.size)

    mask = (np.abs(om)>= A)&(np.abs(om)<A+B)

    x = (np.abs(om[mask])-A)/B
    y = scipy.special.betainc(nx,nx, x)
    z[mask] = insDOM*np.cos(np.pi/2.*y)

    z[np.abs(om)<A] = insDOM
    return z

@njit()
def tukey(data,alpha,N):
    """apply tukey window function to data"""
    imin = np.int64(alpha*(N-1)/2)
    imax = np.int64((N-1)*(1-alpha/2))
    Nwin = N-imax

    for i in range(0,N):
        f_mult = 1.0
        if i<imin:
            f_mult = 0.5*(1.+np.cos(np.pi*(i/imin-1.)))
        if i>imax:
            f_mult = 0.5*(1.+np.cos(np.pi/Nwin*(i-imax)))
        data[i] *= f_mult

@njit()
def DX_assign_loop(m,Nt,ND,DX,data,phif):
    """helper for assigning DX in the main loop"""
    half_Nt = np.int64(Nt/2)
    half_ND = np.int64(ND/2)
    for j in range(-half_Nt,half_Nt):
        i = j+half_Nt
        jj = j+m*half_Nt

        if 0<jj<half_ND:
            DX[i] = phif[abs(j)]*data[jj]
        elif jj==0:
            #NOTE this term appears to be needed to recover correct constant (at least for m=0), but was previously missing
            DX[i] = phif[abs(j)]*data[jj]/2.
        elif jj==half_ND:
            #NOTE this term appears to be needed to recover correct value (at least for m=Nf), but was previously missing
            DX[i] = phif[abs(j)]*data[jj]/2.
        else:
            DX[i] = 0.

@njit()
def DX_unpack_loop(m,Nt,Nf,DX_trans,wave):
    """helper for unpacking fftd DX in main loop"""
    if m==0:
        #half of lowest and highest frequency bin pixels are redundant, so store them in even and odd components of m=0 respectively
        for n in range(0,Nt,2):
            wave[n,0] = np.real(DX_trans[n]*np.sqrt(2))
    elif m==Nf:
        for n in range(0,Nt,2):
            wave[n+1,0] = np.real(DX_trans[n]*np.sqrt(2))
    else:
        for n in range(0,Nt):
            if m%2:
                if (n+m)%2:
                    wave[n,m] = -np.imag(DX_trans[n])
                else:
                    wave[n,m] = np.real(DX_trans[n])
            else:
                if (n+m)%2:
                    wave[n,m] = np.imag(DX_trans[n])
                else:
                    wave[n,m] = np.real(DX_trans[n])


def transform_wavelet_freq_time(data,Nf,Nt,ts,nx=4.):
    """transform time domain data into wavelet domain via fft and then frequency transform"""
    dt = ts[1]-ts[0]
    ND = ts.size
    Nt = ND//Nf
    Tobs = dt*ND

    data_fft = fft.rfft(data)
    fs = np.arange(0.,ND//2+1)/Tobs
    return transform_wavelet_freq(data_fft,Nf,Nt,fs,nx)

def transform_wavelet_freq(data,Nf,Nt,fs,nx=4.):
    """do the wavelet transform using the fast wavelet domain transform"""
    ND = Nf*Nt
    Tobs = 1.0/fs[1]
    dt = Tobs/ND

    dom = 2*np.pi/Tobs  # max frequency is K/2*dom = pi/dt = OM

    half_Nt = np.int64(Nt/2)
    phif = phitilde_vec(dom*np.arange(0,half_Nt+1),Nf,dt,nx)
    if phif[-1]!=0.:
        raise ValueError('filter is likely not large enough to normalize correctly')


    nrm = 0.0
    for l in range(-half_Nt,half_Nt+1):
        nrm += phif[abs(l)]**2

    nrm = np.sqrt(nrm/2.0)
    nrm *= np.sqrt(Nt*Nf)

    phif /= nrm


    wave = np.zeros((Nt,Nf)) # wavelet wavepacket transform of the signal

    DX = np.zeros(Nt,dtype=np.complex128)
    for m in range(0,Nf+1):
        DX_assign_loop(m,Nt,ND,DX,data,phif)
        DX_trans = Nt*fft.ifft(DX,Nt)
        DX_unpack_loop(m,Nt,Nf,DX_trans,wave)
    return wave
