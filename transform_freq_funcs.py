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

    #if m==0:
    #    #NOTE this term appears to be needed to recover correct constant (at least for m=0), but was previously missing
    #    DX[half_Nt] = phif[0]*data[0]/2.

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


def transform_wavelet_freq_time(data,Nf,Nt,ts,nx=4.,mult=8):
    """transform time domain data into wavelet domain via fft and then frequency transform"""
    dt = ts[1]-ts[0]
    ND = ts.size
    Nt = ND//Nf
    #DT = dt*Nf
    Tobs = dt*ND

    #if do_tukey:
    #    data_sig = data.copy()
    #    alpha = (2.0*(4.0*DT)/Tobs)
    #    #TODO should there be a tukey here or not?
    #    tukey(data_sig, alpha, ND)
    #else:
    #    data_sig = data

    data_fft = fft.rfft(data)
    fs = np.arange(0.,ND//2+1)/Tobs
    return transform_wavelet_freq(data_fft,Nf,Nt,fs,nx,mult)


#    tf = 0
#
#    # the time domain data stream
#    ND = ts.size
#    #TODO ts diff is not constant!!!!
#    dt = ts[1]-ts[0]
#    Tobs = dt*ND
#
#    Nt = np.int64(ND/Nf)
#
#    print("%e %d %d %d"%(Tobs, ND, Nf, Nt))
#
#    DT = dt*Nf           # width of wavelet pixel in time
#    DF = 1/(2*dt*Nf) # width of wavelet pixel in frequency
#    OM = np.pi/dt
#    L = 2*Nf
#    DOM = OM/Nf
#    insDOM = 1./np.sqrt(DOM)
#    B = OM/L
#    A = (DOM-B)/2.
#    K = mult*2*Nf
#    half_K = np.int64(K/2)
#
#    T = dt*K
#
#    dom = 2*np.pi/Tobs  # max frequency is K/2*dom = pi/dt = OM
#
#    print("Pixel size DT (seconds) %e DF (Hz) %e"%(DT, DF))
#    print("full filter bandwidth %e"%((A+B)/np.pi))
#
#
#    half_Nt = np.int64(Nt/2)
#    phif = phitilde_vec(dom*np.arange(0,half_Nt+1),Nf,dt,nx)#(double*)malloc(sizeof(double)* (Nt/2+1))
#    #print("norm check",np.sqrt((2*np.sum(phif[1:]**2)+phif[0]**2)*2*dom))
#
#    nrm = 0.0
#    for l in range(-half_Nt,half_Nt+1):
#        nrm += phif[abs(l)]**2
#    #TODO check if the division by 2 should exclude l=0
#    nrm = np.sqrt(nrm/2.0)
#    nrm *= np.sqrt(Nt*Nf)
#    print("norm="+str(nrm))
#
#    phif /= nrm
#
#
#    wave = np.zeros((Nt,Nf)) # wavelet wavepacket transform of the signal
#
#
#    # Window the data and FFT
#    # Tukey window parameter. Flat for (1-alpha) of data
#    alpha = (2.0*(4.0*DT)/Tobs)
#
#    if tf==0:
#        tukey(data, alpha, ND)
#        data = fft.rfft(data,ND)
#
#    DX = np.zeros(Nt,dtype=np.complex128)
#    for m in range(0,Nf):
#        DX_assign_loop(m,Nt,ND,DX,data,phif)
#        DX_trans = Nt*fft.ifft(DX,Nt)
#        DX_unpack_loop(m,Nt,Nf,DX_trans,wave)
#
#    return wave

def transform_wavelet_freq(data,Nf,Nt,fs,nx=4.,mult=8):
    #TODO mult doesn't do anything

    ND = Nf*Nt
    Tobs = 1.0/fs[1]
    dt = Tobs/ND

    #print("%e %d %d %d"%(Tobs, ND, Nf, Nt))

    DT = dt*Nf           # width of wavelet pixel in time
    DF = 1/(2*dt*Nf) # width of wavelet pixel in frequency
    OM = np.pi/dt
    L = 2*Nf
    DOM = OM/Nf
    #insDOM = 1./np.sqrt(DOM)
    B = OM/L
    A = (DOM-B)/2.
    #K = mult*2*Nf
    #half_K = np.int64(K/2)

    #T = dt*K

    dom = 2*np.pi/Tobs  # max frequency is K/2*dom = pi/dt = OM

    #print("Pixel size DT (seconds) %e DF (Hz) %e"%(DT, DF))
    #print("full filter bandwidth %e"%((A+B)/np.pi))


    half_Nt = np.int64(Nt/2)
    phif = phitilde_vec(dom*np.arange(0,half_Nt+1),Nf,dt,nx)#(double*)malloc(sizeof(double)* (Nt/2+1))
    if phif[-1]!=0.:
        raise ValueError('filter is likely not large enough to normalize correctly')


    nrm = 0.0
    for l in range(-half_Nt,half_Nt+1):
        nrm += phif[abs(l)]**2
    #TODO check if the division by 2 should exclude l=0
    nrm = np.sqrt(nrm/2.0)
    nrm *= np.sqrt(Nt*Nf)
    #print("norm="+str(nrm))

    phif /= nrm


    wave = np.zeros((Nt,Nf)) # wavelet wavepacket transform of the signal

    DX = np.zeros(Nt,dtype=np.complex128)
    for m in range(0,Nf+1):
        DX_assign_loop(m,Nt,ND,DX,data,phif)
        DX_trans = Nt*fft.ifft(DX,Nt)
        DX_unpack_loop(m,Nt,Nf,DX_trans,wave)
        #if m==0:
        #    #half of lowest and highest frequency bin pixels are redundant, so store them in even and odd components of m=0 respectively
        #    wave[::2,0] = np.real(DX_trans[np.arange(0,Nt,2)]*np.sqrt(2))
        #elif m==Nf:
        #    wave[1::2,0] = np.real(DX_trans[np.arange(0,Nt,2)]*np.sqrt(2))
        #else:

    return wave
