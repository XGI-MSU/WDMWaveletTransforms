"""helper functions for transform_freq"""
import numpy as np
from numba import njit
import scipy.special

def phitilde_vec_norm(Nf,Nt,dt,nx):
    """helper function to get phitilde vec normalized as needed"""
    ND = Nf*Nt
    Tobs = ND*dt
    oms = 2*np.pi/Tobs*np.arange(0,Nt//2+1)
    phif = phitilde_vec(oms,Nf,dt,nx)
    #nrm should be 1
    nrm = np.sqrt((2*np.sum(phif[1:]**2)+phif[0]**2)*2*np.pi/Tobs)#np.linalg.n
    nrm /= np.pi**(3/2)/np.pi/np.sqrt(dt) #normalization is ad hoc but appears correct
    phif /= nrm
    return phif

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
