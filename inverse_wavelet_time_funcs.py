"""functions for computing the inverse wavelet transforms"""
from numba import njit
import numpy as np

import fft_funcs as fft

from transform_freq_funcs import phitilde_vec


def inverse_wavelet_time(wave_in,Nf,Nt,dt,nx=4.,mult=32):
    """inverse wavlet transform to time domain via fourier transform of frequency domain"""
    ND = Nf*Nt

    #mult = 16 # Filter is mult times pixel with in time
    mult = min(mult,ND//(2*Nf)) #make sure K isn't bigger than ND
    #print('mult got',mult)

    #print("%d %d %d"%(ND, Nf, Nt))

    DT = dt*Nf           # width of wavelet pixel in time
    DF = 1/(2*dt*Nf) # width of wavelet pixel in frequency
    OM = np.pi/dt
    M = Nf
    L = 2*M
    DOM = OM/M
    insDOM = 1./np.sqrt(DOM)
    B = OM/L
    A = (DOM-B)/2.
    K = mult*2*M
    half_K = np.int64(K/2)
    #print(K,ND)

    Tw = dt*K

    #print("filter length = %d"%K)
    #print("Pixel size DT (seconds) %e DF (Hz) %e"%(DT, DF))
    #print("Filter length (seconds) %e full filter bandwidth %e"%(Tw, (A+B)/np.pi))

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

    nrm = np.sqrt(K/dom)#*np.linalg.norm(phi)


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
    ND = Nf*Nt
    #k_ind_min1 = 0
    #k_ind_max1 = min(max(K//2-n*Nf,0),K)
    #k_ind_min2 = k_ind_max1
    #k_ind_max2 = min(max(K//2-n*Nf+ND,k_ind_min2),K)
    #k_ind_min3 = k_ind_max2
    #k_ind_max3 = K

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

   #     mult = K//(2*Nf)

    #for k_off in range(0,ND-Nf,Nf):
    #for k in range(0,ND):
    #    for even_flag in range(0,2):
    #        if even_flag:
    #            k_off = np.int64(k//(2*Nf))*2*Nf
    #            itrf = (k-k_off)#%(2*Nf)
    #        else:
    #            k_off = (np.int64(k//(2*Nf))*2*Nf-Nf)#%ND
    #            itrf = (k-k_off)%(2*Nf)
    #            k_off = k_off%ND


    #        assert 0<=itrf<2*Nf

    #        n = ((K+2*k_off)//(2*Nf))%(Nt)
    #        if n%2:
    #            idxf_base = (-K//2+ND+Nf)%(2*Nf)
    #        else:
    #            idxf_base = (-K//2+ND)%(2*Nf)

    #        for idxm in range(0,mult):
    #            #itrf =  2*Nf-idxf_base
    #            idxf = (itrf+idxf_base)%(2*Nf)
    #            #for idxf in range(0,2*Nf):
    #                #if idxf<idxf_base:
    #                #    itrf = 2*Nf-idxf_base+idxf
    #                #else:
    #                #    itrf = idxf-idxf_base
    #                #k = k_off+itrf
    #                #if k_off==ND-Nf and itrf>=Nf:
    #                #    k = (itrf+k_off-ND)#%ND
    #                #else:
    #                #k = (itrf+k_off)#%ND
    #            k_ind = 2*Nf*idxm+itrf
    #            #assert k==(k_ind-K//2+n*Nf)%ND
    #            res_loc = fft_fin_real[n,idxf]
    #            res[k] += phis[k_ind]*res_loc
    #            #itrf += 1
    #            #if itrf == 2*Nf:
    #            #    itrf -= 2*Nf

    #            n -= 2
    #            if n<0:
    #                n+=Nt


    #for k_off in range(0,ND-Nf,Nf):
    #    n = ((K+2*k_off)//(2*Nf))%(Nt)
    #    if n%2:
    #        idxf_base = (-K//2+ND+Nf)%(2*Nf)
    #    else:
    #        idxf_base = (-K//2+ND)%(2*Nf)

    #    for idxm in range(0,mult):
    #        itrf =  2*Nf-idxf_base
    #        for idxf in range(0,2*Nf):
    #            #if idxf<idxf_base:
    #            #    itrf = 2*Nf-idxf_base+idxf
    #            #else:
    #            #    itrf = idxf-idxf_base
    #            #k = k_off+itrf
    #            #if k_off==ND-Nf and itrf>=Nf:
    #            #    k = (itrf+k_off-ND)#%ND
    #            #else:
    #            k = (itrf+k_off)#%ND
    #            k_ind = 2*Nf*idxm+itrf
    #            #assert k==(k_ind-K//2+n*Nf)%ND
    #            res_loc = fft_fin_real[n,idxf]
    #            res[k] += phis[k_ind]*res_loc
    #            itrf += 1
    #            if itrf == 2*Nf:
    #                itrf -= 2*Nf

    #        n -= 2
    #        if n<0:
    #            n+=Nt

    #idxf_base = (-K//2+ND+Nf)%(2*Nf)
    #for k_off in range(Nf,ND,2*Nf):
    ##for n in range(1,Nt,2):
    #    n = ((K+2*k_off)//(2*Nf))%(Nt)

    #    for idxm in range(0,mult):
    #        assert n%2
    #        for idxf in range(0,2*Nf):
    #            if idxf<idxf_base:
    #                itrf = 2*Nf-idxf_base+idxf
    #            else:
    #                itrf = idxf-idxf_base
    #        #k = (itrf-K//2+n*Nf)%ND

    #            res_loc = fft_fin_real[n,idxf]
    #            #k_off = (2*Nf*idxm-K//2+n*Nf)%ND
    #            #if k_off==3*Nf and idxf==0:
    #            #    print(idxm,n,k_off)
    #            #n_recover = ((K+2*k_off-4*idxm*Nf)/(2*Nf))%(Nt)
    #            #if n_recover!=n:
    #            #    print(n,n_recover,k_off,idxm)
    #            #assert n_recover==n
    #            k_ind = 2*Nf*idxm+itrf
    #            #k = (k_ind-K//2+n*Nf)%ND
    #            #assert k==(k_ind+k_off-2*idxm*Nf)%ND
    #            if k_off==ND-Nf and itrf>=Nf:
    #                k = (itrf+k_off-ND)#%ND
    #            else:
    #                k = (itrf+k_off)#%ND
    #            #if k>=ND:
    #            #    print((k_off-Nf)/(2*Nf),idxm,n,k,itrf,idxf,k%ND)
    #            assert 0<=k<ND
    #            #if k_off==Nf:
    #            #    print(k,k_ind,n)
    #            #if k!=(k_off+idxf+2*Nf*n)%ND:
    #            #    print(k,k_off,idxf,idxm,n,(k_off+idxf+2*Nf*idxm+n*Nf)%ND)
    #            #assert k==(k_off+idxf+2*Nf*n)%ND

    #            res[k] += phis[k_ind]*res_loc
    #            #k += 2*Nf
    #            #if k>=ND:
    #            #    k-=ND
    #        n -= 2
    #        if n<0:
    #            n+=Nt

#    idxf_base = (-K//2+ND+Nf)%(2*Nf)
#
#    for k_off in range(0,ND,2*Nf):
#        n = ((K+2*k_off)//(2*Nf)-1)%(Nt)
#        assert n%2==1
#
#        for idxm in range(0,mult):
#            for idxf in range(0,2*Nf):
#                if idxf<idxf_base:
#                    itrf = 2*Nf-idxf_base+idxf
#                else:
#                    itrf = idxf-idxf_base
#                #k = (k_off+idxf)%ND
#                k = (itrf-K//2+n*Nf)%ND
#                assert 0<=k<ND
#                #if k!=(k_ind-K//2+n*Nf)%ND:
#                #    print(k_off,idxf,itrf,n,idxm,K,n*Nf,ND,k,(itrf-K//2+n*Nf)%ND)
#                k_ind = 2*Nf*idxm+itrf
#                assert 0<=k_ind<K
#                #assert k==(k_ind-K//2+n*Nf)%ND
#                res_loc = fft_fin_real[n,idxf]
#                res[k] += phis[k_ind]*res_loc
#
#            n -= 2
#            if n<0:
#                n+=Nt
        #    idxm =
#        for idxm in range(0,mult):
#            k_ind = 2*Nf*idxm+itrf
#            for n in range(0,Nt,2):
#                res_loc = fft_fin_real[n,idxf]
#                #k = (itrf-K//2+n*Nf)%ND
#                k = (k_ind-K//2+n*Nf)%ND
#                if k==0:
#                    print(idxf,idxm,n,k_ind)
#                res[k] += phis[k_ind]*res_loc
#                #k += 2*Nf
#                #if k>=ND:
#                #    k-=ND

#    for n in range(1,Nt,2):
#        idxf_base = (-K//2+ND+Nf)%(2*Nf)
#
#        mult = K//(2*Nf)
#        for idxf in range(0,2*Nf):
#            if idxf<idxf_base:
#                itrf = 2*Nf-idxf_base+idxf
#            else:
#                itrf = idxf-idxf_base
#            res_loc = fft_fin_real[n,idxf]
#            k = (itrf-K//2+n*Nf)%ND
#            for idxm in range(0,mult):
#                k_ind = 2*Nf*idxm+itrf
#                #k = (k_ind-K//2+n*Nf)%ND
#                res[k] += phis[k_ind]*res_loc
#                k += 2*Nf
#                if k>=ND:
#                    k-=ND

    #for idxf in range(0,idxf_base):
    #    res_loc = fft_fin_real[idxf]
    #    for idxm in range(0,mult):
    #        k_ind = 2*Nf*idxm+itrf
    #        k = (k_ind-K//2+n*Nf)%ND
    #        res[k] += phis[k_ind]*res_loc
    #    itrf += 1
            #idxf+=1
            #k += 1

            #if idxf==2*Nf:
            #    idxf = 0
            #if k==ND:
            #    k = 0


    #for k_ind in range(k_ind_min1,k_ind_max1):
    #    k = k_ind+(-K//2+n*Nf+ND)
    #    res_loc = fft_fin_real[idxf]
    #    res[k] += phis[k_ind]*res_loc
    #    idxf+=1
    #    if idxf>=2*Nf:
    #        idxf = 0

    #for k_ind in range(k_ind_min2,k_ind_max2):
    #    k = k_ind+(-K//2+n*Nf)
    #    res_loc = fft_fin_real[idxf]
    #    res[k] += phis[k_ind]*res_loc
    #    idxf+=1
    #    if idxf>=2*Nf:
    #        idxf = 0

    #for k_ind in range(k_ind_min3,k_ind_max3):
    #    k = k_ind+(-K//2+n*Nf-ND)
    #    res_loc = fft_fin_real[idxf]
    #    res[k] += phis[k_ind]*res_loc
    #    idxf+=1
    #    if idxf>=2*Nf:
    #        idxf = 0

@njit()
def pack_wave_time_helper(n,Nf,Nt,wave_in,afins):
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



@njit()
def inverse_wavelet_time_helper_naive(wave_in,phi,Nf,Nt,K):
    """inverse_wavelet_time helper naive"""
    ND=Nf*Nt
    res = np.zeros(ND)
    for n in range(0,Nt):
        k_min = max(n*Nf-K//2,0)
        k_max = min(n*Nf+K//2,ND)

        for k in range(k_min,k_max):

            k_ind = k-n*Nf+K//2
            mult_t = phi[k_ind]
            res_loc = 0.

            for m in range(1,Nf):
                if wave_in[n,m] == 0.:
                    continue

                if (m+n)%2:
                    prefactor = np.sin(np.pi/Nf*m*k)
                else:
                    prefactor = (-1)**(m*n)*np.cos(np.pi/Nf*m*k)

                res_loc += prefactor*wave_in[n,m]

            res[k] += mult_t*res_loc
    return res

