""""harness for computing inverse wavelet transform, take input .dat file in wavelet domain (Nt rows by Nf columns)
and write to .dat file in frequency domain (columns frequency, real part(h(f)), imag part(h(f))"""
import sys
from time import perf_counter
import numpy as np
np.random.seed(315)
import scipy.stats
import fft_funcs as fft

from transform_freq_funcs import tukey
from inverse_wavelet_funcs import inverse_wavelet_freq
from inverse_wavelet_time_funcs import inverse_wavelet_time
from transform_time_funcs import transform_wavelet_time
from transform_freq_funcs import transform_wavelet_freq

if __name__=='__main__':
    #assume input .dat file is Nt rows by Nf columns
    #if len(sys.argv)!=4:
    #    print("transform_time.py filename_in filename_freq_out dt")
    #    sys.exit(1)

    #transform parameters
    #file_in = sys.argv[1]
    #file_out = sys.argv[2]

    #dt = np.float64(sys.argv[3])
    dt = 5.
    Nt = 512
    Nf = 2048#12318
    ND = Nt*Nf
    DT = Nf*dt
    Tobs = ND*dt
    alpha = 8*(2.0*(4.0*DT)/Tobs)

    wave_in = np.random.normal(0.,1.,(Nt,Nf))

    inverse_wavelet_time(wave_in,Nf,Nt,dt,mult=32)
    #import sys
    #sys.exit()
    n_run = 10
    t0 = perf_counter()
    for itrm in range(0,n_run):
        inverse_wavelet_time(wave_in,Nf,Nt,dt,mult=32)
    t1 = perf_counter()
    print('got time domain inverse transform in %5.3fs'%((t1-t0)/n_run))


    #import sys
    #sys.exit()

    #get a wavelet representation of a signal
    print('begin loading data file')
    t0 = perf_counter()
    #ts = np.arange(0,Nf*Nt)*dt
    ts = np.arange(0,ND)*dt
    fs = np.arange(0,ND//2+1)*1/(Tobs)
    m = 51
    #data = np.sin(2*np.pi*m*ts/(2*DT))
    #tukey(data, alpha, ND)
    data = np.random.normal(0.,1.,ND)
    data_f = fft.rfft(data)
    #wave_in2 = transform_wavelet_time(data,Nf,Nt,ts,mult=64)
    #wave_in3 = transform_wavelet_freq(fft.rfft(data),Nf,Nt,fs)
    #import sys
    #sys.exit()
    #import sys
    #sys.exit()
    wave_in2 = transform_wavelet_time(data,Nf,Nt,ts,mult=32)
    wave_in1 = transform_wavelet_freq(data_f,Nf,Nt,fs)

    #wave_in = np.loadtxt(file_in)
    t1 = perf_counter()
    print('loaded input file in %5.3fs'%(t1-t0))

    #Nt = wave_in.shape[0]
    #Nf = wave_in.shape[1]

    ND = Nt*Nf
    Tobs = dt*ND

    t0 = perf_counter()
    transform_wavelet_time(data,Nf,Nt,ts,mult=32)
    t1 = perf_counter()
    print('got time domain forward transform in %5.3fs'%(t1-t0))

    t0 = perf_counter()
    transform_wavelet_freq(fft.rfft(data),Nf,Nt,fs)
    t1 = perf_counter()
    print('got time domain forward transform via fftin %5.3fs'%(t1-t0))

    t0 = perf_counter()
    transform_wavelet_freq(data_f,Nf,Nt,fs)
    t1 = perf_counter()
    print('got frequency domain forward transform in %5.3fs'%(t1-t0))

    #time and frequency grids

    signal_freq1 = inverse_wavelet_freq(wave_in1,Nf,Nt,dt)
    signal_freq2 = inverse_wavelet_freq(wave_in2,Nf,Nt,dt)

    t0 = perf_counter()
    signal_freq1 = inverse_wavelet_freq(wave_in1,Nf,Nt,dt)
    t1 = perf_counter()


    print('got frequency domain inverse transform in %5.3fs'%(t1-t0))

    #signal_freq3 = inverse_wavelet_freq(wave_in3,Nf,Nt,dt)
    #import sys
    #sys.exit()

    t2 = perf_counter()
    signal_time1 = fft.irfft(signal_freq1)
    signal_time2 = fft.irfft(signal_freq2)
    #signal_time3 = fft.irfft(signal_freq3)
    t3 = perf_counter()

    print('got fourier transform in %5.3fs'%(t3-t2))

    signal_time_alt = inverse_wavelet_time(wave_in1,Nf,Nt,dt,mult=64)
    signal_time_alt2 = inverse_wavelet_time(wave_in2,Nf,Nt,dt,mult=64)
    #import sys
    #sys.exit()

    t2 = perf_counter()
    inverse_wavelet_time(wave_in1,Nf,Nt,dt,mult=64)
    t3 = perf_counter()
    signal_freq_alt = fft.rfft(signal_time_alt)
    signal_freq_alt2 = fft.rfft(signal_time_alt2)

    wave_alt = transform_wavelet_freq(signal_freq_alt,Nf,Nt,fs)

    print('got time domain inverse transform in %5.3fs'%(t3-t2))

    mean_sig = np.mean(signal_time1)
    std_sig = np.std(signal_time1)
    std_std_time = np.std(signal_time1)*np.sqrt(2/ND)

    print('mean %+.3e std %10.8f, sigma (mean-<mean>) %5.3f sigma (std-<std>) %5.3f'%(mean_sig,std_sig,mean_sig*np.sqrt(ND),(std_sig-1.)/std_std_time))
    print('consistency of time domain signal with normal',scipy.stats.normaltest(signal_time1))
    print('consistency of real part of frequency domain signal with normal',scipy.stats.normaltest(np.real(signal_freq1)))
    print('consistency of imag part of frequency domain signal with normal',scipy.stats.normaltest(np.imag(signal_freq1)))

    t4 = perf_counter()
    #np.savetxt(file_out,np.vstack([fs,np.real(signal_freq),np.imag(signal_freq)]).T)
    t5 = perf_counter()
    print('saved file in %5.3fs'%(t5-t4))

    print('rmstfd',np.linalg.norm(signal_time1-data)/np.sqrt(ND))
    print('rmstd ',np.linalg.norm(signal_time_alt-data)/np.sqrt(ND))
    print('rmstt ',np.linalg.norm(signal_time_alt-signal_time_alt2)/np.sqrt(ND))
    #print('rms2',np.linalg.norm(signal_time2-data)/np.sqrt(ND))
    #print('rms3',np.linalg.norm(signal_time3-data)/np.sqrt(ND))


    do_plots = True
    if do_plots:
        import matplotlib.pyplot as plt
        plt.loglog(fs,np.abs(signal_freq1))
        plt.loglog(fs,np.abs(signal_freq_alt))
        plt.loglog(fs,np.abs(signal_freq_alt2))
        #plt.loglog(fs,np.abs(signal_freq2))
        #plt.loglog(fs,np.abs(fft.rfft(data)))
        #plt.loglog(fs,np.abs(data_f))
        #plt.loglog(fs,np.abs(fft.rfft(signal_time_alt)))
        plt.xlabel(r"$f$ (Hz)")
        plt.ylabel(r"$\tilde{h}(f)$")
        plt.show()

        #plt.loglog(np.abs(signal_freq1[3*Nt:7*Nt]))
        #plt.loglog(np.abs(signal_freq_alt[3*Nt:7*Nt]))
        #plt.show()

        #plt.semilogx(np.angle(signal_freq1[3*Nt:7*Nt]))
        #plt.semilogx(np.angle(signal_freq_alt[3*Nt:7*Nt]))
        #plt.show()

        #wave_diff = wave_alt-wave_in1
        #wave_diff[wave_diff==0.] = np.nan
        #plt.imshow(np.rot90(wave_diff),aspect='auto')
        #plt.show()

        #plt.semilogx(fs[-3*Nt:],np.angle(signal_freq1[-3*Nt:]))
        ##plt.loglog(fs,np.abs(signal_freq2))
        #plt.semilogx(fs[-3*Nt:],np.angle(fft.rfft(data)[-3*Nt:]))
        ##plt.loglog(fs,np.abs(data_f))
        ##plt.loglog(fs,np.abs(fft.rfft(signal_time_alt)))
        #plt.xlabel(r"$f$ (Hz)")
        #plt.ylabel(r"$\tilde{h}(f)$")
        #plt.show()

        do_time = True
        if do_time:
            plt.plot(ts,signal_time1)
            plt.plot(ts,signal_time_alt)
            #plt.plot(ts,data)
            #plt.plot(ts,fft.irfft(data_f))
            #plt.plot(ts,signal_time_alt)
            plt.xlabel(r"$t$ (s)")
            plt.ylabel(r"$h(t)$")
            plt.show()
