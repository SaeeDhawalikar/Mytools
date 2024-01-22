import numpy as np
import scipy.fft as fft
import scipy.stats as stats
import pandas as pd


class power(object):
    ''' different functions to calculate power sepctrum of discrete data'''
    
    def power_prem(FX,box_size, interlace_with_FX=None, Win_correct_scheme='CIC', grid_size=512):
        FK = np.fft.rfftn(FX) * (box_size/FX.shape[0])**FX.ndim
        if interlace_with_FX is not None:
            FK += np.fft.rfftn(interlace_with_FX) * (box_size/interlace_with_FX.shape[0])**interlace_with_FX.ndim
            FK /= 2

        K = np.array(np.meshgrid(*[np.fft.fftfreq(n, d=box_size/n) *2*np.pi for n in FX.shape[:-1]],
                                             np.fft.rfftfreq(FX.shape[-1], d=box_size/FX.shape[-1]) *2*np.pi))

        #since rfftn is used, the last axis will have half the k values as comared to the first 2 axes.

        assert FK.shape == K.shape[1:], "Reshape needed fftfreq"

        win_correct_power = ['NGP', 'CIC', 'TSC'].index(Win_correct_scheme) + 1
        k_nyq = np.pi * grid_size / box_size

        FK /= ( np.sinc(K[0]/(2*k_nyq)) * np.sinc(K[1]/(2*k_nyq)) * np.sinc(K[2]/(2*k_nyq)))**(win_correct_power)

        if interlace_with_FX is not None:
            FK /= ( 1 + np.exp(-np.pi*1j*K.sum(axis=0)/(2*k_nyq)) )/2

        k = np.sqrt((K**2).sum(axis=0))

        Pk = (FK.real**2 + FK.imag**2) / (box_size)**FX.ndim
        Pk[0,0,0] = 0


        return pd.DataFrame(data={'k':k.ravel(), 'Pk':Pk.ravel()}).groupby('k').mean().reset_index()
    
    def power_spectrum_uncorrected(field,l,pixels):
        """ This function produces the power spectrum P(k), for the given discrete density field.

        Inputs
        field: pixel X pixel X pixel array corresponding to the number density field
        l: length of the box
        pixels: number of grid cells per side while calculating the fourier transform

        Output
        Pk: power spectrum array
        k: wavenumber array
        """

        lpix=l/pixels #pixel size

        fou=fft.fftn(field)*lpix**3
        k_1d=np.fft.fftfreq(pixels, d=lpix)*2*np.pi  #k=2pi*nu
        K=np.array(np.meshgrid(k_1d, k_1d, k_1d))
        k=np.sqrt(np.sum(K*K, axis=0))


        k_nq=np.pi/lpix  #nyquist frequncy

        Pk=np.abs(fou)**2/(l**3)
        Pk[0,0,0]=0

        k=k.flatten()
        Pk=Pk.flatten()

        kbins = np.arange(np.min(k),np.max(k),np.abs(k[0]-k[1]))
        kvals = 0.5*(kbins[1:] + kbins[:-1]) #midpoints of the bins

        Abins, _, _ = stats.binned_statistic(k, Pk, statistic = "mean",bins = kbins)
        return pd.DataFrame({'k':kvals, 'pk': Abins})
    
    def power_spectrum(field,l,pixels, Win_correct_scheme='CIC', bins=50):
        """ This function produces the power spectrum P(k), for the given discrete density field.

        Inputs
        field: pixel X pixel X pixel array corresponding to the number density field
        l: length of the box
        pixels: number of grid cells per side while calculating the fourier transform

        Output
        pk: power spectrum array
        k: wavenumber array
        std: standard deviation
        N: number of counts per bin
        """
        n_corr = ['NGP', 'CIC'].index(Win_correct_scheme) + 1
        lpix=l/pixels #pixel size

        fou=fft.fftn(field)*lpix**3  #multiplying by lpix**3 so that dimensions of delta_k is [L**3]
        k_1d=np.fft.fftfreq(pixels, d=lpix)*2*np.pi  #k=2pi*nu
        K=np.array(np.meshgrid(k_1d, k_1d, k_1d)) #3D vector K
        k=np.sqrt(np.sum(K*K, axis=0)) #magnitude of the wavevector


        k_nq=np.pi/lpix  #nyquist frequncy in 1D

        #deconvolving the window function
        fou =fou/ ( np.sinc(K[0]/(2*k_nq)) * np.sinc(K[1]/(2*k_nq)) * np.sinc(K[2]/(2*k_nq)))**n_corr


        Pk=np.abs(fou)**2/(l**3) #dividing with l**3 so that dimensions of Pk are [L**3]
        # and also so that box size does not affect the power spectrum
        Pk[0,0,0]=0
        k[0,0,0]=1e-8  #setting a not zero value so that there is no problem while taking the log
        k=k.flatten()

        k_log=np.log10(k)
        Pk=Pk.flatten()

        #arranging bins logarithmically
        kbin_log=np.array(np.linspace(np.min(k_log),np.max(k_log), bins))

        #kbins = np.arange(np.min(k),np.max(k),np.abs(k[0]-k[1]))
        kvals_log = 0.5*(kbin_log[1:] + kbin_log[:-1]) #midpoints of the bins
        kvals=10**(kvals_log)

        Abins, _, _ = stats.binned_statistic(k_log, Pk, statistic = "mean",bins = kbin_log)
        std, _, _ =stats.binned_statistic(k_log, Pk, statistic="std", bins=kbin_log)
        N, _, _=stats.binned_statistic(k_log, Pk, statistic="count", bins=kbin_log)
        #removing nan values
        ind=np.argwhere(np.isnan(Abins))
        kvals=np.delete(kvals, ind)
        Abins=np.delete(Abins, ind)
        std=np.delete(std, ind)
        N=np.delete(N, ind)

        kvals[0]=0
        return pd.DataFrame({'k':kvals, 'pk':Abins, 'std':std, 'N':N})
    
    def power_spectrum_modified(field,l,pixels,Ng, Win_correct_scheme='CIC', bins1=20, bins2=20,shot_noise=False):
        """ This function produces the power spectrum P(k), for the given discrete density field.

        Inputs
        field: pixel X pixel X pixel array corresponding to the number density field
        l: length of the box
        pixels: number of grid cells per side while calculating the fourier transform
        Ng: number of halos/particles in the simulation
        bins1: bins below k=0.2
        bins2=bins above k=0.2

        Output
        pk: power spectrum array
        k: wavenumber array
        error: error on pk
        Nk: number of modes per bin
        """
        n_corr = ['NGP', 'CIC'].index(Win_correct_scheme) + 1
        lpix=l/pixels #pixel size

        fou=fft.fftn(field)*lpix**3  #multiplying by lpix**3 so that dimensions of delta_k is [L**3]
        k_1d=np.fft.fftfreq(pixels, d=lpix)*2*np.pi  #k=2pi*nu
        K=np.array(np.meshgrid(k_1d, k_1d, k_1d)) #3D vector K
        k=np.sqrt(np.sum(K*K, axis=0)) #magnitude of the wavevector


        k_nq=np.pi/lpix  #nyquist frequncy in 1D

        #deconvolving the window function
        fou =fou/ ( np.sinc(K[0]/(2*k_nq)) * np.sinc(K[1]/(2*k_nq)) * np.sinc(K[2]/(2*k_nq)))**n_corr


        Pk=np.abs(fou)**2/(l**3) #dividing with l**3 so that dimensions of Pk are [L**3]
        # and also so that box size does not affect the power spectrum
        
        
        Pk[0,0,0]=0
        k[0,0,0]=1e-8  #setting a not zero value so that there is no problem while taking the log
        k=k.flatten()

        k_log=np.log10(k)
        Pk=Pk.flatten()

        #arranging bins logarithmically
        kbin1_log=np.array(np.linspace(np.min(k_log),-0.8, bins1))
        epsilon=(0.2-np.min(k_log))/bins1
        kbin2_log=np.array(np.linspace(-0.8+epsilon, np.max(k_log), bins2))

        kbin_log=np.concatenate((kbin1_log, kbin2_log))

        #kbins = np.arange(np.min(k),np.max(k),np.abs(k[0]-k[1]))
        kvals_log = 0.5*(kbin_log[1:] + kbin_log[:-1]) #midpoints of the bins
        kvals=10**(kvals_log)

        kbin=10**kbin_log #edges of the bins
        Vk=np.zeros(len(kvals), dtype=float) #volume of the bin
        for i in range(len(Vk)):
            Vk[i]=(4/3)*np.pi*((kbin[i+1])**3-(kbin[i])**3)
        V=l**3
        Nk=V*Vk/(2*np.pi)**3

        #errors on k
        k_low=np.zeros(len(kvals),  dtype=float)
        k_high=np.zeros(len(kvals), dtype=float)
        for i in range(len(kvals)):
            k_low[i]=kbin[i]
            k_high[i]=kbin[i+1]

        Abins, _, _ = stats.binned_statistic(k_log, Pk, statistic = "mean",bins = kbin_log)
        
        #removing nan values
        ind=np.argwhere(np.isnan(Abins))
        kvals=np.delete(kvals, ind)
        Abins=np.delete(Abins, ind)
        Nk=np.delete(Nk, ind)
        k_high=np.delete(k_high, ind)
        k_low=np.delete(k_low, ind)
        
        #removing shot noise
        if shot_noise is False:
            Abins=Abins-l**3/Ng

        P_dim=(2*np.pi)**3*Abins/l**3 #dimensionless power spectrum
        
   
        P_dim[0]=1e-6
        delp_p2=2/Nk+ 4/(Ng*Nk*P_dim)+2/(Ng**2*Nk*P_dim**2)
        delp_p=np.sqrt(delp_p2)
        delp=delp_p*Abins
        kvals[0]=0
        
        return pd.DataFrame({'k':kvals, 'pk':Abins, 'error':delp, 'Nk': Nk})
