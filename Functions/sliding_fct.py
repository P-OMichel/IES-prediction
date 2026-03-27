'''
Functions computed on sliding windows with a specific window size and step in between two consecutive windows
'''
import numpy as np
import hfda
import pyinform
import antropy as ant
from Functions.entropy import *
from Functions.spectral_power_methods import *
import pywt
from Functions.utils import *
from Functions.compute_fragmentation import compute_fragmentation_level,compute_fragmentation_norm,compute_fragmentation
from Functions.compute_state import *
from Functions.suppressions import compute_suppression_mask_rate,detect_suppressions_power
from Spindles.spindles_f_distribution import get_spindles_barycenter,get_low_f_proportion,get_low_f_proportion_power,get_h_l_f_quantile
from Spindles.spindles_fragmentation import get_spindles_fragmentation
from Spindles.get_spindle_type import spindle_off_proportion, get_envelopes_power, get_envelopes, get_spindle_type
from statsmodels.tsa.seasonal import seasonal_decompose

def up_zero(y):
    M=np.min(y[np.where(y<=0)[0]])
    return y-M

def mean_IES_time(signal,t,mask_size,fs,threshold,list_morph,f_int,method,Ws,step):
    '''
    average time of an IES within the window
    '''
    # create a list of all the sliding windows
    windows=[signal[i:i+Ws] for i in range(0,len(signal)-Ws,step)]   

    mean_IES_list=[]
    for win in windows:
        mask_IES=compute_suppression_mask_rate(win,mask_size,fs,threshold,list_morph,f_int,method)[0]
        pos_IES=detect_pos_1(mask_IES)
        if pos_IES!=[]:
            length_IES=np.array([(pos[1]-pos[0]) for pos in pos_IES])/fs
            mean_IES_list.append(np.mean(length_IES))
        else:
            mean_IES_list.append(0)
    
    t_list=[t[Ws+i*step] for i in range(len(windows))] 

    return t_list, mean_IES_list


def crossing(signal,t,Ws,step):
    '''
    Number of times the signal crosses 0
    '''

    # create a list of all the sliding windows
    windows=[signal[i:i+Ws] for i in range(0,len(signal)-Ws,step)]   

    # number of time the signal crosses 0
    N_crosses_list=[zero_crossing(win) for win in windows]

    # associated time list (time bin is for the end of a window)
    t_list=[t[Ws+i*step] for i in range(len(windows))] 

    return t_list,N_crosses_list
    

def get_states(signal,t,Ws,step,fs,list_freq_int,h,quantile,threshold,mode='same',threshold_power=[40,1.5,3],threshold_frag=[0.05,0.15]):

    # create a list of all the sliding windows
    windows=[signal[i:i+Ws] for i in range(0,len(signal)-Ws,step)]

    # state, list_frag,list_power
    output=[get_state(win,fs,list_freq_int,h,quantile,threshold,mode,threshold_power,threshold_frag) for win in windows]

    # associated time list (time bin is for the end of a window)
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    return t_list, output

def get_states_supp(signal,t,Ws,step,fs,list_freq_int,threshold_power,threshold_frag,mask_size,threshold,list_morph,f_int,method):

    # create a list of all the sliding windows
    windows=[signal[i:i+Ws] for i in range(0,len(signal)-Ws,step)]

    # state, list_frag,list_power
    output=[get_state_supp(win,fs,list_freq_int,threshold_power,threshold_frag,mask_size,threshold,list_morph,f_int,method) for win in windows]

    # associated time list (time bin is for the end of a window)
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    return t_list, output

def get_full_states_supp(signal,t,Ws,step,fs,list_freq_int,threshold_power,threshold_frag,mask_size,threshold,list_morph,f_int,method):

    # create a list of all the sliding windows
    windows=[signal[i:i+Ws] for i in range(0,len(signal)-Ws,step)]

    # state, list_frag,list_power
    output=[get_full_state_supp(win,fs,list_freq_int,threshold_power,threshold_frag,mask_size,threshold,list_morph,f_int,method) for win in windows]

    # associated time list (time bin is for the end of a window)
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    return t_list, output

def power_1D(signal,t,Ws,step):
    '''
    Output: 
    P_list: 1D numpy array
    '''
    # create a list of all the sliding windows
    windows=[signal[i:i+Ws]**2 for i in range(0,len(signal)-Ws,step)]
    # compute the mean power in each window
    P_list=np.array([np.median(win) for win in windows])
    # associated time list (time bin is for the end of a window)
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    return t_list,P_list

def power_nD(signals,t,Ws,step):
    '''
    Input:
    signals: nD numpy array
    Output:
    P_list: numpy array of same number of line as signals
    '''
    # create a list of all the sliding windows
    windows=[signals[:,i:i+Ws]**2 for i in range(0,np.shape(signals)[1]-Ws,step)]
    # compute the mean power in each window for axis 1
    P_list=np.transpose([np.median(win,axis=1) for win in windows])
    # associated time list (time bin is for th end of a window)
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    return t_list, P_list

def hfda_1D(signal,t,Ws,step,kmax):

    # create a list of all the sliding windows
    windows=[signal[i:i+Ws] for i in range(0,len(signal)-Ws,step)]
    # compute hfda
    hfda_list=np.array([hfda.measure(win,kmax) for win in windows])
    # associated time list (time bin is for th end of a window)
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    return t_list,hfda_list

def hfda_nD(signals,t,Ws,step,kmax):

    # create a list of all the sliding windows
    windows=[signals[:,i:i+Ws] for i in range(0,np.shape(signals)[1]-Ws,step)]
    # compute hfda along all lines of the sliding windows 
    hfda_list=np.array([[hfda.measure(win[i],kmax) for win in windows]for i in range(np.shape(signals)[0])])
    # associated time list (time bin is for th end of a window)
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    return t_list,hfda_list

##############from PyInform
def entropy_1D(signal,t,Ws,step,k):
    
    # create a list of all the sliding windows
    windows=[signal[i:i+Ws] for i in range(0,len(signal)-Ws,step)]
    # compute hfda
    metric_list=np.array([pyinform.block_entropy(up_zero(win),k) for win in windows])
    # associated time list (time bin is for th end of a window)
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    return t_list,metric_list

def entropy_1D_norm(signal,t,Ws,step,k,Max):
    
    # create a list of all the sliding windows
    windows=[signal[i:i+Ws] for i in range(0,len(signal)-Ws,step)]
    # pushes the windows values above zero
    up_zero_windows=[up_zero(win) for win in windows]
    # normalize the windows
    norm_windows=[win/np.max(win)*Max for win in up_zero_windows]
    # compute block entropy
    metric_list=np.array([pyinform.block_entropy(win,k) for win in norm_windows])
    # associated time list (time bin is for th end of a window)
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    return t_list,metric_list

def entropy_nD(signals,t,Ws,step,k):

    # create a list of all the sliding windows
    windows=[signals[:,i:i+Ws] for i in range(0,np.shape(signals)[1]-Ws,step)]
    # compute hfda along all lines of the sliding windows 
    hfda_list=np.array([[pyinform.block_entropy(up_zero(win[i]),k) for win in windows]for i in range(np.shape(signals)[0])])
    # associated time list (time bin is for th end of a window)
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    return t_list,hfda_list

def active_info_1D(signal,t,Ws,step,k):
    
    # create a list of all the sliding windows
    windows=[signal[i:i+Ws] for i in range(0,len(signal)-Ws,step)]
    # compute hfda
    hfda_list=np.array([pyinform.active_info(up_zero(win),k) for win in windows])
    # associated time list (time bin is for th end of a window)
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    return t_list,hfda_list

def active_info_nD(signals,t,Ws,step,k):

    # create a list of all the sliding windows
    windows=[signals[:,i:i+Ws] for i in range(0,np.shape(signals)[1]-Ws,step)]
    # compute hfda along all lines of the sliding windows 
    hfda_list=np.array([[pyinform.active_info(win[i],k) for win in windows]for i in range(np.shape(signals)[0])])
    # associated time list (time bin is for th end of a window)
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    return t_list,hfda_list

################## from Antropy
def spectral_entropy_1D(signal,t,fs,Ws,step):
    
    # create a list of all the sliding windows
    windows=[signal[i:i+Ws] for i in range(0,len(signal)-Ws,step)]
    # compute hfda
    spect_ent_list=np.array([ant.spectral_entropy(win,fs,method='welch',normalize=True) for win in windows])
    # associated time list (time bin is for th end of a window)
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    return t_list,spect_ent_list

def spectral_entropy_nD(signals,t,Ws,step,k):

    # create a list of all the sliding windows
    windows=[signals[:,i:i+Ws] for i in range(0,np.shape(signals)[1]-Ws,step)]
    # compute hfda along all lines of the sliding windows 
    spect_ent_list=np.array([[ant.spectral_entropy(win[i],fs) for win in windows]for i in range(np.shape(signals)[0])])
    # associated time list (time bin is for th end of a window)
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    return t_list,spect_ent_list

def permutation_entropy_1D(signal,t,Ws,step):
    
    # create a list of all the sliding windows
    windows=[signal[i:i+Ws] for i in range(0,len(signal)-Ws,step)]
    # compute hfda
    ent_list=np.array([ant.perm_entropy(win,normalize=True) for win in windows])
    # associated time list (time bin is for th end of a window)
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    return t_list,ent_list


def svd_entropy_1D(signal,t,Ws,step):
    
    # create a list of all the sliding windows
    windows=[signal[i:i+Ws] for i in range(0,len(signal)-Ws,step)]
    # compute hfda
    ent_list=np.array([ant.svd_entropy(win,normalize=True) for win in windows])
    # associated time list (time bin is for th end of a window)
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    return t_list,ent_list

def approximate_entropy_1D(signal,t,Ws,step):
    
    # create a list of all the sliding windows
    windows=[signal[i:i+Ws] for i in range(0,len(signal)-Ws,step)]
    # compute hfda
    ent_list=np.array([ant.app_entropy(win) for win in windows])
    # associated time list (time bin is for th end of a window)
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    return t_list,ent_list

def sample_entropy_1D(signal,t,Ws,step):
    
    # create a list of all the sliding windows
    windows=[signal[i:i+Ws] for i in range(0,len(signal)-Ws,step)]
    # compute hfda
    ent_list=np.array([ant.sample_entropy(win) for win in windows])
    # associated time list (time bin is for th end of a window)
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    return t_list,ent_list

def Hjorth_1D(signal,t,Ws,step):

    # create a list of all the sliding windows
    windows=[signal[i:i+Ws] for i in range(0,len(signal)-Ws,step)]
    # compute hfda
    metric_list=np.array([ant.hjorth_params(win) for win in windows])
    # associated time list (time bin is for th end of a window)
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    return t_list,metric_list

def Katz_1D(signal,t,Ws,step):

    # create a list of all the sliding windows
    windows=[signal[i:i+Ws] for i in range(0,len(signal)-Ws,step)]
    # compute hfda
    metric_list=np.array([ant.katz_fd(win) for win in windows])
    # associated time list (time bin is for th end of a window)
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    return t_list,metric_list

def Higuchi_1D(signal,t,Ws,step,k):

    # create a list of all the sliding windows
    windows=[signal[i:i+Ws] for i in range(0,len(signal)-Ws,step)]
    # compute hfda
    metric_list=np.array([ant.higuchi_fd(win,k) for win in windows])
    # associated time list (time bin is for th end of a window)
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    return t_list,metric_list

def petrosian_1D(signal,t,Ws,step):

    # create a list of all the sliding windows
    windows=[signal[i:i+Ws] for i in range(0,len(signal)-Ws,step)]
    # compute hfda
    metric_list=np.array([ant.petrosian_fd(win) for win in windows])
    # associated time list (time bin is for th end of a window)
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    return t_list,metric_list

def zeros_crossing_1D(signal,t,Ws,step):

    # create a list of all the sliding windows
    windows=[signal[i:i+Ws] for i in range(0,len(signal)-Ws,step)]
    # compute hfda
    metric_list=np.array([ant.num_zerocross(win) for win in windows])
    # associated time list (time bin is for th end of a window)
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    return t_list,metric_list

def dfa_1D(signal,t,Ws,step):

    # create a list of all the sliding windows
    windows=[signal[i:i+Ws] for i in range(0,len(signal)-Ws,step)]
    # compute hfda
    metric_list=np.array([ant.detrended_fluctuation(win) for win in windows])
    # associated time list (time bin is for th end of a window)
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    return t_list,metric_list

############## From Functions.entropy

def Shannon_entropy(signal,t,Ws,step,interval):

    # create a list of all the sliding windows
    windows=[signal[i:i+Ws] for i in range(0,len(signal)-Ws,step)]
    # compute metric
    metric_list=np.array([entropy(win,interval) for win in windows])
    # associated time list (time bin is for th end of a window)
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    return t_list,metric_list

def efficiency_1D(signal,t,Ws,step,interval):

    # create a list of all the sliding windows
    windows=[signal[i:i+Ws] for i in range(0,len(signal)-Ws,step)]
    # compute metric
    metric_list=np.array([efficiency(win,interval) for win in windows])
    # associated time list (time bin is for th end of a window)
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    return t_list,metric_list

def hist_efficiency_1D(signal,t,Ws,step,Nbins):

    # create a list of all the sliding windows
    windows=[signal[i:i+Ws] for i in range(0,len(signal)-Ws,step)]
    # compute metric
    metric_list=np.array([hist_efficiency(win,Nbins) for win in windows])
    # associated time list (time bin is for th end of a window)
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    return t_list,metric_list



################## sliding entropy on wavelet coeff and reconstricted wavelet coeff
def entropy_wavelet(signal,t,Ws,step,wavelet,mode,level,k):
    '''
    Compute the entropy of the wavelet coefficient per sliding windows
    '''

    # create a list of all the sliding windows
    windows=[signal[i:i+Ws] for i in range(0,len(signal)-Ws,step)]
    # DWT
    sliding_coeffs=[pywt.wavedec(win,wavelet,mode,level) for win in windows]
    # compute block entropy (no normalization)
    entropy_list=np.array([[pyinform.block_entropy(np.abs(coeffs[i]),k) for coeffs in sliding_coeffs] for i in range(len(sliding_coeffs[0]))])
    # timestamps associated to each windows
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    return t_list, entropy_list

def wavelet_DWT(signal,t,Ws,step,wavelet,mode,level):
    '''
    Makes the DWT applied on sliding windows through the signal
    Output:
    sliding_coeffs=[coeffs0,coeff1,...] where coeffsi is the list of the coefficients of the ith window (starting from 0)
    '''

    # create a list of all the sliding windows
    windows=[signal[i:i+Ws] for i in range(0,len(signal)-Ws,step)]
    # DWT
    sliding_coeffs=[pywt.wavedec(win,wavelet,mode,level) for win in windows]

    t_list=[t[Ws+i*step] for i in range(len(windows))]

    return t_list,sliding_coeffs

def wavelet_coeffs_entropy(sliding_coeffs,k):
    '''
    Computes the entropy on all coefficients per sliding windows
    Outputs:
    entropy_list: numpy array of shape (level of DWT, number of sliding windows)
    '''
    N_line=len(sliding_coeffs[0])
    entropy_list=np.array([[pyinform.block_entropy(np.abs(coeffs[i]),k) for coeffs in sliding_coeffs] for i in range(N_line)])

    return entropy_list

def wavelet_IDWT(sliding_coeffs,wavelet,level):
    '''
    Makes the IDWT on all coeffs using only them as non zero
    '''
    sliding_a_rec_signal=[pywt.upcoef('a',coeffs[0],wavelet) for coeffs in sliding_coeffs]
    sliding_d_rec_signals=[[pywt.upcoef('d',coeffs[i],wavelet) for coeffs in sliding_coeffs] for i in range(1,level+1)]

    return sliding_a_rec_signal,sliding_d_rec_signals

def wavelet_rec_signals_entropy(sliding_a_rec_signal,sliding_d_rec_signals,k):
    
    N_line=len(sliding_d_rec_signals[0])
    a_rec_signal_entropy=np.array([[pyinform.block_entropy(up_zero(a_signal),k) for a_signal in sliding_a_rec_signal]])
    d_rec_signal_entropy=np.transpose([[pyinform.block_entropy(up_zero(d_signal[i]),k) for d_signal in sliding_d_rec_signals] for i in range(N_line)])

    entropy_list=np.concatenate((a_rec_signal_entropy,d_rec_signal_entropy),axis=0)

    return entropy_list



#################sliding quantile functions ####################

def frequency_quantiles(quantiles,y,fs,cut_f,nfft,overlap,t,Ws,step):
    '''
    quantiles: [q1,q2,q3,...] 
    y: 1D numpy array
    fs: frequency sampling step
    nfft: number of points used in a sliding window for nperseg and for the FFT
    overlap: overlap of consecutive windows
    '''

    # get the different sliding windows
    windows=[y[i:i+Ws] for i in range(0,len(y)-Ws,step)]

    # compute the spectrograms
    spectrograms=[sc.signal.spectrogram(win,fs,nperseg=nfft,noverlap=overlap)[2] for win in windows]

    # compute spectrogram on first window only to get the delta f
    f_spectro=sc.signal.spectrogram(windows[0],fs,nperseg=nfft,noverlap=overlap)[0]
    delta_f=f_spectro[-1]-f_spectro[-2]
    
    # Compute Fourier spectrum (done this way as we can easily choose a frequency resolution in the spectrogram)
    F_spectrums=[np.mean(spectro[:int(cut_f/delta_f),:],axis=1) for spectro in spectrograms]

    # PCS
    pcs_list=[np.cumsum(F_spectrum) for F_spectrum in F_spectrums]
    pcs_list=[pcs/pcs[-1] for pcs in pcs_list]

    # get the frequencies associated to each quantile
    f_quantiles=[]
    for q in quantiles:
        f_quantiles.append([np.argmin(np.abs(pcs-q))*delta_f for pcs in pcs_list])

    # associated time list (time bin is for the end of a window)
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    return f_quantiles,t_list

def integral_f_quantiles(y,fs,start_f,cut_f,nfft,overlap,t,Ws,step):
    '''
    Returns the integral of the frequency according to quantile function.

    y: 1D numpy array
    fs: frequency sampling step
    nfft: number of points used in a sliding window for nperseg and for the FFT
    overlap: overlap of consecutive windows
    '''

    # get the different sliding windows
    windows=[y[i:i+Ws] for i in range(0,len(y)-Ws,step)]

    # compute the spectrograms
    spectrograms=[sc.signal.spectrogram(win,fs,nperseg=nfft,noverlap=overlap)[2] for win in windows]

    # compute spectrogram on first window only to get the delta f
    f_spectro=sc.signal.spectrogram(windows[0],fs,nperseg=nfft,noverlap=overlap)[0]
    delta_f=f_spectro[-1]-f_spectro[-2]
    
    # Compute Fourier spectrum (done this way as we can easily choose a frequency resolution in the spectrogram)
    F_spectrums=[np.mean(spectro[int(start_f/delta_f):int(cut_f/delta_f),:],axis=1) for spectro in spectrograms]

    # PCS
    pcs_list=[np.cumsum(F_spectrum) for F_spectrum in F_spectrums]
    pcs_list=[pcs/pcs[-1] for pcs in pcs_list]

    # get the frequencies associated to each quantile
    integral_list=[np.trapz(y=f_spectro[int(start_f/delta_f):int(cut_f/delta_f)],x=pcs) for pcs in pcs_list]
    
    # associated time list (time bin is for the end of a window)
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    return t_list, integral_list



def count_extremum_norm(y,t,Ws,step,fs):
    '''
    count the number of extremum and divide by window length to have it per second
    '''
    # get the different sliding windows
    windows=[y[i:i+Ws] for i in range(0,len(y)-Ws,step)]

    List_N_extremum=[]
    for win in windows:
        L_min,L_max=find_extremum(win)
        List_N_extremum.append(len(L_min)+len(L_max))
    List_N_extremum=np.array(List_N_extremum)
    List_N_extremum=List_N_extremum/Ws*fs

    # associated time list (time bin is for the end of a window)
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    return List_N_extremum,t_list

################## Fragmentation #################

def suppressions(signal,t,Ws,step,fs,threshold,list_morph,f_int,method,*arg):

    windows=[signal[i:i+Ws] for i in range(0,len(signal)-Ws,step)]

    supp_list=[compute_suppression_mask_rate(win,Ws,fs,threshold,list_morph,f_int,method,*arg)[2:] for win in windows]

    t_list=[t[Ws+i*step] for i in range(len(windows))]

    return t_list, np.transpose(supp_list)

def fragmentation_level(env_data,t,Ws,step,h,quantile,threshold):
    ''' 
    Ws: sliding window size
    '''
    windows=[env_data[:,i:i+Ws] for i in range(0,len(env_data[0])-Ws,step)]

    frag_list=[compute_fragmentation_level(win,h,quantile,threshold) for win in windows]

    t_list=[t[Ws+i*step] for i in range(len(windows))]

    return t_list,np.transpose(frag_list)


def fragmentation_norm(env_data,t,Ws,step,h,quantile,threshold):
    ''' 
    Ws: sliding window size
    '''
    windows=[env_data[:,i:i+Ws] for i in range(0,len(env_data[0])-Ws,step)]

    frag_list=[compute_fragmentation_norm(win,h,quantile,threshold) for win in windows]

    t_list=[t[Ws+i*step] for i in range(len(windows))]

    return t_list, np.transpose(frag_list)

def get_fragmentation(signal,t,Ws,step,h,quantile,threshold,mode='same'):

    # create a list of all the sliding windows
    windows=[signal[i:i+Ws] for i in range(0,len(signal)-Ws,step)]
    # compute fragmentation
    fragmentation_list=np.array([compute_fragmentation(win,h,quantile,threshold,mode)*100 for win in windows])
    # associated time list (time bin is for th end of a window)
    t_list=[t[Ws+i*step] for i in range(len(windows))]    

    return t_list, fragmentation_list

#---------------- ks-test between 2 consecutive windows -------------------# 

def ks_test(y,t,Ws,step):

    windows=[y[i:i+Ws] for i in range(0,len(y)-Ws,step)]

    # ks-test on 2 consecutive windows ( no computation on first window)
    kstest_list=[sc.stats.kstest(windows[i],windows[i+1])[0] for i in range(len(windows)-1)]


    t_list=[t[Ws+i*step] for i in range(len(windows))]
    t_list=t_list[1:]  # get rid of timestamp form end of first window

    return t_list,kstest_list


# --------------------- PSD barycenter ------------------------------#

def PSD_barycenter(y,t,Ws,step,fs,height):

    windows=[y[i:i+Ws] for i in range(0,len(y)-Ws,step)] 
    barycenter_list=[get_PSD_barycenter(win,fs,height)[-1] for win in windows] 
    t_list=[t[Ws+i*step] for i in range(len(windows))] 

    return t_list,barycenter_list


# -------------------- Spindles segmentation and PSD barycenter ------------------#

def spindles_PSD_barycenter(y,t,Ws,step,fs,lower_criteria,upper_criteria,T,N_bins,T_dist):

    windows=[y[i:i+Ws] for i in range(0,len(y)-Ws,step)] 
    barycenter_list=[get_spindles_barycenter(win,fs,lower_criteria,upper_criteria,T,N_bins,T_dist)[-1] for win in windows] 
    t_list=[t[Ws+i*step] for i in range(len(windows))] 

    return t_list,barycenter_list

def spindles_low_f_proportion(y,t,Ws,step,fs,lower_criteria,upper_criteria,T):

    windows=[y[i:i+Ws] for i in range(0,len(y)-Ws,step)] 
    low_f_proportion_list=[get_low_f_proportion(win,fs,lower_criteria,upper_criteria,T) for win in windows] 
    t_list=[t[Ws+i*step] for i in range(len(windows))] 

    return t_list, low_f_proportion_list

def spindles_low_f_proportion_power(y,t,Ws,step,fs,lower_criteria,upper_criteria,T,N):

    windows=[y[i:i+Ws] for i in range(0,len(y)-Ws,step)] 
    low_f_proportion_list=[get_low_f_proportion_power(win,fs,lower_criteria,upper_criteria,T,N) for win in windows] 
    t_list=[t[Ws+i*step] for i in range(len(windows))] 

    return t_list, low_f_proportion_list

def spindles_f_quantiles(y,t,Ws,step,fs,lower_criteria,upper_criteria,T,l_quantile,h_quantile):

    windows=[y[i:i+Ws] for i in range(0,len(y)-Ws,step)] 
    f_quantile_list=[get_h_l_f_quantile(win,fs,lower_criteria,upper_criteria,T,l_quantile,h_quantile) for win in windows] 
    t_list=[t[Ws+i*step] for i in range(len(windows))] 

    return t_list, np.transpose(f_quantile_list)

def spindles_fragmentation(y,t,Ws,step,lower_criteria,upper_criteria):

    windows=[y[i:i+Ws] for i in range(0,len(y)-Ws,step)] 
    fragmentation_list=[get_spindles_fragmentation(win,lower_criteria,upper_criteria) for win in windows] 
    t_list=[t[Ws+i*step] for i in range(len(windows))] 

    return t_list, np.transpose(fragmentation_list)

def diff_quantiles(y,t,Ws,step,q0,q1):

    windows=[y[i:i+Ws] for i in range(0,len(y)-Ws,step)] 
    diff_quantile_list=[(np.quantile(win,q0)-np.quantile(win,q1))/np.quantile(win,q1) for win in windows] 
    t_list=[t[Ws+i*step] for i in range(len(windows))] 

    return t_list, diff_quantile_list

def supp(y,Ws,step,fs,threshold,list_morph,f_int,method):

    windows=[y[i:i+Ws] for i in range(0,len(y)-Ws,step)]
    pos_IES, pos_alpha_supp = [], []
    for i in range(len(windows)):
        IES,alpha_supp=compute_suppression_mask_rate(windows[i],len(windows[i]),fs,threshold,list_morph,f_int,method)[:2]
        IES=detect_pos_1(IES)
        alpha_supp=detect_pos_1(alpha_supp)
        IES=[[pos[0]+i*step,pos[-1]+i*step] for pos in IES]
        alpha_supp=[[pos[0]+i*step,pos[-1]+i*step] for pos in alpha_supp]
        pos_IES+=IES
        pos_alpha_supp+=alpha_supp

    return pos_IES, pos_alpha_supp

def supp_power(y,Ws,step,fs,T_IES_max,T_alpha_max,T_hf_max):

    windows=[y[i:i+Ws] for i in range(0,len(y)-Ws,step)]
    pos_IES, pos_alpha_supp = [], []
    for i in range(len(windows)):
        IES,alpha_supp=detect_suppressions_power(windows[i],fs,T_IES_max,T_alpha_max,T_hf_max)[2:4]
        IES=[[pos[0]+i*step,pos[-1]+i*step] for pos in IES]
        alpha_supp=[[pos[0]+i*step,pos[-1]+i*step] for pos in alpha_supp]
        pos_IES+=IES
        pos_alpha_supp+=alpha_supp

    return pos_IES, pos_alpha_supp

def quantile(y,t,Ws,step,q):

    windows=[y[i:i+Ws] for i in range(0,len(y)-Ws,step)] 
    quantile_list=[np.quantile(win,q) for win in windows] 
    t_list=[t[Ws+i*step] for i in range(len(windows))] 

    return t_list, quantile_list  

def spindles_prop_off(y,t,Ws,step, upper_T, lower_T, method='power'):


    windows=[y[i:i+Ws] for i in range(0,len(y)-Ws,step)] 
    res_list=[spindle_off_proportion(win, upper_T, lower_T, method) for win in windows] 
    t_list=[t[Ws+i*step] for i in range(len(windows))] 

    return t_list, np.transpose(res_list)

def get_var_tot(y,t,Ws,step):

    windows=[y[i:i+Ws] for i in range(0,len(y)-Ws,step)] 
    var_tot_list=[var_tot(win) for win in windows]
    t_list=[t[Ws+i*step] for i in range(len(windows))] 

    return t_list, var_tot_list

def get_trend(y,t,Ws,step):

    windows=[y[i:i+Ws] for i in range(0,len(y)-Ws,step)]
    t_windows=[t[i:i+Ws] for i in range(0,len(y)-Ws,step)]
    #first order coefficient
    trend_list=np.array([np.polyfit(t_windows[i],windows[i],1)[0] for i in range(len(windows))])
    t_list=[t[Ws+i*step] for i in range(len(windows))] 

    return t_list, trend_list

def get_trend_median(y,t,Ws,step):

    windows=[y[i:i+Ws] for i in range(0,len(y)-Ws,step)]
    # gradient
    gradient_list=[np.gradient(win) for win in windows]
    #trend_list = np.array([np.median(gradient) for gradient in gradient_list])
    trend_list = np.array([np.quantile(gradient,0.9) for gradient in gradient_list])
    t_list=[t[Ws+i*step] for i in range(len(windows))] 

    return t_list, trend_list    

def get_trend_time_series(y,t,Ws,step):

    windows=[y[i:i+Ws] for i in range(0,len(y)-Ws,step)]
    windows=[win+np.min(win)+1 for win in windows]
    t_windows=[t[i:i+Ws] for i in range(0,len(y)-Ws,step)]
    #first order coefficient
    trend_list=[seasonal_decompose(win, model='multiplicative',period=16,extrapolate_trend='freq')._trend for win in windows]
    trend_list=np.array([np.polyfit(t_windows[i],trend_list[i],1)[0] for i in range(len(windows))])
    t_list=[t[Ws+i*step] for i in range(len(windows))] 

    return t_list, trend_list

def get_spindles_off_proportion(y, t, Ws, step, upper_T, lower_T):

    windows=[y[i:i+Ws] for i in range(0,len(y)-Ws,step)]
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    pos_spindles_list = [get_envelopes_power(win,upper_T,lower_T)[0] for win in windows]

    prop_off_list = []

    for pos_spindle in pos_spindles_list:
        length = 0
        for pos in pos_spindle:
            length += pos[-1] - pos[0]
        prop_off_list.append(1 - length / Ws)

    return t_list,prop_off_list

def get_spindles_off_proportion_amplitude(y, t, Ws, step, upper_T, lower_T):

    windows=[y[i:i+Ws] for i in range(0,len(y)-Ws,step)]
    t_windows=[t[i:i+Ws] for i in range(0,len(y)-Ws,step)]
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    pos_spindles_list = [get_envelopes(t_windows[i],windows[i],upper_T,lower_T)[2] for i in range(len(windows))]

    prop_off_list = []

    for pos_spindle in pos_spindles_list:
        length = 0
        for pos in pos_spindle:
            length += pos[-1] - pos[0]
        prop_off_list.append(1 - length / Ws)

    return t_list,prop_off_list

def get_N_spindles_power(y, t, Ws, step, upper_T, lower_T):

    windows=[y[i:i+Ws] for i in range(0,len(y)-Ws,step)]
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    pos_spindles_list = [get_envelopes_power(win,upper_T,lower_T)[0] for win in windows]

    N_spindles_list = [len(pos_spindles) for pos_spindles in pos_spindles_list]
    return t_list, N_spindles_list

def get_N_spindles_amplitude(y, t, Ws, step, upper_T, lower_T):

    windows=[y[i:i+Ws] for i in range(0,len(y)-Ws,step)]
    t_windows=[t[i:i+Ws] for i in range(0,len(y)-Ws,step)]
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    pos_spindles_list = [get_envelopes(t_windows[i],windows[i],upper_T,lower_T)[2] for i in range(len(windows))]

    N_spindles_list = [len(pos_spindles) for pos_spindles in pos_spindles_list]
    print(pos_spindles_list[0])
    return t_list, N_spindles_list

def get_med_P_spindles_power(y, t, Ws, step, upper_T, lower_T):

    windows=[y[i:i+Ws] for i in range(0,len(y)-Ws,step)]
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    pos_spindles_list = [get_envelopes_power(win,upper_T,lower_T)[0] for win in windows]

    P_spindles_list = []
    for i in range(len(pos_spindles_list)):
        L = []
        for pos in pos_spindles_list[i]:
            L.append(np.sum(windows[i][pos[0]:pos[-1]]**2))
        P_spindles_list.append(np.quantile(L,0.5))

    return t_list, P_spindles_list

def get_med_P_spindles_amplitude(y, t, Ws, step, upper_T, lower_T):

    windows=[y[i:i+Ws] for i in range(0,len(y)-Ws,step)]
    t_windows=[t[i:i+Ws] for i in range(0,len(y)-Ws,step)]
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    pos_spindles_list = [get_envelopes(t_windows[i],windows[i],upper_T,lower_T)[2] for i in range(len(windows))]

    P_spindles_list = []
    for i in range(len(pos_spindles_list)):
        L = []
        for pos in pos_spindles_list[i]:
            L.append(np.sum(windows[i][pos[0]:pos[-1]]**2))
        P_spindles_list.append(np.quantile(L,0.9))

    return t_list, P_spindles_list

def smoothing(y,t,Ws,step,N,method='savgol_filter'):
    
    windows=[y[i:i+Ws] for i in range(0,len(y)-Ws,step)]
    t_list=[t[Ws+i*step] for i in range(len(windows))]
    if method == 'savgol_filter':
        #smooth_list=[sc.signal.savgol_filter(win,N,1)[-1] for win in windows]
        smooth_list=[np.median(sc.signal.savgol_filter(win,N,1)) for win in windows]
    print(len(windows[0]))
    print(len(sc.signal.savgol_filter(windows[0],N,1)))
    return t_list, smooth_list


def kurtosis(y,t,Ws,step):

    windows=[y[i:i+Ws] for i in range(0,len(y)-Ws,step)]
    t_list=[t[Ws+i*step] for i in range(len(windows))]   

    kurtosis_list=[sc.stats.kurtosis(win) for win in windows] 

    return t_list, kurtosis_list

def get_spindles_proportion(y, t, Ws, step, upper_T, lower_T, proportion_T, frequency_T):

    windows=[y[i:i+Ws] for i in range(0,len(y)-Ws,step)]
    windows_t=[t[i:i+Ws] for i in range(0,len(y)-Ws,step)]
    t_list=[t[Ws+i*step] for i in range(len(windows))]

    D={}
    D['number of spindles']=[]
    D['number low frequency spindles']=[]
    D['number mixed frequency spindles']=[]
    D['number high frequency spindles']=[]

    for i in range(len(windows)):
        Binary_seq = get_spindle_type(windows[i],windows_t[i], upper_T, lower_T,proportion_T,frequency_T)[0]
        if len(Binary_seq) != 0:
            # pos of low mixed and high spindles
            indices_l_f = np.where(Binary_seq==0)[0]
            indices_m_f = np.where(Binary_seq==1)[0]
            indices_h_f = np.where(Binary_seq==2)[0]

            # get number of spindles
            N_spindles = len(Binary_seq)
            # get the number of low f and high f spindles
            N_l_f_spindles = len(indices_l_f)
            N_m_f_spindles = len(indices_m_f)
            N_h_f_spindles = len(indices_h_f)

            D['number of spindles'].append(N_spindles)
            D['number low frequency spindles'].append(N_l_f_spindles)
            D['number mixed frequency spindles'].append(N_m_f_spindles)
            D['number high frequency spindles'].append(N_h_f_spindles)

        else:
            D['number of spindles'].append(0.00001)
            D['number low frequency spindles'].append(0.00001)
            D['number mixed frequency spindles'].append(0.00001)
            D['number high frequency spindles'].append(0.00001)      

    D['number of spindles'] = np.array(D['number of spindles'])
    D['number low frequency spindles'] = np.array(D['number low frequency spindles'])
    D['number mixed frequency spindles'] = np.array(D['number mixed frequency spindles'])
    D['number high frequency spindles'] = np.array(D['number high frequency spindles'])
        
    return t_list, D