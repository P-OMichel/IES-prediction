import numpy as np
import scipy as sc
from Functions.filter import filter,filter_FFT
from Functions.utils import diff_envelops,detect_pos_1
from scipy.signal import find_peaks, peak_prominences


''' 
Function that takes a signal, a length, thresholds for the level of suppressions alpha and IES and returns
the proportion of suppressions that occured during that window for both types and their masks (1 where there is a suppression 0 elsewhere)

We need to be careful that when a window stops you may be in a suppression band or a hole that will not be detected/ be detected even if the band or the hole follows in the next window.
To avoid this issue we need to take apart of the next signal that has size 2*band+gap to have a complete minimal detection to work with.
'''
def compute_suppression_mask_rate(y,mask_size,fs,threshold,list_morph,f_int,method,*arg):
    '''
    Inputs:
    - y           <-- eeg signal
    - mask_size   <-- size of the mask (it corresponds to a window on which the proportion of suppression is computed)
    - fs          <-- sampling step frequency of the signal
    - list_morph  <-- list of parameters for the morphological transformation list_morph=[min_band_IES,min_band_alpha,max_gap_IES,max_gap_alpha]
    - f_int       <-- list of freq interval to use f_int=[f_0,f_1]
    - threshold   <-- list of threshold values threshold=[T_f_int0,T_f_int1,...T_IES] to compare to the median of the envelopes difference (same order as respective freq int and then IES)
    - method      <-- str being either 'FFT' or 'DWT' selecting which method to use for the spectral decomposition
    - arg         <-- if 'DWT' is used we have to select the wavelet name and mode, and in equality_case to select True or False whether to use FFT after or not.
                      The arg is entered as a list [wavelet_name,mode,equality_case] of str  

    Outputs:
    - mask_IES                <-- mask of the IES suppressions after the erosion/dilatation/erosion routine (1 is for presence of a suppression and 0 no suppression detected)
    - mask_ alpha             <-- mask of the alpha suppressions after the erosion/dilatation/erosion routine (1 is for presence of a suppression and 0 no suppression detected)
    - suppression_rate_IES    <-- proportion of IES suppression in the window
    - suppression_rate_alpha  <-- proportion of alpha suppression in the window
    '''
    y=np.array(y)
    N=y.size

    # necessary length to add to the window on each part not to miss a complete minimal pattern of detection
    added_length=int(max((2*list_morph[0]+list_morph[2]),(2*list_morph[1]+list_morph[3]))) 

    # translation of the morphological parameters into intervals length
    added_length=int(fs*added_length)

    added_length=0

    # total size of window to take to compute for suppressions detection
    global_window_size=mask_size+2*added_length

    if N>=global_window_size:                 # we have enough data to compute on the asked window size and before/after

        # compute masks
        mask_IES,mask_alpha=suppressions_masks_amplitude(y[-global_window_size:],fs,threshold,list_morph,f_int,method,*arg)  # compute the mask detection
        mask_IES,mask_alpha=mask_IES[added_length:global_window_size-added_length],mask_alpha[added_length:global_window_size-added_length] # take only the part in the window of interest of mask_size
        
        # compute proportion rates
        suppression_rate_IES=np.sum(mask_IES)/mask_size
        suppression_rate_alpha=np.sum(mask_alpha)/mask_size

    elif global_window_size>N>=mask_size:      # number of points in between the size of the mask and the global window size. 
                                               # In that case the function returns the mask substracting added length on both size of the input signal

        # compute masks
        mask_IES,mask_alpha=suppressions_masks_amplitude(y,fs,threshold,list_morph,f_int,method,*arg)
        mask_IES,mask_alpha=mask_IES[added_length:-added_length],mask_alpha[added_length:-added_length] # pblm if mask size is smaller than 2*added_length 
        
        # compute proportion rates
        suppression_rate_IES=np.sum(mask_IES)/mask_size
        suppression_rate_alpha=np.sum(mask_alpha)/mask_size
    
    else:                                   # we do not have enough data to compute on the asked window size
        print('the size of the mask is too low compared to the asked window size')

    return mask_IES,mask_alpha,suppression_rate_IES,suppression_rate_alpha 


def suppressions_masks_prominence(y,fs,threshold,list_morph,f_int,method,*arg):

    N=y.size

    # filter to get rid of low and high frequencies
    y=filter(y,fs,[1,50],method,*arg)

    # compute spectrogram
    delta_f=0.25
    delta_t=0.25
    nfft=int(fs/delta_f)
    overlap=int(nfft-delta_t*fs)
    spectro=sc.signal.spectrogram(y,fs,nperseg=nfft,noverlap=overlap)[2]

    y2=np.sum(spectro[6:120,:],axis=0)
    signal_power=-np.log(y2)

    #---  search for peaks and prominence in the signal power series
    peaks=find_peaks(signal_power)[0]
    prominences = peak_prominences(signal_power, peaks)[0]

    # select indices of peaks which prominence is above threshold
    T = 2
    indices=np.where(prominences >= T)[0]
    print(indices)
    if len(indices)>0:
        # get threshold for IES in the low values in between burst of prominence higher than T
        T_IES=np.median(y2[indices]) 
    else:
        T_IES=0
    print(T_IES)

    # alpha band 
    alpha_y=filter(y,fs,f_int,method,*arg)

    # alpha band_envelope
    diff_alpha_env=diff_envelops(alpha_y)
    med_alpha=np.median(diff_alpha_env)

    # alpha threshold
    T_alpha=min(threshold[0],med_alpha)

    # Determination of the Omega set of indices for the suppressions
    abs_alpha_y=np.abs(alpha_y)
    Om_IES=np.where(y2<T_IES)[0]               # list of indices where the condition is satisfied for IES
    Om_alpha=np.where(abs_alpha_y<T_alpha)[0]     # list of indices where the condition is satisfied for alpha suppression
                      
    # Suppression masks
    mask_IES=np.zeros(N)
    mask_alpha=np.zeros(N)
    mask_IES[Om_IES]=1           #  set 1 where there is an IES suppression, 0 elsewhere
    mask_alpha[Om_alpha]=1       #  set 1 where there is an alpha suppression, 0 elsewhere
    
    # Erosion and Dilation 
    # Outputs is a list of masks with 0 where there is no suppression and 1 where there is one with a morpholical transformation routine erosion/dilatation/erosion
    #mask_IES=erosion_dilation(mask_IES,list_morph[0],list_morph[2],fs)
    mask_alpha=erosion_dilation(mask_alpha,list_morph[1],list_morph[3],fs)

    return mask_IES,mask_alpha

def suppressions_masks_amplitude(y,fs,threshold,list_morph,f_int,method,*arg):

    # filter to get rid of low frequencies
    y=filter(y,fs,[1,50],method,*arg)
    N=y.size

    # envelope
    diff_env=diff_envelops(y)
    med=np.median(diff_env)
    med=np.quantile(diff_env,0.9)*0.3


    # IES threshold
    T_IES=min(threshold[-1],med)
    if T_IES<5:
        T_IES=5

    # alpha band 
    alpha_y=filter(y,fs,f_int,method,*arg)
    abs_alpha_y=np.abs(alpha_y)
    '''
    # rms normalization and threshold
    indices=np.where(abs_alpha_y<100)[0] # take only values lower than 100 into account
    rms_alpha=np.sqrt(np.mean(alpha_y[indices]**2))
    #rms_alpha=np.sqrt(np.mean(alpha_y**2))
    alpha_y=alpha_y/rms_alpha
    T_alpha=2
    '''
    #'''
    # alpha band_envelope
    diff_alpha_env=diff_envelops(alpha_y)
    med_alpha=np.median(diff_alpha_env)
    med_alpha=np.quantile(diff_alpha_env,0.9)*0.15

    # alpha threshold
    T_alpha=min(threshold[0],med_alpha)
    if T_alpha<4:
        T_alpha=4
    #'''

    # Determination of the Omega set of indices for the suppressions
    abs_y=np.abs(y)
    Om_IES=np.where(abs_y<T_IES)[0] #np.where((abs_y<T_IES) & (abs_alpha_y<T_alpha))[0]       # list of indices where the condition is satisfied for IES
    Om_alpha=np.where(abs_alpha_y<T_alpha)[0] #np.where((abs_y>T_IES) & (abs_alpha_y<T_alpha))[0]     # list of indices where the condition is satisfied for alpha suppression     

    # Suppression masks
    mask_IES=np.zeros(N)
    mask_alpha=np.zeros(N)
    mask_IES[Om_IES]=1           #  set 1 where there is an IES suppression, 0 elsewhere
    mask_alpha[Om_alpha]=1       #  set 1 where there is an alpha suppression, 0 elsewhere

    # find ground check if present (need full ground check pattern)
    mask_ground_check=find_ground_check_amplitude(y)

    # 2 becomes a position in between ground check and with IES
    new_mask=mask_ground_check+mask_IES
    mask_IES[np.where(new_mask==2)[0]]=0
    
    # Erosion and Dilation 
    # Outputs is a list of masks with 0 where there is no suppression and 1 where there is one with a morpholical transformation routine erosion/dilatation/erosion
    mask_IES=erosion_dilation(mask_IES,list_morph[0],list_morph[2],fs)

    # remove alpha_supp where there is an IES
    mask_alpha[np.where((mask_alpha-mask_IES)!=1)[0]]=0
    new_mask=mask_ground_check+mask_alpha
    mask_alpha[np.where(new_mask==2)[0]]=0
    mask_alpha=erosion_dilation(mask_alpha,list_morph[1],list_morph[3],fs)

    return mask_IES,mask_alpha

#------------ Suppressions using the spectrogram -------------------------

def IES_proportion_prominence(y,fs):

    # compute spectrogram
    delta_f=0.25
    delta_t=0.25
    nfft=int(fs/delta_f)
    overlap=int(nfft-delta_t*fs)
    spectro=sc.signal.spectrogram(y,fs,nperseg=nfft,noverlap=overlap)[2]

    # get signal power values in 1.5 to 30 Hz smoothed by the spectrogram
    y2=np.sum(spectro[6:120,:],axis=0)
    signal_power=np.log(y2)
    # opposite to have IES appear as high peaks (for the prominence ?)
    signal_power=-signal_power

    # extract prominence of each peaks
    peaks=find_peaks(signal_power)[0]
    prominences = peak_prominences(signal_power, peaks)[0]

    # get indice of peaks which prominence is higher than 2
    T = 1
    indices = np.where(prominences >= T)[0]
    peaks_indices = peaks[indices]

    # get threshold for IES
    T_IES=np.median(y2[peaks_indices])*3
    
    # find IES (maybe add erosion and dilatation routine later)
    mask_IES=np.zeros_like(y2)
    mask_IES[np.where(y2 <= T_IES)[0]]=1

    # erosion dilatation routine
    mask_IES=erosion_dilation(mask_IES,1.1,0.9,4)

    return np.sum(mask_IES)/len(y2)

def detect_suppressions(y,fs,T_IES_max,T_alpha_max,T_hf_max):

    # compute spectrogram
    delta_f=0.25
    delta_t=0.25
    nfft=int(fs/delta_f)
    overlap=int(nfft-delta_t*fs)
    t_spectro,spectro=sc.signal.spectrogram(y,fs,nperseg=nfft,noverlap=overlap)[1:]

    # smoothed power of signal between [1.5,30]] Hz
    y2=np.sum(spectro[6:120,:],axis=0)
    N=len(y2)
    
    # smoothed power of signal between [7,14]] Hz
    y2_alpha=np.sum(spectro[28:56,:],axis=0) 

    # smoothed power of high frequency between [30,40] Hz
    y2_hf=np.sum(spectro[120:160,:],axis=0)
    r=y2_hf/y2

    #--- IES threshold
    # take only values that are lower than very high values from high power region and artefact, ground checks
    # what happens if power in Burst is same as high values signal ?
    try:
        indices=np.where(y2<T_IES_max*4)[0]
        T_IES=np.quantile(y2[indices],0.9)*0.25
    except: # indices is empty
        T_IES=np.quantile(y2,0.9)*0.25

    # Threshold for IES
    T_IES=min(T_IES,T_IES_max)
    # if very long suppressions T_IES will be very low, we set a minimal value for it
    T_IES_min=T_IES_max/6
    if T_IES<T_IES_min: # need to change this if microvolted (maybe check if microvolted or not at start)
        T_IES=T_IES_min

    Om_IES=np.where(y2<T_IES)[0]               # list of indices where the condition is satisfied for IES

    # Suppression masks
    mask_IES=np.zeros(N)
    mask_IES[Om_IES]=1           #  set 1 where there is an IES suppression, 0 elsewhere


    #--- alpha-suppressions threshold
    try:
        indices=np.where(y2_alpha<T_alpha_max*4)[0]
        T_alpha=np.quantile(y2_alpha[indices],0.9)*0.25
    except: # indices is empty
        T_alpha=np.quantile(y2_alpha,0.9)*0.25

    # Threshold for alpha-suppressions
    T_alpha=min(T_alpha,T_alpha_max)
    # if very long suppressions T_alpha will be very low, we set a minimal value for it
    T_alpha_min=T_alpha_max/6
    if T_alpha<T_alpha_min: # need to change this if microvolted (maybe check if microvolted or not at start)
        T_alpha=T_alpha_min

    Om_alpha=np.where(y2_alpha<T_alpha)[0]               # list of indices where the condition is satisfied for IES

    # Suppression masks
    mask_alpha=np.zeros(N)
    mask_alpha[Om_alpha]=1           #  set 1 where there is an alpha suppression, 0 elsewhere


    #--- Erosion and dilatation routine

    mask_alpha=erosion_dilation(mask_alpha,0.9,0.5,4) #1.1,0.9
    mask_IES=erosion_dilation(mask_IES,1.1,0.9,4)

    #--- correct possible ground truth
    # find ground check if present (need full ground check pattern)
    mask_ground_check=find_ground_check(y2)

    # 2 becomes a position in between ground check taken as a suppression
    new_mask=mask_ground_check+mask_IES
    mask_IES[np.where(new_mask==2)[0]]=0

    new_mask=mask_ground_check+mask_alpha
    mask_alpha[np.where(new_mask==2)[0]]=0

    #--- correct detection of IES while having too much hf (in case of ketamie, probably need correction first)
    mask_hf=np.zeros(N)
    mask_hf[np.where(r)[0]>T_hf_max]=1
    new_mask=mask_hf+mask_IES
    mask_IES[np.where(new_mask==2)[0]]=0

    # get position of suppressions
    pos_IES=detect_pos_1(mask_IES)
    pos_alpha=detect_pos_1(mask_alpha)

    # get proportion of IES in window
    IES_proportion=np.sum(mask_IES)/N
    alpha_suppression_proportion=np.sum(mask_alpha)/N

    return y2,y2_alpha,t_spectro,pos_IES,pos_alpha,IES_proportion,alpha_suppression_proportion

'''
def detect_suppressions_power(y,fs,T_IES_max,T_alpha_max,T_hf_max):


    h=np.ones(32)/32 # 32 is 0.25 s equivalent to the spectrogram one

    # smoothed power of signal between [1.5,30]] Hz
    y2=np.convolve(filter_FFT(y,fs,[1.5,30])**2,h,mode='same')
    N=len(y2)
    
    # smoothed power of signal between [7,14]] Hz
    y2_alpha=np.convolve(filter_FFT(y,fs,[7,14])**2,h,mode='same')

    # smoothed power of high frequency between [30,40] Hz
    y2_hf=np.convolve(filter_FFT(y,fs,[30,40])**2,h,mode='same')
    r=y2_hf/y2

    #--- IES threshold
    # take only values that are lower than very high values from high power region and artefact, ground checks
    # what happens if power in Burst is same as high values signal ?
    try:
        indices=np.where(y2<T_IES_max*12)[0]
        T_IES=np.quantile(y2[indices],0.9)*0.25
    except: # indices is empty
        T_IES=np.quantile(y2,0.9)*0.25

    # Threshold for IES
    T_IES=min(T_IES,T_IES_max)
    # if very long suppressions T_IES will be very low, we set a minimal value for it
    T_IES_min=T_IES_max/3
    T_IES_min=10
    if T_IES<T_IES_min: # need to change this if microvolted (maybe check if microvolted or not at start)
        T_IES=T_IES_min
    
    Om_IES=np.where(y2<T_IES)[0]               # list of indices where the condition is satisfied for IES

    # Suppression masks
    mask_IES=np.zeros(N)
    mask_IES[Om_IES]=1           #  set 1 where there is an IES suppression, 0 elsewhere


    #--- alpha-suppressions threshold
    try:
        indices=np.where(y2_alpha<T_alpha_max*15)[0]
        T_alpha=np.quantile(y2_alpha[indices],0.9)*0.25
    except: # indices is empty
        T_alpha=np.quantile(y2_alpha,0.9)*0.25

    # Threshold for alpha-suppressions
    T_alpha=min(T_alpha,T_alpha_max)
    # if very long suppressions T_alpha will be very low, we set a minimal value for it
    T_alpha_min=T_alpha_max/3
    T_alpha_min=3
    if T_alpha<T_alpha_min: # need to change this if microvolted (maybe check if microvolted or not at start)
        T_alpha=T_alpha_min

    Om_alpha=np.where(y2_alpha<T_alpha)[0]               # list of indices where the condition is satisfied for IES

    # Suppression masks
    mask_alpha=np.zeros(N)
    mask_alpha[Om_alpha]=1           #  set 1 where there is an alpha suppression, 0 elsewhere


    #--- Erosion and dilatation routine

    #mask_alpha=erosion_dilation(mask_alpha,0.9,0.5,fs) #1.1,0.9
    mask_IES=erosion_dilation(mask_IES,1.1,0.9,fs)

    # remove alpha_supp where there is an IES
    mask_alpha[np.where((mask_alpha-mask_IES)!=1)[0]]=0
    mask_alpha=erosion_dilation(mask_alpha,0.9,0.5,fs)


    #--- correct possible ground truth
    # find ground check if present (need full ground check pattern)
    mask_ground_check=find_ground_check(y2)

    # 2 becomes a position in between ground check taken as a suppression
    new_mask=mask_ground_check+mask_IES
    mask_IES[np.where(new_mask==2)[0]]=0

    new_mask=mask_ground_check+mask_alpha
    mask_alpha[np.where(new_mask==2)[0]]=0

    #--- correct detection of IES while having too much hf (in case of ketamie, probably need correction first)
    mask_hf=np.zeros(N)
    mask_hf[np.where(r)[0]>T_hf_max]=1
    new_mask=mask_hf+mask_IES
    mask_IES[np.where(new_mask==2)[0]]=0

    # get position of suppressions
    pos_IES=detect_pos_1(mask_IES)
    pos_alpha=detect_pos_1(mask_alpha)

    # get proportion of IES in window
    IES_proportion=np.sum(mask_IES)/N
    alpha_suppression_proportion=np.sum(mask_alpha)/N

    return y2,y2_alpha,pos_IES,pos_alpha,IES_proportion,alpha_suppression_proportion
'''

def detect_suppressions_power(y,fs,T_IES_max,T_alpha_max,T_hf_max):


    h=np.ones(32)/32 # 32 is 0.25 s equivalent to the spectrogram one

    # smoothed power of signal between [1.5,30]] Hz
    y2=np.convolve(filter_FFT(y,fs,[1.5,30])**2,h,mode='same')
    #y2=sc.signal.savgol_filter(y2,32,1)
    N=len(y2)
    
    # smoothed power of signal between [7,14]] Hz
    y2_alpha=np.convolve(filter_FFT(y,fs,[7,14])**2,h,mode='same')

    #--- IES threshold
    # find if zone is ok or mostly suppression
    q=np.quantile(y2,0.75)
    
    if q <= 5:
        T_IES=min(10,q*3)

    else:
        # take only values that are lower than very high values from high power region and artefact, ground checks
        # what happens if power in Burst is same as high values signal ?
        try:
            indices=np.where(y2<T_IES_max*12)[0]
            T_IES=np.quantile(y2[indices],0.9)*0.12
        except: # indices is empty
            T_IES=np.quantile(y2,0.9)*0.12

        # Threshold for IES
        T_IES=min(T_IES,T_IES_max)


    Om_IES=np.where(y2<T_IES)[0]               # list of indices where the condition is satisfied for IES

    # Suppression masks
    mask_IES=np.zeros(N)
    mask_IES[Om_IES]=1           #  set 1 where there is an IES suppression, 0 elsewhere


    #--- alpha-suppressions threshold
    try:
        indices=np.where(y2_alpha<T_alpha_max*15)[0]
        T_alpha=np.quantile(y2_alpha[indices],0.9)*0.12
    except: # indices is empty
        T_alpha=np.quantile(y2_alpha,0.9)*0.12


    Om_alpha=np.where(y2_alpha<T_alpha)[0]               # list of indices where the condition is satisfied for IES

    # Suppression masks
    mask_alpha=np.zeros(N)
    mask_alpha[Om_alpha]=1           #  set 1 where there is an alpha suppression, 0 elsewhere


    #--- Erosion and dilatation routine

    #mask_alpha=erosion_dilation(mask_alpha,0.9,0.5,fs) #1.1,0.9
    mask_IES=erosion_dilation(mask_IES,1.1,0.9,fs)

    # remove alpha_supp where there is an IES
    mask_alpha[np.where((mask_alpha-mask_IES)!=1)[0]]=0
    mask_alpha=erosion_dilation(mask_alpha,0.9,0.5,fs)

    # get position of suppressions
    pos_IES=detect_pos_1(mask_IES)
    pos_alpha=detect_pos_1(mask_alpha)

    # get proportion of IES in window
    IES_proportion=np.sum(mask_IES)/N
    alpha_suppression_proportion=np.sum(mask_alpha)/N

    return y2,y2_alpha,pos_IES,pos_alpha,IES_proportion,alpha_suppression_proportion


#--------------------------- routine ---------------------

def erosion_dilation(mask,min_band,max_gap,fs):
    '''
    Inputs:
    - mask      <-- mask of 1 at a position of a value of interest and 0 elsewhere
    - min_band  <-- min lenght of correct values (minimal lenght of segment of 1)
    - max_gap   <-- max lenght of incorrect values (maximal lenght of segment of 0)
    
    Outputs:
    - new_mask  <-- mask created after a process of erosion/dilatation/erosion (explain each phase)
    '''
    # translate the min_band and max_gap from time duration to interval length
    min_band=int(fs*min_band) 
    max_gap=int(fs*max_gap)-1   # minus 1 to have the correct effect on erosion (erosion with int 3 for instance takes away 2 by construction of the function)

    # routine to erode, dilate and erode the mask
    new_mask=sc.ndimage.binary_erosion(mask,np.ones(min_band),iterations=1)
    new_mask=sc.ndimage.binary_dilation(new_mask,np.ones(min_band+max_gap),iterations=1)
    new_mask=sc.ndimage.binary_erosion(new_mask,np.ones(max_gap),iterations=1)

    return new_mask

def find_ground_check_amplitude(y):

    mask=np.zeros(y.size)
    mask[np.where(abs(y)>=100)[0]]=1
    mask=sc.ndimage.binary_dilation(mask,np.ones(1280),iterations=1)*1 # around 10 s

    return mask

def find_ground_check(y2):

    mask=np.zeros(len(y2))
    mask[np.where(y2>=1200)[0]]=1
    mask=sc.ndimage.binary_dilation(mask,np.ones(40),iterations=1)*1 # around 10 s

    return mask
