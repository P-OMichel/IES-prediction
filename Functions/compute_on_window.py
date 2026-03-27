'''
This function computes all the necessary data to extract from a raw EGG signal in between an interval
--> take a window size to process
--> detect and correct artifacts on the sample
--> computes the spectrogram, the contribution power percentage  and the power proportion in between frequency intervals
--> computes the level of fragmentations of some frequency bands signal
--> detection IES and alpha suppressions
'''

from Functions.detect_artifacts import *
from Functions.WaveletQuantileNormalization import *
from Functions.spectral_power_methods import *
from Functions.suppressions import *


def compute_window_light(y,fs,window_size,
                   list_detection,
                   list_WQN,list_Spectro,
                   list_power_contribution
                   ):

    if window_size>len(y): 
        print('Error: window larger than signal size')

    # need y before -window_size to be a corrected signal   
    else:
        eeg_signal=y[-window_size:]                                         # take the sample of interest in the eeg signal on the window to process
        
    #--- detection of the artifacts
        index_mask=find_artifacts(eeg_signal,*list_detection)      

        # if it has some artifacts  
        if len(index_mask)!=0:                                              
        #--- clean the artifacts using WQN algorithm
            eeg_clean=WQN_3(eeg_signal,index_mask,*list_WQN)                
            y[-window_size:]=eeg_clean  

    #--- obtain the spectrogram on the window to process 
        f,t,spectro=compute_spectro(y,fs,window_size,*list_Spectro)         

    #--- Power Cumulative Sum (PCS)
        pcs=power_CS(spectro)   
        # normalize the PCS                                            
        pcs=pcs/pcs[-1,:]   

        # step of frequency in the spectrogram useful for the index_contribution and power_proportion functions
        delta_f=f[-1]-f[-2]    

    #--- get list of index for the contribution of power
        list_index=index_contribution_argmin(list_power_contribution,pcs,delta_f)  


    return y[-window_size:],f,t,spectro,list_index

'''
def compute_window(y,fs,window_size,
                   list_detection,
                   list_WQN,list_Spectro,
                   list_power_contribution,
                   list_freq_interval,
                   list_fragmentations,
                   list_suppressions):
    #
    Inputs:
    - y              <-- raw eeg signal

    - fs             <-- frequency sampling step

    - window_size    <-- size of window to plot

    - list_detection <-- [Ws,step,threshold_detection,wavelet_name,level,mode] 
                         Ws: sliding window size for DWT
                         step: interval step between two windows
                         threshold: list of thresholds on the approximation coefficient ECDF and the first detail coefficient one [approx_threshold,detail_threshold]
                         wavelet_name: wavelet to use for the DWT
                         level: max level to use for the DWT
                         mode: mode to use for DWT

    - list_WQN       <-- [wavelet_name,mode,min_interval,alpha]
                         wavelet_name: name of the wavelet to use (str)
                         mode: mode to use for the DWT and IDWT (str)
                         min_interval: minimium interval for the last wavelet coefficient iun the decomposition
                         alpha: power on the min threshold            

    - list_Spectro   <-- [global_overlap,stft_window,nfft_ratio,stft_overlap]
                         global_overlap: overlap percentage taken from previous window for STFT
                         stft_window: name of the window shape to use for STFT (str)
                         nfft_ratio: size of nfft window in ratio of the compute_window size
                         stft_overlap: overlap percentage between two consecutive stft_window

    - list_power_contribution <-- list of all power contribution percentage([a,b,c,...]) to take into account 

    - list_freq_interval      <-- list of intervals of frequency to look the power proportion contained in it [[fi_0,f_j0],[fi_1,fj_1],...]

    - list_fragmentations     <-- [list_frequency_frag,frag_prop]
                                  list_frequency_frag: list of intervals of frequency to look the amount of fragmentation contained in it [[fi_0,f_j0],[fi_1,fj_1],...]
                                  frag_prop: scalar to divide the maximum value of the spectrogram to have the threshold for fragmentation
    
    - list_suppressions       <-- [threshold_suppressions,list_morph,f_int,method]

    Outputs:
    - y                           <-- corrected eeg_signal
    - f                           <-- list of frequencies for spectrogram
    - t                           <-- list of time for spectrogram starting at the correct window
    - spectro                     <-- spectrogram of the signal in the window to process
    - list_index                  <-- array where each line is the list of a given power contribution percentage   
    - list_proportion             <-- array where each line is the list of a power proportion for a given interval
    - list_frag_score             <-- list of percentages of presence of the signal within the window for the respective frequency interval [score_0,score_1,...]
    - list_Nb_interruption        <-- list of int of number of times there is an interruption for the respective frequency interval [n_0,n_1,...]
    - mask_IES                    <-- mask for the IES suppression 
    - mask_alpha                  <-- mask for the alpha suppressions
    - suppression_rate_IES        <-- proportion of IES suppression in the window
    - suppression_rate_alpha      <-- proportion of alpha suppression in the window
    #

    if window_size>len(y): 
        print('Error: window larger than signal size')

    # need y before -window_size to be a corrected signal   
    else:
        eeg_signal=y[-window_size:]                                         # take the sample of interest in the eeg signal on the window to process
        
        #################################################### clean data ############################################################################

        index_mask=find_artifacts(eeg_signal,*list_detection)               # detection of the artifacts
        
        if len(index_mask)!=0:                                              # if it has some artifacts
            eeg_clean=WQN_3(eeg_signal,index_mask,*list_WQN)                # clean the artifacts using WQN algorithm
            y[-window_size:]=eeg_clean  ##### make sure the size are corrects for eeg_clean

        #################################################### operations on the spectral power ############################################################################

        f,t,spectro=compute_spectro(y,fs,window_size,*list_Spectro)         # obtain the spectrogram on the window to process 

        pcs=power_CS(spectro)                                               # Power Cumulative Sum (PCS)
        pcs=pcs/pcs[-1,:]                                                   # normalize the PCS

        delta_f=f[-1]-f[-2]                                                 # step of frequency in the spectrogram useful for the index_contribution and power_proportion functions

        list_index=index_contribution(list_power_contribution,pcs,delta_f)  # get list of index for the contribution of power

        list_proportion=power_proportion(list_freq_interval,pcs,delta_f)    # Power proportion in between frequency intervals

        threshold_frag=np.max(spectro)/list_fragmentations[1]                                     ##### to change 
        list_frag_score,list_Nb_interruption=fragmentation_score(list_fragmentations[0],spectro,delta_f,threshold_frag) # fragmentation score and number of interruptions
        #list_frag_score_2,list_Nb_interruption_2=fragmentation_score_2(list_fragmentations[0],spectro,delta_f,threshold_frag) # fragmentation score and number of interruptions ##### on small array first version seems to work slighty faster. Check on larger array on also with multiple 0 due to threshold as this take a longer time in the first version.Also if  positions are needed to check wich one is faster

        #################################################### operations on suppressions ############################################################################
        
        added_length=max((2*list_suppressions[1][0]+list_suppressions[1][2]),(2*list_suppressions[1][1]+list_suppressions[1][3]))   # take the maximum of minimal suppression pattern criteria
        added_length=int(fs*added_length)  # conversion into interval length
        # the suppressions are calculated on a window of size window_size+2*added_length and return a signal of length window_size but that ends and starts -added_length prior to the window to process.
        mask_IES,mask_alpha,suppression_rate_IES,suppression_rate_alpha=compute_suppression_mask_rate(y[-window_size-2*added_length:],window_size,fs,*list_suppressions)

    return y,f,t,spectro,list_index,list_proportion,list_frag_score,list_Nb_interruption,mask_IES,mask_alpha,suppression_rate_IES,suppression_rate_alpha
'''








