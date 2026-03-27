import numpy as np
import scipy as sc

'''
For a given signal compute the spectrogram of this one and returns the part that would correspond 
to the size of the spectrogram from only the window_size signal
'''
def compute_spectro(y,fs,window_size,global_overlap,stft_window,nfft_ratio,stft_overlap):  
    '''
    Inputs:
    - y          <-- total accessible signal
    - fs              <-- frequency sampling step
    - window_size     <-- size of the interval on which the STFT is plotted
    - global_overlap  <-- proportion of window_size to add to the plot window for computing the stft
    - stft_window     <-- name of the stft window to use (str)
    - nfft ratio      <-- size of nfft window in ratio of the compute_window size
    - stft_overlap    <-- overlap percentage between two consecutive stft_window

    Outputs:
    - f               <-- list of frequencies
    - t               <-- list of times on the window_size
    - spectro         <-- spectrogram on the window_size
    '''

    size_previous_window_taken=int(np.ceil(global_overlap*window_size))     # interval size of previous data to add to reach the desired overlap
    nfft=int(nfft_ratio*window_size)                                        # size of stft_window glinding through the full window
    wind=sc.signal.get_window(stft_window,nfft)                             # window to use for the stft 
    
    if len(y)>=size_previous_window_taken+window_size:                 # enough data to have full window and expected overlap on the window before
        global_window=y[-window_size-size_previous_window_taken:]
        f,t,stft=sc.signal.stft(global_window,fs,window=wind,nperseg=nfft,noverlap=int(np.ceil(stft_overlap*nfft)))      
    
    elif size_previous_window_taken+window_size>len(y)>=window_size:   # enough data to have a full window but not the total overlap
        f,t,stft=sc.signal.stft(y,fs,window=wind,nperseg=nfft,noverlap=int(np.ceil(stft_overlap*nfft)))      
    
    else:                                                                   # not enough data, the window size is larger than the input signal length
        print('input signal does not have sufficient length')      

    index_begin_window=np.min(np.where(np.abs(t-(t[-1]-window_size/fs))<=(t[1]-t[0])))   # index of the starting interval for the window of interest to print # can do with percentage ?
    spectro=np.abs(stft[:,index_begin_window:]**2)

    return f,t[index_begin_window:],spectro


'''
For a given power percentage give the list of frequencies for which this value is reached in the spectrogram.
If multiple contributions are asked it returns the list for each in an array shape
'''
def index_contribution(contribution_percentage,PCS,delta_f):
    '''
    Inputs:
    - contribution percentage  <-- list of the power contribution percentage
    - PCS                      <-- PCS of the spectrogram of interest (normalized)
    - delta_f                  <-- frequency sampling step in the stft
    Outputs:
    - list_index               <-- array of size=(number of contributions,number of time samples)
    '''
    
    list_index=np.zeros((np.size(contribution_percentage),np.shape(PCS)[1]))

    for c in range(np.size(contribution_percentage)):
        seuil=contribution_percentage[c]
        for j in range(np.shape(PCS)[1]):
            for i in range(np.shape(PCS)[0]-1):
                # necessary because all in negative if this condition is respected
                # and the product of two consecutive values is always positive
                if i==0 and (seuil-PCS[i,j])<=0:
                    list_index[c,j]=i*delta_f   # interp here if necessary to check
                    break
                elif (seuil-PCS[i,j])==0:
                    list_index[c,j]=i*delta_f
                    break
                elif (PCS[i+1,j]-seuil)*(PCS[i,j]-seuil)<=0:
                    list_index[c,j]=np.interp(seuil,[PCS[i,j],PCS[i+1,j]],[i*delta_f,(i+1)*delta_f])
                    break
                

    return list_index 


def index_contribution_map(contribution_percentage,PCS,delta_f):
    '''
    Inputs:
    - contribution percentage  <-- list of the power contribution percentage
    - PCS                      <-- PCS of the spectrogram of interest (normalized)
    - delta_f                  <-- frequency sampling step in the stft
    Outputs:
    - list_index               <-- array of size=(number of contributions,number of time samples)
    '''

    N_line=np.shape(PCS)[0]
    PCS_t=np.transpose(PCS)
    N_contribution=len(contribution_percentage)


def index_contribution_argmin(contribution_percentage,PCS,delta_f):

    list_index=np.zeros((np.size(contribution_percentage),np.shape(PCS)[1]))
    N_contribution=len(contribution_percentage)

    for i in range(N_contribution):
        list_index[i,:]=np.argmin(np.abs(PCS-contribution_percentage[i]),axis=0)*delta_f
    
    return list_index

'''
For a given spectrogram and a frequency interval return the percentage of presence of power contained or not in this interval in the total spectrogram window.
Returns also the number of times there is an interruption 
'''
def fragmentation_score(freq_interval,spectro,delta_f,threshold_frag):   ##### maybe adpat to send index position of the different parts
    '''
    Inputs:
    - freq_interval   <-- list of intervals to take [[fi_0,fj_0],[fi_1,fj_1],...] 
    - PCS             <-- PCS of the spectrogram of interest (normalized)
    - delta_f         <-- frequency sampling step in the stft
    - threshold_frag  <-- tresholds under which a power value of the signal is considered too low for the part to be present. 
                          threshold_frag=[T0,T1,...] (same length as freq_interval)
    Outputs:
    - list_frag_score      <-- list of percentages of presence of the signal within the window for the respective frequency interval [score_0,score_1,...]
    - list_Nb_interruption <-- list of int of number of times there is an interruption for the respective frequency interval [n_0,n_1,...]
    '''

    list_frag_score=np.zeros(shape=(np.shape(freq_interval)[0],np.shape(spectro)[1]))
    list_Nb_interruption=np.zeros(np.shape(freq_interval)[0])
    Nb_col=np.shape(spectro)[1]  # number of columns

    for n,(i,j) in enumerate(freq_interval): ##### check if a verification or ordering can be done
        # convert frequency interval into index one
        i=int(i/delta_f)            # start interval index   # check start indexing from f list
        j=int(np.ceil(j/delta_f))   # end interval index

        signal_part=spectro[i:j,:]                     # take the part of the PCS contained within the frequency interval

        average_value=np.mean(signal_part,axis=0) # sum on the column and average on the number of points in a column [to check]

        mask_on_off=np.ones(np.shape(average_value))   # mask that will have one where the value is higher than the threshold and 0 if bellow

        # set t0 zero if value if lower than threshold
        mask_on_off[np.where(average_value<threshold_frag[n])[0]]=0

        # proportion of off time
        list_frag_score[n,:]=1-np.sum(mask_on_off)/Nb_col  

        # function that does 1 if we have the sequence on/off--> 10 and 0 for all others possibilities {11,01,00}. f(x_i,x_i+1)-->x_i*(1-x_i+1).
        mask_on_off=np.insert(mask_on_off,0,1)  # we need to add 1 as the first element of the mask if it starts with a 0 as otherwise it will not be counted as an interruption (only does on/off and not off/on)
        mask_on_off=mask_on_off[:-1]*(1-mask_on_off[1:])

        list_Nb_interruption[n]=np.sum(mask_on_off)

    return list_frag_score,list_Nb_interruption


'''
For a given interval of frequency returns the proportion contained within this frequency interval.
Multiple input intervals can be given. In that case it return an array of length the number of intervals with the respective proportions.
'''
def power_proportion(freq_interval,PCS,delta_f):
    '''
    Inputs:
    - freq_interval <-- list of intervals to take [[fi_0,fj_0],[fi_1,fj_1],...] 
    - PCS           <-- PCS of the spectrogram of interest (normalized)
    - delta_f       <-- frequency sampling step in the stft
    Output:
    - list_proportion <-- array where each line is the list of a power proportion for a given interval 
    '''

    list_proportion=np.zeros((np.shape(freq_interval)[0],np.shape(PCS)[1]))

    for n,(i,j) in enumerate(freq_interval): ##### check if a verification or ordering can be done
        # convert frequency interval into index one
        i=int(i/delta_f)            # start interval index   # check start indexing from f list
        j=int(np.ceil(j/delta_f))   # end interval index
        list_proportion[n,:]=PCS[j,:]-PCS[i,:]  # the proportion is the difference in the PCS value between the two indexes from the interval

    return list_proportion

def power_proportion_map(freq_interval,PCS,delta_f):
    '''
    Inputs:
    - freq_interval <-- list of intervals to take [[fi_0,fj_0],[fi_1,fj_1],...] 
    - PCS           <-- PCS of the spectrogram of interest (normalized)
    - delta_f       <-- frequency sampling step in the stft
    Output:
    - list_proportion <-- array where each line is the list of a power proportion for a given interval 
    '''  
    def get_power_proportion(f_int):
        # convert frequency interval into index one
        i=int(f_int[0]/delta_f)            # start interval index   # check start indexing from f list
        j=int(np.ceil(f_int[1]/delta_f))   # end interval index
        return PCS[j,:]-PCS[i,:]  # the proportion is the difference in the PCS value between the two indexes from the interval

    return list(map(get_power_proportion,freq_interval))



'''
Calculates the cumulative sum of the power of the spectrogram on the frequencies
'''
def power_CS(spectro):
    '''
    Inputs:
    - spectro     <-- spectrogram
    Outputs:
    - PCS         <-- Power Cumulative Sum array that has the size of spectro and each value is the sum of previous frequencies power 
    '''
    pcs=np.cumsum(spectro,axis=0)
    return pcs


#-------------------------------------------------------- Function taking all points in temporal domain ---------------------------------------------------------#


    


