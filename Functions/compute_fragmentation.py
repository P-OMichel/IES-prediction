'''
File to compute the fragmentation level in different bands and IES
'''
import numpy as np
import scipy as sc
from Functions.utils import detect_pos_1

def compute_fragmentation(signal,h,quantile,threshold,mode='same'):

    # smooth power of signal
    signal2=np.convolve(signal**2,h,mode='same')

    # mask of where data is lower than a threshold 
    N_signal2=len(signal2)
    mask=np.zeros(N_signal2)

    signal2=np.convolve(signal2,h,mode) # maybe to delete

    mask[np.where(signal2<=threshold*np.quantile(signal2,quantile))[0]]=1    

    # erosion dilatation routine
    N_h=len(h)
    mask=morph_erosion_dilation(mask,int(N_h*0.8),int(N_h*0.3))

    # fragmentation score
    fragmentation_score=np.sum(mask)/N_signal2

    return fragmentation_score

def compute_fragmentation_norm(env_signals,h,quantile,threshold,mode='same'):
    '''
    env_signals: numpy array where each line if the difference envelop of one band
    h: the filter to convolute with to get the mean
    threshold: the list of thresholds for detection
    quantile: numpy array of the quantile ti take from each line
    '''
    N_line,N_column=np.shape(env_signals)
    mask_signals=np.zeros(shape=(N_line,N_column))
    N_on_off=np.zeros(N_line)

    # detect On/Off positions in the different signal line
    for i in range(N_line):
        N_h=len(h[i])
        mask_signals[i,:]=mask_on_off(env_signals[i,:],h[i],quantile[i],threshold[i],mode)
        mask_signals[i,:]=morph_erosion_dilation(mask_signals[i,:],int(N_h*0.8),int(N_h*0.3))
        N_on_off[i]=len(detect_pos_1(mask_signals[i,:]))
    fragmentation_score=np.sum(mask_signals,axis=1)/N_column

    return fragmentation_score#,N_on_off

def compute_fragmentation_level(env_signals,h,quantile,threshold,mode='same'):
    '''
    env_signals: numpy array where each line if the difference envelop of one band
    h: the filter to convolute with to get the mean
    threshold: the list of thresholds for detection
    quantile: numpy array of the quantile ti take from each line
    '''
    N_line,N_column=np.shape(env_signals)
    mask_signals=np.zeros(shape=(N_line,N_column))

    # detect On/Off positions in the different signal line
    for i in range(N_line):
        mask_signals[i,:]=mask_on_off(env_signals[i,:],h[i],quantile[i],threshold[i],mode)
    fragmentation_score=np.sum(mask_signals,axis=1)/N_column

    mask_mutual_off=np.prod(mask_signals[1:,:],axis=0)
    IES_score=np.sum(mask_mutual_off)/N_column

    return np.append(fragmentation_score,IES_score)
    

def mask_on_off(diff_env,h,quantile,threshold,mode):
    mean=np.convolve(diff_env,h,mode)
    N=diff_env.size
    mask=np.zeros(N)
    mask[np.where(mean<=threshold*np.quantile(mean,quantile))[0]]=1
    return mask
    

def morph_erosion_dilation(mask,min_band,max_gap):
    
    max_gap=max_gap-1

    new_mask=sc.ndimage.binary_erosion(mask,np.ones(min_band),iterations=1)
    new_mask=sc.ndimage.binary_dilation(new_mask,np.ones(min_band+max_gap),iterations=1)
    new_mask=sc.ndimage.binary_erosion(new_mask,np.ones(max_gap),iterations=1)

    return new_mask   

