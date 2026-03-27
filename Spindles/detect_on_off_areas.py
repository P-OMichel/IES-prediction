'''
File to detect On/Off zone regarding the presence (On) or not (Off) of spindles 
'''
from Functions.utils import find_maximum,detect_pos_1
import numpy as np
import scipy as sc


def detect_off_areas(env_signals,h,quantile,threshold,mode='same'):
    N_line,N_column=np.shape(env_signals)
    mask_signals=np.zeros(shape=(N_line,N_column))
    pos_off_signals=[]

    # detect On/Off positions in the different signal line
    for i in range(N_line):
        mask_signals[i,:]=mask_on_off(env_signals[i,:],h[i],quantile[i],threshold[i],mode='same')
        pos_off_signals.append(detect_pos_1(mask_signals[i,:]))

    # detect mutual Off area (where all signal lines are in an Off state)
    # A mutual off area is an area with ones in the array obtained by mutliplying all the masks together
    mask_mutual_off=np.prod(mask_signals,axis=0)
    pos_off_mutual=detect_pos_1(mask_mutual_off)

    return pos_off_signals,pos_off_mutual

'''
Same function as detect_on_off_mean but return the mask of 0 when above threshold and 1 when bellow instead of position of 0,1 and 1,0 edges
'''
def mask_on_off(diff_env,h,quantile,threshold,mode='same'):
    mean=np.convolve(diff_env,h,mode)
    N=diff_env.size
    mask=np.zeros(N)
    mask[np.where(mean<=threshold*np.quantile(mean,quantile))[0]]=1
    return mask

'''
same as detect_off_areas but without the computation to get mutual off areas, it only gives suppressions position int he different line of the signals input
'''
def detect_on_off_mean_signals(env_signals,h,quantile,threshold,mode='same'):
    '''
    env_signals: the difference envelope of the signals
    h: the filters for averaging (here it is a list of np.ones() of given sizes depending on the signal line)
    threshold: threshold value in [0,1] under which it is considered an off area (normalized) 
    '''
    N_line,N_column=np.shape(env_signals)
    mask_signals=np.zeros(shape=(N_line,N_column))
    pos_off_signals=[]
    for i in range(N_line):
        mask_signals[i,:]=mask_on_off(env_signals[i,:],h[i],quantile[i],threshold[i],mode='same')
        pos_off_signals.append(detect_pos_1(mask_signals[i,:]))

    return pos_off_signals

'''
returns the positions of the off areas on a D signal
'''
def detect_on_off_mean(diff_env,h,quantile,threshold,mode='same'):
    '''
    diff_env: the difference envelope of the signal
    h: the filter for averaging (here it is np.ones(Number_neighbours))
    threshold: threshold value in [0,1] under which it is considered an off area (normalized) 
    '''
    mean=np.convolve(diff_env,h,mode)
    N=diff_env.size
    mask=np.zeros(N)
    mask[np.where(mean<=threshold*np.quantile(mean,quantile))[0]]=1

    return detect_pos_1(mask)



#------------------------ detection of On/Off areas different than with sliding mean ---------------------#


def detect_on_off_percentile(diff_env,percentile,N_neighbours,threshold):
    '''
    diff_env: the difference envelope of the signal
    percentile: number between 0 and 100 (50 gives median)
    N_neighbours: sliding window size (must be an odd number)
    '''
    percentile=sc.ndimage.percentile_filter(diff_env,percentile, size=N_neighbours)
    return np.where(percentile<=threshold*np.max(percentile))[0]

def detect_on_off_max(diff_env,h,threshold):
    '''
    diff_env: the difference envelope of the signal
    h: the filter for averaging (here it is np.ones(Number_neighbours))
    threshold: threshold value in [0,1] under which it is considered an off area (normalized) 
    '''
    pos_max=find_maximum(diff_env)
    max=diff_env[pos_max]
    mean=np.convolve(max,h,'same')
    return pos_max[np.where(mean<=threshold*np.max(mean))[0]]