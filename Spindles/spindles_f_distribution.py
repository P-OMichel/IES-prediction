'''
Function to get the distribution of spindles frequency
'''

import scipy as sc
import numpy as np
from Spindles.spindles_segmentation import detect_spindles,detect_spindles_power
from Functions.utils import barycenter
from Spindles.get_spindle_type import *


def get_low_f_proportion_power(y,fs,lower_criteria,upper_criteria,T,N):

    # get the spindles positions
    pos_spindles=detect_spindles_power(y,lower_criteria,upper_criteria,N)

    # extract the main spindles frequencies
    f_list=[]
    for pos in pos_spindles:
        f_spindle=get_f_max_power(y[pos[0]:pos[-1]],fs,T)
        for f in f_spindle:
            f_list.append(f)

    proportion_low_f_spindles=len(np.where(np.array(f_list)<=10.5)[0])/len(f_list)

    return proportion_low_f_spindles


def get_low_f_proportion(y,fs,lower_criteria,upper_criteria,T):

    # get the spindles positions
    pos_spindles=detect_spindles(y,lower_criteria,upper_criteria)

    # extract the main spindles frequencies
    f_list=[]
    for pos in pos_spindles:
        f_spindle=get_f_max_power(y[pos[0]:pos[-1]],fs,T)
        for f in f_spindle:
            f_list.append(f)

    proportion_low_f_spindles=len(np.where(np.array(f_list)<=10.5)[0])/len(f_list)

    return proportion_low_f_spindles


def get_h_l_f_quantile(y,fs,lower_criteria,upper_criteria,T,l_quantile,h_quantile):

    # get the spindles positions
    pos_spindles=detect_spindles(y,lower_criteria,upper_criteria)

    # extract the main spindles frequencies
    f_list=[]
    for pos in pos_spindles:
        f_spindle=get_f_max_power(y[pos[0]:pos[-1]],fs,T)
        for f in f_spindle:
            f_list.append(f)

    # get low and high quantile frequencies
    f_list=np.array(f_list)
    l_f_quantile=np.quantile(f_list,q=l_quantile)
    h_f_quantile=np.quantile(f_list,q=h_quantile)

    return l_f_quantile,h_f_quantile


def get_spindles_barycenter(y,fs,lower_criteria,upper_criteria,T,N_bins,T_dist):
    
    # get distribution
    histo,bins_mid_values=get_f_distribution(y,fs,lower_criteria,upper_criteria,T,N_bins)

    # get barycenter
    indices=np.where(histo>=T_dist)[0]
    x_b=barycenter(bins_mid_values[indices],histo[indices])

    return histo,bins_mid_values,x_b


def get_f_distribution(y,fs,lower_criteria,upper_criteria,T,N_bins,method='0'):

    # get the spindles positions
    if method=='0':
        pos_spindles=get_envelopes_power(y, lower_criteria, upper_criteria)
    elif method=='1':
        pos_spindles=detect_spindles(y,lower_criteria,upper_criteria)

    # extract the main spindles frequencies
    f_list=[]
    for pos in pos_spindles:
        f_spindle=get_f_max_power(y[pos[0]:pos[-1]],fs,T)
        for f in f_spindle:
            f_list.append(f)

    # maybe add line to avoid having values oustide the range of the alpha wave frequencies
    histo,bins_list=np.histogram(f_list,bins=N_bins)
    # take mid values in between bins (list of same llength as histogram)
    bins_mid_values=np.array([(bins_list[i+1]-bins_list[i])/2+bins_list[i] for i in range(len(bins_list)-1)])

    return histo/np.max(histo),bins_mid_values


def get_f_max_power(spindle,fs,T):

    fft_y=sc.fft.rfft(spindle)
    fft_y=np.abs(fft_y)
    delta_f=(fs/2)/len(fft_y) 
    # take peaks higher than T* the max, can be more than one, taking [-2:] works to get at max 2, if only one works as well
    indices=np.sort(np.where(fft_y>T*np.max(fft_y))[0])[-2:]   

    return indices*delta_f

    