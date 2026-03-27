'''
Detection of spindles using two thresholds extracted from the signal envelope
'''

import numpy as np
import scipy as sc
from Functions.utils import *
from bisect import bisect

def detect_spindles(y,lower_criteria,upper_criteria):
    
    # get signal envelope
    diff_env = diff_envelops(y)

    # get Thresholds
    # currently std but possibility to change how to get the threshold
    std_signal=np.std(diff_env)
    #std_signal=np.quantile(np.abs(y),0.75)
    l_T, u_T = lower_criteria*std_signal, upper_criteria*std_signal

    # determine positions of maximum above upper threshold
    pos_maximum = np.where(diff_env > u_T)[0]
    #pos_maximum = sc.signal.find_peaks(diff_env,height=u_T)[0]

    # determine positions of minimum bellow lower threshold
    pos_minimum = np.where(diff_env < l_T)[0]
    #pos_minimum = sc.signal.find_peaks(-diff_env,height=l_T)[0]

    if len(pos_maximum)==0 or len(pos_minimum)==0:
        return [[0,len(y)]]
    
    else:

        # Ensure it start with a minimum (if it start with a maximum it is necessarily cutted, we get rid of this spindle)
        # rmk: we could also take it
        if pos_maximum[0] < pos_minimum[0]:
            pos_maximum=pos_maximum[1:]

        # Ensure it stops with a minimum (if it stops with a maximum it is necessarily cutted, we get rid of this spindle)
        # rmk: we could also take it
        if pos_maximum[-1] > pos_minimum[-1]:
            pos_maximum=pos_maximum[:-1]   

        # now it necessarily starts with a left minimum of a spindle and stops with a right minimum of a spindle 
        pos_spindles=[]
        while len(pos_maximum) != 0:
            # find closest left min value and rigth min value to a max (for that we search the pos the max would have in the min list to keep it sorted)
            index=bisect(pos_minimum,pos_maximum[0])
            pos_spindles.append([pos_minimum[index-1],pos_maximum[0],pos_minimum[index]])
            # update the maximum list (stop algortihm when it becomes empty)
            pos_maximum=pos_maximum[bisect(pos_maximum,pos_minimum[index]):]
            # update the minimum list
            pos_minimum=pos_minimum[index:]

    return pos_spindles


def detect_spindles_power(y,lower_criteria,upper_criteria,N):
    
    # get signal envelope
    diff_env = np.convolve(y**2,np.ones(N)/N,mode='same')

    # get Thresholds
    # currently std but possibility to change how to get the threshold
    std_signal=np.std(diff_env)
    l_T, u_T = lower_criteria*std_signal, upper_criteria*std_signal

    # determine positions of maximum above upper threshold
    pos_maximum = np.where(diff_env > u_T)[0]
    #pos_maximum = sc.signal.find_peaks(diff_env,height=u_T)[0]

    # determine positions of minimum bellow lower threshold
    pos_minimum = np.where(diff_env < l_T)[0]
    #pos_minimum = sc.signal.find_peaks(-diff_env,height=l_T)[0]

    if len(pos_maximum)==0 or len(pos_minimum)==0:
        return [[0,len(y)]]
    
    else:

        # Ensure it start with a minimum (if it start with a maximum it is necessarily cutted, we get rid of this spindle)
        # rmk: we could also take it
        if pos_maximum[0] < pos_minimum[0]:
            pos_maximum=pos_maximum[1:]

        # Ensure it stops with a minimum (if it stops with a maximum it is necessarily cutted, we get rid of this spindle)
        # rmk: we could also take it
        if pos_maximum[-1] > pos_minimum[-1]:
            pos_maximum=pos_maximum[:-1]   

        # now it necessarily starts with a left minimum of a spindle and stops with a right minimum of a spindle 
        pos_spindles=[]
        while len(pos_maximum) != 0:
            # find closest left min value and rigth min value to a max (for that we search the pos the max would have in the min list to keep it sorted)
            index=bisect(pos_minimum,pos_maximum[0])
            if index==len(pos_minimum):
                pos_spindles.append([pos_minimum[index-1],pos_maximum[0],pos_minimum[-1]])
                # update the maximum list (stop algortihm when it becomes empty)
                pos_maximum=pos_maximum[bisect(pos_maximum,pos_minimum[-1]):]
                # update the minimum list
                pos_minimum=pos_minimum[index:]
            else:
                pos_spindles.append([pos_minimum[index-1],pos_maximum[0],pos_minimum[index]])
                # update the maximum list (stop algortihm when it becomes empty)
                pos_maximum=pos_maximum[bisect(pos_maximum,pos_minimum[index]):]
                # update the minimum list
                pos_minimum=pos_minimum[index:]

    return pos_spindles