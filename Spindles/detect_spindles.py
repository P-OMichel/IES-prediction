'''
Function to detect spindles using a "U" strategy
'''
import numpy as np
from Functions.filter import filter
from Functions.utils import *
import statistics as stat

def get_spindle_positions(y,list_f_int,list_threshold):
    '''
    Inputs:
    - y          <-- the signal where we want to find some spindles
    - threshold  <-- the threshold used to compare distances between two opposite extremum
    - list_f_int <-- list of frequency intervals
    '''

    spindles_positions=[]
    y_bands=[]
    for i in range(len(list_f_int)):
            # filter in desired band
            y_filtered=filter(y,128,list_f_int[i],'FFT')

            # compute standard deviation of data
            #std=np.std(y_filtered)
            std=stat.stdev(y_filtered)
            y_bands.append(y_filtered)
            spindles_positions.append(find_spindles(y_filtered,std*list_threshold[i]))

    return spindles_positions,np.array(y_bands)


'''
Function to detect spindles in signals.
For EEG generally for signals each line is a band filtered form the original raw recording
'''
def get_spindle_positions_signals(signals,list_threshold):
    '''
    Inputs:
    - signals          <-- the signals from the filtered bands
    - threshold        <-- the threshold used to compare distances between two opposite extremum
    '''
    N_signals=np.shape(signals)[0]

    spindles_positions=[]
    for i in range(N_signals):
            signal=signals[i,:]
            # compute standard deviation of data
            std=np.std(signal)
            spindles_positions.append(find_spindles(signal,std*list_threshold[i]))

    return spindles_positions


def find_spindles(y,threshold):
    '''
    Inputs:
    - y         <-- the signal where we want to find some spindles
    - threshold <-- the threshold used to compare distances between two opposite extremum
    '''

    # get envelopes difference
    diff_env=diff_envelops(y)

    # get local min and max on the difference envelope
    list_max,list_min=find_extremum(diff_env)
    # as diff env is 0 at pos 0 the first extremum encountered is a max
    # we set the first min at the first position for the computation of 
    # consecutive distances to start by min to max pattern.
    # same thing for the end ( need to check that last is always a max with find_extremum and 0 diff env at last position)
    list_min=np.insert(list_min,0,0)
    list_min=np.append(list_min,len(y)-1)

    # get the distance between two opposite extremum
    list_distances=get_consecutive_distances(diff_env,list_min,list_max)

    # list of the distances above the threshold
    list_above_threshold=np.where(list_distances>=threshold)[0]

    # make sure it starts with a min to max pattern, otherwise make min start at pos 0
    # if the first index is not even then it is a max to min start, we switch it to a min to max one 
    # with the min at the first position
    if len(list_above_threshold)==0:
        none_array=np.array([])
        none_array=none_array.astype(int)  # setting to int is necessary for plot (indexing issue otherwise --> the indices are set to float)
        return none_array
    
    elif len(list_above_threshold)!=0:
        if list_above_threshold[0]%2!=0:
            list_above_threshold=np.insert(list_above_threshold,0,0)


        # eliminate consecutive similar direction distances
        for i in range(len(list_above_threshold)-1):
            # if consecutive sum is even they are in the same direction
            if ((list_above_threshold[i]+list_above_threshold[i+1])%2==0):
                # if the first is even it is a min to max pattern we take the left one
                if list_above_threshold[i]%2==0:
                    list_above_threshold[i+1]=-2 # need to be pair for next comparison
                # if the first is odd it is a max to min pattern we take the right one
                elif list_above_threshold[i]%2==1:
                    list_above_threshold[i]=-1
        
        # remove unwanted values (setting them bellow 0 above allows to easily remove then)
        list_above_threshold=list_above_threshold[np.where(list_above_threshold>=0)[0]]
        list_above_threshold=list_above_threshold+list_above_threshold%2

        # transfer position of the distances to position of spindle
        pos=list_above_threshold/2
        pos=pos.astype(int)
        list_spindles=list_min[pos]

    return list_spindles#,list_distances,list_min,list_max


def get_consecutive_distances(y,list_min,list_max):
    '''
    Output:
    - L_distances   <-- the list of consecutive distances between min to max and max to min pattern (in this order)
    '''

    L_distances=[]
    # distance for min to max patterns
    left_distance=y[list_max]-y[list_min[:-1]]
    # distance for max to min patterns
    right_distance=y[list_max]-y[list_min[1:]]

    for i in range(len(left_distance)):
        # starts always with a min to max pattern ( thanks to the if condition of the find_spindle function)
        # min to max distances are on even indices
        L_distances.append(left_distance[i])
        # max to min distances are on odd indices
        L_distances.append(right_distance[i])        

    L_distances=np.array(L_distances)


    return L_distances
