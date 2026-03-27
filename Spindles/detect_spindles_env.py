'''
Function to detect spindles using a "U" strategy. Compared to detect_spindle this file works with already computed envelops
'''
import numpy as np
from Functions.utils import *

#---------------------------------------- 2 thresholds method ---------------------------------------#

def get_spindles_2_thresholds_signals(signals,env_signals,lower_std_criteria,upper_std_criteria):

    N_signals=np.shape(signals)[0]
    spindles_positions=[]

    for i in range(N_signals):
            # compute standard deviation of data
            std=np.std(signals[i,:])
            spindles_positions.append(find_spindles_2_thresholds(env_signals[i,:],std,lower_std_criteria[i],upper_std_criteria[i]))

    return spindles_positions

def find_spindles_2_thresholds(diff_env,std,lower_std_criteria,upper_std_criteria):

#--- extract left min, right min from bellow segments and maximum from above segments
    pos_bellow,pos_above=get_segments(diff_env,std,lower_std_criteria)

    # maximum of above lower threshold segments
    def get_maximum(segment):
        max=np.max(diff_env[segment[0]:segment[1]+1])
        return max

    list_max=np.array(list(map(get_maximum,pos_above)))


    # left and right minimums of bellow lower threshold segments
    def get_minimum(segment):
        minimum=find_minimum(diff_env[segment[0]:segment[1]+1])
        try:
            L_min=[minimum[0],minimum[-1]]
        except:
            L_min=[segment[0],segment[1]+1]
        return L_min

    list_min=list(map(get_minimum,pos_bellow))

    '''
    list_min=[]
    for segment in pos_bellow:
        minimum=find_minimum(diff_env[segment[0]:segment[1]+1])
        try:
            list_min.append([minimum[0],minimum[-1]])
        except:
            list_min.append([segment[0],segment[1]+1])
    '''

#--- detect spindles
    pos_max_above=np.where(list_max>=upper_std_criteria*std)[0]

    # need to start and end with a bellow segment to be sure to open or close a spindle
    # if it starts with an above segment whose left pos if l we insert [l,l] as the first element of list_min
    # if it ends with a bellow segment whose right pos if r we append [r,r] as the last element of list_min
    if pos_above[0][0]<pos_bellow[0][0]:
        list_min.insert(0,[pos_above[0][0],pos_above[0][0]])
    elif pos_bellow[-1][1]<pos_above[-1][1]:
        list_min.append([pos_above[-1][1],pos_above[-1][1]])

    # create list containing starting and ending position of spindles pos_spindles=[[min_0,0,min_0,1],[min_1,0,min_1,1],...]

    def get_spindle_position(index):
        return [pos_bellow[index][1],pos_bellow[index+1][0]]
    
    return list(map(get_spindle_position,pos_max_above))

    '''
    spindle_pos=[]
    for index in pos_max_above:
        spindle_pos.append([pos_bellow[index][1],pos_bellow[index+1][0]])

    return spindle_pos
    '''
        
'''
Get the segments that are bellow lower_std_criteria.std of the signal
'''
def get_segments(diff_env,std,lower_std_criteria):
    '''
    Inputs:
    - diff_env            <-- the difference envelope of the signal
    - std                 <-- the standard deviation of the signal
    - lower_std_criteria  <-- by how much is multiplied the std to get the lower threshold
    Output:
    - [[segment0[start],segment0[end]],[segment1[start],segment1[end]],...] where a segment is the portion bellow the threshold
    '''
    N=diff_env.size

    mask=np.ones(N)
    # mask with 1 when stricly above threshold and 0 when bellow
    mask[np.where(diff_env<=lower_std_criteria*std)[0]]=0
    # detection: opening at pattern 0,1 and closing at pattern 1,0
    pos_above=detect_pos_1(mask)
    # pos bellow
    pos_bellow=detect_pos_1(1-mask)

    return pos_bellow,pos_above








#-----------------------------  Distance method -------------------------------#
'''
Function to detect spindles in signals.
For EEG generally for signals each line is a band filtered form the original raw recording
'''
def get_spindles_signals(signals,env_signals,list_threshold):
    '''
    Inputs:
    - signals          <-- the signals from the filtered bands
    - env_signals      <-- the difference envelopes of the signals
    - threshold        <-- the threshold used to compare distances between two opposite extremum
    '''
    N_signals=np.shape(signals)[0]

    spindles_positions=[]
    for i in range(N_signals):
            # compute standard deviation of data
            std=np.std(signals[i,:])
            spindles_positions.append(find_spindles(env_signals[i,:],std*list_threshold[i]))

    return spindles_positions


def find_spindles(diff_env,threshold):
    '''
    Inputs:
    - diff_env  <-- the difference envelope of the signal where we want to detect spindles
    - threshold <-- the threshold used to compare distances between two opposite extremum
    '''

    # get local min and max on the difference envelope
    list_max,list_min=find_extremum(diff_env)
    # as diff env is 0 at pos 0 the first extremum encountered is a max
    # we set the first min at the first position for the computation of 
    # consecutive distances to start by min to max pattern.
    # same thing for the end ( need to check that last is always a max with find_extremum and 0 diff env at last position)
    list_min=np.insert(list_min,0,0)
    list_min=np.append(list_min,len(diff_env)-1)   ##### to check

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

    




    






    



