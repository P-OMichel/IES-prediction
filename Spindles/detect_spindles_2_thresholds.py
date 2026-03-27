'''
Function to detect spindles within an EEG record
'''

import numpy as np
import statistics as stat
import scipy as sc
from Functions.utils import diff_envelops

'''
Function to call for detecting spindles
'''
def detect_spindles(y,lower_std_criteria,upper_std_criteria):
    '''
    Inputs:
    - y                   <-- the signal on which we want to find spindles (1D numpy array)
    - lower_std_criteria  <-- by how much the std is multiplied as a threshold for the lower bound (float)
    - upper_std_criteria  <-- by how much the std is multiplied as a threshold for the upper bound (float)
    '''

    # get standard deviation of the signal
    std=np.std(y)

    # get envelope of signal
    diff_env=diff_envelops(y)

    # get list of edges when crossing a value of std
    list_edges=get_bellow_segment(diff_env,std,lower_std_criteria)

    # get spindle segment (first choice has some trouble error)
    #list_spindles=get_spindle_interval_1(diff_env,std,list_edges,lower_std_criteria,upper_std_criteria)
    list_spindles=get_spindle_interval(diff_env,std,list_edges,upper_std_criteria)
    return list_spindles


'''
Function to get spindles segment with min max min pattern
'''
def get_spindle_interval(diff_env,std,list_edges,upper_std_criteria):

    # get interval with segment1<std max>3.std segment2<std pattern
    list_segment=[]
    for i in range(1,len(list_edges)-2,2):
        local_pos_max=list_edges[i]+np.argmax(diff_env[list_edges[i]:list_edges[i+1]])
        if diff_env[local_pos_max]>upper_std_criteria*std:
            list_segment.append([list_edges[i-1],list_edges[i],local_pos_max,list_edges[i+1],list_edges[i+2]])  # segment under std before and segment under std after having a ma above 3.std

    # get min max min pattern detection
    list_spindles=[]
    for segment in list_segment:
        list_spindles.append([segment[0]+np.argmin(diff_env[segment[0]:segment[1]]),segment[2],segment[3]+np.argmin(diff_env[segment[3]:segment[4]])])
        
    return list_spindles


'''
Function to get spindles segment with min max min pattern
'''
def get_spindle_interval_1(diff_env,std,list_edges,lower_std_criteria,upper_std_criteria):

    # get interval with segment1<st max>3.std segment2<std pattern
    list_segment=[]
    for i in range(1,len(list_edges)-2,2):
        local_pos_max=list_edges[i]+np.argmax(diff_env[list_edges[i]:list_edges[i+1]])
        if diff_env[local_pos_max]>upper_std_criteria*std:
            list_segment.append([list_edges[i-1],list_edges[i],local_pos_max,list_edges[i+1],list_edges[i+2]])  # segment under std before and segment under std after having a ma above 3.std

    # get min max min pattern detection
    list_spindles=[]
    for segment in list_segment:
        c=0
        try:
            # get all min bellow lower_std_criteria.std --- left segment
            left_min_peaks,_=sc.signal.find_peaks(-diff_env[segment[0]:segment[1]],height=-lower_std_criteria*std)
            # take the further right one
            left_min=segment[0] + left_min_peaks[-1]
            c+=1
        except: # should be an index error but in case we take the min
            print('error 0')
            #left_min=segment[0]+np.argmin(diff_env[segment[0]:segment[1]])

        try:    
            # get all min bellow lower_std_criteria.std --- right segment
            right_min_peaks,_=sc.signal.find_peaks(-diff_env[segment[3]:segment[4]],height=-lower_std_criteria*std)
            # take the further left one
            right_min=segment[3] + right_min_peaks[0]
            c+=1
        except: # should be an index error but in case we take the min
            print('error 1')
            #right_min=segment[3]+np.argmin(diff_env[segment[3]:segment[4]])

        # fill the list of spindles position
        if c==2:
            list_spindles.append([left_min,segment[2],right_min])     
        
    return list_spindles


'''
Get the segments that are bellow lower_std_criteria.std of the signal
'''
def get_bellow_segment(diff_env,std,lower_std_criteria):
    '''
    Inputs:
    - diff_env     <-- the difference envelope of the signal
    - std          <-- the standard deviation of the signal
    '''

    # length of difference envelope
    n=diff_env.size

    # find the edges of each segment bellow lower_std_criteria*std
    list_edges=np.where(np.diff(np.sign(diff_env-lower_std_criteria*std)))[0]     # check if we start or end in a spindle
    list_edges=np.insert(list_edges,0,0)
    list_edges=np.append(list_edges,n)
    # get minimum of each segment delimited by the edges

    return list_edges

