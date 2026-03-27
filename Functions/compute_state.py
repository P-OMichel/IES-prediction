'''
Function to determine the brain state based on the EEG signal
'''

from Functions.filter import get_filtered_signal
from Functions.utils import diff_envelops_signals,zero_crossing
from Functions.compute_fragmentation import compute_fragmentation_level
from Functions.suppressions import compute_suppression_mask_rate
import numpy as np

def get_state(y,fs,list_freq_int,h,quantile,threshold,mode='same',threshold_power=[40,1.5,3],threshold_frag=[0.05,0.15]):


    #--- filter signals

    filtered_signals=get_filtered_signal(y,fs,list_freq_int)

    #--- get fragmentation

    # get difference envelops
    env_signals=diff_envelops_signals(filtered_signals)

    # get fragmentation level 
    list_frag=compute_fragmentation_level(env_signals,h,quantile,threshold,mode)


    #--- get power

    # power of each filtered frequency interval signal
    list_power=np.median(filtered_signals**2,axis=1)

    # add power of y signal
    #list_power=np.append(list_power,np.median(y**2))
    list_power=np.append(list_power,np.sum(list_power))

    #--- determine state

    # parameters 
    T_power=threshold_power[0]
    T_gamma_alpha=threshold_power[1]
    T_beta_alpha=threshold_power[1]

    T_IES=threshold_frag[0]
    T_F_alpha=threshold_frag[1]

    # if power of eeg recording is very low --> Suppression area --> Too heavy
    if list_power[-1] < T_power:
        state = 0
    
    # if P_gamma/P_alpha is higher than a threshold --> very Shallow or Awake
    elif list_power[3]/list_power[1] > T_gamma_alpha:
        state = 4
    
    # if P_beta/P_alpha is higher than a threshold --> Shallow
    elif list_power[2]/list_power[1] > T_beta_alpha:
        state = 3
    
    # if the amount a IES is superior to threshold
    elif list_frag[-1] > T_IES:
        state = 0
    
    # if the amount a IES is superior to threshold
    elif list_frag[1] > T_F_alpha:
        state = 1
    
    # if none of the above criteria is met --> Ok, may not be true but if state chart correclty made then yes
    else:
        state = 2

    return state, list_frag, list_power



def get_state_supp(y,fs,list_freq_int,threshold_power,threshold_frag,mask_size,threshold,list_morph,f_int,method):

    #--- filter signals

    filtered_signals=get_filtered_signal(y,fs,list_freq_int)

    #--- get fragmentation

    # get suppression scores
    # list_frag[0] is IES proportion and list_frag[0] is alpha -suppression proportion
    list_frag=compute_suppression_mask_rate(y,mask_size,fs,threshold,list_morph,f_int,method)[2:]  
    

    #--- get power

    # power of each filtered frequency interval signal
    list_power=np.median(filtered_signals**2,axis=1)

    # add power of y signal
    #list_power=np.append(list_power,np.median(y**2))
    list_power=np.append(list_power,np.sum(list_power))

    #--- determine state

    # parameters 
    T_gamma_alpha=threshold_power[0]
    T_beta_alpha=threshold_power[1]

    T_IES=threshold_frag[0]
    T_F_alpha=threshold_frag[1]

    # High presence of IES --> Too heavy
    if list_frag[0] > T_IES:
        state = 0
    
    # if P_gamma/P_alpha is higher than a threshold --> very Shallow or Awake
    elif list_frag[1] > T_F_alpha and list_power[3]/list_power[1] <= T_gamma_alpha and list_power[2]/list_power[1] <= T_beta_alpha:
        state =1

    elif list_power[3]/list_power[1] > T_gamma_alpha:
        state = 4
    
    # if P_beta/P_alpha is higher than a threshold --> Shallow
    elif list_power[2]/list_power[1] > T_beta_alpha:
        state = 3
    
    # if none of the above criteria is met --> Ok, may not be true but if state chart correclty made then yes
    else:
        state = 2


    return state, list_frag, list_power


def get_full_state_supp(y,fs,list_freq_int,threshold_power,threshold_frag,mask_size,threshold,list_morph,f_int,method):

    #--- filter signals

    filtered_signals=get_filtered_signal(y,fs,list_freq_int)

    #--- get fragmentation

    # list_frag[0] is IES proportion and list_frag[1] is alpha -suppression proportion
    list_frag=compute_suppression_mask_rate(y,mask_size,fs,threshold,list_morph,f_int,method)[2:]  

    #--- get power

    # power of each filtered frequency interval signal
    list_power=np.median(filtered_signals**2,axis=1)

    # add power of y signal
    #list_power=np.append(list_power,np.median(y**2))
    list_power=np.append(list_power,np.sum(list_power))

    #--- determine  main state

    # parameters 
    T_gamma_alpha=threshold_power[0]
    T_beta_alpha=threshold_power[1]

    T_IES=threshold_frag[0]
    T_F_alpha=threshold_frag[1]

    # High presence of IES --> Too heavy
    if list_frag[0] > T_IES:
        main_state = 0
    
    # alpha-suppressions (might need some changes in thresholds for power)
    elif list_frag[1] > T_F_alpha and list_power[3]/list_power[1] <= 0.75 and list_power[2]/list_power[1] <= 1.5:
        main_state = 1

    # if P_gamma/P_alpha is higher than a threshold --> very Shallow or Awake
    elif list_power[3]/list_power[1] > T_gamma_alpha:
        main_state = 4
    
    # if P_beta/P_alpha is higher than a threshold --> Shallow
    elif list_power[2]/list_power[1] > T_beta_alpha or list_power[3]/list_power[1] > 0.25:
        main_state = 3
    
    # if none of the above criteria is met --> Ok, may not be true but if state chart correclty made then yes
    else:
        main_state = 2

    #--- determine state ( put thresholds in function input)

    if main_state == 0:

        if list_frag[0] > 0.4:
            state = 0
        elif list_frag[0] > 0.3:
            state = 1 
        elif list_frag[1] > 0.2:
            state = 2
        else:
            state = 3
    
    elif main_state == 1:

        if list_frag[1] > 0.2:
            state = 4
        elif list_frag[1] > 0.16:
            state = 5
        elif list_frag[1] > 0.12:
            state = 6
        else:
            state = 7
        
    elif main_state == 4:

        if list_power[3]/list_power[1] > 50:
            state = 16
        elif list_power[3]/list_power[1] > 35:
            state = 15
        elif list_power[3]/list_power[1] > 15:
            state = 14
        else:
            state = 13

    elif main_state == 3:

        if list_power[2]/list_power[1] > 3 or list_power[3]/list_power[1] > 0.9:
            state = 12
        elif list_power[2]/list_power[1] > 2 or list_power[3]/list_power[1] > 0.75:
            state = 11
        elif list_power[2]/list_power[1] > 1.5 or list_power[3]/list_power[1] > 0.5:
            state = 10
        else:
            state = 9

    elif main_state == 2:
        state = 8

    # for awake before induction
    if zero_crossing(np.abs(y)-90) > 50:
        state=16

    return state, list_frag, list_power


'''
currently set on mean of two values and three averaged values for trend (total of 1 min with a 10 s step)
'''
def get_trend(list_states):
    '''
    Function to cal to obtain the trend of the state evolution
    '''
    # average the state value
    averaged_states=np.array([np.mean(list_states[i:i+2]) for i in [0,2,4]])
    # get sign difference between states
    sign_list=np.sign(averaged_states[1:]-averaged_states[:-1])
    # get trend
    if sign_list[0]==0 and sign_list[1]==0:
        return 0
    elif (sign_list[0]==0 and sign_list[1]==1)  or (sign_list[0]==1 and sign_list[1]==0) or (sign_list[0]==1 and sign_list[1]==1):
        return 1
    elif (sign_list[0]==0 and sign_list[1]==-1)  or (sign_list[0]==-1 and sign_list[1]==0) or (sign_list[0]==-1 and sign_list[1]==-1):
        return -1
    else: #(+1 then -1 or opposite)
        return 5

