'''
Function to obtain all the needed data once a batch arrives in the real time monitoring of anesthesia
'''
import numpy as np
from Functions.compute_on_window import compute_window_light
from Functions.compute_state import get_state_supp


class RetrieveData:
    
    
    def __init__(self,fs,window_size,
                   list_detection,
                   list_WQN,list_Spectro,
                   list_power_contribution,
                   state_batch_size,list_freq_int,threshold_power,threshold_frag,mask_size,threshold,list_morph,f_int,method,
                   N_states,
                   state_evolution_mode):
        super().__init__()

        self.fs=fs
        self.window_size=window_size
        self.list_detection=list_detection
        self.list_WQN=list_WQN
        self.list_Spectro=list_Spectro
        self.list_power_contribution=list_power_contribution
        self.state_batch_size=state_batch_size
        self.list_freq_int=list_freq_int
        self.threshold_power=threshold_power
        self.threshold_frag=threshold_frag
        self.mask_size=mask_size
        self.threshold=threshold
        self.list_morph=list_morph
        self.f_int=f_int
        self.method=method
        self.N_states=N_states
        self.state_evolution_mode=state_evolution_mode

        # initialize counter 
        self.N_window_size=0

        # initialize eeg_data to keep for the state determination
        self.eeg_data=np.zeros(self.state_batch_size)

        # initialize state list
        self.state_list=np.zeros(self.N_states)

        # state evolution markers
        self.D_trend={}
        self.D_trend['-1']='Heavier'
        self.D_trend['0']='Stable'
        self.D_trend['1']='Lighter'

        
    def get_clean_eeg_spectro(self,y):
        '''
        Method to call to obtain the clean eeg, the spectrogram and median/SEF
        '''
    
        #--- detect and correct artefacts, compute spectrogram
        y_corr,f,t,spectro,list_index=compute_window_light(y,
                                                            self.fs,
                                                            self.window_size,
                                                            self.list_detection,
                                                            self.list_WQN,
                                                            self.list_Spectro,
                                                            self.list_power_contribution)

        #--- update data for state determination
        # remove oldest points
        self.eeg_data=self.eeg_data[self.window_size:]
        # append new points
        self.eeg_data=np.concatenate((self.eeg_data,y_corr))

        # update the counter of batch
        self.N_window_size+=1

        print(self.N_window_size)

        return y_corr,f,t,spectro,list_index
    
    def get_state(self):
        '''
        Method to call to obtain the state
        '''
        
        if self.N_window_size>=int(self.state_batch_size/self.window_size):
            state=get_state_supp(self.eeg_data,
                                self.fs,
                                self.list_freq_int,
                                self.threshold_power,
                                self.threshold_frag,
                                self.state_batch_size,
                                self.threshold,
                                self.list_morph,
                                self.f_int,
                                self.method)[0]
            
            #--- update states history for state evolution trend
            # remove oldest state
            self.state_list=self.state_list[1:]
            # append new points
            self.state_list=np.append(self.state_list,state)

        else:
            state=None

        return state

        
    def state_evolution(self):
        '''
        Method to get the trend of state evolution
        '''

        if self.state_evolution_mode=='Unit': # done considering of history of 9 states
            # compute sign of state differences
            sign_0=np.sign(self.state_list[3]-self.state_list[0])
            sign_1=np.sign(self.state_list[6]-self.state_list[3])
            # if both sign are the same
            if sign_0==sign_1:
                trend_marker=self.D_trend[str(sign_0)]
            # if signs are different we do not conclude on the trend
            else:
                trend_marker='No possible trend computation'

        elif self.state_evolution_mode=='Average':
            # make averages
            mean_0=np.mean(self.state_list[:3])
            mean_1=np.mean(self.state_list[3:6])
            mean_2=np.mean(self.state_list[6:])
            # compute sign of state differences
            
            sign_0=np.sign(mean_1-mean_0)
            sign_1=np.sign(mean_2-mean_1)
            # if both sign are the same
            if sign_0==sign_1:
                trend_marker=self.D_trend[str(sign_0)]
            # if signs are different we do not conclude on the trend
            else:
                trend_marker='No possible trend computation'

        return trend_marker

























    