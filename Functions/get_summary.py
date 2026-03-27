'''
File to call to get the summary of the data for the balance sheet
'''

import csv 
import numpy as np
from Functions.suppressions import compute_suppression_mask_rate
from Functions.utils import detect_pos_1


class SummaryData:

    def __init__(self,fs,window_size,
                mask_size,
                threshold,
                list_morph,
                f_int,
                method):
        
        super().__init__()

        self.fs=fs
        self.window_size=window_size
        self.mask_size=mask_size
        self.threshold=threshold
        self.list_morph=list_morph
        self.f_int=f_int
        self.method=method


        # initialize dictionnary that stores the data
        self.D={}
        self.D['IES duration'] = 0
        self.D['Number of IES'] = 0
        self.D['Longest IES'] = np.array([0,0,0])
        self.D['Alpha suppression duration'] = 0
        self.D['Number of Alpha suppression'] = 0
        self.D['Longest Alpha suppression'] = np.array([0,0,0])
        self.D['Time spent under Heavy state'] = 0
        self.D['Time spent above Shallow state'] = 0
        self.D['Total time of anesthesia'] = 0
        self.D['alpha increase at recovery'] = None
        self.D['sensitivity score'] = None

        self.field_names=['IES duration','Number of IES','Longest IES','Alpha suppression duration','Number of Alpha suppression',
                          'Longest Alpha suppression','Time spent under Heavy state','Time spent above Shallow state','Total time of anesthesia',
                          'alpha increase at recovery','sensitivity score']


    def get_suppressions(self,y):
        '''
        Method to process suppressions related data
        y: the corrected eeg data to analyse
        '''

        #--- update time duration of anesthesia
        self.D['Total time of anesthesia'] += len(y)/self.fs

        #--- get suppressions data
        mask_IES, mask_alpha = compute_suppression_mask_rate(y,
                                                            self.mask_size,
                                                            self.fs,
                                                            self.threshold,
                                                            self.list_morph,
                                                            self.f_int,
                                                            self.method)[:2]
            
        # intervals corresponding to the suppressions
        pos_edges_IES = detect_pos_1(mask_IES)
        pos_edges_alpha_suppressions = detect_pos_1(mask_alpha)

        # convert intervals into their duration (may need to have a checkup that the pos_edges are not empty) 
        intervals_lengths_IES = np.array([x[1]-x[0] for x in pos_edges_IES])/self.fs
        intervals_lengths_alpha_suppressions = np.array([x[1]-x[0] for x in pos_edges_alpha_suppressions])/self.fs

        #--- get cumulative duration of suppressions
        self.D['IES duration'] += np.sum(intervals_lengths_IES)
        self.D['Alpha suppression duration'] += np.sum(intervals_lengths_alpha_suppressions)

        #--- get numbers of suppressions
        self.D['Number of IES'] += len(intervals_lengths_IES)
        self.D['Number of Alpha suppression'] += len(intervals_lengths_alpha_suppressions)

        #--- get 3 longest suppressions
        # position above the minimum of the previous IES three longest
        pos_IES_above_min_longest_IES = np.where(intervals_lengths_IES > np.min(self.D['Longest IES']))[0]
        pos_alpha_suppressions_above_min_longest_alpha_suppressions = np.where(intervals_lengths_alpha_suppressions > np.min(self.D['Longest Alpha suppression']))[0]

        # concatenate list of previous longest suppressions and the list of suppressions above the min of the previous maximum
        IES_list = np.concatenate((self.D['Longest IES'],intervals_lengths_IES[pos_IES_above_min_longest_IES]))
        alpha_suppressions_list = np.concatenate((self.D['Longest Alpha suppression'],intervals_lengths_alpha_suppressions[pos_alpha_suppressions_above_min_longest_alpha_suppressions]))

        # sort lists and take the three maxima
        sorted_IES_list = np.sort(IES_list)
        sorted_alpha_suppressions_list = np.sort(alpha_suppressions_list)

        # take 3 longest suppressions
        self.D['Longest IES'] = sorted_IES_list[-3:]
        self.D['Longest Alpha suppression'] = sorted_alpha_suppressions_list[-3:]

    def state_description(self,states_list):
        '''
        Method to get the time spent under heavy states and shallow states
        states_list: states history list
        '''    
        
        # the time spent under a given state is the number of times this state occurs multiplied by the duration of the window it describes
        self.D['Time spent under Heavy state'] += len(np.where(states_list*4 <= 4)[0])*self.window_size/self.fs 
        self.D['Time spent above Shallow state'] += len(np.where(states_list*4 >= 12)[0])*self.window_size/self.fs

    def store_alpha_evolution_recovery(self,alpha_evolution):
        '''
        Method to store the alpha evolution at the end of recovery
        '''
        self.D['alpha increase at recovery']=alpha_evolution

    def store_sensitivity_score(self,sensitivity_score):
        '''
        Method to store the sensitivity score into the dictionnary
        '''

        self.D['sensitivity score']=sensitivity_score

    def save_to_csv(self,name):
        '''
        Method to save the dictionnary into a CSV file
        '''

        with open(name + '.csv', 'w') as csvfile: 
            writer = csv.DictWriter(csvfile, fieldnames = self.field_names) 
            writer.writeheader() 
            writer.writerow(self.D) 
     
