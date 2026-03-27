'''
Function to retrieve the parameters into python objects from the json file
'''

import json

def convert_json_python(json_file):
    '''
    Input:
    - json_file  <-- str name of the json file to read that contains the parameters
    
    
    '''

    with open(json_file) as parameters_file:
        parameters=json.load(parameters_file)

    fs=parameters['fs']                                 # frequency sampling step of the recording device

    window_size=parameters['window_size']               # window size to process

    list_detection=parameters['list_detection']         # parameters for the detection of artifacts function

    list_WQN=parameters['list_WQN']                     # parameters for the correction of artefacts

    list_Spectro=parameters['list_Spectro']             # parameters for the computation of the spectrogram

    list_power_contribution=parameters['list_power_contribution']  # list of percentages. Each percentage is a power proportion and the frequency at which this proportion is reached is searched

    list_freq_interval=parameters['list_freq_interval']            # list of frequency intervals. How much power is contained within this interval is then computed

    list_fragmentations=parameters['list_fragmentations']          # list of frequency intervals of interest to look for fragmentations and threshold value.

    list_suppressions=parameters['list_suppressions']

    list_spindles=parameters['list_spindles']

    return fs,window_size,list_detection,list_WQN,list_Spectro,list_power_contribution,list_freq_interval,list_fragmentations,list_suppressions,list_spindles                    


def convert_json_python_colors(json_file):
    '''
    Input:
    - json_file  <-- str name of the json file to read that contains the parameters
    
    
    '''

    with open(json_file) as colors_file:
        colors=json.load(colors_file)

    colors_power_contribution=colors['colors_power_contribution']
    colors_freq_interval=colors['colors_freq_interval']
    colors_fragmentations=colors['colors_fragmentations']
    colors_suppressions=colors['colors_suppressions']

    return colors_power_contribution,colors_freq_interval,colors_fragmentations,colors_suppressions