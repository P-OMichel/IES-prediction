'''
File to get the frequency at the maximum amplitude of a spindle based on the instantaneous frequency from the Hilbert transform
'''
import scipy as sc
import numpy as np

def get_frequency_max(y,fs):
    '''
    Inputs:
    y --> the signal to extract the frequency at the maximum amplitude from
    fs --> sampling frequency of the signal
    Outputs:
    max_amp --> the maximum amplitude
    freq_max --> frequency at the maximum amplitude
    ''' 

    # Hilbert transform
    analytic_signal=sc.signal.hilbert(y)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * fs)

    # get index of max amplitude
    abs_y=abs(y)
    pos_max=np.argmax(abs_y)

    # get maximum amplitude
    max_amp=abs_y[pos_max]

    # get frequency corresponding to the maximum amplitude
    try:
        freq_max=instantaneous_frequency[pos_max]
    except IndexError:
        # instanteneous_frequency has one component less than the signal, in case the maximum amplitude is 
        # at the end of it we take the last component of the frequency list
        freq_max=instantaneous_frequency[-1]

    return max_amp,freq_max
