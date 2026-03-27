import numpy as np
import scipy as sc
import pywt
from scipy.signal import butter, sosfiltfilt


def get_filtered_signal(y,fs,list_freq_int):

    N=y.size
    N_freq_int=len(list_freq_int)
    filtered_signals=np.zeros(shape=(N_freq_int,N))
    for k in range(N_freq_int):
        filtered_signals[k,:]=filter_butterworth(y,fs,list_freq_int[k])
    return filtered_signals

def filter_butterworth(y,fs,f_int,order=4):

    nyq = 0.5 * fs
    low = f_int[0] / nyq
    high = f_int[-1] / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    filtered_signal = sosfiltfilt(sos, y)

    return filtered_signal

def filter_FFT(y,fs,f_int):
    '''
    Inputs:
    - y          <-- signal to filter
    - fs         <-- sampling step frequency of the signal
    - f_int      <-- frequency interval to keep using the filter f_int=[f0,f1],f0<f1
    - method     <-- str being either 'FFT', 'DWT' or 'BUTTER' selecting which method to use for the spectral decomposition
    - arg        <-- if 'DWT' is used we have to select the wavelet name and mode, and in equality_case to select True or False whether to use FFT after or not.
                     The arg is entered as a list [wavelet_name,mode,equality_case] of str

                     if 'BUTTER' is used we have to select the order of the filter.
                     The arg is entered as an int butter_order
                    

    Outputs:
    - filtered_y <--the signal filtered keeping only the values of frequencies in f_int
    '''

    N=y.size

    fft_y=sc.fft.rfft(y)                # compute fft of the signal to access its spectrum, we can use rfft since y is a real time series (faster as it computes the fft only for the f>0)
        
    delta_f=(fs/2)/len(fft_y)           # value of one bin frequency interval, we use this to find the index position in list_f of the frequencies in f_int
        
    i=int(f_int[0]/delta_f)             # index position corresponding to the f0 value
    j=int(np.ceil(f_int[1]/delta_f))    # index position corresponding to the f1 value
        
    fft_y[:i]=0                         # only keep the Fourier transform in the frequency interval of interest
    fft_y[j:]=0
        
    filtered_y=sc.fft.irfft(fft_y)      # inverse FFT on the cutted Fourier transform, this gives the signal with only the components in f_int

    return filtered_y

'''
Function to filter a signal keeping only frequencies in [f0,f1] using either FFT, DWT or Butterworth filter
'''
def filter(y,fs,f_int,method,*arg):
    '''
    Inputs:
    - y          <-- signal to filter
    - fs         <-- sampling step frequency of the signal
    - f_int      <-- frequency interval to keep using the filter f_int=[f0,f1],f0<f1
    - method     <-- str being either 'FFT', 'DWT' or 'BUTTER' selecting which method to use for the spectral decomposition
    - arg        <-- if 'DWT' is used we have to select the wavelet name and mode, and in equality_case to select True or False whether to use FFT after or not.
                     The arg is entered as a list [wavelet_name,mode,equality_case] of str

                     if 'BUTTER' is used we have to select the order of the filter.
                     The arg is entered as an int butter_order
                    

    Outputs:
    - filtered_y <--the signal filtered keeping only the values of frequencies in f_int
    '''

    N=y.size


    if method=='FFT':

        fft_y=sc.fft.rfft(y)                # compute fft of the signal to access its spectrum, we can use rfft since y is a real time series (faster as it computes the fft only for the f>0)
        
        delta_f=(fs/2)/len(fft_y)           # value of one bin frequency interval, we use this to find the index position in list_f of the frequencies in f_int
        
        i=int(f_int[0]/delta_f)             # index position corresponding to the f0 value
        j=int(np.ceil(f_int[1]/delta_f))    # index position corresponding to the f1 value
        
        fft_y[:i]=0                         # only keep the Fourier transform in the frequency interval of interest
        fft_y[j:]=0
        
        filtered_y=sc.fft.irfft(fft_y)      # inverse FFT on the cutted Fourier transform, this gives the signal with only the components in f_int

    elif method=='BUTTER':
        
        butter_order=int(*arg)              # take the order of the filter to use that has been passed as an argument (set to int to avoid any issue)
        
        sos=sc.signal.butter(butter_order,f_int,btype='bandpass',fs=fs,output='sos')  # create the filter second-order section  
        filtered_y=sc.signal.sosfilt(sos,y)                                           # filters the signal using the second-order section

    elif method=='DWT':

        wavelet_name,mode,equality_case=arg
        
        max_level=pywt.dwt_max_level(N,wavelet_name)  # max level of decomposition with the given wavelet and size of signal

        i_1=int(np.log2(fs/f_int[1]))                 # max possible value for level decomposition containing f1
        i_0=int(np.ceil(np.log2((fs/f_int[0]))))      # min possible value for level decomposition containing f0 (the number of this coefficient is i_0-1 as i_0 leads to the high frequency of the next coeff)

        if i_0>max_level+1:
            print('not enought input data or lower frequency too low')
        elif i_1<2:
            print('not enough data or higher frequency too high')        
        else:
            max_level=i_0-1                          # max level to use is the one containing f_0, we do not have to go higher as the next detail coefficient will not contain values in the frequency
            list_coeff=[i for i in range(i_1,i_0)]   # list of coefficient to use [i_1,i_1+1,...,i_0-1], can contain only one index if i_0-1=i_1 

            coeffs=pywt.wavedec(y,wavelet_name,mode,max_level)  # Discrete Wavelet Transform (DWT) on the signal 

            for k in range(max_level+1):                        # max level number of detail coeff and 1 approximation one. max_level+1 is the size of coeffs
                if max_level+1-k not in list_coeff:                 # max_level+1-i translate the loop into one on the coefficient number with the approx one counted as max_level+1
                    coeffs[k]=np.zeros(len(coeffs[k]))                  # set the coefficients with no frequency in f_int to zero
   
            filtered_y=pywt.waverec(coeffs,wavelet_name,mode) # restitution of the signal with only the coefficients containing frequencies of f_int

        # to this point if int(np.log2(fs/f_int[1]))!=np.log2(fs/f_int[1]) and/or np.ceil(np.log2((fs/f_int[0])))!=np.log2((fs/f_int[0]))
        # we can also have values outside of f_int in DWT_filtered_y. Using the FFT method we can extract them.
              
        if np.log2(fs/f_int[1])!=int(np.log2(fs/f_int[1])) or np.log2((fs/f_int[0]))!=np.ceil(np.log2((fs/f_int[0]))):
            filtered_y=filter(filtered_y,fs,f_int,'FFT')

        elif np.log2(fs/f_int[1])==int(np.log2(fs/f_int[1])) and np.log2((fs/f_int[0]))==np.ceil(np.log2((fs/f_int[0]))):
            if equality_case=='True':
                filtered_y=filter(filtered_y,fs,f_int,'FFT')  # either way using wavelet can lead to multiple frequencies outside of f_int so using FFT afterwards can be good

    else:
        print('incorrect method name')


    return filtered_y



