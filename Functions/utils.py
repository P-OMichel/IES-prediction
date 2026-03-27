import numpy as np
import pywt
import scipy as sc

'''
Function that detects the position of a 0 to 1 edge in a mask (the position is the one of the 1 in the mask)
as well as the position of a 1 to 0 edge (position is the one of the 1 in the mask)
'''
def detect_mask_edge(mask):
    '''
    Input:
    - mask   <--- 1D numpy array of 0 and 1
    Outputs:
    - edge_0_1   <--- position of the 1 in a 0 to 1 edge
    - edge_1_0   <--- position of the 1 in a 1 to 0 edge
    '''
    # in case array start with 1 to detect it as 0 to 1 edge
    mask=np.insert(mask,0,0)
    # in case array finishes with 1 to detect it as 1 to 0 edge
    mask=np.append(mask,0)

    # transformation that makes:
    # 0,0 <-- 0
    # 0,1 <-- 2
    # 1,0 <-- 1
    # 1,1 <-- 3
    mask=mask[:-1]+2*mask[1:]

    edge_0_1=np.where(mask==2)[0]
    edge_1_0=np.where(mask==1)[0]-1

    return edge_0_1,edge_1_0

'''
Function that returns a list of starting and ending position of 1 in a mask
ex:
Input=[0,1,1,0,1]
Output=[[1,2],[4,4]]
'''
def detect_pos_1(mask):

    edge_0_1,edge_1_0=detect_mask_edge(mask)
    N=edge_0_1.size
    pos=[]
    for i in range(N):
        pos.append([edge_0_1[i],edge_1_0[i]])
    
    return pos

def diff_envelops_signals(signals):

    N_line,N_column=np.shape(signals)
    env_signals=np.zeros(shape=(N_line,N_column))

    for i in range(N_line):
        env_signals[i,:]=diff_envelops(signals[i,:])

    return env_signals

'''
return the list of the interpolated upper and lower envelopes at every index of the list y
''' 
def diff_envelops(y):
    '''
    Inputs:
    - y        <-- signal for which the difference between the upper and lower envelops is computed (signal or filtered signal in a band f_int=[f_0,f_1] from the filter function and normalized by its RMS)

    Outputs:
    - diff_env  <-- difference of the interpolated upper and lower envelops
    '''
    N=y.size

    # creates upper and lower envelops of a signal
    list_pos_maxima,list_pos_minima=find_extremum(y)

    # set origin as a max and min to start with a difference envelope of 0
    list_pos_maxima=np.insert(list_pos_maxima,0,0)
    list_pos_minima=np.insert(list_pos_minima,0,0)

    # set last point as a max and min to end with a difference envelope of 0
    list_pos_maxima=np.append(list_pos_maxima,N-1)
    list_pos_minima=np.append(list_pos_minima,N-1)  

    # get list of maxima and minima
    list_maxima,list_minima=y[list_pos_maxima],y[list_pos_minima]

    # create an interpolation of the envelope at every time of the signal (otherwise we only have a list at the position of the extremum)
    index=[i for i in range(N)]
    ######### maybe index with np.arange is faster
    upper_env=np.interp(index,list_pos_maxima,list_maxima)  # upper envelop interpolation
    lower_env=np.interp(index,list_pos_minima,list_minima)  # lower envelop interpolation

    # difference of the envelops
    diff_env=upper_env-lower_env

    return diff_env

def find_extremum(y):
    '''
    Input:
    - y    <-- the signal where we want to find the local minimum and local maxima
    '''

    # compute the list of differences with right neighbour
    L_right=y[:-1]-y[1:]

    # it misses last point so we add it to keep same indexing as y (pos i is the difference of y_i with y_i+1)
    L_right=np.append(L_right,0)

    # compute the list of differences with left neighbour
    L_left=y[1:]-y[:-1]
    # it misses first point so we add it to keep same indexing as y (pos i is the difference of y_i with y_i+1)
    L_left=np.insert(L_left,0,0)

    # list that discriminates if max, min or not extremum (+3 for max, -1 if min, -3 or 1 if not extremum)
    L=np.sign(L_right)*np.sign(L_left)+2*np.sign(L_right)

    # get list of maximum
    list_pos_maxima=np.where(L==3)[0]
    
    # get list of minimum
    list_pos_minima=np.where(L==-1)[0]

    return list_pos_maxima,list_pos_minima

def get_extremum_list(y):
    '''
    function to get the extremum values of a 1D array (of positive values)
    '''
    list_pos_maxima,list_pos_minima=find_extremum(y)
    list_maximum=y[list_pos_maxima]
    list_minimum=y[list_pos_minima]
    list_extremum=np.concatenate((list_maximum,list_minimum))
    # remove outliers
    list_extremum=list_extremum[np.where(list_extremum<4*np.median(list_extremum))]
    return list_extremum,list_maximum,list_minimum

def find_maximum(y):
    '''
    Input:
    - y    <-- the signal where we want to find the local minimum and local maxima
    '''

    # compute the list of differences with right neighbour
    L_right=y[:-1]-y[1:]

    # it misses last point so we add it to keep same indexing as y (pos i is the difference of y_i with y_i+1)
    L_right=np.append(L_right,0)

    # compute the list of differences with left neighbour
    L_left=y[1:]-y[:-1]
    # it misses first point so we add it to keep same indexing as y (pos i is the difference of y_i with y_i+1)
    L_left=np.insert(L_left,0,0)

    # list that discriminates if max, min or not extremum (+3 for max, -1 if min, -3 or 1 if not extremum)
    L=np.sign(L_right)*np.sign(L_left)+2*np.sign(L_right)

    # get list of maximum
    list_pos_maxima=np.where(L==3)[0]

    return list_pos_maxima

def find_minimum(y):
    '''
    Input:
    - y    <-- the signal where we want to find the local minimum and local maxima
    '''

    # compute the list of differences with right neighbour
    L_right=y[:-1]-y[1:]

    # it misses last point so we add it to keep same indexing as y (pos i is the difference of y_i with y_i+1)
    L_right=np.append(L_right,0)

    # compute the list of differences with left neighbour
    L_left=y[1:]-y[:-1]
    # it misses first point so we add it to keep same indexing as y (pos i is the difference of y_i with y_i+1)
    L_left=np.insert(L_left,0,0)

    # list that discriminates if max, min or not extremum (+3 for max, -1 if min, -3 or 1 if not extremum)
    L=np.sign(L_right)*np.sign(L_left)+2*np.sign(L_right)

    # get list of minimum
    list_pos_minima=np.where(L==-1)[0]

    return list_pos_minima

'''
Computes a function on a sliding window of given length shape and overlap between two consecutive ones across a 1D signal
'''
def sliding_window_fct(y,Ws,shape,overlap,function):
    '''
    Inputs:
    - y         <--- 1D np.array that is slide through by sliding window of size Ws and  specific shape (shape is of size Ws)
    - Ws        <--- size of the sliding window
    - shape     <--- shape of the sliding window 
                    (ex: for boxcar set shape=1 for gaussian set to gaussian of size Ws with specific mean and std (as np.array type) )
    - overlap   <--- how much point are in common between two consecutive sliding windows
    - function  <--- function applied to the signal contained in the sliding window (must be fct of one variable of same type as y)
    '''
    N=y.size
    # divide signal into the sliding windows
    windows=[y[i:i+Ws]*shape for i in range(0,N-Ws,Ws-overlap)]
    # compute the function onto each window
    res=[function(wind) for wind in windows]

    return res

'''
Function to get the sliding std of a 1D signal using convolution 
'''
def sliding_std(y,h_length,mode='valid'):
    '''
    Inputs:
    - y         <--- the signal (as a 1D np.array)
    - h_length         <--- the length of the filter
    - mode      <--- which mode to use for the convolution: 
                    'full' to have all possible values
                    'valid' that gives only the ones that can be computed with a full filter
                    'same' to have output of same size as input
    '''
    # create filter for mean
    h=np.ones(h_length)/h_length

    yy=y**2
    # sliding mean of the squared signal
    mean_yy=np.convolve(yy,h,mode)

    # sliding mean of the signal then squared
    mean_y_squared=np.convolve(y,h,mode)**2

    return np.sqrt(mean_yy-mean_y_squared)

'''
Function to get the sliding mean power of a 1D signal using convolution 
'''
def sliding_power(y,h_length,mode='valid'):
    '''
    Inputs:
    - y         <--- the signal (as a 1D np.array)
    - h_length         <--- the length of the filter
    - mode      <--- which mode to use for the convolution: 
                    'full' to have all possible values
                    'valid' that gives only the ones that can be computed with a full filter
                    'same' to have output of same size as input
    '''
    # create filter for mean
    h=np.ones(h_length)/h_length

    # square the signal to get the instantaneous power
    yy=y**2

    return np.convolve(yy,h,mode)

def wav_thresholding(y,wavelet,mode,level,quantiles):

    coeffs=pywt.wavedec(y,wavelet,mode,level)
    
    for i in range(len(coeffs)):
        abs_coeffs=np.abs(coeffs[i])
        coeffs[i][np.where(abs_coeffs<np.quantile(abs_coeffs,quantiles[i]))[0]]=0

    return coeffs

'''
Function that return a mask after erosion/dilatation/erosion
'''
def erosion_dilation(mask,min_band,max_gap,fs):
    '''
    Inputs:
    - mask      <-- mask of 1 at a position of a value of interest and 0 elsewhere
    - min_band  <-- min lenght of correct values (minimal lenght of segment of 1)
    - max_gap   <-- max lenght of incorrect values (maximal lenght of segment of 0)
    
    Outputs:
    - new_mask  <-- mask created after a process of erosion/dilatation/erosion (explain each phase)
    '''
    # translate the min_band and max_gap from time duration to interval length
    min_band=int(fs*min_band) 
    max_gap=int(fs*max_gap)-1   # minus 1 to have the correct effect on erosion (erosion with int 3 for instance takes away 2 by construction of the function)

    # routine to erode, dilate and erode the mask
    new_mask=sc.ndimage.binary_erosion(mask,np.ones(min_band),iterations=1)
    new_mask=sc.ndimage.binary_dilation(new_mask,np.ones(min_band+max_gap),iterations=1)
    new_mask=sc.ndimage.binary_erosion(new_mask,np.ones(max_gap),iterations=1)

    return new_mask


def make_derivative(y,dx):

    dy=np.zeros(len(y))

    # first point derivative
    dy[0]=(y[1]-y[0])/dx

    # points having a left and right neighbour
    dy[1:-1]=(y[2:]-y[:-2])/(2*dx)

    # last point derivative
    dy[-1]=(y[-1]-y[-2])/dx

    return dy


def zero_crossing(signal):
    '''
    Input:
    - signal <-- 1D numpy array
    Output
    - N_zero_crossing <-- number of time the signal crosses 0
    '''

    # remove all zero values to avoid not counting crossing ( 3 0 -3 becomes 3 -3 and there is a direct crossing that is seen)
    non_zeros_indices = np.where(signal != 0)[0]
    non_zero_signal = signal[non_zeros_indices]

    # get the sign list
    sign_list = np.sign(non_zero_signal)

    # multiplication of the sign_list with itself shifted
    # it gives:
    #          -1 when there is a crossing of zero
    #          +1 when there is no crossing
    #           0 when at least one of the two multiplied value is zero
    mult_list = sign_list[:-1]*sign_list[1:]

    # count the number of time there is -1
    N_zero_crossing=len(np.where(mult_list == -1)[0])

    return N_zero_crossing


def barycenter(x,y):
    
    return np.sum(x*y)/np.sum(y)

def get_PSD_barycenter(y,fs,height):

    f, Pxx_den = sc.signal.periodogram(y, fs)
    Pxx_den=Pxx_den/np.max(Pxx_den)
    indices=sc.signal.find_peaks(Pxx_den,height=height)[0]
    x_b=barycenter(f[indices],Pxx_den[indices])

    return f,Pxx_den,x_b

def detect_ground_check(y,N,T):
    
    # get smooth power of signal
    y2=np.convolve(y**2,np.ones(N)/N,mode='same')

    # create mask where 0 is for a value lower than T and 1 for higher
    mask=np.zeros(len(y))
    mask[np.where(y2 >= T)[0]]=1

    # create mask of 1 in area of ground check high values should be
    h=np.zeros(1280)
    h[:192]=1
    h[-192:]=1 

    # convolution of mask and h to detect the ground check
    mask=np.convolve(mask,h,mode='same')

    return mask

def var_tot(y):

    # compute derivative
    dy = np.gradient(y)
    # get total variation
    var=np.sum(np.abs(dy))/len(dy)

    return var

def gaussian(x,x0,sigma):

    return np.exp(-1*(x-x0)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)

def get_gaussian_h(h,x0,sigma):
    h=gaussian(h,x0,sigma)
    return h/np.sum(h)

def upsample_signal(signal, original_fs, target_fs):

    # Create the time vector for the original signal
    original_time = np.arange(0, len(signal) / original_fs, 1 / original_fs)
    
    # Create the time vector for the target sampling frequency
    target_time = np.arange(0, len(signal) / original_fs, 1 / target_fs)
    
    # Interpolate the signal using linear interpolation
    interpolator = sc.interpolate.interp1d(original_time, signal, kind='linear', fill_value='extrapolate')
    upsampled_signal = interpolator(target_time)
    
    return upsampled_signal








