import numpy as np
import matplotlib.pyplot as plt
from Functions.detect_ground_check import clean_ground_check
from Functions.detect_artifacts import find_artifacts
from Functions.WaveletQuantileNormalization import WQN_3
import Functions.sliding_fct as sliding
from Functions.filter import filter_butterworth
import scipy as sc
from scipy.ndimage import gaussian_filter1d

from time import time

'''
def get_label_0_data(D,delta_t,smoothing=False):

    prior_t: in seconds, how much in advance form the start iof BS we take the signal
    delta_t: size of window in seconds

    # list to store the data
    L=[]

    for key in D.keys():
        print(key)

        fs=D[key]['fs']
        ti=D[key]['no BS']

        # get signal
        if D[key]['method'] == 0:
            EEG_data=sc.io.loadmat('Recordings/' + key[:-4]) 
            y=EEG_data['record'][:,0]

        elif D[key]['method'] == 1:
            y = np.load(key)

        elif D[key]['method'] == 2:
            EEG_data=sc.io.loadmat('D:/BS/' + key) 
            y=EEG_data['record'][0,:]

        # clean signal
        try:
            y=clean_ground_check(y,fs,16,7.5,8.5,50000)
        except:
            print('no ground check detected')
            
        N_y=len(y)
        t=np.linspace(0,N_y/fs,N_y)

        #--- clean EEG
        # detection parameters
        list_detection=[2*fs,1*fs,[0.0004,0.002],"sym4",4,"periodization"]  
        # pos of artefacts
        index_mask=find_artifacts(y,*list_detection) 

        # WQN parameters
        list_WQN=["sym4","periodization",30,1]

        # clean artefacts
        y=WQN_3(y,index_mask,*list_WQN) 

        # get part of the signal of interest
        t_part, y_part = get_part(y,t,ti,fs,delta_t+90+90)

        # get data
        Ws = int(90*fs)
        step = int(5*fs)
        L.append(get_data(y_part,t_part,fs,Ws,step,smoothing))

    return L


def get_label_1_data(D,prior_t,delta_t,smoothing=False):

    prior_t: in seconds, how much in advance form the start iof BS we take the signal
    delta_t: size of window in seconds

    # list to store the data
    L=[]

    for key in D.keys():
        print(key)

        fs=D[key]['fs']
        ti=D[key]['before BS'] - prior_t

        # get signal
        if D[key]['method'] == 0:
            EEG_data=sc.io.loadmat('Recordings/' + key[:-4]) 
            y=EEG_data['record'][:,0]

        elif D[key]['method'] == 1:
            y = np.load(key)

        elif D[key]['method'] == 2:
            EEG_data=sc.io.loadmat('D:/BS/' + key) 
            y=EEG_data['record'][0,:]

        # clean signal
        try:
            y=clean_ground_check(y,fs,16,7.5,8.5,50000)
        except:
            print('no ground check detected')
            
        N_y=len(y)
        t=np.linspace(0,N_y/fs,N_y)

        #--- clean EEG
        # detection parameters
        list_detection=[2*fs,1*fs,[0.0004,0.002],"sym4",4,"periodization"]  
        # pos of artefacts
        index_mask=find_artifacts(y,*list_detection) 

        # WQN parameters
        list_WQN=["sym4","periodization",30,1]

        # clean artefacts
        y=WQN_3(y,index_mask,*list_WQN) 

        # get part of the signal of interest
        t_part, y_part = get_part(y,t,ti,fs,delta_t+90+90)

        # get data
        Ws = int(90*fs)
        step = int(5*fs)
        L.append(get_data(y_part,t_part,fs,Ws,step,smoothing))

    return L

def get_part(y,t,ti,fs,delta_t):

    i=int(ti*fs) 
    i_delta_t=int((delta_t) *fs)
    y_part=y[i-i_delta_t:i]
    t_part=t[i-i_delta_t:i]

    return t_part,y_part

def get_data(y_part,t_part,fs,Ws,step,smoothing):

    y_part = filter_butterworth(y_part,fs,[7,14])

    # get power of alpha wave
    P = sliding.power_1D(y_part,t_part,Ws,step)[-1]

    # get BE of alpha wave
    be = sliding.entropy_1D(y_part,t_part,Ws,step,k=2)[-1]

    # get fragmentation of alpha wave
    T=0.2
    q=0.8
    N=int(fs/2)
    h=np.ones(N)/N
    fragmentation = sliding.get_fragmentation(y_part,t_part,Ws,step,h,q,T,mode='same')[-1]

    # get spindles number
    upper_T = 2
    lower_T =1
    N_spindles = sliding.get_N_spindles_power(y_part, t_part, Ws, step, upper_T, lower_T)[-1]

    if smoothing == True:
        #P = gaussian_filter1d(P, int(Ws/step),radius=int(Ws/2/step))
        #be = gaussian_filter1d(be, int(Ws/step),radius=int(Ws/2/step))
        #fragmentation = gaussian_filter1d(fragmentation, int(Ws/step),radius=int(Ws/2/step))
        #N_spindles = gaussian_filter1d(N_spindles, int(Ws/step),radius=int(Ws/2/step))

        P = sc.signal.savgol_filter(P, int(Ws/step),1)
        be = sc.signal.savgol_filter(be, int(Ws/step),1)
        fragmentation = sc.signal.savgol_filter(fragmentation, int(Ws/step),1)
        N_spindles = sc.signal.savgol_filter(N_spindles, int(Ws/step),1)

    res = np.zeros((4, len(P)))
    res[0]=np.array(P)
    res[1]=np.array(be)
    res[2]=np.array(fragmentation)
    res[3]=np.array(N_spindles)

    return res
'''


def get_label_0_data(D,delta_t):
    '''
    prior_t: in seconds, how much in advance form the start iof BS we take the signal
    delta_t: size of window in seconds
    '''
    # list to store the data
    L=[]

    for l in D:
        print(l[0])

        fs=128
        ti=l[-1]

        # get signal
        EEG_data=sc.io.loadmat('Data/' + l[0]) 
        y=EEG_data['record'][:,0]

        # clean signal
        try:
            y=clean_ground_check(y,fs,16,7.5,8.5,50000)
        except:
            print('no ground check detected')
            
        N_y=len(y)
        t=np.linspace(0,N_y/fs,N_y)

        #--- clean EEG
        # detection parameters
        list_detection=[2*fs,1*fs,[0.0004,0.002],"sym4",4,"periodization"]  
        # pos of artefacts
        index_mask=find_artifacts(y,*list_detection) 

        # WQN parameters
        list_WQN=["sym4","periodization",30,1]

        # clean artefacts
        y=WQN_3(y,index_mask,*list_WQN) 

        # get part of the signal of interest
        t_part, y_part = get_part(y,t,ti,fs)

        # get data
        Ws = int(90*fs)
        step = int(5*fs)
        L.append(get_data(y_part,t_part,fs,Ws,step,delta_t))

    return L


def get_label_1_data(D,prior_t,delta_t):
    '''
    prior_t: in seconds, how much in advance form the start iof BS we take the signal
    delta_t: size of window in seconds
    '''
    # list to store the data
    L=[]

    for key in D.keys():
        print(key)

        fs=D[key]['fs']
        ti=D[key]['before BS'] - prior_t

        # get signal
        if D[key]['method'] == 0:
            EEG_data=sc.io.loadmat('Recordings/' + key[:-4]) 
            y=EEG_data['record'][:,0]

        elif D[key]['method'] == 1:
            y = np.load('recordings_npy/' +key)

        elif D[key]['method'] == 2:
            EEG_data=sc.io.loadmat('BS/' + key) 
            y=EEG_data['record'][0,:]

        # clean signal
        try:
            y=clean_ground_check(y,fs,16,7.5,8.5,50000)
        except:
            print('no ground check detected')
            
        N_y=len(y)
        t=np.linspace(0,N_y/fs,N_y)

        #--- clean EEG
        # detection parameters
        list_detection=[2*fs,1*fs,[0.0004,0.002],"sym4",4,"periodization"]  
        # pos of artefacts
        index_mask=find_artifacts(y,*list_detection) 

        # WQN parameters
        list_WQN=["sym4","periodization",30,1]

        # clean artefacts
        y=WQN_3(y,index_mask,*list_WQN) 

        # get part of the signal of interest
        t_part, y_part = get_part(y,t,ti,fs)

        # get data
        Ws = int(90*fs)
        step = int(5*fs)
        L.append(get_data(y_part,t_part,fs,Ws,step,delta_t))

    return L


def scale_to_quantile_robust(x, target_value=75, quantile=1, iqr_thresh=1.5):
    x = np.asarray(x)

    # Step 1: Remove outliers using IQR
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    lower_bound = q1 - iqr_thresh * iqr
    upper_bound = q3 + iqr_thresh * iqr
    inliers = x[(x >= lower_bound) & (x <= upper_bound)]

    # Step 2: Compute quantile on inliers
    q_val = np.quantile(np.abs(inliers), quantile)
    if q_val == 0:
        raise ValueError("Quantile is zero after outlier removal. Cannot scale.")

    # Step 3: Scale full array so quantile of inliers maps to target
    scaling_factor = target_value / q_val
    x = x *  scaling_factor
    x = x - np.median(x)
    return x 


def get_graded_anesthesia_data(D,prior_t,delta_t):
    '''
    prior_t: in seconds, how much in advance form the start iof BS we take the signal
    delta_t: size of window in seconds
    '''
    # list to store the data
    L=[]


    for l in D:
        print(l[0])

        factor = 10
        fs = int(2500 / factor)
        ti = l[-1] - prior_t

        # get signal  
        y = np.load('C:/Users/holcman/Desktop/GradedAnesthesia/data/' + l[0]) 
        y = sc.signal.decimate(y, factor, ftype='fir', zero_phase=True)
        y = scale_to_quantile_robust(y)
        N_y=len(y)
        t=np.linspace(0,N_y/fs,N_y)

        # get part of the signal of interest
        t_part, y_part = get_part(y,t,ti,fs)

        # get data
        Ws = int(90*fs)
        step = int(5*fs)
        L.append(get_data_GAM(y_part,t_part,fs,Ws,step,delta_t))

    return L


def get_part(y,t,ti,fs):

    i=int(ti*fs) 
    y_part=y[:i]
    t_part=t[:i]

    return t_part,y_part

def get_data(y_part,t_part,fs,Ws,step,delta_t):

    alpha_part = filter_butterworth(y_part,fs,[7,14])
    delta_part = filter_butterworth(y_part,fs,[0.5,4])

    print('len_input_Data:', len(y_part))

    # get power of alpha wave
    t0 = time()
    P = sliding.power_1D(alpha_part,t_part,Ws,step)[-1]
    t1 = time()
    print(t1 - t0)
    print('len_variable:', len(P))

    # get power of delta band
    t0 = time()
    P_delta = sliding.power_1D(delta_part,t_part,Ws,step)[-1]
    t1 = time()
    print(t1 - t0)
    # get BE of alpha wave
    t0 = time()
    be = sliding.entropy_1D(alpha_part,t_part,Ws,step,k=2)[-1]
    t1 = time()
    print(t1 - t0)
    # get fragmentation of alpha wave
    T=0.2
    q=0.8
    N=int(fs/2)
    h=np.ones(N)/N
    fragmentation = sliding.get_fragmentation(alpha_part,t_part,Ws,step,h,q,T,mode='same')[-1]
    t1 = time()
    print(t1 - t0)
    # get spindles number
    upper_T = 3
    lower_T = 1.5
    proportion_T = 3/2
    frequency_T = 10.5
    #N_spindles = sliding.get_N_spindles_power(y_part, t_part, Ws, step, upper_T, lower_T)[-1]

    t0 = time()
    D = sliding.get_spindles_proportion(alpha_part, t_part, Ws, step, upper_T, lower_T, proportion_T, frequency_T)[-1]
    N_spindles = D['number of spindles']
    N_l_f_spindles = D['number low frequency spindles']
    N_m_f_spindles = D['number mixed frequency spindles']
    N_h_f_spindles = D['number high frequency spindles']
    prop_l_f_spindles = N_l_f_spindles / N_spindles
    prop_m_f_spindles = N_m_f_spindles / N_spindles
    prop_h_f_spindles = N_h_f_spindles / N_spindles
    t1 = time()
    print(t1 - t0)
    # get PSD of alpha band
    height = 0.5
    t0 = time()
    xb = sliding.PSD_barycenter(alpha_part,t_part,Ws,step,fs,height)[-1]
    t1 = time()
    print(t1 - t0)
    # get PSd of delta band
    t0 = time()
    xb_delta =  sliding.PSD_barycenter(delta_part,t_part,Ws,step,fs,height)[-1]
    t1 = time()
    print(t1 - t0)
    # smooth the variables time series
    N_smooth = int(Ws*2/step)

    print('len_smoothing filter: ', N_smooth)

    print(len(P), N_smooth)
    P = sc.signal.savgol_filter(P, N_smooth,1)
    be = sc.signal.savgol_filter(be, N_smooth,1)
    fragmentation = sc.signal.savgol_filter(fragmentation, N_smooth,1)
    N_spindles = sc.signal.savgol_filter(N_spindles, N_smooth,1)
    prop_l_f_spindles = sc.signal.savgol_filter(prop_l_f_spindles, N_smooth,1)
    prop_m_f_spindles = sc.signal.savgol_filter(prop_m_f_spindles, N_smooth,1)
    prop_h_f_spindles = sc.signal.savgol_filter(prop_h_f_spindles, N_smooth,1)
    xb = sc.signal.savgol_filter(xb, N_smooth,1)

    P_delta = sc.signal.savgol_filter(P_delta, N_smooth,1)
    xb_delta = sc.signal.savgol_filter(xb_delta, N_smooth,1)

    N = int(delta_t*fs/step)
    res = np.zeros((10, N))
    res[0]=np.array(P[-N:])
    res[1]=np.array(be[-N:])
    res[2]=np.array(fragmentation[-N:])
    res[3]=np.array(N_spindles[-N:])
    res[4]=np.array(prop_l_f_spindles[-N:])
    res[5]=np.array(prop_m_f_spindles[-N:])
    res[6]=np.array(prop_h_f_spindles[-N:])
    res[7]=np.array(xb[-N:])
    res[8]=np.array(P_delta[-N:])
    res[9]=np.array(xb_delta[-N:])

    return res


def get_data_GAM(y_part,t_part,fs,Ws,step,delta_t):

    alpha_part = filter_butterworth(y_part,fs,[6,20])
    delta_part = filter_butterworth(y_part,fs,[0.5,4])

    # get power of alpha wave

    P = sliding.power_1D(alpha_part,t_part,Ws,step)[-1]

    # get power of delta band
    P_delta = sliding.power_1D(delta_part,t_part,Ws,step)[-1]

    # get BE of alpha wave
    be = sliding.entropy_1D(alpha_part,t_part,Ws,step,k=2)[-1]

    # get fragmentation of alpha wave
    T=0.2
    q=0.8
    N=int(fs/2)
    h=np.ones(N)/N
    fragmentation = sliding.get_fragmentation(alpha_part,t_part,Ws,step,h,q,T,mode='same')[-1]

    # get spindles number
    upper_T = 3
    lower_T = 1.5
    proportion_T = 3/2
    frequency_T = 10.5
    #N_spindles = sliding.get_N_spindles_power(y_part, t_part, Ws, step, upper_T, lower_T)[-1]

    D = sliding.get_spindles_proportion(alpha_part, t_part, Ws, step, upper_T, lower_T, proportion_T, frequency_T)[-1]
    N_spindles = D['number of spindles']
    N_l_f_spindles = D['number low frequency spindles']
    N_m_f_spindles = D['number mixed frequency spindles']
    N_h_f_spindles = D['number high frequency spindles']
    prop_l_f_spindles = N_l_f_spindles / N_spindles
    prop_m_f_spindles = N_m_f_spindles / N_spindles
    prop_h_f_spindles = N_h_f_spindles / N_spindles

    # get PSD of alpha band
    height = 0.5
    xb = sliding.PSD_barycenter(alpha_part,t_part,Ws,step,fs,height)[-1]

    # get PSd of delta band
    xb_delta =  sliding.PSD_barycenter(delta_part,t_part,Ws,step,fs,height)[-1]

    # smooth the variables time series
    N_smooth = int(Ws*2/step)

    P = sc.signal.savgol_filter(P, N_smooth,1)
    be = sc.signal.savgol_filter(be, N_smooth,1)
    fragmentation = sc.signal.savgol_filter(fragmentation, N_smooth,1)
    N_spindles = sc.signal.savgol_filter(N_spindles, N_smooth,1)
    prop_l_f_spindles = sc.signal.savgol_filter(prop_l_f_spindles, N_smooth,1)
    prop_m_f_spindles = sc.signal.savgol_filter(prop_m_f_spindles, N_smooth,1)
    prop_h_f_spindles = sc.signal.savgol_filter(prop_h_f_spindles, N_smooth,1)
    xb = sc.signal.savgol_filter(xb, N_smooth,1)

    P_delta = sc.signal.savgol_filter(P_delta, N_smooth,1)
    xb_delta = sc.signal.savgol_filter(xb_delta, N_smooth,1)

    N = int(delta_t*fs/step)
    res = np.zeros((10, N))
    res[0]=np.array(P[-N:])
    res[1]=np.array(be[-N:])
    res[2]=np.array(fragmentation[-N:])
    res[3]=np.array(N_spindles[-N:])
    res[4]=np.array(prop_l_f_spindles[-N:])
    res[5]=np.array(prop_m_f_spindles[-N:])
    res[6]=np.array(prop_h_f_spindles[-N:])
    res[7]=np.array(xb[-N:])
    res[8]=np.array(P_delta[-N:])
    res[9]=np.array(xb_delta[-N:])

    return res