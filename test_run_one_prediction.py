import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from Functions.detect_artifacts import find_artifacts
from Functions.WaveletQuantileNormalization import WQN_3
from Prediction.extract_data import get_data, get_part
from time import time

data = sc.io.loadmat('Data/rec_20230629_092502.mat')
y = data['record'][:,0]
fs = 128

delta_t = 240 + 10

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
y = WQN_3(y,index_mask,*list_WQN) 

# get part of the signal of interest
ti = 500
t_part, y_part = get_part(y,t,ti,fs)

# get data
Ws = int(90*fs)
step = int(5*fs)
t0 = time()
X = get_data(y_part,t_part,fs,Ws,step,delta_t)
t1 = time()
print(t1 -t0)

t0 = time()
X = get_data(y_part,t_part,fs,Ws,step,delta_t)
t1 = time()
print(t1 -t0)
print(X)
