'''
Testing spindles detection
'''
import numpy as np
import matplotlib.pyplot as plt
from Spindles.spindles_segmentation import detect_spindles_power
from Functions.utils import diff_envelops
import scipy as sc
from Functions.filter import filter


# criteria on thresholds
lower_std_criteria,upper_std_criteria=0.5,1

# test with real data
#data loading
#EEG_data=sc.io.loadmat('Recordings/rec_20240229_090001.mat')
#time_list=EEG_data['time'][15000:18000,0]
#eeg_signal=EEG_data['record'][15000:18000,0] 
EEG_data=sc.io.loadmat('Recordings/rec_20240307_115040.mat')
fs=128
time_list=EEG_data['time'][2616*fs:2780*fs,0]#[550*fs:650*fs,0] #same[1000*fs:1100*fs,0]#good example of missing case [2000*fs:2100*fs,0]#[2616*fs:2780*fs,0]
eeg_signal=EEG_data['record'][2616*fs:2780*fs,0]#[550*fs:650*fs,0] #[1000*fs:1100*fs,0]#[2000*fs:2100*fs,0]#[2616*fs:2780*fs,0] 

# filter in alpha band
f_int=[7,14]
y=filter(eeg_signal,128,f_int,'FFT')

# detect spindles position (method 1)
N=64
pos_spindles=detect_spindles_power(y,lower_std_criteria,upper_std_criteria,N)

# compute the difference envelope for plot visualization
d=np.convolve(y**2,np.ones(N)/N,mode='same')

# compute standard deviation of data
std=np.std(d)

fig,(ax0,ax1,ax2)=plt.subplots(3,sharex=True)
ax0.plot(y,linewidth=1)
ax0.set_title('EEG recording in the '+str(f_int)+' band')
ax1.plot(d)
ax1.set_title(' d '+str(f_int)+' band')
ax1.plot([upper_std_criteria*std for i in y],label='upper threshold: '+str(upper_std_criteria)+' std')
ax1.plot([lower_std_criteria*std for i in y],label='lower threshold: '+str(lower_std_criteria)+' std')
ax1.legend()

# plot consecutive spindles in different color for better visualization
c=0
for i in pos_spindles:
    if c%2==0:
        current_spindle=y[i[0]:i[-1]]
        ax1.plot(i,d[i],'o')
        ax2.plot(np.linspace(i[0],i[-1],len(current_spindle)),current_spindle,color='red')
        c+=1
    else:
        current_spindle=y[i[0]:i[-1]]
        ax1.plot(i,d[i],'o')
        ax2.plot(np.linspace(i[0],i[-1],len(current_spindle)),current_spindle,color='black')
        c+=1
ax2.set_title('Spindles detected')

plt.show()