'''
Testing spindles detection
'''
import numpy as np
import matplotlib.pyplot as plt
from Spindles.detect_spindles_2_thresholds import*
from Spindles.detect_spindles_env import find_spindles_2_thresholds
import scipy as sc
from Functions.filter import filter


# criteria on thresholds
lower_std_criteria,upper_std_criteria=1,3

# test with real data
#data loading
EEG_data=sc.io.loadmat('Recordings/rec_20240229_090001.mat')
time_list=EEG_data['time'][15000:18000,0]
eeg_signal=EEG_data['record'][15000:18000,0] 

# filter in alpha band
f_int=[8,12]
y=filter(eeg_signal,128,f_int,'FFT')

# compute standard deviation of data
std=np.std(y)


# detect spindles position (method 1)
pos_spindles=detect_spindles(y,lower_std_criteria,upper_std_criteria)

# compute the difference envelope for plot visualization
d=diff_envelops(y)

fig,(ax0,ax1,ax2,ax3)=plt.subplots(4,sharex=True)
ax0.plot(y,linewidth=1)
ax0.set_title('EEG recording in the '+str(f_int)+' band')
ax1.plot(d)
ax1.set_title(' difference envelope of the signal in the '+str(f_int)+' band')
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



# method 2
spindles=find_spindles_2_thresholds(d,std,lower_std_criteria,upper_std_criteria)
c=0
for i in spindles:
    if c%2==0:
        current_spindle=y[i[0]:i[-1]]
        ax3.plot(np.linspace(i[0],i[-1],len(current_spindle)),current_spindle,color='red')
        c+=1
    else:
        current_spindle=y[i[0]:i[-1]]
        ax3.plot(np.linspace(i[0],i[-1],len(current_spindle)),current_spindle,color='black')
        c+=1
ax3.set_title('Spindles detected')

plt.show()


