'''
File to test the detection of On/Off areas
'''
from Functions.utils import *
from Functions.compute_fragmentation import *
from Functions.filter import get_filtered_signal
from Spindles.detect_spindles_env import get_spindles_signals,get_spindles_2_thresholds_signals
from Spindles.detect_on_off_areas import detect_off_areas
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import time

#----------------------------- Parameters--------------------------- #
fs=128

# for filter
list_freq_int=[[2,3],[8,12],[15,30],[30,50]]

# for spindles

# 2 thresholds method
lower_T=[1,1,1,1]
upper_T=[3,3,3,3]

# distance method
T_spindles=[2.,2.,2.,2.]

# for On/Off detection
N_neigbours=fs
h_delta=np.ones(int(N_neigbours*1.5)+1)
h_alpha=np.ones(N_neigbours+1)
h_beta=np.ones(int(N_neigbours/2)+1)
h_gamma=np.ones(int(N_neigbours/4)+1)
h=[h_delta,h_alpha,h_beta,h_gamma]
T_on_off=[0.2,0.15,0.15,0.15]
quantile=[0.98,0.98,0.98,0.98]


#-----------------------------load data ---------------------------#
'''
EEG_data=sc.io.loadmat('Recordings/rec_20240321_085300.mat')
start,end=160000,230000
t=EEG_data['time'][start:end,0]
y=EEG_data['record'][start:end,0] 
'''
EEG_data=sc.io.loadmat('Recordings/rec_20240411_090403')
i=int(1300*fs)
j=int(1750*fs)
t=EEG_data['time'][i:j,0]
y=EEG_data['record'][i:j,0] 

#--------------------------- Detection ----------------------------#
t_start=time.time()
#--- filter data
filtered_signals=get_filtered_signal(y,fs,list_freq_int)
# delta to abs(delta)
#filtered_signals[0,:]=np.abs(filtered_signals[0,:])
#filtered_signals[0,:]=np.convolve(filtered_signals[0,:],np.ones(32),'same')
t1=time.time()
#--- envelopes
env_signals=diff_envelops_signals(filtered_signals)
t2=time.time()
#--- detect spindles (2 thresholds method)
spindles_2_T=get_spindles_2_thresholds_signals(filtered_signals,env_signals,lower_T,upper_T)
t3=time.time()
#--- detect spindles (distance method)
spindles_distance=get_spindles_signals(filtered_signals,env_signals,T_spindles)
t4=time.time()
#--- detect On/Off areas
t5=time.time()
pos_on_off,pos_IES=detect_off_areas(env_signals,h,quantile,T_on_off,mode='same')
t_end=time.time()
print('time to filter: ',t1-t_start)
print('time to make env differences: ',t2-t1)
print('time to detect spindles (with 2 thresholds method): ',t3-t2)
print('time to detect spindles (distance method): ',t4-t3)
print('time to detect On/off areas: ',t5-t4)
print('time to filter, make env differences, detect spindles with 2 methods and on off areas: ', t_end-t_start)


#------------------------------- Plot -----------------------------#

D={}
D['0']='red'
D['1']='black'

'''
fig0,axes=plt.subplots(1+4*len(list_freq_int),sharex=True)
axes[0].plot(t,y)
axes[0].set_title('EEG recording (blue) with IES (red)')
for pos in pos_IES:
    axes[0].plot(t[pos[0]:pos[1]],y[pos[0]:pos[1]],color='red')

c=0
for i in range(len(list_freq_int)):

    signal=filtered_signals[i,:]
    env_signal=env_signals[i,:]
    pos_spindles_2_T=spindles_2_T[i]
    pos_spindles_distance=spindles_distance[i]
    on_off_pos=pos_on_off[i]

    axes[i+1+c].plot(t,signal)
    axes[i+1+c].set_title(str(list_freq_int[i])+' On/Off areas detection')
    c+=1

    axes[i+1+c].plot(t,signal)
    a=0
    for pos in pos_spindles_2_T:
        axes[i+1+c].plot(t[pos[0]:pos[1]],signal[pos[0]:pos[1]],color=D[str(a%2)])
        a+=1
    axes[i+1+c].set_title(str(list_freq_int[i])+' Spindles detection (2 thresholds method)')
    c+=1

    axes[i+1+c].plot(t,signal)
    a=0
    for j in range(0,len(pos_spindles_distance)-1,2):
        axes[i+1+c].plot(t[pos_spindles_distance[j]:pos_spindles_distance[j+1]],signal[pos_spindles_distance[j]:pos_spindles_distance[j+1]],color=D[str(a%2)])
        a+=1
    axes[i+1+c].set_title(str(list_freq_int[i])+' Spindles detection (distance method)')
    c+=1

    
    axes[i+1+c].plot(t,env_signal)
    for pos in on_off_pos:
        axes[i+1+c-3].plot(t[pos[0]:pos[-1]],signal[pos[0]:pos[-1]],color='red')
        axes[i+1+c].plot(t[pos[0]:pos[-1]],env_signal[pos[0]:pos[-1]],color='red')
    axes[i+1+c].set_title(str(list_freq_int[i])+' difference envelope')
'''


'''
#--- delta and alpha
fig1,axes1=plt.subplots(1+4*len(list_freq_int[:2]),sharex=True)
axes1[0].plot(t,y)
axes1[0].set_title('EEG recording')

c=0
for i in range(len(list_freq_int[:2])):

    signal=filtered_signals[i,:]
    env_signal=env_signals[i,:]
    pos_spindles_2_T=spindles_2_T[i]
    pos_spindles_distance=spindles_distance[i]
    on_off_pos=pos_on_off[i]

    axes1[i+1+c].plot(t,signal)
    axes1[i+1+c].set_title(str(list_freq_int[i])+' On/Off areas detection')
    c+=1

    axes1[i+1+c].plot(t,signal)
    a=0
    for pos in pos_spindles_2_T:
        axes1[i+1+c].plot(t[pos[0]:pos[1]],signal[pos[0]:pos[1]],color=D[str(a%2)])
        a+=1
    axes1[i+1+c].set_title(str(list_freq_int[i])+' Spindles detection (2 thresholds method)')
    c+=1

    axes1[i+1+c].plot(t,signal)
    a=0
    for j in range(0,len(pos_spindles_distance)-1,2):
        axes1[i+1+c].plot(t[pos_spindles_distance[j]:pos_spindles_distance[j+1]],signal[pos_spindles_distance[j]:pos_spindles_distance[j+1]],color=D[str(a%2)])
        a+=1
    axes1[i+1+c].set_title(str(list_freq_int[i])+' Spindles detection (distance method)')
    c+=1
    
    axes1[i+1+c].plot(t,env_signal)
    for pos in on_off_pos:
        axes1[i+1+c-3].plot(t[pos[0]:pos[-1]],signal[pos[0]:pos[-1]],color='red')
        axes1[i+1+c].plot(t[pos[0]:pos[-1]],env_signal[pos[0]:pos[-1]],color='red')
    axes1[i+1+c].set_title(str(list_freq_int[i])+' difference envelope')
'''

'''
#--- beta and gamma
fig2,axes2=plt.subplots(1+4*len(list_freq_int[2:]),sharex=True)
axes2[0].plot(t,y)
axes2[0].set_title('EEG recording')

c=0
for i in range(len(list_freq_int[2:])):

    signal=filtered_signals[i+2,:]
    env_signal=env_signals[i+2,:]
    pos_spindles_2_T=spindles_2_T[i+2]
    pos_spindles_distance=spindles_distance[i+2]
    on_off_pos=pos_on_off[i+2]

    axes2[i+1+c].plot(t,signal)
    axes2[i+1+c].set_title(str(list_freq_int[i])+' On/Off areas detection')
    c+=1

    axes2[i+1+c].plot(t,signal)
    a=0
    for pos in pos_spindles_2_T:
        axes2[i+1+c].plot(t[pos[0]:pos[1]],signal[pos[0]:pos[1]],color=D[str(a%2)])
        a+=1
    axes2[i+1+c].set_title(str(list_freq_int[i])+' Spindles detection (2 thresholds method)')
    c+=1

    axes2[i+1+c].plot(t,signal)
    a=0
    for j in range(0,len(pos_spindles_distance)-1,2):
        axes2[i+1+c].plot(t[pos_spindles_distance[j]:pos_spindles_distance[j+1]],signal[pos_spindles_distance[j]:pos_spindles_distance[j+1]],color=D[str(a%2)])
        a+=1
    axes2[i+1+c].set_title(str(list_freq_int[i])+' Spindles detection (distance method)')
    c+=1
    
    axes2[i+1+c].plot(t,env_signal)
    for pos in on_off_pos:
        axes2[i+1+c-3].plot(t[pos[0]:pos[-1]],signal[pos[0]:pos[-1]],color='red')
        axes2[i+1+c].plot(t[pos[0]:pos[-1]],env_signal[pos[0]:pos[-1]],color='red')
    axes2[i+1+c].set_title(str(list_freq_int[i])+' difference envelope')
'''

#--- fragmentation on all
fig3,axes3=plt.subplots(1+2*len(list_freq_int),1,sharex=True)
axes3[0].plot(t,y)
axes3[0].set_title('EEG recording (blue) with IES (red)')
for pos in pos_IES:
    axes3[0].plot(t[pos[0]:pos[1]],y[pos[0]:pos[1]],color='red')

c=0
for i in range(len(list_freq_int)):
    signal=filtered_signals[i,:]
    env_signal=env_signals[i,:]
    on_off_pos=pos_on_off[i]

    axes3[i+1+c].plot(t,signal)
    axes3[i+1+c].set_title(str(list_freq_int[i])+' signal with On/Off areas detection')
    c+=1

    axes3[i+1+c].plot(t,env_signal)
    for pos in on_off_pos:
        axes3[i+1+c-1].plot(t[pos[0]:pos[-1]],signal[pos[0]:pos[-1]],color='red')
        axes3[i+1+c].plot(t[pos[0]:pos[-1]],env_signal[pos[0]:pos[-1]],color='red')
    axes3[i+1+c].plot(t,np.ones(len(env_signal))*T_on_off[i]*np.quantile(env_signal,0.98))
    axes3[i+1+c].set_title(str(list_freq_int[i])+' difference envelope with On/Off areas detection')

plt.show()



