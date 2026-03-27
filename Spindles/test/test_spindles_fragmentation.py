'''
File to test evaluation of the alpha-suppression with the spindles gaps
'''

import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from Functions.filter import filter_FFT
from Spindles.spindles_segmentation import detect_spindles

# rec_20240321_085300 Big Suppressions
# rec_20240118_094517  persistent alpha at the end
# rec_20240307_115040
# load recording
EEG_data=sc.io.loadmat('Recordings/rec_20240118_094517') 
t=EEG_data['time'][:,0]
y=EEG_data['record'][:,0]
fs=128

# compute spectrogram
delta_f=0.25
delta_t=1
nfft=int(fs/delta_f)
overlap=int(nfft-delta_t*fs)
f_spectro,t_spectro,spectro=sc.signal.spectrogram(y,fs,nperseg=nfft,noverlap=overlap)
eps=0.00000001

# filter into alpha band
alpha_y=filter_FFT(y,fs,[8,12])
lower_std_criteria, upper_std_criteria = 2, 3

def spindles_fragmentation(signal,lower_std_criteria, upper_std_criteria):
    list_spindles=detect_spindles(signal,lower_std_criteria,upper_std_criteria)  # pos of stat, peak, end of a spindle
    pos_spindles=[[x[0],x[-1]] for x in list_spindles]                            # pos of start and end of a spindle

    return pos_spindles


# sliding window parameters
Ws=30*fs
step=30*fs

windows=[alpha_y[i:i+Ws] for i in range(0,len(alpha_y)-Ws,step)] 
t_list=[t[Ws+i*step] for i in range(len(windows))] 

fig,axes=plt.subplots(4,sharex=True)
axes[0].plot(t,y)
axes[0].set_ylim(-75,75)
axes[1].plot(t,alpha_y)
axes[1].set_ylim(-25,25)
axes[2].pcolormesh(t_spectro,f_spectro,np.log(spectro+eps),shading='nearest',cmap='jet',vmax=np.log(2000),vmin=np.log(eps))

# spindles fragmentation
list_frag=[]
colors=['black','red']
i=0
c=0
for win in windows:
    pos_spindles=spindles_fragmentation(win,lower_std_criteria,upper_std_criteria)
    list_frag.append(Ws/fs-np.sum(np.array([pos[1]-pos[0] for pos in pos_spindles])/fs))
    for pos in pos_spindles:
        axes[1].plot(t[i*step+pos[0]:i*step+pos[1]],alpha_y[i*step+pos[0]:i*step+pos[1]],color=colors[c%2])
        c+=1
    i+=1
axes[3].plot(t_list,list_frag)
plt.show()

