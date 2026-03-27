'''
Function to get the distribution of spindles frequency
'''

import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from Functions.filter import filter_FFT
from Spindles.spindles_f_distribution import get_spindles_barycenter
import Functions.sliding_fct as sliding


# load data
EEG_data=sc.io.loadmat('Recordings/rec_20240411_090403')
fs=128
i=int(1430*fs)
j=int(1460*fs)
t=EEG_data['time'][:,0]   
y=EEG_data['record'][:,0] 

# compute spectrogram
delta_f=0.5
delta_t=1
nfft=int(fs/delta_f)
overlap=int(nfft-delta_t*fs)
f_spectro,t_spectro,spectro=sc.signal.spectrogram(y,fs,nperseg=nfft,noverlap=overlap)

# extract distribution
lower_criteria,upper_criteria=1, 3
T=0.5
N_bins=10
T_dist=0.5
#'''
# get alpha wave
alpha_y=filter_FFT(y[i:j],fs,[7,14])

histo,bins_mid_values,x_b=get_spindles_barycenter(alpha_y,fs,lower_criteria,upper_criteria,T,N_bins,T_dist)
print(x_b)
print(histo)


fig,axes=plt.subplots(2)
axes[0].pcolormesh(t_spectro,f_spectro,np.log(spectro+0.00000001),shading='nearest',cmap='jet',vmin=np.log(0.00000001),vmax=np.log(2000))
axes[1].bar(bins_mid_values,histo/np.max(histo))
#'''

# sliding spindle PSD barycenter
alpha_y=filter_FFT(y,fs,[7,14])

Ws=30*fs
step=10*fs
t_list,barycenter_list=sliding.spindles_PSD_barycenter(alpha_y,t,Ws,step,fs,lower_criteria,upper_criteria,T,N_bins,T_dist)

fig,axes=plt.subplots(2,sharex=True)
axes[0].pcolormesh(t_spectro,f_spectro,np.log(spectro+0.00000001),shading='nearest',cmap='jet',vmin=np.log(0.00000001),vmax=np.log(2000))
axes[1].pcolormesh(t_spectro,f_spectro,np.log(spectro+0.00000001),shading='nearest',cmap='jet',vmin=np.log(0.00000001),vmax=np.log(2000))
axes[1].plot(t_list,barycenter_list,label='Ws='+str(Ws/fs)+' | step='+str(step/fs),color='black')
plt.show()



