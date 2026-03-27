'''
File to test the spindles frequency proportion over time.
'''
import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from Functions.filter import filter_FFT
from Spindles.spindles_segmentation import detect_spindles
from Spindles.hilbert_frequency_maximum import get_frequency_max
from Functions.utils import diff_envelops,get_PSD_barycenter,barycenter
import Functions.sliding_fct as sliding

# rec_20240321_085300 Big Suppressions
# rec_20240118_094517  persistent alpha at the end
# load recording
EEG_data=sc.io.loadmat('Recordings/rec_20240118_094517') 
fs=128
start, end = 0, -1
t=EEG_data['time'][start*fs:end*fs,0]
y=EEG_data['record'][start*fs:end*fs,0]

# height parameter for peaks detection in the PSD
height=0.6

# compute spectrogram
delta_f=0.5
delta_t=1
nfft=int(fs/delta_f)
overlap=int(nfft-delta_t*fs)
f_spectro,t_spectro,spectro=sc.signal.spectrogram(y,fs,nperseg=nfft,noverlap=overlap)
eps=0.00000001

alpha_y=filter_FFT(y,fs,[7,14])


cmap='rainbow'

# display spectrogram
fig,axes=plt.subplots(2,sharex=True)
axes[0].pcolormesh(t_spectro,f_spectro,np.log(spectro + eps),shading='nearest',cmap=cmap,vmin=np.log(eps),vmax=np.log(1000+eps)) 
axes[1].pcolormesh(t_spectro,f_spectro,np.log(spectro + eps),shading='nearest',cmap=cmap,vmin=np.log(eps),vmax=np.log(1000+eps))
Ws, step=30*fs, 5*fs
t_list, barycenter_list=sliding.PSD_barycenter(alpha_y,t,Ws,step,fs,height)
axes[1].plot(t_list,barycenter_list,label='Ws='+str(Ws/fs)+' | step='+str(step/fs),color='black')
Ws, step=60*fs, 5*fs
t_list, barycenter_list=sliding.PSD_barycenter(alpha_y,t,Ws,step,fs,height)
axes[1].plot(t_list,barycenter_list,label='Ws='+str(Ws/fs)+' | step='+str(step/fs),color='blue')
axes[1].legend()

plt.show()

'''
start, end = 350, 380
t=EEG_data['time'][start*fs:end*fs,0]
y=EEG_data['record'][start*fs:end*fs,0]

# filter into alpha band
alpha_y=filter_FFT(y,fs,[6,16])

f, Pxx_den, x_b=get_PSD_barycenter(alpha_y,fs,height)

mean_Pxx=np.zeros(len(Pxx_den))
delta_f=f[-1]-f[-2]
mean_Pxx[int(6/delta_f):int(16/delta_f)]=np.convolve(Pxx_den[int(6/delta_f):int(16/delta_f)],np.ones(50)/50,mode='same')
mean_Pxx=mean_Pxx/np.max(mean_Pxx)
print(barycenter(f,mean_Pxx))

fig,axis=plt.subplots(1,sharex=True)
axis.plot(f, Pxx_den) 
axis.plot(f,mean_Pxx)
axis.set_title(x_b)



start, end = 1090, 1120
t=EEG_data['time'][start*fs:end*fs,0]
y=EEG_data['record'][start*fs:end*fs,0]

# filter into alpha band
alpha_y=filter_FFT(y,fs,[6,16])

f, Pxx_den, x_b=get_PSD_barycenter(alpha_y,fs,height)

mean_Pxx=np.zeros(len(Pxx_den))
delta_f=f[-1]-f[-2]
mean_Pxx[int(6/delta_f):int(16/delta_f)]=np.convolve(Pxx_den[int(6/delta_f):int(16/delta_f)],np.ones(50)/50,mode='same')
mean_Pxx=mean_Pxx/np.max(mean_Pxx)
print(barycenter(f,mean_Pxx))

fig,axis=plt.subplots(1,sharex=True)
axis.plot(f, Pxx_den) 
axis.plot(f,mean_Pxx)
axis.set_title(x_b)





start, end = 2120, 2150
t=EEG_data['time'][start*fs:end*fs,0]
y=EEG_data['record'][start*fs:end*fs,0]

# filter into alpha band
alpha_y=filter_FFT(y,fs,[6,16])

f, Pxx_den, x_b=get_PSD_barycenter(alpha_y,fs,height)

mean_Pxx=np.zeros(len(Pxx_den))
delta_f=f[-1]-f[-2]
mean_Pxx[int(6/delta_f):int(16/delta_f)]=np.convolve(Pxx_den[int(6/delta_f):int(16/delta_f)],np.ones(50)/50,mode='same')
mean_Pxx=mean_Pxx/np.max(mean_Pxx)
print(barycenter(f,mean_Pxx))

fig,axis=plt.subplots(1,sharex=True)
axis.plot(f, Pxx_den) 
axis.plot(f,mean_Pxx)
axis.set_title(x_b)

plt.show()
'''




'''
# Hilbert transform
def get_instantaneous_frequency(y):
    analytic_signal=sc.signal.hilbert(y)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * fs)
    return instantaneous_frequency


# Figure
colors=['black','red']
fig,axes=plt.subplots(6)
axes[0].plot(t,alpha_y)
c=0
k=1
for pos in pos_spindles:
    axes[0].plot(t[pos[0]:pos[-1]],alpha_y[pos[0]:pos[-1]],color=colors[c%2])
    c+=1
for i in [0,5,10,15,20]:
    axes[k].plot(get_instantaneous_frequency(alpha_y[pos_spindles[i][0]:pos_spindles[i][-1]]))
    k+=1

fig,axes=plt.subplots(5)
k=0
for i in [0,5,10,15,20]:
    fft_y=sc.fft.rfft(alpha_y[pos_spindles[i][0]:pos_spindles[i][-1]])  
    delta_f=(fs/2)/len(fft_y)  
    axes[k].plot(np.linspace(0,len(fft_y)*delta_f,len(fft_y)),np.abs(fft_y)/np.max(np.abs(fft_y)))
    k+=1

fig,axes=plt.subplots(5)
k=0
for i in [0,5,10,15,20]:
    f, Pxx_den = sc.signal.periodogram(alpha_y[pos_spindles[i][0]:pos_spindles[i][-1]], fs)
    print(f[-1]-f[-2])
    axes[k].plot(f, Pxx_den/np.max(Pxx_den))              
    k+=1
'''












'''
# fct to get distribution of spindle maximum frequency
def max_freq_spindle(signal,lower_std_criteria,upper_std_criteria):
    
    list_spindles=detect_spindles(signal,lower_std_criteria,upper_std_criteria)  # pos of stat, peak, end of a spindle
    pos_spindles=[[x[0],x[-1]] for x in list_spindles]                            # pos of start and end of a spindle

    max_amp_list, freq_max_list = [], []
    for x in pos_spindles:
        max_amp, freq_max = get_frequency_max(alpha_y[x[0]:x[1]],fs)
        max_amp_list.append(max_amp)
        freq_max_list.append(freq_max)

    return max_amp_list,freq_max_list
'''



'''
# rec_20240307_115040
#alpha_y[1416*fs:1615*fs],
signals=[alpha_y[3780*fs:3880*fs],
         
         alpha_y[3500*fs:3600*fs],
         alpha_y[1100*fs:1300*fs],
         alpha_y[3600*fs:3700*fs],
         alpha_y[2900*fs:3100*fs],
         alpha_y[2075*fs:2225*fs]
         ]
N_signals=len(signals)
titles=['Awake','OK exiting BS','OK entering Shallow','OK entering Awake','Too Heavy','Too Heavy'] #'Shallow',



fig0,axes=plt.subplots(N_signals,2)
for i in range(N_signals):
    print(i)
    max_amp_list,freq_max_list=max_freq_spindle(signals[i],lower_std_criteria,upper_std_criteria)

    axes[i,0].hist(max_amp_list,bins=20)
    axes[i,1].hist(freq_max_list,bins=20)
    axes[i,0].set_title('Amplitude | ' + titles[i])
    axes[i,1].set_title('Max Freq | ' + titles[i])

plt.show()


'''
'''
# get spindles
lower_std_criteria, upper_std_criteria = 1, 3

list_spindles=detect_spindles(alpha_y,lower_std_criteria,upper_std_criteria)  # pos of stat, peak, end of a spindle

pos_spindles=[[x[0],x[-1]] for x in list_spindles]                            # pos of start and end of a spindle

# get max amp and corresponding frequencies for each spindles

max_amp_list, freq_max_list = [], []
for x in pos_spindles:
    max_amp, freq_max = get_frequency_max(alpha_y[x[0]:x[1]],fs)
    max_amp_list.append(max_amp)
    freq_max_list.append(freq_max)

# display spindles and spectrogram
fig,axes=plt.subplots(2,sharex=True)
axes[0].plot(t,alpha_y)
axes[0].set_ylim(-35,35)
c=0
color_list=['black','red']
for x in pos_spindles:
    axes[0].plot(t[x[0]:x[1]],alpha_y[x[0]:x[1]],color=color_list[c%2])
    c+=1
axes[1].pcolormesh(t_spectro,f_spectro,np.log(spectro + eps),shading='nearest',cmap='jet',vmin=np.log(eps),vmax=np.log(2000+eps))


#for i in range(len(pos_spindles)):
    #print(i,pos_spindles[i])




#-- Heavy state 

start, end = 450*fs, 817*fs

i, j = 81, 121

fig1,axes=plt.subplots(2)
axes[0].hist(max_amp_list[81:122],bins=20)
axes[1].hist(freq_max_list[81:122],bins=20)


#-- Heavy state  with new computation of spindles

start, end = 450*fs, 817*fs

alpha_heavy=alpha_y[start:end]

list_spindles=detect_spindles(alpha_heavy,lower_std_criteria,upper_std_criteria)  # pos of stat, peak, end of a spindle

pos_spindles=[[x[0],x[-1]] for x in list_spindles]                            # pos of start and end of a spindle

# get max amp and corresponding frequencies for each spindles

max_amp_list, freq_max_list = [], []
for x in pos_spindles:
    max_amp, freq_max = get_frequency_max(alpha_heavy[x[0]:x[1]],fs)
    max_amp_list.append(max_amp)
    freq_max_list.append(freq_max)



fig1,axes=plt.subplots(3)
axes[0].plot(t[start:end]-start/fs,alpha_heavy)
c=0
for x in pos_spindles:
    axes[0].plot(t[x[0]:x[1]],alpha_heavy[x[0]:x[1]],color=color_list[c%2])
    c+=1
axes[1].hist(max_amp_list,bins=20)
axes[2].hist(freq_max_list,bins=20)



# study part of the recording

#-- Heavy state  with new computation of spindles

start, end = 2100*fs, 2219*fs

alpha_part=alpha_y[start:end]

list_spindles=detect_spindles(alpha_part,lower_std_criteria,upper_std_criteria)  # pos of stat, peak, end of a spindle

pos_spindles=[[x[0],x[-1]] for x in list_spindles]                            # pos of start and end of a spindle

# get max amp and corresponding frequencies for each spindles

max_amp_list, freq_max_list = [], []
for x in pos_spindles:
    max_amp, freq_max = get_frequency_max(alpha_part[x[0]:x[1]],fs)
    max_amp_list.append(max_amp)
    freq_max_list.append(freq_max)



fig1,axes=plt.subplots(3)
axes[0].plot(t[start:end]-start/fs,alpha_part)
c=0
for x in pos_spindles:
    axes[0].plot(t[x[0]:x[1]],alpha_part[x[0]:x[1]],color=color_list[c%2])
    c+=1
axes[1].hist(max_amp_list,bins=20)
axes[2].hist(freq_max_list,bins=20)

plt.show()
'''




