import numpy as np
from scipy.signal import hilbert
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d

def smooth(data, window_size):
    """Smooth data using a uniform filter."""
    return uniform_filter1d(data, size=window_size, mode='nearest')


def get_envelopes_power(xt, upper_T, lower_T):
    """
    Return the envelopes of the signal.

    Parameters:
    - t: 1D numpy array, time vector.
    - xt: 1D numpy array, signal.

    Returns:
    - x_up: Upper envelope of the signal.
    - x_low: Lower envelope of the signal.
    - spd_itv: Spindle intervals.
    - d: Difference between upper and lower envelopes.
    """
    h=np.ones(16)/16
    xt=np.convolve(xt**2,h,mode='same')
    xt=smooth(xt, 32)

    # Compute the standard deviation of the signal
    s = np.std(xt)

    peaks_max, _ = find_peaks(xt, height=upper_T * s)
    peaks_min, _ = find_peaks(-xt, height=-lower_T*s)


    idx_t = np.column_stack((peaks_min[:-1], peaks_min[1:]))
    spd_itv = []

    for start, end in idx_t:
        k = np.where((peaks_max > start) & (peaks_max < end))[0]
        if k.size > 0:
            spd_itv.append((start, end))

    return  np.array(spd_itv),xt,s


def get_envelopes_prominence(t, xt, upper_T, lower_T):
    """
    Return the envelopes of the signal.

    Parameters:
    - t: 1D numpy array, time vector.
    - xt: 1D numpy array, signal.

    Returns:
    - x_up: Upper envelope of the signal.
    - x_low: Lower envelope of the signal.
    - spd_itv: Spindle intervals.
    - d: Difference between upper and lower envelopes.
    """

    # Compute the standard deviation of the signal
    s = np.std(xt)
    span = 5

    # Upper and lower envelopes
    x_pos = xt * (xt > 0)
    x_neg = xt * (xt < 0)

    # Local maxima and minima
    TF_max = (xt[1:-1] > xt[:-2]) & (xt[1:-1] > xt[2:])  # Local maxima
    TF_min = (xt[1:-1] < xt[:-2]) & (xt[1:-1] < xt[2:])  # Local minima

    # Interpolation for upper and lower envelopes
    max_indices = np.where(TF_max)[0] + 1  # Adjust index
    min_indices = np.where(TF_min)[0] + 1  # Adjust index

    x_up = interp1d(t[max_indices], smooth(xt[max_indices], span), bounds_error=False, fill_value="extrapolate")(t)
    x_low = interp1d(t[min_indices], smooth(xt[min_indices], span), bounds_error=False, fill_value="extrapolate")(t)

    # Find spindles cut
    d = np.abs(x_up - x_low)
    peaks_max, _ = find_peaks(d, prominence=upper_T * s)
    peaks_min, _ = find_peaks(-d, height=-lower_T*s)


    idx_t = np.column_stack((peaks_min[:-1], peaks_min[1:]))
    spd_itv = []

    for start, end in idx_t:
        k = np.where((peaks_max > start) & (peaks_max < end))[0]
        if k.size > 0:
            spd_itv.append((start, end))

    return x_up, x_low, np.array(spd_itv), d, s


def get_envelopes(t, xt, upper_T, lower_T):
    """
    Return the envelopes of the signal.

    Parameters:
    - t: 1D numpy array, time vector.
    - xt: 1D numpy array, signal.

    Returns:
    - x_up: Upper envelope of the signal.
    - x_low: Lower envelope of the signal.
    - spd_itv: Spindle intervals.
    - d: Difference between upper and lower envelopes.
    """

    # Compute the standard deviation of the signal
    s = np.std(xt)
    span = 5

    # Upper and lower envelopes
    x_pos = xt * (xt > 0)
    x_neg = xt * (xt < 0)

    # Local maxima and minima
    TF_max = (xt[1:-1] > xt[:-2]) & (xt[1:-1] > xt[2:])  # Local maxima
    TF_min = (xt[1:-1] < xt[:-2]) & (xt[1:-1] < xt[2:])  # Local minima

    # Interpolation for upper and lower envelopes
    max_indices = np.where(TF_max)[0] + 1  # Adjust index
    min_indices = np.where(TF_min)[0] + 1  # Adjust index

    x_up = interp1d(t[max_indices], smooth(xt[max_indices], span), bounds_error=False, fill_value="extrapolate")(t)
    x_low = interp1d(t[min_indices], smooth(xt[min_indices], span), bounds_error=False, fill_value="extrapolate")(t)

    # Find spindles cut
    d = np.abs(x_up - x_low)
    peaks_max, _ = find_peaks(d, height=upper_T * s)
    peaks_min, _ = find_peaks(-d, height=-lower_T*s)


    idx_t = np.column_stack((peaks_min[:-1], peaks_min[1:]))
    spd_itv = []

    for start, end in idx_t:
        k = np.where((peaks_max > start) & (peaks_max < end))[0]
        if k.size > 0:
            spd_itv.append((start, end))

    return x_up, x_low, np.array(spd_itv), d, s


def instfreq(signal, t):
    """
    Calculate instantaneous frequency of a signal.

    Parameters:
    - signal: 1D numpy array, the input signal.
    - t: 1D numpy array, the time vector corresponding to the signal.

    Returns:
    - ifq: 1D numpy array, the instantaneous frequency.
    - t_ifq: 1D numpy array, the time vector for the instantaneous frequency.
    """
    # Compute the analytic signal using the Hilbert Transform
    analytic_signal = hilbert(signal)
    
    # Get the instantaneous phase
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    
    # Calculate the instantaneous frequency
    ifq = np.diff(instantaneous_phase) / (2.0 * np.pi)  # in cycles per sample
    ifq = np.concatenate(([0], ifq))  # add zero to match length of original signal
    
    # Convert from cycles per sample to Hz
    dt = np.mean(np.diff(t))  # assume uniform time intervals
    ifq /= dt

    # Create time vector for instantaneous frequency
    t_ifq = t  # can use original time vector

    return ifq, t_ifq
    

def get_spindle_type(xt, t, upper_T, lower_T,proportion_T,frequency_T):
    # Spindle annotation
    # 0 slow, 1 mixed, 2 fast
    Binary_seq = np.array([])
    FreqMax0 = np.array([])
    FreqMax1 = np.empty((2,1))
    FreqMax2 = np.array([])

    thres = proportion_T #3 / 2
    x_up, x_down, idx_t, d, s = get_envelopes(t, xt, upper_T, lower_T)

    for i in range(idx_t.shape[0]):
        region = slice(idx_t[i, 0], idx_t[i, 1] + 1)
        t_reg = t[region] - t[region][0]
        idx_max = np.argmax(x_up[region])
        mid = idx_t[i, 0] + idx_max

        if mid != idx_t[i, 1]:
            # Get instantaneous frequency
            ifq, t_ifq = instfreq(xt[region], t_reg)
            mask_slow = ifq < frequency_T #10
            mask_fast = ifq > frequency_T #10

            ratio = np.sum(mask_slow) / np.sum(mask_fast)
            if ratio > thres:
                Binary_seq = np.append(Binary_seq,0)  # slow
                FreqMax0 = np.append(FreqMax0,np.mean(ifq[mask_slow])) 
            elif ratio < 1 / thres:
                Binary_seq= np.append(Binary_seq,2)  # fast
                FreqMax2 = np.append(FreqMax2,np.mean(ifq[mask_fast]))
            else:
                Binary_seq= np.append(Binary_seq,1)  # mixed
                FreqMax1 = np.concatenate((FreqMax1,np.transpose([[np.mean(ifq[mask_slow]), np.mean(ifq[mask_fast])]])),axis=1) 
                
    return Binary_seq, FreqMax0, FreqMax1[:,1:], FreqMax2, idx_t, d, s


def get_spindle_type_prominence(xt, t, upper_T, lower_T,proportion_T,frequency_T):
    # Spindle annotation
    # 0 slow, 1 mixed, 2 fast
    Binary_seq = np.array([])
    FreqMax0 = np.array([])
    FreqMax1 = np.empty((2,1))
    FreqMax2 = np.array([])

    thres = proportion_T #3 / 2
    x_up, x_down, idx_t, d, s = get_envelopes_prominence(t, xt, upper_T, lower_T)

    for i in range(idx_t.shape[0]):
        region = slice(idx_t[i, 0], idx_t[i, 1] + 1)
        t_reg = t[region] - t[region][0]
        idx_max = np.argmax(x_up[region])
        mid = idx_t[i, 0] + idx_max

        if mid != idx_t[i, 1]:
            # Get instantaneous frequency
            ifq, t_ifq = instfreq(xt[region], t_reg)
            mask_slow = ifq < frequency_T #10
            mask_fast = ifq > frequency_T #10

            ratio = np.sum(mask_slow) / np.sum(mask_fast)
            if ratio > thres:
                Binary_seq = np.append(Binary_seq,0)  # slow
                FreqMax0 = np.append(FreqMax0,np.mean(ifq[mask_slow])) 
            elif ratio < 1 / thres:
                Binary_seq= np.append(Binary_seq,2)  # fast
                FreqMax2 = np.append(FreqMax2,np.mean(ifq[mask_fast]))
            else:
                Binary_seq= np.append(Binary_seq,1)  # mixed
                FreqMax1 = np.concatenate((FreqMax1,np.transpose([[np.mean(ifq[mask_slow]), np.mean(ifq[mask_fast])]])),axis=1) 
                
    return Binary_seq, FreqMax0, FreqMax1[:,1:], FreqMax2, idx_t, d, s


def get_spindle_type_power(xt, t, upper_T, lower_T, proportion_T, frequency_T):
    # Spindle annotation
    # 0 slow, 1 mixed, 2 fast
    Binary_seq = np.array([])
    FreqMax0 = np.array([])
    FreqMax1 = np.empty((2,1))
    FreqMax2 = np.array([])

    thres = proportion_T #3 / 2
    pos_spindle,xt2,s= get_envelopes_power(xt, upper_T, lower_T)

    for i in range(pos_spindle.shape[0]):
        region = slice(pos_spindle[i, 0], pos_spindle[i, 1] + 1)
        t_reg = t[region] - t[region][0]
        idx_max = np.argmax(xt[region])
        mid = pos_spindle[i, 0] + idx_max

        if mid != pos_spindle[i, 1]:
            # Get instantaneous frequency
            ifq, t_ifq = instfreq(xt[region], t_reg)
            mask_slow = ifq < frequency_T #10
            mask_fast = ifq > frequency_T #10

            ratio = np.sum(mask_slow) / np.sum(mask_fast)
            if ratio > thres:
                Binary_seq = np.append(Binary_seq,0)  # slow
                FreqMax0 = np.append(FreqMax0,np.mean(ifq[mask_slow])) 
            elif ratio < 1 / thres:
                Binary_seq= np.append(Binary_seq,2)  # fast
                FreqMax2 = np.append(FreqMax2,np.mean(ifq[mask_fast]))
            else:
                Binary_seq= np.append(Binary_seq,1)  # mixed
                FreqMax1 = np.concatenate((FreqMax1,np.transpose([[np.mean(ifq[mask_slow]), np.mean(ifq[mask_fast])]])),axis=1) 
                
    return Binary_seq, FreqMax0, FreqMax1[:,1:], FreqMax2, pos_spindle, xt2, s


def spindle_off_proportion(y, upper_T, lower_T, method='power'):

    if method=='power':
        pos_spindles, xt, s = get_envelopes_power(y, upper_T, lower_T)
    
    N=len(y)
    n_spindles=len(pos_spindles)
    mask=np.zeros(N)
    for pos in pos_spindles:
        mask[pos[0]:pos[-1]] = 1

    prop_off = 1 - np.sum(mask)/N

    return prop_off, n_spindles, s
    

