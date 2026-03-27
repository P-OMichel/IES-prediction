import numpy as np



def entropy_signals(signals,interval):
    '''
    signals: nD numpy.array
    interval: integer of how continuous values are associated together
              ex: interval=2 means [0,2[-->0, [2,4[-->2 etc
                  interval=1 gives the integer part
    '''

    res=np.apply_along_axis(entropy,1,signals,interval)

    return res

def entropy(y,interval):
    '''
    y: 1D numpy array
    interval: integer of how continuous values are associated together
              ex: interval=2 means [0,2[-->0, [2,4[-->2 etc
                  interval=1 gives the integer part
    '''
    N=y.size
    # transforms the signal accorded to the desired interval
    y=((y/interval).astype(int))*interval

    # get the values present in y and the number of time they appear
    values,counts=np.unique(y,return_counts=True)

    # changes counts into probabilities
    probabilities=counts/N

    # compute the Shannon entropy
    Shannon_entropy=-np.sum(probabilities*np.log2(probabilities))

    return Shannon_entropy

'''
efficiency is the entropy divided by the maximum one
'''
def efficiency(y,interval):
    '''
    y: 1D numpy array
    interval: integer of how continuous values are associated together
              ex: interval=2 means [0,2[-->0, [2,4[-->2 etc
                  interval=1 gives the integer part
    '''
    N=y.size
    # transforms the signal accorded to the desired interval
    y=((y/interval).astype(int))*interval

    # get the values present in y and the number of time they appear
    values,counts=np.unique(y,return_counts=True)

    # changes counts into probabilities
    probabilities=counts/N

    # compute the Shannon entropy
    Shannon_entropy=-np.sum(probabilities*np.log2(probabilities))/np.log2(N)

    return Shannon_entropy






################################################  Efficiency computed with the histogram ######################################################

def hist_efficiency_signals(signals,Nbins):
    '''
    signals: nD numpy.array
    interval: integer of how continuous values are associated together
              ex: interval=2 means [0,2[-->0, [2,4[-->2 etc
                  interval=1 gives the integer part
    '''

    res=np.apply_along_axis(hist_efficiency,1,signals,Nbins)

    return res

def hist_efficiency(y,Nbins):
    '''
    y: 1D numpy array 
    Nbins: number of bins in the histogram
    quantile: upper quantile to stop the range of the histogram
    '''

    hist=np.histogram(y,Nbins)

    efficiency=-np.sum(hist[0]*np.log2(hist[0]+0.0000000001))/np.log(Nbins)

    return efficiency






