import numpy as  np

def lorentzian(x,A,mu,sigma):

    return (A/np.pi)*sigma/((x-mu)**2+sigma**2)

def log_lorentzian(x,A,mu,sigma):

    return 

def curb(x,A,B):

    return A*(1-B*x)

def curb_final(x,A,c,a):
    
    return A/(c+x**a)