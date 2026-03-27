'''
File to compute the sensitivity score
'''
import numpy as np

class GetSensitivity:

    def __init__(self):
        super().__init__()


    #--- Dictionnary where the key corresponds to which variables are used in the logistic regression
    #    and it has the values of the learned coefficient for the associated variables. 
    #    order: [a_coeff, t_alpha_coeff, t_IES_coeff] | rmk: on Chris code it is [a_coef,t_IES_coeff,t_alpha_coeff]
        self.D={}
        # first checkpoint
        self.D['t_a']=np.array([0,0.4331,0]) #001
        self.D['t_IES']=np.array([0,0,2.8736]) #010
        # second checkpoint
        self.D['a']=np.array([1.4630,0,0]) #100
        self.D['a,t_a']=np.array([1.3361,0.1056,0]) #101
        # second and third checkpoint
        self.D['a,t_a,t_IES']=np.array([0.6626,-0.0918,2.4486])#111

    def fit_sigmoid(self,):
           
