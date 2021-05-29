# -*- coding: utf-8 -*-
"""
Created on Sat May 29 01:41:52 2021

@author: rashe
"""
import pandas as pd

def processStates(weights, biases)
    thetas = pd.DataFrame()
    for z in range(0,len(weights)):
        ones = np.arange(0.01,1.01,1/weights[z].shape[1])
        for q in range(0,len(weights[z])):                
            dp = np.dot(ones, np.array(weights[z][q])) ##REMEMBER TO TRANSPOSE
            dp = dp/np.linalg.norm(ones)
            dp = dp/np.linalg.norm(weights[z][q])
            dp = np.arccos(dp)
            thetas = pd.concat([thetas, pd.DataFrame([dp])],axis=0)
            norm_a = np.linalg.norm(weights[z][q])                         
            thetas = pd.concat([thetas, pd.DataFrame([norm_a])], axis=0)
            thetas = pd.concat([thetas, pd.DataFrame([biases[0][0][0]])], axis=0)
                                         
            thetas = thetas.reset_index(drop=True)
            
    return(thetas)