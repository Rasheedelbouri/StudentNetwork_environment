# -*- coding: utf-8 -*-
"""
Created on Sat May 29 19:22:47 2021

@author: rashe
"""

import pandas as pd
import numpy as np

def Get_Mahalanobis(dataframe):
    
    dataframe = dataframe.reset_index(drop=True)
    nunique = dataframe.apply(pd.Series.nunique)
    if dataframe.shape[1] >= 15:
        cols_to_drop = nunique[nunique <= 2].index
        dataframe = dataframe.drop(cols_to_drop, axis=1)

    features = list(dataframe)
    means = pd.DataFrame(np.zeros(len(features)))
    covariance = np.cov(dataframe.T)
    inv_cov = np.linalg.inv(covariance)
    Mahalanobis = np.zeros(len(dataframe))
    
    

    for j in range(0,len(means)):
            means[0][j] = np.mean(dataframe.iloc[:,j])
            
    means = means.reset_index(drop=True)
    
    for i in range(0,len(dataframe)):
        first = pd.DataFrame(dataframe.iloc[i,:]).reset_index(drop=True)    
        
        V = first[i]-means[0]
        Mahalanobis[i] = np.sqrt(np.dot(np.dot(V.T,inv_cov), V))#[0][0]
        
        
    return(Mahalanobis, features)