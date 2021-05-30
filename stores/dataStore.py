# -*- coding: utf-8 -*-
"""
Created on Sat May 29 01:48:52 2021

@author: rashe
"""
import numpy as np
import pandas as pd

from utils.ranking import Get_Mahalanobis

class dataStore():
    
    
    @classmethod
    def buildBatches(cls, train_x, train_y, numbatches, datatype):
    
        if datatype == 'image':
            x_train_new = []
            for i in range(0,len(train_x)):
                x_train_new.append(train_x[i].reshape(1,3072)[0])
                print(i)
            x_t = np.array(x_train_new)
            x_t = pd.DataFrame(x_t)
            x_scaled = preprocessing.scale(x_t)
            Mahas = pd.DataFrame(Get_Mahalanobis(pd.DataFrame(x_scaled))[0]).sort_values(0, ascending=True)
            k = int(len(Mahas)/numbatches)
            
            batches = dict()
            outs = dict()
            for i in range(0, numbatches):
                #indices = Mahas[int(k*i) : int(k*(i+1))].index
                indices = Mahas[int(k*(i)): int(k*(i+1))].index
                batches[i] = train_x[Mahas.loc[indices].index]
                batches[i] = batches[i].astype('float32')
                batches[i] /= 255
                outs[i] = train_y[Mahas.loc[indices].index]
            
            return batches, outs
        
        Mahas, features = Get_Mahalanobis(pd.DataFrame(train_x))
        del features        
        Mahas = pd.DataFrame(Mahas).sort_values(0)
                
        train_x, train_y = pd.DataFrame(train_x).reset_index(drop=True), pd.DataFrame(train_y).reset_index(drop=True)
                
        k = int(len(Mahas)/numbatches)
        
        batches = dict()
        outs = dict()
        for i in range(0, numbatches):
            #indices = Mahas[int(k*i) : int(k*(i+1))].index
            indices = Mahas[int(k*(i)): int(k*(i+1))].index
            batches[i] = train_x.loc[Mahas.loc[indices].index]
            outs[i] = train_y.loc[Mahas.loc[indices].index]
            
        return(batches, outs)
        