# -*- coding: utf-8 -*-
"""
Created on Sat May 29 23:48:26 2021

@author: rashe
"""
import numpy as np
import pandas as pd

def continuousActionSelection(student, action, batches, outs, datatype):

    actions,batchsize = action[0],action[1]
    
    if datatype == 'convolutional':
         if actions < batchsize/2:
            batchin = np.array([])
            batchout = np.array([])
            for p in range(0,batchsize):
                batchin = np.vstack([batchin, batches[p]]) if batchin.size else batches[p]
                batchout = np.vstack([batchout, outs[p]]) if batchout.size else outs[p]
            student.train_on_batch(batchin, batchout)        
        
         elif actions >=batchsize/2 and actions<= len(batches)-batchsize/2:
            batchin = np.array([])
            batchout = np.array([])
            if batchsize/2 == 0:
                student.train_on_batch(batches[actions], outs[actions])
            else:
                for p in range(int(actions-batchsize/2),int(actions + batchsize/2)):
                    batchin = np.vstack([batchin, batches[p]]) if batchin.size else batches[p]
                    batchout = np.vstack([batchout, outs[p]]) if batchout.size else outs[p]                        
                student.train_on_batch(batchin, batchout)        
         else:
            batchin = np.array([])
            batchout = np.array([])
            if batchsize/2 == 0:
                student.train_on_batch(batches[actions], outs[actions])
            else:
                for p in range(len(batches)-batchsize,len(batches)):
                    batchin = np.vstack([batchin, batches[p]]) if batchin.size else batches[p]
                    batchout = np.vstack([batchout, outs[p]]) if batchout.size else outs[p]
                student.train_on_batch(batchin, batchout)
         
         return student
    
    if actions < batchsize/2:
        batchin = pd.DataFrame()
        batchout = pd.DataFrame()
        for p in range(0,batchsize):
            batchin = pd.concat([batchin, batches[p]],axis=0)
            batchout = pd.concat([batchout, outs[p]], axis=0)
        student.train_on_batch(batchin, batchout)                    
                    
    elif actions >=batchsize/2 and actions <= len(batches)-batchsize/2:
        batchin = pd.DataFrame()
        batchout = pd.DataFrame()
        if batchsize/2 == 0:
            student.train_on_batch(batches[actions], outs[actions])
        else:
            for p in range(int(actions-batchsize/2),int(actions + batchsize/2)):
                batchin = pd.concat([batchin, batches[p]],axis=0)
                batchout = pd.concat([batchout, outs[p]], axis=0)                        
            student.train_on_batch(batchin, batchout)                                        
    else:
        batchin = pd.DataFrame()
        batchout = pd.DataFrame()
        if batchsize/2 == 0:
            student.train_on_batch(batches[actions], outs[actions])
        else:
            for p in range(len(batches)-batchsize,len(batches)):
                batchin = pd.concat([batchin, batches[p]],axis=0)
                batchout = pd.concat([batchout, outs[p]], axis=0)
            student.train_on_batch(batchin, batchout)
            
    return student
            