#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 16:54:21 2021

@author: kebl4170
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10
import keras




def getDataSet(source):

    if source == 'digits':
        data_in, data_out = pd.DataFrame(load_digits()['data']), pd.DataFrame(load_digits()['target'])
    elif source == 'cifar':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.9, random_state=1)
        uniques = pd.DataFrame(pd.DataFrame(y_train)[0].unique())
        y_train = keras.utils.to_categorical(y_train, 10)
        val_y = keras.utils.to_categorical(y_val, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        
        x_train = x_train[0:12000]
        train_y = y_train[0:12000]
        
        x_test = x_test[0:3000]
        test_y = y_test[0:3000]


        train_x = x_train
        
        test_x = x_test.astype('float32')
        test_x /= 255
        val_x = x_val.astype('float32')
        val_x  /= 255

    else:
        pamd = PAMAP2_data(source)
        train_x, train_y, val_x, val_y, test_x, test_y, uniques = pamd.getTrainingData()  
    
    if source == 'digits':
        sc = StandardScaler()
        sc.fit(data_in)
        data_in = pd.DataFrame(sc.transform(data_in))
        
        
        uniques = pd.DataFrame(pd.DataFrame(data_out)[0].unique())
        lb = preprocessing.LabelBinarizer()
        lb.fit(list(uniques[0]))
        
        
        data_out = pd.DataFrame(lb.transform(data_out))
        
        train_x, test_x, train_y, test_y = train_test_split(data_in, data_out, train_size=0.6)
        val_x, test_x, val_y, test_y = train_test_split(test_x, test_y, train_size=0.5)
        val_y = np.array(val_y)
        test_y = np.array(test_y)

    return train_x, train_y, val_x, val_y, test_x, test_y, uniques
    
    
def formatData(train_x, train_y, numbatches, num_nodes, source, transfer, augment, corrupt_node): 
    if transfer == True:
        batches, outs = getCurriculumBatches(train_x, train_y, numbatches, source)
    else:
        batches = dict()
        outs = dict()
        for i in range(num_nodes):
            batches[i], outs[i] = getCurriculumBatches(train_x[i], train_y[i], numbatches, source)
            if augment == True:
                for k in range(0,len(batches[i])):
                    batches[i][k] = pd.concat([batches[i][k]]*int((((len(train_x[i][k])*numbatches))/numbatches)/len(batches[i][k])))
                    outs[i][k] = pd.concat([outs[i][k]]*int((((len(outs[i][k])*numbatches))/numbatches)/len(outs[i][k])))
        if corrupt_node:
            for i in range(len(batches[0])):
                batches[0][i] *= 0
        
    return batches, outs

