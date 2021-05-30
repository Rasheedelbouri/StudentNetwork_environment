# -*- coding: utf-8 -*-
"""
Created on Sun May 30 01:43:33 2021

@author: rashe
"""
import numpy as np

from stores.loadData import getDataSet
from environmentBuilder import networkEnv

def main():
    train_x, train_y, val_x, val_y, test_x, test_y, uniques = getDataSet('digits')

    env = networkEnv('feedforward', train_x, train_y, val_x, val_y, numbatches = 50)
    
    states ={}
    for j in range(100):
        act = (np.random.randint(env.numbatches), np.random.randint(env.numbatches))
        states[j] = env.step(act)

if __name__ == "__main__":
    main()