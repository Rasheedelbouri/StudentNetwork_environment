# -*- coding: utf-8 -*-
"""
Created on Wed May 26 00:39:08 2021

@author: rashe
"""
import keras
import numpy as np

from buildNetworkScript import buildNetwork
from loadData import getDataSet
from referenceProduct import processStates

train_x, train_y, val_x, val_y, test_x, test_y, uniques = getDataSet('digits')


class networkEnv(buildNetwork):
    
    def __init__(self, network_type, train_x, train_y, layers=2, nodes=50, seed=0):
        
        super().__init__()
        self.network_type = network_type
        self.input_sz = train_x.shape[1]
        self.output_sz = train_y.shape[1]
        if self.output_sz == 2:
            self.output_sz = 1
        
        self.hidden_layers = layers
        self.hidden_nodes = nodes
        self.seed = seed
        
        self.densePositions = None
        
        self.make()
        
        
    
    def make(self):
        if self.network_type == 'feedforward':
            self.network = self.build(np.zeros(self.input_sz), np.zeros(self.output_sz), conv=False, generative=False, continual=False)
            self.network = self.compiler(self.network, q_net=False, actor=False)
        elif self.network_type == 'convolutional':
            self.network = self.build(np.zeros(self.input_sz), np.zeros(self.output_sz), conv=True, generative=False, continual=False)
            self.network = self.compiler(self.network, q_net=False, actor=False)
        else:
            raise NotImplementedError('specify from feedforward or convolutional')
        
    
    def reset(self):
        self.make()
        
    
    
    
    
    
    
    
    def stateRepresentation(self):
        weights, biases = self.state()
        state = processStates(weights, biases)
        
        return state
                    
    
    def step()
    
    
    
    
    
    @property
    def state(self):
        weights = []
        biases = []
        if self.densePositions is None:
            self.densePositions = []
            for i,layer in enumerate(self.network.layers):
                if isinstance(self.network.layers[i], keras.layers.core.Dense):
                    self.densePositions.append(i)
                    weights.append(layer.get_weights()[0])
                    biases.append(layer.get_weights()[1])
        else:
            for i in self.densePositions:
                weights.append(self.network.layers[i].get_weights()[0])
                biases.append(self.network.layers[i].get_weights()[1])
        
        return weights, biases
    
    
    
    @property
    def _summary(self):
        self.network.summary()
    
    

