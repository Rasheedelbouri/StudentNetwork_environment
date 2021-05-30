# -*- coding: utf-8 -*-
"""
Created on Wed May 26 00:39:08 2021

@author: rasheed el-bouri
"""
import keras
import numpy as np
import pandas as pd

from models.buildNetworkScript import buildNetwork
from stores.dataStore import dataStore
from stores.loadData import getDataSet
from utils.referenceProduct import processStates
from utils.errors import shapeError
from continuousAgent import continuousActionSelection


class networkEnv(buildNetwork, dataStore):
    """
    Class designed to create a gym-like environment for treating a neural network
    as a reinforcement learning environment. Upon instantiation, a network is created,
    as well as a data store (containing batches of data organised according to a
    curriculum) for the training set being used. The training set (for
    the neural network to perform classification on) must be loaded in first.
    
    
    Key methods are:
        
        reset: allows you to reset the environment and start training the neural network
        from the initial seed
        
        step: takes as input either an integer or a tuple of two values (for 
        continuous action selection).The integer action selects the corresponding
        batch from the data store.
    
    
    """
    def __init__(self, network_type, train_x, train_y, val_x, val_y, numbatches,\
                 layers=2, nodes=50, seed=0):
        
        super().__init__()
        
        if network_type not in ('feedforward', 'convolutional'):
            raise NameError('must specify between feedforward or convolutional networks')
        
        if not isinstance(train_x, (pd.DataFrame, np.ndarray)):
            raise TypeError('input data must be either pd.DataFrame or np.array')
        
        if not isinstance(train_y, (pd.DataFrame, np.ndarray)):
            raise TypeError('label data must be either pd.DataFrame or np.array')

        if not isinstance(numbatches, int) or numbatches < 1:
            raise ValueError('number of batches cannot be float or negative')
        
        if not isinstance(layers, int) or layers < 1:
            raise ValueError('number of hidden layers cannot be float or negative')

        if not isinstance(nodes, int) or nodes < 1:
            raise ValueError('number of hidden nodes cannot be float or negative')  
        
        if not isinstance(seed, int) or nodes < 0:
            raise ValueError('seed must be integer and greater than 0')
        
        
        self.network_type = network_type
        self.input_sz = train_x.shape[1]
        self.output_sz = train_y.shape[1]
        if self.output_sz == 2:
            self.output_sz = 1
        self.numbatches = numbatches
        self.hidden_layers = layers
        self.hidden_nodes = nodes
        self.seed = seed
        
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        
        self.densePositions = None
        
        self.make()
        
        self.batches, self.labels = self.buildBatches(self.train_x, self.train_y, self.numbatches, datatype='table')
        
    
    def make(self):
        self.tr_rewards = []
        self.va_rewards = []
        self.rewards = []
        self.initial = 0
        if self.network_type == 'feedforward':
            self.network = self.build(np.zeros(self.input_sz), np.zeros(self.output_sz), conv=False, generative=False, continual=False)
            self.network = self.compiler(self.network, q_net=False, actor=False)
        elif self.network_type == 'convolutional':
            self.network = self.build(np.zeros(self.input_sz), np.zeros(self.output_sz), conv=True, generative=False, continual=False)
            self.network = self.compiler(self.network, q_net=False, actor=False)
        else:
            raise NotImplementedError('specify from feedforward or convolutional')
        
    
    def reset(self):
        """
        Re-initialises your environment, including the rewards recorded.

        Returns
        -------
        None.

        """
        self.make()
            
    
    def stateRepresentation(self):
        """
        Transforms the weights and biases of the neural network into the format
        used for representing the state in the reinforcement learing setup.

        Returns
        -------
        state : TYPE
            pd.DataFrame.

        """
        weights, biases = self._state
        state = processStates(weights, biases)
        
        return state
                    
    
    def step(self, action):
        """
        
        Parameters
        ----------
        action : int or tuple
            This is the batch or collection of batches to train the student network
            on.

        Raises
        ------
        TypeError
            action must be of type int or tuple.
        ValueError
            action must not be greater than the number of batches available to
            select from.
        shapeError
            if input is of type tuple, tuple must be of length 2.

        Returns
        -------
        state: pd.DataFrame
            Returns representation of the new state of the network.
        action : int or tuple
            Returns the action that was taken.
        reward: float
            Returns the reward associated with taking this action from the previous
            state.

        """
        if not isinstance(action, (int, tuple)):
            raise TypeError('action must be an integer or a tuple')
        
      
        if isinstance(action, int):
            if action > self.numbatches:
                raise ValueError("number of actions can't be greater than number of curriculum batches")
    
            self.network.train_on_batch(self.batches[action], self.labels[action])
            return (self.stateRepresentation(), action, self.calculateReward())
        
        elif isinstance(action, tuple):
            if len(action) != 2:
                raise shapeError('length of tuple must be 2. If step length == 1, change to int type')
            self.continuousActionSelection(action)
            return (self.stateRepresentation(), action, self.calculateReward())
        
    
    def calculateReward(self):
        """
        Method that calculates the reward based on the product of the improvement
        of accuracy on the training and validation sets. If both are negative then 
        we multiply again by -1.

        Returns
        -------
        reward : float
            Returns reward based on gradient of improvement on training and valdiation sets.

        """
        if self.initial == 0:
            self.tr_rewards.append(self.network.test_on_batch(self.train_x, self.train_y)[1])
            self.va_rewards.append(self.network.test_on_batch(self.val_x, self.val_y)[1])
            self.initial = 1
        else:
            self.tr_rewards.append(self.network.test_on_batch(self.train_x, self.train_y)[1] - self.tr_rewards[-1])
            self.va_rewards.append(self.network.test_on_batch(self.val_x, self.val_y)[1] - self.va_rewards[-1])
    
        if self.tr_rewards[-1] < 0 and self.va_rewards[-1] < 0:
            reward = -1*(self.tr_rewards[-1]*self.va_rewards[-1])
        else:
            reward = (self.tr_rewards[-1]*self.va_rewards[-1])
        self.rewards.append(self.tr_rewards[-1] * self.va_rewards[-1])
        return reward

    
    def continuousActionSelection(self, action):
        """
        Method that allows for the selection of a batch from a continuous action space.

        Parameters
        ----------
        action : int or tuple
            DESCRIPTION.

        Returns
        -------
        None.

        """

        actions,batchsize = action[0],action[1]
        
        if self.network_type == 'convolutional':
             if actions < batchsize/2:
                batchin = np.array([])
                batchout = np.array([])
                for p in range(0,batchsize):
                    batchin = np.vstack([batchin, self.batches[p]]) if batchin.size else self.batches[p]
                    batchout = np.vstack([batchout, self.labels[p]]) if batchout.size else self.labels[p]
                self.network.train_on_batch(batchin, batchout)        
            
             elif actions >=batchsize/2 and actions<= len(self.batches)-batchsize/2:
                batchin = np.array([])
                batchout = np.array([])
                if batchsize/2 == 0:
                    self.network.train_on_batch(self.batches[actions], self.labels[actions])
                else:
                    for p in range(int(actions-batchsize/2),int(actions + batchsize/2)):
                        batchin = np.vstack([batchin, self.batches[p]]) if batchin.size else self.batches[p]
                        batchout = np.vstack([batchout, self.labels[p]]) if batchout.size else self.labels[p]                        
                    self.network.train_on_batch(batchin, batchout)        
             else:
                batchin = np.array([])
                batchout = np.array([])
                if batchsize/2 == 0:
                    self.network.train_on_batch(self.batches[actions], self.labels[actions])
                else:
                    for p in range(len(self.batches)-batchsize,len(self.batches)):
                        batchin = np.vstack([batchin, self.batches[p]]) if batchin.size else self.batches[p]
                        batchout = np.vstack([batchout, self.labels[p]]) if batchout.size else self.labels[p]
                    self.network.train_on_batch(batchin, batchout)
             
        
        if actions < batchsize/2:
            batchin = pd.DataFrame()
            batchout = pd.DataFrame()
            for p in range(0,batchsize):
                batchin = pd.concat([batchin, self.batches[p]],axis=0)
                batchout = pd.concat([batchout, self.labels[p]], axis=0)
            self.network.train_on_batch(batchin, batchout)                    
                        
        elif actions >=batchsize/2 and actions <= len(self.batches)-batchsize/2:
            batchin = pd.DataFrame()
            batchout = pd.DataFrame()
            if batchsize/2 == 0:
                self.network.train_on_batch(self.batches[actions], self.labels[actions])
            else:
                for p in range(int(actions-batchsize/2),int(actions + batchsize/2)):
                    batchin = pd.concat([batchin, self.batches[p]],axis=0)
                    batchout = pd.concat([batchout, self.labels[p]], axis=0)                        
                self.network.train_on_batch(batchin, batchout)                                        
        else:
            batchin = pd.DataFrame()
            batchout = pd.DataFrame()
            if batchsize/2 == 0:
                self.network.train_on_batch(self.batches[actions], self.labels[actions])
            else:
                for p in range(len(self.batches)-batchsize,len(self.batches)):
                    batchin = pd.concat([batchin, self.batches[p]],axis=0)
                    batchout = pd.concat([batchout, self.labels[p]], axis=0)
                self.network.train_on_batch(batchin, batchout)
                            

    @property
    def _state(self):
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
    def _actions(self):
        print(self.numbatches)
    
    @property
    def _rewards(self):
        print(self.rewards)

    
    @property
    def _summary(self):
        self.network.summary()
        
    

