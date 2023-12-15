# -*- coding: utf-8 -*-
"""
@author: Blopa
"""

import numpy as np
import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, layers, activation, output_activation ,**kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(1,len(layers)):
            self.layers.append( nn.Linear(layers[i-1], layers[i]) )
            if i+1 < len(layers):
                if activation in 'st':
                    nn.init.xavier_uniform_(self.layers[-1].weight)
                else:
                    nn.init.kaiming_uniform_(self.layers[-1].weight)
                self.layers.append( self.choose_activation(activation, **kwargs) )
        if output_activation: self.layers.append( self.choose_activation(output_activation, **kwargs) )
        
    def choose_activation(self, activation, **kwargs):
        if activation=='s':
            return nn.Sigmoid()
        elif activation=='t':
            return nn.Tanh()
        elif activation=='r':
            return nn.ReLU()
        elif activation=='l':
            return nn.LeakyReLU( kwargs.get('negative_slope', 0.01) )
        
    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class NeuralNetworkTrainer():
    def __init__(self, modelCls, optCls, lossF, error_threshold, max_iters, modelKwargs={}, optKwargs={}):
        self.nn = modelCls(**modelKwargs)
        self.opt = optCls(self.nn.parameters(), **optKwargs)
        self.loss = lossF()
        self.error_threshold, self.max_iters = error_threshold, max_iters
        
    def fit(self, X, y):
        it = 0
        last_error = 0
        last_delta = self.error_threshold + 1
        self.nn.train()
        self.nn.to(device)
        X = torch.tensor(X, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.float32).view([-1,1]).to(device)
        while (not self.max_iters or it < self.max_iters) and (not self.error_threshold or last_delta > self.error_threshold):
            self.opt.zero_grad()
            y_pred = self.nn(X)
            loss = self.loss(y_pred, y)
            loss.backward()
            self.opt.step()			
            it += 1
            last_delta = abs(loss.item() - last_error)
            last_error = loss.item()
            # print(it,last_error)
        torch.cuda.empty_cache()
        
    def predict(self, X):
        self.nn.to('cpu')
        self.nn.eval()
        return np.array(self.nn(torch.tensor(X, dtype=torch.float32)) >= 0.5, dtype=np.int64).flatten()
