import numpy as np
from math import sqrt

class Module:
    def sgd_step(): pass
    

class Sequential():
    def __init__(self, modules, loss):
        self.modules = modules
        self.loss = loss
        
    def forward(self, data):
        for layer in self.modules:
            data = layer.forward(data)
        return data
    
    def backward(self, delta):
        for module in self.modules[::-1]:
            delta = module.backward(delta)
    
    def sgd_step(self, lrate):
        for m in self.modules:
            m.sgd_step(lrate)

        

class Linear(Module):
    def __init__(self, in_layers, out_layers):
        self.m = in_layers
        self.n = out_layers
        stdv = 1.0 / sqrt(self.m)
        self.W = np.random.uniform(-stdv, stdv, (self.n, self.m)) # W: (n x m)
        self.W0 = np.random.uniform(-stdv, stdv, (self.n, 1)) # W0: (n x 1)
        self.sgd_defined = True
    
    def forward(self, A): # A is (m x b)
        self.A = A
        # (n x m) . (m x b) + (n x 1) = (n x b)
        return np.dot(self.W, self.A) + self.W0 # return value is (n x b)
    
    def __repr__(self):
        return f'W: {self.W}, W0: {self.W0}, A: {self.A}'
    
    def backward(self, dLdZ): # dLdZ: (n x b)
        self.dLdW = np.dot(dLdZ, self.A.T)  # dLdW: (n x m)
        self.dLdW0 = np.sum(dLdZ, axis=1) # dLdW0: (n x 1)
        return np.dot(self.W.T, dLdZ) # dLd(A_prev): (m x b)
    
    def sgd_step(self, lr):
        self.W = self.W - lr * self.dLdW
        self.W0 = self.W0 - lr * self.dLdW0


class ReLU(Module):

    def __init__(self):
        self.sgd_defined = False

    def forward(self, A):
        self.A = A
        return np.where(A < 0, 0, A)
    
    def backward(self, dLdZ):
        return dLdZ * np.where(self.A < 0, 0, 1)

class Tanh(Module):

    def forward(self, A):
        self.th = np.tanh(A)
        self.sgd_defined = False
        return self.th
    
    def backward(self, dLdZ):
        return dLdZ * (1 - np.square(self.th))

class MSELoss():

    def forward(self, y, ypred):
        self.y = y
        self.ypred = ypred
        return np.mean(np.square(y - ypred), axis=0)
    
    def backward(self):
        return - 2 * (self.y - self.ypred)
