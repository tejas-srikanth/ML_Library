import numpy as np
from math import sqrt

class Module:
    def sgd_step(self, lr): pass
    

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
    def __init__(self, in_layers, out_layers, zero_init=False):
        self.m = in_layers
        self.n = out_layers
        stdv = 1.0 / sqrt(self.m)
        if not zero_init:
            self.W = np.random.uniform(-stdv, stdv, (self.n, self.m)) # W: (n x m)
            self.W0 = np.random.uniform(-stdv, stdv, (self.n, 1)) # W0: (n x 1)
        else:
            self.W = np.zeros((self.n, self.m)) # W: (n x m)
            self.W0 = np.ones((self.n, 1)) # W0: (n x 1)

        self.sgd_defined = True
    
    def forward(self, A): # A is (m x b)
        self.A = A
        # (n x m) . (m x b) + (n x 1) = (n x b)
        return np.dot(self.W, self.A) + self.W0 # return value is (n x b)
        return np.dot(self.W, self.A) + self.W0 # return value is (n x b)

    
    def __repr__(self):
        return f'W: {self.W}, W0: {self.W0}, A: {self.A}'
    
    def backward(self, dLdZ): # dLdZ: (n x b)
        self.dLdW = np.dot(dLdZ, self.A.T)  # dLdW: (n x m)
        self.dLdW0 = np.sum(dLdZ, axis=1).reshape(-1,1) # dLdW0: (n x 1)
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
        self.y = y.reshape(1,-1)
        self.ypred = ypred
        return np.mean(np.square(y - ypred), axis=1)
    
    def backward(self):
        return - 2 * (self.y - self.ypred)


class LinearRegression():

    def gram_schmidt(self, X):
        num_cols = X.shape[1]
        X[:, 0] = (1 / np.linalg.norm(X[:, 0])) * X[:, 0] 
        for i in range(1, num_cols):
            proj = np.zeros_like(X[:, 0])
            for j in range(i):
                X_ij = np.dot(X[:, i], X[:, j])
                X_j1 = np.dot(X[:, j], X[:, j])
                proj += (X_ij/X_j1) * X[:, j]
            perp = X[:, i] - proj
            perp = (1 / np.linalg.norm(perp)) * perp 
            X[:, i] = perp
        return X


    def fit(self, X, y):
        # A = self.gram_schmidt(X)
        A = np.copy(np.array(X))
        # A = self.gram_schmidt(A)
        inv_AtA = np.linalg.inv(np.dot(A.T, A))
        self.x_hat = np.dot(np.dot(inv_AtA, A.T), y)
        return self.x_hat
    
    def predict(self, X):
        return np.dot(X, self.x_hat)
