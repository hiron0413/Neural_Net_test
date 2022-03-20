import numpy as np

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
    
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx

class IdentityWithLoss:
    def __init__(self):
        self.loss = None
        self.x = None
        self.t = None
        
    def forward(self, x, t):
        self.t = t
        self.x = x
        self.loss = sum_squared_error(self.x, t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.x - self.t) / batch_size

        return dx

def sum_squared_error(x, t):
    if x.ndim == 1:
        t = t.reshape(1, t.size)
        x = x.reshape(1, x.size)

    batch_size = x.shape[0]
    return 0.5 * np.sum((x - t) ** 2) / batch_size
