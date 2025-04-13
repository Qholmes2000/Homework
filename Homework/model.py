# model.py
import numpy as np
from config import PARAM_GRID

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

class FCLayer:
    def __init__(self, input_size, output_size, init_method='he'):
        self.W = self._initialize_weights(input_size, output_size, init_method)
        self.b = np.zeros((1, output_size))
        self.dW = None
        self.db = None
        self.input = None
        
    def _initialize_weights(self, in_dim, out_dim, method):
        if method == 'xavier':
            scale = np.sqrt(1. / in_dim)
        elif method == 'he':
            scale = np.sqrt(2. / in_dim)
        else:
            scale = 0.01
        return np.random.randn(in_dim, out_dim) * scale

    def forward(self, X):
        self.input = X
        return X.dot(self.W) + self.b

    def backward(self, dout):
        self.dW = self.input.T.dot(dout)
        self.db = np.sum(dout, axis=0, keepdims=True)
        dx = dout.dot(self.W.T)
        return dx

class ActivationLayer:
    def __init__(self, activation='relu'):
        self.activation = activation
        self.input = None
        self.output = None
        
    def forward(self, X):
        self.input = X
        if self.activation == 'relu':
            self.output = np.maximum(0, X)
        elif self.activation == 'sigmoid':
            self.output = 1 / (1 + np.exp(-X))
        elif self.activation == 'tanh':
            self.output = np.tanh(X)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
        return self.output
    
    def backward(self, dout):
        if self.activation == 'relu':
            grad = (self.input > 0).astype(float)
        elif self.activation == 'sigmoid':
            sig = 1 / (1 + np.exp(-self.input))
            grad = sig * (1 - sig)
        elif self.activation == 'tanh':
            grad = 1 - np.square(self.output)
        return dout * grad

class NeuralNetwork:
    @staticmethod
    def softmax(x):
        """稳定的softmax实现"""
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def __init__(self, layer_sizes, activations, init_methods, reg_strength=0.01):
        self.layers = []
        self.reg_strength = reg_strength
        
        # 创建全连接层和激活层交替结构
        for i in range(len(layer_sizes)-1):
            self.layers.append(FCLayer(layer_sizes[i], layer_sizes[i+1], init_methods[i]))
            if i < len(layer_sizes)-2:  # 最后一层不需要激活
                self.layers.append(ActivationLayer(activations[i]))
    
    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out
    
    def backward(self, dout):
        grad = dout
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def compute_loss(self, y, reg_strength):
        m = y.shape[0]
        log_probs = -np.log(softmax(self.layers[-1].input) + 1e-8)
        loss = np.sum(log_probs[np.arange(m), y.argmax(axis=1)]) / m
        
        reg_loss = 0
        for layer in self.layers:
            if isinstance(layer, FCLayer):
                reg_loss += 0.5 * reg_strength * np.sum(layer.W ** 2)
        return loss + reg_loss

    def predict(self, X):
        scores = self.forward(X)
        return np.argmax(scores, axis=1)