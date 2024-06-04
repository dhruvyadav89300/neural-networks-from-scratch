import numpy as np
import pandas as pd
data = pd.read_csv("train.csv")
inputs = np.array(data)
m, n = inputs.shape
np.random.shuffle(inputs)
y_train = inputs[:, 0]
X_train = inputs[:, 1:]


class ReLU:
    def activate(self, inputs):
        self.output = np.maximum(0, inputs)
        return self.output

class softmax():
    def activate(self, inputs):
        exps = np.exp(inputs - np.max()) 

class denseLayer:
    def __init__(self, n_i, n, af):
        self.weights = np.random.randn(n_i, n)
        self.biases = np.zeros((1, n))
        if af == "relu":
            self.activation = ReLU()
        if af == "softmax":
            self.activation = softmax()
    def forwardPass(self, inputs):
        print(self.weights.shape)
        self.z = np.dot(inputs, self.weights) + self.biases
        self.output = self.activation.activate(self.z)

## The code is yet to be completed

layer1 = denseLayer(784, 16, af="relu")
layer1.forwardPass(X_train)
layer2 = denseLayer(16, 16, af="relu")
layer2.forwardPass(layer1.output)
layer3 = denseLayer(16, 10, af="softmax")
layer3.forwardPass(layer2.output)
