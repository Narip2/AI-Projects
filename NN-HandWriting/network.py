import numpy as np

class Network:

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(x,y) for y, x in zip(sizes[:,-1], sizes[1:])]

    def feedforward(self, a):
        # as a list, a stands for the inputs
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

        
# sigmoid function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

test = iter(range(3))
for i in test:
    print(i)