import random
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
    
    def SGD(self, training_data, epochs, mini_batch_size, eta):
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
            

    def update_mini_batch(self, mini_batch, eta):
        # init
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w = [nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
        self.weights = [w-eta*nw/len(mini_batch) for w,nw in zip(self.weights,nabla_w)]
        self.biases = [b-eta*nb/len(mini_batch) for b,nb in zip(self.biases,nabla_b)]


    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x] # a list used to store all actions for different layers
        zs = [] # store all the z vectors
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1]) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta,activations[-2].transpose())
        for l in range(2,self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1],delta)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta,activations[-l-1].transpose())
        return (nabla_b, nabla_w)

        


    def cost_derivative(self, output_activations, y):
        return (output_activations - y)
        
# sigmoid function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

