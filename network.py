import re
import random
import numpy as np
from pprint import pprint
from docopt import docopt

help = """Perceptron

Usage:
  network.py [--layout=<layout>] [--save_file=<save_file>]

Options:
  -h --help                               Display this help.
  --layout=<layout>                       Provide a layout to the network [default: 1_1].
  --save_file=<save_file>                 Pickle file to network state.

Creates a neural composed by a bunch of sigmoids :)
"""

class LayeredNetwork(object):
    """
    Make a network where each neurons of one layer are linked to each neurons of previous layer
    """
    def __init__(self, layout, eta=0.001):
        """
        A layout parameter is a string describing network architecture
        the pattern is for example 784_12_15_10, that create a layered network
        involving 784 inputs neurons, 2 hidden layers compose respectively of 12
        and 15 neurons and returns its results with a pool of 10 outputs. Of course
        hidden layers aren't mandatory so you can write 784_10 if you want and you
        will get 784 inputs linked to 10 outputs.
        eta handles network learning creativity
        """
        pattern = re.compile("([\d_]*)")
        matched = re.findall(pattern, layout)
        if not matched:
            raise ValueError("Incorrect layout provided")
        sizes = [ int(x) for x in matched[0].split("_") ]
        self.eta = eta
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [ np.random.randn(y, 1) for y in sizes[1:] ]
        self.weights = [ np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:]) ]

    def schotastic_gradient_descent(self, batch, nb_trainings, batch_size):
        """
        Following a specific algorithm trains the network to handle various
        situation, batches is a list of (x, y), where x is the input vector
        and y is the output vector. nb_trainings specifies how many train will
        be performed and batch_size
        """
        print "Starting training"
        n = len(batches)
        for training_num in range(nb_trainings):
            random.shuffle(batches)
            training_data = [ batch[i:i+batch_size] for i in xrange(0, n batch_size) ]
            for batch in training_data:
                self.training(batch)
        print "End of training"

    def feedforward(self, a):
        """
        Return the output of the network if "a" is input
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a # output vecor of the last layer

    def training(self, batch):
        #initiate nabla list
        nabla_b = [ np.zeros(b.shape) for b in self.biases ]
        nabla_w = [ np.zeros(w.shape) for w in self.weights ]
        # launch different training inputs x and compute results regarding expected value y
        for x, y in batch:
            delta_nabla_b, delta_nabla_w = self.backpropagation(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_b = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        #modify network biases and weights following network outputs
        self.weights = [w - (self.eta/len(batch)) * nw for w,nw in zip(self.weights, nabla_w)]
        self.weights = [b - (self.eta/len(batch)) * nb for w,nb in zip(self.biases, nabla_b)]

    def backpropagation(self, x, y):
        """
        x input vector
        y expected output vector
        return delta_nabla_b and delta_nabla_w, which are lists of optimized delta to append
        to current network biases and weight to reach with the fastest way the expected result
        """
        #initiate nabla list
        delta_nabla_b = [ np.zeros(b.shape) for b in self.biases ]
        delta_nabla_w = [ np.zeros(w.shape) for w in self.weights ]
        #initiate feed forwarding
        a = x
        activations = [x]
        zs = []
        #propagate inputs toward outputs
        for b,w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            zs.append(z)
            a = sigmoid(z)
            activations.append(a)
        #output checking
        delta = (activations[-1] - y), sigmoid_prime(zs[-1])
        #backpropagation initialization
        delta_nabla_b[-1] = delta
        delta_nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        #backpropagation
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_prime(z)
            delta_nabla_b[-l] = delta
            delta_nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (delta_nabla_b, delta_nabla_w)

def sigmoid(z):
    """
    An elementwise activation function
    z a vector
    """
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """
    An elementwise prime activation function
    z a vector
    """
    return sigmoid(z) * (1.0 - sigmoid(z))

if __name__ == '__main__':
    arguments = docopt(help)
    network = LayeredNetwork(layout=arguments["--layout"])
