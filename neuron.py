import random
import math

class Neuron(object):
    """
    An abstract class describe how to implement a Neuron
    """
    def __init__(self, nb_inputs=1, bias=1):
        """
        Define how many inputs a neuron owns
        """
        # Arbitrary chosen
        self.learning_control = 0.01
        # Bias
        self.bias = bias
        # set inputs
        self.setNbInputs(nb_inputs=nb_inputs)

    def setNbInputs(self, nb_inputs):
        """
        Set the neuron number of inputs
        """
        # At first ways weights are initialized to random values
        self.weights = [round(random.uniform(-1.0, 1.0), 3) for weight in range(nb_inputs)]

    def activate(self, inputs_sum):
        """
        Abstract activate function trigger an answer following inputs and assiociated weights
        """
        raise NotImplementedError

    def feedforward(self, inputs):
        """
        Allows to send data using inputs part
        Returns the output
        """
        processed_inputs = inputs[:]
        inputs_sum = sum([input_value * self.weights[i] for i, input_value in enumerate(processed_inputs)])
        return self.activate(inputs_sum)

class Perceptron(Neuron):
    """
    Simplest neuron possible, only output 1 or 0, following a threshold
    """
    def __init__(self, nb_inputs=1, bias=1):
        """
        Create a Perceptron
        """
        Neuron.__init__(self, nb_inputs=nb_inputs, bias=bias)

    def activate(self, inputs_sum):
        """
        A step function, X < T => 0 or X => T => 1
        """
        return 1 if inputs_sum + self.bias > 0 else 0

class Sigmoid(Neuron):
    """
    Sigmoid neuron, its output is less suggestible than perceptron
    allowing to learn more easily
    """
    def __init__(self, nb_inputs=1, bias=1):
        """
        Create a Sigmoid
        """
        Neuron.__init__(self, nb_inputs=nb_inputs, bias=bias)

    def __sigmoid(x):
        return 1/float(1 + math.exp(-x))

    def activate(self, inputs_sum):
        """
        A step function, X < T => 0 or X => T => 1
        """
        return self.__sigmoid(inputs_sum + self.bias)
