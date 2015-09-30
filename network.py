import re
from neuron import Sigmoid
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
    def __init__(self, layout):
        """
        A layout parameter is a string describing network architecture
        the pattern is for example 784_12_15_10, that create a layered network
        involving 784 inputs neurons, 2 hidden layers compose respectively of 12
        and 15 neurons and returns its results with a pool of 10 outputs. Of course
        hidden layers aren't mandatory so you can write 784_10 if you want and you
        will get 784 inputs linked to 10 outputs.
        """
        pattern = re.compile("([\d_]*)")
        matched = re.findall(pattern, layout)
        if not matched:
            raise ValueError("Incorrect layout provided")
        nb_neurons_on_layers = matched[0].split("_")
        self.layers = [int(nb_neuron_on_this_layer) for nb_neuron_on_this_layer in nb_neurons_on_layers][::-1]
        self.neurons = {}
        self.__generate()

    def __generate(self):
        """
        Create the network
        """
        neuron_id = 0
        layer_id = len(self.layers)
        neurons_on_next_layer = []

        #create layers and links between them
        for layer in self.layers:
            neurons_on_next_layer_tmp = []
            for i in range(layer):
                neuron = Sigmoid(nb_inputs=1)
                neurons_on_next_layer_tmp.append((neuron, neuron_id))
                self.neurons[neuron_id] = {
                    "neuron": neuron,
                    "linked_to" : neurons_on_next_layer,
                    "layer" : layer_id,
                }
                neuron_id += 1
            neurons_on_next_layer = neurons_on_next_layer_tmp[:]
            layer_id -= 1

    def draw_network(self):
        """
        Create a DOT file showing network architecture
        """
        with open("network.dot", "w") as fd:
            fd.write("digraph {\n")
            for neuron_id in self.neurons.keys():
                for linked_neuron in self.neurons[neuron_id]["linked_to"]:
                    fd.write("%s -> %s;\n" % (neuron_id, linked_neuron[1]))
                if not self.neurons[neuron_id]["linked_to"]:
                    fd.write("%s;\n" % (neuron_id))
            fd.write("}")


if __name__ == '__main__':
    arguments = docopt(help)
    network = LayeredNetwork(layout=arguments["--layout"])
    network.draw_network()
