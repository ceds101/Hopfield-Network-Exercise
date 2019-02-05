import numpy as np


class Hopfield:
    # Hopfield network attributes
    network_size = 25 # Number of nodes in the network (N)
    weight_matrix = np.zeros((network_size, network_size)) # N x N weight matrix

    # Constructor/Initialization
    def __init__(self):
        pass

    # Train the network for characters
    def train_network(self, input_vector):
        pass

    # Recognize a character
    def recognize_char(self, input_vector):
        pass