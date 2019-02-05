import numpy as np
import random

class Hopfield:
    # Hopfield network attributes
    network_size = 0 # Number of nodes in the network (N)
    weight_matrix = np.zeros((0, 0)) # N x N weight matrix

    # Constructor/Initialization
    def __init__(self, network_size = 25):
        self.network_size = network_size
        self.weight_matrix = np.zeros((self.network_size, self.network_size))

    # Train the network for characters
    def train_network(self, char_set):
        i = 0
        j = 0

        #Updating the weights for each connection for each pattern (update weight in upper triangular matrix)
        for char_vector in char_set: # char_vector: Vector form of a character (1 0 0 1 0 ...) = A
            for i in range(self.network_size):
                for j in range(i+1,self.network_size):
                    #print(i,' ',j)
                    self.weight_matrix[i][j] += (2*char_vector[i] - 1)*(2*char_vector[j] - 1) # storage prescription formula
                    self.weight_matrix[j][i] = self.weight_matrix[i][j] #Copy to lower triangle

    # Recognize a character
    def recognize_char(self, input_vector):
        node_changed = True #Assume a node has changed
        curr_state = input_vector

        node_index_order = list(range(0, self.network_size))
        random.shuffle(node_index_order)

        while node_changed:
            node_changed = False
            for curr_node_index in node_index_order:
                prev_state = curr_state.copy()
                node_weight_vector = self.weight_matrix[:,curr_node_index]
                curr_state[curr_node_index] = self.__calculate_node_value(node_weight_vector, curr_state) #the node is the jth column of the weight matrix
                if np.array_equal(prev_state, curr_state) == False:
                    node_changed = True
        
        return curr_state
                
        

           
                
        





    # PRIVATE FUNCTIONS
    def __calculate_node_value(self, node_weight_vector, curr_state):
        node_value = np.dot(node_weight_vector, curr_state)
        if node_value >= 0:
            output_node_value = 1
        else:
            output_node_value = 0
        return output_node_value
