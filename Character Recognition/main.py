import backend
import numpy as np
import math

def main():
    network_size = 25
    char_recognizer = backend.Hopfield(network_size)
    a_file = np.genfromtxt('patterns/a_file.csv', delimiter=',').flatten()
    b_file = np.genfromtxt('patterns/b_file.csv', delimiter=',').flatten()
    c_file = np.genfromtxt('patterns/c_file.csv', delimiter=',').flatten()
    d_file = np.genfromtxt('patterns/d_file.csv', delimiter=',').flatten()
    #e_file = np.genfromtxt('patterns/e_file.csv', delimiter=',').flatten()
    
    char_set = np.array([a_file, b_file, c_file, d_file])
    char_recognizer.train_network(char_set) #Train hopfield network on the given set of patterns

    print(char_recognizer.weight_matrix)
    in_file = np.genfromtxt('in_file.csv', delimiter=',')
    print("Inputted Pattern:")
    print(in_file)

    out_char_vector = char_recognizer.recognize_char(in_file.flatten())
    output = out_char_vector.view().reshape(int(math.sqrt(network_size)), int(math.sqrt(network_size))) #Reshape for readability of the result

    print("\nRecognized Pattern:")
    print(output)

if __name__ == '__main__':
    main()

