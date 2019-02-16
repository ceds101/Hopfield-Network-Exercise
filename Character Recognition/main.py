import backend
import numpy as np
import math

def main():
    network_size = 25
    char_recognizer = backend.Hopfield(network_size)
    char_set = np.empty(shape=(0, 25))

    for i in range(3):
        char_filename = "patterns/pattern_" + chr(i + 65) + ".csv"
        char_file = np.genfromtxt(char_filename, delimiter=',').flatten()
        char_set = np.append(char_set, char_file.reshape((-1, 25)), axis=0)

    print(char_set)

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

