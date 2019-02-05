import backend
import numpy as np

def main():
    char_recognizer = backend.Hopfield(25)
    a_file = np.genfromtxt('patterns/a_file.csv', delimiter=',').flatten()
    b_file = np.genfromtxt('patterns/b_file.csv', delimiter=',').flatten()
    c_file = np.genfromtxt('patterns/c_file.csv', delimiter=',').flatten()

    char_set = np.array([a_file, b_file, c_file])
    char_recognizer.train_network(char_set) #Train hopfield network on the given set of patterns

    #print(char_recognizer.weight_matrix)
    in_file = np.genfromtxt('in_file.csv', delimiter=',').flatten()
    print(char_recognizer.recognize_char(in_file))

if __name__ == '__main__':
    main()

