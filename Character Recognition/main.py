import backend
import numpy as np

def main():
    char_recognizer = backend.Hopfield(25)
    a_file = np.genfromtxt('a_file.csv', delimiter=',')
    b_file = np.genfromtxt('b_file.csv', delimiter=',')
    c_file = np.genfromtxt('c_file.csv', delimiter=',')

    char_set = np.array([a_file, b_file, c_file])

    char_recognizer.train_network(char_set) #Train hopfield network on the given set of patterns

    print(char_recognizer.weight_matrix)
    print(char_recognizer.recognize_char(np.genfromtxt('a_file.csv', delimiter=',')))

if __name__ == '__main__':
    main()

