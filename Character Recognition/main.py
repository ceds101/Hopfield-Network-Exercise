import backend
import numpy as np

def main():
    char_recognizer = backend.Hopfield(5)
    char_set = np.array([[0,1,1,0,1],  #V_1
                         [1,0,1,0,1]   #V_2
                        ])

    char_recognizer.train_network(char_set) #Train hopfield network on the given set of patterns

    print(char_recognizer.weight_matrix)
    print(char_recognizer.recognize_char([1, 1, 1, 1, 1]))

if __name__ == '__main__':
    main()