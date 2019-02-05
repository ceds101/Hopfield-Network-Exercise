import backend
import numpy as np

def main():
    char_recognizer = backend.Hopfield(5)
    print('hello world')
    char_recognizer.train_network(np.array([[0, 1 ,1, 0,1], [1, 0, 1 ,0, 1]]))
    print(char_recognizer.weight_matrix)
    print(char_recognizer.recognize_char([1, 1, 1, 1, 1]))

if __name__ == '__main__':
    main()