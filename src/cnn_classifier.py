import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from keras_model import Keras
from mode import Mode
 

def greeting():
    print("The program consists of 3 modes.")
    print("The 1 mode is a training without augmentation.")
    print("The 2 mode is a training with augmentation.")
    print("The final mode is a training without augmentation")


def choose_mode():
    mode = int(input("Choose the program mode [1, 2, 3 (final)]: "))
    
    if mode <= 0 or mode > 3:
        raise Exception("The passed mode " + str(mode) + " does not exist!")

    return mode


def main():
    greeting()
    mode = choose_mode()
    print()

    program = Mode(mode, sys.argv[-1], True)
    print()
    program.data_info()
    print()

    program.run_mode(256, 50, 0)
    NN = program.get_NN()
    
    report = NN.get_report()
    print(report['loss'], report['accuracy'])


if __name__ == '__main__':
    main()

