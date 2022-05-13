import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from keras_model import Keras
from preprocessing import Preprocessing


def data_info(train, test, train_l, test_l):
    print("INFORMATION BY DATASET")

    print('Training data:', train.shape, train_l.shape)
    print('Testing data:', test.shape, test_l.shape)
    
    np_labels = np.unique(train_l)
    len_labels = len(np_labels)

    print('Labels:', np_labels)
    print('Total labels:', len_labels)
 
    
def get_dataset(datadir):
    dataset = Preprocessing(datadir)
    dataset.get_normalized_data()
    dataset.reshape_data()
    return dataset 


def main():
    #path_to_dataset = util.get_path(sys.argv)

    datadir = sys.argv[-1]
    dataset = get_dataset(datadir)
    data = dataset.get_data()
    labels = dataset.get_labels()

    (train, validating, train_l, validating_l) = dataset.split_data(data, labels, size_of_test=0.2, rand_state=None)
    (valid, test, valid_l, test_l) = dataset.split_data(validating, validating_l, size_of_test=0.5, rand_state=None)

    #data_info(train, test, train_l, test_l)
    #data_info(valid, tst, valid_l, tst_l)

    print("The program consists of 2 modes.")
    print("The 1 mode is a training without augmentation.")
    print("The 2 mode is a training with augmentation.")
    mode = int(input("Choose the program mode [1, 2]: "))

    
    NN = Keras(train, valid, train_l, valid_l)

    if mode == 1:
        NN.train_network(batch=256, iteration=50, verb=0)

    elif mode == 2:
        NN.train_network_with_augmentation(batch=256, iteration=50, verb=0)

    else:
        raise Exception("The passed mode does not exist!")

    report = NN.get_report()
    print(report['classification report'])


if __name__ == '__main__':
    main()

