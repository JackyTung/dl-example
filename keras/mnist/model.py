import keras
#from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import math
import os
#from mnist import MNIST
from numpy import array
from importlib import import_module

num_classes = 10
img_rows, img_cols = 28, 28

loader = import_module('loader')

def get_model():
    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 1)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


class DataGenerator:
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        # NOTE: should add $GRAPE_DATASET_DIR before your original data path
        path_data =  os.environ.get("GRAPE_DATASET_DIR")
        self.mndata = loader.MNIST(path_data)

    def training_data(self):
        return self.data_generator(True, self.batch_size)

    def training_steps(self):
        return math.ceil(float(600) / self.batch_size)

    def validation_data(self):
        return self.data_generator(False, self.batch_size)

    def validation_steps(self):
        return math.ceil(float(100) / self.batch_size)

    def data_generator(self, isTrain=True, batchSize=100):
        # the data, shuffled and split between train and test sets
        x_train, y_train = self.mndata.load_training()
        x_train = array(x_train)
        x_test, y_test = self.mndata.load_testing()
        x_test = array(x_test)
        print ('K.image_data_format()', K.image_data_format())
        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        if(isTrain):
            dataset = (x_train, y_train)
            dataset_size = 600
        else:
            dataset = (x_test, y_test)
            dataset_size = 100
        i = 0
        while(True):
            if (i + batchSize > dataset_size):
                i = 0
            yield dataset[0][i:i + batchSize], dataset[1][i:i + batchSize]
            i += batchSize
