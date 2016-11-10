from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils.layer_utils import layer_from_config
from keras.utils import np_utils
import numpy as np
import cPickle

class CNN(object):
    """docstring for CNN"""
    def __init__(self):
        self.width = 32
        self.height = 32
        self.class_num = 10

    def parseLabelData(self, path):
        all_label = cPickle.load(open(path, 'rb'))
        X_train = []
        Y_train = []
        for i in range(len(all_label)):
            for j in range(len(all_label[i])):
                img = np.array(all_label[i][j], dtype='float32')
                X_train.append(img.reshape(3, self.width, self.height))
                Y_train.append(i)
                
        X_train = np.array(X_train, dtype='float32')
        Y_train = np.array(Y_train, dtype='uint8')
        Y_train = np_utils.to_categorical(Y_train, self.class_num)
        return X_train, Y_train

    def parseUnlabelData(self, path):
        all_unlabel = cPickle.load(open(path, 'rb'))
        X_test = []
        for i in range(len(all_unlabel)):
            img = np.array(all_unlabel[i])
            X_test.append(img.reshape(3, self.width, self.height))
        X_test = np.array(X_train, dtype='float32')
        return X_test

    def constructCNN(self, X_train, Y_train):
        model = Sequential()
        model.add(Convolution2D(25, 3, 3, input_shape=(3, self.width, self.height)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Convolution2D(50, 3, 3))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        for i in range(10):
            model.add(Dense(100))
            model.add(Activation("relu"))
        model.add(Dense(self.class_num))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, Y_train, nb_epoch=40, batch_size=128)
        return model

if __name__ == '__main__':
    cnn = CNN()
    label_data_path = '../../data/all_label.p'
    unlabel_data_path = '../../data/all_unlabel.p'
    X_train, Y_train = cnn.parseLabelData(label_data_path)
    #X_test = cnn.parseUnlabelData(unlabel_data_path)
    #model = cnn.constructCNN(X_train, Y_train)
    #activations = model.predict(X_test)
    #print activations[0]