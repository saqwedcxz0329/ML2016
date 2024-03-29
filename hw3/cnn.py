from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils.layer_utils import layer_from_config
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.models import load_model
import numpy as np
import pickle
import gc
import sys

class Data(object):
    def __init__(self, channel, width, height, class_num):
        self.channel = channel
        self.width = width
        self.height = height
        self.class_num = class_num

    def parseLabelData(self, path):
        print ("Parsing label data...")
        all_label = pickle.load(open(path, 'rb'))
        X_label = []
        Y_label = []
        for i in range(len(all_label)):
            for j in range(len(all_label[i])):
                img = np.array(all_label[i][j], dtype='float32')
                X_label.append(img.reshape(self.channel, self.width, self.height))
                Y_label.append(i)
                
        X_label = np.array(X_label, dtype='float32')
        Y_label = np.array(Y_label, dtype='uint8')
        Y_label = np_utils.to_categorical(Y_label, self.class_num)
        del all_label
        return X_label/255., Y_label

    def parseUnlabelData(self, path):
        print ("Parsing unlabel data...")
        all_unlabel = pickle.load(open(path, 'rb'))
        X_unlabel = np.array(all_unlabel, dtype='float32').reshape(len(all_unlabel), self.channel, self.width, self.height)
        del all_unlabel
        return X_unlabel/255.

    def parseTestData(self, path):
        print ("Parsing test data...")
        test = pickle.load(open(path, 'rb'))
        X_test = np.array(test['data'], dtype='float32').reshape(len(test['data']), self.channel, self.width, self.height)
        del test
        return X_test/255.

class CNN(object):
    def __init__(self, channel, width, height, class_num):
        self.channel = channel
        self.width = width
        self.height = height
        self.class_num = class_num

    def cnn_model(self):
        print "Start to train CNN..."
        model = Sequential()
        model.add(Convolution2D(16, 3, 3, input_shape=(3, self.width, self.height)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Convolution2D(64, 3, 3))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dropout(0.25))
        for i in range(10):
            model.add(Dense(100))
            model.add(Activation("relu"))
        model.add(Dropout(0.25))
        model.add(Dense(self.class_num))
        model.add(Activation('softmax'))
        return model

    def addUnlabelData(self, model, X_unlabel, label_flag):
        print "Adding unlabel data..."
        #gc.collect()
        output = model.predict(X_unlabel, batch_size=128)
        X_selftrain = []
        Y_unlabel = []
        for i in np.where(label_flag==0)[0]:
            value = np.amax(output[i])
            if value > 0.85 and label_flag[i] == 0:
                X_selftrain.append(X_unlabel[i])
                Y_unlabel.append(np.argmax(output[i]))
                label_flag[i] = 1
        X_selftrain = np.array(X_selftrain, dtype='float32')
        Y_unlabel = np.array(Y_unlabel, dtype='uint8')
        Y_unlabel = np_utils.to_categorical(Y_unlabel, self.class_num)
        del output
        return X_selftrain, Y_unlabel, label_flag

if __name__ == '__main__':
    data_directory = sys.argv[1]
    model_name = sys.argv[2]
    channel = 3
    width = 32
    height = 32
    class_num = 10
    label_data_path = data_directory +  '/all_label.p'
    unlabel_data_path = data_directory + '/all_unlabel.p'
    test_data_path = data_directory + '/test.p'

    data = Data(channel, width, height, class_num) #(channel, width, height, class_num)
    cnn = CNN(channel, width, height, class_num)   #(channel, width, height, class_num)

    X_label, Y_label = data.parseLabelData(label_data_path)
    X_unlabel = data.parseUnlabelData(unlabel_data_path)
    X_test = data.parseTestData(test_data_path)

    #X_unlabel = np.concatenate((X_unlabel, X_test), axis=0)

    model = cnn.cnn_model()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_label, Y_label, nb_epoch=50, batch_size=128)

    print "Self-training..."
    label_flag = np.zeros(X_unlabel.shape[0])
    for i in range(10):
        print np.count_nonzero(label_flag)
        if np.count_nonzero(label_flag) >= X_unlabel.shape[0]:
            break
        X_selftrain, Y_unlabel, label_flag = cnn.addUnlabelData(model, X_unlabel, label_flag)
        print X_label.shape
        print X_selftrain.shape
        if X_selftrain.shape[0] != 0 :
            X_label = np.concatenate((X_label, X_selftrain), axis=0)
            Y_label = np.concatenate((Y_label, Y_unlabel), axis=0)
        model.fit(X_label, Y_label, nb_epoch=15, batch_size=128)
    model.save(model_name)  # creates a HDF5 file 'my_model.h5'
    
