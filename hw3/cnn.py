from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils.layer_utils import layer_from_config
from keras.utils import np_utils
from keras.optimizers import SGD
import numpy as np
import cPickle
import gc

class CNN(object):
    """docstring for CNN"""
    def __init__(self):
        self.width = 32
        self.height = 32
        self.class_num = 10

    def parseLabelData(self, path):
        print "Parsing label data..."
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
        print "Parsing unlabel data..."
        all_unlabel = cPickle.load(open(path, 'rb'))
        X_unlabel = []
        for i in range(len(all_unlabel)):
            img = np.array(all_unlabel[i])
            X_unlabel.append(img.reshape(3, self.width, self.height))
        X_unlabel = np.array(X_unlabel, dtype='float32')
        return X_unlabel

    def parseTestData(self, path):
        test = cPickle.load(open(path, 'rb'))
        X_test = []
        for img in test['data']:
            img = np.array(img)
            X_test.append(img.reshape(3, self.width, self.height))
        X_test = np.array(X_test, dtype='float32')
        return X_test

    def constructCNN(self, X_train, Y_train):
        print "Start to train CNN..."
        model = Sequential()
        model.add(Convolution2D(25, 3, 3, input_shape=(3, self.width, self.height)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Convolution2D(50, 3, 3))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dropout(0.25))
        for i in range(10):
            #model.add(Dropout(0.25))
            model.add(Dense(100))
            model.add(Activation("relu"))
        model.add(Dropout(0.25))
        model.add(Dense(self.class_num))
        model.add(Activation('softmax'))
        #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, Y_train, nb_epoch=45, batch_size=128)
        return model

    def addUnlabelData(self, model, X_unlabel, label_flag):
        print "Adding unlabel data..."
        gc.collect()
        output = model.predict(X_unlabel, batch_size=128)
        X_selftrain = []
        Y_unlabel = []
        for i in range(output.shape[0]):
            value = np.amax(output[i])
            if value > 0.8 and label_flag[i] == 0:
                X_selftrain.append(X_unlabel[i])
                Y_unlabel.append(np.argmax(output[i]))
                label_flag[i] = 1
        X_selftrain = np.array(X_selftrain, dtype='float32')
        Y_unlabel = np.array(Y_unlabel, dtype='uint8')
        Y_unlabel = np_utils.to_categorical(Y_unlabel, self.class_num)
        return X_selftrain, Y_unlabel, label_flag

    def predict(self, model, X_test):
        print "Predicting..."
        predict_file = open("predict.csv", "w")
        predict_file.write("ID,class\n")
        output = model.predict(X_test, batch_size=128)
        for i in range(output.shape[0]):
            predict_file.write(str(i) + "," + str(np.argmax(output[i])) + "\n")


if __name__ == '__main__':
    cnn = CNN()
    label_data_path = '../../data/all_label.p'
    unlabel_data_path = '../../data/all_unlabel.p'
    test_data_path = '../../data/test.p'

    X_label, Y_label = cnn.parseLabelData(label_data_path)
    X_unlabel = cnn.parseUnlabelData(unlabel_data_path)
    X_test = cnn.parseTestData(test_data_path)
    model = cnn.constructCNN(X_label, Y_label)

    print "Self-training..."
    label_flag = np.zeros(X_unlabel.shape[0])
    for i in range(5):
        print np.count_nonzero(label_flag)
        if np.count_nonzero(label_flag) >= X_unlabel.shape[0]:
            break
        X_selftrain, Y_unlabel, label_flag = cnn.addUnlabelData(model, X_unlabel, label_flag)
        X_label = np.concatenate((X_label, X_selftrain), axis=0)
        Y_label = np.concatenate((Y_label, Y_unlabel), axis=0)
        model = cnn.constructCNN(X_label, Y_label)

    cnn.predict(model, X_test)
