from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils.layer_utils import layer_from_config
from keras.utils import np_utils
from keras.optimizers import SGD
import numpy as np
import cPickle
import gc
import sys

class AutoEncoder(object):
    def __init__(self, channel, width, height, class_num):
        self.channel = channel
        self.width = width
        self.height = height
        self.class_num = class_num

    def autoencoder_model(self):
        input_dim = self.channel * self.width * self.height
        encoding_dim = 256
        input_img = Input(shape=(input_dim,))

        encoded = Dense(1024, activation='relu')(input_img)
        encoded = Dense(512, activation='relu')(encoded)
        encoded = Dense(encoding_dim, activation='relu')(encoded)

        decoded = Dense(512, activation='relu')(encoded)
        decoded = Dense(1024, activation='relu')(decoded)
        decoded = Dense(input_dim, activation='sigmoid')(decoded)

        autoencoder = Model(input=input_img, output=decoded)
        encoder = Model(input=input_img, output=encoded)
        return autoencoder, encoder

    def clustering(self, encoder, X_train, Y_train):
        print "Clustering..."
        encoded_imgs = encoder.predict(X_train)
        for i in range(5000, X_train.shape[0]):
            min_distance = sys.maxint
            for j in range(10):
                # Find the datas belong to class j
                class_index = np.where(Y_train == j)[0]
                # Compute the cluster's centroid
                centroid = np.mean(encoded_imgs[class_index], axis=0)
                # Find the closest cluster
                euclidean_distance = np.linalg.norm(encoded_imgs[i] - centroid)
                if euclidean_distance < min_distance:
                    min_distance = euclidean_distance
                    Y_train[i] = j
            print i
        return Y_train

class Data(object):
    def __init__(self, channel, width, height, class_num):
        self.channel = channel
        self.width = width
        self.height = height
        self.class_num = class_num

    def parseLabelData(self, path):
        print ("Parsing label data...")
        all_label = cPickle.load(open(path, 'rb'))
        X_label = []
        Y_label = []
        for i in range(len(all_label)):
            for j in range(len(all_label[i])):
                img = np.array(all_label[i][j], dtype='float32')
                X_label.append(img.reshape(self.channel, self.width, self.height))
                Y_label.append(i)
                
        X_label = np.array(X_label, dtype='float32')
        Y_label = np.array(Y_label, dtype='uint8')
        return X_label/255., Y_label

    def parseUnlabelData(self, path):
        print ("Parsing unlabel data...")
        all_unlabel = cPickle.load(open(path, 'rb'))
        X_unlabel = []
        for i in range(len(all_unlabel)):
            img = np.array(all_unlabel[i])
            X_unlabel.append(img.reshape(self.channel, self.width, self.height))
        X_unlabel = np.array(X_unlabel, dtype='float32')
        return X_unlabel/255.

    def parseTestData(self, path):
        print ("Parsing test data...")
        test = cPickle.load(open(path, 'rb'))
        X_test = []
        for img in test['data']:
            img = np.array(img)
            X_test.append(img.reshape(3, self.width, self.height))
        X_test = np.array(X_test, dtype='float32')
        return X_test/255.

class CNN(object):
    def __init__(self, channel, width, height, class_num):
        self.channel = channel
        self.width = width
        self.height = height
        self.class_num = class_num

    def cnn_model(self):
        print ("Start to train CNN...")
        model = Sequential()
        model.add(Convolution2D(25, 3, 3, input_shape=(self.channel, self.width, self.height)))
        model.add(MaxPooling2D((2, 2)))
        #model.add(Convolution2D(64, 3, 3))
        #model.add(MaxPooling2D((2, 2)))
        model.add(Convolution2D(50, 3, 3))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        #model.add(Dropout(0.3))
        for i in range(15):
            model.add(Dense(100))
            model.add(Activation("relu"))
        #model.add(Dropout(0.3))
        model.add(Dense(self.class_num))
        model.add(Activation('softmax'))
        return model

    def predict(self, model, X_test):
        print ("Predicting...")
        predict_file = open("predict.csv", "w")
        predict_file.write("ID,class\n")
        output = model.predict(X_test, batch_size=128)
        for i in range(output.shape[0]):
            predict_file.write(str(i) + "," + str(np.argmax(output[i])) + "\n")


if __name__ == '__main__':
    channel = 3
    width = 32
    height = 32
    class_num = 10
    label_data_path = '../../data/all_label.p'
    unlabel_data_path = '../../data/all_unlabel.p'
    test_data_path = '../../data/test.p'

    data = Data(channel, width, height, class_num) #(channel, width, heigth, class_num)
    cnn = CNN(channel, width, height, class_num)   #(channel, width, heigth, class_num)
    ## Read data
    X_label, Y_label = data.parseLabelData(label_data_path)
    X_unlabel = data.parseUnlabelData(unlabel_data_path)
    X_test = data.parseTestData(test_data_path)
   
    X_unlabel = X_unlabel[0:40000]
    ## Concatenate date
    #X_unlabel = np.concatenate((X_unlabel, X_test), axis=0)
    X_train = np.concatenate((X_label, X_unlabel), axis=0)
    Y_unlabel = np.empty(X_unlabel.shape[0])
    Y_unlabel.fill(-1)
    Y_train = np.concatenate((Y_label, Y_unlabel), axis=0)
    
    ############ Autoencoder ############
    print ("Start to train autoencoder")
    X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
    ## Add noisy
    noise_factor = 0.5
    #X_train = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape[1])
    #X_train = X_train + 0.1
    for i in range(X_train.shape[0]):
        #print X_train.shape[1]
        #print X_train[i].shape
        X_train[i] = X_train[i] + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape[1])
    
    X_train = np.clip(X_train, 0., 1.)
    X_train.astype('float32')
    print (X_train.shape)
    print (Y_train.shape)
     
    gc.collect()
    ## Training
    ae = AutoEncoder(3,32,32,10) #(channel, width, heigth, class_num)
    autoencoder, encoder = ae.autoencoder_model()
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.fit(X_train, X_train,
                    nb_epoch=75,
                    batch_size=128,
                    shuffle=True)
    ## Clustering
    Y_train = ae.clustering(encoder, X_train, Y_train)
    Y_train = np_utils.to_categorical(Y_train, 10)

    #X_train = X_train[0:10000]
    #Y_train = Y_train[0:10000]

    ############ Train CNN ############
    print ("Start to train cnn...")
    X_train = X_train.reshape((len(X_train), channel, width, height))
    print X_train.shape
    print Y_train.shape

    model = cnn.cnn_model()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, nb_epoch=50, batch_size=128)
    cnn.predict(model, X_test)
