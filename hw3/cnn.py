from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from keras.utils.layer_utils import layer_from_config
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.models import load_model
import numpy as np
import pickle
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
        encoding_dim = 128
        input_img = Input(shape=(input_dim,))

        #encoded = Dense(1024, activation='relu')(input_img)
        encoded = Dense(512, activation='relu')(input_img)
        encoded = Dense(256, activation='relu')(input_img)
        encoded = Dense(encoding_dim, activation='relu')(encoded)

        decoded = Dense(256, activation='relu')(encoded)
        decoded = Dense(512, activation='relu')(encoded)
        #decoded = Dense(1024, activation='relu')(decoded)
        decoded = Dense(input_dim, activation='sigmoid')(decoded)

        autoencoder = Model(input=input_img, output=decoded)
        encoder = Model(input=input_img, output=encoded)
        return autoencoder, encoder

    def clustering(self, encoder, X_train, Y_train):
        print "Clustering..."
        encoded_imgs = encoder.predict(X_train)
        for i in range(5000, X_train.shape[0]):
            min_distance = sys.maxint
            belong_class = -1
            for j in range(10):
                # Find the datas belong to class j
                class_index = np.where(Y_train == j)[0]
                # Compute the cluster's centroid
                centroid = np.mean(encoded_imgs[class_index], axis=0)
                # Find the closest cluster
                euclidean_distance = np.linalg.norm(encoded_imgs[i] - centroid)
                if euclidean_distance < min_distance:
                    min_distance = euclidean_distance
                    belong_class = j
            Y_train[i] = belong_class
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
        print ("Start to train CNN...")
        model = Sequential()
        model.add(Convolution2D(25, 3, 3, input_shape=(self.channel, self.width, self.height)))
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

    def predict(self, model, X_test):
        print ("Predicting...")
        predict_file = open("predict.csv", "w")
        predict_file.write("ID,class\n")
        output = model.predict(X_test, batch_size=128)
        for i in range(output.shape[0]):
            predict_file.write(str(i) + "," + str(np.argmax(output[i])) + "\n")


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
    ## Read data
    X_label, Y_label = data.parseLabelData(label_data_path)
    X_unlabel = data.parseUnlabelData(unlabel_data_path)
    X_test = data.parseTestData(test_data_path)
    
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
    """
    for i in range(X_train.shape[0]):
        #print X_train.shape[1]
        #print X_train[i].shape
        X_train[i] = X_train[i] + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape[1:])
    """
    X_train = np.clip(X_train, 0., 1.)
    X_train.astype('float32')
    Y_train.astype('uint8')
    print (X_train.shape)
    print (Y_train.shape)
     
    #gc.collect()
    ## Training
    ae = AutoEncoder(3,32,32,10) #(channel, width, height, class_num)
    autoencoder, encoder = ae.autoencoder_model()
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_train, X_train,
                    nb_epoch=50,
                    batch_size=128,
                    shuffle=True)
    ## Clustering
    Y_train = ae.clustering(encoder, X_train, Y_train)
    Y_train = np_utils.to_categorical(Y_train, 10)

    ############ Train CNN ############
    print ("Start to train cnn...")
    X_train = X_train.reshape((len(X_train), channel, width, height))
    print X_train.shape
    print Y_train.shape

    model = cnn.cnn_model()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, nb_epoch=50, batch_size=128)
    model.save(model_name)  # creates a HDF5 file 'my_model.h5'
    del model  # deletes the existing model
