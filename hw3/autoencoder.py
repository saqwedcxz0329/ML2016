from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import np_utils
from scipy.misc import toimage
import numpy as np
import matplotlib.pyplot as plt
import cPickle

def parseLabelData(path):
        print "Parsing label data..."
        all_label = cPickle.load(open(path, 'rb'))
        X_train = []
        Y_train = []
        for i in range(len(all_label)):
            for j in range(len(all_label[i])):
                img = np.array(all_label[i][j], dtype='float32')
                X_train.append(img.reshape(3, 32, 32))
                Y_train.append(i)
                
        X_train = np.array(X_train, dtype='float32')
        Y_train = np.array(Y_train, dtype='uint8')
        Y_train = np_utils.to_categorical(Y_train, 10)
        return X_train, Y_train


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

if __name__ == '__main__':
    x_train, y_train = parseLabelData('../../data/all_label.p')
    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_train

    noise_factor = 0.5
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 

    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = x_train_noisy

    print x_train.shape
    print x_test.shape
    print x_train_noisy.shape
    print x_test_noisy.shape
    print x_test
    ae = AutoEncoder(3,32,32,10)
    autoencoder, encoder = ae.autoencoder_model()
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(x_train_noisy, x_train,
                    nb_epoch=50,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test_noisy, x_test))
    
    #encoded_imgs = encoder.predict(x_test)

    encoded_input = Input(shape=(256,))
    decoder_layer = autoencoder.layers[-3]
    decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
    # encode and decode some digits
    # note that we take them from the *test* set
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)

    encoded_input = Input(shape=(512,))
    decoder_layer = autoencoder.layers[-2]
    decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
    decoded_imgs = decoder.predict(decoded_imgs)

    encoded_input = Input(shape=(1024,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
    decoded_imgs = decoder.predict(decoded_imgs)
    print decoded_imgs[0]

    n = 10  # how many digits we will display
    for i in range(n):
        toimage(x_test_noisy[i].reshape(3, 32, 32)).save("test_%d.jpg" %i)
        toimage(decoded_imgs[i].reshape(3, 32, 32)).save("decode_%d.jpg" %i)
