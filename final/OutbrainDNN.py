from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

def parse_data(file_path):
    file = open(file_path, "r")
    X = []
    Y = []
    for line in file.readlines():
        tmp = line.split()
        X.append(tmp[:8])
        Y.append(tmp[8].split("\n")[0])
    file.close()
    X = np.array(X, dtype='float32')
    Y = np.array(Y, dtype='uint8')
    #del filevi 
    return X, Y

def dnn_model():
    model = Sequential()
    model.add(Dense(input_dim=8,output_dim=100))
    model.add(Activation('relu'))
    for i in range(9):
        model.add(Dense(100))
        model.add(Activation("relu"))
    model.add(Dense(1))
    model.add(Activation('softmax'))

    model.summary()
    return model
    

X, Y =parse_data("./segmentaa.txt")
print X
print Y
model = dnn_model()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X, Y, nb_epoch=50, batch_size=25000)
