from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import sys

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
    return X, Y

def dnn_model():
    model = Sequential()
    model.add(Dense(input_dim=8,output_dim=100))
    model.add(Activation('relu'))
    for i in range(9):
        model.add(Dense(100))
        model.add(Activation("relu"))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.summary()
    return model

file_path = sys.argv[1]
model_name = sys.argv[2]
X, Y =parse_data(file_path)
model = dnn_model()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
history_callback = model.fit(X, Y, nb_epoch=1, batch_size=100000)
loss_history = history_callback.history["acc"]
numpy_loss_history = np.array(loss_history)
np.savetxt("loss_history.txt", numpy_loss_history, delimiter = "\n")
model.save(model_name)
