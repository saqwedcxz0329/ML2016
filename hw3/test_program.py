from keras.models import load_model
import numpy as np
import pickle
import sys

channel = 3
width = 32
height = 32

def parseTestData(path):
        print ("Parsing test data...")
        test = pickle.load(open(path, 'rb'))
        X_test = np.array(test['data'], dtype='float32').reshape(len(test['data']), channel, width, height)
        del test
        return X_test/255.

def predict(smodel, X_test):
        print ("Predicting...")
        predict_file = open("predict.csv", "w")
        predict_file.write("ID,class\n")
        output = model.predict(X_test, batch_size=128)
        for i in range(output.shape[0]):
            predict_file.write(str(i) + "," + str(np.argmax(output[i])) + "\n")

data_directory = sys.argv[1]
model_name = sys.argv[2]
output_file = sys.argv[3]
test_data_path = data_directory + '/test.p'
model = load_model(model_name)
X_test = parseTestData(test_data_path)       
predict(model, X_test)
