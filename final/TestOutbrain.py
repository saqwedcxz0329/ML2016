from keras.models import load_model
import numpy as np
import sys

test_features_data = sys.argv[1]
test_origin_data = sys.argv[2]
model_name = sys.argv[3]
output_file = sys.argv[4]

def readTestData(path):
    file = open(path, 'r')
    Dispaly_id = []
    Ad_id = []
    for line in file.readlines():
        tmp = line.split(',')
        Dispaly_id.append(tmp[0])
        Ad_id.append(tmp[1].split('\n')[0])
    file.close()
    return Dispaly_id, Ad_id


def parseTestData(path):
    print ("Parsing test data...")
    file = open(path, "r")
    X = []
    for line in file.readlines():
        tmp = line.split()
        X.append(tmp[1:])
    file.close()
    X = np.array(X, dtype='float32')
    return X

def predict(smodel, X_test, Dispaly_id, Ad_id):
    print ("Predicting...")
    predict_file = open(output_file, "w")
    predict_file.write("display_id,ad_id\n")
    output = model.predict(X_test, batch_size=128)
    predict_set = {}
    print ("Grouping...")
    for index, label in enumerate(output):
        print ("=========%d=========" %index)
        display_id = Dispaly_id[index]
        ad_id = Ad_id[index]
        if predict_set.has_key(display_id):
            predict_set[display_id][ad_id] = label[0]
        else:
            predict_set[display_id] = {}
            predict_set[display_id][ad_id] = label[0]
    print predict_set

model = load_model(model_name)
Dispaly_id, Ad_id = readTestData(test_origin_data)
X_test = parseTestData(test_features_data)       
predict(model, X_test, Dispaly_id, Ad_id)
