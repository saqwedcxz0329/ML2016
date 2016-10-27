# -*- coding: utf-8 -*-
import sys
import os
import random
import math
import numpy as np
import cPickle

class logisticRegression(object):
    def __init__(self):
        self.y_head = []
        self.train_set = []
        self.test_set = []
        self.features_dim = 0
        self.weights = []
        self.past_gradients = []
    
    def parseData(self, filename):
        train_data = open(os.getcwd() + "/%s" %filename, "r")
        for line in train_data.readlines():
            tmp = line.split(",")
            tmp_train_set = tmp[1:len(tmp)-1]
            tmp_train_set.insert(0, 1)
            self.train_set.append(map(float,tmp_train_set))
            self.y_head.append(float(tmp[len(tmp)-1]))
        self.train_set = np.matrix(self.train_set)
        self.y_head = np.matrix(self.y_head)
        self.features_dim = self.train_set[0].size
        train_data.close()
    
    def parseTestData(self, filename):
        test_data = open(os.getcwd() + "/%s" %filename, "r")
        for line in test_data.readlines():
            tmp = line.split(",")
            tmp_test_set = tmp[1:len(tmp)]
            tmp_test_set.insert(0, 1)
            self.test_set.append(map(float,tmp_test_set))
        test_data.close()
            
    def initWeight(self):
        for i in range(0, self.features_dim, 1):
            self.weights.append(random.uniform(0,0.01))
        self.weights = np.matrix(self.weights)
    
    def training(self):
        weights = self.weights
        x = self.train_set
        z = weights.dot(x.getT())
        y_head = self.y_head
        y = self._sigmoid(z)
        gradients = (y - y_head).dot(x) 
        self.past_gradients.append(gradients)
        self._gradientDescent(gradients)
        return self._lossFunction(y, y_head)
    
    def _gradientDescent(self, gradients):
        learn_rate = 0.1
        sigma_past = np.matrix(np.zeros(self.features_dim))
        for past in self.past_gradients:
            sigma_past = sigma_past + np.power(past,2)
            
        for i in range(self.weights.size):
            self.weights[0,i] =  self.weights[0,i] - learn_rate * gradients[0,i] / math.sqrt(sigma_past[0,i])

    def _sigmoid(self, z):
        return 1 / (1+np.exp(-z))
    
    def predict(self):
        predict_file = open("predict.csv", "w")
        predict_file.write("id,label\n")
        for i in range(len(self.test_set)):
            features = np.matrix(self.test_set[i])
            z = self.weights.dot(features.getT())
            y = self._sigmoid(z)
            if y > 0.5:
                predict_file.write(str(i+1) + "," + str(1) + "\n")
            else:
                predict_file.write(str(i+1) + "," + str(0) + "\n")
    
    def _lossFunction(self, y, y_head):
        for i in range(y.size):
            if y[0,i] > 0.5:
                y[0, i] = 1
            else:
                y[0, i] = 0
        loss = np.power((y - y_head), 2)
        summation = 0
        for i in range(loss.size):
            summation += loss[0, i]
        return summation
        
    def output_model(self, model_name):
        cPickle.dump(self.weights, open(model_name, "w"))
        
if __name__ == '__main__':
    train_filename = sys.argv[1]
    model_name = sys.argv[2]

    LR = logisticRegression()
    #train_filename = "spam_train.csv"
    LR.parseData(train_filename)
    LR.initWeight()
    index = 1
    while True:
        error = LR.training()
        print str(index) + " : " + str(error)
        index += 1
        if error <= 280:
            break
    LR.output_model(model_name)
#    test_filename = "spam_test.csv"
#    LR.parseTestData(test_filename)
#    LR.predict()
