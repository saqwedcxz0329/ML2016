# -*- coding: utf-8 -*-
import sys
import os
import math
import numpy as np

class logisticRegression(object):
    def __init__(self):
        self.class1 = []
        self.class0 = []
        self.test_set = []
        self.N1 = 0
        self.N0 = 0
        self.features_dim = 0
        
    def parseData(self, filename):
        train_data = open(os.getcwd() + "/%s" %filename, "r")
        for line in train_data.readlines():
            tmp = line.split(",")
            tmp_train_set = tmp[1:len(tmp)-1]
            if float(tmp[len(tmp)-1]) == 1:
                self.N1 += 1.0
                self.class1.append(map(float,tmp_train_set))
            else:
                self.N0 += 1.0
                self.class0.append(map(float,tmp_train_set))
        self.features_dim = len(self.class1[0])
        train_data.close()
    
    def parseTestData(self, filename):
        test_data = open(os.getcwd() + "/%s" %filename, "r")
        for line in test_data.readlines():
            tmp = line.split(",")
            tmp_test_set = tmp[1:len(tmp)]
            self.test_set.append(map(float,tmp_test_set))
        test_data.close()
    
    def _computeMu(self, class_name):
        mu = np.matrix(np.zeros(self.features_dim))
        for x in class_name:
            x = np.matrix(x)
            mu += x
        mu = mu / len(class_name)
        return mu.getT()
    
    def _computeSigma(self, class_name, mu):
        sigma = np.matrix(np.full((self.features_dim, self.features_dim), 0.0))
        for x in class_name:
            x = np.matrix(x).getT()
            sigma = sigma + ((x-mu).dot((x-mu).getT()))
        sigma = sigma / len(class_name)
        return sigma
    
    def train(self):
        self.u1 = self._computeMu(self.class1)
        self.u0 = self._computeMu(self.class0)
        sigma1 = self._computeSigma(self.class1, self.u1)
        sigma0 = self._computeSigma(self.class0, self.u0)
        total = self.N0 + self.N1
        self.sigma = (self.N1/total) * sigma1 + (self.N0/total) * sigma0
    
    def _computeWT(self):
        return ((self.u1 - self.u0).getT()).dot(self.sigma.getI())
    
    def _computeB(self):
        sigma_inverse = self.sigma.getI()
        return -0.5*((self.u1.getT()).dot(sigma_inverse).dot(self.u1)) + 0.5*((self.u0.getT()).dot(sigma_inverse).dot(self.u0)) + math.log(self.N1/self.N0)
    
    def predict(self):
        predict_file = open("predict.csv", "w")
        predict_file.write("id,label\n")
        wT = self._computeWT()
        b = self._computeB()
        index = 1
        for x in self.test_set:
            x = np.matrix(x).getT()
            z = wT.dot(x) + b
            probability =  1 / (1 + math.exp(-z))
            if probability > 0.5:
                predict_file.write(str(index) + "," + str(1) + "\n")
            else:
                predict_file.write(str(index) + "," + str(0) + "\n")
            index += 1
            
if __name__ == '__main__':
    LR = logisticRegression()
    train_filename = "spam_train.csv"
    LR.parseData(train_filename)
    LR.train()
    test_filename = "spam_test.csv"
    LR.parseTestData(test_filename)
    LR.predict()