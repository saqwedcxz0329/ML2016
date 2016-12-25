# -*- coding: utf-8 -*-
import sys
import os
import math
import numpy as np
import cPickle

class logisticRegression(object):
    def __init__(self):
        self.class1 = []
        self.class0 = []
        self.test_set = []
        self.N1 = 0
        self.N0 = 0
        self.features_dim = 0
        
    def parseData(self, filename):
        train_data = open("/%s" %filename, "r")
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
            
    def output_model(self, model_name):
        cPickle.dump((self.u1, self.u0, self.sigma, self.N1, self.N0), open(model_name, "w"))
            
if __name__ == '__main__':
    train_filename = sys.argv[1]
    model_name = sys.argv[2]
    LR = logisticRegression()
    LR.parseData(train_filename)
    LR.train()
    LR.output_model(model_name)
