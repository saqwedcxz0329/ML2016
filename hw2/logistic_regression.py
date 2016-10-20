# -*- coding: utf-8 -*-
import sys
import os
import random
import math

class logisticRegression(object):
    def __init__(self):
        self.y_head = []
        self.train_set = []
        self.features_dim = 0
        self.weights = []
    
    def parseData(self, filename):
        train_data = open(os.getcwd() + "/%s" %filename, "r")
        for line in train_data.readlines():
            tmp = line.split(",")
            tmp_train_set = tmp[1:len(tmp)-1]
            tmp_train_set.insert(0, 1)
            self.train_set.append(tmp_train_set)
            self.y_head.append(tmp[len(tmp)-1])
        self.features_dim = len(self.train_set[0])
            
    def initWeight(self):
        for i in range(0, self.features_dim, 1):
            self.weights.append(random.uniform(0,1))
    
    def training(self):
#        print self.weights
        loss = 0
        gradients = []
        self._initGradients(gradients)
        for i in range(self.features_dim):
            features = self.train_set[i]
            y = self._computeY(features)
            loss = loss + self._crossEntropy(y, self.y_head[i])
            self._computeGradients(features, gradients, y, self.y_head[i])
        
        self._gradientDescent(gradients)
#        print self.weights
        return loss
    
    def _computeY(self, features):
        summation = 0
        for i in range(self.features_dim):
            summation = summation + self.weights[i] * float(features[i])
        return 1 / (1+math.exp(-summation))
    
    def _crossEntropy(self, y, y_head):
        if y == 0 or 1-y == 0:
            return 0
        return -(float(y_head)*math.log(y)+ (1-float(y_head))*math.log(1-y))

    def _computeGradients(self, features, gradients, y, y_head):
        for i in range(self.features_dim):
            gradients[i] = gradients[i] + (float(y_head) - y) * float(features[i])
    
    def _initGradients(self, gradients):
        for i in range(self.features_dim):
            gradients.append(0)
            
    def _gradientDescent(self, gradients):
        learn_rate = 0.00001
        for i in range(self.features_dim):
            self.weights[i] =  self.weights[i] + learn_rate * gradients[i]
        
if __name__ == '__main__':
    LR = logisticRegression()
    filename = "spam_train.csv"
    LR.parseData(filename)
    LR.initWeight()
#    while True:
    print LR.training()
    #output_file = open("ans1.txt", "w")