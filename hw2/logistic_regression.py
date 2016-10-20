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
        self.past_gradients = []
    
    def parseData(self, filename):
        train_data = open(os.getcwd() + "/%s" %filename, "r")
        for line in train_data.readlines():
            tmp = line.split(",")
            tmp_train_set = tmp[1:len(tmp)-1]
            tmp_train_set.insert(0, 1)
            self.train_set.append(map(float,tmp_train_set))
            self.y_head.append(float(tmp[len(tmp)-1]))
        self.features_dim = len(self.train_set[0])
            
    def initWeight(self):
        for i in range(0, self.features_dim, 1):
            self.weights.append(random.uniform(0,1))
    
    def training(self):
        loss = 0
        gradients = []
        self._initList(gradients)
        for i in range(len(self.train_set)):
            features = self.train_set[i]
            y = self._computeY(features)
            self._computeGradients(features, gradients, y, self.y_head[i])
            loss = loss + self._crossEntropy(y, self.y_head[i])
            
        self.past_gradients.append(list(gradients))
        self._gradientDescent(gradients)
        return loss
    
    def _computeY(self, features):
        try:
            if len(self.weights) != len(features):
                raise Exception("not equal")
        except:
            pass
        summation = 0
        for i in range(self.features_dim):
            summation = summation + self.weights[i] * features[i]
        return 1 / (1+math.exp(-summation))
    
    def _crossEntropy(self, y, y_head):
#        return (y_head-y) * (y_head-y)
        if y == 0 or 1-y == 0:
            return 0
        return -(y_head*math.log(y)+ (1-y_head)*math.log(1-y))

    def _computeGradients(self, features, gradients, y, y_head):
        try:
            if len(gradients) != len(features):
                raise Exception("not equal")
        except:
            pass
        for i in range(self.features_dim):
            gradients[i] = gradients[i] + (y_head - y) * features[i]
    
    def _initList(self, gradients):
        for i in range(self.features_dim):
            gradients.append(0)
            
    def _gradientDescent(self, gradients):
        learn_rate = 0.1
        sigma_past = []
        self._initList(sigma_past)
        for i in range(len(self.past_gradients)):
            for j in range(self.features_dim):
                sigma_past[j] = sigma_past[j] + self.past_gradients[i][j] * self.past_gradients[i][j]
            
        for i in range(self.features_dim):
            self.weights[i] =  self.weights[i] + learn_rate * gradients[i] / math.sqrt(sigma_past[i])
        
if __name__ == '__main__':
    try:
        if 1<0
#    LR = logisticRegression()
#    filename = "spam_train.csv"
#    LR.parseData(filename)
#    LR.initWeight()
#    while True:
#        print LR.training()
    #output_file = open("ans1.txt", "w")