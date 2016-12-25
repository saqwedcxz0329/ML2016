import sys
import os
import numpy as np
import cPickle

class test_program(object):
    def __init__(self):
        self.test_set = []

    def parseTestData(self, filename):
        test_data = open("/%s" %filename, "r")
        for line in test_data.readlines():
            tmp = line.split(",")
            tmp_test_set = tmp[1:len(tmp)]
            self.test_set.append(map(float,tmp_test_set))
        test_data.close()
 
    def predict(self, output_file):
        predict_file = open(output_file, "w")
        predict_file.write("id,label\n")
        wT = self._computeWT()
        b = self._computeB()
        index = 1
        for x in self.test_set:
            x = np.matrix(x).getT()
            z = wT.dot(x) + b
            probability =  1 / (1 + np.exp(-z))
            if probability > 0.5:
                predict_file.write(str(index) + "," + str(1) + "\n")
            else:
                predict_file.write(str(index) + "," + str(0) + "\n")
            index += 1
            
    def _computeWT(self):
        try:
            sigma_inverse = self.sigma.getI()
        except:
            sigma_inverse = np.linalg.pinv(self.sigma)
        return ((self.u1 - self.u0).getT()).dot(sigma_inverse)
    
    def _computeB(self):
        try:
            sigma_inverse = self.sigma.getI()
        except:
            sigma_inverse = np.linalg.pinv(self.sigma)
        return -0.5*((self.u1.getT()).dot(sigma_inverse).dot(self.u1)) + 0.5*((self.u0.getT()).dot(sigma_inverse).dot(self.u0)) + np.log(self.N1/self.N0)
    

    def load_model(self, model_name):
        self.u1, self.u0, self.sigma, self.N1, self.N0 = cPickle.load(open(model_name, "r"))

if __name__ == '__main__':
    model_name = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]

    TP = test_program()
    TP.load_model(model_name)
    TP.parseTestData(test_file)
    TP.predict(output_file)
