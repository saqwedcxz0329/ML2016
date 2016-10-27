import sys
import os
import numpy as np

class test_program(object):
	def __init__(self):
		self.test_set = []
		self.weights = []

	def parseTestData(self, filename):
		test_data = open(os.getcwd() + "/%s" %filename, "r")
		for line in test_data.readlines():
			tmp = line.split(",")
			tmp_test_set = tmp[1:len(tmp)]
			tmp_test_set.insert(0, 1)
			self.test_set.append(map(float,tmp_test_set))
		test_data.close()
	
	def predict(self, output_file):
		predict_file = open(output_file, "w")
		predict_file.write("id,label\n")
		for i in range(len(self.test_set)):
			features = np.matrix(self.test_set[i])
			z = self.weights.dot(features.getT())
			y = self._sigmoid(z)
			if y > 0.5:
				predict_file.write(str(i+1) + "," + str(1) + "\n")
			else:
				predict_file.write(str(i+1) + "," + str(0) + "\n")
	
	def _sigmoid(self, z):
        return 1 / (1+np.exp(-z))

if __name__ == '__main__':
	model_name = sys.argv[1]
    test_file = sys.argv[2]
	output_file = sys.argv[3]

    TP = test_program()
	TP.parseTestData(test_file)
	TP.predict(output_file):
	