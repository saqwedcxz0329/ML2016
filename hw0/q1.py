from array import array
import sys
import os

class q1(object):
	def __init__(self):
		pass

if __name__ == '__main__':
	request_col = int(sys.argv[1])
	filename = sys.argv[2]
	data = open(os.getcwd() + "/%s" %filename, "r")
	output_file = open("ans1.txt", "w")
	column = []
	flag = False

	for line in data.readlines():
		tmp = line.split(" ")
		for i in range(0, len(tmp)-1, 1):
			if not flag:
				column.append([])
			column[i].append(float(tmp[i+1]))
		flag = True

	target_col = column[request_col]
	target_col.sort()

	for i in range(0, len(target_col), 1):
		output_file.write(str(target_col[i]))
		if i != len(target_col) - 1:
			output_file.write(", ")
	