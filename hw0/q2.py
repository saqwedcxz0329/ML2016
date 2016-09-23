from PIL import Image
import os
import sys

class q2(object):
	"""docstring for hw0_Q1"""

if __name__ == '__main__':
	img_path = sys.argv[1]
	img = Image.open(os.getcwd() + "/%s" %img_path)
	img2 = img.rotate(180)
	img2.save("ans2.png")
		