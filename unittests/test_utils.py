"""
Unittests for utils
"""

import numpy as np
import os
import sys


# Enable import from parent directory
sys.path.append(str(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

from utils import *



if __name__ == '__main__':

	# # Test getDistinguishableColors()
	# colors = getDistinguishableColors(25)
	# print(colors)

	# # Test rotX, rotY, rotZ
	# print(rotX(np.pi/2))
	# print(rotX(90, mode = 'degrees'))
	# print(rotY(np.pi/2))
	# print(rotY(90, mode = 'degrees'))
	# print(rotZ(np.pi/2))
	# print(rotZ(90, mode = 'degrees'))


	print('Tests complete!!!')
