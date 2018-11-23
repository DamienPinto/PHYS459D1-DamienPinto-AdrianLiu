from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math

def generate_function_cloud(function_type, **specifiers):

	#Value that will hold a value corresponding to the type of function chosen.
	function_value = 0
	print str(function_type), str(function_type)=='linear', str(function_type)=='quadratic', str(function_type)=='exponential', str(function_type)=='simple_periodic'

	#'direction' indicates the if the function is overall increasing in the case of the linear and exponential functions. 
	#For quadratic functions it is used to determine if the function curves upwards or downards. 
	if 'direction' in specifiers and specifiers['direction'] == 'increasing':
		direction = 1
	elif 'direction' in specifiers and specifiers['direction'] == 'decreasing':
		direction = -1
	elif 'direction' not in specifiers:
		direction = np.random.choice([-1, 1])

	#This section is to determine the origin point of the function to be generated. Default is (x_0, y_0) = (0,0).
	if 'y_0' in specifiers and type(specifiers['y_0']) == float:
		y_0 = specifiers['y_0']
	elif 'y_0' not in specifiers:
		y_0 = 0
	if 'x_0' in specifiers and type(specifiers['x_0']) == float:
		x_0 = specifiers['x_0']
	elif 'x_0' not in specifiers:
		x_0 = 0

	#This section is if the user wants the function's origin to be at some random point.
	#The user canset whether they want the origin of the function to have a only positive value with 'positive_y_0' and 'positive_x_0'.
	#They can also chose the range of values that the origin can be chosen in. By default it is [-1. 1] for x_0 and y_0, but with the 
	#values 'scale_y_0' and 'scale_x_0' you can chose the scale of the range. For example for x it will be scale_x_0*[-1, 1].
	if 'random_start' in specifiers and specifiers['random_start'] == True:

		if 'positive_y_0' in specifiers and specifiers['positive_y_0'] == True:
			y_0 = np.random.rand()
		elif 'positive_y_0' in specifiers and specifiers['positive_y_0'] == False or 'positive_y_0' not in specifiers:
			y_0 = np.random.choice([-1,1])*np.random.rand()

		if 'positive_x_0' in specifiers and specifiers['positive_x_0'] == True:
			x_0 = np.random.rand()
		elif 'positive_x_0' in specifiers and specifiers['positive_x_0'] == False or 'positive_x_0' not in specifiers:
			x_0 = np.random.choice([-1,1])*np.random.rand()

		if 'scale_y_0' in specifiers and type(specifiers['scale_y_0']) == float:
			y_0 *= specifiers['scale_y_0']
		if 'scale_x_0' in specifiers and type(specifiers['scale_x_0']) == float:
			x_0 *= specifiers['scale_x_0']

	#The user can specify the the x and y range that the function will generate values in. These ranges will also be the ranges plotted.
	#The default range is [-10, 10]
	if 'x_range' in specifiers and type(specifiers['x_range']) == list and type(specifiers['x_range'][0]) == float:
		x_range = specifiers['x_range']
	elif 'x_range' not in specifiers:
		x_range = np.array([-10.0, 10.0])
	if 'y_range' in specifiers and type(specifiers['y_range']) == list and type(specifiers['y_range'][0]) == float:
		y_range = specifiers['y_range']
	elif 'y_range' not in specifiers:
		y_range = np.array([-10.0, 10.0])

	#The user can define the step size with which to prgress through the range of x-values.
	#The defautl is one 100th of the default range, or the specified range if given.
	if 'step_size' in specifiers and type(specifiers['step_size']) == float:
		step_size = specifiers['step_size']
	elif 'step_size' not in specifiers:
		step_size = (np.amax(x_range)-np.amin(x_range))/(100.0)

	#The spread is used to determine standard deviation of the normal distribution that the program will use to generate noise around
	#each point of the chosen funciton.
	#The default value will be a quarter of the y range. So the program, for example for some point (x,y) and a linear function,
	#will only produce values within the range ax+y_0 +/- (width of y_range)/4
	if 'spread' in specifiers and specifiers['spread'] == float:
		spread = specifiers['spread']
	elif 'spread' not in specifiers or specifiers['spread'] != float:
		spread = (np.amax(y_range)-np.amin(y_range))/4.0

	#This is where the program splits to choose one of the functions specified by 'dunction_type'.
	if str(function_type) == 'linear':
		a = 10*direction*np.random.rand()
		fctn_vals = [a*(x-x_0) + y_0 for x in np.arange(np.amin(x_range), np.amax(x_range)+step_size, step_size)]
		fctn_vals = np.array(fctn_vals)
		# fctn_vals = [_ + np.random.normal(loc=0, scale=((np.amax(fctn_vals)-np.amin(fctn_vals))/8.0)) for _ in fctn_vals]
		fctn_vals = [_ + np.random.normal(loc=0, scale=1) for _ in fctn_vals]
		# for i in range(len(fctn_vals)):
		# 	if np.abs(fctn_vals[i]) > 10:
		# 		fctn_vals[i] = None
		function_value = 0

	if str(function_type) == 'quadratic':
		a = 10*direction*np.random.rand()
		fctn_vals = [a*(x-x_0)**2 + y_0 for x in np.arange(np.amin(x_range), np.amax(x_range)+step_size, step_size)]
		fctn_vals = np.array(fctn_vals)
		fctn_vals = [_ + np.random.normal(loc=0, scale=1) for _ in fctn_vals]
		function_value = 1

	if str(function_type) == 'exponential':
		a = 10.0*direction*np.random.rand()
		b = 2.0*direction*np.random.rand()
		fctn_vals = [a*np.exp(b*(x-x_0)) + y_0 for x in np.arange(np.amin(x_range), np.amax(x_range) + step_size, step_size)]
		fctn_vals = np.array(fctn_vals)
		fctn_vals = [_ + np.random.normal(loc=0, scale=1) for _ in fctn_vals]
		# for i in range(len(fctn_vals)):
		# 	if np.abs(fctn_vals[i]) > 10:
		# 		fctn_vals[i] = None
		# fctn_vals[fctn_vals > 10.0] = None
		function_value = 2

	if str(function_type) == 'simple_periodic':
		a = 8*direction*np.random.rand()
		b = 3*direction*np.random.rand()
		while np.abs(b) < 0.2:
			b = 3*direction*np.random.rand()
		fctn_vals = [a*np.sin(b*(x-x_0)) + y_0 for x in np.arange(np.amin(x_range), np.amax(x_range) + step_size, step_size)]
		fctn_vals = np.array(fctn_vals)
		# fctn_vals = [_ + np.random.normal(loc=0, scale=((np.amax(fctn_vals)-np.amin(fctn_vals))/8.0)) for _ in fctn_vals]
		fctn_vals = [_ + np.random.normal(loc=0, scale=np.abs(a/5.0)) for _ in fctn_vals]
		function_value = 3

	if 'return_type' in specifiers and specifiers['return_type'] == True:
		return fctn_vals, function_value
	else:
		return fctn_vals

if __name__ == '__main__':

	#This was just to test this function.
	x = np.arange(-10,10+0.2,0.2)

	# plt.plot(x, generate_function_cloud('linear', random_start=True), 'bo')
	# plt.plot(x, generate_function_cloud('quadratic', random_start=True), 'bo')
	plt.plot(x, generate_function_cloud('exponential', random_start=True), 'bo')
	# plt.plot(x, generate_function_cloud('simple_periodic', direction='increasing'), 'bo')
	plt.xlim((-10,10))
	plt.ylim((-10,10))
	plt.show()
	