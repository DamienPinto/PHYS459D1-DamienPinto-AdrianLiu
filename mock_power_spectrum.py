import math
import scipy.constants as sciCst
import os
import sys
import numpy.random
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np


def make_power_spectrum(fourier_universe):
	
	#The biggest distance from the origin is the one ant the furthest corner so the biggest k would be the norma at the max length in each dimension.
	k_max = round(float(math.sqrt((len(fourier_universe)/2.0)**2 + (len(fourier_universe[0])/2.0)**2 + (len(fourier_universe[0][0])/2.0)**2)))
	# print k_max
	#Make the power spectrum have that many points. The power_spectrum_log keeps a log gives all the values of k an array and keeps track of all the
	#values in the Fourier transformed universe with that value of k.
	power_spectrum_log = [[] for _ in range(int(k_max))]
	power_spectrum 	   = [0 for _ in range(int(k_max))]
	counting_errors	   = [0 for _ in range(int(k_max))]

	# print -len(fourier_universe)/2


	#Cycle through all the points in the Fourier transformed universe...
	#Since what is given to the make_power_spectrum function is the Fourier space *after* the shifts, then the central frequency should be the DC term,
	#the first should be the most negative frequency, and the last should be the largest positive frequncy.
	for k1 in range(-len(fourier_universe)/2, len(fourier_universe)/2-1):
		for k2 in range(-len(fourier_universe[k1])/2, len(fourier_universe[k1])/2-1):
			for k3 in range(-len(fourier_universe[k1][k2])/2, len(fourier_universe[k1][k2])/2-1):
				# print (k1+len(fourier_universe)/2, k2+len(fourier_universe[k1])/2, k3/len(fourier_universe[k1][k2])/2)
				#...determine it's distance from the origin, its k...
				k = int(round(float(math.sqrt(np.abs(k1)**2 + np.abs(k2)**2 + np.abs(k3)**2))))
				# print k, len(power_spectrum), k1+len(fourier_universe)/2, len(fourier_universe), k2+len(fourier_universe)/2, len(fourier_universe[k1]), k2+len(fourier_universe)/2, len(fourier_universe[k1][k2])
				#...append the norm squared of that value to the correct array in the power_spectrum_log.
				power_spectrum_log[k-1].append(float(np.abs(fourier_universe[k1+len(fourier_universe)/2][k2+len(fourier_universe[k1])/2][k3+len(fourier_universe[k1][k2])/2])**2))


	#For each value of k, average over the values of the points with that k and place the result in the power spectrum,
	for i in range(len(power_spectrum_log)):
		if len(power_spectrum_log[i]) > 0:
			power_spectrum[i]  = float(np.sum(power_spectrum_log[i])/len(power_spectrum_log[i]))
		else:
			power_spectrum[i]  = 0

		counting_errors[i] = power_spectrum[i]*float(2/np.sqrt(len(power_spectrum_log[i])))




	#Normalize the power spectrum by deviding by the total number of points.
	# power_spectrum = np.array(power_spectrum)
	# power_spectrum /= len(fourier_universe)**3


	#Power spectrum should be flat, and it's value should be the standard deviation of the Gaussian distribution used to produce the initial
	#universe/matter distribution but squared so postig the square root should return just (approximately) the standard deviation
	print np.sqrt(float(np.sum(power_spectrum)/len(power_spectrum)))

	return power_spectrum, counting_errors


if __name__ == '__main__':

	if str(sys.argv[1]) == "gaussian":
		#Section for entry from command line or calling from other file, just annoying to have to enter that every time when testing.
		# step_power = int(sys.argv[2])
		dx 		 = float(sys.argv[2])
		dy		 = float(sys.argv[3])
		dz		 = float(sys.argv[4])

		xmax	 = float(sys.argv[5])
		ymax 	 = float(sys.argv[6])
		zmax 	 = float(sys.argv[7])


		#Get max number of steps in each spatial dimension.
		#Formula returns closest power of 2 just under the number of steps there should be in each direction (determined by i_max/di).
		Nx		 = (1<<(int(xmax/dx)).bit_length())/2
		Ny 		 = (1<<(int(ymax/dx)).bit_length())/2
		Nz 		 = (1<<(int(zmax/dz)).bit_length())/2

		k_lbound = float(1.0/4.0/math.pi/max([xmax, ymax, zmax]))
		k_ubound = float(1.0/4.0/math.pi/min([dx,dy,dz]))

		mean 	 = float(sys.argv[8])
		std_dev  = float(sys.argv[9])

		universe = numpy.random.normal(loc=mean, scale=std_dev, size=(Nx, Ny, Nz))
		universe_slice = universe[0,:,:]

	fourier_universe = np.array(np.fft.fftshift(np.fft.fftn(np.fft.fftshift(universe), axes=(0, 1, 2))))
	#Normalize the space.
	fourier_universe /= np.sqrt(len(fourier_universe)**3)
	
	power_spectrum,	counting_errors = make_power_spectrum(fourier_universe)
	print power_spectrum
	print counting_errors
	fig = plt.figure()

	ax = fig.add_subplot(1,2,1)
	ax.set_ylabel('z')
	ax.set_xlabel('y')
	ax.set_title('Universe Slice')
	plt.imshow(universe_slice)

	ax = fig.add_subplot(1,2,2)
	k = range(0, len(power_spectrum))
	ax.set_ylabel("Power")
	ax.set_xlabel("k")
	ax.set_title("Power Spectrum of Spherically Averaged Gaussian Noise")
	# ax.plot(k, power_spectrum)
	# print len(k)
	# print len(counting_errors)
	ax.errorbar(k, power_spectrum, yerr=counting_errors, ecolor='r')
	# ax.xaxis.set_major_formatter(FormatStrFormatter('%0.4f'))
	k_step 	   = float((k_ubound - k_lbound)/len(k))
	# print float(k_step)
	tick_marks = np.zeros(len(k))
	tick_marks = ['%0.2f' % float(k_step*i) for i in range(0, len(k), len(k)/10)]
	plt.xticks(np.arange(0, len(power_spectrum), 10), tick_marks, rotation=90)
	plt.show()