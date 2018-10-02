import numpy as np
import math
import os
import sys
import numpy.random
import matplotlib.pyplot as plt
from mock_power_spectrum import make_power_spectrum
from mock_power_spectrum import round_almost_correctly


#Receive a power spectrum as an array where each value is the std_dev squared of the distribution that produced the power spectrum but also the value of #the Fourier transform at that value's index. The index is k = np.sqrt(k1**2 + k2**2 + k3**2) and is contributed to by any combination of values if k1, k2 #& k3 who, when combined in a vector (k1, k2, k3) have a norm of k

#Want to take that power spectrum and produce a Fourier Space that has that power spectrum. Has to respect F(-k) = [F(k)]* to ensure real output when #translating to actual matter distribution.

#I'm only goping to make cubic Fourier Spaces here.


def make_fourier_space(power_spectrum):

	# print power_spectrum


	
	side_length = int(round_almost_correctly(len(power_spectrum)/np.sqrt(3)))
	print side_length
	fourier_universe = np.zeros((2*side_length+1, 2*side_length+1, 2*side_length+1), dtype = complex)
	#array containing arrays, each of which corresponds to a specific integer value of k
	log = [[] for _ in range(len(power_spectrum))]

	#Go through all the coordinates, determine their integer distance from the origin and append that set of coordinates to the log in the corresponding array.
	for k1 in range(side_length):
		for k2 in range(side_length):
			for k3 in range(side_length):
				log[int(round_almost_correctly(np.sqrt(k1**2+k2**2+k3**2)))].append((k1, k2, k3))

	for i in range(len(log)):

		#get list of coordinates with same value for their norm from the origin
		norm_val_list = log[i]

		std_dev_sqrd  = power_spectrum[log.index(log[i])]

		#Dr.Adrian's method of centering distributions around 0 and giving them the same standard deviation as the value of the power spectrum
		re_vals = np.random.normal(loc=0, scale=np.sqrt(std_dev_sqrd/2.0), size=len(norm_val_list))
		re_vals = [re_vals[_] - np.sum(re_vals)/len(re_vals) for _ in range(len(re_vals))]
		im_vals = np.random.normal(loc=0, scale=np.sqrt(std_dev_sqrd/2.0), size=len(norm_val_list))
		im_vals = [im_vals[_] - np.sum(im_vals)/len(im_vals) for _ in range(len(im_vals))]

		#My method using a noral distribution to chose values in Fourier space
		# vals = np.random.normal(loc = std_dev_sqrd, scale=10, size=len(norm_val_list))

		#My method but just randomly chosing positive values to place in Fourier space
		# while not(all(val > 0 for val in vals)):

		# 	vals = np.random.normal(loc = std_dev_sqrd, scale=10, size=len(norm_val_list))

		# vals = np.random.rand(len(norm_val_list))
		# vals = [vals[_] + (std_dev_sqrd-np.sum(vals)/len(vals)) for _ in range(len(vals))]

		for i in range(len(norm_val_list)):

			real_sign = np.random.choice([-1,1], 1)[0]
			complex_sign = np.random.choice([-1,1], 1)[0]

			enterred_val = np.complex(re_vals[i], im_vals[i])

			# partitioning = np.random.rand()

			# enterred_val = np.complex(real_sign*np.sqrt(partitioning*vals[i]), complex_sign*np.sqrt((1-partitioning)*vals[i]))

			fourier_universe[norm_val_list[i][0]][norm_val_list[i][1]][norm_val_list[i][2]]	= enterred_val

			fourier_universe[-(norm_val_list[i][0]+1)][-(norm_val_list[i][1]+1)][-(norm_val_list[i][2]+1)] = np.conj(enterred_val)


	return fourier_universe

 
	




if __name__ == '__main__':
	
	power_spectrum = sys.argv[1].replace('[', '').replace(']', '').replace('\n', '').split(', ')
	power_spectrum = [float(i) for i in power_spectrum]
	power_spectrum = np.array(power_spectrum, dtype = float)

	fourier_universe = make_fourier_space(power_spectrum)

	universe = np.array(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(fourier_universe), axes=(0,1,2))))

	universe_slice = np.array(np.real(universe[32,:len(universe)/2,:len(universe)/2]), dtype=float)

	second_round_fourier_universe = fourier_universe = np.array(np.fft.fftshift(np.fft.fftn(np.fft.fftshift(universe), axes=(0, 1, 2))))
	second_round_fourier_universe = second_round_fourier_universe[0:int(len(second_round_fourier_universe)/2.0), 0:int(len(second_round_fourier_universe)/2.0), 0:int(len(second_round_fourier_universe)/2.0)]
	print len(second_round_fourier_universe)
	print 'Here'

	second_round_power_spectrum, counting_errors = make_power_spectrum(second_round_fourier_universe)

	# fig = plt.figure()
	# plt.imshow(universe_slice)
	# plt.show()


	k_ubound = 1.0/4.0/math.pi
	k_lbound = 1.0/4.0/math.pi/len(universe[0])

	fig = plt.figure()

	ax = fig.add_subplot(1,2,1)
	ax.set_ylabel('z')
	ax.set_xlabel('y')
	ax.set_title('Universe Slice')
	plt.imshow(universe_slice)

	ax = fig.add_subplot(1,2,2)
	k = range(0, len(second_round_power_spectrum))
	ax.set_ylabel("Power")
	ax.set_xlabel("k")
	ax.set_title("Power Spectrum of Spherically Averaged Gaussian Noise")
	# ax.plot(k, second_round_power_spectrum)
	# print len(k)
	# print len(counting_errors)
	ax.errorbar(k, second_round_power_spectrum, yerr=counting_errors, ecolor='r')
	# ax.xaxis.set_major_formatter(FormatStrFormatter('%0.4f'))
	k_step 	   = float((k_ubound - k_lbound)/len(k))
	print float(k_step)
	tick_marks = np.zeros(len(k))
	tick_marks = ['%0.2f' % float(k_step*i) for i in range(0, len(k), 10)]
	plt.xticks(np.arange(0, len(second_round_power_spectrum), 10), tick_marks, rotation=90)
	plt.show()
