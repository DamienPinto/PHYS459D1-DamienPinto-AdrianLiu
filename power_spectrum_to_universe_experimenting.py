import numpy as np
import math
import os
import sys
import numpy.random
import matplotlib.pyplot as plt
from mock_power_spectrum import make_power_spectrum


#Receive a power spectrum as an array where each value is the std_dev squared of the Gaussian noise distribution that produced the power spectrum but 

#also the value of the Fourier transform at that value's index. The index is k = np.sqrt(k1**2 + k2**2 + k3**2) and is contributed to by any combination

#of values if k1, k2 #& k3 who, when combined in a vector (k1, k2, k3) have a norm of k.

#Want to take that power spectrum and produce a Fourier Space that has a power spectrum similar in statistical characteristics to the one entered. Has to respect F(-k) = [F(k)]* to ensure real output when #translating to actual matter distribution.

#I'm only goping to make cubic Fourier Spaces here.


def make_fourier_space(power_spectrum):

	#For now just making a cubic region, can't assume shape of Fourier space used to generate power spectrum without errors.
	side_length = int(round((len(power_spectrum)/np.sqrt(3))))
	print side_length, len(power_spectrum)/np.sqrt(3)

	#I know we discussed that spaces we use should have dimensions that are powers of two, however this method was used to ensure that all 
	#k values can be used and that a vaue can also be placed at -k.
	#So, for example, for k = 3, we need a space to cell to insert a value at k1 = 3 but also k1 = -3.
	#So I figured twice the k_max + 1 for the origin.
	fourier_universe = np.zeros((2*side_length, 2*side_length, 2*side_length), dtype = complex)
	#Array containing arrays, each of which corresponds to a specific integer value of k.
	shell_register   = [[] for _ in range(len(power_spectrum))]

	#Go through all the coordinates, determine their integer distance from the origin and append that set of coordinates to the shell_register 
	#in the corresponding array.
	for k1 in range(-side_length+1, side_length):
		for k2 in range(-side_length+1, side_length):
			for k3 in range(side_length):
				shell_register[int(round((np.sqrt(k1**2+k2**2+k3**2))))].append((k1, k2, k3))

	shell_sizes = [len(shell_register[_]) for _ in range(len(shell_register))]
	shell_vars_re = []
	shell_vars_im = []

	for i in range(len(shell_register)):

		#Get list of coordinates with same value for their norm from the origin as well as the power spectrum value for that shell.
		norm_val_list = shell_register[i]
		std_dev_sqrd  = power_spectrum[shell_register.index(shell_register[i])]

		#Dr.Adrian's method of centering distributions around 0 and giving them the same standard deviation as the value of the power spectrum
		re_vals = np.random.normal(loc=0, scale=np.sqrt(std_dev_sqrd/2.0), size=len(norm_val_list))
		im_vals = np.random.normal(loc=0, scale=np.sqrt(std_dev_sqrd/2.0), size=len(norm_val_list))
		shell_vars_re.append(np.var(re_vals))
		shell_vars_im.append(np.var(im_vals))

		#My method using a noral distribution to chose values in Fourier space
		# vals = np.random.normal(loc = std_dev_sqrd, scale=10, size=len(norm_val_list))

		#My method but just randomly chosing positive values to place in Fourier space
		# while not(all(val > 0 for val in vals)):

		# 	vals = np.random.normal(loc = std_dev_sqrd, scale=10, size=len(norm_val_list))

		# vals = np.random.rand(len(norm_val_list))
		# vals = [vals[_] + (std_dev_sqrd-np.sum(vals)/len(vals)) for _ in range(len(vals))]

		for j in range(len(norm_val_list)):

			enterred_val = np.complex(re_vals[j], im_vals[j])

			# partitioning = np.random.rand()

			# enterred_val = np.complex(real_sign*np.sqrt(partitioning*vals[i]), complex_sign*np.sqrt((1-partitioning)*vals[i]))


			#We thought of shifting here, but after experimenting with the fast fourier transform (fft) and inverse fast fourier transform (ifft)
			#functions I saw that the output of fft and the input that ifft takes before the shifts is one where the first value is the origin,
			#the next entry being the value for the next smallest frequecy in Fourier, then the next... once it reaches the largest grequecy in 
			#the Fourier space it cycles back to the largest negative frequency, so kind of [0, f1, f2, ... fN, -fN, -fN-1, ... -f1] so if I saw that 
			#correctly then this should work. However, this does lead to having to change the way I populate the fourier space in mock_power_spectrum.py
			#though...
			fourier_universe[norm_val_list[j][0]][norm_val_list[j][1]][norm_val_list[j][2]]	= enterred_val

			fourier_universe[-norm_val_list[j][0]][-norm_val_list[j][1]][-norm_val_list[j][2]] = np.conj(enterred_val)

		# fourier_universe[0,0,:] = [np.real(_) for _ in fourier_universe[0,0,:]]
		# fourier_universe[0,:,0] = [np.real(_) for _ in fourier_universe[0,:,0]]
		# fourier_universe[:,0,0] = [np.real(_) for _ in fourier_universe[:,0,0]]

		fourier_universe[0,0,:] = np.random.normal(loc=0, scale=np.sqrt(std_dev_sqrd), size=len(fourier_universe[0,0,:]))
		fourier_universe[0,:,0] = np.random.normal(loc=0, scale=np.sqrt(std_dev_sqrd), size=len(fourier_universe[0,:,0]))
		fourier_universe[:,0,0] = np.random.normal(loc=0, scale=np.sqrt(std_dev_sqrd), size=len(fourier_universe[:,0,0]))

		

	return fourier_universe, shell_sizes, shell_vars_re, shell_vars_im

 
	




if __name__ == '__main__':
	
	# power_spectrum = sys.argv[1].replace('[', '').replace(']', '').replace('\n', '').split(', ')
	# power_spectrum = [float(i) for i in power_spectrum]
	# power_spectrum = np.array(power_spectrum, dtype = float)

	power_spectrum = np.array([10]*55)

	fourier_universe, shell_sizes, shell_vars_re, shell_vars_im = make_fourier_space(power_spectrum)

	universe = np.array(np.fft.ifftshift(np.fft.ifftn(fourier_universe), axes=(0,1,2)))
	partial_universe = universe[:len(universe), :len(universe), :len(universe)]

	universe_slice = np.array(np.real(partial_universe[32,:len(partial_universe),:len(partial_universe)]), dtype=float)

	second_round_fourier_universe = np.array(np.fft.fftshift(np.fft.fftn(np.fft.fftshift(partial_universe), axes=(0, 1, 2))))
	# second_round_fourier_universe /= np.sqrt(len(second_round_fourier_universe)**3)
	# second_round_fourier_universe = second_round_fourier_universe[int(len(second_round_fourier_universe)/4.0):3*int(len(second_round_fourier_universe)/4.0), int(len(second_round_fourier_universe)/4.0):3*int(len(second_round_fourier_universe)/4.0), int(len(second_round_fourier_universe)/4.0):3*int(len(second_round_fourier_universe)/4.0)]
	# print len(second_round_fourier_universe)
	# print 'Here'

	second_round_power_spectrum, counting_errors = make_power_spectrum(np.fft.ifftshift(second_round_fourier_universe))

	print "Real:", shell_vars_re
	print 'Imaginary:', shell_vars_im

	# second_round_power_spectrum, counting_errors = make_power_spectrum(np.fft.fftshift(fourier_universe[:len(fourier_universe)/2, :len(fourier_universe)/2, :len(fourier_universe)/2]))



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
