import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import bivariate_normal
from mpl_toolkits.mplot3d import Axes3D

step_power = 6
min_val = 2
spread = 15

universe = np.zeros((2**step_power, 2**step_power, 101))

#Parameters to set
mu_x = 0
sigma_x = np.sqrt(spread)

mu_y = 0
sigma_y = np.sqrt(spread)

#Create grid and multivariate normal
x    = range(2**step_power)
x    = [float(val - 2**(step_power-1)) for val in x]
x 	 = np.array(x, dtype = float)
y    = range(2**step_power)
y    = [float(val - 2**(step_power-1)) for val in y]
y 	 = np.array(y, dtype = float)
X, Y = np.meshgrid(x,y)
Z 	 = bivariate_normal(X,Y,sigma_x,sigma_y,mu_x,mu_y)
# Z    = np.random.normal(loc=0, scale=10, size=X.shape)
# Z[32,:] += 50
# Z[:,32] += 50

# print np.amax(Z)

#3D Scatter Plots
# for i in range(2**step_power):
# 	for j in range(2**step_power):
# 		# print 'Here1'
# 		universe[i][j][int(Z[i, j]/np.amax(Z)*100)] = 1


# universe = np.fft.fftshift(universe)

# fourier_universe = np.fft.fftn(universe, axes=(0, 1, 2))/np.sqrt(2*2**step_power**2)

# fourier_universe = np.fft.fftshift(fourier_universe)



# z, x, y = universe.nonzero()

# fig = plt.figure()
# ax = fig.add_subplot(1, 2, 1, projection='3d')
# ax.scatter(z, x, y, zdir = 'z', c = 'red')
# ax.set_xlabel('X1 axis')
# ax.set_ylabel('Y1 axis')
# ax.set_zlabel('Z1 axis')

# z, x, y = (fourier_universe>0.0004).nonzero()
# # print np.amax(fourier_universe)
# # print fourier_universe.shape

# ax = fig.add_subplot(1, 2, 2, projection='3d')
# surf2 = ax.scatter(z, x, y, zdir='z' , c = 'red')
# ax.set_xlabel('X2 axis')
# ax.set_ylabel('Y2 axis')
# ax.set_zlabel('Z2 axis')

# plt.show()

fig = plt.figure()
ax = fig.add_subplot(1,2,1)
ax.set_xlabel('X1 axis')
ax.set_ylabel('Y1 axis')
plt.imshow(Z)

FZ = np.fft.ifftshift((np.fft.fftn(np.fft.fftshift(Z), axes=(0,1)))/np.sqrt(2*2**step_power**2))
FZ = np.array(FZ, dtype = float)
# FZ = [np.sqrt(np.abs(val))**2 for val in FZ]
# print FZ
ax = fig.add_subplot(1,2,2)
ax.set_xlabel('X2 axis')
ax.set_ylabel('Y2 axis')
plt.imshow(FZ)
plt.show()