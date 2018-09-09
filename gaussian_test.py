import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import bivariate_normal
from mpl_toolkits.mplot3d import Axes3D

universe = np.zeros((80, 80, 100))

#Parameters to set
mu_x = 10
sigma_x = np.sqrt(5)

mu_y = 10
sigma_y = np.sqrt(5)

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1, projection='3d')

#Create grid and multivariate normal
x = np.linspace(0,20,80)
y = np.linspace(0,20,80)
X, Y = np.meshgrid(x,y)
Z = bivariate_normal(X,Y,sigma_x,sigma_y,mu_x,mu_y)

for i in range(80):
	for j in range(80):
		for k in range(int(Z[i, j]/np.amax(Z)*100)):
			
			universe[i][j][k] = 1
print 'Done.'

# x2 = np.linspace(-1, 1, 500)
# y2 = np.linspace(-1, 1, 500)
# X2, Y2 = np.meshgrid(x, y)

fourier_universe = np.fft.fftshift(np.fft.rfftn(universe, axes=(0, 1)))/np.sqrt(2*6400)
# #Make a 3D plot
# surf1 = ax.plot_surface(X, Y, Z,cmap='viridis',linewidth=0)
# ax.set_xlabel('X1 axis')
# ax.set_ylabel('Y1 axis')
# ax.set_zlabel('Z1 axis')

z, x, y = universe.nonzero()
ax.scatter(z, x, y, zdir = 'z', c = 'red')


z, x, y = fourier_universe.nonzero()
# print fourier_universe
print np.amax(fourier_universe)


# ax = fig.add_subplot(1, 2, 2, projection='3d')
surf2 = ax.scatter(z, x, y, zdir='z' , c = 'red')
ax.set_xlabel('X2 axis')
ax.set_ylabel('Y2 axis')
ax.set_zlabel('Z2 axis')

plt.show()