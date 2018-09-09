import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import bivariate_normal
from mpl_toolkits.mplot3d import Axes3D

universe = np.zeros((80, 80, 101))

#Parameters to set
mu_x = 10
sigma_x = np.sqrt(5)

mu_y = 10
sigma_y = np.sqrt(5)


#Create grid and multivariate normal
x = np.linspace(0,20,80)
y = np.linspace(0,20,80)
X, Y = np.meshgrid(x,y)
Z = bivariate_normal(X,Y,sigma_x,sigma_y,mu_x,mu_y)

for i in range(80):
	for j in range(80):

		universe[i][j][int(Z[i, j]/np.amax(Z)*100)] = 1
print 'Done.'


fourier_universe = np.fft.ax.plot_surface(X, Y, Z,cmap='viridis',linewidth=0)

z, x, y = universe.nonzero()


fig = plt.figure()
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.scatter(z, x, y, zdir = 'z', c = 'red')
ax.set_xlabel('X1 axis')
ax.set_ylabel('Y1 axis')
ax.set_zlabel('Z1 axis')


z, x, y = fourier_universe.nonzero()
print np.amax(fourier_universe)


ax = fig.add_subplot(1, 2, 2, projection='3d')
surf2 = ax.scatter(z, x, y, zdir='z' , c = 'red')
ax.set_xlabel('X2 axis')
ax.set_ylabel('Y2 axis')
ax.set_zlabel('Z2 axis')

plt.show()
