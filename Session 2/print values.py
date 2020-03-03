import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

#Getting X and Y
data = np.loadtxt(fname = "data_kmeans.txt")
X=data[:,:1]
Y=data[:,1:]

fig = plt.figure()
ax = fig.gca()

ax.scatter(X, Y)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
plt.show()