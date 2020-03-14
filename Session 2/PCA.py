import numpy as np
import matplotlib.pyplot as plt

#Getting X and Y
X = np.loadtxt(fname = "data_pca.txt")



I=len(X)
N=len(X[0])
K=3

#computing mu
mu=np.sum(X,axis=0)/I

#computing dataBar
Xb=X-mu

#computing sigma COVARIANCE DE X
sigma=np.zeros((N,N))
for x in Xb:
    sigma += x * x.reshape((N,1))
sigma/=I

#computing the eigenvectors
eigenVects = np.linalg.eig(sigma)
eigenVals = np.linalg.eigvals(sigma)

#sorting eigens
indexMax=np.argmax(eigenVals)
l=eigenVals[indexMax]
u=eigenVects[indexMax]

# Yb=Xb*u
# Y=Yb+mu

Yb = Xb.dot(u)

#print(Y)

plt.scatter(range(7), range(7)*l, marker="*")
plt.scatter(X[:,0], X[:,1], marker="x")
plt.show()


