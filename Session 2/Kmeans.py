import numpy as np
import matplotlib.pyplot as plt

def updateMu(mu,XList,clusters):
    newMu=mu
    counter=np.ones((len(mu),1))
    I=len(XList)

    for i in range(I): #for each point
        point=XList[i]
        cNumber= clusters[i]

        newMu[cNumber]+=point
        counter[cNumber]+=1

    newMu /= counter

    return newMu

#Getting X and Y
data = np.loadtxt(fname = "data_kmeans.txt")

I=len(data)
N=len(data[0])
K=3

plt.scatter(data[:,0], data[:,1], marker="x")

mu=np.random.uniform(np.amin(data),np.amax(data),(K,N))
muOLD=mu*2

while(np.sum(mu-muOLD)!=0):
    clusters=[]
    for x in data:
        closerMu=np.argmin(np.sum(np.square(mu-x),axis=1))
        clusters.append(closerMu)

    muOLD=np.copy(mu)
    mu=updateMu(mu,data,clusters)

plt.scatter(mu[:,0],mu[:,1],marker="*")
plt.show()

X_Test=np.random.uniform(np.amin(data),np.amax(data),(10,2))