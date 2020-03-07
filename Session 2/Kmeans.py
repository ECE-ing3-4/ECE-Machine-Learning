import numpy as np
import matplotlib.pyplot as plt

def distance(x,y):
    sum=0
    N=len(x)
    for n in range(N):
        sum += (x[n]+y[n])**2
    return sum
    #return sum(np.square(a-b))

def indexCloser(x,mu):
    closerindx=0
    minDist=distance(x,mu[0])
    for i in range(1,len(mu)):
        m=mu[i]
        d=distance(x,m)
        if d<minDist:
            closerindx=i
            minDist=d
    return closerindx

def updateMu(mu,XList,clusters):
    sum=np.zeros(len(XList[0]))
    I=len(XList)
    for j in range(len(mu)): #for each mu
        #m=mu[j]
        print("")
        nb=0
        for i in range(I): # for each point
            x=XList[i]
            if clusters[i] == j:#add it if ..
                nb+=1
                sum+=x
                print(nb,sum)
        if nb !=0:sum/=nb
        mu[j]=sum
    return mu

#Getting X and Y
data = np.loadtxt(fname = "data_kmeans.txt")

I=len(data)
N=len(data[0])
K=3

#mu=np.random.uniform(np.amin(data),np.amax(data),(K,N))

clusters=[]
for x in data:
    clusters.append(indexCloser(x,mu))
    print(x, indexCloser(x,mu))
#print(updateMu(mu,data,clusters))

# a=280
# print(distance(data[a],mu[0]))
# print(distance(data[a],mu[1]))
# print(distance(data[a],mu[2]))