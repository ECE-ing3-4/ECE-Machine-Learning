import numpy as np
import matplotlib.pyplot as plt

def distance(x,y):
    sum=0
    N=len(x)
    for n in range(N):
        sum += (x[n]-y[n])**2
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
    newMu=mu
    counter=np.ones((len(mu),1))
    I=len(XList)

    for i in range(I): #for each point
        point=XList[i]
        cNumber= clusters[i]

        newMu[cNumber]+=point
        counter[cNumber]+=1

    newMu /= counter
    # for c in range(len(mu)):#pour chaque mu
    #     if counter[c]!=0:
    #         newMu[cNumber] /= counter[c]

    return newMu


    # for j in range(len(mu)): #for each mu
    #     #m=mu[j]
    #     print("")
    #     nb=0
    #     for i in range(I): # for each point
    #         x=XList[i]
    #         if clusters[i] == j:#add it if ..
    #             nb+=1
    #             sum+=x
    #             print(nb,sum)
    #     if nb !=0:sum/=nb
    #     mu[j]=sum
    # return mu

#Getting X and Y
data = np.loadtxt(fname = "data_kmeans.txt")

I=len(data)
N=len(data[0])
K=3

X1=data[:,:1]
X2=data[:,1:]
fig = plt.figure()
ax = fig.gca()
ax.scatter(X1, X2,marker="x")

mu=np.random.uniform(np.amin(data),np.amax(data),(K,N))

# ax.scatter(mu[:,:1],mu[:,1:],marker="x")
# plt.show()

for i in range(10):
    clusters=[]
    for x in data:
        clusters.append(indexCloser(x,mu))
        #print(x, indexCloser(x,mu))
    print(mu)
    print()
    mu=updateMu(mu,data,clusters)
    print()

ax.scatter(mu[:,:1],mu[:,1:],marker="x")
plt.show()

# a=280
# print(distance(data[a],mu[0]))
# print(distance(data[a],mu[1]))
# print(distance(data[a],mu[2]))