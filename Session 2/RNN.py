import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def activation(x):
    return sigmoid(x)

nbEpoch=8
totalDataSize=20
pTraining=0.75

N=8
K=3
J=1
I=int(totalDataSize*pTraining)

X=np.random.randint(2, size=(totalDataSize,N))
Y=np.sum(X,axis=1)

XTraining,XTesting = X[:I], X[I:]
YTraining,YTesting = Y[:I], Y[I:]


V=np.random.uniform(-1,1,(N+1+K,K))
W=np.random.uniform(-1,1,(K+1,J))
F=np.zeros((I,K))

ones=np.ones((I,1))


for epoch in range(nbEpoch):
    Xb=np.concatenate((ones, XTraining, F), axis=1)
    Xbb=np.dot(Xb,V)
    F=np.apply_along_axis(activation, 0, Xbb)
    Fb=np.concatenate((ones, F), axis=1)
    Fbb=np.dot(Fb,W)
    G=np.apply_along_axis(activation, 0, Fbb)

print(G)