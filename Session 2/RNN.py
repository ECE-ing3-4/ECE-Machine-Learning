import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def activation(x):
    return sigmoid(x)

nbEpoch=10
totalDataSize=10
pTraining=0.75

N=8
I=int(totalDataSize*pTraining)

X=np.random.randint(2, size=(totalDataSize,N))
Y=np.sum(X,axis=1)
XTraining,XTesting = X[:I], X[I:]
YTraining,YTesting = Y[:I], Y[I:]

ones=np.ones((I,1))

#initialisation
Vx=np.random.uniform(-1,1,N)
Vf=np.random.uniform(-1,1,(1,1))

#forward propagation
# for epoch in range(nbEpoch):
F=np.zeros((I,1))
for t in range(1,N):
    input=XTraining[:,[t]] #column t
    F = np.dot(F,Vf) + np.dot(input,Vx[t])

print(F)
print()

#backpropagation


print(F)