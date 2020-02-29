import numpy as np

X=[]
Y=[]
K=8

def sigmoid(x):
    return 1/(1+np.exp(-x))

##Getting X and Y
data = np.loadtxt(fname = "data.txt")
X1,X2,YData=np.hsplit(data, 3)
X=np.concatenate((X1, X2), axis=1)
X=
Y=data[:,-1]

I=len(X)
N=len(X[0])

##Computing Y
Y=[]
YUnique=np.unique(YData)

for y in YData:
    Y.append((YUnique==y[0]).astype(int))

Y=np.asarray(Y)
J=len(Y[0])

##Computing Xb
ones=np.ones((I, 1))
Xb=np.concatenate((ones, X), axis=1)

##Computing Xbb
V=np.random.uniform(-1,1,(N+1,K))
Xbb=np.dot(Xb,V)

##Computing F
F=np.apply_along_axis(sigmoid, 0, Xbb)

##Computing Fb
Fb=np.concatenate((ones, F), axis=1)

##Computing Fbb
W=np.random.uniform(-1,1,(K+1,J))
Fbb=np.dot(Fb,W)

##Computing G
G=np.apply_along_axis(sigmoid, 0, Fbb)

##Computing E
E=0
for i in range(0,I):
    for j in range(0,J):
        predicted=G[i][j]
        target=Y[i][j]
        E+=np.square(predicted-target)
E/=2

##Voila voila
print(E)









