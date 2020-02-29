import numpy as np

K=25

av=0.04
aw=0.04
aEvolution=0.999

nbEpoch=500
printEpoch=20

def sigmoid(x):
    return 1/(1+np.exp(-x))

############# INITIALIZING #############
EGraph=[]

##Getting X and Y
X=[]
Y=[]
data = np.loadtxt(fname = "data.txt")
X1,X2,YData=np.hsplit(data, 3)
X=np.concatenate((X1, X2), axis=1)
Y=data[:,-1]

I=len(X)
N=len(X[0])

##Formatting Y
Y=[]
YUnique=np.unique(YData)

for y in YData:
    Y.append((YUnique==y[0]).astype(int))

Y=np.asarray(Y)
J=len(Y[0])

##Generating V and W randomly
V=np.random.uniform(-1,1,(N+1,K))
W=np.random.uniform(-1,1,(K+1,J))


############# LEARNING #############

for epoch in range(1,nbEpoch+1):
    ##### Forward Propagation #####

    ##Computing Xb
    ones=np.ones((I, 1))
    Xb=np.concatenate((ones, X), axis=1)

    ##Computing Xbb
    Xbb=np.dot(Xb,V)

    ##Computing F
    F=np.apply_along_axis(sigmoid, 0, Xbb)

    ##Computing Fb
    Fb=np.concatenate((ones, F), axis=1)

    ##Computing Fbb
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

    ##Printing error
    if epoch % printEpoch==0:
        print("Epoch", epoch, ":", "%.2f" % E)


    ##### BACK Propagation #####

    ##Computing the new W
    for k in range(0,K+1):
        for j in range(0,J):
            #for W[k][j]
            dEdWkj=0
            for i in range(0,I):
                dEdWkj += (G[i][j] - Y[i][j]) * G[i][j] * (1 - G[i][j]) *Fb[i][k]

            W[k][j] -= aw * dEdWkj

    ##Computing the new V
    for n in range(0,N+1):
        for k in range(0,K):
            #for V[n][k]
            dEdV=0
            for i in range(0,I):
                for j in range(0,J):
                    dEdV += (G[i][j] - Y[i][j]) * G[i][j] * (1 - G[i][j]) * W[k][j] * F[i][k] * (1 - F[i][k]) * Xb[i][n]

            V[n][k] -= aw * dEdV

    ##Change ac and aw
    av *= aEvolution
    aw *= aEvolution






















