import numpy as np
import matplotlib.pyplot as plt


def arrondi(x):
    return (x>0.5).astype(int)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def activation(x):
    return sigmoid(x)

def fwp(X,V,W):
    I=len(X)
    # Forward Propagation

    #Computing Xb
    ones=np.ones((I, 1))
    Xb=np.concatenate((ones, X), axis=1)

    #Computing Xbb
    Xbb=np.dot(Xb,V)

    #Computing F
    F=np.apply_along_axis(activation, 0, Xbb)

    #Computing Fb
    Fb=np.concatenate((ones, F), axis=1)

    #Computing Fbb
    Fbb=np.dot(Fb,W)

    #Computing G
    G=np.apply_along_axis(activation, 0, Fbb)

    return G,F,Fb,Xb

def bp(V,W,Y,G,F,Fb,Xb,J,K,N,av,aw):
    I=len(Y)
    #BACK Propagation

    #Computing the new W
    for k in range(0,K+1):
        for j in range(0,J):
            #for W[k][j]
            dEdWkj=0
            for i in range(0,I):
                dEdWkj += (G[i][j] - Y[i][j]) * G[i][j] * (1 - G[i][j]) *Fb[i][k]

            W[k][j] -= aw * dEdWkj

    #Computing the new V
    for n in range(0,N+1):
        for k in range(0,K):
            #for V[n][k]
            dEdV=0
            for i in range(0,I):
                for j in range(0,J):
                    dEdV += (G[i][j] - Y[i][j]) * G[i][j] * (1 - G[i][j]) * W[k][j] * F[i][k] * (1 - F[i][k]) * Xb[i][n]

            V[n][k] -= aw * dEdV

    return V,W

def error(Y,Yp,J):
    I=len(Y)
    G2=np.apply_along_axis(arrondi, 0, Yp)
    E=0
    for i in range(0,I):
        for j in range(0,J):
            predicted=G2[i][j]
            target=Y[i][j]
            E+=np.square(predicted-target)
    E/=2
    return E