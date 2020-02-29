import numpy as np
import matplotlib.pyplot as plt


############# SETTINGS #############
K=10

av=0.07
aw=0.07
aEvolution=0.999

nbEpoch=500
printEpoch=10
showGraph=False


def arrondi(x):
    return (x>0.5).astype(int)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def activation(x):
    return sigmoid(x)

############# INITIALIZING #############
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


##Graph
if showGraph:
    xAxis=[]
    EGraph=[]

    fig = plt.gcf()
    fig.show()
    fig.canvas.draw()
    plt.ylabel('Errors')

############# LEARNING #############
print("Epoch    av        aw        SSE        acc")

for epoch in range(1,nbEpoch+1):
    ##### Forward Propagation #####

    ##Computing Xb
    ones=np.ones((I, 1))
    Xb=np.concatenate((ones, X), axis=1)

    ##Computing Xbb
    Xbb=np.dot(Xb,V)

    ##Computing F
    F=np.apply_along_axis(activation, 0, Xbb)

    ##Computing Fb
    Fb=np.concatenate((ones, F), axis=1)

    ##Computing Fbb
    Fbb=np.dot(Fb,W)

    ##Computing G
    G=np.apply_along_axis(activation, 0, Fbb)
    G2=np.apply_along_axis(arrondi, 0, Fbb)

    ##Computing E
    E=0
    for i in range(0,I):
        for j in range(0,J):
            predicted=G2[i][j]
            target=Y[i][j]
            E+=np.square(predicted-target)
    E/=2

    ##Printing error
    if showGraph:
        xAxis.append(epoch)
        EGraph.append(E)

        plt.plot(xAxis,EGraph)
        fig.canvas.draw()

    if epoch % printEpoch==0:
        # print("Epoch", epoch, end='')
        # print(", av", "%.2f" % av, end='')
        # print(", aw", "%.2f" % aw, end='')
        # print(", SSE :", "%.2f" % E)
        acc=np.sum(G2==Y)/J/I
        acc=int(acc*100)
        print(epoch, "     ", "%.3f" % av , "   ", "%.3f" % aw , "   ", "%.3f" % E, "   ",  acc)


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

if showGraph: plt.show()

acc=np.sum(G2==Y)/J/I
print("accuracy :",int(acc*100),"%")











