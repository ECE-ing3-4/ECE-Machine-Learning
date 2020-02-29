import numpy as np
import matplotlib.pyplot as plt
import functions as eng

############# SETTINGS #############
K=10

av=0.06
aw=0.06
aEvolution=0.999

nbEpoch=500

printEpoch=50
graphEpoch=10
showGraph=True

############# INITIALIZING #############
#Getting X and Y
data = np.loadtxt(fname = "data.txt")
X=data[:,:2]
YData=data[:,2:]

N=len(X[0])

#Formatting Y
Y=[]
YUnique=np.unique(YData)

for y in YData:
    Y.append((YUnique==y[0]).astype(int))

Y=np.asarray(Y)
J=len(Y[0])


#Graph
if showGraph:
    xAxis=[]
    EGraph=[]

    fig = plt.gcf()
    fig.show()
    fig.canvas.draw()
    plt.ylabel('Errors')


############# LEARNING #############
#Generating V and W randomly
V=np.random.uniform(-1,1,(N+1,K))
W=np.random.uniform(-1,1,(K+1,J))

for epoch in range(1,nbEpoch+1):
    # Forward Propagation
    Yp,F,Fb,Xb = eng.fwp(X,V,W)

    #Computing Error
    E = eng.error(Y,Yp,J)

    #Printing Graph
    if showGraph and epoch % graphEpoch==0:
        xAxis.append(epoch)
        EGraph.append(E)

        plt.plot(xAxis,EGraph)
        fig.canvas.draw()

    #Printing error
    if epoch % printEpoch==0:
        print("epoch", epoch, ":", "%.3f" % E)


    #BACK Propagation
    V,W = eng.bp(V,W,Y,Yp,F,Fb,Xb,J,K,N,av,aw)

    #Change ac and aw
    av *= aEvolution
    aw *= aEvolution

if showGraph: plt.show()

print()
print()

##Printing results
# print(Y)
# print(np.apply_along_axis(eng.arrondi, 0, Yp))
# print()
# print()

##Testing
XTest=[[2,2],[4,4],[4.5,1.5],[1.5,1]]

R=eng.fwp(XTest,V,W)

R=R[0]
R=np.apply_along_axis(eng.arrondi, 0, R)

for i in range(len(XTest)):
    rCateg=R[i]
    r=sum(YUnique*rCateg)
    print(XTest[i], " \t", rCateg, " \t", r)


