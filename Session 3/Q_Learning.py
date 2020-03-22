import numpy as np
import matplotlib.pyplot as plt

file="bank_of_america.csv"
date = np.loadtxt(fname = file,dtype=str, delimiter=',', skiprows=1,usecols=0)
close = np.loadtxt(fname = file, delimiter=',', skiprows=1,usecols=4)
N=len(date)

plt.scatter(date, close, marker="x")
plt.xlabel("X")
plt.ylabel("Y")
#♥plt.show()


#initial cash flow in USD
initialBalance = 5000
balance=initialBalance

#initial nb of action
nbActionsAchetees = 0
volumeTransaction=5

#Q = portfolio
Q= np.zeros((N,3))
#balance + nbActions*valueAction

#A : 3: hold, buy, sell

#learning rate
alpha = 0.05

#discount factor
gamma = 0.05

def computeDD(t,nb):
    price=close[t]
    return nb*price

def execute(action,t):
    price=close[t]
    totalPrice=volumeTransaction*price
    newBalance=balance
    newNbActionsAchetees=nbActionsAchetees

    if action==1 and balance>=totalPrice:#buy
        newBalance=balance -totalPrice
        newNbActionsAchetees=nbActionsAchetees+volumeTransaction
        #assert newBalance>0

    elif action==2 and nbActionsAchetees>=volumeTransaction:#sell
        newBalance= balance+totalPrice
        newNbActionsAchetees=nbActionsAchetees-volumeTransaction
        #assert newNbActionsAchetees>0

    return newBalance,newNbActionsAchetees


#updating Q dans un while
#while pas convergence (convergence = pas d'amelioration)
for i in range(100):
    for t in range(N-2,-1,-1):

        for a in range(3):
            #print("t",t," a",a)
            newBalance,newNbActionsAchetees=execute(a,t)
            DD=computeDD(t+1,newNbActionsAchetees) #draw down
            equity=newBalance + DD
            reward= equity - initialBalance

            Q[t][a]= (1-alpha) * Q[t][a] + alpha * (reward + gamma * max(Q[t+1]))
            #print("ok")

        #action=np.argmax(Q[t])


print(Q)
policy=np.argmax(Q,axis=1)


# t=state=date
# a=action = buy/sell/hold
# policy = quelle action faire a chaque instant : P[t]=action

