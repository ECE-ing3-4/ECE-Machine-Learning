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

def computeCurrentDD(nb):
    price=close[-1]
    return nb*price

def execute(balance,nbActionsAchetees,action,t):
    price=close[t]
    totalPrice=volumeTransaction*price
    newBalance=balance
    newNbActionsAchetees=nbActionsAchetees

    if action==0 and balance>=totalPrice:#buy
        newBalance=balance -totalPrice
        newNbActionsAchetees=nbActionsAchetees+volumeTransaction
        #assert newBalance>0

    elif action==1 and nbActionsAchetees>=volumeTransaction:#sell
        newBalance= balance+totalPrice
        newNbActionsAchetees=nbActionsAchetees-volumeTransaction
        #assert newNbActionsAchetees>0

    return newBalance,newNbActionsAchetees

def simulatePolicy(p):
    balance=initialBalance
    nbActions=0
    for ordre in range(len(p)):
        balance,nbActions=execute(balance,nbActions,p[ordre],ordre)
        #print("bal",balance,"  nbaction",nbActions)

    return balance, nbActions

#updating Q dans un while
#while pas convergence (convergence = pas d'amelioration)
Q= np.zeros((N,3))

for i in range(10):
    print(np.round(Q,2))
    print()
    #tester Q
    policy=np.argmax(Q,axis=1)
    newBal,newNbAction=simulatePolicy(policy)
    DD=computeCurrentDD(newNbAction) #draw down
    equity=newBal + DD
    reward = equity - initialBalance

    newQ=[]
    for t in range(0,N-1):
        #mettre a jour Q
        newQ.append((1-alpha) * Q[t] + alpha * (reward + gamma * max(Q[t+1])))

    newQ.append([0,0,0])
    Q=newQ

print(np.round(Q,3))
policy=np.argmax(Q,axis=1)


# t=state=date
# a=action = buy/sell/hold
# policy = quelle action faire a chaque instant : P[t]=action
