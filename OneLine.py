import numpy as np
import random
import matplotlib.pyplot as plt



epslon = 0.9 # 好奇心係數
alpha = 0.1    # 學習率
gamma = 0.9 # 递减值

maxEpisodes = 30
stepsRecord = []


qT=np.zeros((2,11))
print(qT.max(0))

print(qT[:,2],np.argmax(qT[:,1]))
print(qT.max(0).item(4))

def chooseAction(table,state):
    if random.random() > epslon or qT[0,state] == qT[1,state] : 
        return random.randint(0,1)
    else :
        return np.argmax(qT[:,state])
# action=chooseAction(qTable,state)
# print(action)

def executeAction(state,action):
    reward=0
    if action == 0 : state-=1 ; reward = 1
    elif action == 1 : state+=1 ; reward = 0
    if state == 0 : reward = -100
    elif state == 10 : reward = 100
    return reward,state
# print(executeAction(8,1))

def showTerrain(state):
    line="X---------O "
    line=line[:state]+'T'+line[state+1:]
    print('\r{}'.format(line), end='')
# showTerrain(4)

# def RL():
for episode in range(maxEpisodes):
    step=0
    end=False
    state=5
    while not end :
        action=chooseAction(qT,state)
        # action=1
        reward,stateNew=executeAction(state,action)
        qPredict=qT[action,state]
        if stateNew!=0 and stateNew!=10 :
            qTarget=reward + gamma * qT.min(0).item(stateNew)
        else : 
            qTarget=reward
            end = True 
        qT[action,state] += alpha * (qTarget-qPredict)
        state=stateNew
        step+=1
    if state==0 : step*=-1
    stepsRecord.append(step)
    print(episode,state,step)
    
# RL()

plt.figure(figsize=(20,6))
plt.suptitle("S0=5 ; minQ(S',a') ; r(left)=1 ; r(right)=0", fontsize=20)
plt.subplot(1, 2, 1)
X=[0,1,2,3,4,5,6,7,8,9,10]
# 左圖，
plt.xlabel('States')
plt.ylabel('Quality')
plt.plot(X,qT[0,:].tolist(),label='go left')
plt.plot(X,qT[1,:].tolist(),label='go right') 
plt.legend(loc='upper left')
# 右圖
plt.subplot(1, 2, 2)
X=np.arange(1,31)
plt.xlabel('Episodes')
plt.ylabel('Steps')
plt.bar(X,stepsRecord)
plt.show()
