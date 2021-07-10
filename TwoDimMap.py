import numpy as np
import random
import matplotlib.pyplot as plt
import torch,copy

epslon = 0.9 # 好奇心係數
alpha = 0.1    # 學習率
gamma = 0.9 # 递减值
learning_rate=0.01

maxEpisodes = 3
stepsRecord = []

qT=np.zeros((4,11,11))
# print(qT.max(0))

# 設定每層神經元數目
D_in, H1, H2, D_out, = 4, 5, 5, 4 
# 建構預測網路
predNN = torch.nn.Sequential(
    torch.nn.Linear(D_in,H1),
    torch.nn.ReLU(),
    torch.nn.Linear(H1,H2), 
    torch.nn.ReLU(),
    torch.nn.Linear(H2, D_out)
)
# 複製目標網路
targNN=copy.deepcopy(predNN)
targNN.load_state_dict(predNN.state_dict())
loss_fn = torch.nn.MSELoss() # 損失函數 
optimizer = torch.optim.Adam(predNN.parameters(),lr=learning_rate)


state=[3,3,5,5]
stateT=torch.Tensor(state)
print(stateT)
print(predNN(stateT).detach().numpy(),targNN(stateT).detach().numpy(),stateT.numpy())

predQ=predNN(stateT).squeeze()[0]
maxQ=torch.max(targNN(stateT))
newValue=1+gamma*maxQ
print(predQ,newValue)
loss=loss_fn(predQ,newValue)
optimizer.zero_grad()
loss.backward()
optimizer.step()
print(predNN(stateT).data.numpy(),targNN(stateT).data.numpy(),stateT.data.numpy())

def chooseAction(aQ):
    if random.random() > epslon : 
        return random.randint(0,3)
    else :
        return np.argmax(aQ)

a=chooseAction(predNN(stateT).detach().numpy())
print(a)

def executeAction(state,action) :
    reward=0
    end=False
    # 動作
    if action==0 : state[1]-=1 # 上
    if action==1 : state[1]+=1 # 下
    if action==2 : state[0]-=1 # 左
    if action==3 : state[0]+=1 # 右
    # 撞牆
    if state[0]==-1 : state[0]=0 ; reward=-1
    if state[0]==11 : state[0]=10; reward=-1
    if state[1]==-1 : state[1]=0 ; reward=-1
    if state[1]==11 : state[1]=10; reward=-1
    # 拿到寶藏
    if state[0]==5 and state[1]==5 :
        reward=100
        end = True
    return reward,state,end

print(executeAction([4,5],3),executeAction([10,5],3))


for episode in range(maxEpisodes):
    step=0
    end=False
    state=[1,1,5,5]
    while not end :
        stateT=torch.Tensor(state)
        actionQualitys=predNN(stateT).detach().numpy()
        action=chooseAction(actionQualitys)
        reward,stateNew,end=executeAction(state,action)
        predQ=predNN(stateT).squeeze()[action]
        if not end :
            maxQ=torch.max(targNN(stateT))
            qTarget=reward+gamma*maxQ
        else : 
            qTarget=torch.Tensor([reward])
            
        # qTarget=torch.tensor([qTarget]).detach()
        # print(predQ,qTarget)
        # print(stateT.numpy(),action)
        loss=loss_fn(predQ,qTarget)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        state=stateNew
        step+=1
        if step % 100 == 0 :
            targNN.load_state_dict(predNN.state_dict())
        if step > 10000 :
            end = True
    # if state==0 : step*=-1
    stepsRecord.append(step)
    print(episode,state,step)
    
x = np.linspace(0,10,11)
y = np.linspace(0,10,11)
X, Y = np.meshgrid(x, y)
# print(X, Y)
Z0=np.zeros((11,11))
Z1=np.zeros((11,11))
Z2=np.zeros((11,11))
Z3=np.zeros((11,11))

for i in range(11):
    for j in range(11):
        state=[i,j,5,5]
        stateT=torch.Tensor(state)
        Q0=predNN(stateT).squeeze()[0]
        Q1=predNN(stateT).squeeze()[1]
        Q2=predNN(stateT).squeeze()[2]
        Q3=predNN(stateT).squeeze()[3]
        Z0[i,j]=Q0
        Z1[i,j]=Q1
        Z2[i,j]=Q2
        Z3[i,j]=Q3

fig = plt.figure(figsize=(20,6))
# ax = plt.axes(projection='3d')
ax = fig.add_subplot(1, 2, 1,projection='3d')
ax.plot_surface(X, Y, Z0, color='red')
ax.plot_surface(X, Y, Z1, color='yellow')
ax.plot_surface(X, Y, Z2, color='green')
ax.plot_surface(X, Y, Z3, color='blue')
ax.set_title('wireframe')
plt.show()
