import matplotlib.pyplot as plt
from math import sin,cos,pi
import random
import torch
# 設定每層神經元數目
D_in, H1, H2, D_out, = 1, 20, 20, 1 
# 建構序列型模型
def rebuildModel():
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in,H1),
        torch.nn.ReLU(),
        torch.nn.Linear(H1,H2), 
        torch.nn.ReLU(),
        torch.nn.Linear(H2, D_out)#,
        # torch.nn.ReLU()
    )
    return model
model=rebuildModel()

epochs=10000 # 訓練回合數
learning_rate=0.1 # 學習率
loss_fn = torch.nn.MSELoss() # 損失函數 
# 設定優化器
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9)

Sx=[[0],[2*pi]]
Sy=[[2],[2]]
Sx=[]
Sy=[]
# Ay_pred=[]
for i in range(31):
    x=i*2*pi/30
    y=sin(x)
    # if abs(sin(x)) > .8 or cos(x)==0.: 
    #     # print(abs(sin(x)))
    Sx.append([x])
    Sy.append([y])
print(Sx,len(Sx))

Tx=torch.Tensor(Sx) # 將單一數值轉化成矩陣
Ty=torch.Tensor(Sy)
# Ty_pred=model(Tx)
# print(Tx,Ty_pred)

L0=0
L1=1
end=False
step=0
# for i in range(1000):
while not end :
    Ty_pred=model(Tx)
    loss = loss_fn(Ty_pred,Ty)
    L1=loss.data.numpy()
    print(L1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # print(L0,L1)
    step+=1
    if L0-L1 ==0 :
        print('rebuild')
        model=rebuildModel()
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
        step=0
    if L1<0.0001:
        print('success',step)
        # model=rebuildModel()
        # optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
        end = True
        step=0
    L0=L1
# for i in range(epochs):
#     x=random.random()*2*pi # 在範圍內隨機選定輸入值
#     y=sin(x)+2 # 以函式產出目標值
#     Tx=torch.Tensor([x]) # 將單一數值轉化成矩陣
#     Ty=torch.Tensor([y])
#     y_pred=model(Tx) # 產出與測值
#     loss = loss_fn(y_pred,Ty) # 計算預測與目標的差異
#     optimizer.zero_grad() # 清空梯度
#     loss.backward() # 反向傳播機算梯度
#     optimizer.step() # 依照梯度做優化

Ax=[] # 建立空串列
Ay=[]
Ay_pred=[]
for i in range(100):
    x=i*2*pi/100 # 循序產生輸入值
    y=sin(x)
    Tx=torch.Tensor([x])
    Ty=torch.Tensor([y])
    y_pred=model(Tx) # 使用模型算出預測值
    Ax.append(x) # 裝進串列方便畫圖
    Ay.append(y)
    Ay_pred.append(y_pred.item())
plt.plot(Ax,Ay,Ax,Ay_pred,Sx,Sy,"r*") # 畫出預測與目標兩組數據
plt.show()
