import matplotlib.pyplot as plt
from math import sin,cos,pi
import random
import torch
# 設定每層神經元數目
D_in, H1, H2, D_out, = 1, 1, 1, 1 
# 建構序列型模型
def rebuildModel():
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in,H1),
        torch.nn.ReLU(),
        # torch.nn.Linear(H1,H2), 
        # torch.nn.ReLU(),
        torch.nn.Linear(H2, D_out)#,
        # torch.nn.ReLU()
    )
    return model
model=rebuildModel()
# print(model.children())

epochs=10000 # 訓練回合數
learning_rate=0.1 # 學習率
loss_fn = torch.nn.MSELoss() # 損失函數 
# 設定優化器
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

Sx=[]
Sy=[]
# Ay_pred=[]
for i in range(30):
    x=i*2./29
    # x=.3
    y=abs(x-1)+2
    y=2*x
    # if abs(sin(x)) > .8 or cos(x)==0.: 
    #     # print(abs(sin(x)))
    Sx.append([x])
    Sy.append([y])
print(Sx,len(Sx))

Tx=torch.Tensor(Sx) # 將單一數值轉化成矩陣
Ty=torch.Tensor(Sy)

L0=0
L1=1
end=False
step=0
# for i in range(1000):
while not end :
    Ty_pred=model(Tx)
    loss = loss_fn(Ty_pred,Ty)
    L1=loss.data.numpy()
    # print(L1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # print(model.grad)
    # print(L0,L1)
    step+=1
    if step == 1 :
        for param in model.parameters():
            print('param:',param.data.numpy())
            print('grad:',param.grad.data.numpy())
        print('-------------------')

    # if L0-L1 ==0 and step >1000:
    if L0-L1 ==0:   
        # print(L1)
        # for i in model.parameters():
        #     print(i.grad)
        for param in model.parameters():
            print('param:',param.data.numpy())
            print('grad:',param.grad.data.numpy())
        print('rebuild ================================',step)
        # end=True
        model=rebuildModel()
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
        
        step=0
    if L1<0.00001:
        for param in model.parameters():
            print('param:',param.data.numpy())
            print('grad:',param.grad.data.numpy())
        print('success ================================',step)
        # model=rebuildModel()
        # optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
        end = True
        step=0
    L0=L1

# print(model)
# for name, param in model.named_parameters():
#     # if param.requires_grad:
#     print(name, param)

# for param in model.parameters():
#     print('param:',param.data.numpy())
#     print('gradi:',param.grad.data.numpy())




Ax=[] # 建立空串列
Ay=[]
Ay_pred=[]
for i in range(100):
    x=i*4/100-1 # 循序產生輸入值
    y=abs(x-1)+2
    y=2*x
    Tx=torch.Tensor([x])
    Ty=torch.Tensor([y])
    y_pred=model(Tx) # 使用模型算出預測值
    Ax.append(x) # 裝進串列方便畫圖
    Ay.append(y)
    Ay_pred.append(y_pred.item())
plt.plot(Ax,Ay,Ax,Ay_pred,Sx,Sy,"r*") # 畫出預測與目標兩組數據
plt.show()
