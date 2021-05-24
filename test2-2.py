import torch 
a=torch.tensor([[1.,2.,3.],[4.,5.,6.]], requires_grad=True)
print(a,a.shape)

x=torch.tensor([2.,4.],requires_grad=True)
m=torch.randn(2,requires_grad=True)
b=torch.randn(1,requires_grad=True)
y=m*x+b 
print(y,m,x,b)
y_known=torch.Tensor([5,9])
loss=(torch.sum(y))
loss.backward()
print(loss,m.grad,x.grad,b.grad)
# print(m)
# m.step()
# print(m)

layer=torch.nn.Linear(2,3)
print(layer)