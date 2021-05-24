import torch

in1=4; ou1=3
in2=3; ou2=2

model=torch.nn.Sequential(
    torch.nn.Linear(in1,ou1),
    torch.nn.ReLU(),
    torch.nn.Linear(in2,ou2),
    torch.nn.ReLU()
)
print(model,model.parameters())
loss_fu = torch.nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(),lr=0.01)
print(loss_fu,optimiser)
