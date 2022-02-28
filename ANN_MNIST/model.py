import torch
import torch.nn as nn
import torch.nn.functional as F


class NN(nn.Module):
    def __init__(self,input,output):
        super(NN,self).__init__()
        self.input=nn.Linear(input,10)
        self.hidden1=nn.Linear(10,12)
        self.output=nn.Linear(12,output)
        

    def forward(self,x):
        x=F.relu(self.input(x))
        x=F.relu(self.hidden1(x))
        x=self.output(x)
        return x

model=NN(784,10)
x=torch.randn(64,784)
print(model(x).shape)

