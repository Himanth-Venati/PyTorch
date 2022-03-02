from matplotlib import transforms
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


#set device
device=torch.device('cuda'if torch.cuda.is_available else 'cpu')
#hyperparameters
input_size=784
output=10
learning_rate=0.001
batch_size=64
num_epochs=2

#Load Data
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


train_dataset=datasets.MNIST(root='datset/',train=True,transform=transforms.ToTensor(),download=True)
train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

test_dataset=datasets.MNIST(root='dataset/',train=False,transform=transforms.ToTensor(),download=True)
test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

#Initilize network

model=NN(input=input_size,output=output)

#Loss and optimizer

criterion=nn.CrossEntropyLoss()
optimizer




