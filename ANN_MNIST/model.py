from os import access
from matplotlib import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim


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
num_epochs=1

#Load Data
train_dataset=datasets.MNIST(root='datset/',train=True,transform=transforms.ToTensor(),download=True)
train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

test_dataset=datasets.MNIST(root='dataset/',train=False,transform=transforms.ToTensor(),download=True)
test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

#Initilize network

model=NN(input=input_size,output=output)

#Loss and optimizer

criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=learning_rate)

#Train Network

for epoch in range(num_epochs):
    for batch_idx,(data,targets) in enumerate(train_loader):
        #Get data to cuda if possible
        data=data.to(device=device)

        #Get to correct shape
        data=data.reshape(data.shape[0],-1)

        #forward
        scores=model(data)
        loss=criterion(scores,targets)

        #backward
        optimizer.zero_grad()
        loss.backward()

        #gradient descent or adam step

        optimizer.step()

#check accuracy on training & test to see how good our model

def check_accuracy(loader,model):
    if loader.dataset.train:
        print('checking accuracy on training data')  
    else:
        print('checking accuracy on test data')    
    num_correct=0
    num_samples=0
    model.eval()

    with torch.no_grad():
            
        for x,y in loader:
            x=x.to(device=device)
            y=y.to(device=device)
            x=x.reshape(x.shape[0],-1)

            scores=model(x)
            _,predictions=scores.max(1)
            num_correct +=(predictions==y).sum()
            num_samples+=predictions.size(0)
        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')    
    model.train()
    

check_accuracy(train_loader,model)
check_accuracy(test_loader,model)






