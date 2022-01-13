from torch import nn
#from torchsummary import summary
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

'''class MLP_1_hidden(nn.Module):
    def __init__(self, num_features,num_hidden_1,labels_output=1):
        super().__init__()
        self.cf1 = nn.Linear(num_features, num_hidden_1)
        self.cf2 = nn.Linear(num_hidden_1, labels_output)
    def forward(self, x):
        x = self.cf1(x)
        x = F.relu(x)
        x = self.cf2(x)
        x = F.softmax(x, dim=1)
        return x'''


class MLP(nn.Module):
    def __init__(self,input):
        super(MLP, self).__init__()
        self.cf1=nn.Linear(input,5)
        self.cf2=nn.Linear(5,10)
        self.output1=nn.Linear(10,1)

    def forward(self,s):
        x1=self.cf1(s)
        x1=F.relu(x1)
        x1=F.relu(self.cf2(x1)) 
        x1=self.output1(x1)
        return x1

        

        
       

        

#model = MLP(8,1, hidden_sizes=[7,4])

#print( summary(model,(8,)) )

# YOUR CODE HERE
