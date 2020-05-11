import pandas as pd
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import h5py

# USEFUL PARAMETERS

BATCH_SIZE = 1
LEARNING_RATE = 0.1
EPOCHS = 2
NUM_CLASSES = 6
FILE_PATH = '/home/fuhrer/Desktop/Hand-Gesture-Classification/data/train_signs.h5'

# LOADING DATA

data = h5py.File(FILE_PATH,'r')
X = data[('train_set_x')]
Y = data[('train_set_y')]

class image_data(Dataset):
    def __init__(self):
        super(image_data,self).__init__()
        self.x = torch.tensor(X).float()
        self.y = torch.tensor(Y).float()
    
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self,index):
        return self.x[index].reshape(3,64,64),self.y[index]

dataset = image_data()
dataloader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)

# CNN CLASS

def same_padding(input_size, stride, kernel):
    padding = ((input_size-1)*stride + kernel - input_size)/2
    return int(padding)

class cnn(nn.Module):
    
    def __init__(self):
        super(cnn,self).__init__()
        self.conv1 = nn.Conv2d(3,8,4,stride=1,padding=same_padding(64,1,4),bias=True,padding_mode='zeros')
        self.pool1 = nn.MaxPool2d(8,stride=8)
        self.conv2 = nn.Conv2d(8,16,2,stride=1,padding=same_padding(64,1,2),bias=True,padding_mode='zeros')
        self.pool2 = nn.MaxPool2d(4,stride=4)
        self.fc = nn.Linear(63504,6)
    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        pad1 = same_padding(64,8,8)
        x = F.pad(x,(pad1,pad1,pad1,pad1),mode='constant',value=0)
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        pad2 = same_padding(64,4,4)
        x = F.pad(x,(pad2,pad2,pad2,pad2),mode='constant',value=0)
        x = self.pool2(x)
        x = x.reshape(-1)
        x = self.fc(x)
        return x
    
net = cnn()

# LOSS FUNCTION AND OPTIMIZER

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE)

# TRAINING 

for epoch in range(EPOCHS):

    for i,(input,label) in enumerate(dataloader):
        optimizer.zero_grad()
        output = net(input)
        loss = criterion(output.reshape(1,6),label.long())
        loss.backward()
        optimizer.step()

        if(i%100==0):
            print({i},{loss.data})