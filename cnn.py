import pandas as pd
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import h5py

# USEFUL PARAMETERS

BATCH_SIZE = 1
LEARNING_RATE = 0
EPOCHS = 1
NUM_CLASSES = 6
FILE_PATH = '/home/mk/Desktop/hand gesture recognition/train_signs.h5'

# LOADING DATA

data = h5py.File(FILE_PATH,'r')
X = data.get('train_set_x').value
Y = data.get('train_set_y').value

class image_data(Dataset):
    def __init__(self):
        super(self,image_data).__init__()
        self.x = torch.tensor(X)
        self.y = torch.tensor(Y)
    
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self,index):
        return self.x[index],self.y[index]

dataset = image_data()
dataloader = DataLoader(dataset,batch_size=1,shuffle=True)

# CNN CLASS

class cnn(nn.Module):
    
    def __init__(self):
        super(cnn,self).__init__()
        self.conv1 = nn.Conv2d(3,8,4,stride=1,padding=2,bias=True,padding_mode='zeros')
        self.pool1 = nn.MaxPool2d(8,stride=8,padding=4)
        self.conv2 = nn.Conv2d(8,16,2,stride=1,padding=2,bias=True,padding_mode='zeros')
        self.pool2 = nn.MaxPool2d(4,stride=4,padding=[...])
        self.fc = nn.Linear(40000,6)
    
    def forward(self,x):
        x = nn.ReLU(self.conv1(x))
        x = self.pool1(x)
        x = nn.ReLU(self.conv2(x))
        x = self.pool2(x)
        x = x.reshape(-1)
        x = self.fc(x)
    
net = cnn()

# LOSS FUNCTION AND OPTIMIZER

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE)

# TRAINING 

for epoch in range(EPOCHS):

    for i,(input,label) in enumerate(dataloader):
        
        output = net(input)
        optimizer.zero_grad()
        loss = criterion(output,label)
        loss.backward()
        optimizer.step()

        if(i%100):
            print({i},{loss.data})