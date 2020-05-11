import torch.nn as nn
import torch.nn.functional as F

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
        self.fc = nn.Linear(64*64*16*64,64*6)
    
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