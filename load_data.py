import hyper_parameters as p 
import  h5py
import torch
from torch.utils.data import DataLoader,Dataset

data = h5py.File(p.FILE_PATH,'r')
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