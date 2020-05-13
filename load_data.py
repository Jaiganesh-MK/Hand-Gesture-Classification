import hyper_parameters as p 
import  h5py
import torch
from torch.utils.data import DataLoader,Dataset
torch.manual_seed(14)

data = h5py.File(p.TRAIN_FILE_PATH,'r')
X = data[('train_set_x')]
Y = data[('train_set_y')]

test_data = h5py.File(p.TEST_FILE_PATH,'r')

class image_data(Dataset):
    def __init__(self):
        super(image_data,self).__init__()
        self.x = torch.tensor(X[0:900]).float()
        self.y = torch.tensor(Y[0:900]).float()
    
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self,index):
        return self.x[index].reshape(3,64,64),self.y[index]