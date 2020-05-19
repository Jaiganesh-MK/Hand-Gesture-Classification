import hyper_parameters as p 
import  h5py
import torch
from torch.utils.data import DataLoader,Dataset
torch.manual_seed(14)

data = h5py.File(p.TRAIN_FILE_PATH,'r')
X = data[('train_set_x')]
Y = data[('train_set_y')]

test_data = h5py.File(p.TEST_FILE_PATH,'r')
X_test = test_data[('test_set_x')]
Y_test = test_data[('test_set_y')]
print(X_test.shape)
class image_data(Dataset):
    def __init__(self):
        super(image_data,self).__init__()
        self.x = torch.tensor(X).float()
        self.y = torch.tensor(Y).float()
    
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self,index):
        return self.x[index].reshape(3,64,64),self.y[index]

class test_dataset(Dataset):
    def __init__(self):
        super(test_dataset,self).__init__()
        self.x = torch.tensor(X_test).float()
        self.y = torch.tensor(Y_test).float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index].reshape(3,64,64),self.y[index]