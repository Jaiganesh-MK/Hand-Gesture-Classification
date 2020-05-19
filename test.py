from train import net
import torch
import hyper_parameters as p
import load_data
from torch.utils.data import DataLoader

checkpoint = torch.load(p.MODEL_SAVE_PATH)
net.load_state_dict(checkpoint['model_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
net.eval()

test_data = load_data.test_dataset()
test_dataloader = DataLoader(test_data,batch_size=p.BATCH_SIZE,shuffle=False)
count = 0
for i,(x_test,y_test) in enumerate(test_dataloader):
    y = net(x_test)
    y = y.reshape(6,p.BATCH_SIZE).T
    y_pred = []
    for m in range(p.BATCH_SIZE):
        for n in range(6):
            if y[m][n]==y[m].max():
                y_pred.append(n)
    for i in range(p.BATCH_SIZE):
        if(y_pred[i]==y_test[i].float()):
            count = count + 1  

print(f'test-accuracy: {(count/120)*100}')