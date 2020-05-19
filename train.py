import cnn
import load_data 
import hyper_parameters as p
import torch.nn as nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import numpy as np
import load_data
torch.manual_seed(14)

net = cnn.cnn()
# dataset = load_data.image_data()
# dataloader = load_data.DataLoader(dataset,batch_size=p.BATCH_SIZE,shuffle=True)
# test_dataset = load_data.test_dataset()
# test_data_loader = load_data.DataLoader(test_dataset,batch_size=p.BATCH_SIZE,shuffle=False)
# criterion = nn.CrossEntropyLoss()
    
# def val():   
   
#     count = 0
#     for i,(x_test,y_test) in enumerate(dataloader):
#         y = net(x_test)
#         y = y.reshape(6,p.BATCH_SIZE).T
#         y_pred = []
#         for m in range(p.BATCH_SIZE):
#             for n in range(6):
#                 if y[m][n]==y[m].max():
#                     y_pred.append(n)
#         for i in range(p.BATCH_SIZE):
#             if(y_pred[i]==y_test[i].float()):
#                 count = count + 1  
#     return (count/1080)*100



# # def val():
# #     pred = []
# #     for i in range(179):
# #         X = torch.tensor(load_data.X[i+900]).reshape(1,3,64,64).float()
# #         pred.append(net(X))

# #     y_pred = []
# #     for i in range(179):
# #         for j in range(6):
# #             if(pred[i][j]==pred[i].max()):
# #                 y_pred.append(j)

# #     count = 0
# #     for i in range(179):
# #         if(y_pred[i]==torch.tensor(load_data.Y[i+900]).float()):
# #             count = count + 1

# #     return(count/179)

optimizer = optim.Adam(net.parameters(), lr=p.LEARNING_RATE)

# costs = []
# acc = []
# for epoch in range(p.EPOCHS):
#     cost = 0
#     for i,(input,label) in enumerate(dataloader):
#         optimizer.zero_grad()
#         output = net(input)
#         loss = criterion(output.reshape(6,p.BATCH_SIZE).T,label.long())
#         loss.backward()
#         optimizer.step()
#         cost += loss.data
#     val_acc = val()
#     costs.append(cost/1080)
#     acc.append(val_acc)
#     print(f'epoch: {epoch},  loss: {cost/108},  val_acc: {val_acc}')   

# plt.plot(np.squeeze(costs))
# plt.ylabel('cost')
# plt.xlabel('epochs')
# plt.title("Learning rate =" + str(p.LEARNING_RATE))
# plt.show()

# plt.plot(np.squeeze(acc))
# plt.ylabel('accuracy')
# plt.xlabel('epochs')
# plt.title('Learning rate ='+ str(p.LEARNING_RATE))
# plt.show()

# torch.save({
#     'epoch': epoch,
#     'model_state_dict': net.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'loss': loss
#     },p.MODEL_SAVE_PATH)