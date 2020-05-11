import cnn
import load_data 
import hyper_parameters as p
import torch.nn as nn
import torch.optim as optim

net = cnn.cnn()
dataset = load_data.image_data()
dataloader = load_data.DataLoader(dataset,batch_size=p.BATCH_SIZE,shuffle=True)
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=p.LEARNING_RATE)

for epoch in range(p.EPOCHS):

    for i,(input,label) in enumerate(dataloader):
        optimizer.zero_grad()
        output = net(input)
        loss = criterion(output.reshape(64,6),label.long())
        loss.backward()
        optimizer.step()

        if(i%100==0):
            print({i},{loss.data})
