import netCDF4 as nc
import torch as th
import numpy as np
from chlorophyll_dataset import ChlorophyllDataset
from tqdm import tqdm
from model import SRCNN
import matplotlib.pyplot as plt

# normalize afai values between 0 and 1
def normalize(afai):
    return (afai - np.min(afai)) / (np.max(afai) - np.min(afai))

# Load the data
batch_size = 1
train_dataset = ChlorophyllDataset('../archivos_prueba/750m_300m/750m/', '../archivos_prueba/750m_300m/300m/')

train_size = int(0.8 * len(train_dataset))
test_size = len(train_dataset) - train_size
train_dataset, test_dataset = th.utils.data.random_split(train_dataset, [train_size, test_size])

train_loader = th.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = th.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Create the model
model = SRCNN(in_channels=2)
criterion = th.nn.MSELoss()
optimizer = th.optim.Adam(model.parameters(), lr=0.001)

model.train()

# Train the model for 10 epochs
epochs = 1000
array_loss = []
for epoch in tqdm(range(epochs)):
    for (low_res, high_res) in train_loader:
        optimizer.zero_grad()
        outputs = model(low_res)
        loss = criterion(outputs, high_res)
        loss.backward()
        optimizer.step()
    array_loss.append(loss.item())

# plot the loss
plt.plot(array_loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('loss_1000_better.png')
plt.clf()

# test the model
model.eval()
test_loss = 0
with th.no_grad():
    for (low_res, high_res) in test_loader:
        outputs = model(low_res)
        test_loss += criterion(outputs, high_res)
    print(f'Test Loss: {test_loss / len(test_loader):.4f}')

# save the model
th.save(model.state_dict(), 'SRCNN_750_300_1000_better.pth')