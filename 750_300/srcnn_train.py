import netCDF4 as nc
import torch as th
import numpy as np
from chlorophyll_dataset import ChlorophyllDataset
from image_aligner import ImageAligner
from tqdm import tqdm
from model import SRCNN
import matplotlib.pyplot as plt
import torch.cuda.amp as amp

aligner = ImageAligner('../archivos_prueba/750m_300m/750m/', '../archivos_prueba/750m_300m/300m/')
aligner.align_images()

# Load the data
batch_size = 1
train_dataset = ChlorophyllDataset('../archivos_prueba/750m_300m/750m/', '../archivos_prueba/750m_300m/300m/aligned/')

train_size = int(0.8 * len(train_dataset))
test_size = len(train_dataset) - train_size
train_dataset, test_dataset = th.utils.data.random_split(train_dataset, [train_size, test_size])

train_loader = th.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = th.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Create the model
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
model = SRCNN(in_channels=2).to(device)
criterion = th.nn.MSELoss().to(device)
optimizer = th.optim.Adam(model.parameters(), lr=0.0001)

model.train()

# Train the model for 10 epochs
epochs = 1200
array_loss = []
scaler = amp.GradScaler()
for epoch in tqdm(range(epochs)):
    for (low_res, high_res) in train_loader:
        low_res, high_res = low_res.to(device), high_res.unsqueeze(1).to(device)
        optimizer.zero_grad()
        with amp.autocast():
            outputs = model(low_res)
            loss = criterion(outputs, high_res)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # Free up GPU memory
        del low_res, high_res, outputs
        th.cuda.empty_cache()
    array_loss.append(loss.item())  # move this line outside the with block


# plot the final loss
plt.plot(array_loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('loss.png')
plt.clf()

# test the model
model.eval()
test_loss = 0
with th.no_grad():
    for (low_res, high_res) in test_loader:
        low_res, high_res = low_res.to(device), high_res.to(device)
        with amp.autocast():
            outputs = model(low_res)
            test_loss += criterion(outputs, high_res)
        # Free up GPU memory
        del low_res, high_res, outputs
        th.cuda.empty_cache()
    print(f'Test Loss: {test_loss / len(test_loader):.4f}')
    # Free up GPU memory
    del test_loss
    th.cuda.empty_cache()

# save the model
th.save(model.state_dict(), 'model.pth')
