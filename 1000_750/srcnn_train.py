import torch as th
import numpy as np
from chlorophyll_dataset import ChlorophyllDataset
from tqdm import tqdm
from model import SRCNN
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='SRCNN Training')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--epochs', default=1000, type=int, help='Number of epochs')
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
parser.add_argument('--in_channels', default=2, type=int, help='Number of input channels')
parser.add_argument('--device', default='cuda', type=str, help='Device to use (cuda or cpu)')
parser.add_argument('--save_model', default='model.pth', type=str, help='Path to save the trained model')
parser.add_argument('--low_res_path', default='../archivos_prueba/1km_750m/1km/', type=str, help='Path to the low resolution images')
parser.add_argument('--high_res_path', default='../archivos_prueba/1km_750m/750m/', type=str, help='Path to the high resolution images')

args = parser.parse_args()

# normalize afai values between 0 and 1
def normalize(afai):
    return (afai - np.min(afai)) / (np.max(afai) - np.min(afai))

# Load the data
batch_size = args.batch_size
train_dataset = ChlorophyllDataset(args.low_res_path, args.high_res_path)

train_size = int(0.8 * len(train_dataset))
test_size = len(train_dataset) - train_size
train_dataset, test_dataset = th.utils.data.random_split(train_dataset, [train_size, test_size])

train_loader = th.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = th.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Create the model
model = SRCNN(in_channels=args.in_channels).to(args.device)
criterion = th.nn.MSELoss().to(args.device)
optimizer = th.optim.Adam(model.parameters(), lr=args.lr)

model.train()

# Train the model and store the loss to plot it later
epochs = args.epochs
array_loss = []
for epoch in tqdm(range(epochs)):
    for (low_res, high_res) in train_loader:
        optimizer.zero_grad()
        outputs = model(low_res)
        loss = criterion(outputs, high_res)
        loss.backward()
        optimizer.step()
    #print(f'Epoch: {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')
    # save the loss to plot it later
    array_loss.append(loss.item())


# save the model
th.save(model.state_dict(), args.save_model)