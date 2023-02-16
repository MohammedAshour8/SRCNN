import netCDF4 as nc
import torch as th
from torchvision.transforms import functional as F
import numpy as np
import matplotlib.pyplot as plt
from chlorophyll_dataset import ChlorophyllDataset
from tqdm import tqdm
from model import SRCNN

# normalize afai values between 0 and 1
def normalize(afai):
    return (afai - np.min(afai)) / (np.max(afai) - np.min(afai))

# Load the data
batch_size = 1
train_dataset = ChlorophyllDataset('archivos_prueba/1km_750m/1km/', 'archivos_prueba/1km_750m/750m/')
train_loader = th.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create the model
model = SRCNN()
criterion = th.nn.MSELoss()
optimizer = th.optim.Adam(model.parameters(), lr=0.001)

# Train the model for 10 epochs
epochs = 100
for epoch in tqdm(range(epochs)):
    for (low_res, high_res) in train_loader:
        optimizer.zero_grad()
        outputs = model(low_res)
        loss = criterion(outputs, high_res)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')

# make a prediction with the model
low_res_file = nc.Dataset('archivos_prueba/MODIS_TERRA_AFAI_MODIS_TERRA_AFAI.nc')
high_res_file = nc.Dataset('archivos_prueba/VIIRS_NOAA20_AFAI_VIIRS_NOAA20_AFAI.nc')

low_res_lat = low_res_file['lat'][:]
low_res_lon = low_res_file['lon'][:]
high_res_lat = high_res_file['lat'][:]
high_res_lon = high_res_file['lon'][:]
low_res_lon, low_res_lat = np.meshgrid(low_res_lon, low_res_lat)
high_res_lon, high_res_lat = np.meshgrid(high_res_lon, high_res_lat)

low_res_data = low_res_file['afai'][:]
high_res_data = high_res_file['afai'][:]
low_res_data = np.nan_to_num(low_res_data, nan=0)
high_res_data = np.nan_to_num(high_res_data, nan=0)

low_res_data = th.from_numpy(low_res_data)
high_res_data = th.from_numpy(high_res_data)

_, low_res_lat, low_res_lon = low_res_data.shape

low_res_data = F.resize(low_res_data, (high_res_data.shape[1], high_res_data.shape[2]), interpolation=F.InterpolationMode.BICUBIC)
low_res_data = low_res_data.numpy()
high_res_data = high_res_data.numpy()

prediction = model(th.from_numpy(low_res_data).unsqueeze(0))
prediction = prediction.detach().numpy()

# Plot the results
plt.figure(figsize=(10, 10))
#plt.subplot(1, 3, 1)
plt.title('Low resolution')
plt.pcolormesh(high_res_lon, high_res_lat, low_res_data[0, :, :])
plt.colorbar()
plt.savefig('low_res.png')
plt.clf()
#plt.subplot(1, 3, 2)
plt.figure(figsize=(10, 10))
plt.title('High resolution')
plt.pcolormesh(high_res_lon, high_res_lat, high_res_data[0, :, :])
plt.colorbar()
plt.savefig('high_res.png')
plt.clf()
#plt.subplot(1, 3, 3)
plt.figure(figsize=(10, 10))
plt.title('Prediction')
plt.pcolormesh(high_res_lon, high_res_lat, prediction[0, 0, :, :])
plt.colorbar()
plt.savefig('prediction.png')