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

# load the model
model = SRCNN(in_channels=2)
model.load_state_dict(th.load('SRCNN_750_300_1000.pth'))
model.eval()

# make a prediction with the model
low_res_file = nc.Dataset('../archivos_prueba/750m_300m/VIIRS_SNPP_AFAI_VIIRS_SNPP_AFAI.nc')
high_res_file = nc.Dataset('../archivos_prueba/750m_300m/MCI_OLCI_MCI_OLCI.nc')

low_res_lat = low_res_file['lat'][:]
low_res_lon = low_res_file['lon'][:]
high_res_lat = high_res_file['lat'][:]
high_res_lon = high_res_file['lon'][:]

low_res_lon, low_res_lat = np.meshgrid(low_res_lon, low_res_lat)
high_res_lon, high_res_lat = np.meshgrid(high_res_lon, high_res_lat)

low_res_data = low_res_file['afai'][:]
high_res_data = high_res_file['MCI'][:]
low_res_data = np.nan_to_num(low_res_data, nan=0)
high_res_data = np.nan_to_num(high_res_data, nan=0)

# interpolate all values smaller than -100
low_res_data[low_res_data < -100] = 0
high_res_data[high_res_data < -100] = 0

# Interpolate nan values
low_res_data = np.ma.masked_invalid(low_res_data)
high_res_data = np.ma.masked_invalid(high_res_data)

low_res_data = normalize(low_res_data)
high_res_data = normalize(high_res_data)

low_res_data = th.from_numpy(low_res_data)
high_res_data = th.from_numpy(high_res_data)

_, low_res_lon, low_res_lat = low_res_data.shape

low_res_data_resized = F.resize(low_res_data, (int(low_res_lon * 750/300), int(low_res_lat * 750/300)), interpolation=F.InterpolationMode.BICUBIC)
low_res_data_resized = low_res_data_resized.numpy()
high_res_data = high_res_data.numpy()

prediction = model(th.from_numpy(low_res_data_resized).unsqueeze(0))
prediction = prediction.detach().numpy()

# Plot the results
plt.figure(figsize=(10, 10))
#plt.subplot(1, 3, 1)
plt.title('Low resolution')
plt.pcolormesh(low_res_data[0, :, :])
plt.colorbar()
plt.savefig('low_res.png')
plt.clf()
#plt.subplot(1, 3, 2)
plt.figure(figsize=(10, 10))
plt.title('High resolution')
plt.pcolormesh(high_res_data[0, :, :])
plt.colorbar()
plt.savefig('high_res.png')
plt.clf()
#plt.subplot(1, 3, 3)
plt.figure(figsize=(10, 10))
plt.title('Prediction')
plt.pcolormesh(prediction[0, 0, :, :])
plt.colorbar()
plt.savefig('prediction_1000.png')