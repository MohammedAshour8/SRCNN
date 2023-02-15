import netCDF4 as nc
import torch as th
from torchvision.transforms import functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from tqdm import tqdm
from model import SRCNN

# normalize afai values between 0 and 1
def normalize(afai):
    return (afai - np.min(afai)) / (np.max(afai) - np.min(afai))

with nc.Dataset('archivos_prueba/1km_750m/23_01_25/MODIS_AQUA_AFAI_MODIS_AQUA_AFAI.nc') as terra_file:
    # Get the data
    lat_terra = terra_file.variables['lat'][:]
    lon_terra = terra_file.variables['lon'][:]
    afai_terra = terra_file.variables['afai'][:]

# Plot the data
plt.figure(figsize=(10, 10))
plt.pcolormesh(lon_terra, lat_terra, afai_terra[0, :, :])
plt.colorbar()
plt.savefig('TERRA.png')
plt.clf()

# Get the data
with nc.Dataset('archivos_prueba/1km_750m/23_01_25/VIIRS_NOAA20_AFAI_VIIRS_NOAA20_AFAI.nc') as noaa_file:
    lat_noaa = noaa_file.variables['lat'][:]
    lon_noaa = noaa_file.variables['lon'][:]
    afai_noaa = noaa_file.variables['afai'][:]

# Plot the data
plt.figure(figsize=(10, 10))
plt.pcolormesh(lon_noaa, lat_noaa, afai_noaa[0, :, :])
plt.colorbar()
plt.savefig('NOAA.png')
plt.clf()

# create a meshgrid
lon_terra, lat_terra = np.meshgrid(lon_terra, lat_terra)
lon_noaa, lat_noaa = np.meshgrid(lon_noaa, lat_noaa)

afai_noaa = th.from_numpy(afai_noaa)
afai_terra = th.from_numpy(afai_terra)

_, afai_noaa_lat, afai_noaa_lon = afai_noaa.shape

afai_terra = F.resize(afai_terra, (afai_noaa_lat, afai_noaa_lon), interpolation=F.InterpolationMode.BICUBIC)
afai_terra = afai_terra.numpy()

print(afai_noaa.shape)
print(afai_terra.shape)

# Plot the data
plt.figure(figsize=(10, 10))
plt.pcolormesh(lon_noaa, lat_noaa, afai_terra[0, :, :])
plt.colorbar()
plt.savefig('TERRA_RESIZE.png')
plt.clf()

low_res_dataset = th.utils.data.TensorDataset(afai_terra, afai_noaa)

# Create the model
model = SRCNN()
criterion = th.nn.MSELoss()
optimizer = th.optim.Adam(model.parameters(), lr=0.001)

# Train the model noticng that afai_noaa is the high resolution image and afai_terra is the low resolution image
for epoch in range(100):
    running_loss = 0.0