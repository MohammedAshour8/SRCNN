import netCDF4 as nc
import torch as th
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from tqdm import tqdm
from model import SRCNN


with nc.Dataset('archivos_prueba/30_enero/MODIS_AQUA_AFAI_MODIS_AQUA_AFAI.nc') as terra_file:
    # Get the data
    lat_terra = terra_file.variables['lat'][:]
    lon_terra = terra_file.variables['lon'][:]
    afai_terra = terra_file.variables['afai'][:]

# Plot the data
plt.figure(figsize=(10, 10))
plt.pcolormesh(lon_terra, lat_terra, afai_terra[0, :, :])
plt.colorbar()
plt.savefig('AQUA.png')
plt.clf()

# Get the data
with nc.Dataset('archivos_prueba/30_enero/VIIRS_NOAA20_AFAI_VIIRS_NOAA20_AFAI.nc') as noaa_file:
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

if afai_terra.shape != afai_noaa.shape:
    afai_terra = interp2d(lon_terra.flatten(), lat_terra.flatten(), afai_terra[0, :, :].flatten())(lon_noaa.flatten(), lat_noaa.flatten()).reshape(afai_noaa.shape)

# Create the model
model = SRCNN()
criterion = th.nn.MSELoss()
optimizer = th.optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in tqdm(range(100)):
    # Forward pass
    outputs = model(th.tensor(afai_noaa, dtype=th.float32))
    loss = criterion(outputs, th.tensor(afai_terra, dtype=th.float32))

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Plot the data
plt.figure(figsize=(10, 10))
plt.pcolormesh(lon_noaa, lon_terra, outputs[0, :, :].detach().numpy())
plt.colorbar()
plt.savefig('NOAA_PREDICT.png')
plt.clf()