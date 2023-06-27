import netCDF4 as nc
import torch as th
from torchvision.transforms import functional as F
import numpy as np
import matplotlib.pyplot as plt
from model import SRCNN
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--low_res_file', type=str, required=True)
parser.add_argument('--high_res_file', type=str, required=False, default=None)
parser.add_argument('--model', type=str, required=True)

args = parser.parse_args()

# load the model
model = SRCNN(in_channels=2)
model.load_state_dict(th.load(args.model, map_location=th.device('cpu')))
model.eval()

# make a prediction with the model
low_res_file = nc.Dataset(args.low_res_file)
if args.high_res_file is None:
    high_res_file = None
else:
    high_res_file = nc.Dataset(args.high_res_file)

low_res_lat = low_res_file['lat'][:]
low_res_lon = low_res_file['lon'][:]

low_res_lon, low_res_lat = np.meshgrid(low_res_lon, low_res_lat)

low_res_data = low_res_file['afai'][:]
low_res_data = np.nan_to_num(low_res_data, nan=0)

low_res_data[low_res_data < -100] = 0

low_res_data = th.from_numpy(low_res_data)

_, low_res_lon, low_res_lat = low_res_data.shape

low_res_data_resized = F.resize(low_res_data, (int(low_res_lon * 750/300), int(low_res_lat * 750/300)), interpolation=F.InterpolationMode.BICUBIC)
low_res_data_resized = low_res_data_resized.numpy()

if high_res_file is not None:
    high_res_lat = high_res_file['lat'][:]
    high_res_lon = high_res_file['lon'][:]
    high_res_lon, high_res_lat = np.meshgrid(high_res_lon, high_res_lat)

    high_res_data = high_res_file['MCI'][:]
    high_res_data = np.nan_to_num(high_res_data, nan=0)

    high_res_data[high_res_data < -100] = 0

    high_res_data = th.from_numpy(high_res_data)
    high_res_data = high_res_data.numpy()

prediction = model(th.from_numpy(low_res_data_resized).unsqueeze(0))
prediction = prediction.detach().numpy()


plt.figure(figsize=(10, 10))

plt.title('Low resolution')
plt.pcolormesh(low_res_data[0, :, :])
plt.colorbar()
plt.savefig('low_res.png')
plt.clf()

if high_res_file is not None:
    plt.figure(figsize=(10, 10))
    plt.title('High resolution')
    plt.pcolormesh(high_res_data[0, :, :])
    plt.colorbar()
    plt.savefig('high_res.png')
    plt.clf()

# plot the prediction with the same range of color mesh as the low resolution image
plt.figure(figsize=(10, 10))
plt.title('Prediction')
plt.pcolormesh(prediction[0, 0, :, :])
plt.colorbar()
plt.savefig('prediction.png')
plt.clf()