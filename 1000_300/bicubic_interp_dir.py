import torch as th
import numpy as np
import netCDF4 as nc
from torchvision.transforms import functional as F
import os

class Interpolate():
    def __init__(self, low_res_path, high_res_path):
        self.low_res_path = low_res_path
        self.high_res_path = high_res_path
        self.low_res_files = sorted(os.listdir(low_res_path))
        self.high_res_files = sorted(os.listdir(high_res_path))

    def __len__(self):
        return len(self.low_res_files)
    
    def __getitem__(self, idx):
        low_res_file = nc.Dataset(self.low_res_path + self.low_res_files[idx])
        high_res_file = nc.Dataset(self.high_res_path + self.high_res_files[idx])

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

        low_res_data = np.where(low_res_data < -1000, 0, low_res_data)
        high_res_data = np.where(high_res_data < -1000, 0, high_res_data)

        low_res_data = th.from_numpy(low_res_data)
        high_res_data = th.from_numpy(high_res_data)

        _, low_res_lat, low_res_lon = low_res_data.shape

        low_res_data = F.resize(low_res_data, (high_res_data.shape[0], high_res_data.shape[1]), interpolation=F.InterpolationMode.BICUBIC)
        low_res_data = low_res_data.numpy()
        high_res_data = high_res_data.numpy()

        return low_res_data, high_res_data