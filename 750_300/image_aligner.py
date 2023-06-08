import os
import numpy as np
import scipy.ndimage
import scipy.signal
import torch as th
from torchvision.transforms import functional as F
import netCDF4 as nc

class ImageAligner:
    def __init__(self, low_res_path, high_res_path):
        self.low_res_path = low_res_path
        self.high_res_path = high_res_path
        self.low_res_files = sorted(os.listdir(low_res_path))
        self.high_res_files = sorted(os.listdir(high_res_path))
        self.lon_dim = None
        self.lat_dim = None

    def align_images(self):
        
        for low_res_file_name, high_res_file_name in zip(self.low_res_files, self.high_res_files):
            if low_res_file_name.endswith('.nc') and high_res_file_name.endswith('.nc'):
                
                low_res_path = os.path.join(self.low_res_path, low_res_file_name)
                high_res_path = os.path.join(self.high_res_path, high_res_file_name)
                low_res_image = nc.Dataset(low_res_path)
                high_res_image = nc.Dataset(high_res_path)

                low_res_image_data = low_res_image.variables['afai'][:]
                high_res_image_data = high_res_image.variables['MCI'][:]

                low_res_image_data = np.nan_to_num(low_res_image_data, nan=0)
                high_res_image_data = np.nan_to_num(high_res_image_data, nan=0)

                low_res_image_data = th.from_numpy(low_res_image_data)

                _, low_res_lon, low_res_lat = low_res_image_data.shape

                low_res_image_data = F.resize(low_res_image_data, (int(low_res_lon * 750 / 300), int(low_res_lat * 750 / 300)), interpolation=F.InterpolationMode.BICUBIC)

                low_res_image_data = low_res_image_data.numpy()

                low_res_image_data = np.mean(low_res_image_data, axis=0)
                high_res_image_data = np.mean(high_res_image_data, axis=0)

                low_res_image_data_filtered = scipy.ndimage.gaussian_filter(low_res_image_data, sigma=1)

                high_res_image_data_original = high_res_image_data.copy()

                corr = scipy.signal.correlate2d(low_res_image_data_filtered, high_res_image_data, mode='same', boundary='symm')

                shift = np.unravel_index(np.argmax(corr), corr.shape)
                dx, dy = np.array(shift) - np.array(low_res_image_data_filtered.shape) // 2

                high_res_image_data_aligned = scipy.ndimage.shift(high_res_image_data_original, shift=(dx, dy))

                if not self.lat_dim:
                    self.lat_dim = high_res_image.dimensions['lat']
                    self.lon_dim = high_res_image.dimensions['lon']
                    self.lat_var = high_res_image.variables['lat']
                    self.lon_var = high_res_image.variables['lon']

                aligned_path = os.path.join(self.high_res_path, 'aligned', high_res_file_name)
                new_dataset = nc.Dataset(aligned_path, 'w')
                new_dataset.createDimension('lat', high_res_image_data_aligned.shape[0])
                new_dataset.createDimension('lon', high_res_image_data_aligned.shape[1])
                new_dataset.createVariable('lat', 'f4', ('lat',))[:] = self.lat_var[:]
                new_dataset.createVariable('lon', 'f4', ('lon',))[:] = self.lon_var[:]
                new_dataset.createVariable('MCI', 'f4', ('lat', 'lon'))[:] = high_res_image_data_aligned
                new_dataset.close()

                self.lat_dim = None
                self.lon_dim = None



    def align_image(self, low_res_image_data, high_res_image_data):
       
        low_res_image_data = np.nan_to_num(low_res_image_data, nan=0)
        high_res_image_data = np.nan_to_num(high_res_image_data, nan=0)

        low_res_image_data = th.from_numpy(low_res_image_data)

        _, low_res_lon, low_res_lat = low_res_image_data.shape

        low_res_image_data = F.resize(low_res_image_data, (int(low_res_lon * 750 / 300), int(low_res_lat * 750 / 300)), interpolation=F.InterpolationMode.BICUBIC)

        low_res_image_data = low_res_image_data.numpy()

        
        low_res_image_data = np.mean(low_res_image_data, axis=0)
        high_res_image_data = np.mean(high_res_image_data, axis=0)

        low_res_image_data_filtered = scipy.ndimage.gaussian_filter(low_res_image_data, sigma=1)

        high_res_image_data_original = high_res_image_data.copy()

        corr = scipy.signal.correlate2d(low_res_image_data_filtered, high_res_image_data, mode='same', boundary='symm')

        shift = np.unravel_index(np.argmax(corr), corr.shape)
        dx, dy = np.array(shift) - np.array(low_res_image_data_filtered.shape) // 2

        high_res_image_data_aligned = scipy.ndimage.shift(high_res_image_data_original, shift=(dx, dy))

        return high_res_image_data_aligned