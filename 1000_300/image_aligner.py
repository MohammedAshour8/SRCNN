import os
import numpy as np
import scipy.ndimage
import scipy.signal
import torch as th
from torchvision.transforms import functional as F
import netCDF4 as nc

low_res_path = '../archivos_prueba/1km_300m/1km/'
high_res_path = '../archivos_prueba/1km_300m/300m/'
low_res_files = sorted(os.listdir(low_res_path))
high_res_files = sorted(os.listdir(high_res_path))
lat_dim = None
lon_dim = None

for low_res_file_name, high_res_file_name in zip(low_res_files, high_res_files):
    if low_res_file_name.endswith('.nc') and high_res_file_name.endswith('.nc'):
        # Load the two netcdf format images into your program.
        low_res_path = os.path.join(low_res_path, low_res_file_name)
        high_res_path = os.path.join(high_res_path, high_res_file_name)
        low_res_image = nc.Dataset(low_res_path)
        high_res_image = nc.Dataset(high_res_path)

        low_res_image_data = low_res_image.variables['afai'][:]
        high_res_image_data = high_res_image.variables['MCI'][:]

        # substitute nan values with 0
        low_res_image_data = np.nan_to_num(low_res_image_data, nan=0)
        high_res_image_data = np.nan_to_num(high_res_image_data, nan=0)

        low_res_image_data = th.from_numpy(low_res_image_data)

        _, low_res_lon, low_res_lat = low_res_image_data.shape

        low_res_image_data = F.resize(low_res_image_data, (int(low_res_lon * 750 / 300), int(low_res_lat * 750 / 300)), interpolation=F.InterpolationMode.BICUBIC)

        low_res_image_data = low_res_image_data.numpy()

        # Convert the images to grayscale if they are in color format.
        low_res_image_data = np.mean(low_res_image_data, axis=0)
        high_res_image_data = np.mean(high_res_image_data, axis=0)

        low_res_image_data = scipy.ndimage.gaussian_filter(low_res_image_data, sigma=2)
        high_res_image_data = scipy.ndimage.gaussian_filter(high_res_image_data, sigma=2)

        corr = scipy.signal.correlate2d(low_res_image_data, high_res_image_data, mode='same', boundary='symm')

        shift = np.unravel_index(np.argmax(corr), corr.shape)
        dx, dy = np.array(shift) - np.array(low_res_image_data.shape) // 2

        high_res_image_data_aligned = scipy.ndimage.shift(high_res_image_data, shift=(dx, dy))

        # Define the lat and lon dimensions and variables
        if not lat_dim:
            lat_dim = high_res_image.dimensions['lat']
            lon_dim = high_res_image.dimensions['lon']
            lat_var = high_res_image.variables['lat']
            lon_var = high_res_image.variables['lon']

        # Create the aligned MCI variable with the lat and lon dimensions
        aligned_path = os.path.join(high_res_path, 'aligned', high_res_file_name)
        new_dataset = nc.Dataset(aligned_path, 'w')
        new_dataset.createDimension('lat', high_res_image_data_aligned.shape[0])
        new_dataset.createDimension('lon', high_res_image_data_aligned.shape[1])
        new_dataset.createVariable('lat', 'f4', ('lat',))[:] = lat_var[:]
        new_dataset.createVariable('lon', 'f4', ('lon',))[:] = lon_var[:]
        new_dataset.createVariable('MCI', 'f4', ('lat', 'lon'))[:] = high_res_image_data_aligned
        new_dataset.close()