import numpy as np
import scipy.ndimage
import scipy.signal
import torch as th
from torchvision.transforms import functional as F
import netCDF4 as nc
import matplotlib.pyplot as plt

image1 = nc.Dataset('../archivos_prueba/1km_300m/1km/AQUA_037.nc').variables['afai'][:]
image2 = nc.Dataset('../archivos_prueba/1km_300m/300m/MCI_037.nc').variables['MCI'][:]

# substitute nan values with 0
image1 = np.nan_to_num(image1, nan=0)
image2 = np.nan_to_num(image2, nan=0)

image1 = th.from_numpy(image1)

_, low_res_lon, low_res_lat = image1.shape

image1 = F.resize(image1, (int(low_res_lon * 1000/300), int(low_res_lat * 1000/300)), interpolation=F.InterpolationMode.BICUBIC)

image1 = image1.numpy()

# Convert the images to grayscale if they are in color format.
image1 = np.mean(image1, axis=0)
image2 = np.mean(image2, axis=0)

image1 = scipy.ndimage.gaussian_filter(image1, sigma=1)
image2 = scipy.ndimage.gaussian_filter(image2, sigma=1)

corr = scipy.signal.correlate2d(image1, image2, mode='same', boundary='symm')

shift = np.unravel_index(np.argmax(corr), corr.shape)
dx, dy = np.array(shift) - np.array(image1.shape)//2

image2_aligned = scipy.ndimage.shift(image2, shift=(dx, dy))

# Visualize the aligned images.
fig, ax = plt.subplots(2, 2, figsize=(8, 8))
ax[0, 0].imshow(image1)
ax[0, 0].set_title('Image 1')
ax[0, 1].imshow(image2)
ax[0, 1].set_title('Image 2')
ax[1, 0].imshow(image2_aligned)
ax[1, 0].set_title('Image 2 Aligned')
ax[1, 1].imshow(corr)
ax[1, 1].set_title('Cross-correlation')
plt.show()