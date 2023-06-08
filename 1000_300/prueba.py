import netCDF4 as nc
import torch as th
import matplotlib.pyplot as plt
from model import SRCNN
from bicubic_interp_file import Interpolate
from image_aligner import ImageAligner
import numpy as np

# load the model
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
model = SRCNN(in_channels=2)
model.load_state_dict(th.load('model_0001_9.pth', map_location=th.device('cpu')))

# make a prediction with the model
low_res_file = nc.Dataset('../archivos_prueba/1km_300m/1km_malos/AQUA_032.nc')
high_res_file = nc.Dataset('../archivos_prueba/1km_300m/300m_malos/MCI_032.nc')

low_res_data, low_res_data_resized, high_res_data = Interpolate(low_res_file, high_res_file, 'afai', 'MCI').interpolate()

# Align the images
high_res_data_aligned = ImageAligner('../archivos_prueba/1km_300m/samples', '../archivos_prueba/1km_300m/samples').align_image(low_res_data, high_res_data)

prediction = model(th.from_numpy(low_res_data_resized).unsqueeze(0))
prediction = prediction.detach().numpy()

# Histogram analysis
"""histogram_low_res = np.histogram(low_res_data, bins=256, range=(np.min(low_res_data), np.max(low_res_data)))
histogram_high_res = np.histogram(high_res_data_aligned, bins=256, range=(np.min(high_res_data_aligned), np.max(high_res_data_aligned)))
histogram_prediction = np.histogram(prediction[0, 0, :, :], bins=256, range=(np.min(prediction[0, 0, :, :]), np.max(prediction[0, 0, :, :])))"""

print(low_res_data[0, :, :].shape)
print(high_res_data[0, :, :].shape)
print(prediction[0, 0, :, :].shape)

histogram_low_res = np.histogram(low_res_data[0, :, :], bins=256, range=(-0.3, 0.3))
histogram_high_res = np.histogram(high_res_data[0, :, :], bins=256, range=(np.min(high_res_data[0, :, :]) - 0.05, np.max(high_res_data[0, :, :]) + 0.05))
histogram_prediction = np.histogram(prediction[0, 0, :, :], bins=256, range=(np.min(prediction[0, 0, :, :]) - 0.05, np.max(prediction[0, 0, :, :]) + 0.05))

# Histogram values
values_low_res = histogram_low_res[0]
values_high_res = histogram_high_res[0]
values_prediction = histogram_prediction[0]

# Histogram bins
bins_low_res = histogram_low_res[1]
bins_high_res = histogram_high_res[1]
bins_prediction = histogram_prediction[1]

# Plot histograms applying log scale
plt.figure()
#plt.plot(bins_low_res[:-1], values_low_res, label='Low Resolution Image')
plt.plot(bins_high_res[:-1], values_high_res, label='High Resolution Image')
plt.plot(bins_prediction[:-1], values_prediction, label='Prediction')
plt.yscale('log')
plt.xlabel('Intensity Value')
plt.ylabel('Frequency')
plt.title('Histogram Analysis')
plt.legend()
plt.show()
plt.clf()
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.title('Imagen de baja resolución')
plt.pcolormesh(low_res_data[0, :, :])
plt.colorbar()
plt.subplot(2, 2, 2)
plt.title('Imagen redimensionada')
plt.pcolormesh(prediction[0, 0, :, :])
plt.colorbar()
plt.subplot(2, 2, 3)
plt.title('Imagen de alta resolución')
plt.pcolormesh(high_res_data[0, :, :])
plt.colorbar()
plt.show()