import netCDF4 as nc
import torch as th
import matplotlib.pyplot as plt
from model import SRCNN
from bicubic_interp_file import Interpolate

# load the model
model = SRCNN(in_channels=2)
model.load_state_dict(th.load('model_v10.pth', map_location=th.device('cpu')))

# make a prediction with the model
low_res_file = nc.Dataset('../archivos_prueba/1km_300m/1km_malos/AQUA_030.nc')
high_res_file = nc.Dataset('../archivos_prueba/1km_300m/300m_malos/MCI_030.nc')

low_res_data, low_res_data_resized, high_res_data = Interpolate(low_res_file, high_res_file, 'afai', 'MCI').interpolate()

prediction = model(th.from_numpy(low_res_data_resized).unsqueeze(0))
prediction = prediction.detach().numpy()

# Plot the low resolution, high resolution, prediction and resized low resolution
plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plt.title('Imagen de baja resolución')
plt.pcolormesh(low_res_data[0, :, :])
#plt.colorbar()
plt.subplot(2, 1, 2)
plt.title('Imagen redimensionada')
plt.pcolormesh(prediction[0, 0, :, :])
#plt.colorbar()
plt.show()