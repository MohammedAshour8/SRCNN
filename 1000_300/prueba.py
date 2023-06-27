import netCDF4 as nc
import torch as th
import matplotlib.pyplot as plt
from model import SRCNN
from bicubic_interp_file import Interpolate
import numpy as np


def mean_density(predicted, high_res_data):
    md_predicted = predicted.mean()
    md_high_res = high_res_data.mean()

    mae = np.mean(np.abs(predicted - high_res_data))

    # Cálculo del error cuadrático medio (MSE)
    mse = np.mean((predicted - high_res_data) ** 2)

    print("Media de predicción:", md_predicted)
    print("Media de alta resolución:", md_high_res)
    print("Error absoluto medio (MAE):", mae)
    print("Error cuadrático medio (MSE):", mse)
    print()


def save_images(low_res_data, prediction, high_res_data):
    plt.figure(figsize=(7, 10))
    plt.pcolormesh(low_res_data[0, :, :])
    plt.savefig('predictions/predictions_' + str(i) + '/low_res.png')
    plt.clf()
    plt.figure(figsize=(7, 10))
    plt.pcolormesh(prediction[0, 0, :, :])
    plt.savefig('predictions/predictions_' + str(i) + '/prediction.png')
    plt.clf()
    plt.figure(figsize=(7, 10))
    plt.pcolormesh(high_res_data[0, :, :])
    plt.savefig('predictions/predictions_' + str(i) + '/high_res.png')
    plt.close()

def plot_histogram(i):
    low_res_file = nc.Dataset('../archivos_prueba/1km_300m/prueba_' + str(i) + '/MODIS_AQUA_AFAI_MODIS_AQUA_AFAI.nc')
    high_res_file = nc.Dataset('../archivos_prueba/1km_300m/prueba_' + str(i) + '/MCI_OLCI_MCI_OLCI.nc')
    #low_res_file = nc.Dataset('../archivos_prueba/1km_300m/1km_malos/AQUA_032.nc')
    #high_res_file = nc.Dataset('../archivos_prueba/1km_300m/300m_malos/MCI_032.nc')

    low_res_data, low_res_data_resized, high_res_data = Interpolate(low_res_file, high_res_file, 'afai', 'MCI').interpolate()


    prediction = model(th.from_numpy(low_res_data_resized).unsqueeze(0))
    prediction = prediction.detach().numpy()

    histogram_low_res = np.histogram(low_res_data[0, :, :], bins=256, range=(np.min(low_res_data[0, :, :]) - 0.05, np.max(low_res_data[0, :, :]) + 0.05))
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
    
    # save the histogram
    plt.figure(figsize=(10, 10))
    #plt.plot(bins_low_res[:-1], values_low_res, label='Low resolution')
    plt.plot(bins_high_res[:-1], values_high_res, label='High resolution')
    plt.plot(bins_prediction[:-1], values_prediction, label='Prediction')
    plt.yscale('log')
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.title('Histogram Analysis')
    plt.legend()
    plt.savefig('histograms/histogram_' + str(i) + '.png')
    plt.close()

    mean_density(prediction, high_res_data)
    save_images(low_res_data, prediction, high_res_data)


device = th.device('cuda' if th.cuda.is_available() else 'cpu')
model = SRCNN(in_channels=2)
model.load_state_dict(th.load('model_v10.pth', map_location=th.device('cpu')))

for i in range(1, 4):
    plot_histogram(i)