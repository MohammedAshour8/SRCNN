import netCDF4 as nc
import torch as th
from torchvision.transforms import functional as F
import numpy as np
import matplotlib.pyplot as plt
from model import SRCNN
from bicubic_interp_file import Interpolate
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--low_res_file', type=str, required=True)
parser.add_argument('--high_res_file', type=str, required=False, default=None)
parser.add_argument('--model', type=str, required=True)

args = parser.parse_args()

model = SRCNN(in_channels=2)
model.load_state_dict(th.load(args.model, map_location=th.device('cpu')))
model.eval()

low_res_file = nc.Dataset(args.low_res_file)
if args.high_res_file is None:
    high_res_file = None
else:
    high_res_file = nc.Dataset(args.high_res_file)

low_res_file = nc.Dataset(args.low_res_file)
low_res_data, low_res_data_resized, high_res_data = Interpolate(low_res_file, high_res_file, 'afai', 'MCI').interpolate()

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

plt.figure(figsize=(10, 10))
plt.title('Prediction')
plt.pcolormesh(prediction[0, 0, :, :])
plt.colorbar()
plt.savefig('prediction.png')
plt.clf()