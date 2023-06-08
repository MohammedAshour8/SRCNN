import numpy as np
import torch as th
from torchvision.transforms import functional as F


class Interpolate():
    def __init__(self, low_res_file, high_res_file, low_res_var, high_res_var):
        self.low_res_file = low_res_file
        self.high_res_file = high_res_file
        self.low_res_var = low_res_var
        self.high_res_var = high_res_var

    def interpolate(self):
        low_res_data = self.low_res_file[self.low_res_var][:]
        high_res_data = self.high_res_file[self.high_res_var][:]
        low_res_data = np.nan_to_num(low_res_data, nan=0)
        high_res_data = np.nan_to_num(high_res_data, nan=0)

        low_res_data[low_res_data < -100] = 0
        high_res_data[high_res_data < -100] = 0

        # Interpolate nan values
        low_res_data = np.ma.masked_invalid(low_res_data)
        high_res_data = np.ma.masked_invalid(high_res_data)

        low_res_data = th.from_numpy(low_res_data)
        high_res_data = th.from_numpy(high_res_data)

        if len(high_res_data.shape) == 2:
            low_res_data_resized = F.resize(low_res_data, (high_res_data.shape[0], high_res_data.shape[1]), interpolation=F.InterpolationMode.BICUBIC)
        else:
            low_res_data_resized = F.resize(low_res_data, (high_res_data.shape[1], high_res_data.shape[2]), interpolation=F.InterpolationMode.BICUBIC)
        low_res_data = low_res_data.numpy()
        low_res_data_resized = low_res_data_resized.numpy()
        high_res_data = high_res_data.numpy()

        return low_res_data, low_res_data_resized, high_res_data