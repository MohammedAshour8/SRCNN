import os
import torch as th
import numpy as np
from bicubic_interp_dir import Interpolate

def normalize(afai):
    return (afai - np.min(afai)) / (np.max(afai) - np.min(afai))

class ChlorophyllDataset(th.utils.data.Dataset):
    def __init__(self, low_res_path, high_res_path):
        self.low_res_path = low_res_path
        self.high_res_path = high_res_path
        self.low_res_files = sorted(os.listdir(low_res_path))
        self.high_res_files = sorted(os.listdir(high_res_path))

    def __len__(self):
        return len(self.low_res_files)
    
    def __getitem__(self, idx):
        low_res_data, high_res_data = Interpolate(self.low_res_path, self.high_res_path).__getitem__(idx)

        return low_res_data, high_res_data