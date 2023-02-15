import netCDF4 as nc
import torch as th
from torchvision.transforms import functional as F

def ChlorophyllDataSet(Dataset):
    def __init__(self, low_res_path, high_res_path):
        self.low_res_path = nc.Dataset(low_res_path)
        self.high_res_path = nc.Dataset(high_res_path)

    def __getitem__(self, index):
        # Get the data
        lat_terra = self.low_res_path.variables['lat'][:]
        lon_terra = self.low_res_path.variables['lon'][:]
        afai_terra = self.low_res_path.variables['afai'][:]

        # Get the data
        lat_noaa = self.high_res_path.variables['lat'][:]
        lon_noaa = self.high_res_path.variables['lon'][:]
        afai_noaa = self.high_res_path.variables['afai'][:]

        # create a meshgrid
        lon_terra, lat_terra = np.meshgrid(lon_terra, lat_terra)
        lon_noaa, lat_noaa = np.meshgrid(lon_noaa, lat_noaa)

        afai_noaa = th.from_numpy(afai_noaa)
        afai_terra = th.from_numpy(afai_terra)

        _, afai_noaa_lat, afai_noaa_lon = afai_noaa.shape

        afai_terra = F.resize(afai_terra, (afai_noaa_lat, afai_noaa_lon), interpolation=F.InterpolationMode.BICUBIC)
        afai_terra = afai_terra.numpy()

        return afai_noaa, afai_terra
    
    def __len__(self):
        return len(self.low_res_path)