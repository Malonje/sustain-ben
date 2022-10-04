from pathlib import Path
import shutil
import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
from torchvision import transforms
import tarfile
import datetime
import pytz
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, mean_squared_error
from sustainbench.common.utils import subsample_idxs
from sustainbench.common.metrics.all_metrics import Accuracy
from sustainbench.common.grouper import CombinatorialGrouper
from sustainbench.datasets.sustainbench_dataset import SustainBenchDataset
import os
import torch.nn.functional as F

# IMG_DIM = (7, 7)


partition_to_idx = {
    "train": 0,
    "dev": 1,
    "val": 1,
    "test": 2,
}

BANDS = {'l8': {'8': {'UBLUE': 0, 'BLUE': 1, 'GREEN': 2, 'RED': 3, 'NIR': 4, 'SWIR1': 5, 'SWIR2': 6, 'THERMAL': 7}},
         's1': {'VV': 0, 'VH': 1, 'RATIO': 2},
         's2': {'10': {'BLUE': 0, 'GREEN': 1, 'RED': 2, 'RDED1': 3, 'RDED2': 4, 'RDED3': 5, 'NIR': 6, 'RDED4': 7,
                       'SWIR1': 8, 'SWIR2': 9},
                '4': {'BLUE': 0, 'GREEN': 1, 'RED': 2, 'NIR': 3}},
         'planet': {'4': {'BLUE':   0, 'GREEN': 1, 'RED': 2, 'NIR': 3}}}

MEANS = {'regionwise':

             {'l8': {'cauvery': np.array([12568.23693711, 12884.11980095, 13913.08572643, 13744.98251397,
                            19348.8538171,  14400.93252211, 12137.75946084, 32752.88348736])},
         's1': {'cauvery': np.array([5.34968438e+04, 5.36555564e+04, 3.42925821e+01]),
                'ghana': np.array([-10.50, -17.24, 1.17]),
                'southsudan': np.array([-9.02, -15.26, 1.15]),
                'tanzania': np.array([-9.80, -17.05, 1.30])},
         's2': {'cauvery': np.array([3077.80182893, 3084.27244785, 2944.2416295,  3356.42566003, 3884.70291555,
                             4146.09093281, 4089.9417718,  4226.28939138, 2381.03087606, 1744.84373434]),
                'ghana': np.array(
             [2620.00, 2519.89, 2630.31, 2739.81, 3225.22, 3562.64, 3356.57, 3788.05, 2915.40, 2102.65]),
                'southsudan': np.array(
                    [2119.15, 2061.95, 2127.71, 2277.60, 2784.21, 3088.40, 2939.33, 3308.03, 2597.14, 1834.81]),
                'tanzania': np.array(
                    [2551.54, 2471.35, 2675.69, 2799.99, 3191.33, 3453.16, 3335.64, 3660.05, 3182.23, 2383.79]),
                'germany': np.array(
                    [1991.37, 2026.92, 2136.22, 6844.82, 9951.98, 11638.58, 3664.66, 12375.27, 7351.99, 5027.96])},
         'planet': {'ghana': np.array([1264.81, 1255.25, 1271.10, 2033.22]),
                    'southsudan': np.array([1091.30, 1092.23, 1029.28, 2137.77]),
                    'tanzania': np.array([1014.16, 1023.31, 1114.17, 1813.49])},
         's2_cldfltr': {'ghana': np.array(
             [1362.68, 1317.62, 1410.74, 1580.05, 2066.06, 2373.60, 2254.70, 2629.11, 2597.50, 1818.43]),
                        'southsudan': np.array(
                            [1137.58, 1127.62, 1173.28, 1341.70, 1877.70, 2180.27, 2072.11, 2427.68, 2308.98, 1544.26]),
                        'tanzania': np.array(
                            [1148.76, 1138.87, 1341.54, 1517.01, 1937.15, 2191.31, 2148.05, 2434.61, 2774.64,
                             2072.09])}},

         'plotwise':
             {'l8': {'cauvery': np.array([12510.17013478, 12811.73275677, 13840.56329285, 13658.85679646,
                                            19316.06065994, 14344.93290226, 12067.63016428, 32553.76575401])},
         's1': {'cauvery': np.array([5.31082689e+04, 5.33025678e+04, 3.42494164e+01]),
                'ghana': np.array([-10.50, -17.24, 1.17]),
                'southsudan': np.array([-9.02, -15.26, 1.15]),
                'tanzania': np.array([-9.80, -17.05, 1.30])},
         's2': {'cauvery': np.array([3067.43027738, 3068.67802172, 2927.52472211, 3333.95464914, 3864.26186636,
                            4127.72804715, 4069.16838154, 4209.3863639,  2330.53895157, 1697.87979416]),
                'ghana': np.array(
             [2620.00, 2519.89, 2630.31, 2739.81, 3225.22, 3562.64, 3356.57, 3788.05, 2915.40, 2102.65]),
                'southsudan': np.array(
                    [2119.15, 2061.95, 2127.71, 2277.60, 2784.21, 3088.40, 2939.33, 3308.03, 2597.14, 1834.81]),
                'tanzania': np.array(
                    [2551.54, 2471.35, 2675.69, 2799.99, 3191.33, 3453.16, 3335.64, 3660.05, 3182.23, 2383.79]),
                'germany': np.array(
                    [1991.37, 2026.92, 2136.22, 6844.82, 9951.98, 11638.58, 3664.66, 12375.27, 7351.99, 5027.96])},
         'planet': {'ghana': np.array([1264.81, 1255.25, 1271.10, 2033.22]),
                    'southsudan': np.array([1091.30, 1092.23, 1029.28, 2137.77]),
                    'tanzania': np.array([1014.16, 1023.31, 1114.17, 1813.49])},
         's2_cldfltr': {'ghana': np.array(
             [1362.68, 1317.62, 1410.74, 1580.05, 2066.06, 2373.60, 2254.70, 2629.11, 2597.50, 1818.43]),
                        'southsudan': np.array(
                            [1137.58, 1127.62, 1173.28, 1341.70, 1877.70, 2180.27, 2072.11, 2427.68, 2308.98, 1544.26]),
                        'tanzania': np.array(
                            [1148.76, 1138.87, 1341.54, 1517.01, 1937.15, 2191.31, 2148.05, 2434.61, 2774.64,
                             2072.09])}}

         }

STDS = {'regionwise':

            {'l8': {'cauvery': np.array([ 9811.34429255,  9679.50989908,  8985.48993455,  9001.85187442,
                                                    8666.72647788,  6477.19337138,  5428.2011345,  18263.04960189])},
        's1': {'cauvery': np.array([2.53692211e+04, 2.52311387e+04, 7.32906139e+00]),
               'ghana': np.array([3.57, 4.86, 5.60]),
               'southsudan': np.array([4.49, 6.68, 21.75]),
               'tanzania': np.array([3.53, 4.78, 16.61])},
        's2': {'cauvery': np.array([3303.74822819, 3057.09074463, 2946.57655367, 2997.19526042, 2610.09702032,
                             2469.30697291, 2483.67488076, 2369.08866661, 1603.58160356, 1419.04843785]),

                'ghana': np.array(
            [2171.62, 2085.69, 2174.37, 2084.56, 2058.97, 2117.31, 1988.70, 2099.78, 1209.48, 918.19]),
               'southsudan': np.array(
                   [2113.41, 2026.64, 2126.10, 2093.35, 2066.81, 2114.85, 2049.70, 2111.51, 1320.97, 1029.58]),
               'tanzania': np.array(
                   [2290.97, 2204.75, 2282.90, 2214.60, 2182.51, 2226.10, 2116.62, 2210.47, 1428.33, 1135.21]),
               'germany': np.array(
                   [1943.62, 1755.82, 1841.09, 5703.38, 5104.90, 5136.54, 1663.27, 5125.05, 3682.57, 3273.71])},
        'planet': {'ghana': np.array([602.51, 598.66, 637.06, 966.27]),
                   'southsudan': np.array([526.06, 517.05, 543.74, 1022.14]),
                   'tanzania': np.array([492.33, 492.71, 558.90, 833.65])},
        's2_cldfltr': {
            'ghana': np.array([511.19, 495.87, 591.44, 590.27, 745.81, 882.05, 811.14, 959.09, 964.64, 809.53]),
            'southsudan': np.array(
                [548.64, 547.45, 660.28, 677.55, 896.28, 1066.91, 1006.01, 1173.19, 1167.74, 865.42]),
            'tanzania': np.array([462.40, 449.22, 565.88, 571.42, 686.04, 789.04, 758.31, 854.39, 1071.74, 912.79])}
                       },

        'plotwise': {'l8': {'cauvery': np.array([ 9809.87028069,  9684.29850479,  9003.76874011,  9014.18431093,
                    8708.08620149,  6487.98790027,  5429.55101277, 18357.46182696])},
        's1': {'cauvery': np.array([2.56819425e+04, 2.55194371e+04, 7.51113472e+00]),
               'ghana': np.array([3.57, 4.86, 5.60]),
               'southsudan': np.array([4.49, 6.68, 21.75]),
               'tanzania': np.array([3.53, 4.78, 16.61])},
        's2': {'cauvery': np.array([3290.45689244, 3049.48023399, 2943.81480081, 2997.7516072,  2612.79286555,
                             2472.20530761, 2488.78082753, 2372.71326428, 1589.24245901, 1407.99526645]),
                'ghana': np.array(
            [2171.62, 2085.69, 2174.37, 2084.56, 2058.97, 2117.31, 1988.70, 2099.78, 1209.48, 918.19]),
               'southsudan': np.array(
                   [2113.41, 2026.64, 2126.10, 2093.35, 2066.81, 2114.85, 2049.70, 2111.51, 1320.97, 1029.58]),
               'tanzania': np.array(
                   [2290.97, 2204.75, 2282.90, 2214.60, 2182.51, 2226.10, 2116.62, 2210.47, 1428.33, 1135.21]),
               'germany': np.array(
                   [1943.62, 1755.82, 1841.09, 5703.38, 5104.90, 5136.54, 1663.27, 5125.05, 3682.57, 3273.71])},
        'planet': {'ghana': np.array([602.51, 598.66, 637.06, 966.27]),
                   'southsudan': np.array([526.06, 517.05, 543.74, 1022.14]),
                   'tanzania': np.array([492.33, 492.71, 558.90, 833.65])},
        's2_cldfltr': {
            'ghana': np.array([511.19, 495.87, 591.44, 590.27, 745.81, 882.05, 811.14, 959.09, 964.64, 809.53]),
            'southsudan': np.array(
                [548.64, 547.45, 660.28, 677.55, 896.28, 1066.91, 1006.01, 1173.19, 1167.74, 865.42]),
            'tanzania': np.array([462.40, 449.22, 565.88, 571.42, 686.04, 789.04, 758.31, 854.39, 1071.74, 912.79])}}
        }
class CauveryDataset(SustainBenchDataset):
    """
    The Farmland Parcel Delineation dataset.
    This is a processed version of the farmland dataset used in https://arxiv.org/abs/2004.05471.
    Input (x, image):
        224 x 224 x 3 RGB satellite image.
    Label (y, image):
        if filled_mask == True, y is shows the boundary of the farmland, 224 x 224 image
        if filled_mask == False, y is shows the filled boundary of the farmland, 224 x 224 image
    Metadata:
        each image is annotated with a location coordinate, denoted as 'max_lat', 'max_lon', 'min_lat', 'min_lon'.
    Original publication:
    @inproceedings{aung2020farm,
        title={Farm Parcel Delineation Using Spatio-temporal Convolutional Networks},
        author={Aung, Han Lin and Uzkent, Burak and Burke, Marshall and Lobell, David and Ermon, Stefano},
        booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
        pages={76--77},
        year={2020}
    }
    """
    _dataset_name = 'crop_sowing_transplanting_harvesting'
    _versions_dict = {
        '1.1': {
            'download_url': None,
            'compressed_size': None  # TODO: change compressed size
        }
    }

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='cauvery',splitting_technique='regionwise',
                 resize_planet=False, calculate_bands=True, normalize=True, task="date_prediction",
                 l8_bands=[0,1,2,3,4,5,6,7], s1_bands=[0,1,2], s2_bands=[0,1,2,3,4,5,6,7,8,9],
                 ps_bands=[0,1,2,3], truth_mask=10, img_dim=(32,32), date_pred_for=None, use_actual_season=False):
        self._resize_planet = resize_planet
        self._calculate_bands = calculate_bands
        self._normalize = normalize
        self._splitting_technique = splitting_technique
        self.task = task
        self._version = version
        self.date_pred_for = date_pred_for
        self.l8_bands = l8_bands
        self.s1_bands = s1_bands
        self.s2_bands = s2_bands
        self.ps_bands = ps_bands
        self.truth_mask = truth_mask
        self.img_dim = img_dim
        self.use_actual_season = use_actual_season

        if calculate_bands:
            self.l8_bands.extend([-2, -1])
            self.s1_bands = s1_bands
            self.s2_bands.extend([-2, -1])
            self.ps_bands.extend([-2, -1])

        if split_scheme == 'cauvery':
            self._data_dir = root_dir
        else:
            raise ValueError(f'The split_scheme {split_scheme} is not supported.')  # TODO: uncomment

        self._split_scheme = split_scheme
        self._country = 'cauvery'

        split_df = pd.read_csv(os.path.join(self.data_dir, self._country, f'cauvery_dataset_{splitting_technique}.csv')).dropna(subset=['SPLIT_YIELD'])
        # print(split_df['PLOT_ID'][2380:2390])
        # split_df = pd.read_csv(os.path.join(self.data_dir, self._country, 'cauvery_dataset.csv'))

        # split_df=split_df[split_df['UNIQUE_ID']==2299].dropna(subset=['SPLIT_YIELD']).reset_index(drop=True)
        # split_df['SPLIT_YIELD'][0]='test'
        # split_df['SPLIT'][0]='test'
        # print(split_df)
        split_df['id'] = split_df['UNIQUE_ID']
        split_df['partition'] = split_df['SPLIT_YIELD'].apply(lambda split: partition_to_idx[split])

        self._split_array = split_df['partition'].values

        self._metadata_fields = ['id', 'PLOT_ID', 'SOWING_DAY', 'TRANSPLANTING_DAY', 'HARVESTING_DAY', 'YIELD']
        self._y_array = torch.from_numpy(split_df[self._metadata_fields].to_numpy())
        self._y_size = len(self._y_array)
        self._metadata_array = torch.from_numpy(split_df[self._metadata_fields].to_numpy())

        super().__init__(root_dir, download, split_scheme)

    def __getitem__(self, idx):
        # Any transformations are handled by the SustainBenchSubset
        # since different subsets (e.g., train vs test) might have different transforms
        x = self.get_input(idx)
        y = self.get_label(idx)
        metadata = None #self.get_metadata(idx)

        if self.task == "yield" and self.use_actual_season is True:
            sowing = int(self.y_array[idx][2].item()) - 1 if self.y_array[idx][2].item() > 0 else 0
            harvesting = int(self.y_array[idx][4].item()) - 1
            x['s1'][:, :, :, :sowing] = torch.zeros_like(x['s1'][:, :, :, :sowing])
            x['s2'][:, :, :, :sowing] = torch.zeros_like(x['s2'][:, :, :, :sowing])
            x['l8'][:, :, :, :sowing] = torch.zeros_like(x['l8'][:, :, :, :sowing])
            x['planet'][:, :, :, :sowing] = torch.zeros_like(x['planet'][:, :, :, :sowing])
            x['s1'][:, :, :, harvesting+1:] = torch.zeros_like(x['s1'][:, :, :, harvesting+1:])
            x['s2'][:, :, :, harvesting+1:] = torch.zeros_like(x['s2'][:, :, :, harvesting+1:])
            x['l8'][:, :, :, harvesting+1:] = torch.zeros_like(x['l8'][:, :, :, harvesting+1:])
            x['planet'][:, :, :, harvesting+1:] = torch.zeros_like(x['planet'][:, :, :, harvesting+1:])
        return x, y, metadata

    def get_input(self, idx):
        """
        Returns x for a given idx.
        """
        loc_id = f'{int(self.y_array[idx][0].item()):06d}'
        images = np.load(os.path.join(self.data_dir, self.country, 'npy', f'{self.country}_{loc_id}.npz'))
        # s1 = np.load(os.path.join(self.data_dir, '/home/parichya/Documents/cauvery_data_new(32X32)/S1/', f'{int(loc_id)}.npy'))
        # s2 = np.load(os.path.join(self.data_dir, '/home/parichya/Documents/cauvery_data_new(32X32)/S2/', f'{int(loc_id)}.npy'))

        s1 = images['s1'].astype(np.int64)
        s2 = images['s2'].astype(np.int64)
        planet = images['planet'].astype(np.int64)
        l8 = images['l8'].astype(np.int64)

        # planet = images['planet'].astype(np.int64)

        mask = np.load(os.path.join(self.data_dir, self.country, f'truth@10m', f'{self.country}_{loc_id}.npz'))['plot_id']
        mask_l8 = np.load(os.path.join(self.data_dir, self.country, 'truth@30m', f'{self.country}_{loc_id}.npz'))['plot_id']
        mask_ps = np.load(os.path.join(self.data_dir, self.country, 'truth@3m', f'{self.country}_{loc_id}.npz'))['plot_id']
        plot_id = self._metadata_array[idx][1].item()
        # mask = np.where(mask == plot_id, 1, 0)
        # coord = np.argwhere(mask == 1)
        # top_left_coord = np.min(coord, 0)
        # bottom_right_coord = np.max(coord, 0)
        coords = np.argwhere(mask==plot_id)
        x_min, y_min = np.min(coords,axis=0)
        x_max, y_max = np.max(coords,axis=0)

        # cropped = img[ : , : , x_min:x_max+1, y_min:y_max+1]
        # print(s1.shape, s2.shape)
        s1 = (s1.transpose(3,0,1,2))[ : , : , x_min:x_max+1, y_min:y_max+1]
        s2 = (s2.transpose(3,0,1,2))[ : , : , x_min:x_max+1, y_min:y_max+1]
        coords = np.argwhere(mask_l8 == plot_id)
        if len(coords) == 0:
            l8 = np.zeros((l8.shape[3], l8.shape[0], s2.shape[2], s2.shape[3]))
        else:
            x_min, y_min = np.min(coords, axis=0)
            x_max, y_max = np.max(coords, axis=0)
            l8 = (l8.transpose(3, 0, 1, 2))[:, :, x_min:x_max + 1, y_min:y_max + 1]
        coords = np.argwhere(mask_ps==plot_id)
        x_min, y_min = np.min(coords,axis=0)
        x_max, y_max = np.max(coords,axis=0)
        planet = (planet.transpose(3,0,1,2))[ : , : , x_min:x_max+1, y_min:y_max+1]


        if l8.shape[2] == 0 or l8.shape[3] == 0:
            # print(planet.shape)
            l8 = np.zeros((l8.shape[0], l8.shape[1], s2.shape[2], s2.shape[3]))

        # s1 = s1.transpose(1, 2, 3, 0)
        # s2 = s2.transpose(1, 2, 3, 0)
        # planet = planet.transpose(1, 2, 3, 0)
        # l8 = l8.transpose(1, 2, 3, 0)
        s1 = torch.from_numpy(s1)
        s2 = torch.from_numpy(s2)
        l8 = torch.from_numpy(l8)
        planet = torch.from_numpy(planet)

        # print(l8.shape, s1.shape, s2.shape)
        # planet = torch.from_numpy(planet.astype(np.int32))

        # if self.resize_planet:
        #     # print(s1.shape, s2.shape)
        #     s1 = s1.permute(3, 0, 1, 2)
        #     s1 = transforms.Resize(self.img_dim)(s1)
        #     s1 = s1.permute(1, 2, 3, 0)
        #     s2 = s2.permute(3, 0, 1, 2)
        #     s2 = transforms.Resize(self.img_dim)(s2)
        #     s2 = s2.permute(1, 2, 3, 0)
        #     # print(self.img_dim)
        #     l8 = l8.permute(3, 0, 1, 2)
        #     l8 = transforms.Resize(self.img_dim)(l8)
        #     l8 = l8.permute(1, 2, 3, 0)
        #     planet = planet.permute(3, 0, 1, 2)
        #     planet = transforms.Resize(self.img_dim)(planet)
        #     planet = planet.permute(1, 2, 3, 0)

        if self.resize_planet:
            s1 = transforms.Resize(self.img_dim, interpolation=transforms.InterpolationMode.NEAREST)(s1)
            s1 = s1[:, :, None, :,:]
            # s1= transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0)(s1)
            s1=torch.squeeze(s1, 2)


            s1 = s1.permute(1, 2, 3, 0)
            # s2 = s2.permute(3, 0, 1, 2)
            s2 = transforms.Resize(self.img_dim,interpolation=transforms.InterpolationMode.NEAREST)(s2)

            s2 = s2[:, :, None, :,:]

            s2= transforms.ColorJitter(brightness=0.6, contrast=0, saturation=0, hue=0)(s2)
            s2=torch.squeeze(s2, 2)
            # s2 = s2.permute(3, 0, 1, 2)
            s2 = s2.permute(1, 2, 3, 0)


            # l8 = l8.permute(3, 0, 1, 2)
            l8 = transforms.Resize(self.img_dim,interpolation=transforms.InterpolationMode.NEAREST)(l8)
            l8 = l8[:, :, None, :,:]
            l8= transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0)(l8)
            l8=torch.squeeze(l8, 2)
            l8 = l8.permute(1, 2, 3, 0)

            planet = transforms.Resize(self.img_dim,interpolation=transforms.InterpolationMode.NEAREST)(planet)
            planet = planet[:, :, None, :,:]
            planet = transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0)(planet)
            planet = torch.squeeze(planet, 2)
            planet = planet.permute(1, 2, 3, 0)
        else:
            s1 = s1.permute(3, 0, 1, 2)
            s1 = transforms.CenterCrop(PLANET_DIM)(s1)
            s1 = s1.permute(1, 2, 3, 0)
            s2 = s2.permute(3, 0, 1, 2)
            s2 = transforms.CenterCrop(PLANET_DIM)(s2)
            s2 = s2.permute(1, 2, 3, 0)
            l8 = l8.permute(3, 0, 1, 2)
            l8 = transforms.CenterCrop(PLANET_DIM)(l8)
            l8 = l8.permute(1, 2, 3, 0)
            planet = planet.permute(3, 0, 1, 2)
            planet = transforms.CenterCrop(PLANET_DIM)(planet)
            planet = planet.permute(1, 2, 3, 0)
        # print(planet.shape, s1.shape, s2.shape)


        # Include NDVI and GCVI for s2 and planet, calculate before normalization and numband selection
        if self.calculate_bands:
            ndvi_s2 = (s2[BANDS['s2']['10']['NIR']] - s2[BANDS['s2']['10']['RED']]) / (
                    s2[BANDS['s2']['10']['NIR']] + s2[BANDS['s2']['10']['RED']])
            ndvi_l8 = (l8[BANDS['l8']['8']['NIR']] - l8[BANDS['l8']['8']['RED']]) / (
                        l8[BANDS['l8']['8']['NIR']] + l8[BANDS['l8']['8']['RED']])
            ndvi_planet = (planet[BANDS['planet']['4']['NIR']] - planet[BANDS['planet']['4']['RED']]) / (
                        planet[BANDS['planet']['4']['NIR']] + planet[BANDS['planet']['4']['RED']])

            gcvi_s2 = (s2[BANDS['s2']['10']['NIR']] / s2[BANDS['s2']['10']['GREEN']]) - 1
            gcvi_l8 = (l8[BANDS['l8']['8']['NIR']] / l8[BANDS['l8']['8']['GREEN']]) - 1
            gcvi_planet = (planet[BANDS['planet']['4']['NIR']]/planet[BANDS['planet']['4']['GREEN']]) - 1

            ## added this to remove nans
            ndvi_s2[(s2[BANDS['s2']['10']['NIR'], :, :, :] + s2[BANDS['s2']['10']['RED'], :, :, :]) == 0] = 0
            gcvi_s2[s2[BANDS['s2']['10']['GREEN'], :, :, :] == 0] = 0
            ndvi_l8[
                (l8[BANDS['l8']['8']['NIR'], :, :, :] + l8[BANDS['l8']['8']['RED'], :, :, :]) == 0] = 0
            gcvi_l8[l8[BANDS['l8']['8']['GREEN'], :, :, :] == 0] = 0
            ndvi_planet[(planet[BANDS['planet']['4']['NIR'], :, :, :] + planet[BANDS['planet']['4']['RED'], :, :, :]) == 0] = 0
            gcvi_planet[planet[BANDS['planet']['4']['GREEN'], :, :, :] == 0] = 0

        if self.normalize:
            s1 = self.normalization(s1, 's1')
            s2 = self.normalization(s2, 's2')
            # planet = self.normalization(planet, 'planet')
            l8 = self.normalization(l8, 'l8')

        # Concatenate calculated bands
        if self.calculate_bands:
            s2 = torch.cat((s2, torch.unsqueeze(ndvi_s2, 0), torch.unsqueeze(gcvi_s2, 0)), 0)
            planet = torch.cat((planet, torch.unsqueeze(ndvi_planet, 0), torch.unsqueeze(gcvi_planet, 0)), 0)
            l8 = torch.cat((l8, torch.unsqueeze(ndvi_l8, 0), torch.unsqueeze(gcvi_l8, 0)), 0)

        # s1 = self.pad(s1)
        # s2 = self.pad(s2)
        # planet = self.pad(planet)

        return {'s1': s1[self.s1_bands], 's2': s2[self.s2_bands], 'l8': l8[self.l8_bands], 'planet': planet[self.ps_bands]}


    def get_label(self, idx):
        """
        Returns x for a given idx.
        """
        if self.task == "yield":
            return torch.Tensor([self.y_array[idx][-1].item()])#.type(torch.LongTensor)
        elif self.date_pred_for=='sowing':
            dpf = self.y_array[idx][2].item() - 1 if self.y_array[idx][2].item() > 0 else 0
        elif self.date_pred_for=='transplanting':
            dpf = self.y_array[idx][3].item() - 1
        elif self.date_pred_for=='harvesting':
            dpf = self.y_array[idx][4].item() - 1
        elif self.date_pred_for == 'unified':
            return torch.Tensor([self.y_array[idx][2].item() - 1 if self.y_array[idx][2].item() > 0 else 0,
                                 self.y_array[idx][4].item() - 1]).type(torch.LongTensor)
        # transplanting = self.y_array[idx][3].item() - 1
        # harvesting = self.y_array[idx][4].item() - 1
        # y = torch.Tensor([sowing, transplanting, harvesting]).type(torch.LongTensor)
        y = torch.Tensor([dpf]).type(torch.LongTensor)
        return y

    def metrics(self, y_true, y_pred):
        y_pred = y_pred.argmax(axis=1)
        y_true = y_true.astype('int')
        y_pred = y_pred.astype('int')
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        assert (y_true.shape == y_pred.shape)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        f1 = f1_score(y_true, y_pred, average='macro')
        acc = []
        for t, p in zip(y_true, y_pred):
            if np.abs(t - p) <= 7:
                acc.append(1)
            else:
                acc.append(0)
        acc = sum(acc)/len(acc)
        # acc = accuracy_score(y_true, y_pred)
        # print('Macro Dice/ F1 score:', f1)
        # print('RMSE:', rmse)
        # print('Accuracy score:', acc)
        return f1, rmse, acc

    def eval(self, y_pred, y_true, metadata=None, binarized=False): # TODO
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model.
            - y_true (Tensor): Ground-truth boundary images
            - metadata (Tensor): Metadata
            - binarized: Whether to use binarized prediction
        Output:
            - results (list): List of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        f1, rmse, acc = self.metrics(y_true, y_pred)
        results = [f1, rmse, acc]
        results_str = f'Dice/ F1 score: {f1}, Accuracy score: {acc}'
        return results, results_str

    def pad_m(self, tensor):
        '''
        Right pads or crops tensor to GRID_SIZE.
        '''
        pad_size = GRID_SIZE[self.country] - tensor.shape[-1]
        tensor = torch.nn.functional.pad(input=tensor, pad=(0, pad_size), value=0)
        return tensor

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def pad(self, tensor):
        '''
        Right pads or crops tensor to GRID_SIZE.
        '''
        # print("shape of tensor before padding",tensor.shape,tensor.shape[-1])
        num_samples = 40
        timestamps = tensor.shape[-1]
        if (timestamps < num_samples):
            return torch.nn.functional.pad(input=tensor, pad=(0, num_samples - timestamps), value=0)
        scores = np.ones((timestamps,))
        # print(scores)
        probabilities = self.softmax(scores)
        samples = np.random.choice(timestamps, size=num_samples, replace=False, p=probabilities)
        samples.sort()
        sampled_img_stack = tensor[:, :, :, samples]
        return sampled_img_stack

    def normalization(self, grid, satellite):
        """ Normalization based on values defined in constants.py
        Args:
          grid - (tensor) grid to be normalized
          satellite - (str) describes source that grid is from ("s1" or "s2")
        Returns:
          grid - (tensor) a normalized version of the input grid
        """
        # print(satellite)
        num_bands = grid.shape[0]
        # print(grid.shape)
        means = MEANS[self.splitting_technique][satellite][self.country]
        stds = STDS[self.splitting_technique][satellite][self.country]
        grid = (grid - means[:num_bands].reshape(num_bands, 1, 1, 1)) / stds[:num_bands].reshape(num_bands, 1, 1, 1)

        if satellite not in ['s1', 's2', 'l8','planet']:
            raise ValueError("Incorrect normalization parameters")
        return grid

    @property
    def normalize(self):
        """
        True if images will be normalized.
        """
        return self._normalize

    @property
    def country(self):
        """
        String containing the country pertaining to the dataset.
        """
        return self._country

    @property
    def resize_planet(self):
        """
        True if planet satellite imagery will be resized to other satellite sizes.
        """
        return self._resize_planet

    @property
    def splitting_technique(self):
        """
        True if planet satellite imagery will be resized to other satellite sizes.
        """
        return self._splitting_technique

    @property
    def calculate_bands(self):
        """
        True if aditional bands (NDVI and GCVI) will be calculated on the fly and appended
        """
        return self._calculate_bands
