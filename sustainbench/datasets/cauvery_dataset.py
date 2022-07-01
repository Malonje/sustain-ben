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
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from sustainbench.common.utils import subsample_idxs
from sustainbench.common.metrics.all_metrics import Accuracy
from sustainbench.common.grouper import CombinatorialGrouper
from sustainbench.datasets.sustainbench_dataset import SustainBenchDataset
import os
import torch.nn.functional as F

partition_to_idx = {
    "train": 0,
    "dev": 1,
    "val": 1,
    "test": 2,
}

BANDS = {'s1': {'VV': 0, 'VH': 1, 'RATIO': 2},
         's2': {'10': {'BLUE': 0, 'GREEN': 1, 'RED': 2, 'RDED1': 3, 'RDED2': 4, 'RDED3': 5, 'NIR': 6, 'RDED4': 7,
                       'SWIR1': 8, 'SWIR2': 9},
                '4': {'BLUE': 0, 'GREEN': 1, 'RED': 2, 'NIR': 3}},
         'planet': {'4': {'BLUE': 0, 'GREEN': 1, 'RED': 2, 'NIR': 3}}}

MEANS = {'l8': {'cauvery': torch.Tensor([12368.1361551, 12661.92114163, 13650.40757402, 13527.14502986,
                                         19340.70235415, 14114.5518405, 11896.20883162, 29226.3843278])},
         's1': {'cauvery': torch.Tensor([6.28422240e+04, 6.32034097e+04, 3.60242752e+01]),
                'ghana': torch.Tensor([-10.50, -17.24, 1.17]),
                'southsudan': torch.Tensor([-9.02, -15.26, 1.15])},
         's2': {'cauvery': torch.Tensor([444.24931557, 457.38388575, 436.65186128, 508.0436634, 627.0239963,
                                         680.004681, 670.04099671, 703.3274434, 436.3003278, 331.22561816]),  # DUMMY
                'ghana': torch.Tensor(
             [2620.00, 2519.89, 2630.31, 2739.81, 3225.22, 3562.64, 3356.57, 3788.05, 2915.40, 2102.65]),
                'southsudan': torch.Tensor(
                    [2119.15, 2061.95, 2127.71, 2277.60, 2784.21, 3088.40, 2939.33, 3308.03, 2597.14, 1834.81])},
         'planet': {'cauvery': torch.Tensor([189.38657473, 218.87043081, 247.32755279, 479.41628529]),  # DUMMY
                    'ghana': torch.Tensor([1264.81, 1255.25, 1271.10, 2033.22]),
                    'southsudan': torch.Tensor([1091.30, 1092.23, 1029.28, 2137.77])},
         's2_cldfltr': {'ghana': torch.Tensor(
             [1362.68, 1317.62, 1410.74, 1580.05, 2066.06, 2373.60, 2254.70, 2629.11, 2597.50, 1818.43]),
                        'southsudan': torch.Tensor(
                            [1137.58, 1127.62, 1173.28, 1341.70, 1877.70, 2180.27, 2072.11, 2427.68, 2308.98,
                             1544.26])}}

STDS = {'l8': {'cauvery': torch.Tensor([9909.46258226, 9772.88653258, 9093.53284106, 9096.19521172,
                                        8922.29167776, 6468.6509666,  5430.69077713, 19781.63175576])},
        's1': {'cauvery': torch.Tensor([1.29914865e+04, 1.21026555e+04, 7.12003524e+00]),
               'ghana': torch.Tensor([3.57, 4.86, 5.60]),
               'southsudan': torch.Tensor([4.49, 6.68, 21.75])},
        's2': {'cauvery': torch.Tensor([1652.44922717, 1504.62790188, 1444.78959425, 1554.34698307, 1656.84349257,
                                        1727.76967515, 1717.28851599, 1757.27593784, 1109.57370268, 879.75099205]),  # DUMMY
               'ghana': torch.Tensor(
            [2171.62, 2085.69, 2174.37, 2084.56, 2058.97, 2117.31, 1988.70, 2099.78, 1209.48, 918.19]),
               'southsudan': torch.Tensor(
                   [2113.41, 2026.64, 2126.10, 2093.35, 2066.81, 2114.85, 2049.70, 2111.51, 1320.97, 1029.58])},
        'planet': {'cauvery': torch.Tensor([1.66349306e+04, 1.66781339e+04, 9.51789817e+00, 9.51789817e+00]),  # DUMMY
                   'ghana': torch.Tensor([602.51, 598.66, 637.06, 966.27]),
                   'southsudan': torch.Tensor([526.06, 517.05, 543.74, 1022.14])},
        's2_cldfltr': {
            'ghana': torch.Tensor([511.19, 495.87, 591.44, 590.27, 745.81, 882.05, 811.14, 959.09, 964.64, 809.53]),
            'southsudan': torch.Tensor(
                [548.64, 547.45, 660.28, 677.55, 896.28, 1066.91, 1006.01, 1173.19, 1167.74, 865.42])}}


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

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='cauvery',
                 resize_planet=False, calculate_bands=True, normalize=True):
        self._resize_planet = resize_planet
        self._calculate_bands = calculate_bands
        self._normalize = normalize

        self._version = version
        if split_scheme == 'cauvery':
            self._data_dir = root_dir
        else:
            raise ValueError(f'The split_scheme {split_scheme} is not supported.')  # TODO: uncomment

        self._split_scheme = split_scheme
        self._country = 'cauvery'

        split_df = pd.read_csv(os.path.join(self.data_dir, self._country, 'cauvery_dataset.csv'))
        split_df['id'] = split_df['UNIQUE_ID']
        split_df['partition'] = split_df['SPLIT'].apply(lambda split: partition_to_idx[split])

        self._split_array = split_df['partition'].values

        self._y_array = torch.from_numpy(split_df['id'].values)
        self._y_size = (1)

        self._metadata_fields = ['plot_id', 'y']
        self._metadata_array = torch.from_numpy(split_df[['PLOT_ID', 'id']].to_numpy())

        super().__init__(root_dir, download, split_scheme)

    def __getitem__(self, idx):
        # Any transformations are handled by the SustainBenchSubset
        # since different subsets (e.g., train vs test) might have different transforms
        x = self.get_input(idx)
        y = self.get_label(idx)
        metadata = self.get_metadata(idx)
        return x, y, metadata

    def get_input(self, idx):
        """
        Returns x for a given idx.
        """
        loc_id = f'{self.y_array[idx]:06d}'
        images = np.load(os.path.join(self.data_dir, self.country, 'npy', f'{self.country}_{loc_id}.npz'))

        temp = images['s1'].astype(np.int64)
        s1 = []
        for t in range(temp.shape[0]):
            if np.any(temp[t]):
                s1.append(temp[t])
        temp = images['s2'].astype(np.int64)
        s2 = []
        for t in range(temp.shape[0]):
            if np.any(temp[t]):
                s2.append(temp[t])
        temp = images['planet'].astype(np.int64)
        planet = []
        for t in range(temp.shape[0]):
            if np.any(temp[t]):
                planet.append(temp[t])

        s1 = np.asarray(s1)
        s2 = np.asarray(s2)
        planet = np.asarray(planet)

        mask = np.load(os.path.join(self.data_dir, self.country, 'truth', f'{self.country}_{loc_id}.npz'))['mask']
        coord = np.argwhere(mask > 0)
        top_left_coord = np.min(coord, 0)
        bottom_right_coord = np.max(coord, 0)

        s1 = s1[top_left_coord[0]:bottom_right_coord[0], top_left_coord[1]:bottom_right_coord[1]]
        s2 = s2[top_left_coord[0]:bottom_right_coord[0], top_left_coord[1]:bottom_right_coord[1]]
        planet = planet[top_left_coord[0]:bottom_right_coord[0], top_left_coord[1]:bottom_right_coord[1]]

        s1 = torch.from_numpy(s1)
        s2 = torch.from_numpy(s2.astype(np.int32))
        planet = torch.from_numpy(planet.astype(np.int32))

        if self.resize_planet:
            planet = planet.permute(3, 0, 1, 2)
            planet = transforms.Resize(IMG_DIM)(planet)
            planet = planet.permute(1, 2, 3, 0)
        else:
            planet = planet.permute(3, 0, 1, 2)
            planet = transforms.CenterCrop(PLANET_DIM)(planet)
            planet = planet.permute(1, 2, 3, 0)

        # Include NDVI and GCVI for s2 and planet, calculate before normalization and numband selection
        if self.calculate_bands:
            ndvi_s2 = (s2[BANDS['s2']['10']['NIR']] - s2[BANDS['s2']['10']['RED']]) / (
                    s2[BANDS['s2']['10']['NIR']] + s2[BANDS['s2']['10']['RED']])
            ndvi_planet = (planet[BANDS['planet']['4']['NIR']] - planet[BANDS['planet']['4']['RED']]) / (
                    planet[BANDS['planet']['4']['NIR']] + planet[BANDS['planet']['4']['RED']])

            gcvi_s2 = (s2[BANDS['s2']['10']['NIR']] / s2[BANDS['s2']['10']['GREEN']]) - 1
            gcvi_planet = (planet[BANDS['planet']['4']['NIR']] / planet[BANDS['planet']['4']['GREEN']]) - 1

            ## added this to remove nans
            ndvi_s2[(s2[BANDS['s2']['10']['NIR'], :, :, :] + s2[BANDS['s2']['10']['RED'], :, :, :]) == 0] = 0
            gcvi_s2[s2[BANDS['s2']['10']['GREEN'], :, :, :] == 0] = 0
            ndvi_planet[
                (planet[BANDS['planet']['4']['NIR'], :, :, :] + planet[BANDS['planet']['4']['RED'], :, :, :]) == 0] = 0
            gcvi_planet[planet[BANDS['planet']['4']['GREEN'], :, :, :] == 0] = 0

        if self.normalize:
            s1 = self.normalization(s1, 's1')
            s2 = self.normalization(s2, 's2')
            planet = self.normalization(planet, 'planet')

        # Concatenate calculated bands
        if self.calculate_bands:
            s2 = torch.cat((s2, torch.unsqueeze(ndvi_s2, 0), torch.unsqueeze(gcvi_s2, 0)), 0)
            planet = torch.cat((planet, torch.unsqueeze(ndvi_planet, 0), torch.unsqueeze(gcvi_planet, 0)), 0)

        s1 = self.pad(s1)
        s2 = self.pad(s2)
        planet = self.pad(planet)
        return {'s1': s1, 's2': s2, 'planet': planet}

    def get_label(self, idx):
        """
        Returns x for a given idx.
        """
        loc_id = f'{self.y_array[idx]:06d}'
        sowing = np.load(os.path.join(self.data_dir, self.country, 'truth', f'{self.country}_{loc_id}.npz'))['sowing_date']
        transplanting = np.load(os.path.join(self.data_dir, self.country, 'truth', f'{self.country}_{loc_id}.npz'))['transplanting_date']
        harvesting = np.load(os.path.join(self.data_dir, self.country, 'truth', f'{self.country}_{loc_id}.npz'))['harvesting_date']
        mask = np.load(os.path.join(self.data_dir, self.country, 'truth', f'{self.country}_{loc_id}.npz'))['plot_id']
        plot_id = self._metadata_array[idx][0]
        coord = np.argwhere(mask == plot_id)
        top_left_coord = np.min(coord, 0)
        bottom_right_coord = np.max(coord, 0)
        sowing = np.unique(sowing[top_left_coord[0]:bottom_right_coord[0], top_left_coord[1]:bottom_right_coord[1]])[0]
        transplanting = np.unique(transplanting[top_left_coord[0]:bottom_right_coord[0], top_left_coord[1]:bottom_right_coord[1]])[0]
        harvesting = np.unique(harvesting[top_left_coord[0]:bottom_right_coord[0], top_left_coord[1]:bottom_right_coord[1]])[0]
        return torch.Tensor([sowing, harvesting, transplanting])

    def metrics(self, y_true, y_pred, l1_weight=1):
        """
        Regularized MSE loss; l2 loss with l1 loss too.

        Parameters
        ----------
        pred: torch.floatTensor
            The model predictions
        true: torch.floatTensor
            The true values
        l1_weight: int
            The value by which to weight the l1 loss
        scores_dict: defaultdict(list)
            A dict to which scores can be appended.

        Returns
        ----------
        loss: the regularized mse loss
        """
        loss = F.mse_loss(y_pred, y_true)

        scores_dict["l2"].append(loss.item())

        if l1_weight > 0:
            l1 = F.l1_loss(y_pred, y_true)
            loss += l1
            scores_dict["l1"].append(l1.item())
        scores_dict["loss"].append(loss.item())

        y_true = y_true.detach().cpu().numpy().flatten()
        y_pred = y_pred.detach().cpu().numpy().flatten()
        assert (y_true.shape == y_pred.shape)

        error = y_pred - y_true
        RMSE = np.sqrt(np.mean(error ** 2))
        scores_dict["RMSE"].append(RMSE)
        R2 = r2_score(y_true, y_pred)
        scores_dict["R2"].append(R2)
        return scores_dict

    def eval(self, y_pred, y_true, metadata, binarized="ignored"): # TODO
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
        scores_dict = self.metrics(y_true, y_pred)
        l1 = scores_dict["l1"]
        l2 = scores_dict["l2"]
        rmse = scores_dict["RMSE"]
        r2 = scores_dict["R2"]
        results = [l1, l2, rmse, r2]
        results_str = 'L1 loss: {}, L2 loss: {}, RMSE: {}, R2: {}'.format(l1, l2, rmse, r2)
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
        num_bands = grid.shape[0]
        means = MEANS[satellite][self.country]
        stds = STDS[satellite][self.country]
        grid = (grid - means[:num_bands].reshape(num_bands, 1, 1, 1)) / stds[:num_bands].reshape(num_bands, 1, 1, 1)

        if satellite not in ['s1', 's2', 'planet']:
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
    def calculate_bands(self):
        """
        True if aditional bands (NDVI and GCVI) will be calculated on the fly and appended
        """
        return self._calculate_bands
