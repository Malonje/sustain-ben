from pathlib import Path
import shutil
import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import torchvision.transforms.functional as F
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


class CropSegmentationDataset(SustainBenchDataset):
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
    _dataset_name = 'crop_seg'
    _versions_dict = {
        '1.1': {
            'download_url': 'https://worksheets.codalab.org/rest/bundles/0xaec91eb7c9d548ebb15e1b5e60f966ab/contents/blob/', # TODO, change url
            'compressed_size': 53_893_324_800 # TODO: change compressed size
        }
    }

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official', oracle_training_set=False, seed=111, filled_mask=False, use_ood_val=False):
        self._version = version
        if split_scheme == "cauvery":
            self._data_dir = root_dir
        else:
            self._data_dir = self.initialize_data_dir(root_dir, download) # TODO: uncomment

        self._split_dict = {'train': 0, 'val': 1, 'test': 2}
        self._split_names = {'train': 'Train', 'val': 'Val', 'test': 'Test'}

        self._split_scheme = split_scheme
        self.oracle_training_set = oracle_training_set

        self.root = Path(self._data_dir)
        self.seed = int(seed)
        self._original_resolution = (224, 224) #checked
        if self._split_scheme == "cauvery":
            self.metadata = pd.read_csv(os.path.join(root_dir, "cauvery", 'cauvery_dataset.csv'))
            self.metadata['ids'] = self.metadata['UNIQUE_ID']
            self.metadata['indices'] = self.metadata['UNIQUE_ID']
            self.metadata['split'] = self.metadata['SPLIT']
        else:
            self.metadata = pd.read_csv(self.root / 'clean_data.csv')
        self.filled_mask = filled_mask

        self._split_array = -1 * np.ones(len(self.metadata))
        for split in self._split_dict.keys():
            if split == 'test':
                test_mask = np.asarray(self.metadata['split'] == 'test')
                id = self.metadata['ids'][test_mask]
            elif split == 'val':
                val_mask = np.asarray(self.metadata['split'] == 'val')
                id = self.metadata['ids'][val_mask]
            else:
                split_mask = np.asarray(self.metadata['split'] == split)
                id = self.metadata['ids'][split_mask]
            self._split_array[id] = self._split_dict[split]

        self.full_idxs = self.metadata['indices']
        if self.filled_mask:
            if self._split_scheme == "cauvery":
                self._y_array = np.asarray([os.path.join(self.data_dir, 'cauvery', 'truth', f'cauvery_{y:06}.npz') for y in self.full_idxs])
            else:
                self._y_array = np.asarray([self.root / 'masks_filled' / f'{y}.png' for y in self.full_idxs])
        else:
            self._y_array = np.asarray([self.root / 'masks' / f'{y}.png' for y in self.full_idxs])

        self.metadata['y'] = self._y_array
        self._y_size = 1

        if self._split_scheme == "cauvery":
            self._metadata_fields = self.metadata.columns
            self._metadata_array = self.metadata
        else:
            self._metadata_fields = ['y', 'max_lat', 'max_lon', 'min_lat', 'min_lon']
            self._metadata_array = self.metadata[self._metadata_fields].to_numpy()

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        """
        Returns x for a given idx.
        """
        if self._split_scheme == "cauvery":
            image = np.load(os.path.join(self.data_dir, "cauvery", 'npy', f'cauvery_{idx:06}.npz'))['s2']
            # Conversion to RGB and setting pixel value range [0-255]
            image = image[[3, 2, 1], :, :]
            image[0, :, :] = (image[0, :, ] / np.max(image[0, :, ])) * 255
            image[1, :, :] = (image[1, :, ] / np.max(image[1, :, ])) * 255
            image[2, :, :] = (image[2, :, ] / np.max(image[2, :, ])) * 255
            image = np.transpose(image, (1, 2, 0))
            return image
        idx = self.full_idxs[idx]
        img = Image.open(self.root / 'imgs' / f'{idx}.jpeg').convert('RGB')
        return img

    def get_output_image(self, path):
        """
        Returns x for a given idx.
        """
        if self._split_scheme == "cauvery":
            img = np.load(path)['plot_id']
            img = np.where(img > 0, 1, 0)
            return img
        img = Image.open(path).convert('RGB')
        return img

    def crop_segmentation_metrics(self, y_true, y_pred, binarized=True):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        assert (y_true.shape == y_pred.shape)
        if not binarized:
            y_pred[y_pred > 0.5] = 1
            y_pred[y_pred != 1] = 0
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)
        f1 = f1_score(y_true, y_pred, average='binary', pos_label=1)
        acc = accuracy_score(y_true, y_pred)
        precision_recall = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1)
        print('Dice/ F1 score:', f1)
        print('Accuracy score:', acc)
        print("Precision recall fscore", precision_recall)
        return f1, acc, precision_recall

    def eval(self, y_pred, y_true, metadata, binarized=False): # TODO
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
        f1, acc, precision_recall = self.crop_segmentation_metrics(y_true, y_pred, binarized=binarized)
        results = [f1, acc, precision_recall]
        results_str = 'Dice/ F1 score: {}, Accuracy score: {}, Precision recall fscore: '.format(f1, acc, precision_recall)
        return results, results_str

        # metric = Accuracy(prediction_fn=prediction_fn)
        # # Overall evaluation + evaluate by year
        # all_results, all_results_str = self.standard_group_eval(
        #     metric,
        #     self._eval_groupers['year'],
        #     y_pred, y_true, metadata)
        # # Evaluate by region and ignore the "Other" region
        # region_grouper = self._eval_groupers['region']
        # region_results = metric.compute_group_wise(
        #     y_pred,
        #     y_true,
        #     region_grouper.metadata_to_group(metadata),
        #     region_grouper.n_groups)
        # all_results[f'{metric.name}_worst_year'] = all_results.pop(metric.worst_group_metric_field)
        # region_metric_list = []
        # for group_idx in range(region_grouper.n_groups):
        #     group_str = region_grouper.group_field_str(group_idx)
        #     group_metric = region_results[metric.group_metric_field(group_idx)]
        #     group_counts = region_results[metric.group_count_field(group_idx)]
        #     all_results[f'{metric.name}_{group_str}'] = group_metric
        #     all_results[f'count_{group_str}'] = group_counts
        #     if region_results[metric.group_count_field(group_idx)] == 0 or "Other" in group_str:
        #         continue
        #     all_results_str += (
        #         f'  {region_grouper.group_str(group_idx)}  '
        #         f"[n = {region_results[metric.group_count_field(group_idx)]:6.0f}]:\t"
        #         f"{metric.name} = {region_results[metric.group_metric_field(group_idx)]:5.3f}\n")
        #     region_metric_list.append(region_results[metric.group_metric_field(group_idx)])
        # all_results[f'{metric.name}_worst_region'] = metric.worst(region_metric_list)
        # all_results_str += f"Worst-group {metric.name}: {all_results[f'{metric.name}_worst_region']:.3f}\n"
        #
        # return all_results, all_results_str
