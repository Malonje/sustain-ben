from cmath import nan
import os
import loss_fns
import croptype_models
import datetime
import torch
# import datasets
import metrics
import util_deploy
import os
# import rasterio
import numpy as np
import torch
import torch.nn.functional as F
import numpy as np
import pickle
from sustainbench import logger
import matplotlib.pyplot as plt
from torch import autograd

from constants import *
from tqdm import tqdm
from torch import autograd
import visualize

from sustainbench import get_dataset
from sustainbench.common.data_loaders import get_train_loader, get_eval_loader
import torchvision.transforms as transforms
import torch.nn.functional as F
# from sustainbench import get_dataset
# from sustainbench.common.data_loaders import get_train_loader, get_eval_loader
# import torchvision.transforms as transforms
# import torch.nn.functional as F

import PIL
from PIL import Image
# import util
MEANS_ = {'l8': {'cauvery': np.array([12568.23693711, 12884.11980095, 13913.08572643, 13744.98251397,
                                     19348.8538171,  14400.93252211, 12137.75946084, 32752.88348736]
                                    )},
         's1': {'cauvery': np.array([5.34968438e+04, 5.36555564e+04, 3.42925821e+01]),
                'ghana': np.array([-10.50, -17.24, 1.17]),
                'southsudan': np.array([-9.02, -15.26, 1.15]),
                'tanzania': np.array([-9.80, -17.05, 1.30])},
         's2': {'cauvery': np.array([3077.80182893, 3084.27244785, 2944.2416295,  3356.42566003, 3884.70291555,
                                     4146.09093281, 4089.9417718,  4226.28939138, 2381.03087606, 1744.84373434]
                                    ),
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

STDS_ = {'l8': {'cauvery': np.array([ 9811.34429255,  9679.50989908,  8985.48993455, 9001.85187442,
                                      8666.72647788,  6477.19337138,  5428.2011345,  18263.04960189]
                                    )},
        's1': {'cauvery': np.array([2.53692211e+04, 2.52311387e+04, 7.32906139e+00]),
               'ghana': np.array([3.57, 4.86, 5.60]),
               'southsudan': np.array([4.49, 6.68, 21.75]),
               'tanzania': np.array([3.53, 4.78, 16.61])},
        's2': {'cauvery': np.array([3303.74822819, 3057.09074463, 2946.57655367, 2997.19526042, 2610.09702032,
                                     2469.30697291, 2483.67488076, 2369.08866661, 1603.58160356, 1419.04843785]
                                ),
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

BANDS = {'l8': {'8': {'UBLUE': 0, 'BLUE': 1, 'GREEN': 2, 'RED': 3, 'NIR': 4, 'SWIR1': 5, 'SWIR2': 6, 'THERMAL': 7}},
         's1': {'VV': 0, 'VH': 1, 'RATIO': 2},
         's2': {'10': {'BLUE': 0, 'GREEN': 1, 'RED': 2, 'RDED1': 3, 'RDED2': 4, 'RDED3': 5, 'NIR': 6, 'RDED4': 7,
                       'SWIR1': 8, 'SWIR2': 9},
                '4': {'BLUE': 0, 'GREEN': 1, 'RED': 2, 'NIR': 3}},
         'planet': {'4': {'BLUE': 0, 'GREEN': 1, 'RED': 2, 'NIR': 3}}}

MEANS = {'l8': {'cauvery': np.array([12323.99321417, 12618.39105649, 13599.00590732, 13469.41012126,
                                     19246.94627376, 14150.09082232, 11900.22527039, 30151.18871195])},
         's1': {'cauvery': np.array([5.59124221e+04, 5.62048856e+04, 3.54572721e+01]),
                'ghana': np.array([-10.50, -17.24, 1.17]),
                'southsudan': np.array([-9.02, -15.26, 1.15]),
                'tanzania': np.array([-9.80, -17.05, 1.30])},
         's2': {'cauvery': np.array([2549.02273395, 2603.65376925, 2485.22625814, 2868.84385738, 3550.92105453,
                                     3862.91197932, 3783.76088997, 4001.17052372, 2334.81574498, 1701.94449602]),
                'ghana': np.array(
             [2620.00, 2519.89, 2630.31, 2739.81, 3225.22, 3562.64, 3356.57, 3788.05, 2915.40, 2102.65]),
                'southsudan': np.array(
                    [2119.15, 2061.95, 2127.71, 2277.60, 2784.21, 3088.40, 2939.33, 3308.03, 2597.14, 1834.81]),
                'tanzania': np.array(
                    [2551.54, 2471.35, 2675.69, 2799.99, 3191.33, 3453.16, 3335.64, 3660.05, 3182.23, 2383.79]),
                'germany': np.array(
                    [1991.37, 2026.92, 2136.22, 6844.82, 9951.98, 11638.58, 3664.66, 12375.27, 7351.99, 5027.96])},
         'planet': {'cauvery' : np.array([1282.3570983,  1470.73877623, 1664.53240303, 3189.36871313]),
                    'ghana': np.array([1264.81, 1255.25, 1271.10, 2033.22]),
                    'southsudan': np.array([1091.30, 1092.23, 1029.28, 2137.77]),
                    'tanzania': np.array([1014.16, 1023.31, 1114.17, 1813.49])},
         's2_cldfltr': {'ghana': np.array(
             [1362.68, 1317.62, 1410.74, 1580.05, 2066.06, 2373.60, 2254.70, 2629.11, 2597.50, 1818.43]),
                        'southsudan': np.array(
                            [1137.58, 1127.62, 1173.28, 1341.70, 1877.70, 2180.27, 2072.11, 2427.68, 2308.98, 1544.26]),
                        'tanzania': np.array(
                            [1148.76, 1138.87, 1341.54, 1517.01, 1937.15, 2191.31, 2148.05, 2434.61, 2774.64,
                             2072.09])}}

STDS = {'l8': {'cauvery': np.array([ 9900.16857204, 9776.42869284, 9109.375738, 9120.84374404,
                                     8959.01997492, 6547.22589968, 5485.36549631, 19514.8944211])},
        's1': {'cauvery': np.array([2.31868271e+04, 2.28824542e+04, 7.29889798e+00]),
               'ghana': np.array([3.57, 4.86, 5.60]),
               'southsudan': np.array([4.49, 6.68, 21.75]),
               'tanzania': np.array([3.53, 4.78, 16.61])},
        's2': {'cauvery': np.array([3144.86006302, 2690.39501861, 2612.12269672, 2628.01887174, 2277.53296502,
                                    2174.6560615,  2201.13007016, 2111.43942529, 1429.09178551, 1271.39393091]),
                'ghana': np.array(
            [2171.62, 2085.69, 2174.37, 2084.56, 2058.97, 2117.31, 1988.70, 2099.78, 1209.48, 918.19]),
               'southsudan': np.array(
                   [2113.41, 2026.64, 2126.10, 2093.35, 2066.81, 2114.85, 2049.70, 2111.51, 1320.97, 1029.58]),
               'tanzania': np.array(
                   [2290.97, 2204.75, 2282.90, 2214.60, 2182.51, 2226.10, 2116.62, 2210.47, 1428.33, 1135.21]),
               'germany': np.array(
                   [1943.62, 1755.82, 1841.09, 5703.38, 5104.90, 5136.54, 1663.27, 5125.05, 3682.57, 3273.71])},
        'planet': {'cauvery' : np.array([1216.77464953, 1101.1991926,  1391.19893251, 1415.2707272 ]),
                   'ghana': np.array([602.51, 598.66, 637.06, 966.27]),
                   'southsudan': np.array([526.06, 517.05, 543.74, 1022.14]),
                   'tanzania': np.array([492.33, 492.71, 558.90, 833.65])},
        's2_cldfltr': {
            'ghana': np.array([511.19, 495.87, 591.44, 590.27, 745.81, 882.05, 811.14, 959.09, 964.64, 809.53]),
            'southsudan': np.array(
                [548.64, 547.45, 660.28, 677.55, 896.28, 1066.91, 1006.01, 1173.19, 1167.74, 865.42]),
            'tanzania': np.array([462.40, 449.22, 565.88, 571.42, 686.04, 789.04, 758.31, 854.39, 1071.74, 912.79])}}

def find_idx(file_name, start_date):
    edate = file_name.split('_')[1]
    edate = dt.datetime.strptime(edate, "%Y%m%d").date()
    sdate = dt.datetime.strptime(start_date, "%Y%m%d").date()
    return (edate.date() - sdate.date()).days


def makedataset(PATH, strt_date):
    pix={
            'S1': 32,
            'S2': 32,
            'L8': 11,
            'PS': 107,}
    bands={
            'S1': 3,
            'S2': 10,
            'L8': 8,
            'PS': 4,}

    Sattelites=["S1", "S2", "L8"]
    fans=[]
    for s in Sattelites:
        path_to_sat=PATH+"/"+s
        all_file=os.listdir(path_to_sat)
        ans_array = torch.zeros(184, bands[s], pix[s], pix[s])
        max_h,max_w=0,0
        for f in all_file:
            src = rasterio.open(all_file+"/"+f)
            n_val=src.read()
            max_h=max(max_h, n_val.shape[1])
            max_w=max(max_w, n_val.shape[2])
        for f in all_file:
            idx=find_idx(f,strt_date)
            src = rasterio.open(all_file+"/"+f)
            n_val=src.read()
            n_val=torch.from_numpy(n_val)
            # padd/crop to store
            pad=(max_w-n_val.shape[2],0,max_h-n_val.shape[1],0)
            n_pad = F.pad(n_val, pad, "constant", 0)
            ans_array[idx]=n_pad

        fans.append(ans_array)
    return fans




def get_input_others(images, img_dim, l8_bands, s2_bands, s1_bands, ps_bands, calculate_bands):

    l8_bands_=l8_bands.copy()
    s1_bands_=s1_bands.copy()
    s2_bands_=s2_bands.copy()
    ps_bands_=ps_bands.copy()
    resize_planet=True
    if calculate_bands:
        l8_bands_.extend([-2, -1])
        s1_bands_ = s1_bands_
        s2_bands_.extend([-2, -1])
        ps_bands_.extend([-2, -1])



    s1 = images['s1'].astype(np.int64)
    s2 = images['s2'].astype(np.int64)
    planet = images['planet'].astype(np.int64)
    l8 = images['l8'].astype(np.int64)

    # planet = images['planet'].astype(np.int64)

    # mask = np.load(os.path.join(self.data_dir, self.country, f'truth@10m', f'{self.country}_{loc_id}.npz'))['plot_id']
    # mask_l8 = np.load(os.path.join(self.data_dir, self.country, 'truth@30m', f'{self.country}_{loc_id}.npz'))['plot_id']
    # mask_ps = np.load(os.path.join(self.data_dir, self.country, 'truth@3m', f'{self.country}_{loc_id}.npz'))['plot_id']
    # plot_id = self._metadata_array[idx][1].item()
    # mask = np.where(mask == plot_id, 1, 0)
    # coord = np.argwhere(mask == 1)
    # top_left_coord = np.min(coord, 0)
    # bottom_right_coord = np.max(coord, 0)
    # coords = np.argwhere(mask==plot_id)
    # x_min, y_min = np.min(coords,axis=0)
    # x_max, y_max = np.max(coords,axis=0)

    # cropped = img[ : , : , x_min:x_max+1, y_min:y_max+1]
    # print(s1.shape, s2.shape)
    # s1 = (s1.transpose(3,0,1,2))[ : , : , x_min:x_max+1, y_min:y_max+1]
    # s2 = (s2.transpose(3,0,1,2))[ : , : , x_min:x_max+1, y_min:y_max+1]
    # coords = np.argwhere(mask_l8 == plot_id)
    # if len(coords) == 0:
    #     l8 = np.zeros((l8.shape[3], l8.shape[0], s2.shape[2], s2.shape[3]))
    # else:
    #     x_min, y_min = np.min(coords, axis=0)
    #     x_max, y_max = np.max(coords, axis=0)
    #     l8 = (l8.transpose(3, 0, 1, 2))[:, :, x_min:x_max + 1, y_min:y_max + 1]
    # coords = np.argwhere(mask_ps==plot_id)
    # x_min, y_min = np.min(coords,axis=0)
    # x_max, y_max = np.max(coords,axis=0)
    # planet = (planet.transpose(3,0,1,2))[ : , : , x_min:x_max+1, y_min:y_max+1]


    if l8.shape[2] == 0 or l8.shape[3] == 0:
        # print(planet.shape)
        l8 = np.zeros((l8.shape[0], l8.shape[1], s2.shape[2], s2.shape[3]))
    # print(l8.shape)
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

    if resize_planet:
        # print(s1.shape, s2.shape)
        s1 = s1.permute(3, 0, 1, 2)
        s1 = transforms.Resize(img_dim)(s1)
        s1 = s1.permute(1, 2, 3, 0)
        s2 = s2.permute(3, 0, 1, 2)
        s2 = transforms.Resize(img_dim)(s2)
        s2 = s2.permute(1, 2, 3, 0)
        # print(self.img_dim)
        l8 = l8.permute(3, 0, 1, 2)
        l8 = transforms.Resize(img_dim)(l8)
        l8 = l8.permute(1, 2, 3, 0)
        planet = planet.permute(3, 0, 1, 2)
        planet = transforms.Resize(img_dim)(planet)
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
    if calculate_bands:
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

    # if self.normalize:
    s1 = normalization(s1, 's1', 'others')
    s2 = normalization(s2, 's2', 'others')
    # planet = self.normalization(planet, 'planet', 'others)
    l8 = normalization(l8, 'l8', 'others')

    # Concatenate calculated bands
    if calculate_bands:
        s2 = torch.cat((s2, torch.unsqueeze(ndvi_s2, 0), torch.unsqueeze(gcvi_s2, 0)), 0)
        planet = torch.cat((planet, torch.unsqueeze(ndvi_planet, 0), torch.unsqueeze(gcvi_planet, 0)), 0)
        l8 = torch.cat((l8, torch.unsqueeze(ndvi_l8, 0), torch.unsqueeze(gcvi_l8, 0)), 0)

    # s1 = self.pad(s1)
    # s2 = self.pad(s2)
    # planet = self.pad(planet)
    return {'s1': s1[s1_bands_], 's2': s2[s2_bands_], 'l8': l8[l8_bands_], 'planet': planet[ps_bands_]}


def get_input_croptype(images, img_dim, l8_bands, s2_bands, s1_bands, ps_bands,calculate_bands):
        # """
        # Returns X for a given idx.
        # """
        # loc_id = f'{self.y_array[idx]:06d}'
        # if loc_id == "004660":  # problem in soutsudan one
        #     loc_id = "001603"  # random one
        # images = np.load(os.path.join(self.data_dir, self.country, 'npy', f'{self.country}_{loc_id}.npz'))
        l8_bands_=l8_bands.copy()
        s1_bands_=s1_bands.copy()
        s2_bands_=s2_bands.copy()
        ps_bands_=ps_bands.copy()
        resize_planet=True
        if calculate_bands:
            l8_bands_.extend([-2, -1])
            s1_bands_ = s1_bands_
            s2_bands_.extend([-2, -1])
            ps_bands_.extend([-2, -1])

        temp = images['s1'].astype(np.int64)
        s1 = []
        for t in range(temp.shape[-1]):  # (B x H x W x T) shape of the input image
            if np.any(temp[:, :, :, t]):
                s1.append(temp[:, :, :, t])
        temp = images['s2'].astype(np.int64)
        s2 = []
        for t in range(temp.shape[-1]):
            if np.any(temp[:, :, :, t]):
                s2.append(temp[:, :, :, t])
        temp = images['l8'].astype(np.int64)
        l8 = []
        for t in range(temp.shape[-1]):
            if np.any(temp[:, :, :, t]):
                l8.append(temp[:, :, :, t])
        temp = images['planet'].astype(np.int64)
        planet = []
        for t in range(temp.shape[-1]):
            if np.any(temp[:, :, :, t]):
                planet.append(temp[:, :, :, t])

        s1 = np.asarray(s1)
        s2 = np.asarray(s2)
        l8 = np.asarray(l8)
        planet = np.asarray(planet)

        s1 = torch.from_numpy(s1).permute(1, 0, 2, 3)
        s2 = torch.from_numpy(s2).permute(1, 0, 2, 3)
        l8 = torch.from_numpy(l8).permute(1, 0, 2, 3)
        try:
            planet = torch.from_numpy(planet).permute(1, 0, 2, 3)
        except:
            planet = torch.zeros((4, 184, 108, 108))

        if resize_planet:
            s1 = s1.permute(3, 0, 1, 2)
            s1 = transforms.Resize(img_dim)(s1)
            s1 = s1.permute(1, 2, 3, 0)
            s2 = s2.permute(3, 0, 1, 2)
            s2 = transforms.Resize(img_dim)(s2)
            s2 = s2.permute(1, 2, 3, 0)
            l8 = l8.permute(3, 0, 1, 2)
            l8 = transforms.Resize(img_dim)(l8)
            l8 = l8.permute(1, 2, 3, 0)
            planet = planet.permute(3, 0, 1, 2)
            planet = transforms.Resize(img_dim)(planet)
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

        # Include NDVI and GCVI for s2 and planet, calculate before normalization and numband selection
        if calculate_bands:
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

        # if self.normalize:
        s1 = normalization(s1, 's1', 'crop_type')
        s2 = normalization(s2, 's2', 'crop_type')
        l8 = normalization(l8, 'l8', 'crop_type')
        planet = normalization(planet, 'planet', 'crop_type')

        # Concatenate calculated bands
        if calculate_bands:
            s2 = torch.cat((s2, torch.unsqueeze(ndvi_s2, 0), torch.unsqueeze(gcvi_s2, 0)), 0)
            planet = torch.cat((planet, torch.unsqueeze(ndvi_planet, 0), torch.unsqueeze(gcvi_planet, 0)), 0)
            l8 = torch.cat((l8, torch.unsqueeze(ndvi_l8, 0), torch.unsqueeze(gcvi_l8, 0)), 0)

        s1 = pad(s1)
        s2 = pad(s2)
        l8 = pad(l8)
        planet = pad(planet)

        return {'s1': s1[s1_bands_], 's2': s2[s2_bands_], 'l8': l8[l8_bands_], 'planet': planet[ps_bands_]}

def normalization( grid, satellite, task_for):
    """ Normalization based on values defined in constants.py
    Args:
      grid - (tensor) grid to be normalized
      satellite - (str) describes source that grid is from ("s1" or "s2")
    Returns:
      grid - (tensor) a normalized version of the input grid
    """

    num_bands = grid.shape[0]
    if task_for=='crop_type':
        means = MEANS[satellite]['cauvery']
        stds = STDS[satellite]['cauvery']
    else:
        means = MEANS_[satellite]['cauvery']
        stds = STDS_[satellite]['cauvery']
    grid = (grid - means[:num_bands].reshape(num_bands, 1, 1, 1)) / stds[:num_bands].reshape(num_bands, 1, 1, 1)

    if satellite not in ['s1', 's2', 'l8', 'planet']:
        raise ValueError("Incorrect normalization parameters")
    return grid

def create_patch(t,s):

    # t is the tensor of shape(B,H,W,T), s is the patch size
    s=s[0]
    psh=s-(t.shape[1]%s) # finding the pad size
    psw= s-(t.shape[2]%s)
    if(t.shape[1]%s == 0):
        psh=0
    if(t.shape[2]%s == 0):
        psw=0
    pad=(0,0,psw,0,psh,0)
    t_pad = F.pad(t, pad, "constant", 0)  # padding the tensor
    oshapes=t_pad.shape
    # print(oshapes)
    newn=t_pad.reshape(oshapes[0],oshapes[1]//s, s, oshapes[2]//s, s, oshapes[3]).swapaxes(2,3) # (B, H, W, T) -> (B, H/s, s, W/s,s, T) -> (B, H/s,W/s, s, s, T)
    newn=newn.reshape(oshapes[0],-1, s, s,oshapes[3]).swapaxes(0,1)  # (B, H/s,W/s, s, s, T) ->(B, H/s*W/s , s, s, T) ->( H/s*W/s, B, s, s, T)
    return newn

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def pad(tensor):
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
    probabilities = softmax(scores)
    samples = np.random.choice(timestamps, size=num_samples, replace=False, p=probabilities)
    samples.sort()
    sampled_img_stack = tensor[:, :, :, samples]
    return sampled_img_stack

def cropout(t,inp,s):  # pass output from model of shape(Nxsxs) with the original input image of shape (B, H, W, T) and s
  # get final size, create empty torch, copy data at required place, return
    s=s[0]
    N=t.shape[0]  # total number of patches
    H=inp.shape[1]  # finding the shape of the final 2D image which will be created after concating patches
    if(inp.shape[1]%s !=0):
        H+= s-(inp.shape[1]%s)
    W=(N*s*s)//H
    fans=torch.empty(H,W)  # creating empth 2d image of HxW
    hh=H//s  # number of patches on H axis
    ww=W//s
    for i in range(0,H,s):
        for j in range(0,W,s):
            ind= (i//s * ww) + j//s  # finding the index of patch at this i,j location
            fans[i:i+s, j:j+s]=t[ind]  # copying data at the required location
    return fans


# def get_input(images, satellite, l8_bands, s2_bands, s1_bands, ps_bands, task_for, calculate_bands ):
#     """
#     Returns X for a given idx.
#     """
#     # loc_id = f'{self.y_array[idx]:06d}'
#     # if loc_id == "004660":  # problem in soutsudan one
#     #     loc_id = "001603"  # random one
#     # images = np.load(os.path.join(self.data_dir, self.country, 'npy', f'{self.country}_{loc_id}.npz'))
#     if satellite=='s1':
#         bands = s1_bands.copy()
#     if satellite=='s2':
#         bands = s2_bands.copy()
#     if satellite=='l8':
#         bands = l8_bands.copy()
#     if satellite=='planet':
#         bands = ps_bands.copy()
#
#     temp = images.numpy()
#     temp = temp.astype(np.int64)
#     img = []
#     for t in range(temp.shape[-1]):  # (B x H x W x T) shape of the input image
#         if np.any(temp[:, :, :, t]):
#             img.append(temp[:, :, :, t])
#
#     # print(img[0].shape)
#     img = np.asarray(img)
#
#
#     if satellite=='planet':
#         try:
#             img = torch.from_numpy(img).permute(1, 0, 2, 3)
#         except:
#             img = torch.zeros((4, 184, 108, 108))
#
#     img = torch.from_numpy(img).permute(1,  2, 3, 0)
#     # print(img.shape)
#
#     # Include NDVI and GCVI for s2 and planet, calculate before normalization and numband selection
#     if calculate_bands:
#         if satellite=='s2':
#
#             bands.extend([-2, -1])
#             ndvi_img = (img[BANDS[satellite]['10']['NIR']] - img[BANDS[satellite]['10']['RED']]) / (
#                         img[BANDS[satellite]['10']['NIR']] + img[BANDS[satellite]['10']['RED']])
#             gcvi_img = (img[BANDS[satellite]['10']['NIR']] / img[BANDS[satellite]['10']['GREEN']]) - 1
#             ndvi_img[(img[BANDS[satellite]['10']['NIR'], :, :, :] + img[BANDS[satellite]['10']['RED'], :, :, :]) == 0] = 0
#             gcvi_img[img[BANDS[satellite]['10']['GREEN'], :, :, :] == 0] = 0
#         if satellite=='l8':
#
#
#             bands.extend([-2, -1])
#             ndvi_img = (img[BANDS[satellite]['8']['NIR']] - img[BANDS[satellite]['8']['RED']]) / (
#                         img[BANDS[satellite]['8']['NIR']] + img[BANDS[satellite]['8']['RED']])
#             gcvi_img = (img[BANDS[satellite]['8']['NIR']] / img[BANDS[satellite]['8']['GREEN']]) - 1
#             ndvi_img[
#             (img[BANDS[satellite]['8']['NIR'], :, :, :] + img[BANDS[satellite]['8']['RED'], :, :, :]) == 0] = 0
#             gcvi_img[img[BANDS[satellite]['8']['GREEN'], :, :, :] == 0] = 0
#         if satellite=='planet':
#             bands.extend([-2, -1])
#             ndvi_img = (img[BANDS[satellite]['4']['NIR']] - img[BANDS[satellite]['4']['RED']]) / (
#                         img[BANDS[satellite]['4']['NIR']] + img[BANDS[satellite]['4']['RED']])
#
#             gcvi_img = (img[BANDS[satellite]['4']['NIR']]/img[BANDS[satellite]['4']['GREEN']]) - 1
#             ndvi_img[(img[BANDS[satellite]['4']['NIR'], :, :, :] + img[BANDS[satellite]['4']['RED'], :, :, :]) == 0] = 0
#             gcvi_img[img[BANDS[satellite]['4']['GREEN'], :, :, :] == 0] = 0
#
#
#
#
#
#     # if normalize:
#     img = normalization(img, satellite, task_for)
#     if calculate_bands:
#         img = torch.cat((img, torch.unsqueeze(ndvi_img, 0), torch.unsqueeze(gcvi_img, 0)), 0)
#     print(img.shape)
#     if task_for=='crop_type':
#         img = pad(img)
#
#     return img[bands,:,:,:]
#
# # def load_model(**args,img_d):
# #     model = croptype_models.get_model( **vars(args))
# #     # model.load_state_dict(torch.load(f"../model_weights/crop_yield_best_val({run_name}).pth.tar"))
# #     return model
def norm_plot(a):
    row_sums = a.sum(axis=1)
    new_matrix = a / row_sums[:, np.newaxis]
    return new_matrix
def norm_max(a):
    return np.true_divide(a,np.amax(a))
def plot_arr(H):

    fig = plt.figure(figsize=(6, 3.2))

    ax = fig.add_subplot(111)
    ax.set_title('colorMap')
    plt.imshow(H)
    ax.set_aspect('equal')

    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plt.colorbar(orientation='vertical')
    plt.show()
def make_input_patch(img,i,j,crop_to_dim):

    #shape of img is bxhxwxt
    H=img.shape[1]
    W=img.shape[2]
    input_=np.zeros((img.shape[0], crop_to_dim,crop_to_dim, img.shape[-1]))

    hu = 0
    hl = W
    wl = 0
    wr = H
    crop_to_dim = crop_to_dim //2
    if i+crop_to_dim < H:
        wr = i+crop_to_dim
    if i-crop_to_dim >= 0:
        wl = i-crop_to_dim
    if j+crop_to_dim < W:
        hl = j+crop_to_dim
    if j-crop_to_dim >= 0:
        hu = j-crop_to_dim

    crp_img = img[: , wl:wr+1 , hu:hl+1 , :]
    input_[:, :crp_img.shape[1] , :crp_img.shape[2]  , :] = crp_img
    # print('print input shape:',input_.shape)

    return input_

def concat_satellite(inputs, use_s1, use_s2,use_l8,use_planet):
    # print('l8',inputs['l8'].shape)
    # print('s2',inputs['s2'].shape)
    temp_inputs = None
    if use_s1:
        temp_inputs = inputs['s1']
    if use_s2:
        if temp_inputs is None:
            temp_inputs = inputs['s2']
        else:
            temp_inputs = torch.cat((temp_inputs, inputs['s2']), dim=0)
    if use_l8:
        if temp_inputs is None:
            temp_inputs = inputs['l8']
        else:
            temp_inputs = torch.cat((temp_inputs, inputs['l8']), dim=0)
    if use_planet:
        if temp_inputs is None:
            temp_inputs = inputs['planet']
        else:
            temp_inputs = torch.cat((temp_inputs, inputs['planet']), dim=0)
    # print(temp_inputs.shape)
    return(temp_inputs)

def metrics(y_true, y_pred):
    # y_pred = y_pred.argmax(axis=1)
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
def get_model_input(im,i,j, dim):
    input = make_input_patch(im, i ,j , dim)
    #convert input to batch size 1
    input = torch.tensor(input)
    input = input[None, :, : , :, :]
    inputs = input.permute(0, 1, 4, 2, 3)
    return inputs

def get_bands(l8_band, s1_band, s2_band, ps_band):
    l8_bands = [0,1,2,3,4,5,6,7]
    s1_bands = [0,1,2]
    s2_bands = [0,1,2,3,4,5,6,7,8,9]
    ps_bands = [0,1,2,3]
    if l8_band is not None:
        l8_bands=l8_band[1:-1]
        l8_bands=l8_bands.split(',')
        l8_bands=[int(x) for x in l8_bands]
    if s1_band is not None:
        s1_bands=s1_band[1:-1]
        s1_bands=s1_bands.split(',')
        s1_bands=[int(x) for x in s1_bands]
    if s2_band is not None:
        s2_bands=s2_band[1:-1]
        s2_bands=s2_bands.split(',')
        s2_bands=[int(x) for x in s2_bands]
    if ps_band is not None:
        ps_bands=ps_band[1:-1]
        ps_bands=ps_bands.split(',')
        ps_bands=[int(x) for x in ps_bands]
    return l8_bands, s1_bands, s2_bands, ps_bands
def get_dim(use_s1, use_s2, use_l8, use_planet):
    sat_names = ""
    if use_s1:
        sat_names += "S1"
    if use_s2:
        sat_names += "S2"
    if use_l8:
        sat_names += "L8"
    if use_planet:
        sat_names += "planet"
    img_dimension = (7,7)
    crop_typ_dim = (32,32)
    if 'planet' in sat_names:
        img_dimension = (19,19)
        crop_typ_dim = (108,108)
    elif sat_names == 'L8':
        img_dimension = (3,3)
        crop_typ_dim = (12,12)

    return img_dimension, crop_typ_dim

def main(args):

    l8_bands, s1_bands, s2_bands, ps_bands = get_bands(args.l8_bands, args.s1_bands, args.s2_bands, args.ps_bands)
    img_dimension, crop_typ_dim =  get_dim(args.use_s1, args.use_s2, args.use_l8, args.use_planet)



    # # PATH=input("Enter path to region")
    # # SEASON=input("Enter season")
    # # YEAR=input("Enter year")

    # PATH="get path from user"
    # SEASON="01"
    # YEAR=2020
    # strt_date='01-'+SEASON+'-'+YEAR
    # img_ = makedataset(PATH, strt_date)
    uids=[
          2139,2144,2145,2146,2162,2163,2164,2165,2168,2169,2201,2202,2203,2204,2261,2268,2269,2270,2271,
            2299,2057,2058,2059,2060,2061,2062,2063,2075,2076,2089,2093,2090,2091,2092,2110,2111,2122,2123]
    #Load model croptype
    args.l8_bands = None
    args.ps_bands = None
    args.s1_bands = None
    args.s2_bands='[0,1,2,3,4,5,6,7,8,9]'
    model_crp = croptype_models.get_model( **vars(args))
    model_crp.load_state_dict(torch.load('/home/parichya/Documents/model_weights/blooming-sky-345 [S2(All)].pth.tar'))
    model_crp=model_crp.cuda()
    #Load model sow
    args.model_name = 'unet-fc'
    args.num_timesteps=184

    args.use_s2 = True
    args.use_s1 = False
    args.use_planet = False
    args.use_l8 = True
    args.s2_bands = '[0,1,2,3,4,5,6,7,8,9]'
    args.l8_bands = '[0]'
    img_dimension, crop_typ_dim =  get_dim(args.use_s1, args.use_s2, args.use_l8, args.use_planet)
    model_sow = croptype_models.get_model(input_shape=img_dimension, **vars(args))
    model_sow.load_state_dict(torch.load('/home/parichya/Documents/model_weights/sowing_date_prediction.pth.tar'))

    #Load model transplant
    args.num_timesteps=184
    args.use_s2 = False
    args.use_s1 = False
    args.use_planet = False
    args.use_l8 = True
    args.l8_bands = '[0,1,2,3,4,5,6]'
    args.model_name = 'unet-fc'
    img_dimension, crop_typ_dim =  get_dim(args.use_s1, args.use_s2, args.use_l8, args.use_planet)
    model_trans = croptype_models.get_model(input_shape=img_dimension, **vars(args))
    model_trans.load_state_dict(torch.load('/home/parichya/Documents/model_weights/date_prediction_best_val_f1(ethereal-grass-12 [L8-Trans(All Spectral)]).pth.tar'))

    #Load model harvest
    args.use_s2 = False
    args.use_s1 = True
    args.use_planet = False
    args.use_l8 = False
    args.num_timesteps=184
    args.model_name = 'unet-fc'
    args.s1_bands = '[0]'
    img_dimension, crop_typ_dim =  get_dim(args.use_s1, args.use_s2, args.use_l8, args.use_planet)
    model_harv = croptype_models.get_model(input_shape=img_dimension, **vars(args))
    model_harv.load_state_dict(torch.load('/home/parichya/Documents/model_weights/date_prediction_best_val_f1(wise-forest-21 [S1-Harvest(VV)]).pth.tar'))

    #Load model yield
    args.use_s2 = True
    args.use_s1 = False
    args.use_planet = False
    args.use_l8 = True

    args.l8_bands = '[0]'
    args.s2_bands = '[0,1,2,3,4,5,6,7,8,9]'
    args.model_name = 'unet-fc-yield'
    img_dimension, crop_typ_dim =  get_dim(args.use_s1, args.use_s2, args.use_l8, args.use_planet)
    model_yield = croptype_models.get_model(input_shape=img_dimension, **vars(args))
    model_yield.load_state_dict(torch.load('/home/parichya/Documents/model_weights/crop_yield [Fusion(L8-UB,S2-All)].pth.tar'))

    for uid in uids:
        args.use_l8=False
        args.use_s1=False
        args.use_planet=False
        args.use_s2=True

        args.l8_bands = None
        args.ps_bands = None
        args.s1_bands = None
        args.s2_bands='[0,1,2,3,4,5,6,7,8,9]'
        args.num_timesteps=40

        print(uid)
        path = '/home/parichya/Documents/cauvery/npy/'
        img_ = np.load(path+'cauvery_00'+str(uid)+'.npz')
        truth = np.load('/home/parichya/Documents/cauvery/truth@10m/'+'cauvery_00'+str(uid)+'.npz')
        l8_bands, s1_bands, s2_bands, ps_bands = get_bands(args.l8_bands, args.s1_bands, args.s2_bands, args.ps_bands)
        img_dimension, crop_typ_dim = get_dim(args.use_s1, args.use_s2, args.use_l8, args.use_planet)
        X = get_input_croptype(img_, crop_typ_dim, l8_bands, s2_bands, s1_bands, ps_bands, args.include_indices)
        X = concat_satellite(X, args.use_s1, args.use_s2, args.use_l8, args.use_planet)

        Y=np.zeros((5,X.shape[1],X.shape[2]))
        # print('X shape:', X.shape)
        out = create_patch(X,crop_typ_dim)
        # print('out shape:', out.shape)
        out = out.permute(0, 1, 4, 2, 3)
        # model = croptype_models.get_model( **vars(args))
        # if args.model_name in DL_MODELS:
        #     print('Total trainable model parameters: {}'.format(
        #         sum(p.numel() for p in model.parameters() if p.requires_grad)))
        # model.load_state_dict(torch.load('/home/parichya/Documents/model_weights/blooming-sky-345 [S2(All)].pth.tar'))

        res = model_crp(out.cuda().float())
        Y[0]=cropout(res[0][0][None,:,:], X, crop_typ_dim).detach().numpy()
        # print(res[0][0][None,:,:])
        res = torch.argmax(res, dim=1)
        res = torch.where(res==0,1,0)
        # res = res[0][0]
        # res = res[None, :, : ]
        # print(res.shape)

        # cover_crop_type = cropout(res, X, crop_typ_dim)
        cover_crop_type = res[0]
        # res = res[0][0]
        # res = res[None, :, : ]
        # Y[0] = cropout(res , X, crop_typ_dim)
        # print(cover_crop_type.shape)
        # Y[0] = cover_crop_type

        num_pixels = torch.where(cover_crop_type == 1)
        args.num_timesteps=184
        # X_ = get_input_others(img_, img_dimension, l8_bands, s2_bands, s1_bands, ps_bands, args.include_indices)
        # X_ = concat_satellite(X_, args.use_s1, args.use_s2, args.use_l8, args.use_planet)
        # print('X_ shape:',X_.shape)

        for k in range(len(num_pixels[0])):
            i = int(num_pixels[0][k])
            j = int(num_pixels[1][k])
            # sowing_actual = truth['sowing_dates'][i][j]
            # harvesting_actual = truth['harvesting_dates'][i][j]
            # input = make_input_patch(X_, i ,j , img_dimension[0])
            # #convert input to batch size 1
            # input = torch.tensor(input)
            # input = input[None, :, : , :, :]
            # inputs = input.permute(0, 1, 4, 2, 3)

            #SOWING
            args.model_name = 'unet-fc'
            args.num_timesteps=184
            args.use_s2 = True
            args.use_s1 = False
            args.use_planet = False
            args.use_l8 = True

            args.s2_bands = '[0,1,2,3,4,5,6,7,8,9]'
            args.l8_bands = '[0]'

            l8_bands, s1_bands, s2_bands, ps_bands = get_bands(args.l8_bands, args.s1_bands, args.s2_bands, args.ps_bands)
            img_dimension, crop_typ_dim =  get_dim(args.use_s1, args.use_s2, args.use_l8, args.use_planet)
            X_ = get_input_others(img_, img_dimension, l8_bands, s2_bands, s1_bands, ps_bands, args.include_indices)
            # print('before concat',X_.shape)
            X_ = concat_satellite(X_, args.use_s1, args.use_s2, args.use_l8, args.use_planet)
            inputs = get_model_input(X_, i, j, img_dimension[0])

            # model = croptype_models.get_model(input_shape=img_dimension, **vars(args))
            # if args.model_name in DL_MODELS:
            #     print('Total trainable model parameters: {}'.format(
            #         sum(p.numel() for p in model.parameters() if p.requires_grad)))
            # model.load_state_dict(torch.load('/home/parichya/Documents/model_weights/sowing_date_prediction.pth.tar'))
            preds = model_sow(inputs.cuda().float())
            print('for swoing')

            # print(preds.shape)

            Y[1][i][j] = int(torch.argmin(preds, dim=1)[0].item())
            print(torch.max(preds, dim=1)[0])
            print('for sowing over')
            args.l8_bands  = None
            args.s2_bands  = None

            #Transplantig
            args.use_s2 = False
            args.use_s1 = False
            args.use_planet = False
            args.use_l8 = True

            args.l8_bands = '[0,1,2,3,4,5,6]'

            l8_bands, s1_bands, s2_bands, ps_bands = get_bands(args.l8_bands, args.s1_bands, args.s2_bands, args.ps_bands)
            img_dimension, crop_typ_dim =  get_dim(args.use_s1, args.use_s2, args.use_l8, args.use_planet)
            # print('transplant img dim:', img_dimension)
            X_ = get_input_others(img_, img_dimension, l8_bands, s2_bands, s1_bands, ps_bands, args.include_indices)
            X_ = concat_satellite(X_, args.use_s1, args.use_s2, args.use_l8, args.use_planet)
            inputs = get_model_input(X_, i, j, img_dimension[0])

            # model = croptype_models.get_model(input_shape=img_dimension, **vars(args))
            # if args.model_name in DL_MODELS:
            #     print('Total trainable model parameters: {}'.format(
            #         sum(p.numel() for p in model.parameters() if p.requires_grad)))
            # model.load_state_dict(torch.load('/home/parichya/Documents/model_weights/date_prediction_best_val_f1(ethereal-grass-12 [L8-Trans(All Spectral)]).pth.tar'))
            preds = model_trans(inputs.cuda().float())

            Y[2][i][j] = int(torch.argmax(preds, dim=1)[0].item())
            args.l8_bands = None

            #Harvesting
            args.use_s2 = False
            args.use_s1 = True
            args.use_planet = False
            args.use_l8 = False

            args.s1_bands = '[0]'

            l8_bands, s1_bands, s2_bands, ps_bands = get_bands(args.l8_bands, args.s1_bands, args.s2_bands, args.ps_bands)
            img_dimension, crop_typ_dim =  get_dim(args.use_s1, args.use_s2, args.use_l8, args.use_planet)
            X_ = get_input_others(img_, img_dimension, l8_bands, s2_bands, s1_bands, ps_bands, args.include_indices)
            X_ = concat_satellite(X_, args.use_s1, args.use_s2, args.use_l8, args.use_planet)
            inputs = get_model_input(X_, i, j, img_dimension[0])
            # model = croptype_models.get_model(input_shape=img_dimension, **vars(args))
            # if args.model_name in DL_MODELS:
            #     print('Total trainable model parameters: {}'.format(
            #         sum(p.numel() for p in model.parameters() if p.requires_grad)))
            # model.load_state_dict(torch.load('/home/parichya/Documents/model_weights/date_prediction_best_val_f1(wise-forest-21 [S1-Harvest(VV)]).pth.tar'))
            preds = model_harv(inputs.cuda().float())
            # print('for harv')



            Y[3][i][j] = int(torch.argmin(preds, dim=1)[0].item())
            # print(Y[3][i][j])
            # print('harv over')
            args.s1_bands = None

            #YIELD
            args.use_s2 = True
            args.use_s1 = False
            args.use_planet = False
            args.use_l8 = True

            args.l8_bands = '[0]'
            args.s2_bands = '[0,1,2,3,4,5,6,7,8,9]'
            # args.model_name = 'unet-fc-yield'
            l8_bands, s1_bands, s2_bands, ps_bands = get_bands(args.l8_bands, args.s1_bands, args.s2_bands, args.ps_bands)
            img_dimension, crop_typ_dim =  get_dim(args.use_s1, args.use_s2, args.use_l8, args.use_planet)
            X_ = get_input_others(img_, img_dimension, l8_bands, s2_bands, s1_bands, ps_bands, args.include_indices)
            sowing = int(Y[1][i][j])-1 if int(Y[1][i][j]) >0 else 0
            harvesting = int(Y[3][i][j]) -1
            if args.use_actual_season:
                X_['s1'][:, :, :, :sowing] = torch.zeros_like(X_['s1'][:, :, :, :sowing])
                X_['s2'][:, :, :, :sowing] = torch.zeros_like(X_['s2'][:, :, :, :sowing])
                X_['l8'][:, :, :, :sowing] = torch.zeros_like(X_['l8'][:, :, :, :sowing])
                X_['planet'][:, :, :, :sowing] = torch.zeros_like(X_['planet'][:, :, :, :sowing])
                X_['s1'][:, :, :, harvesting+1:] = torch.zeros_like(X_['s1'][:, :, :, harvesting+1:])
                X_['s2'][:, :, :, harvesting+1:] = torch.zeros_like(X_['s2'][:, :, :, harvesting+1:])
                X_['l8'][:, :, :, harvesting+1:] = torch.zeros_like(X_['l8'][:, :, :, harvesting+1:])
                X_['planet'][:, :, :, harvesting+1:] = torch.zeros_like(X_['planet'][:, :, :, harvesting+1:])

                # X_['s1']=X_['s1'][:,:,:,sow_pred:harv_pred+1]
                # X_['s2']=X_['s2'][:,:,:,sow_pred:harv_pred+1]
                # X_['l8']=X_['l8'][:,:,:,sow_pred:harv_pred+1]
                # X_['planet']=X_['planet'][:,:,:,sow_pred:harv_pred+1]
                # args.num_timesteps=harv_pred-sow_pred+1
            X_ = concat_satellite(X_, args.use_s1, args.use_s2, args.use_l8, args.use_planet)
            inputs = get_model_input(X_, i, j, img_dimension[0])
            # model = croptype_models.get_model(input_shape=img_dimension, **vars(args))
            # if args.model_name in DL_MODELS:
            #     print('YYyyyyyyyYYYYYYYYYYYYYYYYY')
            #     print('Total trainable model parameters: {}'.format(
            #         sum(p.numel() for p in model.parameters() if p.requires_grad)))
            # model.load_state_dict(torch.load('/home/parichya/Documents/model_weights/crop_yield [Fusion(L8-UB,S2-All)].pth.tar'))
            preds = model_yield(inputs.cuda().float())


            Y[4][i][j] = preds[0].item()
            args.s2_bands = None
            args.l8_bands = None

        # print(Y[0])
        # print(Y[1])
        # print(Y[2])
        # print(Y[3])

        # Y = Y.numpy()
        print('sow avg:',np.average(Y[1][Y[1]>0]))
        print('harvest avg:',np.average(Y[3][Y[3]>0]))
        if np.average(Y[1][Y[1]>0])<np.average(Y[3][Y[3]>0]):
            print('saved')
            np.save('/home/parichya/Documents/deploy/deploy_test/deploy'+str(uid)+'.npy', Y)

    plot_arr((Y[0]))
    plot_arr(norm_max(Y[1]))
    plot_arr(norm_plot(Y[2]))
    plot_arr(norm_plot(Y[3]))
    plot_arr(norm_max(Y[4]))


if __name__ == "__main__":
    # parse args
    parser = util_deploy.get_parser()
    main(parser.parse_args())















