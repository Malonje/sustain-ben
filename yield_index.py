
from pathlib import Path
import shutil
import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
from torchvision import transforms
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import tarfile
import datetime
from torch.utils.data import DataLoader
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
import torch
import pandas as pd
from torch.autograd import Variable

BANDS = {'l8': {'8': {'UBLUE': 0, 'BLUE': 1, 'GREEN': 2, 'RED': 3, 'NIR': 4, 'SWIR1': 5, 'SWIR2': 6, 'THERMAL': 7}},
         's1': {'VV': 0, 'VH': 1, 'RATIO': 2},
         's2': {'10': {'BLUE': 0, 'GREEN': 1, 'RED': 2, 'RDED1': 3, 'RDED2': 4, 'RDED3': 5, 'NIR': 6, 'RDED4': 7,
                       'SWIR1': 8, 'SWIR2': 9},
                '4': {'BLUE': 0, 'GREEN': 1, 'RED': 2, 'NIR': 3}},
         'planet': {'4': {'BLUE':   0, 'GREEN': 1, 'RED': 2, 'NIR': 3}}}

n_bands = {
    's2': '10',
    'l8': '8',
    'planet': '4',
}

def calc_index(index, img, sat):
    if index == 'gcvi':
        computed_index = (img[BANDS[sat][n_bands[sat]]['NIR']] / img[BANDS[sat][n_bands[sat]]['GREEN']]) - 1
    elif index == 'ndvi':
        computed_index = (img[BANDS[sat][n_bands[sat]]['NIR']] - img[BANDS[sat][n_bands[sat]]['RED']]) / (
                    img[BANDS[sat][n_bands[sat]]['NIR']] + img[BANDS[sat][n_bands[sat]]['RED']])
    elif index == 'recl':
        computed_index = (img[BANDS[sat][n_bands[sat]]['NIR']] / img[BANDS[sat][n_bands[sat]]['RED']]) - 1
    ans = []
    computed_index = computed_index.transpose(2, 0, 1)
    for t in range(computed_index.shape[0]):
        ans.append(np.mean(computed_index[t]))
    ans=np.asarray(ans)
    return ans


class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

class IndicesDataset():
    """Indices dataset."""

    def __init__(self,  data_dir, country, split, satellite, for_indice, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.split = split
        self.data_dir = data_dir
        self.transform = transform
        self.country = country
        self.for_indice = for_indice
        self.satellite = satellite
        self.csv = pd.read_csv(self.data_dir+self.country+'/cauvery_dataset.csv')
        self.csv = self.csv[self.csv['SPLIT_YIELD']==self.split].reset_index(drop=True)

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # img_name = os.path.join(self.root_dir,cauevry_00000locid.npz
        #                         self.csv.iloc[idx, 0])
        u_id = self.csv['UNIQUE_ID'][idx]
        loc_id = f'{int(u_id):06d}'
        images = np.load(os.path.join(self.data_dir, self.country, 'npy', f'{self.country}_{loc_id}.npz'))

        s1 = images['s1'].astype(np.int64)
        s2 = images['s2'].astype(np.int64)
        planet = images['planet'].astype(np.int64)
        l8 = images['l8'].astype(np.int64)
        mask = np.load(os.path.join(self.data_dir, self.country, f'truth@10m', f'{self.country}_{loc_id}.npz'))['plot_id']
        mask_l8 = np.load(os.path.join(self.data_dir, self.country, 'truth@30m', f'{self.country}_{loc_id}.npz'))['plot_id']
        mask_ps = np.load(os.path.join(self.data_dir, self.country, 'truth@3m', f'{self.country}_{loc_id}.npz'))['plot_id']
        plot_id =  self.csv['PLOT_ID'][idx]

        coords = np.argwhere(mask==plot_id)
        x_min, y_min = np.min(coords,axis=0)
        x_max, y_max = np.max(coords,axis=0)

        s1 = (mask * s1.transpose(3,0,1,2))[ : , : , x_min:x_max+1, y_min:y_max+1]
        s2 = (mask * s2.transpose(3,0,1,2))[ : , : , x_min:x_max+1, y_min:y_max+1]
        coords = np.argwhere(mask_l8==plot_id)
        if len(coords) == 0:
            l8 = np.zeros((l8.shape[3], l8.shape[0], s2.shape[2], s2.shape[3]))
        else:
            x_min, y_min = np.min(coords,axis=0)
            x_max, y_max = np.max(coords,axis=0)
            l8 = (mask_l8 * l8.transpose(3,0,1,2))[ : , : , x_min:x_max+1, y_min:y_max+1]
        coords = np.argwhere(mask_ps==plot_id)
        x_min, y_min = np.min(coords,axis=0)
        x_max, y_max = np.max(coords,axis=0)
        planet = (mask_ps * planet.transpose(3,0,1,2))[ : , : , x_min:x_max+1, y_min:y_max+1]


        #APPLY FUNCTION TO CALCULATE INDICES
        # if self.transform:
        #     sample = self.transform(sample)

        Y = self.csv['YIELD'][idx]


        if self.satellite == 'l8':
            X = l8
        if self.satellite == 's1':
            X = s1
        if self.satellite == 's2':
            X = s2
        if self.satellite == 'planet':
            X = planet

        #CLACULATE INCIDES
        X = X.transpose(1, 2, 3, 0)
        X = calc_index(self.for_indice, X, self.satellite)


        return [X, Y]




train_dataset = IndicesDataset(data_dir= '../Dataset/', country='cauvery',
                               split='train', satellite='l8', for_indice='recl')
test_dataset = IndicesDataset(data_dir= '../Dataset/', country='cauvery',
                               split='test', satellite='l8', for_indice='recl')

train_len = len(train_dataset)
test_len = len(test_dataset)
# val_len =
# test_len =
print(train_len, test_len)
train_loader = DataLoader(train_dataset, batch_size=train_len,shuffle=True, num_workers=0)
test_loader  = DataLoader(test_dataset, batch_size=test_len,shuffle=True, num_workers=0)


# X_train=np.nan_to_num(X_train)
# regr = SGDRegressor()
# regr.fit(X_train, y_train)
# print('start')
# regr = LinearRegression()
regr = linear_model.Lasso(alpha=0.1)
rmse = []
for X_train,y_train in tqdm(train_loader):
    # X_train=np.save('/home/parichya/Documents/X_l8.npy')
    # y_train=np.save('/home/parichya/Documents/Y_l8.npy')
    # print('saved')
    X_train=np.nan_to_num(X_train)
    y_train = y_train.numpy()
    # print(X)
    # print(X[1])
    # print(Y.shape)
    # X_train, y_train = X, Y
    # print(X_train.shape)
    # print(1)
    regr.fit(X_train, y_train)
    pred = regr.predict(X_train)
    # y_train = y_train.numpy()
    rmse.append(np.sqrt(np.mean((y_train - pred) ** 2)))
print('RMSE Train: ', sum(rmse) / len(rmse))
    # print('Regression Score Train: ', regr.score(X_train, y_train))
# print(3)
# true=np.asarray(y_test).flatten()
rmse = []
for X_test,y_test in tqdm(test_loader):
    X_test=np.nan_to_num(X_test)
    y_test = y_test.numpy()
    print(X_test.shape)
    pred = regr.predict(X_test)
    rmse.append(np.sqrt(np.mean((y_test - pred) ** 2)))
print('RMSE Test: ', sum(rmse) / len(rmse))
# print('Regression Score Test: ', regr.score(X_test, y_test))
# pred = reg.predict
# pred = np.asarray(reg).flatten()
# rmse = np.sqrt(np.mean((true - pred) ** 2))
# print(regr.score(X_test, y_test))
#


# inputDim = 184       # takes variable 'x'
# outputDim = 1       # takes variable 'y'
# learningRate = 0.1
# epochs = 100
#
# model = linearRegression(inputDim, outputDim)
# model=model.float()
# ##### For GPU #######
# # if torch.cuda.is_available():
# #     model.cuda()
# criterion = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
#
#
# for epoch in range(epochs):
#     # print(type(inputs))
#     inputs = X_train
#     labels = y_train
#     print(inputs.shape)
#     inputs = torch.Tensor(inputs)
#     labels = torch.Tensor(labels)
#     # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
#     optimizer.zero_grad()
#
#     # get output from the model, given the inputs
#     outputs = model(inputs.float())
#     outputs_ = outputs.detach().numpy()
#     # pred_= pred_.cpu().detach().numpy()
#     rmse_ = np.sqrt(np.mean((y_train - outputs_) ** 2))
#     # get loss for the predicted output
#     loss = criterion(outputs, labels)
#     # print(loss)
#     # get gradients w.r.t to parameters
#     loss.backward()
#
#     # update parameters
#     optimizer.step()
#
#     print('epoch {}, loss {} RMSE {}'.format(epoch, loss.item(), rmse_))
# X_test = torch.Tensor(X_test)
# pred_ = model(X_test)
# # y_test = y_test.numpy()
# pred_ = pred_.detach().numpy()
# # pred_= pred_.cpu().detach().numpy()
# rmse_ = np.sqrt(np.mean((y_test - pred_) ** 2))
# print('RMSE Test nn: ', rmse_)
# # print('Regression Score: ', regr.score(X_test, y_test))

