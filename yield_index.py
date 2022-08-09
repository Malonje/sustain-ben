
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
from sklearn.linear_model import SGDRegressor
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
def calc_gcvi(img, sat):
    # print(img)
    h = img.shape[1]
    w = img.shape[2]
    num_pixels = h * w
    # print(img.shape)
    # print(num_pixels)
    if sat == 's2':
        gcvi = (img[BANDS['s2']['10']['NIR']] / img[BANDS['s2']['10']['GREEN']]) - 1
    if sat == 'l8':
        gcvi = (img[BANDS['l8']['8']['NIR']] / img[BANDS['l8']['8']['GREEN']]) - 1
    if sat == 's1':
        gcvi = (img[BANDS['planet']['4']['NIR']] / img[BANDS['planet']['4']['GREEN']]) - 1
    # gcvi = gcvi / num_pixels
    ans = []
    # print(gcvi.shape)
    gcvi = gcvi.transpose(2, 0, 1)
    # print(gcvi.shape)
    for k in range(gcvi.shape[0]):
        # print(k)
        # sum=0
        # for i in range(gcvi.shape[0]):
        #     for j in range(gcvi.shape[1]):
        #         sum += gcvi[i][j][k]
        # print(gcvi[k].shape)
        ans.append(np.average(gcvi[k])/num_pixels)
    # ans = torch.Tensor(ans)
    # print(ans)
    ans=np.asarray(ans)
    # print(type(ans))
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
        self.csv = self.csv[self.csv['SPLIT']==self.split].reset_index(drop=True)

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
        l8 = (mask_l8 * l8.transpose(3,0,1,2))[ : , : , x_min:x_max+1, y_min:y_max+1]
        planet = (mask_ps * planet.transpose(3,0,1,2))[ : , : , x_min:x_max+1, y_min:y_max+1]


        if l8.shape[2] == 0 or l8.shape[3] == 0:
            l8 = np.zeros((l8.shape[0], l8.shape[1], s2.shape[2], s2.shape[3]))

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
        if self.for_indice == 'gcvi':
            X = calc_gcvi(X, self.satellite)


        return [X, Y]




train_dataset = IndicesDataset(data_dir= '/home/parichya/Documents/', country='cauvery',
                               split='train', satellite='s2', for_indice='gcvi')
test_dataset = IndicesDataset(data_dir= '/home/parichya/Documents/', country='cauvery',
                               split='test', satellite='s2', for_indice='gcvi')

train_len = 1919
test_len = 500
# val_len =
# test_len =
train_loader = DataLoader(train_dataset, batch_size=train_len,shuffle=True, num_workers=0)
test_loader  = DataLoader(test_dataset, batch_size=test_len,shuffle=True, num_workers=0)
# for X,Y in tqdm(train_loader):
#     X= X.numpy()
#     Y=Y.numpy()
#     np.save('/home/parichya/Documents/X.npy', X)
#     np.save('/home/parichya/Documents/Y.npy', Y)
#     # X, Y =
X=np.load('/home/parichya/Documents/X.npy')
Y=np.load('/home/parichya/Documents/Y.npy')



X=np.nan_to_num(X)
# print(X)
# print(X[1])
# print(Y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25)
print(X_train.shape)
regr = SDGRegressor()
# print(1)
regr.fit(X_train, y_train)
# print(3)
true=np.asarray(y_test).flatten()
pred = np.asarray(reg).flatten()
rmse = np.sqrt(np.mean((true - pred) ** 2))
print(regr.score(X_test, y_test))
#


inputDim = 184       # takes variable 'x'
outputDim = 1       # takes variable 'y'
learningRate = 0.1
epochs = 100

model = linearRegression(inputDim, outputDim)
model=model.float()
##### For GPU #######
# if torch.cuda.is_available():
#     model.cuda()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

for epoch in range(epochs):
    # for x_train, y_train in tqdm(train_loader):

    # Converting inputs and labels to Variable
    # if torch.cuda.is_available():
    #     inputs = Variable(torch.from_numpy(x_train).cuda())
    #     labels = Variable(torch.from_numpy(y_train).cuda())
    # print(x_train.shape)
    # print(x_train)

    inputs = X#x_train
    labels = Y#y_train
    print(inputs.shape)
    inputs = torch.Tensor(inputs)
    labels = torch.Tensor(labels)
    print(type(inputs))
    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
    optimizer.zero_grad()

    # get output from the model, given the inputs
    outputs = model(inputs.float())

    # get loss for the predicted output
    loss = criterion(outputs, labels)
    print(loss)
    # get gradients w.r.t to parameters
    loss.backward()

    # update parameters
    optimizer.step()

    print('epoch {}, loss {}'.format(epoch, loss.item()))



