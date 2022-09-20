import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
# %matplotlib inline
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, mean_squared_error


# Function to normalize the grid values
def normalize(array):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    array_min, array_max = array.min(), array.max()
    return ((array - array_min)/(array_max - array_min))

# Normalize the bands
# redn = normalize(red)
# greenn = normalize(green)
# bluen = normalize(blue)
def norm_plot(a):
    row_sums = a.sum(axis=1)
    new_matrix = a / row_sums[:, np.newaxis]
    return new_matrix
def norm_max(a):
    return np.true_divide(a,np.amax(a))
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

def xx(a,b,c): print(f'F1:{a}, RMSE:{b}, acc:{c}')

# def gen_tru_mask(a,b):
#     reutrn np.multiply(a,b)

uid='2374'
Y=np.load('/home/parichya/Documents/deploy/deploy'+uid+'.npy')
print(type(Y[1]))
target = np.load('/home/parichya/Documents/cauvery/truth@10m/cauvery_00'+uid+'.npz')
image = np.load('/home/parichya/Documents/cauvery/npy/cauvery_00'+uid+'.npz')['s2']
# crop_type,sowing_date,transplanting_date,harvesting_date,crop_yield
#f1 rmse, acc
# print(np.unique(target['plot_id']))
mask_=np.where(target['plot_id']==432,1,0)
coords=np.argwhere(mask_>0)
x_min, y_min = np.min(coords,axis=0)
x_max, y_max = np.max(coords,axis=0)
# print('coorsd', coords)
# print(np.unique(mask_))
tcp=target['crop_type'][x_min:x_max+1, y_min:y_max+1]
print(tcp.shape)
tsd=target['sowing_date'][x_min:x_max+1, y_min:y_max+1]
print(tsd.shape)
ttd=target['transplanting_date'][x_min:x_max+1, y_min:y_max+1]
thd=target['harvesting_date'][x_min:x_max+1, y_min:y_max+1]
tcy=target['crop_yield'][x_min:x_max+1, y_min:y_max+1]
print((tcy.shape))
print(Y[0].shape)
pred_0=Y[0][x_min:x_max+1, y_min:y_max+1]
pred_1=Y[1][x_min:x_max+1, y_min:y_max+1]
pred_2=Y[2][x_min:x_max+1, y_min:y_max+1]
pred_3=Y[3][x_min:x_max+1, y_min:y_max+1]
pred_4=Y[4][x_min:x_max+1, y_min:y_max+1]


a,b,c= metrics(tcp,pred_0)
xx(a,b,c)
print('sowing_date')
a,b,c= metrics(tsd,pred_1)
xx(a,b,c)
print('transplanting_date')
a,b,c= metrics(ttd,pred_2)
xx(a,b,c)
print('harvesting_date')
print(np.unique(Y[3]*mask_))
print(np.unique(thd))
a,b,c= metrics(thd,pred_3)
xx(a,b,c)
print('YIELD')
print(pred_4)
print(tcy)
a,b,c= metrics(tcy,pred_4)
xx(a,b,c)

# plot_arr((Y[0]))
# plot_arr(norm_max(Y[1]))
# plot_arr(norm_plot(Y[2]))
# plot_arr(norm_plot(Y[3]))
# # print(np.any(Y[4]))
# plot_arr(norm_max(Y[4]))



