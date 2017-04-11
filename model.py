import glob
import pickle

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from lib import *


def get_image_files():
    car_path = "./data/vehicles/"
    not_car_path = "./data/non-vehicles/"

    car_files = glob.glob(car_path + '/**/*.png', recursive=True)
    y_car = np.ones(len(car_files), dtype=np.int)
    not_car_files = glob.glob(not_car_path + '/**/*.png', recursive=True)
    y_not_car = np.zeros(len(not_car_files), dtype=np.int)

    X = np.concatenate([car_files, not_car_files])
    y = np.concatenate([y_car, y_not_car])
    X_names, y = shuffle(X, y)
    return X_names, y

X_names, y = get_image_files()



colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 12
pix_per_cell = 8
cell_per_block = 1
hog_channel = 'ALL' # Can be 0, 1, 2 or "ALL"

X_gradient = extract_gradient_features(X_names, 
                             cspace=colorspace,
                             orient=orient,
                             pix_per_cell=pix_per_cell,
                             cell_per_block=cell_per_block,
                             hog_channel=hog_channel)

colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
spatial_size = (32, 32)
hist_bins = 100
hist_range=(0, 256)

X_color = extract_color_features(X_names,
                               cspace=colorspace,
                               spatial_size=spatial_size,
                               hist_bins=hist_bins,
                               hist_range=hist_range)

X = np.hstack((X_color, X_gradient))

X_scaler = StandardScaler().fit(X)
X = X_scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

svc = LinearSVC()
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)
print(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='binary'))

with open("svc.pkl", "wb") as f:
    pickle.dump(svc, f)
with open("X_scaler.pkl", "wb") as f:
    pickle.dump(X_scaler, f)
