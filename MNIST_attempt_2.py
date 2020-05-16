from itertools import product
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
import pandas as pd

from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
from tqdm import tqdm
from ripser import Rips
from persim import PersImage
from sklearn.datasets import fetch_openml

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

## we do some preprocessing. Namely:
# Recast the images into images
# We binerize it into 0.1 grayscale values
# Keep the x, y with non-zero values
X[X>0] = 1
X = X.reshape((70000, 28, 28))

X_index = np.nonzero(X)

X_index = [0] * 70000
for i in tqdm(range(70000)):
    img = X[i]
    x_arr, y_arr = np.nonzero(img)
    X_index[i] = np.array([x_arr, y_arr]).T

X_index = X_index[:1000]
y = y[:1000]

quality = 20
pixels = quality * quality
spread = 0.05
PI_vectors_H0 = np.zeros((len(X_index), pixels))
PI_vectors_H1 = np.zeros((len(X_index), pixels))
PI_vectors_Hcon = np.zeros((len(X_index), 2 * pixels))
for i in tqdm(range(len(X_index))):
    pim_0 = PersImage(spread=spread,
                      pixels=[quality, quality],
                      verbose=False,
                      kernel_type="laplace",
                      weighting_type="logistic",
                      specs={
                          "maxBD": 2,
                          "minBD": 0,
                      })
    pim_1 = PersImage(spread=spread,
                      pixels=[quality, quality],
                      verbose=False,
                      kernel_type="laplace",
                      weighting_type="logistic",
                      specs={
                          "maxBD": 2,
                          "minBD": 0,
                      })

    PI_data = X_index[i]
    rips = Rips(verbose=False)
    dgms = rips.fit_transform(PI_data)
    PI_data_H0 = pim_0.transform(dgms[0])
    PI_data_H1 = pim_1.transform(dgms[1])

    PI_data_H1 = PI_data_H1.reshape(pixels)
    PI_data_H0 = PI_data_H0.reshape(pixels)
    PI_data_Hcon = np.concatenate((PI_data_H0, PI_data_H1))

    PI_vectors_H0[i, :] = PI_data_H0
    PI_vectors_H1[i, :] = PI_data_H1
    PI_vectors_Hcon[i, :] = PI_data_Hcon


X_train_H0, X_test_H0, y_train_H0, y_test_H0 = train_test_split(PI_vectors_H0, y, test_size=0.33)
X_train_H1, X_test_H1, y_train_H1, y_test_H1 = train_test_split(PI_vectors_H0, y, test_size=0.33)
X_train_con, X_test_con, y_train_con, y_test_con = train_test_split(PI_vectors_H0, y, test_size=0.33)


def do_training(PI_vectors_train, PI_vectors_test, y_H_train):
    X_H_train = PI_vectors_train
    X_H_test = PI_vectors_test

    rf = RandomForestClassifier(n_estimators=10000).fit(X_H_train, y_H_train.ravel())

    y_H_hat_rf = rf.predict(X_H_test)
    return y_H_hat_rf

def do_analysis(y_H, yhat1, labels):
    report_rf = classification_report(y_H.ravel(),
                                     yhat1,
                                     labels=labels,
                                     target_names=labels,
                                     output_dict=True)

    report_km_rf = pd.DataFrame(report_rf).transpose()
    print(report_km_rf)

    return report_km_rf


y_H0_hat_rf = do_training(X_train_H0, X_test_H1, y_train_H0)
y_H1_hat_rf = do_training(X_train_H1, X_test_H1, y_train_H1)
y_con_hat_rf = do_training(X_train_con, X_test_con, y_train_con)

report_H0 = do_analysis(y_test_H0, y_H0_hat_rf, list(range(10)))
report_H1 = do_analysis(y_test_H1, y_H1_hat_rf, list(range(10)))
report_con = do_analysis(y_test_con, y_con_hat_rf, list(range(10)))