from itertools import product
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC

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

print("Loading data")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

# ## we do some preprocessing. Namely:
# # Recast the images into images
# # We binerize it into 0.1 grayscale values
# # Keep the x, y with non-zero values
# X[X>0] = 1
# X = X.reshape((70000, 28, 28))
#
# X_index = np.nonzero(X)
#
# X_index = [0] * 70000
# for i in tqdm(range(70000)):
#     img = X[i]
#     x_arr, y_arr = np.nonzero(img)
#     X_index[i] = np.array([x_arr, y_arr]).T / 28
#
# # X_index = X_index[:1000]
# # y = y[:1000]
#
# quality = 20
# pixels = quality * quality
# spread = 0.05
# PI_vectors_H0 = np.zeros((len(X_index), pixels))
# PI_vectors_H1 = np.zeros((len(X_index), pixels))
# PI_vectors_Hcon = np.zeros((len(X_index), 2 * pixels))
# for i in tqdm(range(len(X_index))):
#     pim_0 = PersImage(spread=spread,
#                       pixels=[quality, quality],
#                       verbose=False,
#                       kernel_type="laplace",
#                       weighting_type="logistic",
#                       specs={
#                           "maxBD": 2,
#                           "minBD": 0,
#                       })
#     pim_1 = PersImage(spread=spread,
#                       pixels=[quality, quality],
#                       verbose=False,
#                       kernel_type="laplace",
#                       weighting_type="logistic",
#                        specs={
#                           "maxBD": 2,
#                           "minBD": 0,
#                       })
#
#     PI_data = X_index[i]
#     rips = Rips(verbose=False)
#     dgms = rips.fit_transform(PI_data)
#     PI_data_H0 = pim_0.transform(dgms[0])
#     PI_data_H1 = pim_1.transform(dgms[1])
#
#     PI_data_H1 = PI_data_H1.reshape(pixels)
#     PI_data_H0 = PI_data_H0.reshape(pixels)
#     PI_data_Hcon = np.concatenate((PI_data_H0, PI_data_H1))
#
#     PI_vectors_H0[i, :] = PI_data_H0
#     PI_vectors_H1[i, :] = PI_data_H1
#     PI_vectors_Hcon[i, :] = PI_data_Hcon
#
# with open("processed_data/PI_data_H0.npy", "wb") as f:
#     np.save(f, PI_vectors_H0)
#
# with open("processed_data/PI_data_H1.npy", "wb") as f:
#     np.save(f, PI_vectors_H1)
#
# with open("processed_data/PI_data_Hcon.npy", "wb") as f:
#     np.save(f, PI_vectors_Hcon)

print("Loading preprocessed data")
with open("processed_data/PI_data_H0.npy", "rb") as f:
    PI_vectors_H0 = np.load(f)

with open("processed_data/PI_data_H1.npy", "rb") as f:
    PI_vectors_H1 = np.load(f)

with open("processed_data/PI_data_Hcon.npy", "rb") as f:
    PI_vectors_Hcon = np.load(f)



X_train_H0, X_test_H0, y_train_H0, y_test_H0 = train_test_split(PI_vectors_H0, y, test_size=0.33)
X_train_H1, X_test_H1, y_train_H1, y_test_H1 = train_test_split(PI_vectors_H1, y, test_size=0.33)
X_train_con, X_test_con, y_train_con, y_test_con = train_test_split(PI_vectors_Hcon, y, test_size=0.33)


def do_training(PI_vectors_train, PI_vectors_test, y_H_train):
    print("Start training")
    X_H_train = PI_vectors_train
    X_H_test = PI_vectors_test

    rf = RandomForestClassifier(n_estimators=10000, verbose=3, n_jobs=-1).fit(X_H_train, y_H_train.ravel())
    lr = LogisticRegression(max_iter=10000, verbose=1).fit(X_H_train, y_H_train.ravel())
    sgd = SGDClassifier(max_iter=10000, verbose=1).fit(X_H_train, y_H_train.ravel())
    svm = SVC().fit(X_H_train, y_H_train.ravel())
    #ada = AdaBoostClassifier(n_estimators=10000).fit(X_H_train, y_H_train.ravel())

    y_H_hat_rf = rf.predict(X_H_test)
    y_H_hat_lr = lr.predict(X_H_test)
    y_H_hat_sgd = sgd.predict(X_H_test)
    y_H_hat_svm = svm.predict(X_H_test)
    # y_H_hat_ada = ada.predict(X_H_test)
    return y_H_hat_rf, y_H_hat_lr, y_H_hat_sgd, y_H_hat_svm
    # return y_H_hat_rf, y_H_hat_lr, y_H_hat_ada

def do_analysis(y_H, yhat1, yhat2, yhat3, yhat4, labels):
    report_rf = classification_report(y_H.ravel(),
                                     yhat1,
                                     labels=labels,
                                     target_names=labels,
                                     output_dict=True)
    report_lr = classification_report(y_H.ravel(),
                                     yhat2,
                                     labels=labels,
                                     target_names=labels,
                                     output_dict=True)
    report_sgd = classification_report(y_H.ravel(),
                                     yhat3,
                                     labels=labels,
                                     target_names=labels,
                                     output_dict=True)
    report_svm = classification_report(y_H.ravel(),
                                     yhat4,
                                     labels=labels,
                                     target_names=labels,
                                     output_dict=True)

    report_km_rf = pd.DataFrame(report_rf).transpose()
    report_km_lr = pd.DataFrame(report_lr).transpose()
    report_km_sgd = pd.DataFrame(report_sgd).transpose()
    report_km_svm = pd.DataFrame(report_svm).transpose()
    # report_km_ada = pd.DataFrame(report_ada).transpose()

    print(report_km_rf)
    print(report_km_lr)
    print(report_km_sgd)
    print(report_km_svm)
    # print(report_km_ada)

    return report_km_rf, report_km_lr, report_km_sgd, report_km_svm


y_H0_hat_rf, y_H0_hat_lr, y_H0_hat_sgd, y_H0_hat_svm = do_training(X_train_H0, X_test_H1, y_train_H0)
y_H1_hat_rf, y_H1_hat_lr, y_H1_hat_sgd, y_H1_hat_svm = do_training(X_train_H1, X_test_H1, y_train_H1)
y_con_hat_rf, y_con_hat_lr, y_con_hat_sgd, y_con_hat_svm = do_training(X_train_con, X_test_con, y_train_con)

report_H0 = do_analysis(y_test_H0, y_H0_hat_rf, y_H0_hat_lr, y_H0_hat_sgd, y_H0_hat_svm, list(range(10)))
report_H1 = do_analysis(y_test_H1, y_H1_hat_rf, y_H1_hat_lr, y_H1_hat_sgd, y_H1_hat_svm,  list(range(10)))
report_con = do_analysis(y_test_con, y_con_hat_rf, y_con_hat_lr, y_con_hat_sgd, y_con_hat_svm, list(range(10)))
report_H0[0].to_csv("result_MNIST/MNIST_report_HO_rf.csv")
report_H0[1].to_csv("result_MNIST/MNIST_report_HO_lr.csv")
report_H0[2].to_csv("result_MNIST/MNIST_report_HO_sgd.csv")
report_H0[2].to_csv("result_MNIST/MNIST_report_HO_svm.csv")

#
report_H1[0].to_csv("result_MNIST/MNIST_report_H1_rf.csv")
report_H1[1].to_csv("result_MNIST/MNIST_report_H1_lr.csv")
report_H1[2].to_csv("result_MNIST/MNIST_report_H1_sgd.csv")
report_H1[3].to_csv("result_MNIST/MNIST_report_H1_svm.csv")

#
report_con[0].to_csv("result_MNIST/MNIST_report_con_rf.csv")
report_con[1].to_csv("result_MNIST/MNIST_report_con_lr.csv")
report_con[2].to_csv("result_MNIST/MNIST_report_con_sgd.csv")
report_con[3].to_csv("result_MNIST/MNIST_report_con_svm.csv")

# report_H0.to_csv("result_MNIST/MNIST_report_HO_lr.csv")
# report_H1.to_csv("result_MNIST/MNIST_report_H1_lr.csv")
# report_con.to_csv("result_MNIST/MNIST_report_con_lr.csv")
