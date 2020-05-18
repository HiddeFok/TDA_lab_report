from itertools import product
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd

from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
from tqdm import tqdm
from ripser import Rips
from persim import PersImage

from gtda.images import HeightFiltration, DilationFiltration, RadialFiltration
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceImage
from sklearn.datasets import fetch_openml


print("Loading data")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

X = X[:40000]
y = y[:40000]
X_copy = np.copy(X)

X[X <= .4] = 0
X[X > .4] = 1
X = X.reshape((40000, 28, 28))

X_bin = X

# X_index = np.zeros((40000, 28, 28))
# for i in tqdm(range(40000)):
#     img = X[i]
#     x_arr, y_arr = np.nonzero(img)
#     X_index[i] = np.array([x_arr, y_arr]).T
#
# hf = HeightFiltration()
# df = DilationFiltration()
# rf = RadialFiltration(np.array([8, 15]))
# X_used = X_bin
# y_used = y
#
# ## Create the Filtrations
# X_hf = hf.fit_transform(X_used, y_used)
# X_df = df.fit_transform(X_used, y_used)
# X_rf = rf.fit_transform(X_used, y_used)
#
# fig = plt.figure()
# fig.set_size_inches((12, 9))
# a = fig.add_subplot(2, 2, 1)
# a.set_title("Height filtration")
# imgplot = plt.imshow(X_hf[0], cmap="viridis")
# a = fig.add_subplot(2,2, 2)
# a.set_title("Dilation filtration")
# imgplot = plt.imshow(X_df[0], cmap="viridis")
# a = fig.add_subplot(2,2, 3)
# a.set_title("Radial filtration")
# imgplot = plt.imshow(X_bin[0], cmap="binary")
# a = fig.add_subplot(2,2, 4)
# a.set_title("Radial filtration")
# imgplot = plt.imshow(X_rf[0], cmap="viridis")
# plt.colorbar(orientation="vertical")
# plt.savefig("figures/MNINST_example.png")
#
#
# index = np.zeros((28 * 28, 2))
# k = 0
# for i in range(28):
#     for j in range(28):
#         index[k] = np.array([i, j])
#         k += 1
# index = index.astype(int)
# quality = 10
# pixels = quality * quality
# spread = 0.05
# PI_vectors = np.zeros((40000, 8 * quality * quality))
# for i in tqdm(range(X_bin.shape[0])):
#     bin_vec = X_bin[i]
#     hf_vec = X_hf[i]
#     df_vec = X_df[i]
#     rf_vec = X_rf[i]
#     sampled_index = np.random.choice(len(index), 100, replace=False)
#     sampled_index = index[sampled_index].astype(int)
#
#     sampled_bin_vec = np.concatenate([sampled_index, bin_vec[sampled_index[:, 0], sampled_index[:, 1], None]], axis=1)
#     sampled_hf_vec = np.concatenate([sampled_index, hf_vec[sampled_index[:, 0], sampled_index[:, 1], None]], axis=1)
#     sampled_df_vec = np.concatenate([sampled_index, df_vec[sampled_index[:, 0], sampled_index[:, 1], None]], axis=1)
#     sampled_rf_vec = np.concatenate([sampled_index, rf_vec[sampled_index[:, 0], sampled_index[:, 1], None]], axis=1)
#     pim = PersImage(spread=spread,
#                       pixels=[quality, quality],
#                       kernel_type="gaussian",
#                       weighting_type="logistic",
#                       verbose=False,
#                       specs={
#                           "maxBD": 2,
#                           "minBD": 0,
#                       })
#
#     rips = Rips(verbose=False)
#     dgms_bin = rips.fit_transform(sampled_bin_vec)
#     dgms_hf = rips.fit_transform(sampled_hf_vec)
#     dgms_df = rips.fit_transform(sampled_df_vec)
#     dgms_rf = rips.fit_transform(sampled_rf_vec)
#
#     sampled_bin_vec_H0 = pim.transform(dgms_bin[0]).reshape(100)
#     sampled_bin_vec_H1 = pim.transform(dgms_bin[1]).reshape(100)
#
#     sampled_hf_vec_H0 = pim.transform(dgms_hf[0]).reshape(100)
#     sampled_hf_vec_H1 = pim.transform(dgms_hf[1]).reshape(100)
#
#     sampled_df_vec_H0 = pim.transform(dgms_df[0]).reshape(100)
#     sampled_df_vec_H1 = pim.transform(dgms_df[1]).reshape(100)
#
#     sampled_rf_vec_H0 = pim.transform(dgms_rf[0]).reshape(100)
#     sampled_rf_vec_H1 = pim.transform(dgms_rf[1]).reshape(100)
#
#     con = [
#         sampled_bin_vec_H0,
#         sampled_bin_vec_H1,
#         sampled_hf_vec_H0,
#         sampled_hf_vec_H1,
#         sampled_df_vec_H0,
#         sampled_df_vec_H1,
#         sampled_rf_vec_H0,
#         sampled_rf_vec_H1,
#     ]
#     sampled_con = np.concatenate(con)
#     PI_vectors[i] = sampled_con

# with open("processed_data/PI_data_everything.npy", "wb") as f:
#     np.save(f, PI_vectors)

with open("processed_data/PI_data_everything.npy", "rb") as f:
    PI_vectors = np.load(f)

X_train, X_test, y_train, y_test = train_test_split(PI_vectors, y, test_size=0.25)


def do_training(PI_vectors_train, PI_vectors_test, y_H_train):
    print("Start training")
    X_H_train = PI_vectors_train
    X_H_test = PI_vectors_test

    rf = RandomForestClassifier(n_estimators=10000, verbose=3, n_jobs=-1).fit(X_H_train, y_H_train.ravel())
    lr = LogisticRegression(max_iter=10000, verbose=1, n_jobs=-1).fit(X_H_train, y_H_train.ravel())
    sgd = SGDClassifier(max_iter=10000, verbose=1, n_jobs=-1).fit(X_H_train, y_H_train.ravel())
    svm = SVC().fit(X_H_train, y_H_train.ravel())
    #ada = AdaBoostClassifier(n_estimators=10000).fit(X_H_train, y_H_train.ravel())

    y_H_hat_rf = rf.predict(X_H_test)
    y_H_hat_lr = lr.predict(X_H_test)
    y_H_hat_sgd = sgd.predict(X_H_test)
    y_H_hat_svm = svm.predict(X_H_test)
    return y_H_hat_rf, y_H_hat_lr, y_H_hat_sgd, y_H_hat_svm

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

    return report_km_rf, report_km_lr, report_km_sgd, report_km_svm


y_hat_rf, y_hat_lr, y_hat_sgd, y_hat_svm = do_training(X_train, X_test, y_train)

report = do_analysis(y_test, y_hat_rf, y_hat_lr, y_hat_sgd, y_hat_svm, list(range(10)))
report[0].to_csv("result_MNIST/MNIST_report_rf.csv")
report[1].to_csv("result_MNIST/MNIST_report_lr.csv")
report[2].to_csv("result_MNIST/MNIST_report_sgd.csv")
report[3].to_csv("result_MNIST/MNIST_report_svm.csv")

