from itertools import product
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import classification_report
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
import pandas as pd

from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
from tqdm import tqdm
from ripser import Rips
from persim import PersImage

def F(arr, r):
    x = arr[0]
    y = arr[1]
    x_new = np.mod(x + r * y * (1 - y), 1)
    return np.array([x_new, y])

def G(arr, r):
    x = arr[0]
    y = arr[1]
    y_new = np.mod(y + r * x * (1 - x), 1)
    y_new = y_new - np.floor(y_new)
    return np.array([x, y_new])

def n_step_dynamic_system(n, x, y, r):
    result = np.zeros((n, 2))
    result[0, :] = [x, y]
    for i in range(1, n):
        # x_step = np.sqrt(2 - result[i - 1, 0])
        # y_step = r * (result[i - 1, 0] + np.sqrt(x_step))
        # x_step = result[i - 1, 0] + r * result[i - 1, 1] * (1 - result[i - 1, 1])
        # y_step = result[i - 1, 1] + r * result[i - 1, 0] * (1 - result[i - 1, 0])
        x_step, y_step = G(F(result[i-1, :], r), r)
        # x_step = result[i - 1, 0] * np.exp(r * (1 - result[i - 1, 1]))

        result[i, :] = [x_step, y_step]
    return result


def do_training(PI_vectors_train, PI_vectors_test, y_H_train):
    X_H_train = PI_vectors_train
    X_H_test = PI_vectors_test

    ada = AdaBoostClassifier(n_estimators=1000, random_state=42).fit(X_H_train, y_H_train.ravel())
    grad = GradientBoostingClassifier(n_estimators=1000).fit(X_H_train, y_H_train.ravel())

    y_H_hat_ada = ada.predict(X_H_test)
    y_H_hat_grad = grad.predict(X_H_test)
    return y_H_hat_ada, y_H_hat_grad

def do_analysis(y_H, yhat1, yhat2, labels):
    report_ada = classification_report(y_H.ravel(),
                                         yhat1,
                                         labels=list(range(5)),
                                         target_names=labels,
                                         output_dict=True)
    report_grad = classification_report(y_H.ravel(),
                                         yhat2,
                                         labels=list(range(5)),
                                         target_names=labels,
                                         output_dict=True)

    report_km_ada = pd.DataFrame(report_ada).transpose()
    report_km_grad = pd.DataFrame(report_grad).transpose()

    print(report_km_ada)
    print(report_km_grad)

    return report_km_ada, report_km_grad


def do_full_run(data, quality=50, spread=0.05, kernel="gaussian", weighting="linear"):
    """
    Does the full PI analysis and some training
    saves the results in the results folder
    :param quality: resulution (int)
    :param spread:  variance in smoothing (float)
    :param kernel:  cdf to be used for smoothing (string: gaussian, laplace, lognorm, gamma)
    :param weighting: weighting to be used (string: linear, pm_linear, logistic)
    :return: reports for H0 and H1
    """
    pixels = quality * quality
    PI_vectors_H0 = np.zeros((5 * m, pixels))
    PI_vectors_H1 = np.zeros((5 * m, pixels))
    PI_vectors_con = np.zeros((5 * m, 2 * pixels))

    target = np.zeros(5 * m)
    labels = [2, 3.5, 4.0, 4.1, 4.3]
    index = 0
    print("Creating the PI images for the different r's")
    for i in range(5):
        shape_data = data[i]
        r = labels[i]
        print("training for r:{}".format(r))
        for j in tqdm(range(m)):
            pim_0 = PersImage(spread=spread,
                              pixels=[quality, quality],
                              kernel_type=kernel,
                              weighting_type=weighting,
                              verbose=False,
                              specs={
                                  "maxBD": 2,
                                  "minBD": 0,
                              })
            pim_1 = PersImage(spread=spread,
                              pixels=[quality, quality],
                              kernel_type=kernel,
                              weighting_type=weighting,
                              verbose=False,
                              specs={
                                  "maxBD": 2,
                                  "minBD": 0,
                              })

            PI_data = shape_data[j, :, :]
            rips = Rips(verbose=False)
            dgms = rips.fit_transform(PI_data)

            PI_data_H0 = pim_0.transform(dgms[0])
            PI_data_H1 = pim_1.transform(dgms[1])

            PI_data_H1 = PI_data_H1.reshape(pixels)
            PI_data_H0 = PI_data_H0.reshape(pixels)

            # PI_data_H0 = PI_data_H0[:, None]
            # PI_data_H1 = PI_data_H1[:, None]
            #
            target[index] = int(i)

            PI_vectors_H0[index, :] = PI_data_H0
            PI_vectors_H1[index, :] = PI_data_H1
            PI_vectors_con[index, :] = np.concatenate((PI_data_H0, PI_data_H1))
            index += 1

    PI_vectors_H0_train, PI_vectors_H0_test, y_train_H0, y_test_H0 = train_test_split(PI_vectors_H0,
                                                                                      target,
                                                                                      test_size=0.33)
    PI_vectors_H1_train, PI_vectors_H1_test, y_train_H1, y_test_H1 = train_test_split(PI_vectors_H1,
                                                                                      target,
                                                                                      test_size=0.33)
    PI_vectors_con_train, PI_vectors_con_test, y_train_con, y_test_con = train_test_split(PI_vectors_con,
                                                                                          target,
                                                                                          test_size=0.33)

    y_H0_hat_ada, y_H0_hat_grad = do_training(PI_vectors_H0_train, PI_vectors_H0_test, y_train_H0)
    y_H1_hat_ada, y_H1_hat_grad = do_training(PI_vectors_H1_train, PI_vectors_H1_test, y_train_H1)
    y_con_hat_ada, y_con_1_hat_grad = do_training(PI_vectors_con_train, PI_vectors_con_test, y_train_con)


    reports_H0_ada, reports_H0_grad = do_analysis(y_test_H0, y_H0_hat_ada, y_H0_hat_grad, labels)
    reports_H1_ada, reports_H1_grad = do_analysis(y_test_H1, y_H1_hat_ada, y_H1_hat_grad, labels)
    reports_con_ada, reports_con_grad = do_analysis(y_test_con, y_con_hat_ada, y_con_1_hat_grad, labels)

    reports_H0_ada.to_csv("results_dynamic/{}_{}_{}_{}_H0_ada.csv".format(quality, spread, kernel, weighting))
    reports_H0_grad.to_csv("results_dynamic/{}_{}_{}_{}_H0_grad.csv".format(quality, spread, kernel, weighting))
    reports_H1_ada.to_csv("results_dynamic/{}_{}_{}_{}_H1_ada.csv".format(quality, spread, kernel, weighting))
    reports_H1_grad.to_csv("results_dynamic/{}_{}_{}_{}_H1_grad.csv".format(quality, spread, kernel, weighting))
    reports_con_ada.to_csv("results_dynamic/{}_{}_{}_{}_con_ada.csv".format(quality, spread, kernel, weighting))
    reports_con_grad.to_csv("results_dynamic/{}_{}_{}_{}_con_grad.csv".format(quality, spread, kernel, weighting))

    return reports_H0_ada, reports_H0_grad, reports_H1_ada, reports_H1_grad, reports_con_ada, reports_con_grad


rs = [2, 3.5, 4, 4.1, 4.3]
m, n = 400, 1000
data = np.zeros((len(rs), m, n, 2))
for i in range(len(rs)):
    r = rs[i]
    print("training for r:{}".format(r))
    for j in tqdm(range(m)):
        x, y = np.random.uniform(size=2)
        data[i, j, :, :] = n_step_dynamic_system(n, x, y, r)

#_, _, _, _, _, _ = do_full_run(data, quality=20, spread=0.05, kernel="gaussian", weighting="linear")
#_, _, _, _, _, _ = do_full_run(data, quality=20, spread=0.05, kernel="laplace", weighting="linear")
#_, _, _, _, _, _ = do_full_run(data, quality=20, spread=0.05, kernel="gamma", weighting="linear")
_, _, _, _, _, _ = do_full_run(data, quality=20, spread=0.05, kernel="laplace", weighting="logistic")
#_, _, _, _, _, _ = do_full_run(data, quality=20, spread=0.05, kernel="gamma", weighting="logistic")
_, _, _, _, _, _ = do_full_run(data, quality=20, spread=0.05, kernel="gaussian", weighting="logistic")

