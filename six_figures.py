from itertools import product
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
import pandas as pd

from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
from tqdm import tqdm

from ripser import Rips
from persim import PersImage

## function to create the 6 figures
## Needed to replicate the results in the paper

def gen_six_figures(n: int, m: int, V: float):
    """
    This function generates the following 6 figures:
        - Random point cloud in R3
        - circle
        - sphere
        - 3 clusters
        - 3 clusters each with 3 clusters
        - torus

    :param n: number of data points in each figure
    :param m: number of examples per figure
    :param V: Variance parameter for the error sampling
    :return: np.array with the data points dim: [V, m, n, 3
    """

    ## random point cloud in R3
    rand_point_cloud = np.random.uniform(size=(m, n, 3))

    ## random circle centered at 0.5
    circ_points = mvn(mean=[0, 0]).rvs(size=(m, n))
    error = mvn(cov=np.diag([V ** 2, V ** 2])).rvs(size=(m, n))
    for i in range(m):
        norms_points = np.linalg.norm(circ_points[i, :, :], axis=1)
        circ_points[i, :, :] = circ_points[i, :, :] / (2 * norms_points[:, None])
    circ_points = circ_points + error
    circ_points = circ_points + np.array([0.5, 0.5])


    ## random sphere centerred at [0.5, 0.5, 0.5]
    sphere_points = mvn(mean=[0, 0, 0]).rvs(size=(m, n))
    error = mvn(cov=np.diag([V ** 2, V ** 2, V ** 2])).rvs(size=(m, n))
    for i in range(m):
        norms_points = np.linalg.norm(sphere_points[i, :, :], axis=1)
        sphere_points[i, :, :] = sphere_points[i, :, :] / (2 * norms_points[:, None])
    sphere_points = sphere_points + error
    sphere_points = sphere_points + np.array([0.5, 0.5, 0.5])

    ## 3 clusters
    m_clusters = np.zeros((m, n, 3))

    def create_cluster(centers, n=n, v=0.05):
        a = max(np.random.choice(np.arange(0, int(0.45 * n))), int(0.1 * n))
        b = max(np.random.choice(np.arange(0, int(0.45 * (n - a)))), int(0.1 * (n - a)))
        c = n - a - b
        cluster_1 = mvn(mean=centers[0, :], cov=np.diag([v ** 2, v ** 2, v ** 2])).rvs(size=a)
        cluster_2 = mvn(mean=centers[1, :], cov=np.diag([v ** 2, v ** 2, v ** 2])).rvs(size=b)
        cluster_3 = mvn(mean=centers[2, :], cov=np.diag([v ** 2, v ** 2, v ** 2])).rvs(size=c)
        clusters = np.vstack((cluster_1, cluster_2, cluster_3))
        np.random.shuffle(clusters)
        return clusters

    for i in range(m):
        centers = np.random.uniform(size=(3, 3))
        # enforce a max of 45% and a min of 10% of all points in one of the clusters
        m_clusters[i, :, :] = create_cluster(centers, n)

    ## 3 clusters in 3 clusters:
    m_clusters_clusters = np.zeros((m, n, 3))
    for i in range(m):
        a = max(np.random.choice(np.arange(0, int(0.45 * n))), int(0.1 * n))
        b = max(np.random.choice(np.arange(0, int(0.45 * (n - a)))), int(0.1 * (n - a)))
        c = n - a - b
        centers = np.random.uniform(size=(3, 3))
        centers = np.repeat(centers, 3, axis=0)
        pert = mvn(mean=[0, 0, 0], cov=np.diag([0.05 ** 2, 0.05 ** 2, 0.05 ** 2])).rvs(size=9)
        centers = pert + centers
        clusters_1 = create_cluster(centers[0:3, :], a, v=0.02)
        clusters_2 = create_cluster(centers[3:6, :], b, v=0.02)
        clusters_3 = create_cluster(centers[6:9, :], c, v=0.02)
        clusters = np.vstack((clusters_1, clusters_2, clusters_3))
        np.random.shuffle(clusters)
        m_clusters_clusters[i, :, :] = clusters

    ## sample points for the Torus centered at 0.5, 0.5 0.5
    # R=0.5, r=0.25
    theta = np.random.uniform(0, 2 * np.pi, size=(m, n))
    phi = np.random.uniform(0, 2 * np.pi, size=(m, n))
    R, r = 0.5, 0.25
    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    error = mvn(cov=np.diag([V ** 2, V ** 2, V ** 2])).rvs(size=(m, n))
    torus_points = np.transpose(np.array([x, y, z]), (1, 2, 0)) + error
    torus_points = torus_points + np.array([0.5, 0.5, 0.5])

    toy_shapes = {
        "random": rand_point_cloud,
        "circle": circ_points,
        "sphere": sphere_points,
        "clusters": m_clusters,
        "c_clusters": m_clusters_clusters,
        "torus": torus_points
    }
    return toy_shapes


def do_training(PI_vectors_train, PI_vectors_test, y_H_train):
    X_H_train = PI_vectors_train
    X_H_test = PI_vectors_test

    print(PI_vectors_train)
    print(PI_vectors_test)
    print(y_H_train)

    gnb = GaussianNB().fit(X_H_train, y_H_train.ravel())
    lr = LogisticRegression(random_state=42, max_iter=1000).fit(X_H_train, y_H_train.ravel())

    y_H_hat_km = gnb.predict(X_H_test)
    y_H_hat_lr = lr.predict(X_H_test)
    return y_H_hat_km, y_H_hat_lr

def do_analysis(y_H, yhat1, yhat2, labels):
    report_km = classification_report(y_H.ravel(),
                                         yhat1,
                                         labels=list(range(6)),
                                         target_names=list(labels.values()),
                                         output_dict=True)
    report_lr = classification_report(y_H.ravel(),
                                         yhat2,
                                         labels=list(range(6)),
                                         target_names=list(labels.values()),
                                         output_dict=True)

    report_km_df = pd.DataFrame(report_km).transpose()
    report_km_lr = pd.DataFrame(report_lr).transpose()

    print(report_km_df)
    print(report_km_lr)

    return report_km_df, report_km_lr


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
    PI_vectors_H0 = np.zeros((6 * m, pixels))
    PI_vectors_H1 = np.zeros((6 * m, pixels))

    target = np.zeros(6 * m)
    labels = {i: shape for i, shape in enumerate(data)}
    index = 0
    print("Creating the PI images for the 6 figures")
    for i in tqdm(range(6)):
        shape = labels[i]
        shape_data = data[shape]
        for j in range(m):
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
            index += 1

    PI_vectors_H0_train, PI_vectors_H0_test, y_train_H0, y_test_H0 = train_test_split(PI_vectors_H0, target,
                                                                                      test_size=0.33)
    PI_vectors_H1_train, PI_vectors_H1_test, y_train_H1, y_test_H1 = train_test_split(PI_vectors_H1, target,
                                                                                      test_size=0.33)

    y_H0_hat_km, y_H0_hat_lr = do_training(PI_vectors_H0_train, PI_vectors_H0_test, y_train_H0)
    y_H1_hat_km, y_H1_hat_lr = do_training(PI_vectors_H1_train, PI_vectors_H1_test, y_train_H1)

    reports_H0_gnb, reports_H0_lr = do_analysis(y_test_H0, y_H0_hat_km, y_H0_hat_lr, labels)
    reports_H0_gnb, reports_H0_lr = do_analysis(y_test_H0, y_H0_hat_km, y_H0_hat_lr, labels)
    reports_H1_gnb, reports_H1_lr = do_analysis(y_test_H1, y_H1_hat_km, y_H1_hat_lr, labels)

    reports_H0_gnb.to_csv("results/{}_{}_{}_{}_H0_gnb.csv".format(quality, spread, kernel, weighting))
    reports_H0_lr.to_csv("results/{}_{}_{}_{}_H0_lr.csv".format(quality, spread, kernel, weighting))
    reports_H1_gnb.to_csv("results/{}_{}_{}_{}_H1_gnb.csv".format(quality, spread, kernel, weighting))
    reports_H1_lr.to_csv("results/{}_{}_{}_{}_H1_lr.csv".format(quality, spread, kernel, weighting))

    print(reports_H0_gnb)
    print(reports_H0_lr)
    print(reports_H1_gnb)
    print(reports_H1_lr)
    return reports_H0_gnb, reports_H0_lr, reports_H1_gnb, reports_H1_lr

## Now we apply Ripser and Persim to do the classification
n, m, V = 150, 150, 0.1
data = gen_six_figures(n, m, V)
epsilon = 0.1
ax_range = (0 - epsilon, 1 + epsilon)
## create example of each shape
# for shape in data:
#     shape_data_ex = data[shape][0, :, :]
#     if shape != "circle":
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection="3d")
#
#         ax.scatter(
#             shape_data_ex[:, 0],
#             shape_data_ex[:, 1],
#             shape_data_ex[:, 2],
#         )
#         ax.set_xlabel('X Axis')
#         ax.set_ylabel('Y Axis')
#         ax.set_zlabel('Z Axis')
#
#         ax.set_xlim(ax_range)
#         ax.set_ylim(ax_range)
#         ax.set_zlim(ax_range)
#         plt.savefig("figures/{}_example.png".format(shape))
#     else:
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#
#         ax.scatter(
#             shape_data_ex[:, 0],
#             shape_data_ex[:, 1],
#         )
#         ax.set_xlabel('X Axis')
#         ax.set_ylabel('Y Axis')
#
#         ax.set_xlim(ax_range)
#         ax.set_ylim(ax_range)
#         plt.savefig("figures/{}_example.png".format(shape))


qualities = [5, 10, 25, 50]
spreads = [0.05, 0.1, 0.2]
kernels = ["gamma", "lognorm"]
weightings = ["linear", "logistic"]

for quality in tqdm(qualities):
    for spread in tqdm(spreads):
        for kernel in tqdm(kernels):
            for weighting in tqdm(weightings):
                _, _, _, _ = do_full_run(data, quality, spread, kernel, weighting)
