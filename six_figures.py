from itertools import product
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from mpl_toolkits.mplot3d import Axes3D

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

## Now we apply Ripser and Persim to do the classification
n, m, V = 150, 100, 0.2
data = gen_six_figures(n, m, V)
epsilon = 0.1
ax_range = (0 - epsilon, 1 + epsilon)
## create example of each shape
for shape in data:
    shape_data_ex = data[shape][0, :, :]
    if shape != "circle":
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(
            shape_data_ex[:, 0],
            shape_data_ex[:, 1],
            shape_data_ex[:, 2],
        )
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')

        ax.set_xlim(ax_range)
        ax.set_ylim(ax_range)
        ax.set_zlim(ax_range)
        plt.savefig("{}_example.png".format(shape))
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.scatter(
            shape_data_ex[:, 0],
            shape_data_ex[:, 1],
        )
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')

        ax.set_xlim(ax_range)
        ax.set_ylim(ax_range)
        plt.savefig("{}_example.png".format(shape))


quality = 50
pixels = quality * quality
PI_vectors_H0 = np.zeros((6 * 100, pixels, 1))
PI_vectors_H1 = np.zeros((6 * 100, pixels, 1))
labels = {i: shape for i, shape in enumerate(data)}
index = 0
print("Creating the PI images for the 6 figures")
for i in tqdm(range(6)):
    shape = labels[i]
    shape_data = data[shape]
    for j in range(5):
        pim_0 = PersImage(spread=0.1,
                          pixels=[quality, quality],
                          verbose=False,
                          specs={
                            "maxBD": 2,
                            "minBD": 0,
                  })
        pim_1 = PersImage(spread=0.1,
                          pixels=[quality, quality],
                          verbose=False,
                          specs={
                            "maxBD": 2,
                            "minBD": 0,
                  })

        PI_data = shape_data[j, :, :]
        rips = Rips(verbose=False)
        dgms = rips.fit_transform(PI_data)
        PI_data_H0 = pim_0.transform(dgms[0])
        ax = plt.subplot(131)
        plt.title("PI for $H_1$\nwith 10x10 pixels")
        pim_0.show(PI_data_H0, ax)

        PI_data_H1 = pim_1.transform(dgms[1])

        PI_data_H1 = PI_data_H1.reshape(pixels)
        PI_data_H0 = PI_data_H0.reshape(pixels)

        PI_data_H0 = PI_data_H0[:, None]
        PI_data_H1 = PI_data_H1[:, None]

        PI_data_H0[:, 0] = int(i)
        PI_data_H1[:, 0] = int(i)

        PI_vectors_H0[index, :, :] = PI_data_H0
        PI_vectors_H1[index, :, :] = PI_data_H1
        index += 1

def do_clf_analysis(PI_vectors, mapping):

    X_H = PI_vectors[:, :, 0]
    y_H = PI_vectors[:, 0, :].astype(int)
    kmeans = KMeans(n_clusters=6, random_state=42).fit(X_H)
    lr = LogisticRegression(random_state=42).fit(X_H, y_H.ravel())

    y_H_hat_km = kmeans.predict(X_H)
    y_H_hat_lr = lr.predict(X_H)

    mapper = lambda x: int(mapping[x])
    vfunct = np.vectorize(mapper)
    y_H_hat_km = vfunct(y_H_hat_km)


    H1_report_km = classification_report(y_H.ravel(),
                                         y_H_hat_km,
                                         labels=list(range(6)),
                                         target_names=list(labels.values()))
    H1_report_lr = classification_report(y_H.ravel(),
                                         y_H_hat_lr,
                                         labels=list(range(6)),
                                         target_names=list(labels.values()))
    print(H1_report_km)
    print(H1_report_lr)
    return H1_report_km, H1_report_lr


mapping_H0 = {
    5: 0,
    1: 1,
    2: 2,
    4: 3,
    0: 4,
    3: 5
}
reports_H0 = do_clf_analysis(PI_vectors_H0, mapping_H0)
mapping_H1 = {
    5: 0,
    1: 1,
    2: 2,
    4: 3,
    0: 4,
    3: 5
}
reports_H1 = do_clf_analysis(PI_vectors_H1, mapping_H1)
