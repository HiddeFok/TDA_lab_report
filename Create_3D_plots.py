from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from ripser import Rips
from persim import PersImage
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt

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
        "circle": circ_points,
        "sphere": sphere_points,
        "torus": torus_points
    }
    return toy_shapes

data = gen_six_figures(100, 4, 0.05)
PI_data = data["torus"][0]

#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
#
# ax.scatter(
#     PI_data[:, 0],
#     PI_data[:, 1],
#     PI_data[:, 2],
# )
# ax.set_xlabel('X Axis')
# ax.set_ylabel('Y Axis')
# ax.set_zlabel('Z Axis')
#
# epsilon = 0.1
# ax_range = (0 - epsilon, 1 + epsilon)
#
# ax.set_xlim(ax_range)
# ax.set_ylim(ax_range)
# ax.set_zlim(ax_range)

kernels = ["gaussian", "laplace"]
weightings = ["linear", "logistic"]

spread = 0.05
quality = 150
pixels = [quality, quality]

fig = plt.figure()
fig.set_figwidth(9)
fig.set_figheight(6)

index = 1
for i, kern in enumerate(kernels):
    for j, weighting in enumerate(weightings):
        pim = PersImage(spread=spread,
                        pixels=pixels,
                        verbose=False,
                        kernel_type=kern,
                        weighting_type=weighting)
        rips = Rips(verbose=False)
        dgms = rips.fit_transform(PI_data)

        img = pim.transform(dgms[1])
        mini, maxi = np.amin(img), min(np.amax(img), 0.05)
        print(mini, maxi)
        X = np.arange(0, 150)
        Y = np.arange(0, 150)
        X, Y = np.meshgrid(X, Y)
        Z = img[::-1]

        ax =  fig.add_subplot(2, 2, index, projection='3d')
        #ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=15, cstride=15, cmap="inferno")
        ax.set_zlim(mini, maxi)
        ax.set_xlim(1, 150)
        ax.set_ylim(1, 150)

        plt.yticks([], [])
        plt.xticks([], [])
        ax.set_zticks([])
        ax.set_title("{} density and {} weighting".format(kern, weighting))

        index += 1


plt.savefig("figures/3d_plots_torus.svg".format(index), format="svg")
