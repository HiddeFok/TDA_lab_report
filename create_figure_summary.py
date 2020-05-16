import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')

qualities = [5, 10, 25, 50]
spreads = [0.05, 0.1, 0.2]
kernels = ["gaussian", "laplace"]
weightings = ["linear", "logistic"]

acc_scores_H0_gnb = defaultdict(lambda: defaultdict(list))
acc_scores_H1_gnb = defaultdict(lambda: defaultdict(list))
acc_scores_H0_lr = defaultdict(lambda: defaultdict(list))
acc_scores_H1_lr = defaultdict(lambda: defaultdict(list))


for kern in kernels:
    for weight in weightings:
        for spread in spreads:
            report_H0_gnb = pd.read_csv("results/10_{}_{}_{}_H0_gnb.csv".format(spread, kern, weight))
            report_H1_gnb = pd.read_csv("results/10_{}_{}_{}_H1_gnb.csv".format(spread, kern, weight))
            report_H0_lr = pd.read_csv("results/10_{}_{}_{}_H0_lr.csv".format(spread, kern, weight))
            report_H1_lr = pd.read_csv("results/10_{}_{}_{}_H1_lr.csv".format(spread, kern, weight))


            acc_scores_H0_gnb[kern][weight].append(report_H0_gnb["f1-score"].iloc[0])
            acc_scores_H1_gnb[kern][weight].append(report_H1_gnb["f1-score"].iloc[0])
            acc_scores_H0_lr[kern][weight].append(report_H0_lr["f1-score"].iloc[0])
            acc_scores_H1_lr[kern][weight].append(report_H1_lr["f1-score"].iloc[0])

fig, axs = plt.subplots(2, 2)
fig.set_figwidth(19)
fig.set_figheight(15)

for i, kern in enumerate(kernels):
    for j, weight in enumerate(weightings):
        axs[i, j].plot(spreads, acc_scores_H0_gnb[kern][weight], label="H0_GNB")
        axs[i, j].plot(spreads, acc_scores_H1_gnb[kern][weight], label="H1_GNB")
        axs[i, j].plot(spreads, acc_scores_H0_lr[kern][weight], label="H0_LR")
        axs[i, j].plot(spreads, acc_scores_H1_lr[kern][weight], label="H1_LR")
        axs[i, j].legend()
        axs[i, j].set_xlabel("Variance parameter")
        axs[i, j].set_ylabel("Accuracy")
        axs[i, j].set_title("Accuracy of classification with kernel {} and weighting {}".format(kern, weight))
plt.savefig("figures/accuracies_against_variances.png")


acc_scores_H0_gnb = defaultdict(lambda: defaultdict(list))
acc_scores_H1_gnb = defaultdict(lambda: defaultdict(list))
acc_scores_H0_lr = defaultdict(lambda: defaultdict(list))
acc_scores_H1_lr = defaultdict(lambda: defaultdict(list))


for kern in kernels:
    for weight in weightings:
        for quality in qualities:
            report_H0_gnb = pd.read_csv("results/{}_0.05_{}_{}_H0_gnb.csv".format(quality, kern, weight))
            report_H1_gnb = pd.read_csv("results/{}_0.05_{}_{}_H1_gnb.csv".format(quality, kern, weight))
            report_H0_lr = pd.read_csv("results/{}_0.05_{}_{}_H0_lr.csv".format(quality, kern, weight))
            report_H1_lr = pd.read_csv("results/{}_0.05_{}_{}_H1_lr.csv".format(quality, kern, weight))


            acc_scores_H0_gnb[kern][weight].append(report_H0_gnb["f1-score"].iloc[0])
            acc_scores_H1_gnb[kern][weight].append(report_H1_gnb["f1-score"].iloc[0])
            acc_scores_H0_lr[kern][weight].append(report_H0_lr["f1-score"].iloc[0])
            acc_scores_H1_lr[kern][weight].append(report_H1_lr["f1-score"].iloc[0])

fig, axs = plt.subplots(2, 2)
fig.set_figwidth(19)
fig.set_figheight(15)

for i, kern in enumerate(kernels):
    for j, weight in enumerate(weightings):
        axs[i, j].plot(qualities, acc_scores_H0_gnb[kern][weight], label="H0_GNB")
        axs[i, j].plot(qualities, acc_scores_H1_gnb[kern][weight], label="H1_GNB")
        axs[i, j].plot(qualities, acc_scores_H0_lr[kern][weight], label="H0_LR")
        axs[i, j].plot(qualities, acc_scores_H1_lr[kern][weight], label="H1_LR")
        axs[i, j].legend()
        axs[i, j].set_xlabel("Resolution")
        axs[i, j].set_ylabel("Accuracy")
        axs[i, j].set_title("Accuracy of classification with kernel {} and weighting {}".format(kern, weight))
plt.savefig("figures/accuracies_against_resolution.png")
