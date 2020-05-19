import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')

kernels = ["gaussian", "laplace", "gamma", "lognorm"]
weightings = ["linear", "logistic"]
acc_scores_H0_ada = defaultdict(list)
acc_scores_H1_ada = defaultdict(list)
acc_scores_H0_grad = defaultdict(list)
acc_scores_H1_grad = defaultdict(list)
acc_scores_con_ada = defaultdict(list)
acc_scores_con_grad = defaultdict(list)

for kern in kernels:
    for weight in weightings:
        report_H0_ada = pd.read_csv("results_dynamic/20_0.05_{}_{}_H0_ada.csv".format(kern, weight))
        report_H1_ada = pd.read_csv("results_dynamic/20_0.05_{}_{}_H1_ada.csv".format(kern, weight))
        report_H0_grad = pd.read_csv("results_dynamic/20_0.05_{}_{}_H0_grad.csv".format(kern, weight))
        report_H1_grad = pd.read_csv("results_dynamic/20_0.05_{}_{}_H1_grad.csv".format(kern, weight))
        report_con_ada = pd.read_csv("results_dynamic/20_0.05_{}_{}_con_ada.csv".format(kern, weight))
        report_con_grad = pd.read_csv("results_dynamic/20_0.05_{}_{}_con_grad.csv".format(kern, weight))

        acc_scores_H0_ada[weight].append(report_H0_ada[report_H0_ada.iloc[:, 0] == "accuracy"]["f1-score"].iloc[0])
        acc_scores_H1_ada[weight].append(report_H1_ada[report_H1_ada.iloc[:, 0] == "accuracy"]["f1-score"].iloc[0])
        acc_scores_H0_grad[weight].append(report_H0_grad[report_H0_grad.iloc[:, 0] == "accuracy"]["f1-score"].iloc[0])
        acc_scores_H1_grad[weight].append(report_H1_grad[report_H1_grad.iloc[:, 0] == "accuracy"]["f1-score"].iloc[0])
        acc_scores_con_ada[weight].append(report_con_ada[report_con_ada.iloc[:, 0] == "accuracy"]["f1-score"].iloc[0])
        acc_scores_con_grad[weight].append(report_con_grad[report_con_grad.iloc[:, 0] == "accuracy"]["f1-score"].iloc[0])



width = 0.2
x = np.arange(4)
for i, weight in enumerate(weightings):
    fig, axs = plt.subplots(1, 1)
    fig.set_figwidth(15)
    fig.set_figheight(13)
    axs.bar(x - width, acc_scores_H0_ada[weight], width, label="H0")
    axs.bar(x, acc_scores_H1_ada[weight], width, label="H1")
    axs.bar(x + width, acc_scores_con_ada[weight], width, label="H0 & H1")
    axs.set_ylabel('Accuracy')
    axs.set_xticks(x)
    axs.set_xticklabels(kernels)
    axs.set_yticks(np.arange(0, 1.1, 0.1))
    axs.legend()
    axs.set_ylim((0, 1))
    axs.set_title("Accuracy of classification with AdaBoost and weighting {}".format(weight))
    plt.savefig("figures/accuracies_dynamic_ada_{}.svg".format(weight), format="svg")




width = 0.2
x = np.arange(4)
for i, weight in enumerate(weightings):
    fig, axs = plt.subplots(1, 1)
    fig.set_figwidth(15)
    fig.set_figheight(13)
    axs.bar(x - width, acc_scores_H0_grad[weight], width, label="H0")
    axs.bar(x, acc_scores_H1_grad[weight], width, label="H1")
    axs.bar(x + width, acc_scores_con_grad[weight], width, label="H0 & H1")
    axs.set_ylabel('Accuracy')
    axs.set_xticks(x)
    axs.set_xticklabels(kernels)
    axs.set_yticks(np.arange(0, 1.1, 0.1))
    axs.legend()
    axs.set_ylim((0, 1))
    axs.set_title("Accuracy of classification with Gradient Boosting and weighting {}".format(weight))
    plt.savefig("figures/accuracies_dynamic_grad_{}.svg".format(weight), format="svg")
