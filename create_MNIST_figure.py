import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')

full_pipeline = []
H1_pipeline = []
H0_pipeline = []
con_pipeline = []

full_pipeline.append(pd.read_csv("result_MNIST/MNIST_report_lr.csv")["f1-score"].iloc[10])
full_pipeline.append(pd.read_csv("result_MNIST/MNIST_report_rf.csv")["f1-score"].iloc[10])
full_pipeline.append(pd.read_csv("result_MNIST/MNIST_report_sgd.csv")["f1-score"].iloc[10])
full_pipeline.append(pd.read_csv("result_MNIST/MNIST_report_svm.csv")["f1-score"].iloc[10])

H1_pipeline.append(pd.read_csv("result_MNIST/MNIST_report_H1_lr.csv")["f1-score"].iloc[10])
H1_pipeline.append(pd.read_csv("result_MNIST/MNIST_report_H1_rf.csv")["f1-score"].iloc[10])
H1_pipeline.append(pd.read_csv("result_MNIST/MNIST_report_H1_sgd.csv")["f1-score"].iloc[10])
H1_pipeline.append(pd.read_csv("result_MNIST/MNIST_report_H1_svm.csv")["f1-score"].iloc[10])

H0_pipeline.append(pd.read_csv("result_MNIST/MNIST_report_HO_lr.csv")["f1-score"].iloc[10])
H0_pipeline.append(pd.read_csv("result_MNIST/MNIST_report_HO_rf.csv")["f1-score"].iloc[10])
H0_pipeline.append(pd.read_csv("result_MNIST/MNIST_report_HO_sgd.csv")["f1-score"].iloc[10])
H0_pipeline.append(pd.read_csv("result_MNIST/MNIST_report_HO_svm.csv")["f1-score"].iloc[10])

con_pipeline.append(pd.read_csv("result_MNIST/MNIST_report_con_lr.csv")["f1-score"].iloc[10])
con_pipeline.append(pd.read_csv("result_MNIST/MNIST_report_con_rf.csv")["f1-score"].iloc[10])
con_pipeline.append(pd.read_csv("result_MNIST/MNIST_report_con_sgd.csv")["f1-score"].iloc[10])
con_pipeline.append(pd.read_csv("result_MNIST/MNIST_report_con_svm.csv")["f1-score"].iloc[10])


fig, axs = plt.subplots(1,1)
fig.set_figwidth(8)
fig.set_figheight(6)
width = 0.2
x = np.arange(4)
clfs = ["Logistic Regression", "Random Forest", "SGD", "SVM"]

axs.bar(x - 3 * width / 2, H0_pipeline, width, label="H0")
axs.bar(x - width / 2, H1_pipeline, width, label="H1")
axs.bar(x + width / 2, con_pipeline, width, label="H0 & H1")
axs.bar(x + 3 * width / 2, full_pipeline, width, label="Custom")
axs.set_ylabel('Accuracy')
axs.set_xticks(x)
axs.set_xticklabels(clfs)
axs.set_yticks(np.arange(0, 1.1, 0.1))
axs.legend()
axs.set_ylim((0, 1))
axs.set_title("Accuracy of MNIST classification")
plt.savefig("figures/MNIST_results.svg", format="svg")