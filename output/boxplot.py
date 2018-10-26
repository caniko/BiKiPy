import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys


df = pd.read_csv('position_preference_1.csv')

ratio = df.Top /df.Bottom

# this is not useful here
#plt.figure()
#plt.boxplot(ratio)
#plt.ion()
#plt.show()

exp =  np.array(["Nov"] * df.shape[0])
idxs = np.where(np.array(["Sociability" in v for v in df.Experiment.values]) == True)[0]

exp[idxs] = "Sociability"

df["Groups"] = exp
df["Ratio (Top/Bottom)"] = ratio

# boxplot and kdplot using seaborn
#data_boxplot = sns.boxplot(data=df, x="group", y="ratio")
#data_kdplot = sns.kdeplot(data=df.ratio, shade=True)
plt.figure()
if sys.argv[1] == "boxplot":
    sns.boxplot(data=df, x="Groups", y="Ratio (Top/Bottom)")
    plt.savefig("data_seaborn_boxplot.png")
#if sys.argv[1] == "kdplot":
#    sns.kdeplot(data=df.ratio, shade=True)
#    sns.kdeplot(data=df.ratio, shade=True).savefig("data_kdplot.png")
