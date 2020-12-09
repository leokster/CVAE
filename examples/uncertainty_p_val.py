import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from variational_autoencoder.evaluations import calc_area, get_p_vals, calc_area_separated
import pandas as pd

smpl = 1000
mc_smpl = 1000
distr = [
    np.random.normal(2,2,[smpl, mc_smpl]),
    np.random.normal(2,0.1,[smpl, mc_smpl]),
    np.random.gamma(2,2,[smpl, mc_smpl]),
    np.random.gamma(7, 2, [smpl, mc_smpl])
]

labels_distr = [
    r"$N(2,2)$ distributed",
    r"$N(2,0.1)$ distributed",
    r"$\Gamma(2,2)$ distributed",
    r"$\Gamma(7,2)$ distributed",
]

p_area = np.zeros(shape=(len(distr), len(distr)))

p_area_pos = np.zeros(shape=(len(distr), len(distr)))
p_area_neg = np.zeros(shape=(len(distr), len(distr)))
p_vals = np.zeros(shape=(len(distr), len(distr)))


fig, axes = plt.subplots(len(distr)+1,len(distr)+1, figsize=(20,20))

for i in range(len(distr)):
    for j in range(len(distr)):
        p_vals= get_p_vals(distr[i],distr[j][:,0])
        p_area_pos[i,j], p_area_neg[i,j] = calc_area_separated(p_vals)
        p_area[i,j] = calc_area(p_vals)
        sns.lineplot(x=p_vals, y=np.linspace(0, 1, smpl), ax=axes[i+1, j+1])
        axes[i + 1, j + 1].set_xlim((0,1))
        axes[i + 1, j + 1].set_ylim((0,1))
        p_vals_extended = np.append(0,np.append(p_vals,1))
        axes[i + 1, j + 1].fill_between(p_vals_extended, np.linspace(0, 1, smpl+2), p_vals_extended, color="Gray", alpha=0.3)
        axes[i + 1, j + 1].plot([0,1], [0,1], color="Gray")

for i in range(len(distr)):
    axes[i + 1, 0].clear()
    axes[0,i+1].clear()
    sns.histplot(distr[i].flatten()[:10000], ax=axes[i+1, 0])
    sns.histplot(distr[i].flatten()[:10000], ax=axes[0, i + 1])
    axes[i + 1, 0].set_ylabel(labels_distr[i], rotation=90, fontsize=24)
    axes[0, i + 1].set_ylabel("")
    axes[0, i + 1].set_xlabel(labels_distr[i], rotation=0, fontsize=24)
    axes[0, i + 1].xaxis.set_label_position('top')

axes[0,0].set_axis_off()


fig.show()
fig.savefig("p_plt.png")

print(pd.DataFrame(p_area, columns=labels_distr, index=labels_distr).round(4).to_latex())