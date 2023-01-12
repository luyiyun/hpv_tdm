from math import ceil

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


rooms = ["Sf", "If", "Pf", "LC", "RC", "DC", "Rf",
         "Vf", "Sm", "Im", "Pm", "Dm", "Rm"]
agebins = np.array([
    0, 1, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 25, 27, 30,
    35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, np.inf
])
nages = agebins.shape[0] - 1
ages = []
for i, j in zip(range(nages), range(1, nages+1)):
    a, b = agebins[i], agebins[j]
    if np.isclose(b, np.inf):
        ages.append("[%d, inf)" % a)
    else:
        ages.append("[%d, %d)" % (a, b))

res_t = np.load("./results/res_t.npy")
res_yn = np.load("./results/res_yn.npy")
# res_yp = np.load("./results/res_yp.npy")

nt, nrooms, nages = res_yn.shape
nc = 4
nr = ceil(nrooms / nc)
fig, axs = plt.subplots(nrows=nr, ncols=nc, figsize=(3*nc, 3*nr))
colors = sns.color_palette("husl", nages)
for ri in range(nrooms):
    i, j = ri // nc, ri % nc
    ax = axs[i, j]
    for ai in range(nages):
        ax.plot(res_t, res_yn[:, ri, ai], label=ages[ai], color=colors[ai])
    ax.set_title(rooms[ri])

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="center right",
           ncol=1, bbox_to_anchor=(1.0, 0.5))
# fig.tight_layout()
plt.show()
