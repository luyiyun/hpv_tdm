from math import ceil
import os.path as osp

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


save_root = "./results"
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

res_t = np.load(osp.join(save_root, "res_t.npy"))
res_y = np.load(osp.join(save_root, "res_yn.npy"))

nt, nrooms, nages = res_y.shape
nc = 4
nr = ceil(nrooms / nc)
fig, axs = plt.subplots(nrows=nr, ncols=nc, figsize=(3*nc, 3*nr))
colors = sns.color_palette("husl", nages)
for ri in range(nrooms):
    i, j = ri // nc, ri % nc
    ax = axs[i, j]
    for ai in range(nages):
        ax.plot(res_t, res_y[:, ri, ai], label=ages[ai], color=colors[ai])
    ax.set_title(rooms[ri])

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="center right",
           ncol=1, bbox_to_anchor=(1.0, 0.5))
fig.savefig(osp.join(save_root, "res.png"))
# fig.tight_layout()
# plt.show()
