from time import perf_counter

import numpy as np
import pandas as pd
import seaborn as sns


class Timer:

    def __init__(self, desc=None) -> None:
        self.desc = desc

    def __enter__(self):
        self.t1 = perf_counter()
        return self

    def __exit__(self, a, b, c):
        self.eval = perf_counter() - self.t1
        if self.desc is not None:
            print("%s : %.4fs" % (self.desc, self.eval))


def agebins_to_labels(agebins):
    labels = []
    for i, j in zip(agebins[:-1], agebins[1:]):
        if np.isinf(j):
            labels.append("[%d, inf)" % i)
        else:
            labels.append("[%d, %d)" % (i, j))
    return np.array(labels)


def plot_initial_population(agebins, P_f, P_m):
    labels = agebins_to_labels(agebins)
    n = P_f.shape[0]
    df = pd.DataFrame({
        "age": np.r_[labels, labels],
        "gender": np.array(["female"] * n + ["male"] * n),
        "popu": np.r_[P_f, P_m],
    })
    fg = sns.relplot(data=df, x="age", y="popu",
                     hue="gender", kind="line", aspect=2)
    fg.set_xticklabels(rotation=45)
    return fg


def plot_detailed_process(agebins, t, y):
    labels = agebins_to_labels(agebins)
    rooms = np.array([
        "Sf", "If", "Pf", "LC", "RC", "DC", "Rf",
        "Vf", "Sm", "Im", "Pm", "Dm", "Rm"
    ])
    nages = agebins.shape[0] - 1

    nt, nrooms, nages = y.shape
    df = pd.DataFrame({
        "y": y.flatten(),
        "t": np.repeat(t, nrooms*nages),
        "room": np.tile(np.repeat(rooms, nages), nt),
        "age": np.tile(labels, nt*nrooms)
    })
    fg = sns.relplot(data=df, x="t", y="y", hue="age",
                     col="room", col_wrap=4, aspect=1, kind="line",
                     facet_kws={"sharey": False})
    return fg

