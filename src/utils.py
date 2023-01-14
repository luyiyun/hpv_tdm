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


def diagonal2coo(args):
    ii, jj, data = [], [], []
    for (x0, x1), n, major, minor in args:
        i0, i1 = np.arange(x0, x0+n), np.arange(x1, x1+n)
        ii.append(i0)
        jj.append(i1)
        if isinstance(major, (float, int)):
            major = np.full((n,), major)
        data.append(major)
        if minor is not None:
            i0, i1 = np.arange(x0+1, x0+n), np.arange(x0, x0+n-1)
            ii.append(i0)
            jj.append(i1)
            if isinstance(minor, (float, int)):
                minor = np.full((n,), minor)
            data.append(minor)
    ii = np.concatenate(ii)
    jj = np.concatenate(jj)
    data = np.concatenate(data)
    return ii, jj, data


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


