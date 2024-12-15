import numpy as np
import pandas as pd


def life_table(deathes, agebins, method="prime"):
    agedelta = agebins[1:] - agebins[:-1]
    nd_ = deathes[1:-1] * agedelta[1:-1]
    q = np.r_[deathes[0], 2*nd_/(2+nd_), 1]  # 第一年直接使用婴儿的死亡率即可
    p = 1 - q
    l = np.r_[1, np.cumprod(p[:-1])]
    if method == "prime":
        # 这里是prime的计算方式
        L = np.r_[(l[:-1] + l[1:])/2*agedelta[:-1], l[-1]/2]
    elif method == "textbook":
        L = np.r_[l[0]+0.15*(l[1]-l[0]),
                  (l[1:-1]+l[2:])/2*agedelta[1:-1], l[-1]/deathes[-1]]
    else:
        raise NotImplementedError
    T = np.cumsum(L[::-1])[::-1]
    E = T / l
    
    return pd.DataFrame(
        {"q": q, "p": p, "l": l, "L": L, "T": T, "E": E},
        index=pd.IntervalIndex.from_breaks(agebins, closed="left")
    )

