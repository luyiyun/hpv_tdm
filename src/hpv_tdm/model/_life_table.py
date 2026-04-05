from __future__ import annotations

import numpy as np
import pandas as pd


def life_table(
    deathes: np.ndarray,
    agebins: np.ndarray,
    method: str = "prime",
) -> pd.DataFrame:
    agedelta = agebins[1:] - agebins[:-1]
    nd_ = deathes[1:-1] * agedelta[1:-1]
    q = np.r_[deathes[0], 2 * nd_ / (2 + nd_), 1]
    p = 1 - q
    survivors = np.r_[1, np.cumprod(p[:-1])]
    if method == "prime":
        L = np.r_[
            (survivors[:-1] + survivors[1:]) / 2 * agedelta[:-1],
            survivors[-1] / 2,
        ]
    elif method == "textbook":
        L = np.r_[
            survivors[0] + 0.15 * (survivors[1] - survivors[0]),
            (survivors[1:-1] + survivors[2:]) / 2 * agedelta[1:-1],
            survivors[-1] / deathes[-1],
        ]
    else:
        raise ValueError(f"unsupported life table method: {method}")
    T = np.cumsum(L[::-1])[::-1]
    E = T / survivors
    return pd.DataFrame(
        {"q": q, "p": p, "l": survivors, "L": L, "T": T, "E": E},
        index=pd.IntervalIndex.from_breaks(agebins, closed="left"),
    )
