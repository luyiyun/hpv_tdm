from __future__ import annotations

import numpy as np


def agebins_to_labels(agebins: np.ndarray) -> np.ndarray:
    labels = []
    for lower, upper in zip(agebins[:-1], agebins[1:]):
        if np.isinf(upper):
            labels.append(f"[{int(lower)}, inf)")
        else:
            labels.append(f"[{int(lower)}, {int(upper)})")
    return np.asarray(labels)


__all__ = ["agebins_to_labels"]
