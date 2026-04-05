from __future__ import annotations

import logging
import random

import numpy as np


def compute_c(d: np.ndarray, q: float, delta: np.ndarray) -> np.ndarray:
    dq = d + q
    # 这里用 divide 显式处理 dq=0 的极限，避免无意义的 warning。
    return np.divide(d, np.exp(dq * delta) - 1, out=1.0 / delta, where=dq != 0.0)


def compute_fq(
    d: np.ndarray,
    m: np.ndarray,
    q: float,
    delta: np.ndarray,
    lam: float = 1.0,
) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    c = compute_c(d, q, delta)
    cdq = c + d + q
    c_cdq = np.r_[1.0, c[:-1]] / cdq
    c_cdq_prod = np.cumprod(c_cdq)
    return lam * np.sum(m * c_cdq_prod) - 1, (c, cdq, c_cdq, c_cdq_prod)


def _compute_df_dq(
    d: np.ndarray,
    m: np.ndarray,
    q: float,
    delta: np.ndarray,
    lam: float,
    intermediater: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> float:
    n = d.shape[0]
    c, cdq, _, f = intermediater
    c_ = c[:-1]
    delta_ = delta[:-1]
    dq_ = d[:-1] + q
    cdq_ = cdq[:-1]
    f_ = f[:-1]

    dc_dq = (c_ - c_**2 * delta_) / dq_ - c_ * delta_
    dc_dq[dq_ == 0.0] = -0.5
    df_dc = f[:, None] * (dq_ / (c_ * cdq_))
    row, col = np.triu_indices_from(df_dc, 0)
    df_dc[row, col] = 0
    df_dc[range(n - 1), range(n - 1)] = -f_ / cdq_
    df_dq = -f * np.cumsum(1 / cdq)
    df_dq_all = np.dot(df_dc, dc_dq) + df_dq
    return lam * np.sum(df_dq_all * m)


def compute_population_by_age(
    total0: float,
    d: np.ndarray,
    q: float,
    c: np.ndarray,
) -> np.ndarray:
    c_cdq = c[:-1] / (c[1:] + d[1:] + q)
    c_cdq_cumprod = np.cumprod(c_cdq, axis=0)
    y0 = total0 / (c_cdq_cumprod.sum() + 1)
    return np.r_[y0, y0 * c_cdq_cumprod]


def find_q_newton(
    lam: float,
    fertilities: np.ndarray,
    deathes: np.ndarray,
    agedelta: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-8,
    q_init: float | None = 0.0,
    verbose: bool = False,
) -> tuple[float, dict[str, float | int | bool]]:
    if q_init is None:
        q_init = random.random()

    q = q_init
    for index in range(max_iter):
        fq, intermediater = compute_fq(deathes, fertilities, q, agedelta, lam)
        err = np.abs(fq)
        if verbose:
            logging.info("[find_q_newton] %d: q=%.4f err=%.4f", index, q, err)
        if err <= tol:
            return q, {"err": err, "niter": index, "converage": True}
        dfdq = _compute_df_dq(deathes, fertilities, q, agedelta, lam, intermediater)
        q -= fq / dfdq

    fq, _ = compute_fq(deathes, fertilities, q, agedelta, lam)
    err = np.abs(fq)
    if err > tol:
        logging.info(
            "[find_q_newton] reach max_iter without convergence, q=%.4f fq=%.4f",
            q,
            err,
        )
    return q, {"err": err, "niter": max_iter, "converage": err <= tol}
