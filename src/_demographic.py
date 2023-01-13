import random
import logging

import numpy as np


def compute_c(d, q, delta):
    dq = d + q
    mask = (dq == 0.)
    res = dq / (np.exp(dq*delta)-1)
    res[mask] = 1. / delta[mask]
    return res


def compute_fq(d, m, q, delta, lam=1.):
    c = compute_c(d, q, delta)
    cdq = c + d + q
    c_cdq = np.r_[1., c[:-1]] / cdq
    c_cdq_prod = np.cumprod(c_cdq)  # fi
    res = lam * np.sum(m * c_cdq_prod) - 1, (c, cdq, c_cdq, c_cdq_prod)
    return res


def _compute_df_dq(d, m, q, delta, lam, compute_fq_intermediater):
    n = d.shape[0]

    c, cdq, _, f = compute_fq_intermediater
    c_ = c[:-1]
    delta_ = delta[:-1]
    dq_ = d[:-1] + q
    cdq_ = cdq[:-1]
    f_ = f[:-1]

    dc_dq = (c_ - c_** 2 * delta_) / dq_ - c_ * delta_
    dc_dq[dq_ == 0.] = -0.5  # 通过极限运算得到的
    df_dc = f[:, None] * (dq_/(c_*cdq_))
    row, col = np.triu_indices_from(df_dc, 0)
    df_dc[row, col] = 0
    df_dc[range(n-1), range(n-1)] = - f_ / cdq_

    df_dq = -f * np.cumsum(1/cdq)
    df_dq_all = np.dot(df_dc, dc_dq) + df_dq
    res = lam * np.sum(df_dq_all * m)
    return res

def compute_P(total0, d, q, c):
    c_cdq = c[:-1] / (c[1:] + d[1:] + q)
    c_cdq_cumprod = np.cumprod(c_cdq, axis=0)
    y0 = total0 / (c_cdq_cumprod.sum() + 1)
    return np.r_[y0, y0 * c_cdq_cumprod]


# def find_q_dichotomy(
#     fertilities, deathes, agedelta, max_iter=100, tol=1e-8, bounds=(0., 1.)
# ):
#     d = deathes
#     m = fertilities
#     delta = agedelta
#
#     minb, maxb = bounds
#     minf, maxf = (
#         compute_fq(d, m, minb, delta)[0], compute_fq(d, m, maxb, delta)[0]
#     )
#
#     if abs(minf) <= tol:
#         return minb, {"tol": abs(minf), "niter": 0}
#     elif abs(maxf) <= tol:
#         return maxb, {"tol": abs(maxf), "niter": 0}
#
#     for i in range(max_iter):
#         if (minf > 0 and maxf > 0) or (minf < 0 and minf < 0):
#             raise ValueError
#
#         midb = (minb + maxb) / 2
#         midf = compute_fq(d, m, midb, delta)[0]
#         err = abs(midf)
#
#         if err <= tol:
#             return midb, {"tol": err, "niter": i + 1}
#         elif ((midf > 0) and (minf < 0)) or ((midf < 0) and (minf > 0)):
#             maxb = midb
#             maxf = midf
#             continue
#         elif ((midf > 0) and (maxf < 0)) or ((midf < 0) and (maxf > 0)):
#             minb = midb
#             minf = midf
#             continue
#
#     else:
#         print(
#             "reach the max iteration, don't converage. "
#             "q value: %.4f, qdelta: %.4f" % (minb, err)
#         )
#
#     return minb, {"tol": err, "niter": max_iter}


def find_q_newton(
    lam, fertilities, deathes, agedelta, max_iter=100, tol=1e-8, q_init=0.,
    verbose=False
):
    d = deathes
    m = fertilities
    delta = agedelta
    if q_init is None:
        q_init = random.random()

    q = q_init
    for i in range(max_iter):
        fq, intermediater = compute_fq(d, m, q, delta, lam)
        err = np.abs(fq)
        if verbose:
            logging.info(
                "[find_q_newton] %d: q = %.4f, err = %.4f" % (i, q, err)
            )
        if err <= tol:
            return q, {"err": err, "niter": i, "converage": True}
        dfdq = _compute_df_dq(d, m, q, delta, lam, intermediater)
        q -= fq / dfdq
    
    fq, _ = compute_fq(d, m, q, delta, lam)
    err = np.abs(fq)
    if err <= tol:
        return q, {"err": err, "niter": max_iter, "converage": True}
    else:
        logging.info(
            "[find_q_newton] reach the max iteration, don't converage. "
            "q value: %.4f, fq: %.4f" % (q, err)
        )
    return q, {"err": err, "niter": max_iter, "converage": False}

