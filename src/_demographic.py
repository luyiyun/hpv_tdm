import numpy as np


def compute_c(d, q, delta):
    dq = d + q
    return dq / (np.exp(dq*delta)-1)


def compute_fq(d, m, q, delta):
    c = compute_c(d, q, delta)
    cdq = c + d + q
    c_cdq = np.r_[1., c[:-1]] / cdq
    c_cdq_prod = np.cumprod(c_cdq)  # fi
    res = np.sum(m * c_cdq_prod) - 1, (c, cdq, c_cdq, c_cdq_prod)
    return res


def _compute_df_dq(d, m, q, delta, compute_fq_intermediater):
    n = d.shape[0]

    c, cdq, _, f = compute_fq_intermediater
    c_ = c[:-1]
    delta_ = delta[:-1]
    dq_ = d[:-1] + q
    cdq_ = cdq[:-1]
    f_ = f[:-1]

    dc_dq = (c_ - c_** 2 * delta_) / dq_ - c_ * delta_
    df_dc = f[:, None] * (dq_/(c_*cdq_))
    row, col = np.triu_indices_from(df_dc, 0)
    df_dc[row, col] = 0
    df_dc[range(n-1), range(n-1)] = - f_ / cdq_

    df_dq = -f * np.cumsum(1/cdq)
    df_dq_all = np.dot(df_dc, dc_dq) + df_dq
    res = np.sum(df_dq_all * m)
    return res

def compute_P(total0, d, q, c):
    cc = c[:-1] / (c[1:] + d[1:] + q)
    cccum = np.cumprod(cc, axis=0)
    y0 = [total0 / (cccum.sum() + 1)]
    for i in range(0, d.shape[0]-1):
        y0.append(y0[-1] * cc[i])
    return np.array(y0)


def find_q_dichotomy(
    fertilities, deathes, agedelta, max_iter=100, tol=1e-8, bounds=(0., 1.)
):
    d = deathes
    m = fertilities
    delta = agedelta

    minb, maxb = bounds
    minf, maxf = (
        compute_fq(d, m, minb, delta)[0], compute_fq(d, m, maxb, delta)[0]
    )

    if abs(minf) <= tol:
        return minb, {"tol": abs(minf), "niter": 0}
    elif abs(maxf) <= tol:
        return maxb, {"tol": abs(maxf), "niter": 0}

    for i in range(max_iter):
        if (minf > 0 and maxf > 0) or (minf < 0 and minf < 0):
            raise ValueError

        midb = (minb + maxb) / 2
        midf = compute_fq(d, m, midb, delta)[0]
        err = abs(midf)

        if err <= tol:
            return midb, {"tol": err, "niter": i + 1}
        elif ((midf > 0) and (minf < 0)) or ((midf < 0) and (minf > 0)):
            maxb = midb
            maxf = midf
            continue
        elif ((midf > 0) and (maxf < 0)) or ((midf < 0) and (maxf > 0)):
            minb = midb
            minf = midf
            continue

    else:
        print(
            "reach the max iteration, don't converage. "
            "q value: %.4f, qdelta: %.4f" % (minb, err)
        )

    return minb, {"tol": err, "niter": max_iter}


def find_q_newton(
    fertilities, deathes, agedelta, max_iter=100, tol=1e-8, q_init=0.,
    verbose=False
):
    d = deathes
    m = fertilities
    delta = agedelta

    q = q_init
    for i in range(max_iter):
        fq, intermediater = compute_fq(d, m, q, delta)
        err = np.abs(fq)
        if verbose:
            print("%d: q = %.4f, err = %.4f" % (i, q, err))
        if err <= tol:
            return q, {"err": err, "niter": i, "converage": True}
        dfdq = _compute_df_dq(d, m, q, delta, intermediater)
        q -= fq / dfdq
    
    fq, _ = compute_fq(d, m, q, delta)
    err = np.abs(fq)
    if err <= tol:
        return q, {"err": err, "niter": max_iter, "converage": True}
    else:
        print(
            "reach the max iteration, don't converage. "
            "q value: %.4f, fq: %.4f" % (q, err)
        )
    return q, {"err": err, "niter": max_iter, "converage": False}

