import logging

import numpy as np


def cal_incidence(y, ycum, model, method="oneyear", verbose=True):
    nrooms_f = model.nrooms_f
    if method == "oneyear":
        # 1年发病率
        cum_cecx = ycum[:, 3].sum(axis=1)
        delta_cecx = cum_cecx[1:] - cum_cecx[:-1]
        nfemale = y[:, :nrooms_f].sum(axis=(1, 2))
        nfemale = (nfemale[1:] +nfemale[:-1]) / 2
        incidence = delta_cecx / nfemale
        # 保证等长度，后面的程序不用修改
        incidence = np.r_[incidence[0], incidence]
    elif method == "instant":
        # 瞬时发病率
        Pf = y[:, 2]
        DeltaLC = model.beta_P * Pf
        incidence = DeltaLC.sum(axis=1) / y[:, :nrooms_f].sum(axis=(1, 2))
    else:
        raise NotImplementedError

    if verbose:
        logging.info("%s incidence rate is :" % method)
        inds = np.linspace(incidence.shape[0] * 0.2, incidence.shape[0]-1, num=10)
        inds = inds.astype(int)
        for ind, value in zip(inds, incidence[inds]):
            logging.info("%d=%.6f" % (ind, value))
    return incidence


def cost_utility_analysis(
    ycum, life_table, cost_per_vacc=55, cost_per_cecx=7547,
    DALY_nofatal=0.76, DALY_fatal=0.86
):
    # 分为两个部分：疫苗接种花费和癌症治疗花费
    nVacc = ycum[:, -1].sum(axis=1)
    nCecx = ycum[:, 3:6].sum(axis=(1, 2))
    nCecxDeathAge = ycum[:, 6:9].sum(axis=1)
    nCecxDeath = nCecxDeathAge.sum(axis=1)

    cVacc = nVacc * cost_per_vacc
    cCecx = nCecx * cost_per_cecx
    cAll = cVacc + cCecx
    dDeath = nCecxDeath * DALY_fatal
    dNoDeath = (nCecx - nCecxDeath) * DALY_nofatal
    lLoss = (nCecxDeathAge * life_table["E"].values).sum(axis=-1)

    # NOTE: 还无法计算ICER值，需要两个策略的差值比

    return {
        "cost_vacc": cVacc,
        "cost_cecx": cCecx,
        "cost_all": cAll,
        "DALY_death": dDeath,
        "DALY_nodeath": dNoDeath,
        "LifeLoss": lLoss,
    }


def cal_icer(cu_tar, cu_ref):
    # NOTE: 使用asarray，不管是ndarray还是series，都转换成了ndarray
    tar_cost = np.asarray(cu_tar["cost_all"])
    ref_cost = np.asarray(cu_ref["cost_all"])
    ref_daly = np.asarray(cu_ref["DALY_nodeath"]) + \
            np.asarray(cu_ref["DALY_death"]) + \
            np.asarray(cu_ref["LifeLoss"])
    tar_daly = np.asarray(cu_tar["DALY_nodeath"]) + \
            np.asarray(cu_tar["DALY_death"]) + \
            np.asarray(cu_tar["LifeLoss"])

    return (tar_cost - ref_cost) / (ref_daly - tar_daly)
