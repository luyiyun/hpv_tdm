import os
import os.path as osp
import argparse
import re
import pickle
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns

from src import life_table


logging.basicConfig(level=logging.INFO,
                    format="[%(levelname)s][%(asctime)s]%(message)s")


def cal_incidence(y, ycum, model, method="oneyear"):
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

    logging.info("%s incidence rate is :" % method)
    inds = np.linspace(incidence.shape[0] * 0.2, incidence.shape[0]-1, num=10)
    inds = inds.astype(int)
    for ind, value in zip(inds, incidence[inds]):
        logging.info("%d=%.6f" % (ind, value))
    return incidence


def cost_utility_analysis(
    ycum, life_table, cost_per_vacc=55, cost_per_cecx=650,
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


def load_and_calculate(root):
    # 1.载入数据
    with open(osp.join(root, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    t = np.load(osp.join(root, "t.npy"))
    y = np.load(osp.join(root, "y.npy"))
    ycum = np.load(osp.join(root, "ycum.npy"))

    # 2.计算指标
    ltable = life_table(model.deathes_female, model.agebins)
    incidences = cal_incidence(y, ycum, model)
    cost_utilities = cost_utility_analysis(ycum, ltable)

    # 3.整理成dataframe
    df_inci = pd.DataFrame({"t": t, "incidence": incidences})
    cost_utilities["t"] = t
    df_cu = pd.DataFrame(cost_utilities)
 
    return df_inci, df_cu


def plot_incidence(df_tar, df_ref=None):
    if df_ref is None:
        fg = sns.relplot(data=df_tar, x="t", y="incidence",
                         kind="line", aspect=2)
    else:
        df_tar["group"] = "target"
        df_ref["group"] = "reference"
        df_plot = pd.concat([df_tar, df_ref], axis=0)
        fg = sns.relplot(data=df_plot, x="t", y="incidence",
                         hue="group", kind="line", aspect=2)
    # fg.refline(y=4/100000, color="red", linestyle="--")
    fg.set(yscale="log")
    fg.savefig("./incidence.png")


def plot_cost(df_tar, df_ref=None):
    df_tar = df_tar.melt(id_vars="t",
                         var_name="type", value_name="cost")
    if df_ref is None:
        fg = sns.relplot(data=df_tar, x="t", y="cost",
                         hue="type", kind="line", aspect=2)
    else:
        df_ref = df_ref.melt(id_vars="t",
                             var_name="type", value_name="cost")
        df_tar["group"] = "target"
        df_ref["group"] = "reference"
        df_plot = pd.concat([df_tar, df_ref], axis=0)
        fg = sns.relplot(data=df_plot, x="t", y="cost",
                         hue="group", col="type", kind="line", aspect=1.5)
    fg.savefig("./cost.png")


def plot_DALY(df_tar, df_ref=None):
    df_tar = df_tar.melt(id_vars="t",
                         var_name="type", value_name="DALY")
    if df_ref is None:
        fg = sns.relplot(data=df_tar, x="t", y="DALY",
                         hue="type", kind="line", aspect=2)
    else:
        df_ref = df_ref.melt(id_vars="t",
                             var_name="type", value_name="DALY")
        df_tar["group"] = "target"
        df_ref["group"] = "reference"
        df_plot = pd.concat([df_tar, df_ref], axis=0)
        fg = sns.relplot(data=df_plot, x="t", y="DALY",
                         hue="group", col="type", kind="line", aspect=1.5)
    fg.savefig("./DALY.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-target", type=str, default=None)
    parser.add_argument("-reference", type=str, default=None)
    args = parser.parse_args()

    abs_root = "/home/luyiyun/Documents/hpv_tdm/results"
    if args.target is None:
        root_dates = []
        for di in os.listdir(abs_root):
            if re.search(r"2023-\d{2}-\d{2}", di):
                fdi = osp.join("./results", di)
                for dj in os.listdir(fdi):
                    if re.search(r"\d{2}-\d{2}-\d{2}", dj):
                        dj = osp.join(di, dj)
                        date_dj = datetime.strptime(dj, "%Y-%m-%d/%H-%M-%S")
                        root_dates.append(date_dj)
        root_dates = max(root_dates)
        root_dir = osp.join(abs_root, root_dates.strftime("%Y-%m-%d/%H-%M-%S"))
    else:
        root_dir = osp.join(abs_root, args.target)
    os.chdir(root_dir)
    logging.info("[main] CWD is %s" % os.getcwd())

    df_in_target, df_cu_target = load_and_calculate(".")
    if args.reference is not None:
        data_ref = osp.join(abs_root, args.reference)
        df_in_ref, df_cu_ref = load_and_calculate(data_ref)
        plot_incidence(df_in_target, df_in_ref)
        plot_cost(
            df_cu_target.loc[:, ["t", "cost_vacc", "cost_cecx", "cost_all"]],
            df_cu_ref.loc[:, ["t", "cost_vacc", "cost_cecx", "cost_all"]]
        )
        plot_DALY(
            df_cu_target.loc[:, ["t", "DALY_nodeath", "DALY_death"]],
            df_cu_ref.loc[:, ["t", "DALY_nodeath", "DALY_death"]]
        )

        icer = (df_cu_target["cost_all"] - df_cu_ref["cost_all"]) / (
            df_cu_ref[["DALY_nodeath", "DALY_death", "LifeLoss"]].sum(axis=1)-
            df_cu_target[["DALY_nodeath", "DALY_death", "LifeLoss"]].sum(axis=1)
        )
        df_icer = pd.DataFrame({"t": df_cu_target["t"], "ICER": icer})
        # < 10年的部分数值太大，无法看到最终ICER值大约是多少
        fg = sns.relplot(data=df_icer[df_icer["t"] > 10],
                         x="t", y="ICER", aspect=2, kind="line")
        # fg.set(yscale="log")
        fg.savefig("./ICER_vs_%s.png" % args.reference.replace("/", "-"))
    else:
        plot_incidence(df_in_target)
        plot_cost(
            df_cu_target.loc[:, ["t", "cost_vacc", "cost_cecx", "cost_all"]])
        plot_DALY(
            df_cu_target.loc[:, ["t", "DALY_nodeath", "DALY_death"]])


if __name__ == "__main__":
    main()
