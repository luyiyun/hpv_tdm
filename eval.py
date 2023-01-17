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


def cal_incidence(y, model):
    nrooms_f = model.nrooms_f
    Ntf = y[:, :nrooms_f, :].sum(axis=1)
    Ntm = y[:, nrooms_f:, :].sum(axis=1)
    Sf, If, Pf = y[:, 0], y[:, 1], y[:, 2]
    Sm, Im, Pm = y[:, 0], y[:, nrooms_f+1], y[:, nrooms_f+2]

    # 计算一下alpha（感染率）
    iPf = (If + Pf) / Ntf
    iPm = (Im + Pm) / Ntm
    alpha_f_age = model.epsilon_f * model.omega_f * (iPm @ model.rho.T)
    alpha_m_age = model.epsilon_m * model.omega_m * (iPf @ model.rho.T)

    alpha_f_all = (alpha_f_age * Sf).sum(axis=1) / Sf.sum(axis=1)
    alpha_m_all = (alpha_m_age * Sm).sum(axis=1) / Sm.sum(axis=1)

    return {
        "female": alpha_f_all, "male": alpha_m_all,
        # "female_age": alpha_f_age, "male_age": alpha_m_age
    }


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
    incidences = cal_incidence(y, model)
    cost_utilities = cost_utility_analysis(ycum, ltable)

    # 3.整理成dataframe
    incidences["t"] = t
    df_inci = pd.DataFrame(incidences)
    cost_utilities["t"] = t
    df_cu = pd.DataFrame(cost_utilities)
 
    return df_inci, df_cu


def plot_incidence(df_tar, df_ref=None):
    df_tar = df_tar.melt(id_vars="t",
                         var_name="gender", value_name="incidence")
    if df_ref is None:
        fg = sns.relplot(data=df_tar, x="t", y="incidence",
                         hue="gender", kind="line", aspect=2)
    else:
        df_ref = df_ref.melt(id_vars="t",
                             var_name="gender", value_name="incidence")
        df_tar["group"] = "target"
        df_ref["group"] = "reference"
        df_plot = pd.concat([df_tar, df_ref], axis=0)
        fg = sns.relplot(data=df_plot, x="t", y="incidence",
                         hue="group", col="gender", kind="line", aspect=1.5)
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
