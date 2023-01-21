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
from src.evaluation import cal_incidence, cost_utility_analysis, cal_icer


logging.basicConfig(level=logging.INFO,
                    format="[%(levelname)s][%(asctime)s]%(message)s")


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

    cost_utilities["t"] = t
    cost_utilities["incidence"] = incidences
    return cost_utilities


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

    tar_scores = load_and_calculate(".")
    tar_df = pd.DataFrame(tar_scores).set_index("t", drop=False)
    if args.reference is not None:
        data_ref = osp.join(abs_root, args.reference)
        ref_scores = load_and_calculate(data_ref)
        ref_df = pd.DataFrame(ref_scores).set_index("t", drop=False)
        ref_df = ref_df.loc[tar_df.index, :]  # 保证tar和ref有相同的t
        plot_incidence(tar_df[["t", "incidence"]], ref_df[["t", "incidence"]])
        plot_cost(
            tar_df[["t", "cost_vacc", "cost_cecx", "cost_all"]],
            ref_df[["t", "cost_vacc", "cost_cecx", "cost_all"]],
        )
        plot_DALY(
            tar_df[["t", "DALY_nodeath", "DALY_death"]],
            ref_df[["t", "DALY_nodeath", "DALY_death"]],
        )

        icer = cal_icer(tar_scores, ref_scores)
        df_icer = pd.DataFrame({"t": tar_scores["t"], "ICER": icer})
        # < 10年的部分数值太大，无法看到最终ICER值大约是多少
        fg = sns.relplot(data=df_icer[df_icer["t"] > 10],
                         x="t", y="ICER", aspect=2, kind="line")
        # fg.set(yscale="log")
        fg.savefig("./ICER_vs_%s.png" % args.reference.replace("/", "-"))
    else:
        plot_incidence(tar_df[["t", "incidence"]])
        plot_cost(tar_df[["t", "cost_vacc", "cost_cecx", "cost_all"]])
        plot_DALY(tar_df[["t", "DALY_nodeath", "DALY_death"]])


if __name__ == "__main__":
    main()
