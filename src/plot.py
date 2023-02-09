import logging

import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d


def plot_incidence(df_tar, df_ref=None, refline=False):
    if df_ref is None:
        fg = sns.relplot(data=df_tar, x="t", y="incidence",
                         kind="line", aspect=2)
    else:
        df_tar["group"] = "target"
        df_ref["group"] = "reference"
        df_plot = pd.concat([df_tar, df_ref], axis=0)
        fg = sns.relplot(data=df_plot, x="t", y="incidence",
                         hue="group", kind="line", aspect=2)
    if (
        refline and
        4/100000 > df_tar["incidence"].values.min() and
        4/100000 < df_tar["incidence"].values.max()
    ):
        fg.refline(y=4/100000, color="red", linestyle="--")
        ifunc = interp1d(df_tar["incidence"].values, df_tar["t"].values)
        t = ifunc([4/100000])[0]
        fg.refline(x=t, color="red", linestyle="--")
        logging.info("time = %.2f" % t)
    fg.set(yscale="log")
    # fg.set_titles("time = %.2f" % t)
    fg.savefig("./incidence.png")


def plot_death(df_tar, df_ref=None):
    if df_ref is None:
        fg = sns.relplot(data=df_tar, x="t", y="death",
                         kind="line", aspect=2)
    else:
        df_tar["group"] = "target"
        df_ref["group"] = "reference"
        df_plot = pd.concat([df_tar, df_ref], axis=0)
        fg = sns.relplot(data=df_plot, x="t", y="death",
                         hue="group", kind="line", aspect=2)
    fg.set(yscale="log")
    fg.savefig("./death.png")


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
