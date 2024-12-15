import os
import os.path as osp
import pickle

import pandas as pd
import numpy as np
import seaborn as sns
import optuna
# import matplotlib.pyplot as plt
from mplfonts import use_font
from optuna.trial import TrialState


use_font('Noto Serif CJK SC')  #指定中文字体
d_results = {
    "策略1": ("./results/opt_res_m100_one", "./results/ref_100"),
    "策略2": ("./results/opt_res_m80_one", "./results/ref_80"),
    "策略3": ("./results/opt_res_m50_one", "./results/ref_50"),
    "策略4": ("./results/opt_res_m30_one", "./results/ref_30"),
}


def print_cost(d_res):
    df = []
    for k, (tar_root, _) in d_res.items():
        tar_cost_vacc = np.load(osp.join(tar_root, "cu_cost_vacc.npy"))
        tar_cost_cecx = np.load(osp.join(tar_root, "cu_cost_cecx.npy"))
        tar_ycum = np.load(osp.join(tar_root, "ycum.npy"))
        nVacc = tar_ycum[:, -1].sum(axis=1)

        with open(osp.join(tar_root, "reference.pkl"), "rb") as f:
            ref = pickle.load(f)

        ref_cost_cecx = ref["cu"]["cost_cecx"]

        diff_cost_cecx = ref_cost_cecx - tar_cost_cecx
        net_cost = tar_cost_vacc - diff_cost_cecx

        avoid_inci = np.load(osp.join(tar_root, "avoid_inci.npy"))
        avoid_death = np.load(osp.join(tar_root, "avoid_death.npy"))
        avoid_inci = avoid_inci.sum(axis=-1)
        avoid_death = avoid_death.sum(axis=(1, 2))

        df.append({
            "time": k,
            "num_vacc": nVacc[-1],
            "cost_vacc": tar_cost_vacc[-1] / 10000,  # 单位是万元
            "cost_cecx_diff": diff_cost_cecx[-1] / 10000,
            "cost_net": net_cost[-1] / 10000,
            "cost_per_inci": net_cost[-1] / avoid_inci[-1],
            "cost_per_death": net_cost[-1] / avoid_death[-1],
        })
    df = pd.DataFrame.from_records(df).set_index("time")
    print(df)


def plot_inci(d_res):
    df = []
    for k, (tar_root, _) in d_res.items():
        inci = np.load(osp.join(tar_root, "inci.npy"))
        t = np.load(osp.join(tar_root, "t.npy"))
        dfi = pd.DataFrame({"t": t, "inci": inci})
        dfi["strategy"] = k
        df.append(dfi)

    tar_root = d_res["策略1"][0]
    with open(osp.join(tar_root, "reference.pkl"), "rb") as f:
        ref = pickle.load(f)
    inci = ref["inci"]
    t = ref["t"]
    dfi = pd.DataFrame({"t": t, "inci": inci})
    dfi["strategy"] = "无疫苗接种"
    df.append(dfi)

    df = pd.concat(df, axis=0)
    fg = sns.relplot(data=df, x="t", y="inci", hue="strategy", kind="line",
                     aspect=2)
    fg.set_xlabels("时间")
    fg.set_ylabels("发病率")
    fg._legend.set_title("")
    fg.savefig("./results/incidence.png")


def plot_inci_ages(d_res):
    df = []
    for k, (tar_root, _) in d_res.items():
        # t = np.load(osp.join(tar_root, "t.npy"))
        y = np.load(osp.join(tar_root, "y.npy"))
        with open(osp.join(tar_root, "model.pkl"), "rb") as f:
            model = pickle.load(f)

        # 瞬时发病率
        Pf = y[:, 2]
        DeltaLC = model.beta_P * Pf
        inci = DeltaLC / y[:, :model.nrooms_f].sum(axis=1)
        inci = inci[-1]
        dfi = pd.DataFrame({
            "age": pd.Categorical(model.agebin_names,
                                  categories=model.agebin_names,
                                  ordered=True),
            "inci": inci
        })
        dfi["strategy"] = k

        df.append(dfi)
    df = pd.concat(df)

    fg = sns.relplot(data=df, x="age", y="inci", hue="strategy", kind="line",
                     aspect=2)
    fg.set_xlabels("年龄组")
    fg.set_ylabels("发病率")
    fg.set_xticklabels(rotation=45)
    fg._legend.set_title("")
    fg.savefig("./results/incidence_age.png")


def plot_death(d_res):
    df = []
    for k, (tar_root, _) in d_res.items():
        t = np.load(osp.join(tar_root, "t.npy"))
        y = np.load(osp.join(tar_root, "y.npy"))
        ycum = np.load(osp.join(tar_root, "ycum.npy"))

        deathi = ycum[:, 6:9].sum(axis=(1, 2))
        deathi = deathi[1:] - deathi[:-1]
        popu = y.sum(axis=(1, 2))
        popu = (popu[1:] + popu[:-1]) / 2
        deathp = deathi / popu

        dfi = pd.DataFrame({"t": (t[1:] + t[:-1]) / 2, "deathp": deathp})
        dfi["strategy"] = k
        df.append(dfi)

    tar_root = d_res["策略1"][0]
    with open(osp.join(tar_root, "reference.pkl"), "rb") as f:
        ref = pickle.load(f)
    t = ref["t"]
    y = ref["y"]
    ycum = ref["ycum"]
    deathi = ycum[:, 6:9].sum(axis=(1, 2))
    deathi = deathi[1:] - deathi[:-1]
    popu = y.sum(axis=(1, 2))
    popu = (popu[1:] + popu[:-1]) / 2
    deathp = deathi / popu
    dfi = pd.DataFrame({"t": (t[1:] + t[:-1]) / 2, "deathp": deathp})
    dfi["strategy"] = "无疫苗接种"
    df.append(dfi)

    df = pd.concat(df, axis=0)
    fg = sns.relplot(data=df, x="t", y="deathp", hue="strategy", kind="line",
                     aspect=2)
    fg.set_xlabels("时间")
    fg.set_ylabels("因宫颈癌死亡率")
    fg._legend.set_title("")
    fg.savefig("./results/deathp.png")


def plot_death_ages(d_res):
    df = []
    for k, (tar_root, _) in d_res.items():
        # t = np.load(osp.join(tar_root, "t.npy"))
        y = np.load(osp.join(tar_root, "y.npy"))
        ycum = np.load(osp.join(tar_root, "ycum.npy"))
        with open(osp.join(tar_root, "model.pkl"), "rb") as f:
            model = pickle.load(f)

        deathi = ycum[:, 6:9].sum(axis=1)
        deathi = deathi[1:] - deathi[:-1]
        popu = y.sum(axis=1)
        popu = (popu[1:] + popu[:-1]) / 2
        deathp = deathi / popu
        deathp = deathp[-1]

        dfi = pd.DataFrame({
            "age": pd.Categorical(model.agebin_names,
                                  categories=model.agebin_names,
                                  ordered=True),
            "death": deathp
        })
        dfi["strategy"] = k

        df.append(dfi)
    df = pd.concat(df)

    fg = sns.relplot(data=df, x="age", y="death", hue="strategy", kind="line",
                     aspect=2)
    fg.set_xlabels("年龄组")
    fg.set_ylabels("宫颈癌死亡率")
    fg.set_xticklabels(rotation=45)
    fg._legend.set_title("")
    fg.savefig("./results/death_age.png")


def print_avoid(d_res):
    df = []
    for k, (tar_root, _) in d_res.items():
        tar_cost_vacc = np.load(osp.join(tar_root, "cu_cost_vacc.npy"))
        tar_cost_cecx = np.load(osp.join(tar_root, "cu_cost_cecx.npy"))
        # tar_ycum = np.load(osp.join(tar_root, "ycum.npy"))
        # nVacc = tar_ycum[:, -1].sum(axis=1)

        with open(osp.join(tar_root, "reference.pkl"), "rb") as f:
            ref = pickle.load(f)

        ref_cost_cecx = ref["cu"]["cost_cecx"]

        diff_cost_cecx = ref_cost_cecx - tar_cost_cecx
        net_cost = tar_cost_vacc - diff_cost_cecx

        avoid_inci = np.load(osp.join(tar_root, "avoid_inci.npy"))
        avoid_death = np.load(osp.join(tar_root, "avoid_death.npy"))
        avoid_inci = avoid_inci.sum(axis=-1)
        avoid_death = avoid_death.sum(axis=(1, 2))

        tar_daly = np.load(osp.join(tar_root, "cu_DALY_nodeath.npy")) + \
            np.load(osp.join(tar_root, "cu_DALY_death.npy")) + \
            np.load(osp.join(tar_root, "cu_LifeLoss.npy"))
        ref_daly = ref["cu"]["DALY_nodeath"] + ref["cu"]["DALY_death"] +\
            ref["cu"]["LifeLoss"]
        avoid_daly = ref_daly - tar_daly

        df.append({
            "time": k,
            "avoid_inci": avoid_inci[-1],
            "avoid_death": avoid_death[-1],
            "aovid_DALY": avoid_daly[-1],
            "cost_per_inci": net_cost[-1] / avoid_inci[-1],
            "cost_per_death": net_cost[-1] / avoid_death[-1],
            "cost_per_DALY(ICER)": net_cost[-1] / avoid_daly[-1],
        })
    df = pd.DataFrame.from_records(df).set_index("time")
    print(df)


def plot_avoid(d_res):
    df = []
    for k, (tar_root, _) in d_res.items():
        t = np.load(osp.join(tar_root, "t.npy"))
        tar_cost_vacc = np.load(osp.join(tar_root, "cu_cost_vacc.npy"))
        tar_cost_cecx = np.load(osp.join(tar_root, "cu_cost_cecx.npy"))
        # tar_ycum = np.load(osp.join(tar_root, "ycum.npy"))
        # nVacc = tar_ycum[:, -1].sum(axis=1)

        with open(osp.join(tar_root, "reference.pkl"), "rb") as f:
            ref = pickle.load(f)

        ref_cost_cecx = ref["cu"]["cost_cecx"]

        diff_cost_cecx = ref_cost_cecx - tar_cost_cecx
        net_cost = tar_cost_vacc - diff_cost_cecx

        avoid_inci = np.load(osp.join(tar_root, "avoid_inci.npy"))
        avoid_death = np.load(osp.join(tar_root, "avoid_death.npy"))
        avoid_inci = avoid_inci.sum(axis=-1)
        avoid_death = avoid_death.sum(axis=(1, 2))

        tar_daly = np.load(osp.join(tar_root, "cu_DALY_nodeath.npy")) + \
            np.load(osp.join(tar_root, "cu_DALY_death.npy")) + \
            np.load(osp.join(tar_root, "cu_LifeLoss.npy"))
        ref_daly = ref["cu"]["DALY_nodeath"] + ref["cu"]["DALY_death"] +\
            ref["cu"]["LifeLoss"]
        avoid_daly = ref_daly - tar_daly

        dfi = pd.DataFrame({
            "t": t,
            "避免发病数": avoid_inci,
            "避免死亡数": avoid_death,
            "避免损失DALY数": avoid_daly,
            "每避免一例发病的成本": net_cost / avoid_inci,
            "每避免一例死亡的成本": net_cost / avoid_death,
            "增量成本效果比(ICER)": net_cost / avoid_daly,
        })
        dfi["strategy"] = k
        df.append(dfi)

    df = pd.concat(df)
    df = df.melt(id_vars=["t", "strategy"], var_name="var", value_name="value")
    df = df[df["value"] < 30000]
    fg = sns.relplot(data=df, x="t", y="value", hue="strategy",
                     col="var", col_wrap=3,
                     facet_kws={"sharey": False,
                                "sharex": False},
                     aspect=1.0, kind="line")
    fg.set_titles(col_template="{col_name}")
    fg.set_xlabels("时间")
    fg.set_ylabels("")
    # fg.refline(y=4e-5)
    fg._legend.set_title("")
    fg.savefig("./results/avoid.png")



def plot_optim_hist(d_res):
    ori_cwd = os.getcwd()

    df = []
    for k, (tar_root, _) in d_res.items():

        os.chdir(tar_root)
        study_name = "HPV_tdm"
        storage_name = "sqlite:///{}.db".format(study_name)
        with open("sampler.pkl", "rb") as f:
            sampler = pickle.load(f)
        study = optuna.create_study(study_name=study_name,
                                    storage=storage_name,
                                    sampler=sampler,
                                    directions=["minimize", "minimize"],
                                    load_if_exists=True)

        dfi = []
        trials = study.get_trials(states=(TrialState.COMPLETE,))
        min_value = np.inf
        for trial in trials:
            value = {"Trial": trial.number, "ICER": trial.values[0]}
            if value["ICER"] < min_value:
                min_value = value["ICER"]
                value["is_best"] = True
            else:
                value["is_best"] = False

            dfi.append(value)
        dfi = pd.DataFrame.from_records(dfi)
        dfi["strategy"] = {
            "策略1": "100",
            "策略2": "80",
            "策略3": "50",
            "策略4": "30",
        }[k]
        df.append(dfi)
        os.chdir(ori_cwd)

    cmap = sns.color_palette()
    df = pd.concat(df)
    fg = sns.relplot(data=df, x="Trial", y="ICER",
                     col="strategy", col_wrap=2, alpha=0.7, size=0.5,
                     facet_kws={"sharey": False, "sharex": False},
                     aspect=1.0)
    fg.set_titles(col_template="实施时间={col_name}年")
    for k, ax in fg.axes_dict.items():
        dfi = df.query("strategy == '%s'" % k)
        dfi = dfi.loc[dfi["is_best"], :]
        ax.plot(
            dfi["Trial"].values,
            dfi["ICER"].values,
            marker="o",
            color=cmap[3],
            alpha=1.0,
            label=None
        )

    fg.savefig("./results/optim_hist.png")


def plot_pareto(d_res):
    ori_cwd = os.getcwd()

    df = []
    for k, (tar_root, _) in d_res.items():

        os.chdir(tar_root)
        study_name = "HPV_tdm"
        storage_name = "sqlite:///{}.db".format(study_name)
        with open("sampler.pkl", "rb") as f:
            sampler = pickle.load(f)
        study = optuna.create_study(study_name=study_name,
                                    storage=storage_name,
                                    sampler=sampler,
                                    directions=["minimize", "minimize"],
                                    load_if_exists=True)

        dfi = []
        trials = study.get_trials(states=(TrialState.COMPLETE,))
        for trial in trials:
            value = {"Trial": trial.number,
                     "ICER": trial.values[0],
                     "Incidence": trial.values[1]}
            dfi.append(value)
        dfi = pd.DataFrame.from_records(dfi)

        best_indice = []
        best_icer, bbest_ind = np.inf, None
        for trial in study.best_trials:
            best_indice.append(trial.number)
            if trial.values[0] < best_icer and trial.values[1] <= 4e-5:
                best_icer = trial.values[0]
                bbest_ind = trial.number
        dfi["best"] = "Trial"
        dfi.loc[dfi["Trial"].isin(best_indice), "best"] = "Best Trial"
        dfi.loc[dfi.Trial == bbest_ind, "best"] = "Choice"

        dfi["strategy"] = {
            "策略1": "100",
            "策略2": "80",
            "策略3": "50",
            "策略4": "30",
        }[k]
        df.append(dfi)
        os.chdir(ori_cwd)

    df = pd.concat(df)

    cmap = sns.color_palette()
    fg = sns.relplot(data=df, x="ICER", y="Incidence", hue="best", size="best",
                     col="strategy", col_wrap=2, alpha=0.7,
                     facet_kws={"sharey": False, "sharex": False},
                     aspect=1.5, palette=[cmap[0], cmap[1], cmap[3]],
                     sizes={"Trial": 30, "Best Trial": 90, "Choice": 500})
    fg.set_titles(col_template="实施时间={col_name}年")
    # fg.refline(y=4e-5)
    fg._legend.set_title("")
    fg.savefig("./results/pareto.png")


def plot_cost(d_res):
    df = []
    for k, (tar_root, _) in d_res.items():
        t = np.load(osp.join(tar_root, "t.npy"))
        tar_cost_vacc = np.load(osp.join(tar_root, "cu_cost_vacc.npy"))
        tar_cost_cecx = np.load(osp.join(tar_root, "cu_cost_cecx.npy"))
        tar_ycum = np.load(osp.join(tar_root, "ycum.npy"))
        nVacc = tar_ycum[:, -1].sum(axis=1)

        with open(osp.join(tar_root, "reference.pkl"), "rb") as f:
            ref = pickle.load(f)

        ref_cost_cecx = ref["cu"]["cost_cecx"]

        diff_cost_cecx = ref_cost_cecx - tar_cost_cecx
        net_cost = tar_cost_vacc - diff_cost_cecx

        avoid_inci = np.load(osp.join(tar_root, "avoid_inci.npy"))
        avoid_death = np.load(osp.join(tar_root, "avoid_death.npy"))
        avoid_inci = avoid_inci.sum(axis=-1)
        avoid_death = avoid_death.sum(axis=(1, 2))

        dfi = pd.DataFrame({
            "t": t,
            "接种人数": nVacc,
            "疫苗成本": tar_cost_vacc,
            "节约治疗成本": diff_cost_cecx,
            "净节约成本": -net_cost
        })
        dfi["strategy"] = k
        # {
        #     "策略1": "100",
        #     "策略2": "80",
        #     "策略3": "50",
        #     "策略4": "30",
        # }[k]
        df.append(dfi)
    df = pd.concat(df)
    df = df.melt(id_vars=["t", "strategy"], var_name="var", value_name="value")
    fg = sns.relplot(data=df, x="t", y="value", hue="strategy",
                     col="var", col_wrap=2,
                     facet_kws={"sharey": False, "sharex": False},
                     aspect=1.0, kind="line")
    fg.set_titles(col_template="{col_name}")
    # fg.refline(y=4e-5)
    fg._legend.set_title("")
    fg.savefig("./results/cost.png")


if __name__ == "__main__":
    # print_cost(d_results)
    # print_avoid(d_results)
    # plot_inci(d_results)
    # plot_death(d_results)
    # plot_optim_hist(d_results)
    # plot_pareto(d_results)
    # plot_cost(d_results)
    # plot_inci_ages(d_results)
    # plot_death_ages(d_results)
    plot_avoid(d_results)
