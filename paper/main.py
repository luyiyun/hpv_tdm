import os
import os.path as osp
import pickle
from typing import Literal
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from optuna.trial import TrialState
import h5py
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from matplotlib.scale import SymmetricalLogScale
from lifelines import WeibullAFTFitter

from hpv_tdm import Evaluator


plt.rcParams["font.family"] = "Times New Roman"
jama_palette = {
    "Limed Spruce": "#374E55",
    "Anzac": "#DF8F44",
    "Cerulean": "#00A1D5",
    "Apple Blossom": "#B24745",
    "Acapulco": "#79AF97",
    "Kimberly": "#6A6599",
    "Makara": "#80796B",
}
nejm_palette = {
    "TallPoppy": "#BC3C29",
    "DeepCerulean": "#0072B5",
    "Zest": "#E18727",
    "Eucalyptus": "#20854E",
    "WildBlueYonder": "#7876B1",
    "Gothic": "#6F99AD",
    "Salomie": "#FFDC91",
    "FrenchRose": "#EE4C97",
}


def plot_for_scenario(
    df: pd.DataFrame,
    x_colname: str,
    y_colname: str,
    scenario_colname: str = "scenario",
    best_colname: str = "is_best",
    nrows: int = 2,
    ncols: int = 3,
    x_label: str | None = None,
    y_lable: str | None = None,
    colors: tuple[str, str] = (
        jama_palette["Limed Spruce"],
        jama_palette["Apple Blossom"],
    ),
    normal_marker: str = ".",
    best_marker: str = "o-",
    normal_alpha: float = 0.5,
    best_alpha: float = 1,
) -> tuple[plt.Figure, plt.Axes]:
    fig, axs = plt.subplots(
        nrows=2, ncols=3, figsize=(12, 8), sharex=False, sharey=False
    )
    scenario_names = df[scenario_colname].unique()
    for ri, ci in product(range(nrows), range(ncols)):
        ax = axs[ri, ci]
        sname = scenario_names[ri * ncols + ci]
        dfi = df.query(f"scenario == '{sname}'")

        if ri == 1:
            ax.set_xlabel(x_label or x_colname)
        if ci == 0:
            ax.set_ylabel(y_lable or y_colname)
        ax.set_title(sname)

        ax.plot(
            dfi[x_colname].values,
            dfi[y_colname].values,
            normal_marker,
            color=colors[0],
            alpha=normal_alpha,
            # label=None,
        )
        # 将最优的ICUR标记出来
        dfi_best = dfi.query(best_colname)
        ax.plot(
            dfi_best[x_colname].values,
            dfi_best[y_colname].values,
            best_marker,
            color=colors[1],
            alpha=best_alpha,
            # label="Best ICUR",
        )

        # 强制要求y轴为科学计数法
        ax.ticklabel_format(
            style="sci", axis="y", useMathText=True, scilimits=(0, 0)
        )

    return fig, axs


def get_studies(
    results_dict: dict[str, dict[str, str]],
) -> list[tuple[str, str, optuna.study.Study]]:
    """
    results_dict: 第一个key是年限，第二个key是是否贴现，value是对应的结果路径
    """

    # 临时数据库路径，创建在每个训练结果路径下，用于缓存optuna的study对象
    storage_name = "sqlite:///HPV_tdm.db"
    ori_dir = os.getcwd()

    # 载入训练结果，训练结果是optuna的study对象
    # d_trials, d_best_trials, d_choice_trial = {}, {}, {}
    studies = []
    for scenario, results_dict_i in results_dict.items():
        # d_trials_i, d_best_trials_i, d_choice_trial_i = [], [], []
        for disc_flag, tar_root in results_dict_i.items():
            tar_root = os.path.abspath(tar_root)
            os.chdir(tar_root)
            with open("sampler.pkl", "rb") as f:
                sampler = pickle.load(f)
            study = optuna.create_study(
                study_name="HPV_tdm",
                storage=storage_name,
                sampler=sampler,
                directions=["minimize", "minimize"],
                load_if_exists=True,
            )
            studies.append((scenario, disc_flag, study))
            os.chdir(ori_dir)

    return studies
    # # 所有的trials
    # trials = study.get_trials(states=(TrialState.COMPLETE,))
    #     d_trials_i.append(trials)
    #     # 所有的best trials
    #     d_best_trials_i.append(study.best_trials)
    #     # 找到那个我们选择的策略
    #     best_icer, choice_trial = np.inf, None
    #     for trial in study.best_trials:
    #         if trial.values[0] < best_icer and trial.values[1] <= incidence_threshold:
    #             best_icer = trial.values[0]
    #             choice_trial = trial
    #     d_choice_trial_i.append(choice_trial)
    # d_trials[scenario] = d_trials_i
    # d_best_trials[scenario] = d_best_trials_i
    # d_choice_trial[scenario] = d_choice_trial_i
    # os.chdir(ori_cwd)


def figure_history(
    studies: list[tuple[str, str, optuna.study.Study]],
    figname: str | None = None,
    exchange_rate: float = 6.8967,
    disc_flag: Literal["no_disc", "discount"] = "discount",
    colors: tuple[str, str] = (
        jama_palette["Limed Spruce"],
        jama_palette["Apple Blossom"],
    ),
):
    df_hist = []
    for scenario, disc_flag, study in studies:
        trials = study.get_trials(states=(TrialState.COMPLETE,))
        dfi = []
        min_value = np.inf
        for trial in trials:
            value = {"Trial": trial.number, "ICUR": trial.values[0]}
            if value["ICUR"] < min_value:
                min_value = value["ICUR"]
                value["is_best"] = True
            else:
                value["is_best"] = False

            dfi.append(value)
        dfi = pd.DataFrame.from_records(dfi)
        dfi["scenario"] = scenario
        dfi["discount"] = disc_flag
        df_hist.append(dfi)
    df_hist = pd.concat(df_hist)
    df_hist["ICUR"] = df_hist["ICUR"] * exchange_rate

    fig, _ = plot_for_scenario(
        df_hist.query("discount == '%s'" % disc_flag),
        x_colname="Trial",
        y_colname="ICUR",
        scenario_colname="scenario",
        best_colname="is_best",
        nrows=2,
        ncols=3,
        x_label="Iterations",
        y_lable="ICUR",
        normal_marker=".",
        best_marker="o-",
        colors=colors,
    )

    if figname is not None:
        fig.savefig(figname)
    else:
        plt.show()


def figure_pareto(
    studies: list[tuple[str, str, optuna.study.Study]],
    figname: str | None = None,
    exchange_rate: float = 6.8967,
    disc_flag: Literal["no_disc", "discount"] = "discount",
    incidence_threshold: float = 4e-5,
    colors: tuple[str, str] = (
        jama_palette["Limed Spruce"],
        jama_palette["Apple Blossom"],
    ),
):
    df_pareto = []
    for scenario, disc_flag, study in studies:
        trials = study.get_trials(states=(TrialState.COMPLETE,))
        dfi = pd.DataFrame.from_records(
            [
                {
                    "Trial": trial.number,
                    "ICUR": trial.values[0],
                    "Incidence": trial.values[1],
                }
                for trial in trials
            ]
        )
        # study.best_trials储存了所有的位于帕累托前沿中的trials
        pareto_front_indice = [trial.number for trial in study.best_trials]
        # 找到最优策略
        best_index = list(
            sorted(
                filter(
                    lambda x: x.values[1] <= incidence_threshold,
                    study.best_trials,
                ),
                key=lambda x: x.values[0],
            )
        )[0].number
        dfi["pareto_front"] = dfi["Trial"].isin(pareto_front_indice)
        dfi["is_best"] = dfi["Trial"] == best_index
        dfi["discount"] = disc_flag
        dfi["scenario"] = scenario
        df_pareto.append(dfi)
    df_pareto = pd.concat(df_pareto)
    df_pareto["ICUR"] = df_pareto["ICUR"] * exchange_rate

    fig, _ = plot_for_scenario(
        df_pareto.query(f"discount == '{disc_flag}'"),
        x_colname="ICUR",
        y_colname="Incidence",
        scenario_colname="scenario",
        best_colname="pareto_front",
        nrows=2,
        ncols=3,
        normal_marker=".",
        best_marker=".",
        normal_alpha=1.0,
        best_alpha=1.0,
        colors=colors,
    )
    if figname is not None:
        fig.savefig(figname)
    else:
        plt.show()


def save_evaluation_to_hdf5(
    eval_obj,
    hdf5_path: str,
):
    with h5py.File(hdf5_path, "w") as f:
        g_model = f.create_group("model")
        for k, v in eval_obj.model.kwargs.items():
            if isinstance(v, (int, float, bool, str)):
                g_model.attrs[k] = v
            elif isinstance(v, np.ndarray):
                g_model.create_dataset(k, data=v)
            elif isinstance(v, (tuple, list)):
                g_model.create_dataset(k, data=np.array(v))
            else:
                raise ValueError(f"Unsupported type of {k}: {type(v)}")

        g_evaluation = f.create_group("evaluation")
        g_evaluation.attrs["cost_pvacc"] = eval_obj.cost_pvacc
        g_evaluation.attrs["cost_pcecx"] = eval_obj.cost_pcecx
        g_evaluation.attrs["daly_nof"] = eval_obj.daly_nof
        g_evaluation.attrs["daly_f"] = eval_obj.daly_f
        g_evaluation.attrs["discount"] = eval_obj.discount_rate or 0.0
        g_eval_result = g_evaluation.create_group("result")
        for k, v in eval_obj.result.items():
            if k == "model":
                continue
            g_eval_result.create_dataset(k, data=v)

    eval_obj.ltable.reset_index(drop=True).to_hdf(
        hdf5_path, "evaluation/ltable"
    )


def convert_pkl_to_hdf5(results_dict: dict[str, dict[str, str]]):
    """
    在早期版本中，我将evaluation对象保存为pkl格式，其中evaluation对象中包含了model、ltable、result等信息。
    这样做的好处是可以直接得到evaluation对象，并直接使用其方法进行分析。
    但是，pkl格式的储存方式需要代码结构保持不变，比如原来定义evaluation对象是在src.evaluation模块中，现在
      修改到新的路径下，则pkl就无法保持。或者某些库的版本更新也会导致pkl无法读取。
    这里我们修改为hdf5格式进行储存，储存结构为：
    model/：模型参数
        ndarray, tuple, list等使用dataset保存
        其他类型使用attrs保存
    evaluation/：评估结果
        ltable: 使用dataframe.to_hdf保存
        result: 其是一个字典，所以其每个元素使用dataset保存
        其他使用attrs保存，其中discount_rate是None时，使用attrs保存为0.0
    """
    for _, res_dict_i in results_dict.items():
        for _, trial_root in res_dict_i.items():
            # 将原来的pkl格式储存的evaluation对象转换为hdf5格式进行储存
            with open(osp.join(trial_root, "tar_eval.pkl"), "rb") as f:
                tar_eval = pickle.load(f)
                save_evaluation_to_hdf5(
                    tar_eval,
                    osp.join(trial_root, "tar_eval.h5"),
                )
            with open(osp.join(trial_root, "ref_eval.pkl"), "rb") as f:
                ref_eval = pickle.load(f)
                save_evaluation_to_hdf5(
                    ref_eval,
                    osp.join(trial_root, "ref_eval.h5"),
                )


def get_burden_df(
    results_dict: dict[str, dict[str, str]],
    disc_flag: str,
    burden_type: Literal["incidence", "mortality"] = "incidence",
) -> pd.DataFrame:
    df = []
    for scenario, res_dict_i in results_dict.items():
        trial_root = res_dict_i[disc_flag]
        tar_evaluator = Evaluator.from_hdf(osp.join(trial_root, "tar_eval.h5"))
        if burden_type == "incidence":
            burden_i = tar_evaluator.cal_incidence(reduce=False, reuse=True)
        elif burden_type == "mortality":
            burden_i = tar_evaluator.cal_mortality(reduce=False, reuse=True)
        else:
            raise ValueError(f"Unsupported burden_type: {burden_type}")

        agebins = tar_evaluator.model.agebin_names
        # agebins = np.array([",".join(ab.split(", ")) for ab in agebins])
        dfi = pd.DataFrame(
            {
                "t": np.repeat(tar_evaluator.t_, burden_i.shape[1]),
                "age": pd.Categorical(
                    np.tile(agebins, burden_i.shape[0]),
                    categories=agebins,
                    ordered=True,
                ),
                "burden": burden_i.flatten(),
            }
        )
        dfi["scenario"] = scenario
        df.append(dfi)
    df = pd.concat(df, axis=0)
    df["scenario"] = pd.Categorical(
        df["scenario"],
        categories=list(results_dict.keys()),
        ordered=True,
    )
    return df


def figure_burden_total(
    results_dict: dict[str, dict[str, str]],
    fig_name: str | None = None,
    disc_flag: Literal["no_disc", "discount"] = "discount",
    colors: tuple[str, ...] = tuple(jama_palette.values()),
    figsize: tuple[float, float] = (8, 8),
    burden_type: Literal["incidence", "mortality"] = "incidence",
):
    assert burden_type in ["incidence", "mortality"], "Unsupported burden_type"

    df = []
    for scenario, res_dict_i in results_dict.items():
        trial_root = res_dict_i[disc_flag]
        tar_evaluator = Evaluator.from_hdf(osp.join(trial_root, "tar_eval.h5"))
        incidence_i = tar_evaluator.cal_incidence(reduce=True, reuse=True)
        dfi = pd.DataFrame(
            {
                "t": tar_evaluator.t_,
                "burden": incidence_i,
                "scenario": np.array([scenario] * len(tar_evaluator.t_)),
            }
        )
        df.append(dfi)
        # 选择reference结果中最长时间的作为对照（即无疫苗接种的结果）
        if scenario == "100 Years":
            ref_evaluator = Evaluator.from_hdf(
                osp.join(trial_root, "ref_eval.h5")
            )
            incidence_i = ref_evaluator.cal_incidence(reduce=True, reuse=True)
            dfi = pd.DataFrame(
                {
                    "t": ref_evaluator.t_,
                    "burden": incidence_i,
                    "scenario": np.array(
                        ["No Vaccine"] * len(ref_evaluator.t_)
                    ),
                }
            )
            df.append(dfi)
    df = pd.concat(df, axis=0)

    fig, ax = plt.subplots(figsize=figsize)
    for i, (scenario, dfi) in enumerate(df.groupby("scenario")):
        ax.plot(
            dfi["t"].astype(int),
            dfi["burden"],
            label=scenario,
            color=colors[i],
        )
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Incidence" if burden_type == "incidence" else "Mortality")
    # ax.set_title("Incidence vs. Time")
    ax.legend(frameon=False, fancybox=False)
    if fig_name is not None:
        fig.savefig(fig_name)
    else:
        plt.show()


def figure_burden_age(
    incidence_age_df: pd.DataFrame,
    fig_name: str | None = None,
    colors: tuple[str, ...] = tuple(jama_palette.values()),
    figsize: tuple[float, float] = (12, 6),
    plot_kind: Literal["bar", "point"] = "point",
    burden_type: Literal["incidence", "mortality"] = "incidence",
):
    assert burden_type in ["incidence", "mortality"], "Unsupported burden_type"

    df_last = incidence_age_df.groupby(["scenario", "age"]).apply(
        lambda dfi: dfi.sort_values("t").iloc[-1]
    )

    fig, ax = plt.subplots(figsize=figsize)
    if plot_kind == "bar":
        sns.barplot(
            data=df_last,
            x="age",
            y="burden",
            hue="scenario",
            ax=ax,
            palette=colors,
        )
    elif plot_kind == "point":
        sns.pointplot(
            data=df_last,
            x="age",
            y="burden",
            hue="scenario",
            ax=ax,
            palette=colors,
        )
    else:
        raise ValueError(f"Unsupported plot_kind: {plot_kind}")
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Incidence" if burden_type == "incidence" else "Mortality")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    # ax.set_title("Incidence vs. Age Group")
    ax.legend(frameon=False, fancybox=False)
    ax.ticklabel_format(
        style="sci", axis="y", scilimits=(0, 0), useMathText=True
    )
    fig.tight_layout()

    if fig_name is not None:
        fig.savefig(fig_name)
    else:
        plt.show()


def figure_burden_age_time(
    incidence_age_df: pd.DataFrame,
    fig_name: str | None = None,
    colors: tuple[str, ...] = tuple(jama_palette.values()),
    figsize: tuple[float, float] = (12, 10),
    age_colname: str = "age",
    min_age_group: str = "[12, 13)",
    ncols: int = 5,
    burden_type: Literal["incidence", "mortality"] = "incidence",
):
    assert burden_type in ["incidence", "mortality"], "Unsupported burden_type"

    df_plot = incidence_age_df[
        incidence_age_df[age_colname] > min_age_group
    ].copy()
    df_plot[age_colname] = df_plot.age.cat.remove_unused_categories()

    agebins = df_plot[age_colname].cat.categories
    nrows = (len(agebins) + ncols - 1) // ncols
    fig, axs = plt.subplots(
        figsize=figsize,
        nrows=nrows,
        ncols=ncols,
        sharex=True,
        sharey=False,
        constrained_layout=True,
    )
    axs = axs.flatten()
    for i in range(len(agebins)):
        ax = axs[i]
        sns.lineplot(
            data=df_plot[df_plot[age_colname] == agebins[i]],
            x="t",
            y="burden",
            hue="scenario",
            palette=colors,
            ax=ax,
            legend=True,  # 设置总legend时借用
        )
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_title(agebins[i])
        ax.ticklabel_format(
            style="sci", axis="y", scilimits=(0, 0), useMathText=True
        )

    fig.supxlabel("Time (years)")
    fig.supylabel("Incidence" if burden_type == "incidence" else "Mortality")

    # 首先利用子图的legend来设置总的legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=len(df_plot["scenario"].unique()),
        frameon=False,
        fancybox=False,
    )
    # 然后移除所有子图的legend
    for ax in axs:
        ax.get_legend().remove()

    if fig_name is not None:
        fig.savefig(fig_name)
    else:
        plt.show()


def get_cost_df(
    results_dict: dict[str, dict[str, str]],
    disc_flag: str,
    exchange_rate: float = 6.8967,
) -> pd.DataFrame:
    df = []
    for scenario, res_dict_i in results_dict.items():
        trial_root = res_dict_i[disc_flag]
        tar_evaluator = Evaluator.from_hdf(osp.join(trial_root, "tar_eval.h5"))
        ref_evaluator = Evaluator.from_hdf(osp.join(trial_root, "ref_eval.h5"))

        tar_n_vacc = tar_evaluator.cal_number_vacc(reuse=False).sum(axis=1)
        tar_cost_vacc = tar_evaluator.cal_cost_vacc(reuse=False)
        tar_cost_cecx = tar_evaluator.cal_cost_cecx(reuse=False)
        ref_cost_cecx = ref_evaluator.cal_cost_cecx(reuse=False)
        diff_cost_cecx = ref_cost_cecx - tar_cost_cecx
        net_cost = tar_cost_vacc - diff_cost_cecx

        dfi = pd.DataFrame(
            {
                "t": tar_evaluator.t_,
                "vaccine_num": tar_n_vacc / 10000,
                "vaccine_cost": tar_cost_vacc / 10000 * exchange_rate,
                "cost_saved": diff_cost_cecx / 10000 * exchange_rate,
                "net_cost": net_cost / 10000 * exchange_rate,
                "scenario": np.array([scenario] * len(tar_evaluator.t_)),
            }
        )
        df.append(dfi)

    df = pd.concat(df, axis=0)
    df["scenario"] = pd.Categorical(
        df["scenario"],
        categories=list(results_dict.keys()),
        ordered=True,
    )
    return df


def table_cost(
    df_cost: pd.DataFrame,
    tab_name: str = "Table_2_cost.xlsx",
):
    df_cost_table = (
        df_cost.groupby("scenario")
        .apply(lambda df: df.sort_values("t").iloc[-1])
        .drop(columns=["scenario"])
    )
    df_cost_table.round(2).to_excel(tab_name, index=True)


def figure_cost_time(
    df_cost: pd.DataFrame,
    fig_name: str | None = None,
    colors: tuple[str, ...] = tuple(jama_palette.values()),
    figsize: tuple[float, float] = (10, 6),
    plot_kind: Literal["bar", "line"] = "line",
    ncols: int = 2,
):
    assert ncols in [1, 2, 4], "Supported ncols are 1, 2, and 4"

    fig, axs = plt.subplots(
        ncols=ncols,
        nrows=4 // ncols,
        sharex=True,
        figsize=figsize,
        constrained_layout=True,
    )
    x = df_cost["t"].values
    hue = df_cost["scenario"].values
    for ax, col_name, formal_name in zip(
        axs.flatten(),
        ["vaccine_num", "vaccine_cost", "cost_saved", "net_cost"],
        ["Vaccine Number", "Vaccine Cost", "Cost Saved", "Net Cost"],
    ):
        y = df_cost[col_name].values
        if plot_kind == "bar":
            sns.barplot(x=x, y=y, ax=ax, palette=colors, hue=hue)
        else:
            sns.lineplot(x=x, y=y, ax=ax, palette=colors, hue=hue)
        # ax.set_xlabel("Time (years)")
        ax.set_ylabel(formal_name)
        ax.ticklabel_format(
            style="sci", axis="y", scilimits=(0, 0), useMathText=True
        )

    fig.supxlabel("Time (years)")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=len(np.unique(hue)),
        frameon=False,
        fancybox=False,
    )

    for ax in axs.flatten():
        ax.legend_.remove()

    if fig_name is not None:
        fig.savefig(fig_name)
    else:
        plt.show()


def table_strategies_choiced(
    results_dict: dict[str, dict[str, str]],
    tab_name: str = "Table_1_strategies_choiced.xlsx",
    disc_flag: Literal["no_disc", "discount"] = "discount",
    incidence_threshold: float = 4e-5,
    exchange_rate: float = 6.8967,
):
    # 临时数据库路径，创建在每个训练结果路径下，用于缓存optuna的study对象
    storage_name = "sqlite:///HPV_tdm.db"
    ori_dir = os.getcwd()

    df = []
    for scenario, results_dict_i in results_dict.items():
        # 载入训练结果，训练结果是optuna的study对象, 得到最优trial的参数
        tar_root = results_dict_i[disc_flag]
        tar_root = os.path.abspath(tar_root)
        os.chdir(tar_root)
        with open("sampler.pkl", "rb") as f:
            sampler = pickle.load(f)
        study = optuna.create_study(
            study_name="HPV_tdm",
            storage=storage_name,
            sampler=sampler,
            directions=["minimize", "minimize"],
            load_if_exists=True,
        )
        best_icer, choice_trial = np.inf, None
        for trial in study.best_trials:
            if (
                trial.values[0] < best_icer
                and trial.values[1] <= incidence_threshold
            ):
                best_icer = trial.values[0]
                choice_trial = trial
        os.chdir(ori_dir)
        params = choice_trial.params

        # 得到经济学评价的结果
        tar_evaluator = Evaluator.from_hdf(osp.join(tar_root, "tar_eval.h5"))
        ref_evaluator = Evaluator.from_hdf(osp.join(tar_root, "ref_eval.h5"))
        model = tar_evaluator.model
        if "target_age_span" in params:
            start, end = params["target_age_span"]
            target_age = list(range(start, end + 1))
        elif "target_age" in params:
            target_age_wild = params["target_age"]
            if isinstance(target_age_wild, int):
                target_age = [target_age_wild]
            elif isinstance(target_age_wild, (tuple, list)):
                target_age = list(target_age_wild)
            else:
                raise ValueError
        target_age_str = [model.agebin_names[i] for i in target_age]
        if "coverage" in params:
            cover = [params["coverage"]]
        else:
            cover = [params["coverage%d" % i] for i in target_age]
        target_vacc = params["target_vacc"]
        target_vacc = {
            "dom2": "Domestic bivalent vaccine",
            "imp2": "Imported bivalent vaccines",
            "imp9": "Imported nine-valent vaccine",
        }[target_vacc]
        icurs = tar_evaluator.cal_icur(
            ref_evaluator, reuse=False, minor_reuse=False
        )

        # 计算经济学评价指标
        resi = {
            "Scenario": scenario,
            "Target age group": ",".join(target_age_str),
            "Vaccine valence type": target_vacc,
            "Coverage": ",".join([f"{coveri * 100:.2f}%" for coveri in cover]),
            "ICUR": icurs[-1] * exchange_rate,
        }
        df.append(resi)

    df = pd.DataFrame.from_records(df)
    df.round(2).to_excel(tab_name, index=True)


def get_icur_df(
    results_dict: dict[str, dict[str, str]],
    disc_flag: Literal["no_disc", "discount"] = "discount",
    exchange_rate: float = 6.8967,
) -> pd.DataFrame:
    df = []
    for scenario, results_dict_i in results_dict.items():
        tar_root = results_dict_i[disc_flag]
        tar_evaluator = Evaluator.from_hdf(osp.join(tar_root, "tar_eval.h5"))
        ref_evaluator = Evaluator.from_hdf(osp.join(tar_root, "ref_eval.h5"))

        tar_cost_vacc = tar_evaluator.cal_cost_vacc(reuse=False)
        tar_cost_cecx = tar_evaluator.cal_cost_cecx(reuse=False)
        diff_cost_cecx = (
            ref_evaluator.cal_cost_cecx(reuse=False) - tar_cost_cecx
        )
        net_cost = tar_cost_vacc - diff_cost_cecx
        icur = tar_evaluator.cal_icur(
            ref_evaluator, reuse=False, minor_reuse=True
        )
        avoid_cecx = tar_evaluator.cal_avoid_cecx(
            ref_evaluator, reuse=False, minor_reuse=True
        ).sum(axis=1)
        avoid_cecx_death = tar_evaluator.cal_avoid_cecxDeath(
            ref_evaluator, reuse=False, minor_reuse=True
        ).sum(axis=1)
        avoid_daly = tar_evaluator.cal_avoid_daly(
            ref_evaluator, reuse=False, minor_reuse=True
        )

        dfi = pd.DataFrame(
            {
                "t": tar_evaluator.t_,
                "avoid_cecx": avoid_cecx,
                "avoid_cecx_death": avoid_cecx_death,
                "avoid_daly": avoid_daly,
                "ic_per_cecx": np.divide(
                    net_cost,
                    avoid_cecx,
                    out=np.full_like(net_cost, np.inf),
                    where=avoid_cecx != 0,
                )
                * exchange_rate,
                "ic_per_cecx_death": np.divide(
                    net_cost,
                    avoid_cecx_death,
                    out=np.full_like(net_cost, np.inf),
                    where=avoid_cecx_death != 0,
                )
                * exchange_rate,
                "icur": icur * exchange_rate,
            }
        )
        dfi["scenario"] = scenario
        df.append(dfi)

    df = pd.concat(df)
    return df


def table_icur(
    df_icur: pd.DataFrame,
    tab_name: str = "Table_3_icur.xlsx",
):
    res = (
        df_icur.groupby("scenario")
        .apply(lambda df: df.sort_values("t").iloc[-1])
        .drop(columns=["t", "scenario"])
    )
    res.round(2).to_excel(tab_name, index=True)


def figure_icur_time(
    df_icur: pd.DataFrame,
    fig_name: str | None = None,
    colors: tuple[str, ...] = tuple(jama_palette.values()),
    figsize: tuple[float, float] = (10, 6),
    plot_kind: Literal["bar", "line"] = "line",
    linear_scale: bool = True,
    inset_axes: bool = False,
):
    if inset_axes:
        assert linear_scale, "Inset axes only support linear scale"

    # df_icur中起点是inf，需要去掉
    df_icur = df_icur.query("t > 0")
    df_icur.loc[:, ["avoid_cecx", "avoid_cecx_death"]] = (
        df_icur[["avoid_cecx", "avoid_cecx_death"]] / 10000
    )
    fig, axs = plt.subplots(
        ncols=3,
        nrows=2,
        sharex=False,
        figsize=figsize,
        constrained_layout=True,
    )
    axs = axs.flatten()
    for ax, col_name, formal_name in zip(
        axs,
        [
            "avoid_cecx",
            "avoid_cecx_death",
            "avoid_daly",
            "ic_per_cecx",
            "ic_per_cecx_death",
            "icur",
        ],
        [
            "Cases of cervical cancer prevented",
            "Cervical cancer deaths prevented",
            "Disablity adjusted life years saved",
            "Incremental cost per case of cervical cancer prevention",
            "Incremental cost per cervical cancer death avoided",
            "Incremental cost per disability-adjusted life year saved",
        ],
    ):
        if plot_kind == "bar":
            sns.barplot(
                data=df_icur,
                x="t",
                y=col_name,
                hue="scenario",
                ax=ax,
                palette=colors,
            )
        else:
            sns.lineplot(
                data=df_icur,
                x="t",
                y=col_name,
                hue="scenario",
                ax=ax,
                palette=colors,
            )
        ax.ticklabel_format(
            style="sci", axis="y", scilimits=(0, 0), useMathText=True
        )
        if not linear_scale and not col_name.startswith("avoid"):
            ax.set_yscale("symlog", linthresh=100)
        if linear_scale and inset_axes and not col_name.startswith("avoid"):
            x1_ori = {"ic_per_cecx": 18, "ic_per_cecx_death": 30, "icur": 20}[
                col_name
            ]
            x2_ori = {"ic_per_cecx": 50, "ic_per_cecx_death": 100, "icur": 50}[
                col_name
            ]
            df_icur_sub = df_icur.query(f"t >= {x1_ori} and t <= {x2_ori}")
            y1_ori = df_icur_sub[col_name].min()
            y2_ori = df_icur_sub[col_name].max()
            axin = ax.inset_axes(
                [0.2, 0.2, 0.7, 0.7],
                xlim=(x1_ori, x2_ori),
                ylim=(y1_ori, y2_ori),
                xticklabels=[],
                xticks=[],
                # yticklabels=[],
                # yticks=[],
            )
            sns.lineplot(
                data=df_icur.query(f"t >= {x1_ori}"),
                x="t",
                y=col_name,
                hue="scenario",
                ax=axin,
                palette=colors,
                legend=False,
            )
            axin.set_xlabel("")
            axin.set_ylabel("")
            axin.ticklabel_format(
                style="sci", axis="y", scilimits=(0, 0), useMathText=True
            )
            ax.indicate_inset_zoom(axin, edgecolor="black")
        ax.set_ylabel(formal_name)
        ax.set_xlabel("")
        # ax.set_title(formal_name)

    fig.supxlabel("Time (years)")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=len(df_icur["scenario"].unique()),
        frameon=False,
        fancybox=False,
    )

    for ax in axs:
        ax.legend_.remove()

    if fig_name is not None:
        fig.savefig(fig_name)
    else:
        plt.show()


def table_parameters_by_age_group(
    results_dict: dict[str, dict[str, str]],
    tab_name: str = "Table_s1_parameters_by_age_group.xlsx",
):
    root = results_dict["30 Years"]["discount"]
    tar_evaluator = Evaluator.from_hdf(osp.join(root, "tar_eval.h5"))
    model = tar_evaluator.model

    df = pd.DataFrame(
        {
            "Age group": model.agebin_names,
            "Number of sexual partners per year for women": model.kwargs[
                "omega_f"
            ],
            "Number of sexual partners per year for men": model.kwargs[
                "omega_m"
            ],
            "Mortality rate of local cancer": model.kwargs["dL"],
            "Mortality rate of region cancer": model.kwargs["dR"],
            "Mortality rate of distant cancer": model.kwargs["dD"],
            "Fertility rate": model.kwargs["fertilities"],
            "Death rate of women": model.kwargs["deathes_female"],
            "Death rate of men": model.kwargs["deathes_male"],
        }
    )

    df.to_excel(tab_name, index=False)


def insurance_cost(
    evaluator: Evaluator,
    vacc_insured: float = 1.0,  # 接种人数参保比例，这里所有的人都是未成年人
    vacc_reimburse: float = 1.0,  # 疫苗接种报销比例
    minor_insured: float = 0.869,  # 未成年人的参保比例，未成年人的参保类型全是城乡居民
    cecx_diag: float = 0.81,  # 宫颈癌的诊断率
    cecx_treatment: float = 0.9,  # 宫颈癌接受治疗率
    cecx_insured: float = (
        0.7569,
        0.2431,
    ),  # 成年人的参保比例，分别是城乡居民和城镇职工
    cecx_reimburse: tuple[float, float] = (
        0.479,
        0.6211,
    ),  # 两种保险的报销比例
    exchange_rate: float = 6.8967,  # 汇率
    discount_rate: float = 0.03,  # 贴现率
    n_minor: int = 10,  # 未成年人数
    unit: int = 1000000,  # 单位，这里是百万元
) -> pd.DataFrame:
    # 累计人数
    ncecxC = evaluator.reset_discount(0.0).cal_number_cecx(reuse=False)
    nvaccC = evaluator.reset_discount(0.0).cal_number_vacc(reuse=False)
    # 每年人数
    ncecx = ncecxC[1:] - ncecxC[:-1]
    nvacc = nvaccC[1:] - nvaccC[:-1]
    # 实际接受治疗的人数
    ncecx_treat = ncecx * cecx_diag * cecx_treatment
    # 贴现率
    disc_rate_arr = np.power(
        1 / (1 + discount_rate), np.arange(ncecx.shape[0])
    )
    # 花费（放在前面，因为不同年龄组的疫苗花费不同，这里希望直接使用cost_pvacc_vec）
    cost_cecx = ncecx_treat * evaluator.cost_pcecx * disc_rate_arr[:, None]
    cost_vacc = nvacc * evaluator.cost_pvacc * disc_rate_arr[:, None]
    # 成年人和未成年人
    cost_cecx_minor = cost_cecx[:, :n_minor].sum(axis=1)
    cost_cecx_adult = cost_cecx[:, n_minor:].sum(axis=1)
    # 已知，所有打疫苗的都是未成年人，直接使用参数中的参保率和报销比例即可
    cost_vacc = cost_vacc.sum(axis=1)
    # cost_vacc_minor = cost_vacc[:, :n_minor].sum(axis=1)
    # cost_vacc_adult = cost_vacc[:, n_minor:].sum(axis=1)

    # 计算疫苗的医保花费
    cost_vacc_insurance = cost_vacc * vacc_insured * vacc_reimburse
    # 计算宫颈癌的医保花费
    cost_cecx_minor_insurance = (
        cost_cecx_minor * minor_insured * cecx_reimburse[0]
    )  # 未成年人只有一部分人参保，而且参保类型都是城乡居民，使用城乡居民的报销比例
    cost_cecx_adult_insurance = 0
    for insure_rate, reimb_rate in zip(cecx_insured, cecx_reimburse):
        cost_cecx_adult_insurance += (
            cost_cecx_adult * insure_rate * reimb_rate
        )  # 成年人的参保类型有多种，先乘以参保比例，然后乘以报销比例

    # 计算总的医保花费
    cost_insurance = (
        cost_vacc_insurance
        + cost_cecx_minor_insurance
        + cost_cecx_adult_insurance
    )

    df_res = pd.DataFrame(
        {
            "n_vacc": nvacc.sum(axis=1),
            "n_cecx": ncecx.sum(axis=1),
            "n_treated": ncecx_treat.sum(axis=1),
            "cost_vacc": cost_vacc * exchange_rate,
            "cost_treated": cost_cecx.sum(axis=1) * exchange_rate,
            "cost_vacc_insurance": cost_vacc_insurance * exchange_rate,
            "cost_cecx_insured": (
                cost_cecx_minor_insurance + cost_cecx_adult_insurance
            )
            * exchange_rate,
            "cost_insurance": cost_insurance * exchange_rate,
        }
    )
    df_res.index = evaluator.t_[1:]
    df_res.index.name = "time"

    return df_res / unit  # 单位转换为百万元


def get_budget_df(
    results_dict: dict[str, dict[str, str]],
    vacc_reimburse: float = 1.0,
    gov_rate: float = 0.0,
    busi_rate: float = 0.0,
    person_rate: float = 0.0,
    eval_scenario: str = "40 Years",
    ref_scenario: str = "100 Years",
    unit: int = 1000000,
) -> pd.DataFrame:
    assert (
        vacc_reimburse + gov_rate + busi_rate + person_rate == 1.0
    ), "all rates should add up to 1.0"
    assert (
        eval_scenario in results_dict.keys()
    ), "eval_scenario not in results_dict"

    eval_tar_root = results_dict[eval_scenario]["discount"]
    eval_ref_root = results_dict[ref_scenario]["discount"]
    eval_tar = Evaluator.from_hdf(osp.join(eval_tar_root, "tar_eval.h5"))
    eval_ref = Evaluator.from_hdf(osp.join(eval_ref_root, "ref_eval.h5"))
    df_insurance_tar = insurance_cost(
        eval_tar, vacc_reimburse=vacc_reimburse, unit=unit
    )
    df_insurance_ref = insurance_cost(
        eval_ref, vacc_reimburse=vacc_reimburse, unit=unit
    )

    indice = df_insurance_tar.index.intersection(df_insurance_ref.index)
    df_insurance_tar = df_insurance_tar.loc[indice, :]
    df_insurance_ref = df_insurance_ref.loc[indice, :]
    df_insurance_diff = df_insurance_tar - df_insurance_ref

    vacc_cost_all = df_insurance_tar["cost_vacc"].values
    other_money = {}
    for name, ratei in zip(
        ["government", "commercial insurance", "personal payment"],
        [gov_rate, busi_rate, person_rate],
    ):
        if ratei is not None:
            value = vacc_cost_all * ratei
            other_money[name] = value
    other_money = pd.DataFrame(other_money, index=df_insurance_tar.index)

    columns1 = (
        ["with vaccination"] * df_insurance_tar.shape[1]
        + ["without vaccination"] * df_insurance_tar.shape[1]
        + ["difference"] * df_insurance_tar.shape[1]
        + other_money.columns.tolist()
    )
    columns2 = (
        df_insurance_tar.columns.tolist()
        + df_insurance_ref.columns.tolist()
        + df_insurance_diff.columns.tolist()
        + [""] * other_money.shape[1]
    )

    df_insurance_select = pd.concat(
        [df_insurance_tar, df_insurance_ref, df_insurance_diff, other_money],
        axis=1,
    )
    df_insurance_select.columns = pd.MultiIndex.from_arrays(
        [columns1, columns2]
    )

    df_sum = df_insurance_select.sum(axis=0)
    df_insurance_select = pd.concat(
        [df_insurance_select, df_sum.to_frame("sum").T], axis=0
    )

    return df_insurance_select


def figure_budget(
    budget_df: pd.DataFrame,
    fig_name: str = "Figure_11_budget_bar.png",
    colors: tuple[str, ...] = tuple(jama_palette.values()),
    figsize: tuple[float, float] = (10, 10),
    plot_kind: Literal["bar", "line"] = "line",
    begin_year: int = 2019,
    y_label: str = "Cost (millions yuan)",
    bar_width: float = 0.3,
):
    features = [
        "cost_cecx_insured",
        "cost_insurance",
    ]
    feature_names = [
        "Cost of cervical cancer insurance",
        "Cost of total insurance",
    ]
    panels = ["with vaccination", "without vaccination"]

    budget_df = budget_df.drop(index="sum")
    x = budget_df.index.values + begin_year - 1  # index starts from 1

    fig, axs = plt.subplots(
        ncols=1, nrows=2, sharex=True, figsize=figsize, constrained_layout=True
    )
    axs = axs.flatten()
    for ax, feature, feature_name in zip(axs, features, feature_names):
        for i, panel in enumerate(panels):
            y = budget_df[(panel, feature)].values
            if plot_kind == "bar":
                ax.bar(
                    x + bar_width * (2 * i - 1) * 0.5,
                    y,
                    bar_width,
                    label=panel,
                    color=colors[i],
                    edgecolor="black",
                )
            elif plot_kind == "line":
                ax.plot(x, y, "o-", label=panel, color=colors[i])
        ax.set_title(feature_name)
        ax.legend()

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=len(panels),
        frameon=False,
        fancybox=False,
    )

    for ax in axs:
        ax.legend_.remove()

    fig.supxlabel("Time (years)")
    fig.supylabel(y_label)

    if fig_name is not None:
        fig.savefig(fig_name)
    else:
        plt.show()


def figure_budget_increment_compare(
    budget_df_dict: dict[str, pd.DataFrame],
    fig_name: str | None = "Figure_12_budget_increment_compare.png",
    colors: tuple[str, ...] = tuple(nejm_palette.values()),
    figsize: tuple[float, float] = (8, 6),
    begin_year: int = 2019,
    y_label: str = "Cost (millions yuan)",
):
    fig, ax = plt.subplots(figsize=figsize)
    for i, (k, dfi) in enumerate(budget_df_dict.items()):
        dfi = dfi.drop(index="sum")
        x = dfi.index.values + begin_year - 1  # index starts from 1
        ax.plot(
            x,
            dfi[("difference", "cost_insurance")].values,
            ".-",
            color=colors[i],
            label=f"insurance cost increment ({k})",
        )
        ax.plot(
            x,
            dfi[("with vaccination", "cost_vacc_insurance")].values,
            "-",
            color=colors[i],
            label=f"vaccination cost ({k})",
        )
        ax.plot(
            x,
            dfi[("with vaccination", "cost_insurance")].values,
            "x-",
            color=colors[i],
            label=f"insurance cost ({k})",
        )

    # 设置网格
    ax.grid(True, which="both", axis="y")
    # 设置坐标轴，去掉上和右侧，将底部设置为y=0的位置
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_position(("data", begin_year - 1))
    ax.spines["bottom"].set_position("zero")

    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    # 设置y轴的间隔
    ax.yaxis.set_major_locator(MultipleLocator(200))

    # 自定义图例
    legend_elements = []
    for i, k in enumerate(budget_df_dict.keys()):
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color=colors[i],
                label=k,
            )
        )
    for k, m in zip(
        ["insurance cost increment", "vaccination cost", "insurance cost"],
        [".", ",", "x"],
    ):
        legend_elements.append(
            Line2D([0], [0], marker=m, color="black", label=k)
        )

    ax.legend(
        handles=legend_elements,
        loc="best",
        ncol=1,
        frameon=False,
        fancybox=False,
    )

    ax.set_ylabel(y_label)
    fig.supxlabel("Time (years)")
    fig.tight_layout()

    if fig_name is not None:
        fig.savefig(fig_name)
    else:
        plt.show()


def figure_budget_three_party(
    df_budget: pd.DataFrame,
    fig_name: str | None = "Figure_13_budget_three_party.png",
    colors: tuple[str, ...] = tuple(nejm_palette.values()),
    figsize: tuple[float, float] = (8, 4),
    begin_year: int = 2019,
    plot_kind: Literal["bar", "line"] = "line",
    bar_width: float = 0.3,
    y_unit_str: str = "millions yuan",
):
    df_budget = df_budget.drop(index="sum")

    fig, axs = plt.subplots(
        figsize=figsize, nrows=2, sharex=True, constrained_layout=True
    )

    x = df_budget.index.values + begin_year - 1  # index starts from 1
    for j, ax in enumerate(axs):
        for i, (col, label) in enumerate(
            zip(
                [
                    ("with vaccination", "cost_vacc_insurance"),
                    "government",
                    "personal payment",
                ],
                [
                    "medical insurance part",
                    "government part",
                    "personal part",
                ],
            )
        ):
            y = (
                df_budget[col].values
                if j == 0
                else df_budget[col].values.cumsum()
            )
            if plot_kind == "line":
                ax.plot(
                    x,
                    y,
                    ".-",
                    color=colors[i],
                    label=label,
                )
            elif plot_kind == "bar":
                ax.bar(
                    x + bar_width * ([-1, 0, 1][i]),
                    y,
                    bar_width,
                    label=label,
                    color=colors[i],
                    edgecolor="black",
                )
            ax.legend()

        ax.set_xlim(begin_year - 1, x.max() + 1)
        ax.set_ylabel(
            f"Cost per year ({y_unit_str})"
            if j == 0
            else f"Cumulative cost ({y_unit_str})"
        )
    fig.supxlabel("Time (years)")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=3,
        frameon=False,
        fancybox=False,
    )

    for ax in axs:
        ax.legend_.remove()

    if fig_name is not None:
        fig.savefig(fig_name)
    else:
        plt.show()


def get_price_demand_df(
    dta_fname: str,
):
    dta_fname = osp.abspath(dta_fname)
    df = pd.read_stata(dta_fname)
    df = df[["家庭月平均收入元", "国产二价HPV疫苗愿意承受的最高价格是全程"]]
    df.rename(
        columns={
            "家庭月平均收入元": "income",
            "国产二价HPV疫苗愿意承受的最高价格是全程": "WTP",
        },
        inplace=True,
    )
    df["status"] = 1

    # 异常值处理
    thre = np.quantile(df["income"], 0.95)
    df["income_wo_outlier"] = np.minimum(df["income"], thre)

    return df.query("WTP > 0")


def plot_price_demand_function(
    df_price_demand: pd.DataFrame,
    income: float | list[float] = 30733 / 12 * 2,  # 2019年人均可支配收入
    fig_name: str | None = "Figure_14_price_demand_function.png",
    colors: tuple[str, ...] = tuple(reversed(nejm_palette.values())),
    figsize: tuple[float, float] = (6, 4),
    linewidth: float = 2,
):
    if isinstance(income, (int, float)):
        income = np.array([income])

    # df_price_demand = df_price_demand.copy()
    # df_price_demand["WTP"] = df_price_demand["WTP"] + 1e-7
    fitter = WeibullAFTFitter()
    fitter.fit(
        df_price_demand[["income_wo_outlier", "status", "WTP"]],
        duration_col="WTP",
        event_col="status",
    )
    preds = []
    for p in np.arange(0.01, 1.0, 0.01):
        pred = fitter.predict_percentile(
            pd.DataFrame({"income_wo_outlier": income}), p=p
        )
        preds.append(pred)
    preds = pd.concat(preds, axis=1)
    preds.columns = np.arange(0.01, 1.0, 0.01)
    preds.index = income
    preds = preds.T

    fig, ax = plt.subplots(figsize=figsize)
    for i, income_i in enumerate(preds.columns):
        ax.plot(
            preds[income_i].values,
            preds.index.values,
            color=colors[i],
            label=f"income={income_i:.0f}",
            linewidth=linewidth,
        )

    ax.grid(True, which="both", axis="both")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        loc="best",
        frameon=False,
        fancybox=False,
    )

    ax.set_xlabel("vaccine price (yuan)")
    ax.set_ylabel("vaccine demand")

    if fig_name is not None:
        fig.savefig(fig_name)
    else:
        plt.show()


def main():
    d_results = {
        "30 Years": {
            "no_disc": "../results/backup3/opt_res_m30_nodisc/",
            "discount": "../results/backup4/opt_res_m30_disc/",
        },
        "40 Years": {
            "no_disc": "../results/backup3/opt_res_m40_nodisc/",
            "discount": "../results/backup4/opt_res_m40_disc/",
        },
        "50 Years": {
            "no_disc": "../results/backup3/opt_res_m50_nodisc/",
            "discount": "../results/backup4/opt_res_m50_disc/",
        },
        "60 Years": {
            "no_disc": "../results/backup3/opt_res_m60_nodisc/",
            "discount": "../results/backup4/opt_res_m60_disc/",
        },
        "80 Years": {
            "no_disc": "../results/backup3/opt_res_m80_nodisc/",
            "discount": "../results/backup4/opt_res_m80_disc/",
        },
        "100 Years": {
            "no_disc": "../results/backup3/opt_res_m100_nodisc/",
            "discount": "../results/backup4/opt_res_m100_disc/",
        },
    }

    # studies = get_studies(d_results)
    # figure_history(studies, figname="Figure_1_history.png")
    # figure_pareto(studies, figname="Figure_2_pareto.png")

    # NOTE: 以下代码仅用于转换pkl格式的evaluation对象为hdf5格式
    # convert_pkl_to_hdf5(d_results)

    # figure_burden_total(
    #     d_results,
    #     fig_name="Figure_3_incidence_total.pdf",
    #     burden_type="incidence",
    #     colors=list(jama_palette.values()),
    # )
    # incidence_age_df = get_burden_df(
    #     d_results, disc_flag="discount", burden_type="incidence"
    # )
    # figure_burden_age(
    #     incidence_age_df,
    #     fig_name="Figure_4_incidence_age.pdf",
    #     burden_type="incidence",
    #     colors=list(jama_palette.values()),
    # )
    # figure_burden_age_time(
    #     incidence_age_df,
    #     fig_name="Figure_5_incidence_age_time.pdf",
    #     age_colname="age",
    #     min_age_group="[12, 13)",
    #     ncols=5,
    #     burden_type="incidence",
    #     colors=list(jama_palette.values()),
    # )

    # figure_burden_total(
    #     d_results,
    #     fig_name="Figure_6_mortality_total.pdf",
    #     burden_type="mortality",
    #     colors=list(jama_palette.values()),
    # )
    # mortality_age_df = get_burden_df(
    #     d_results, disc_flag="discount", burden_type="mortality"
    # )
    # figure_burden_age(
    #     mortality_age_df,
    #     fig_name="Figure_7_mortality_age.pdf",
    #     burden_type="mortality",
    #     colors=list(jama_palette.values()),
    # )
    # figure_burden_age_time(
    #     mortality_age_df,
    #     fig_name="Figure_8_mortality_age_time.pdf",
    #     age_colname="age",
    #     min_age_group="[12, 13)",
    #     ncols=5,
    #     burden_type="mortality",
    #     colors=list(jama_palette.values()),
    # )

    # table_strategies_choiced(
    #     d_results, tab_name="Table_1_strategies_choiced.xlsx"
    # )

    # df_cost = get_cost_df(d_results, disc_flag="discount")
    # table_cost(df_cost, tab_name="Table_2_cost.xlsx")
    # figure_cost_time(
    #     df_cost,
    #     fig_name="Figure_9_cost_time.pdf",
    #     colors=list(jama_palette.values()),
    # )

    df_icur = get_icur_df(d_results, disc_flag="discount")
    # table_icur(df_icur, tab_name="Table_3_icur.xlsx")
    figure_icur_time(
        df_icur,
        fig_name="Figure_10_icur_time.pdf",
        linear_scale=True,
        inset_axes=True,
        colors=list(jama_palette.values()),
    )

    # table_parameters_by_age_group(
    #     d_results, tab_name="Table_s1_parameters_by_age_group.xlsx"
    # )

    # df_budget = get_budget_df(d_results, vacc_reimburse=1.0, unit=1000000)
    # figure_budget(
    #     df_budget,
    #     plot_kind="bar",
    #     fig_name="Figure_11_budget_bar.pdf",
    #     y_label="Cost (millions yuan)",
    #     colors=list(jama_palette.values()),
    #     figsize=(8, 8),
    #     bar_width=0.35
    # )

    # df_budget_3937 = get_budget_df(
    #     d_results,
    #     vacc_reimburse=0.3937,
    #     person_rate=0.6063,
    #     unit=1000000,
    # )
    # df_budget_7428 = get_budget_df(
    #     d_results,
    #     vacc_reimburse=0.7428,
    #     person_rate=0.2572,
    #     unit=1000000,
    # )
    # df_budget_three = get_budget_df(
    #     d_results,
    #     vacc_reimburse=0.2791,
    #     gov_rate=0.2619,
    #     person_rate=0.4590,
    #     unit=1000000,
    # )
    # figure_budget_increment_compare(
    #     {
    #         "personal payment 60.63%": df_budget_3937,
    #         "personal payment 25.72%": df_budget_7428,
    #         (
    #             "personal payment 45.90%, " "government payment 26.19%"
    #         ): df_budget_three,
    #     },
    #     fig_name="Figure_12_budget_increment_compare.pdf",
    #     colors=list(jama_palette.values()),
    #     figsize=(8, 6),
    # )
    # figure_budget_three_party(
    #     df_budget_three,
    #     fig_name="Figure_13_budget_three_party.pdf",
    #     figsize=(10, 6),
    #     plot_kind="bar",
    #     colors=list(jama_palette.values())
    # )

    # df_price_demand = get_price_demand_df("../data/price_demand_data.dta")
    # incomes = np.quantile(
    #     df_price_demand["income_wo_outlier"],
    #     [0, 0.07, 0.15, 0.25, 0.5, 0.75, 1.0],
    # )
    # plot_price_demand_function(
    #     df_price_demand,
    #     income=incomes,
    #     fig_name="Figure_14_price_demand_function.pdf",
    #     linewidth=1.5,
    #     colors=list(jama_palette.values())
    # )


if __name__ == "__main__":
    main()
