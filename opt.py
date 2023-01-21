import sys
# from copy import deepcopy
import os.path as osp
import logging
import pickle
import warnings

import numpy as np
import pandas as pd
import hydra
import optuna
import seaborn as sns
from omegaconf import DictConfig
# from optuna.visualization.matplotlib import plot_optimization_history

from src import AgeGenderHPVModel2, life_table
from src.evaluation import cal_incidence, cost_utility_analysis, cal_icer
from eval import plot_cost, plot_DALY, plot_incidence


# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore", message=warn_msg,
#                             category=RuntimeWarning)
warnings.simplefilter("ignore")
logging.basicConfig(level=logging.INFO,
                    format="[%(levelname)s][%(asctime)s]%(message)s")


class ObjectiveFunction:

    def __init__(
        self, ltable, ref_cost_utilities, return_plot_res=False,
        constraint_weight=1.
    ) -> None:
        self.ltable = ltable
        self.ref_cu = ref_cost_utilities
        self.return_plot_res = return_plot_res
        self.constraint_weight = constraint_weight

    def __call__(self, trial: optuna.Trial) -> float:
        # 得到参数
        target_age = trial.suggest_categorical("target_age", tuple(range(13)))
        target_vacc = trial.suggest_categorical(
            "target_vacc", ["dom2", "imp2", "imp9"]
        )
        coverage  = trial.suggest_float("coverage", low=0.1, high=1)

        res_all = self.calculate_by_parameters(target_age,
                                               target_vacc,
                                               coverage)[-3:]
        if self.return_plot_res:
            return res_all

        inci, _, icer = res_all[-3:]
        trial.set_user_attr(
            "c0", (np.median(inci[-10:]) - 4e-5) * self.constraint_weight
        )

        return np.median(icer[-10:])

    def calculate_by_parameters(self, target_age, target_vacc, coverage):
        # 根据参数进行设置
        psi = np.zeros(26)
        psi[target_age] = coverage
        tau = {"dom2": 0.691, "imp2": 0.691, "imp9": 0.921}[target_vacc]
        cost_per_vacc = {"dom2": 153.2,
                         "imp2": 262.38,
                         "imp9": 574.71}[target_vacc]

        # 构建模型
        model = AgeGenderHPVModel2(cal_cumulate=True,
                                   psi=psi, tau=tau, verbose=False)

        # 得到初始值，进行运算
        init = model.get_init([0.85, 0.15]+[0]*6+[0.85, 0.15, 0, 0])
        t, (y, ycum) = model.predict(init=init, t_span=(0, 100),
                                     t_eval=np.arange(100), backend="solve_ivp",
                                     verbose=False)
        cost_utilities = cost_utility_analysis(ycum, self.ltable,
                                               cost_per_vacc=cost_per_vacc)

        # 计算指标
        icer = cal_icer(cost_utilities, self.ref_cu)
        incidences = cal_incidence(y, ycum, model, verbose=False)
        return model, t, y, ycum, incidences, cost_utilities, icer


def constraints(trial: optuna.Trial):
    return (trial.user_attrs["c0"],)


@hydra.main(config_path="conf", config_name="opt", version_base="1.3")
def main(cfg: DictConfig):

    # 首先得到reference的评价结果
    ref_path = osp.join(cfg.res_root, cfg.reference)
    with open(osp.join(ref_path, "model.pkl"), "rb") as f:
        ref_model = pickle.load(f)
    ref_t = np.load(osp.join(ref_path, "t.npy"))
    ref_y = np.load(osp.join(ref_path, "y.npy"))
    ref_ycum = np.load(osp.join(ref_path, "ycum.npy"))
    ref_ltable = life_table(ref_model.deathes_female, ref_model.agebins)
    ref_incidences = cal_incidence(ref_y, ref_ycum, ref_model, verbose=False)
    ref_cu = cost_utility_analysis(ref_ycum, ref_ltable)

    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(
        logging.StreamHandler(sys.stdout))
    study_name = "HPV_tdm"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    if osp.exists("sampler.pkl"):
        logging.info("using existed sampler ...")
        with open("sampler.pkl", "rb") as f:
            sampler = pickle.load(f)
    else:
        logging.info("create new sampler ...")
        sampler = optuna.samplers.TPESampler(constraints_func=constraints)
    study = optuna.create_study(study_name=study_name,
                                storage=storage_name,
                                sampler=sampler,
                                load_if_exists=True)
    objective = ObjectiveFunction(ref_ltable, ref_cu,
                                  constraint_weight=cfg.constraint_weight)
    study.optimize(objective, n_trials=cfg.n_trials)
    # 单独保存
    with open("sampler.pkl", "wb") as fout:
        pickle.dump(study.sampler, fout)

    logging.info("Best icer: %.6f, Best Params: " % study.best_value)
    for k, v in study.best_params.items():
        if isinstance(v, float):
            logging.info("  %s: %.6f" % (k, v))
        elif isinstance(v, int):
            logging.info("  %s: %d" % (k, v))
        else:
            logging.info("  %s: %s" % (k, v))
    logging.info(
        "constraints: incidences = %.6f" %
        (
            study.best_trial.user_attrs["c0"] / objective.constraint_weight
            + 4e-5
        )
    )

    logging.info("plotting best results...")
    new_obj_func = ObjectiveFunction(ref_ltable, ref_cu, True)
    tar_model, tar_t, tar_y, tar_ycum, tar_inci, tar_cu, icer = \
        new_obj_func.calculate_by_parameters(**study.best_params)
    tar_df = pd.DataFrame(tar_cu)
    tar_df["t"] = tar_t
    tar_df["incidence"] = tar_inci

    ref_df = pd.DataFrame(ref_cu)
    ref_df["t"] = ref_t
    ref_df["incidence"] = ref_incidences
    ref_df = ref_df.set_index("t", drop=False)
    ref_df = ref_df.loc[tar_df["t"], :]  # 保证tar和ref有相同的t

    plot_incidence(tar_df[["t", "incidence"]], ref_df[["t", "incidence"]])
    plot_cost(
        tar_df[["t", "cost_vacc", "cost_cecx", "cost_all"]],
        ref_df[["t", "cost_vacc", "cost_cecx", "cost_all"]],
    )
    plot_DALY(
        tar_df[["t", "DALY_nodeath", "DALY_death"]],
        ref_df[["t", "DALY_nodeath", "DALY_death"]],
    )

    icer = cal_icer(tar_df, ref_df)
    df_icer = pd.DataFrame({"t": tar_t, "ICER": icer})
    # < 10年的部分数值太大，无法看到最终ICER值大约是多少
    fg = sns.relplot(data=df_icer[df_icer["t"] > 10],
                     x="t", y="ICER", aspect=2, kind="line")
    # fg.set(yscale="log")
    fg.savefig("./ICER_vs_%s.png" % cfg.reference.replace("/", "-"))

    # 绘制一下模型的进展图
    fgs = tar_model.plot(tar_t, tar_y)
    for key, fg in fgs.items():
        fg.savefig("plot_%s.png" % key)
    if tar_model.cal_cumulate:
        fgs = tar_model.plot_cumulative(tar_t, tar_ycum)
        for key, fg in fgs.items():
            fg.savefig("plot_%s_cum.png" % key)


if __name__ == "__main__":
    main()
