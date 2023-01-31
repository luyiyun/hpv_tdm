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
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from optuna.visualization.matplotlib import plot_pareto_front

from src import AgeGenderHPVModel2, life_table
from src.evaluation import cal_incidence, cost_utility_analysis, cal_icer
from eval import plot_incidence, plot_DALY, plot_cost


# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore", message=warn_msg,
#                             category=RuntimeWarning)
warnings.simplefilter("ignore")
logging.basicConfig(level=logging.INFO,
                    format="[%(levelname)s][%(asctime)s]%(message)s")


class ObjectiveFunction:

    def __init__(
        self, ltable, ref_cost_utilities, n_ages_vacc=1,
        return_plot_res=False, constraint_weight=1.,
        t_span=(0, 100), n_eval=100
    ) -> None:
        self.ltable = ltable
        self.ref_cu = ref_cost_utilities
        self.nages_vacc = n_ages_vacc
        self.return_plot_res = return_plot_res
        self.constraint_weight = constraint_weight
        self.t_span = t_span
        self.n_eval = n_eval
        self.t_eval = np.linspace(t_span[0], t_span[1], n_eval)

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
        # return np.median(icer[-10:]), np.median(inci[-1:])
        return icer[-1], inci[-1]

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
                                   psi=psi, tau=tau, verbose=False,
                                   total0_f=500000, total0_m=500000)

        # 得到初始值，进行运算
        init = model.get_init([0.85, 0.15]+[0]*6+[0.85, 0.15, 0, 0])
        t, (y, ycum) = model.predict(init=init, t_span=self.t_span,
                                     t_eval=self.t_eval, backend="odeint",
                                     verbose=False)
        cost_utilities = cost_utility_analysis(ycum, self.ltable,
                                               cost_per_vacc=cost_per_vacc,
                                               cost_per_cecx=7547)

        # 计算指标
        icer = cal_icer(cost_utilities, self.ref_cu)
        incidences = cal_incidence(y, ycum, model, verbose=False)
        return model, t, y, ycum, incidences, cost_utilities, icer


@hydra.main(config_path="conf", config_name="opt", version_base="1.3")
def main(cfg: DictConfig):
    # 设置随机种子数
    np.random.seed(cfg.seed)

    # 首先得到reference的评价结果
    ref_path = osp.join(cfg.res_root, cfg.reference)
    with open(osp.join(ref_path, "model.pkl"), "rb") as f:
        ref_model = pickle.load(f)
    ref_t = np.load(osp.join(ref_path, "t.npy"))
    ref_y = np.load(osp.join(ref_path, "y.npy"))
    ref_ycum = np.load(osp.join(ref_path, "ycum.npy"))
    ref_ltable = life_table(ref_model.deathes_female, ref_model.agebins)
    ref_incidences = cal_incidence(ref_y, ref_ycum, ref_model, verbose=False)
    ref_cu = cost_utility_analysis(ref_ycum, ref_ltable, cost_per_cecx=7547)

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
        sampler = optuna.samplers.TPESampler()
        # sampler = optuna.samplers.NSGAIISampler()
    study = optuna.create_study(study_name=study_name,
                                storage=storage_name,
                                sampler=sampler,
                                directions=["minimize", "minimize"],
                                load_if_exists=True)
    objective = ObjectiveFunction(
        ref_ltable, ref_cu,
        constraint_weight=cfg.constraint_weight,
        t_span=(cfg.t_span[0], cfg.t_span[1]), n_eval=cfg.n_eval
    )
    study.optimize(objective, n_trials=cfg.n_trials)
    # 单独保存
    with open("sampler.pkl", "wb") as fout:
        pickle.dump(study.sampler, fout)

    logging.info("Pareto front:")
    # 选择出那个incidence < 4e-5 并且有最低icer值的trial
    best_value, best_trial = np.inf, None
    trials = sorted(study.best_trials, key=lambda t: t.values)
    for trial in trials:
        print("  Trial#{}".format(trial.number))
        print("    Values: Values={}".format(trial.values))
        print("    Params: {}".format(trial.params))

        i_icer, i_inci = trial.values
        if i_inci < 4e-5 and i_icer < best_value:
            best_trial = trial
            best_value = i_icer
    plot_pareto_front(study)
    plt.savefig("./pareto_front.png")

    if best_trial is None:
        logging.info("There is not trial which incidence < 4e-5!")
        return

    logging.info("Best icer: %.6f, incidence: %.6f, Best Params: " %
                 tuple(best_trial.values))
    for k, v in best_trial.params.items():
        if isinstance(v, float):
            logging.info("  %s: %.6f" % (k, v))
        elif isinstance(v, int):
            logging.info("  %s: %d" % (k, v))
        else:
            logging.info("  %s: %s" % (k, v))

    logging.info("plotting best results...")
    new_obj_func = ObjectiveFunction(
        ref_ltable, ref_cu, True,
        t_span=(cfg.t_span[0], cfg.t_span[1]), n_eval=cfg.n_eval
    )
    tar_model, tar_t, tar_y, tar_ycum, tar_inci, tar_cu, icer = \
        new_obj_func.calculate_by_parameters(**best_trial.params)
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
    fg = sns.relplot(data=df_icer,
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
