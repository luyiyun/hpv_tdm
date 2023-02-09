import sys
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
from omegaconf import DictConfig, OmegaConf
from optuna.visualization.matplotlib import plot_pareto_front

from src._life_table import life_table
from src import evaluation as E
from src.plot import plot_cost, plot_DALY, plot_incidence


warnings.simplefilter("ignore")
logging.basicConfig(level=logging.INFO,
                    format="[%(levelname)s][%(asctime)s]%(message)s")


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
    ref_res = {"t": ref_t, "y": ref_y, "ycum": ref_ycum, "model": ref_model}
    ref_inci = E.incidence(ref_res)
    ref_cu = E.cost_utility(
        ref_res,
        life_table=ref_ltable,
        cost_per_vacc=0,  # reference没有疫苗接种
        **OmegaConf.to_object(cfg.cu_kwargs),
    )
    reference = {
        "model": ref_model, "t": ref_t, "y": ref_y, "ycum": ref_ycum,
        "inci": ref_inci, "cu": ref_cu, "ltable": ref_ltable
    }
    if not osp.exists("reference.pkl"):
        with open("reference.pkl", "wb") as f:
            pickle.dump(reference, f)
    if ref_t.shape[0] != cfg.n_eval:
        raise ValueError("The n_trials is not equal to reference's.")

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
    args = {
        "reference": reference,
        "cu_kwargs": OmegaConf.to_object(cfg.cu_kwargs),
        "tdm_kwargs": OmegaConf.to_object(cfg.tdm_kwargs),
    }
    if cfg.search_strategy == "multi":
        args["n_vacc_ages"] = cfg.n_vacc_ages
    objective = {
        "one": E.OneAgeObjectiveFunction,
        "multi": E.MultiAgesObjectiveFunction,
        "conti": E.ContiAgesObjectiveFunction,
        "contiOneCover": E.ContiAgesOneCoverObjectiveFunction,
    }[cfg.search_strategy](**args)
    study.optimize(objective, n_trials=cfg.n_trials)
    # 单独保存
    with open("sampler.pkl", "wb") as fout:
        pickle.dump(study.sampler, fout)

    # 多目标优化结果绘制（帕累托图）
    logging.info("Pareto front:")
    # 选择出那个incidence < 4e-5 并且有最低icer值的trial
    best_value, best_trial = np.inf, None
    trials = sorted(study.best_trials, key=lambda t: t.values)
    for trial in trials:
        logging.info("  Trial#{}".format(trial.number))
        logging.info("    Values: Values={}".format(trial.values))
        logging.info("    Params: {}".format(trial.params))

        i_icer, i_inci = trial.values
        if i_inci < 4e-5 and i_icer < best_value:
            best_trial = trial
            best_value = i_icer
    plot_pareto_front(study)
    plt.savefig("./pareto_front.png")

    if best_trial is None:
        logging.info("There is not trial which incidence < 4e-5!")
        return

    logging.info(
        "Best icer: %.6f, incidence: %.6f, Best Params: " %
        tuple(best_trial.values)
    )
    objective.show_params(best_trial.params)

    logging.info("plotting best results...")
    new_obj_func = {
        "one": E.OneAgeObjectiveFunction,
        "multi": E.MultiAgesObjectiveFunction,
        "conti": E.ContiAgesObjectiveFunction,
        "contiOneCover": E.ContiAgesOneCoverObjectiveFunction,
    }[cfg.search_strategy](**args)
    tar_res, tar_inci, tar_cu, ic = \
        new_obj_func.call_from_parameters(best_trial.params)

    # 1.先绘制该最优策略模型的结果
    logging.info("[main] plot results ...")
    fgs = tar_res["model"].plot(tar_res)
    for key, fg in fgs.items():
        fg.savefig("plot_%s.png" % key)

    # 2. 绘制一下策略前后的incidence、cu、cost、DALY的变化趋势
    tar_df = pd.DataFrame(tar_cu)
    tar_df["t"] = tar_res["t"]
    tar_df["incidence"] = tar_inci

    ref_df = pd.DataFrame(ref_cu)
    ref_df["t"] = ref_t
    ref_df["incidence"] = ref_inci
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

    # 3. 绘制一下策略前后icer、避免患病、避免死亡的变化趋势
    arr_avoid_inci_all = E.avoid_incidence(tar_res, ref_res)
    arr_avoid_inci = arr_avoid_inci_all.sum(axis=-1)
    arr_avoid_death_all = E.avoid_death(tar_res, ref_res)
    arr_avoid_death = arr_avoid_death_all.sum(axis=(1, 2))

    df_icer = pd.DataFrame({
        "t": tar_res["t"], "ICER": ic,
        "AvoidIncidence": arr_avoid_inci,
        "AvoidDeath": arr_avoid_death
    })
    fg = sns.relplot(data=df_icer,
                     x="t", y="ICER", aspect=2, kind="line")
    # fg.set(yscale="log")
    fg.savefig("ICER.png")

    fg = sns.relplot(data=df_icer,
                     x="t", y="AvoidIncidence", aspect=2, kind="line")
    fg.set(yscale="log")
    fg.savefig("AvoidIncidence.png")

    fg = sns.relplot(data=df_icer,
                     x="t", y="AvoidDeath", aspect=2, kind="line")
    fg.set(yscale="log")
    fg.savefig("AvoidDeath.png")

    # 4. 保存结果
    with open("model.pkl", "wb") as f:
        pickle.dump(tar_res["model"], f)
    for k, v in tar_res.items():
        if k == "model":
            continue
        np.save("%s.npy" % k, v)
    np.save("inci", tar_inci)
    for k, v in tar_cu.items():
        np.save("cu_%s.npy" % k, v)
    np.save("icer.npy", ic)
    np.save("avoid_inci.npy", arr_avoid_inci_all)
    np.save("avoid_death.npy", arr_avoid_death_all)


if __name__ == "__main__":
    main()
