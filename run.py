import os
import logging
import pickle

import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig

from src.model import AgeGenderHPVModel2
from src import evaluation as E
from src.plot import plot_incidence, plot_death


logging.basicConfig(level=logging.INFO,
                    format="[%(levelname)s][%(asctime)s]%(message)s")


@hydra.main(config_path="conf", config_name="run", version_base="1.3")
def main(cfg: DictConfig):

    logging.info("[main] CWD is %s" % os.getcwd())

    logging.info("[main] Model init ... ")
    model = AgeGenderHPVModel2(
        cal_cumulate=True,
        total0_f=cfg.total0_f,
        total0_m=cfg.total0_m,
        vacc_prefer=cfg.vacc_prefer,
        # psi=np.array([0]*7 + [0.9] + [0] * 18),
        # tau=0.691,
        # psi=np.array([0] + [1.0] * 10 + [0] * 15),
        # tau=0.921,
    )
    #
    logging.info("[main] start prediction ...")
    # 我国人群HPV率是13.1-18.8%
    cfg_init = cfg.get("init", None)
    if cfg_init is None:
        init = model.get_init([0.85, 0.15]+[0]*6+[0.85, 0.15, 0, 0])
    else:
        init = np.load(os.path.join(cfg_init, "y.npy"))[-1].flatten()
        if model.cal_cumulate:
            init = np.r_[init, np.zeros(model.nages * 10)]
    results = model.predict(
        init=init, t_span=(cfg.t_span[0], cfg.t_span[1]),
        t_eval=np.linspace(cfg.t_span[0], cfg.t_span[1], cfg.n_eval),
        backend="solve_ivp"
    )

    logging.info("[main] incidence ...")
    inci = E.incidence(results)
    E.show_incidence(inci, results["t"])
    df_inci = pd.DataFrame({"t": results["t"], "incidence": inci})
    plot_incidence(df_inci)

    logging.info("[main] death ...")
    dea = E.death(results)
    E.show_death(dea, results["t"])
    df_dea = pd.DataFrame({"t": results["t"], "death": dea})
    plot_death(df_dea)

    logging.info("[main] saving results ...")
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    np.save("init.npy", init)
    for k, v in results.items():
        np.save("%s.npy" % k, v)

    logging.info("[main] plot results ...")
    fgs = model.plot(results)
    for key, fg in fgs.items():
        fg.savefig("plot_%s.png" % key)

if __name__ == "__main__":
    main()
