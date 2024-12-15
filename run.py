import os
import logging
import pickle

import numpy as np
import hydra
from omegaconf import DictConfig

from src.model import AgeGenderHPVModel2
from src.evaluation import Evaluator, Ploter


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
    logging.info("[main] start evaluation ...")
    evaluator = Evaluator(
        model, cfg.get("init", None),
        t_span=cfg.t_span, n_eval=cfg.n_eval,
        cost_per_cecx=7537, cost_per_vacc=0.,
        discount_rate=cfg.disc_rate
    )
    evaluator.cal_incidence(show=True)
    evaluator.cal_mortality(show=True)

    ploter = Ploter(target=evaluator)
    ploter.plot_incidence()
    ploter.plot_mortality()
    ploter.plot_cost()
    # ploter.plot_daly()

    logging.info("[main] saving results ...")
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("evaluator.pkl", "wb") as f:
        pickle.dump(evaluator, f)
    if cfg.save_last:
        last = evaluator.y_[-1].flatten()
        np.save("last.npy", last)

    # logging.info("[main] plot results ...")
    # fgs = model.plot(results)
    # for key, fg in fgs.items():
    #     fg.savefig("plot_%s.png" % key)

if __name__ == "__main__":
    main()
