import os
import logging
import pickle

import numpy as np
import hydra
from omegaconf import DictConfig

from src import AgeGenderHPVModel2


logging.basicConfig(level=logging.INFO,
                    format="[%(levelname)s][%(asctime)s]%(message)s")


@hydra.main(config_path="conf", config_name="main", version_base="1.3")
def main(cfg: DictConfig):

    logging.info("[main] CWD is %s" % os.getcwd())

    logging.info("[main] Model init ... ")
    model = AgeGenderHPVModel2(
        cal_cumulate=True,
        # psi=np.array([0] * 5 + [1.0, 1.0, 1.0] + [0]*18),  # 更早接种疫苗
        # tau=1.0,
        # partner_interval=(15, 60)
    )
    #
    logging.info("[main] start prediction ...")
    # 我国人群HPV率是13.1-18.8%
    init = model.get_init([0.85, 0.15]+[0]*6+[0.85, 0.15, 0, 0])
    t, y = model.predict(init=init, t_span=(0, 100),
                         t_eval=np.arange(100), backend="solve_ivp")
    if model.cal_cumulate:
        y, ycum = y

    logging.info("[main] saving results ...")
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    np.save("init.npy", init)
    np.save("t.npy", t)
    np.save("y.npy", y)
    if model.cal_cumulate:
        np.save("ycum.npy", ycum)

    logging.info("[main] plot results ...")
    fgs = model.plot(t, y)
    for key, fg in fgs.items():
        fg.savefig("plot_%s.png" % key)
    if model.cal_cumulate:
        fgs = model.plot_cumulative(t, ycum)
        for key, fg in fgs.items():
            fg.savefig("plot_%s_cum.png" % key)

if __name__ == "__main__":
    main()
