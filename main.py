import logging
import pickle

import numpy as np
import hydra
from omegaconf import DictConfig

# from src import AgeGenderModel
# from src import AgeGenderHPVModel1
from src import AgeGenderHPVModel2


logging.basicConfig(level=logging.INFO,
                    format="[%(levelname)s][%(asctime)s]%(message)s")


@hydra.main(config_path="conf", config_name="main", version_base="1.3")
def main(cfg: DictConfig):
    
    logging.info("[main] Model init ... ")
    model = AgeGenderHPVModel2(
        psi=np.array([0] * 5 + [1.0] + [0]*20),  # 更早接种疫苗
        tau=0.921,
    )
    #
    logging.info("[main] start prediction ...")
    # init = np.random.randint(1000, 10000, size=(52,))
    # init = np.random.randint(100, 1000, size=(model.ndim,))
    init = model.get_init([0.8, 0.2]+[0]*6+[0.8, 0.2, 0, 0])
    t, y = model.predict(init=init, t_span=(0, 100), backend="solve_ivp")

    logging.info("[main] saving results ...")
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    np.save("init.npy", init)
    np.save("t.npy", t)
    np.save("y.npy", y)

    logging.info("[main] plot results ...")
    fgs = model.plot(t, y)
    for key, fg in fgs.items():
        fg.savefig("plot_%s.png" % key)

if __name__ == "__main__":
    main()

