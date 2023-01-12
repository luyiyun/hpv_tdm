# from copy import deepcopy

import optuna
import optuna.visualization.matplotlib as opt_vis
import matplotlib.pyplot as plt

from prime import Prime


prime = Prime(
        Country="CHINA",
        CohortSizeBirth=7707500,
        CohortSizeVacc=6986267,
        PropHPV1618=0.6842,
        VaccEffic=1.00,
        VaccPrice=40.44,
        VaccDeliCost=15.00,
        CancerTreatCost=650.0915022,
        AgeVaccine=12,
        DALYsCancerDiag=0.08,
        DALYsNoTerm=0.11,
        DALYsTerm=0.78,
        Discount=0.03,
        Coverage=0.8
        # Country="UNITED KINGDOM",
        # CohortSizeBirth=395953,
        # CohortSizeVacc=338891,
        # PropHPV1618=0.766,
        # VaccEffic=1.00,
        # VaccPrice=390,
        # VaccDeliCost=25.00,
        # CancerTreatCost=1325,
        # AgeVaccine=18,
        # DALYsCancerDiag=0.08,
        # DALYsNoTerm=0.04,
        # DALYsTerm=0.78,
        # Discount=0.03,
        # Coverage=0.80
    )


def objective(trial):
    vacc_price = trial.suggest_float("VaccPrice", 0, 100)
    age_vaccine = trial.suggest_int("AgeVaccine", 6, 18)
    res = prime.run(VaccPrice=vacc_price, AgeVaccine=age_vaccine)
    return (
        res["Incremental cost per DALY prevented"],
        res["Incremental cost per cervical cancer prevented"]
    )


study = optuna.create_study(directions=["minimize", "minimize"])
study.optimize(objective, n_trials=500)


# best_params = study.best_params
# params = []
# for trial in study.best_trials:
#     params.append(deepcopy(trial.params))
# params = list(set(params))
# for param in params:
#     for k, v in param.items():
#         print("Best %s: %.4f" % (k, v))
# print()
# res = prime.run(**best_params)
# for k, v in res.items():
#     print("%s: %.4f" % (k, v))

# fig = opt_vis.plot_optimization_history(study)
opt_vis.plot_pareto_front(study)
plt.show()
