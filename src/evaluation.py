import logging
from itertools import combinations, combinations_with_replacement

import numpy as np
import optuna

from .model import AgeGenderHPVModel2


def incidence(result, method="instant"):
    y = result["y"]
    ycum = result["ycum"]
    model = result["model"]
    nrooms_f = model.nrooms_f
    if method == "oneyear":
        # 1年发病率
        cum_cecx = ycum[:, 3].sum(axis=1)
        delta_cecx = cum_cecx[1:] - cum_cecx[:-1]
        nfemale = y[:, :nrooms_f].sum(axis=(1, 2))
        nfemale = (nfemale[1:] +nfemale[:-1]) / 2
        incidence = delta_cecx / nfemale
        # 保证等长度，后面的程序不用修改
        incidence = np.r_[incidence[0], incidence]
    elif method == "instant":
        # 瞬时发病率
        Pf = y[:, 2]
        DeltaLC = model.beta_P * Pf
        incidence = DeltaLC.sum(axis=1) / y[:, :nrooms_f].sum(axis=(1, 2))
    else:
        raise NotImplementedError

    return incidence


def death(result, method="instant"):
    assert method == "instant"
    y = result["y"]
    model = result["model"]
    nrooms_f = model.nrooms_f

    # 瞬时发病率
    ndeath = model.dL * y[:, 3] + model.dR * y[:, 4] + model.dD * y[:, 5]
    death = ndeath.sum(axis=1) / y[:, :nrooms_f].sum(axis=(1, 2))
    return death


def show_incidence(inci, t):
    logging.info("incidence rate is :")
    inds = np.linspace(inci.shape[0] * 0.2, inci.shape[0]-1, num=10)
    inds = inds.astype(int)
    for ind, ti, value in zip(inds, t[inds], inci[inds]):
        logging.info("%d, t: %.4f, inci: %.6f" % (ind, ti, value))


def show_death(dea, t):
    logging.info("Death rate is :")
    inds = np.linspace(dea.shape[0] * 0.2, dea.shape[0]-1, num=10)
    inds = inds.astype(int)
    for ind, ti, value in zip(inds, t[inds], dea[inds]):
        logging.info("%d, t: %.4f, death: %.6f" % (ind, ti, value))


def cost_utility(
    result, life_table, cost_per_vacc, cost_per_cecx,
    DALY_nofatal=0.76, DALY_fatal=0.86
):
    ycum = result["ycum"]
    # 分为两个部分：疫苗接种花费和癌症治疗花费
    nVacc = ycum[:, -1]  # .sum(axis=1)
    nCecx = ycum[:, 3].sum(axis=1)
    nCecxDeathAge = ycum[:, 6:9].sum(axis=1)
    nCecxDeath = nCecxDeathAge.sum(axis=1)

    # NOTE: 14岁以下只需要打2针
    cVacc = np.dot(nVacc, np.array([cost_per_vacc*2/3]*7+[cost_per_vacc]*19))
    cCecx = nCecx * cost_per_cecx
    cAll = cVacc + cCecx
    dDeath = nCecxDeath * DALY_fatal
    dNoDeath = (nCecx - nCecxDeath) * DALY_nofatal
    lLoss = (nCecxDeathAge * life_table["E"].values).sum(axis=-1)

    return {
        "cost_vacc": cVacc,
        "cost_cecx": cCecx,
        "cost_all": cAll,
        "DALY_death": dDeath,
        "DALY_nodeath": dNoDeath,
        "LifeLoss": lLoss,
    }


def icer(tar_cu: dict, ref_cu: dict):
    tar_cost = tar_cu["cost_all"]
    ref_cost = ref_cu["cost_all"]
    ref_daly = ref_cu["DALY_nodeath"] + ref_cu["DALY_death"] + ref_cu["LifeLoss"]
    tar_daly = tar_cu["DALY_nodeath"] + tar_cu["DALY_death"] + tar_cu["LifeLoss"]
    return (tar_cost - ref_cost) / (ref_daly - tar_daly)


def avoid_incidence(tar_result, ref_result):
    return ref_result["ycum"][:, 3] - tar_result["ycum"][:, 3]


def avoid_death(tar_result, ref_result):
    return ref_result["ycum"][:, 6:9] - tar_result["ycum"][:, 6:9]


class ObjectiveFunction:

    d_tau = {"dom2": 0.691, "imp2": 0.691, "imp9": 0.921}
    d_cost_per_vacc = {"dom2": 153.2, "imp2": 262.38, "imp9": 574.71}
    n_ages = 26
    age_index_span = (0, 11)

    def __init__(
        self,
        reference,
        cu_kwargs,
        tdm_kwargs,
        **kwargs
    ) -> None:
        self.ref = reference
        self.cu_kwargs = cu_kwargs
        self.tdm_kwargs = tdm_kwargs
        self.kwargs = kwargs

        self.t_span = (reference["t"].min(), reference["t"].max())
        self.t_eval = reference["t"]
        self.init = np.r_[
            reference["y"][0].flatten(),
            np.zeros(np.prod(reference["ycum"].shape[1:]))
        ]

    def __call__(self, trial: optuna.Trial) -> float:
        raise NotImplementedError

    def call_from_parameters(self, params: dict):
        raise NotImplementedError


class OneAgeObjectiveFunction(ObjectiveFunction):

    def _eval_by_params(self, target_age, target_vacc, coverage):
        # 根据参数进行设置
        psi = np.zeros(self.n_ages)
        psi[target_age] = coverage
        tau = self.d_tau[target_vacc]
        cost_per_vacc = self.d_cost_per_vacc[target_vacc]

        # 构建模型
        model = AgeGenderHPVModel2(
            cal_cumulate=True, psi=psi, tau=tau, verbose=False,
            **self.tdm_kwargs
        )

        # 得到初始值，进行运算
        res = model.predict(
            init=self.init, t_span=self.t_span,
            t_eval=self.t_eval, backend="solve_ivp", verbose=False
        )
        # 计算指标
        inci = incidence(res)
        cu = cost_utility(
            res, self.ref["ltable"],
            cost_per_vacc=cost_per_vacc,
            **self.cu_kwargs
        )
        ic = icer(cu, self.ref["cu"])
        return res, inci, cu, ic

    def __call__(self, trial: optuna.Trial) -> float:
        # 得到参数
        tage = trial.suggest_categorical(
            "target_age", tuple(range(*self.age_index_span))
        )
        tvacc = trial.suggest_categorical("target_vacc", self.d_tau.keys())
        cover = trial.suggest_float("coverage", low=0.0, high=1.0)

        _, inci, _, ic = self._eval_by_params(
            target_age=tage, target_vacc=tvacc, coverage=cover
        )
        # return np.median(icer[-10:]), np.median(inci[-1:])
        return ic[-1], inci[-1]

    def call_from_parameters(self, params: dict):
        return self._eval_by_params(**params)

    def show_params(self, params: dict):
        for k, v in params.items():
            if k.startswith("target_age"):
                logging.info(
                    "  %s: %s" % (k, self.ref["model"].agebin_names[v])
                )
            elif isinstance(v, float):
                logging.info("  %s: %.6f" % (k, v))
            elif isinstance(v, int):
                logging.info("  %s: %d" % (k, v))
            else:
                logging.info("  %s: %s" % (k, v))


class MultiAgesObjectiveFunction(OneAgeObjectiveFunction):

    def __init__(
        self,
        reference,
        cu_kwargs,
        tdm_kwargs,
        n_vacc_ages=1,
        **kwargs
    ) -> None:
        super().__init__(reference=reference,
                         cu_kwargs=cu_kwargs,
                         tdm_kwargs=tdm_kwargs,
                         **kwargs)
        self.n_vacc_ages = n_vacc_ages
        self.suggested_ages = list(combinations(
            range(*self.age_index_span), n_vacc_ages
        ))

    def __call__(self, trial: optuna.Trial) -> float:
        # 得到参数
        tages = np.array(trial.suggest_categorical(
            "target_age", self.suggested_ages
        ))
        tvacc = trial.suggest_categorical(
            "target_vacc", ["dom2", "imp2", "imp9"]
        )
        covers = np.array([
            trial.suggest_float("coverage%d" % i, low=0.0, high=1.0)
            for i in range(self.n_vacc_ages)
        ])

        _, inci, _, ic = self._eval_by_params(
            target_age=tages, target_vacc=tvacc, coverage=covers
        )
        # return np.median(icer[-10:]), np.median(inci[-1:])
        return ic[-1], inci[-1]

    def call_from_parameters(self, params: dict):
        covers = np.array([
            params["coverage%d" % i]
            for i in range(self.n_vacc_ages)
        ])
        return self._eval_by_params(
            target_age=np.array(params["target_age"]),
            coverage=covers,
            target_vacc=params["target_vacc"],
        )

    def show_params(self, params: dict):
        for k, v in params.items():
            if k.startswith("target_age"):
                logging.info(
                    ("  %s: " % k) +
                    ", ".join([self.ref["model"].agebin_names[vi] for vi in v])
                )
            elif isinstance(v, float):
                logging.info("  %s: %.6f" % (k, v))
            elif isinstance(v, int):
                logging.info("  %s: %d" % (k, v))
            else:
                logging.info("  %s: %s" % (k, v))


class ContiAgesObjectiveFunction(OneAgeObjectiveFunction):

    def __init__(
        self,
        reference,
        cu_kwargs,
        tdm_kwargs,
        **kwargs
    ) -> None:
        super().__init__(reference=reference,
                         cu_kwargs=cu_kwargs,
                         tdm_kwargs=tdm_kwargs,
                         **kwargs)
        self.suggest_ages_span = list(combinations_with_replacement(
            range(*self.age_index_span), 2
        ))

    def __call__(self, trial: optuna.Trial) -> float:
        # 得到参数
        tage0, tage1 = trial.suggest_categorical(
            "target_age_span", self.suggest_ages_span
        )
        tvacc = trial.suggest_categorical(
            "target_vacc", ["dom2", "imp2", "imp9"]
        )
        covers = np.array([
            trial.suggest_float("coverage%d" % i, low=0.0, high=1.0)
            for i in range(tage0, tage1+1)
        ])

        _, inci, _, ic = self._eval_by_params(
            target_age=slice(tage0, tage1+1),
            target_vacc=tvacc,
            coverage=covers
        )
        # return np.median(icer[-10:]), np.median(inci[-1:])
        return ic[-1], inci[-1]

    def call_from_parameters(self, params: dict):
        tage0, tage1 = params["target_age_span"]
        covers = np.array([
            params["coverage%d" % i]
            for i in range(tage0, tage1+1)
        ])
        return self._eval_by_params(
            target_age=slice(tage0, tage1+1),
            coverage=covers,
            target_vacc=params["target_vacc"],
        )

    def show_params(self, params: dict):
        for k, v in params.items():
            if k.startswith("target_age"):
                logging.info(
                    ("  %s: " % k) +
                    ", ".join([self.ref["model"].agebin_names[vi] for vi in v])
                )
            elif isinstance(v, float):
                logging.info("  %s: %.6f" % (k, v))
            elif isinstance(v, int):
                logging.info("  %s: %d" % (k, v))
            else:
                logging.info("  %s: %s" % (k, v))


class ContiAgesOneCoverObjectiveFunction(OneAgeObjectiveFunction):

    def __init__(
        self,
        reference,
        cu_kwargs,
        tdm_kwargs,
        **kwargs
    ) -> None:
        super().__init__(reference=reference,
                         cu_kwargs=cu_kwargs,
                         tdm_kwargs=tdm_kwargs,
                         **kwargs)
        self.suggest_ages_span = list(combinations_with_replacement(
            range(*self.age_index_span), 2
        ))

    def __call__(self, trial: optuna.Trial) -> float:
        # 得到参数
        tage0, tage1 = trial.suggest_categorical(
            "target_age_span", self.suggest_ages_span
        )
        tvacc = trial.suggest_categorical(
            "target_vacc", ["dom2", "imp2", "imp9"]
        )
        cover = trial.suggest_float("coverage", low=0.0, high=1.0)

        _, inci, _, ic = self._eval_by_params(
            target_age=slice(tage0, tage1+1),
            target_vacc=tvacc,
            coverage=cover
        )
        # return np.median(icer[-10:]), np.median(inci[-1:])
        return ic[-1], inci[-1]

    def call_from_parameters(self, params: dict):
        tage0, tage1 = params["target_age_span"]
        return self._eval_by_params(
            target_age=slice(tage0, tage1+1),
            coverage=params["coverage"],
            target_vacc=params["target_vacc"],
        )

    def show_params(self, params: dict):
        for k, v in params.items():
            if k.startswith("target_age"):
                logging.info(
                    ("  %s: " % k) +
                    ", ".join([self.ref["model"].agebin_names[vi] for vi in v])
                )
            elif isinstance(v, float):
                logging.info("  %s: %.6f" % (k, v))
            elif isinstance(v, int):
                logging.info("  %s: %d" % (k, v))
            else:
                logging.info("  %s: %s" % (k, v))
