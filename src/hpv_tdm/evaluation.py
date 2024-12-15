import logging
from itertools import combinations, combinations_with_replacement
from copy import deepcopy
from typing import Dict

import numpy as np
import pandas as pd
import optuna
import seaborn as sns
import h5py

from .model import AgeGenderHPVModel2


def cal_discount(t, rate):
    return 1 / np.power((1 + rate), t[:-1])


def show_eval_result(t, result, msg=None):
    if msg is not None:
        logging.info("%s is :" % msg)
    inds = np.linspace(t.shape[0] * 0.2, t.shape[0] - 1, num=10)
    inds = inds.astype(int)
    if result.ndim == 1:
        for ind, ti, value in zip(inds, t[inds], result[inds]):
            logging.info("%d, t=%.2f, %.6f" % (ind, ti, value))
    elif result.ndim == 2:
        for ind, ti, value in zip(inds, t[inds], result[inds]):
            logging.info(
                ("%d, t=%.2f\n" % (ind, ti, value))
                + "  "
                + ",".join(["%.6f" % vi for vi in value])
            )


def notneed_calculate_again(evaluator, attrname):
    if hasattr(evaluator, attrname + "_"):
        if evaluator._disc_flag and hasattr(evaluator, attrname + "_disc_"):
            return True
        elif not evaluator._disc_flag:
            return True
    return False


def prepare_for_icur(evaluator):
    evaluator.cal_cost_vacc()
    evaluator.cal_cost_cecx()
    evaluator.cal_daly_fatal()
    evaluator.cal_daly_nofatal()
    evaluator.cal_lifeloss()


class Evaluator:
    def __init__(
        self,
        result: dict[str, np.ndarray],
        life_table: pd.DataFrame,
        cost_per_vacc: float | np.ndarray,
        cost_per_cecx: float,
        DALY_nofatal: float,
        DALY_fatal: float,
        discount_rate: float = 0.0,
        model: AgeGenderHPVModel2 | None = None,
    ) -> None:
        self.result = result
        self.ltable = life_table
        self.cost_pvacc_ = cost_per_vacc
        self.cost_pcecx_ = cost_per_cecx
        self.daly_nof = DALY_nofatal
        self.daly_f = DALY_fatal
        self.discount_rate = discount_rate
        self.model = model

        self.t_ = result["t"]
        self.y_ = result["y"]
        self.ycum_ = result["ycum"]

        self._discount_arr = cal_discount(self.t_, discount_rate)

        # TODO: 需要更加泛化的编程
        # 14岁以下只需要打2针
        self._cost_pvacc_arr = np.array(
            [self.cost_pvacc_ * 2 / 3] * 7 + [self.cost_pvacc_] * 19
        )

    @classmethod
    def from_hdf(cls, hdf5_file: str) -> "Evaluator":
        with h5py.File(hdf5_file, "r") as f:
            assert "evaluation" in f, "No evaluation data in the hdf5 file."
            g = f["evaluation"]
            result = {k: v[:] for k, v in g["result"].items()}
            cost_pvacc = g.attrs["cost_pvacc"]
            cost_pcecx = g.attrs["cost_pcecx"]
            daly_nof = g.attrs["daly_nof"]
            daly_f = g.attrs["daly_f"]
            discount_rate = g.attrs["discount"]

            if "model" in f:
                g = f["model"]
                kwargs = {k: v for k, v in g.attrs.items()}
                for k, v in g.items():
                    kwargs[k] = v[:]
                model = AgeGenderHPVModel2(**kwargs)
            else:
                model = None

        ltable = pd.read_hdf(hdf5_file, "evaluation/ltable")

        return cls(
            result=result,
            life_table=ltable,
            cost_per_vacc=cost_pvacc,
            cost_per_cecx=cost_pcecx,
            DALY_nofatal=daly_nof,
            DALY_fatal=daly_f,
            discount_rate=discount_rate,
            model=model,
        )

    def reset_discount(self, discount_rate: float) -> "Evaluator":
        return Evaluator(
            result=self.result,
            life_table=self.ltable,
            cost_per_vacc=self.cost_pvacc_,
            cost_per_cecx=self.cost_pcecx_,
            DALY_nofatal=self.daly_nof,
            DALY_fatal=self.daly_f,
            discount_rate=discount_rate,
            model=self.model,
        )

    @property
    def t(self):
        return self.t_

    @property
    def cost_pvacc(self) -> np.ndarray:
        return self._cost_pvacc_arr

    @property
    def cost_pcecx(self) -> float:
        return self.cost_pcecx_

    # def __init__(
    #     self, model, init=None, t_span=(0, 100), n_eval=None,
    #     life_table=None, cost_per_vacc=None, cost_per_cecx=None,
    #     DALY_nofatal=0.76, DALY_fatal=0.86,
    #     discount_rate=None
    # ) -> None:
    #     self.model = model
    #     self.ltable = life_table
    #     self.cost_pvacc = cost_per_vacc
    #     self.cost_pcecx = cost_per_cecx
    #     self.daly_nof = DALY_nofatal
    #     self.daly_f = DALY_fatal
    #     self.discount_rate = discount_rate
    #     # self.ref = ref

    #     # 我国人群HPV率是13.1-18.8%
    #     if init is None:
    #         init = model.get_init([0.85, 0.15]+[0]*6+[0.85, 0.15, 0, 0])
    #     else:
    #         init = np.load(init)
    #     t_span = (t_span[0], t_span[1])
    #     if n_eval is None:
    #         t_eval = None
    #     else:
    #         t_eval = np.linspace(t_span[0], t_span[1], n_eval)
    #     self.result = self.model.predict(
    #         init=init, t_span=t_span, t_eval=t_eval,
    #         backend="solve_ivp", verbose=False
    #     )

    #     self.t_ = self.result["t"]
    #     self.y_ = self.result["y"]
    #     self.ycum_ = self.result["ycum"]

    #     # NOTE: 14岁以下只需要打2针
    #     if self.cost_pvacc is not None:
    #         self.cost_pvacc_vec = np.array(
    #             [self.cost_pvacc * 2 / 3] * 7 + [self.cost_pvacc] * 19
    #         )

    #     self._disc_flag = self.discount_rate is not None
    #     if self._disc_flag:
    #         self.discount_ = cal_discount(self.t_, self.discount_rate)
    #         # self.y_ = self.y_ * self.discount_[:, None, None]
    #         # self.ycum_ = self.ycum_ * self.discount_[:, None, None]

    def _discount_cumulative(self, value):
        # if value.ndim > 1:
        #     value_ = np.concatenate([np.zeros_like(value[0])[None, :], value])
        # else:
        #     value_ = np.r_[0, value]
        # value_nocum = value_[1:] - value_[:-1]
        value_nocum = value[1:] - value[:-1]
        if value_nocum.ndim == 1:
            value_nocum *= self._discount_arr
            value_nocum = np.r_[value[0], value_nocum]
        elif value_nocum.ndim == 2:
            value_nocum *= self._discount_arr[:, None]
            value_nocum = np.concatenate([value[[0]], value_nocum], axis=0)
        elif value_nocum.ndim == 3:
            value_nocum *= self._discount_arr[:, None, None]
            value_nocum = np.concatenate([value[[0]], value_nocum], axis=0)
        elif value_nocum.ndim > 3:
            raise ValueError
        return np.cumsum(value_nocum, axis=0)

    def cal_incidence(
        self,
        reduce: bool = True,
        reuse: bool = True,
    ) -> np.ndarray:
        if hasattr(self, "incidence_") and reuse:
            return self.incidence_

        nrooms_f = self.model.nrooms_f
        Pf = self.y_[:, 2]
        DeltaLC = self.model.beta_P * Pf
        if reduce:
            incidence = DeltaLC.sum(axis=1) / self.y_[:, :nrooms_f].sum(
                axis=(1, 2)
            )
        else:
            incidence = DeltaLC / self.y_[:, :nrooms_f].sum(axis=1)
        self.incidence_ = incidence
        # if show:
        #     show_eval_result(self.t_, self.incidence_, "incidence")
        return self.incidence_

    def cal_mortality(self, reduce=True, reuse=True) -> np.ndarray:
        if hasattr(self, "mortality_") and reuse:
            return self.mortality_
        ndeath = (
            self.model.dL * self.y_[:, 3]
            + self.model.dR * self.y_[:, 4]
            + self.model.dD * self.y_[:, 5]
        )
        if reduce:
            mortality = ndeath.sum(axis=1) / self.y_[
                :, : self.model.nrooms_f
            ].sum(axis=(1, 2))
        else:
            mortality = ndeath / self.y_[:, : self.model.nrooms_f].sum(axis=1)
        self.mortality_ = mortality
        # if show:
        #     show_eval_result(self.t_, self.mortality_, "mortality")
        return self.mortality_

    def cal_number_cecx(self, reuse=True) -> np.ndarray:
        if hasattr(self, "number_cecx_") and reuse:
            return self.number_cecx_
        self.number_cecx_ = self.ycum_[:, 3]
        if self.discount_rate is not None and self.discount_rate > 0.0:
            self.number_cecx_ = self._discount_cumulative(self.number_cecx_)
        return self.number_cecx_

    def cal_number_vacc(self, reuse=True) -> np.ndarray:
        if hasattr(self, "number_vacc_") and reuse:
            return self.number_vacc_
        self.number_vacc_ = self.ycum_[:, -1]
        if self.discount_rate is not None and self.discount_rate > 0.0:
            self.number_vacc_ = self._discount_cumulative(self.number_vacc_)
        return self.number_vacc_

    def cal_number_cecxDeath(self, reuse=True) -> np.ndarray:
        if hasattr(self, "number_cecxDeath_") and reuse:
            return self.number_cecxDeath_
        self.number_cecxDeath_ = self.ycum_[:, 6:9].sum(axis=1)
        if self.discount_rate is not None and self.discount_rate > 0.0:
            self.number_cecxDeath_ = self._discount_cumulative(
                self.number_cecxDeath_
            )
        return self.number_cecxDeath_

    def cal_cost_vacc(self, reuse=True) -> np.ndarray:
        if hasattr(self, "cost_vacc_") and reuse:
            return self.cost_vacc_
        self.cost_vacc_ = np.dot(self.ycum_[:, -1], self._cost_pvacc_arr)
        if self.discount_rate is not None and self.discount_rate > 0.0:
            self.cost_vacc_ = self._discount_cumulative(self.cost_vacc_)
        return self.cost_vacc_

    def cal_cost_cecx(self, reuse=True) -> np.ndarray:
        if hasattr(self, "cost_cecx_") and reuse:
            return self.cost_cecx_
        self.cost_cecx_ = self.ycum_[:, 3].sum(axis=1) * self.cost_pcecx_
        if self.discount_rate is not None and self.discount_rate > 0.0:
            self.cost_cecx_ = self._discount_cumulative(self.cost_cecx_)
        return self.cost_cecx_

    def cal_daly_fatal(self, reuse=True) -> np.ndarray:
        if hasattr(self, "daly_fatal_") and reuse:
            return self.daly_fatal_
        self.daly_fatal_ = self.ycum_[:, 6:9].sum(axis=(1, 2)) * self.daly_f
        if self.discount_rate is not None and self.discount_rate > 0.0:
            self.daly_fatal_ = self._discount_cumulative(self.daly_fatal_)
        return self.daly_fatal_

    def cal_daly_nofatal(self, reuse=True) -> np.ndarray:
        if hasattr(self, "daly_nofatal_") and reuse:
            return self.daly_nofatal_
        n_cecx = self.ycum_[:, 3].sum(axis=1)
        n_cecx_death = self.ycum_[:, 6:9].sum(axis=(1, 2))
        self.daly_nofatal_ = (n_cecx - n_cecx_death) * self.daly_nof
        if self.discount_rate is not None and self.discount_rate > 0.0:
            self.daly_nofatal_ = self._discount_cumulative(self.daly_nofatal_)
        return self.daly_nofatal_

    def cal_lifeloss(self, reuse=True) -> np.ndarray:
        if hasattr(self, "lifeloss_") and reuse:
            return self.lifeloss_
        n_cecx_death = self.ycum_[:, 6:9].sum(axis=1)
        self.lifeloss_ = np.sum(
            n_cecx_death * self.ltable["E"].values, axis=-1
        )
        if self.discount_rate is not None and self.discount_rate > 0.0:
            self.lifeloss_ = self._discount_cumulative(self.lifeloss_)
        return self.lifeloss_

    def _prepare_for_icur(self, reuse=True):
        self.cal_cost_vacc(reuse)
        self.cal_cost_cecx(reuse)
        self.cal_daly_fatal(reuse)
        self.cal_daly_nofatal(reuse)
        self.cal_lifeloss(reuse)

    def cal_icur(
        self, ref_eval: "Evaluator", reuse=True, minor_reuse=True
    ) -> np.ndarray:
        if hasattr(self, "icur_") and reuse:
            return self.icur_

        self._prepare_for_icur(reuse=minor_reuse)
        ref_eval._prepare_for_icur(reuse=minor_reuse)

        tar_cost = self.cost_vacc_ + self.cost_cecx_
        ref_cost = ref_eval.cost_vacc_ + ref_eval.cost_cecx_
        tar_daly = self.daly_nofatal_ + self.daly_fatal_ + self.lifeloss_
        ref_daly = (
            ref_eval.daly_nofatal_ + ref_eval.daly_fatal_ + ref_eval.lifeloss_
        )
        cost_diff = tar_cost - ref_cost
        daly_diff = ref_daly - tar_daly
        self.icur_ = np.divide(
            cost_diff,
            daly_diff,
            out=np.full_like(cost_diff, fill_value=np.inf),
            where=daly_diff != 0,
        )
        return self.icur_

    def cal_avoid_cecx(
        self, ref_eval: "Evaluator", reuse=True, minor_reuse=True
    ) -> np.ndarray:
        if hasattr(self, "avoid_cecx_") and reuse:
            return self.avoid_cecx_

        tar_n_cecx = self.cal_number_cecx(reuse=minor_reuse)
        ref_n_cecx = ref_eval.cal_number_cecx(reuse=minor_reuse)
        self.avoid_cecx_ = ref_n_cecx - tar_n_cecx
        return self.avoid_cecx_

    def cal_avoid_cecxDeath(
        self, ref_eval: "Evaluator", reuse=True, minor_reuse=True
    ) -> np.ndarray:
        if hasattr(self, "avoid_cecxDeath_") and reuse:
            return self.avoid_cecxDeath_

        tar_n_cecx_death = self.cal_number_cecxDeath(reuse=minor_reuse)
        ref_n_cecx_death = ref_eval.cal_number_cecxDeath(reuse=minor_reuse)
        self.avoid_cecxDeath_ = ref_n_cecx_death - tar_n_cecx_death
        return self.avoid_cecxDeath_

    def cal_avoid_daly(
        self, ref_eval: "Evaluator", reuse=True, minor_reuse=True
    ) -> np.ndarray:
        if hasattr(self, "avoid_daly_") and reuse:
            return self.avoid_daly_
        self.cal_daly_fatal(minor_reuse)
        self.cal_daly_nofatal(minor_reuse)
        self.cal_lifeloss(minor_reuse)
        ref_eval.cal_daly_fatal(reuse=minor_reuse)
        ref_eval.cal_daly_nofatal(minor_reuse)
        ref_eval.cal_lifeloss(minor_reuse)
        self.avoid_daly_ = (
            ref_eval.daly_fatal_
            + ref_eval.daly_nofatal_
            + ref_eval.lifeloss_
            - (self.daly_fatal_ + self.daly_nofatal_ + self.lifeloss_)
        )
        return self.avoid_daly_


class Ploter:
    def __init__(self, **kwargs: Dict[str, Evaluator]):
        self.kwargs = kwargs
        self.n_ = len(self.kwargs)

    def plot_incidence(self, savefn="./incidence.png", log=False):
        df = []
        for k, eval in self.kwargs.items():
            inci = eval.cal_incidence(reduce=True).incidence_
            dfi = pd.DataFrame({"t": eval.t_, "inci": inci})
            dfi["group"] = k
            df.append(dfi)
        df = pd.concat(df)
        fg = sns.relplot(
            data=df,
            x="t",
            y="inci",
            hue="group" if self.n_ > 1 else None,
            kind="line",
            aspect=2,
            height=4,
        )
        if log:
            fg.set(yscale="log")
        fg.savefig(savefn)

    def plot_mortality(self, savefn="./mortality.png", log=False):
        df = []
        for k, eval in self.kwargs.items():
            mort = eval.cal_mortality(reduce=True).mortality_
            dfi = pd.DataFrame({"t": eval.t_, "mort": mort})
            dfi["group"] = k
            df.append(dfi)
        df = pd.concat(df)
        fg = sns.relplot(
            data=df,
            x="t",
            y="mort",
            hue="group" if self.n_ > 1 else None,
            kind="line",
            aspect=2,
            height=4,
        )
        if log:
            fg.set(yscale="log")
        fg.savefig(savefn)

    def plot_cost(self, savefn="./cost.png"):
        df = []
        for k, eval in self.kwargs.items():
            eval.cal_cost_vacc()
            eval.cal_cost_cecx()
            cost_vacc = eval.cost_vacc_
            cost_cecx = eval.cost_cecx_
            cost_all = cost_vacc + cost_cecx
            dfi = pd.DataFrame(
                {
                    "t": np.tile(eval.t_, 3),
                    "type": np.repeat(
                        ["vacc", "cecx", "all"], cost_vacc.shape[0]
                    ),
                    "value": np.r_[cost_vacc, cost_cecx, cost_all],
                }
            )
            dfi["group"] = k
            dfi["discount"] = "NoDisc"
            df.append(dfi)
            if eval._disc_flag:
                cost_vacc = eval.cost_vacc_disc_
                cost_cecx = eval.cost_cecx_disc_
                cost_all = cost_vacc + cost_cecx
                dfi = pd.DataFrame(
                    {
                        "t": np.tile(eval.t_, 3),
                        "type": np.repeat(
                            ["vacc", "cecx", "all"], cost_vacc.shape[0]
                        ),
                        "value": np.r_[cost_vacc, cost_cecx, cost_all],
                    }
                )
                dfi["group"] = k
                dfi["discount"] = "Disc"
                df.append(dfi)
        df = pd.concat(df)

        style_as_disc = False
        for eval in self.kwargs.values():
            if eval._disc_flag:
                style_as_disc = True
                break

        fg = sns.relplot(
            data=df,
            x="t",
            y="value",
            hue="group" if self.n_ > 1 else None,
            style="discount" if style_as_disc else None,
            col="type",
            kind="line",
            aspect=1.2,
            height=4,
        )
        fg.savefig(savefn)

    def plot_daly(self, savefn="./daly.png"):
        df = []
        for k, eval in self.kwargs.items():
            eval.cal_daly_fatal()
            eval.cal_daly_nofatal()
            daly_fatal = eval.daly_fatal_
            daly_nofatal = eval.daly_nofatal_
            dfi = pd.DataFrame(
                {
                    "t": np.tile(eval.t_, 2),
                    "type": np.repeat(
                        ["fatal", "nofatal"], daly_fatal.shape[0]
                    ),
                    "value": np.r_[daly_fatal, daly_nofatal],
                }
            )
            dfi["group"] = k
            dfi["discount"] = "NoDisc"
            df.append(dfi)
            if eval._disc_flag:
                daly_fatal = eval.daly_fatal_disc_
                daly_nofatal = eval.daly_nofatal_disc_
                dfi = pd.DataFrame(
                    {
                        "t": np.tile(eval.t_, 2),
                        "type": np.repeat(
                            ["fatal", "nofatal"], daly_fatal.shape[0]
                        ),
                        "value": np.r_[daly_fatal, daly_nofatal],
                    }
                )
                dfi["group"] = k
                dfi["discount"] = "Disc"
                df.append(dfi)
        df = pd.concat(df)

        style_as_disc = False
        for eval in self.kwargs.values():
            if eval._disc_flag:
                style_as_disc = True
                break

        fg = sns.relplot(
            data=df,
            x="t",
            y="value",
            hue="group" if self.n_ > 1 else None,
            style="discount" if style_as_disc else None,
            col="type",
            kind="line",
            aspect=1.2,
            height=4,
        )
        fg.savefig(savefn)

    def plot_icur(self, ref_key, t_span=None, savefn="./icur.png"):
        ref_eval = self.kwargs[ref_key]
        df = []
        for k, eval in self.kwargs.items():
            if k == ref_key:
                continue
            eval.cal_icur(ref_eval)

            dfi = pd.DataFrame(
                {
                    "t": eval.t_,
                    "value": eval.icur_,
                }
            )
            dfi["group"] = k
            dfi["discount"] = "NoDisc"
            df.append(dfi)
            if eval._disc_flag:
                dfi = pd.DataFrame(
                    {
                        "t": eval.t_,
                        "value": eval.icur_disc_,
                    }
                )
                dfi["group"] = k
                dfi["discount"] = "Disc"
                df.append(dfi)
        df = pd.concat(df).reset_index(drop=True)

        style_as_disc = False
        for eval in self.kwargs.values():
            if eval._disc_flag:
                style_as_disc = True
                break

        if t_span is not None:
            df = df[(df.t >= t_span[0]) & (df.t <= t_span[1])]

        fg = sns.relplot(
            data=df,
            x="t",
            y="value",
            hue="group" if self.n_ > 2 else None,
            style="discount" if style_as_disc else None,
            kind="line",
            aspect=1.2,
            height=4,
        )
        fg.savefig(savefn)


class ObjectiveFunction:
    d_tau = {"dom2": 0.691, "imp2": 0.691, "imp9": 0.921}
    d_cost_per_vacc = {"dom2": 153.2, "imp2": 262.38, "imp9": 574.71}
    n_ages = 26
    age_index_span = (0, 11)

    def __init__(self, ref_model, eval_kwargs, tdm_kwargs, **kwargs) -> None:
        self.ref_model = ref_model
        self.eval_kwargs = eval_kwargs
        self.tdm_kwargs = tdm_kwargs
        self.kwargs = kwargs

        self.ref_eval = Evaluator(self.ref_model, **self.eval_kwargs)
        self.ref_eval.cal_cost_vacc()
        self.ref_eval.cal_cost_cecx()
        self.ref_eval.cal_daly_fatal()
        self.ref_eval.cal_daly_nofatal()
        self.ref_eval.cal_lifeloss()

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
            cal_cumulate=True,
            psi=psi,
            tau=tau,
            verbose=False,
            **self.tdm_kwargs,
        )
        # 构建评价器
        eval_kwargs = deepcopy(self.eval_kwargs)
        eval_kwargs["cost_per_vacc"] = cost_per_vacc
        evaluator = Evaluator(model, **eval_kwargs)

        # 计算指标
        inci = evaluator.cal_incidence().incidence_
        evaluator.cal_icur(self.ref_eval)
        if evaluator._disc_flag:
            icur = evaluator.icur_disc_
        else:
            icur = evaluator.icur_
        return evaluator, inci, icur

    def __call__(self, trial: optuna.Trial) -> float:
        # 得到参数
        tage = trial.suggest_categorical(
            "target_age", tuple(range(*self.age_index_span))
        )
        tvacc = trial.suggest_categorical("target_vacc", self.d_tau.keys())
        cover = trial.suggest_float("coverage", low=0.0, high=1.0)

        _, inci, ic = self._eval_by_params(
            target_age=tage, target_vacc=tvacc, coverage=cover
        )
        # return np.median(icer[-10:]), np.median(inci[-1:])
        return ic[-1], inci[-1]

    def call_from_parameters(self, params: dict):
        return self._eval_by_params(**params)

    def show_params(self, params: dict):
        for k, v in params.items():
            if k.startswith("target_age"):
                logging.info("  %s: %s" % (k, self.ref_model.agebin_names[v]))
            elif isinstance(v, float):
                logging.info("  %s: %.6f" % (k, v))
            elif isinstance(v, int):
                logging.info("  %s: %d" % (k, v))
            else:
                logging.info("  %s: %s" % (k, v))


class MultiAgesObjectiveFunction(OneAgeObjectiveFunction):
    def __init__(
        self, ref_model, eval_kwargs, tdm_kwargs, n_vacc_ages=1, **kwargs
    ) -> None:
        super().__init__(
            ref_model=ref_model,
            eval_kwargs=eval_kwargs,
            tdm_kwargs=tdm_kwargs,
            **kwargs,
        )
        self.n_vacc_ages = n_vacc_ages
        self.suggested_ages = list(
            combinations(range(*self.age_index_span), n_vacc_ages)
        )

    def __call__(self, trial: optuna.Trial) -> float:
        # 得到参数
        tages = np.array(
            trial.suggest_categorical("target_age", self.suggested_ages)
        )
        tvacc = trial.suggest_categorical(
            "target_vacc", ["dom2", "imp2", "imp9"]
        )
        covers = np.array(
            [
                trial.suggest_float("coverage%d" % i, low=0.0, high=1.0)
                for i in range(self.n_vacc_ages)
            ]
        )

        _, inci, ic = self._eval_by_params(
            target_age=tages, target_vacc=tvacc, coverage=covers
        )
        # return np.median(icer[-10:]), np.median(inci[-1:])
        return ic[-1], inci[-1]

    def call_from_parameters(self, params: dict):
        covers = np.array(
            [params["coverage%d" % i] for i in range(self.n_vacc_ages)]
        )
        return self._eval_by_params(
            target_age=np.array(params["target_age"]),
            coverage=covers,
            target_vacc=params["target_vacc"],
        )

    def show_params(self, params: dict):
        for k, v in params.items():
            if k.startswith("target_age"):
                logging.info(
                    ("  %s: " % k)
                    + ", ".join(
                        [self.ref["model"].agebin_names[vi] for vi in v]
                    )
                )
            elif isinstance(v, float):
                logging.info("  %s: %.6f" % (k, v))
            elif isinstance(v, int):
                logging.info("  %s: %d" % (k, v))
            else:
                logging.info("  %s: %s" % (k, v))


class ContiAgesObjectiveFunction(OneAgeObjectiveFunction):
    def __init__(self, ref_model, eval_kwargs, tdm_kwargs, **kwargs) -> None:
        super().__init__(
            ref_model=ref_model,
            eval_kwargs=eval_kwargs,
            tdm_kwargs=tdm_kwargs,
            **kwargs,
        )
        self.suggest_ages_span = list(
            combinations_with_replacement(range(*self.age_index_span), 2)
        )

    def __call__(self, trial: optuna.Trial) -> float:
        # 得到参数
        tage0, tage1 = trial.suggest_categorical(
            "target_age_span", self.suggest_ages_span
        )
        tvacc = trial.suggest_categorical(
            "target_vacc", ["dom2", "imp2", "imp9"]
        )
        covers = np.array(
            [
                trial.suggest_float("coverage%d" % i, low=0.0, high=1.0)
                for i in range(tage0, tage1 + 1)
            ]
        )

        _, inci, ic = self._eval_by_params(
            target_age=slice(tage0, tage1 + 1),
            target_vacc=tvacc,
            coverage=covers,
        )
        # return np.median(icer[-10:]), np.median(inci[-1:])
        return ic[-1], inci[-1]

    def call_from_parameters(self, params: dict):
        tage0, tage1 = params["target_age_span"]
        covers = np.array(
            [params["coverage%d" % i] for i in range(tage0, tage1 + 1)]
        )
        return self._eval_by_params(
            target_age=slice(tage0, tage1 + 1),
            coverage=covers,
            target_vacc=params["target_vacc"],
        )

    def show_params(self, params: dict):
        for k, v in params.items():
            if k.startswith("target_age"):
                logging.info(
                    ("  %s: " % k)
                    + ", ".join([self.ref_model.agebin_names[vi] for vi in v])
                )
            elif isinstance(v, float):
                logging.info("  %s: %.6f" % (k, v))
            elif isinstance(v, int):
                logging.info("  %s: %d" % (k, v))
            else:
                logging.info("  %s: %s" % (k, v))


class ContiAgesOneCoverObjectiveFunction(OneAgeObjectiveFunction):
    def __init__(self, ref_model, eval_kwargs, tdm_kwargs, **kwargs) -> None:
        super().__init__(
            ref_model=ref_model,
            eval_kwargs=eval_kwargs,
            tdm_kwargs=tdm_kwargs,
            **kwargs,
        )
        self.suggest_ages_span = list(
            combinations_with_replacement(range(*self.age_index_span), 2)
        )

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
            target_age=slice(tage0, tage1 + 1),
            target_vacc=tvacc,
            coverage=cover,
        )
        # return np.median(icer[-10:]), np.median(inci[-1:])
        return ic[-1], inci[-1]

    def call_from_parameters(self, params: dict):
        tage0, tage1 = params["target_age_span"]
        return self._eval_by_params(
            target_age=slice(tage0, tage1 + 1),
            coverage=params["coverage"],
            target_vacc=params["target_vacc"],
        )

    def show_params(self, params: dict):
        for k, v in params.items():
            if k.startswith("target_age"):
                logging.info(
                    ("  %s: " % k)
                    + ", ".join([self.ref_model.agebin_names[vi] for vi in v])
                )
            elif isinstance(v, float):
                logging.info("  %s: %.6f" % (k, v))
            elif isinstance(v, int):
                logging.info("  %s: %d" % (k, v))
            else:
                logging.info("  %s: %s" % (k, v))
