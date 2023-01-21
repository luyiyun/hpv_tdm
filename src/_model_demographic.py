import logging
from copy import deepcopy

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_period_dtype
import seaborn as sns
from scipy.integrate import odeint, solve_ivp
from tqdm import tqdm

from ._demographic import compute_c, compute_fq, compute_P, find_q_newton


defaults_parameters = dict(
      agebins=np.array([
          0, 1, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 25, 27, 30, 35, 40,
          45, 50, 55, 60, 65, 70, 75, 80, 85, np.inf
      ]),
      fertilities=np.array(
          [0] * 8 +
          [4.61, 18.62, 29.73, 74.39, 112.60,
           106.95, 68.57, 26.08, 5.23, 0.44] +
          [0] * 8
      ) / 1000,
      # deathes=np.array([
      #     7.64, 0.52, 0.08, 0.58, 0.60, 0.14, 0.22, 0.35, 0.33, 0.44, 0.35,
      #     0.47, 0.72, 0.42, 0.57, 1.05, 1.20, 1.74, 3.26, 4.49, 8.91, 13.44,
      #     23.21, 35.70, 73.98, 135.09
      # ]) / 1000,
      deathes_female=np.array([
          6.68, 0.21, 0., 0.48, 0.41, 0.13, 0., 0.61, 0., 0.12, 0.67, 0.36,
          0.50, 0.27, 0.26, 0.57, 0.75, 1.09, 1.79, 2.64, 5.97, 9.51, 18.25,
          27.70, 61.45, 118.39
      ]) / 1000,
      deathes_male=np.array([
          8.49, 0.31, 0.15, 0.66, 0.76, 0.14, 0.41, 0.14, 0.61, 0.71, 0., 0.57,
          0.92, 0.55, 0.87, 1.51, 1.63, 2.38, 4.71, 6.31, 11.82, 17.52, 28.45,
          44.76, 90.01, 160.64
      ]) / 1000,
      lambda_f=0.4681,
      lambda_m=0.5319,
)


class AgeGenderModel:

    def __init__(
        self,
        agebins=None,  # 年龄分组，第一个和最后一个必须分别是0和np.inf
        fertilities=None,  # 每个年龄组的生育力，基于女性计算
        deathes_female=None,  # 女性每个年龄组的死亡率
        deathes_male=None,  # 男性每个年龄组的死亡率
        lambda_f=None,  # 出生婴儿女婴占比
        lambda_m=None,  # 出生婴儿男婴占比
        total0_f=500000,  # 初始女性总人口数量
        total0_m=500000,  # 初始女性总人口数量
        q_is_zero=True,  # 是否直接设置q为0，或者利用生育力和死亡率计算q
        rtol=1e-5,
        atol=1e-5,
        verbose=True
    ):
        self.agebins = np.array(agebins) if agebins is not None else \
            defaults_parameters["agebins"]
        self.fertilities = np.array(fertilities) if fertilities is not None \
            else defaults_parameters["fertilities"]
        self.deathes_female = np.array(deathes_female) if deathes_female is \
            not None else defaults_parameters["deathes_female"]
        self.deathes_male = np.array(deathes_male) if deathes_male is not None \
            else defaults_parameters["deathes_male"]
        self.lambda_f = lambda_f if lambda_f is not None else \
            defaults_parameters["lambda_f"]
        self.lambda_m = lambda_m if lambda_m is not None else \
            defaults_parameters["lambda_m"]
        self.total0_f, self.total0_f = total0_f, total0_m
        self.q_is_zero = q_is_zero
        self.rtol, self.atol = rtol, atol

        self.nages = len(self.agebins) - 1
        assert np.isclose(self.agebins[0], 0.)
        assert np.isclose(self.agebins[-1], np.inf)
        assert len(self.fertilities) == self.nages
        assert len(self.deathes_female) == self.nages
        assert len(self.deathes_male) == self.nages

        self.agebin_names = np.array([
            "[%d, %d)" % (i, j)
            for i, j in zip(self.agebins[:-2], self.agebins[1:-1])
        ] + ["[%d, inf)" % self.agebins[-2]])

        # --- 人口学模型参数的估计 ---
        self.agedelta = self.agebins[1:] - self.agebins[:-1]
        if q_is_zero:
            # NOTE: 当令q=0时，需要对所有的生育率进行调整，使之依然服从方程
            self.q = 0.
            factor = compute_fq(self.deathes_female,
                                self.fertilities, 0.,
                                self.agedelta,
                                lam=self.lambda_f)[0] + 1
            if verbose:
                logging.info("[init] fertilities adjust factor: %.4f" %
                             (1/factor))
            # NOTE: 不然运行多次时，会直接使用到之前的fertilities
            self.fertilities = self.fertilities / factor
        else:
            self.q = find_q_newton(
                self.lambda_f,
                self.fertilities,
                self.deathes_female,
                self.agedelta
            )[0]
            if verbose:
                logging.info("[init] q is %.4f" % self.q)
        self.c_f = compute_c(self.deathes_female, self.q, self.agedelta)
        self.c_m = compute_c(self.deathes_male, self.q, self.agedelta)
        # NOTE: 这个P_f和P_m就是初始总人口，因为P_f.sum()和P_m.sum()正好就是
        # NOTE: total0_f和total0_m
        self.P_f = compute_P(total0_f, self.deathes_female, self.q, self.c_f)
        self.P_m = compute_P(total0_m, self.deathes_male, self.q, self.c_m)
        self.dc_f = self.deathes_female + self.c_f
        self.dc_m = self.deathes_male + self.c_m

    def df_dt(self, _, X):
        Xf, Xm = X[:self.nages], X[self.nages:]
        quitf, quitm = -self.dc_f * Xf, -self.dc_m * Xm
        born = (self.fertilities * Xf).sum()
        born_f, born_m = self.lambda_f * born, self.lambda_m * born
        addf = np.r_[born_f, Xf[:-1] * self.c_f[:-1]]
        addm = np.r_[born_m, Xm[:-1] * self.c_m[:-1]]
        return np.r_[quitf + addf, quitm + addm]

    def df_dt_pbar(self, t, X, pbar, state):
        # state is a list containing last updated time t:
        # state = [last_t, dt]
        # I used a list because its values can be carried between function
        # calls throughout the ODE integration
        last_t, dt = state

        # let's subdivide t_span into 1000 parts
        # call update(n) here where n = (t - last_t) / dt
        n = int((t - last_t)/dt)
        pbar.update(n)

        # we need this to take into account that n is a rounded number.
        state[0] = last_t + dt * n

        return self.df_dt(t, X)

    def predict(
            self, init, t_span, t_eval=None, backend="solve_ivp", verbose=True
        ):
        assert backend in ["solve_ivp", "odeint"]
        if backend == "solve_ivp":
            if verbose:
                with tqdm(total=1000, unit="‰") as pbar:
                    res = solve_ivp(
                        self.df_dt_pbar, t_span=t_span, y0=init,
                        t_eval=t_eval, rtol=self.rtol, atol=self.atol,
                        args=[pbar, [t_span[0], (t_span[1]-t_span[0]) / 1000]]
                    )
            else:
                res = solve_ivp(
                    self.df_dt, t_span=t_span, y0=init,
                    t_eval=t_eval, rtol=self.rtol, atol=self.atol,
                )
            t = res.t
            y = res.y.T

        if backend == "odeint":
            if t_eval is None:
                t_eval = np.linspace(*t_span, num=1000)
            y, _ = odeint(
                self.df_dt, init, t=t_eval,
                tfirst=True, rtol=self.rtol, atol=self.atol, full_output=True,
                printmessg=True
            )
            t = t_eval

        y = self._reshpae_out(y)

        return t, y

    def _reshpae_out(self, y):
        return np.stack([y[:, :self.nages], y[:, self.nages:]], axis=1)

    def plot(self, t, y, plot_process=True, plot_dist=True):
        nt, ngender, nages = y.shape
        df = pd.DataFrame({
            "y": y.flatten(),
            "t": np.repeat(t, nages * ngender),
            "gender": np.tile(np.repeat(
                np.array(["female", "male"]), nages), nt),
            "age": pd.Categorical(
                np.tile(self.agebin_names, nt*ngender),
                categories=self.agebin_names, ordered=True
            )

        })
        fgs = {}
        if plot_process:
            fgs["process"]  = sns.relplot(
                data=df, x="t", y="y", hue="age",
                col="gender", aspect=1, kind="line",
                facet_kws={"sharey": False}
            )
        if plot_dist:
            fgs["dist"] = sns.relplot(
                data=df, x="age", y="y", hue="t",
                col="gender", aspect=2, kind="line",
                facet_kws={"sharey": False}
            )
            fgs["dist"].set_xticklabels(rotation=45)
        for fg in fgs.values():
            fg.set_titles("{col_name}")
        return fgs
