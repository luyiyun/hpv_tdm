from inspect import isfunction
import logging

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.integrate import odeint, solve_ivp
from tqdm import tqdm

from ._demographic import compute_c, compute_fq, compute_P, find_q_newton
from ._sexual import compute_rho


class BaseModel:

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        for k, v in self.kwargs.items():
            setattr(self, k, v)

    def df_dt(self, t, X):
        raise NotImplementedError

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

    def _neat_predict_results(self, t, y):
        raise NotImplementedError

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

        return self._neat_predict_results(t, y)


class AgeGenderModel(BaseModel):

    def __init__(
        self,
        # 年龄分组，第一个和最后一个必须分别是0和np.inf
        agebins=np.array([
            0, 1, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 25, 27, 30, 35, 40,
            45, 50, 55, 60, 65, 70, 75, 80, 85, np.inf
        ]),
        # 每个年龄组的生育力，基于女性计算
        fertilities=np.array(
            [0] * 8 +
            [4.61, 18.62, 29.73, 74.39, 112.60,
             106.95, 68.57, 26.08, 5.23, 0.44] +
            [0] * 8
        ) / 1000,
        # 女性每个年龄组的死亡率
        deathes_female=np.array([
            6.68, 0.21, 0., 0.48, 0.41, 0.13, 0., 0.61, 0., 0.12, 0.67, 0.36,
            0.50, 0.27, 0.26, 0.57, 0.75, 1.09, 1.79, 2.64, 5.97, 9.51, 18.25,
            27.70, 61.45, 118.39
        ]) / 1000,
        # 男性每个年龄组的死亡率
        deathes_male=np.array([
            8.49, 0.31, 0.15, 0.66, 0.76, 0.14, 0.41, 0.14, 0.61, 0.71, 0., 0.57,
            0.92, 0.55, 0.87, 1.51, 1.63, 2.38, 4.71, 6.31, 11.82, 17.52, 28.45,
            44.76, 90.01, 160.64
        ]) / 1000,
        # 每个年龄组的死亡率
        # deathes=np.array([
        #     7.64, 0.52, 0.08, 0.58, 0.60, 0.14, 0.22, 0.35, 0.33, 0.44, 0.35,
        #     0.47, 0.72, 0.42, 0.57, 1.05, 1.20, 1.74, 3.26, 4.49, 8.91, 13.44,
        #     23.21, 35.70, 73.98, 135.09
        # ]) / 1000,
        # 出生婴儿女婴占比
        lambda_f=0.4681,
        # 出生婴儿男婴占比
        lambda_m=0.5319,
        # 初始女性总人口数量
        total0_f=500000,
        # 初始男性总人口数量
        total0_m=500000,
        # 是否直接设置q为0，或者利用生育力和死亡率计算q
        q_is_zero=True,
        rtol=1e-5,
        atol=1e-5,
        verbose=True,
        **kwargs
    ):
        super().__init__(
            agebins=agebins, fertilities=fertilities,
            deathes_female=deathes_female, deathes_male=deathes_male,
            lambda_f=lambda_f, lambda_m=lambda_m,
            total0_f=total0_f, total0_m=total0_m,
            q_is_zero=q_is_zero, rtol=rtol, atol=atol, verbose=verbose,
            **kwargs
        )

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

    def _neat_predict_results(self, t, y):
        return {
            "t": t,
            "y": np.stack([y[:, :self.nages], y[:, self.nages:]], axis=1),
            "model": self
        }

    def plot(self, predict_results, plot_process=True, plot_dist=True):
        t, y = predict_results["t"], predict_results["y"]
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


class AgeGenderHPVModel2(AgeGenderModel):

    """
    与1相比的区别：
        1. 将男性的患病者（D）去掉
        2. 所有的转换使用单个参数来描述，不再是比例和速率的组合
        3. 不再使用sparse matrix来计算，这样更加容易维护
        4. 加入了对于rho（性接触矩阵）的参数
    """

    def __init__(
        self,
        # 是否计算累计量，用于经济学评价
        cal_cumulate=True,
        # -- 传染病模型参数 --
        # 女性接触HPV性伴侣被传染的概率  0.846 0.25
        epsilon_f=0.846,
        # 男性接触HPV性伴侣被传染的概率  0.913 0.25
        epsilon_m=0.913,
        # 女性每年性伴侣数量
        omega_f=np.array([
            0, 0, 0, 0, 0, 0, 0.11, 0.11, 1.36, 1.67, 1.65, 1.40, 1.16, 1.13,
            1.06, 1.02, 0.96, 0.93, 0.83, 0.63, 0, 0, 0, 0, 0, 0
        ]) * 0.457,
        # 男性每年性伴侣数量
        omega_m=np.array([
            0, 0, 0, 0, 0, 0, 0.04, 0.04, 0.49, 1.02, 1.20, 1.43, 1.32, 1.19,
            1.20, 1.08, 1.09, 0.91, 0.85, 0.74, 0, 0, 0, 0, 0, 0
        ]) * 0.457,
        # 性接触矩阵的容许年龄范围
        partner_window=10,
        # 性接触矩阵，每隔一个年龄段的递降幅度
        partner_decline=0.05,
        # 性接触矩阵，性活动界限
        partner_interval=(13, 60),

        # 自然抗体丢失速率 (未知)，假设是3-5年
        phi=1/3,
        # 疫苗覆盖率 (未知，待设置)
        psi=np.array([0] * 26),
        # 疫苗有效率 (2/4: 0.691, 9: 0.921)
        tau=0.691,

        # I->P
        beta_I=0.15,
        # P->LC
        beta_P=0.13,
        # LC->RC
        beta_LC=0.10,
        # RC->DC
        beta_RC=0.30,

        # Local cancer死亡率
        dL=np.array(
            [0]*8+[0.7]*6+[0.6]*2+[0.8]*2+[1.9]*2+[4.2]*2+[11.6]*4
        ) / 100 * 0.01,
        # Region cancer死亡率
        dR=np.array(
            [0]*8+[13.4]*6+[8.9]*2+[11.0]*2+[10.1]*2+[17.6]*2+[28.6]*4
        ) / 100 * 0.01,
        # Distant cancer死亡率
        dD=np.array(
            [0]*8+[42.9]*6+[41.0]*2+[46.7]*2+[52.7]*2+[54.6]*2+[70.3]*4
        ) / 100 * 0.01,

        # I->R
        gamma_I=0.35,
        # P->R
        gamma_P=0.7*(12/18),
        # LC->R
        gamma_LC=0.0,
        # RC->R
        gamma_RC=0.0,
        # DC->R
        gamma_DC=0.0,
        # -- 人口学模型参数 --
        # 年龄分组，第一个和最后一个必须分别是0和np.inf
        agebins=np.array([
            0, 1, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 25, 27, 30, 35, 40,
            45, 50, 55, 60, 65, 70, 75, 80, 85, np.inf
        ]),
        # 每个年龄组的生育力，基于女性计算
        fertilities=np.array(
            [0] * 8 +
            [4.61, 18.62, 29.73, 74.39, 112.60,
             106.95, 68.57, 26.08, 5.23, 0.44] +
            [0] * 8
        ) / 1000,
        # 女性每个年龄组的死亡率
        deathes_female=np.array([
            6.68, 0.21, 0., 0.48, 0.41, 0.13, 0., 0.61, 0., 0.12, 0.67, 0.36,
            0.50, 0.27, 0.26, 0.57, 0.75, 1.09, 1.79, 2.64, 5.97, 9.51, 18.25,
            27.70, 61.45, 118.39
        ]) / 1000,
        # 男性每个年龄组的死亡率
        deathes_male=np.array([
            8.49, 0.31, 0.15, 0.66, 0.76, 0.14, 0.41, 0.14, 0.61, 0.71, 0., 0.57,
            0.92, 0.55, 0.87, 1.51, 1.63, 2.38, 4.71, 6.31, 11.82, 17.52, 28.45,
            44.76, 90.01, 160.64
        ]) / 1000,
        # 每个年龄组的死亡率
        # deathes=np.array([
        #     7.64, 0.52, 0.08, 0.58, 0.60, 0.14, 0.22, 0.35, 0.33, 0.44, 0.35,
        #     0.47, 0.72, 0.42, 0.57, 1.05, 1.20, 1.74, 3.26, 4.49, 8.91, 13.44,
        #     23.21, 35.70, 73.98, 135.09
        # ]) / 1000,
        # 出生婴儿女婴占比
        lambda_f=0.4681,
        # 出生婴儿男婴占比
        lambda_m=0.5319,
        # 初始女性总人口数量
        total0_f=500000,
        # 初始男性总人口数量
        total0_m=500000,
        # 是否直接设置q为0，或者利用生育力和死亡率计算q
        q_is_zero=True,
        rtol=1e-5,
        atol=1e-5,
        verbose=True,
        vacc_prefer=False,
        **kwargs
    ):

        super().__init__(
            agebins=agebins, fertilities=fertilities,
            deathes_female=deathes_female, deathes_male=deathes_male,
            lambda_f=lambda_f, lambda_m=lambda_m,
            total0_f=total0_f, total0_m=total0_m,
            q_is_zero=q_is_zero, rtol=rtol, atol=atol, verbose=verbose,

            epsilon_f=epsilon_f, epsilon_m=epsilon_m,
            omega_f=omega_f, omega_m=omega_m,
            partner_window=partner_window, partner_decline=partner_decline,
            partner_interval=partner_interval,
            phi=phi, psi=psi, tau=tau, beta_I=beta_I, beta_P=beta_P,
            beta_LC=beta_LC, beta_RC=beta_RC,
            dL=dL, dR=dR, dD=dD, gamma_I=gamma_I, gamma_P=gamma_P,
            gamma_LC=gamma_LC, gamma_RC=gamma_RC, gamma_DC=gamma_DC,

            cal_cumulate=cal_cumulate, vacc_prefer=vacc_prefer,
            **kwargs
        )

        # 一些数量设置
        self.nrooms = 12
        self.nrooms_f = 8
        self.nrooms_m = 4
        self.ndim = self.nrooms * self.nages

        self.c_f_ = self.c_f[:-1]
        self.c_m_ = self.c_m[:-1]

        self.dcq_f = self.dc_f + self.q
        self.dcq_m = self.dc_m + self.q
        self.rho = compute_rho(self.agebins,
                               self.partner_window,
                               self.partner_decline,
                               100, self.partner_interval)

    def df_dt(self, t, X):
        if self.cal_cumulate:
            X = X[:self.ndim]
        Xre = X.reshape(self.nrooms, self.nages)
        Ntf = Xre[:self.nrooms_f, :].sum(axis=0)
        Ntm = Xre[self.nrooms_f:, :].sum(axis=0)
        Sf, If, Pf, LC, RC, DC, Rf, Vf, Sm, Im, Pm, Rm = Xre

        # 计算一下alpha（感染率）
        iPf = (If + Pf) / Ntf
        iPm = (Im + Pm) / Ntm
        alpha_f = self.epsilon_f * self.omega_f * np.dot(self.rho, iPm)
        alpha_m = self.epsilon_m * self.omega_m * np.dot(self.rho, iPf)

        # 计算一下接种强度
        if isinstance(self.psi, (np.ndarray, float)):
            psi = self.psi
        elif isfunction(self.psi):
            psi = self.psi(t)
        else:
            raise NotImplementedError

        # 计算一下出生人口
        born = np.dot(Ntf, self.fertilities)
        born_f, born_m = born * self.lambda_f, born * self.lambda_m

        # 根据是否疫苗优先，计算Sf的移除部分
        if self.vacc_prefer:
            alpha_f = alpha_f * (1 - psi)

        dSf = self.phi*Rf-(alpha_f+psi*self.tau+self.dcq_f)*Sf
        dSf[0] += born_f
        dSf[1:] += Sf[:-1] * self.c_f_

        dIf = alpha_f*Sf-(self.beta_I+self.gamma_I+self.dcq_f)*If
        dIf[1:] += If[:-1] * self.c_f_

        dPf = self.beta_I*If-(self.beta_P+self.gamma_P+self.dcq_f)*Pf
        dPf[1:] += Pf[:-1] * self.c_f_

        dLC = self.beta_P*Pf-(self.beta_LC+self.gamma_LC+self.dL+self.dcq_f)*LC
        dLC[1:] += LC[:-1] * self.c_f_

        dRC = self.beta_LC*LC-(self.beta_RC+self.gamma_RC+self.dR+self.dcq_f)*RC
        dRC[1:] += RC[:-1] * self.c_f_

        dDC = self.beta_RC*RC-(self.gamma_DC+self.dD+self.dcq_f)*DC
        dDC[1:] += DC[:-1] * self.c_f_

        dRf = self.gamma_I*If+self.gamma_P*Pf+self.gamma_LC*LC+\
            self.gamma_RC*RC+self.gamma_DC*DC-(self.phi+self.dcq_f)*Rf
        dRf[1:] += Rf[:-1] * self.c_f_

        dVf = psi*self.tau*Sf-self.dcq_f*Vf
        dVf[1:] += Vf[:-1] * self.c_f_

        dSm = self.phi*Rm-(alpha_m+self.dcq_m)*Sm
        dSm[0] += born_m
        dSm[1:] += Sm[:-1] * self.c_m_

        dIm = alpha_m*Sm-(self.beta_I+self.gamma_I+self.dcq_m)*Im
        dIm[1:] += Im[:-1] * self.c_m_

        dPm = self.beta_I*Im-(self.beta_P+self.gamma_P+self.dcq_m)*Pm
        dPm[1:] += Pm[:-1] * self.c_m_

        dRm = self.gamma_I*Im+self.gamma_P*Pm-(self.phi+self.dcq_f)*Rm
        dRm[1:] += Rm[:-1] * self.c_m_

        res = np.concatenate([dSf, dIf, dPf, dLC, dRC, dDC, dRf, dVf,
                              dSm, dIm, dPm, dRm])

        if self.cal_cumulate:
            cR2S_f = self.phi*Rf
            cS2I_f = alpha_f*Sf
            cI2P_f = self.beta_I*If
            cP2LC_f = self.beta_P*Pf
            cLC2RC_f = self.beta_LC*LC
            cRC2DC_f = self.beta_RC*RC
            cLC2d_f = self.dL*LC
            cRC2d_f = self.dR*RC
            cDC2d_f = self.dD*DC
            cS2V_f = psi*self.tau*Sf
            cres = np.concatenate([
                cR2S_f, cS2I_f, cI2P_f, cP2LC_f, cLC2RC_f, cRC2DC_f,
                cLC2d_f, cRC2d_f, cDC2d_f, cS2V_f
            ])
            res = np.concatenate([res, cres])

        return res

    def _neat_predict_results(self, t, y):
        res = {"t": t, "model": self}

        if self.cal_cumulate:
            y, ycum = y[:, :self.ndim], y[:, self.ndim:]
            y = y.reshape(-1, self.nrooms, self.nages)
            ycum = ycum.reshape(-1, 10, self.nages)
            res["y"] = y
            res["ycum"] = ycum
            return res

        res["y"] = y.reshape(-1, self.nrooms, self.nages)
        return res

    def plot(self, predict_results, plot_process=True, plot_dist=True):
        fgs_y = self.plot_y(predict_results["t"], predict_results["y"],
                            plot_process, plot_dist)
        fgs_cum = self.plot_ycum(predict_results["t"], predict_results["ycum"],
                                 plot_process, plot_dist)

        fgs_y.update(fgs_cum)
        return fgs_y

    def plot_y(self, t, y, plot_process=True, plot_dist=True):
        rooms = np.array([
            "Sf", "If", "Pf", "LC", "RC", "DC", "Rf", "Vf",
            "Sm", "Im", "Pm", "Rm"
        ])
        nt, nrooms, nages = y.shape
        df = pd.DataFrame({
            "y": y.flatten(),
            "t": np.repeat(t, nrooms*nages),
            "room": np.tile(np.repeat(rooms, nages), nt),
            "age": pd.Categorical(
                np.tile(self.agebin_names, nt*nrooms),
                categories=self.agebin_names, ordered=True
            )
        })
        fgs = {}
        if plot_process:
            fgs["y_process"] = sns.relplot(
                data=df, x="t", y="y", hue="age",
                col="room", col_wrap=3, aspect=1, kind="line",
                facet_kws={"sharey": False}
            )
        if plot_dist:
            fgs["y_dist"] = sns.relplot(
                data=df, x="age", y="y", hue="t",
                col="room", col_wrap=3, aspect=1.5, kind="line",
                facet_kws={"sharey": False}
            )
            fgs["y_dist"].set_xticklabels(rotation=45)
        for fg in fgs.values():
            fg.set_titles("{col_name}")
        return fgs

    def plot_ycum(self, t, ycum, plot_process=True, plot_dist=True):
        rooms = np.array([
            "Recovery", "Infected", "Persisted",
            "Localized Cancer", "Regional Cancer",
            "Distant Cancer", "Deathed LC", "Deathed RC", "Deathed DC",
            "Vaccined"
        ])
        nt, nrooms, nages = ycum.shape
        df = pd.DataFrame({
            "y": ycum.flatten(),
            "t": np.repeat(t, nrooms*nages),
            "room": np.tile(np.repeat(rooms, nages), nt),
            "age": pd.Categorical(
                np.tile(self.agebin_names, nt*nrooms),
                categories=self.agebin_names, ordered=True
            )
        })
        fgs = {}
        if plot_process:
            fgs["ycum_process"] = sns.relplot(
                data=df, x="t", y="y", hue="age",
                col="room", col_wrap=3, aspect=1, kind="line",
                facet_kws={"sharey": False}
            )
        if plot_dist:
            fgs["ycum_dist"] = sns.relplot(
                data=df, x="age", y="y", hue="t",
                col="room", col_wrap=3, aspect=1.5, kind="line",
                facet_kws={"sharey": False}
            )
            fgs["ycum_dist"].set_xticklabels(rotation=45)
        for fg in fgs.values():
            fg.set_titles("{col_name}")
        return fgs

    def get_init(self, room_propations):
        assert len(room_propations) == self.nrooms
        room_prop = np.array(room_propations)
        room_prop_f = room_prop[:self.nrooms_f]
        room_prop_m = room_prop[self.nrooms_f:]
        room_prop_f /= room_prop_f.sum()
        room_prop_m /= room_prop_m.sum()
        init_f = room_prop_f[:, None] * self.P_f
        init_m = room_prop_m[:, None] * self.P_m
        res = np.concatenate([init_f, init_m]).flatten()
        if self.cal_cumulate:
            res = np.concatenate([res, np.zeros(self.nages*10)])
        return res
