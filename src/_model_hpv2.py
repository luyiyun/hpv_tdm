import numpy as np
import pandas as pd
import seaborn as sns

from ._sexual import compute_rho
from ._model_demographic import AgeGenderModel
from ._model_demographic import defaults_parameters as base_dp


defaults_parameters = dict(
    # 女性接触HPV性伴侣被传染的概率
    epsilon_f=0.846,
    # 男性接触HPV性伴侣被传染的概率
    epsilon_m=0.913,
    # 女性每年性伴侣数量
    omega_f=np.array([
        0, 0, 0, 0, 0, 0, 0.11, 0.11, 1.36, 1.67, 1.65, 1.40, 1.16, 1.13,
        1.06, 1.02, 0.96, 0.93, 0.83, 0.63, 0, 0, 0, 0, 0, 0
    ]),
    # 男性每年性伴侣数量
    omega_m=np.array([
        0, 0, 0, 0, 0, 0, 0.04, 0.04, 0.49, 1.02, 1.20, 1.43, 1.32, 1.19,
        1.20, 1.08, 1.09, 0.91, 0.85, 0.74, 0, 0, 0, 0, 0, 0
    ]),
    # 性接触矩阵的容许年龄范围
    partner_window=10,
    # 性接触矩阵，每隔一个年龄段的递降幅度
    partner_decline=0.05,
    # 性接触矩阵，性活动界限
    partner_interval=(13, 60),

    # 自然抗体丢失速率 (未知)
    phi=1/5,  # 假设是3-5年
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
        [0]*8+[0.7]*6+[0.6]*2+[0.8]*2+[1.9]*2+[4.2]*2+[11.6]*4)/100,
    # Region cancer死亡率
    dR=np.array(
        [0]*8+[13.4]*6+[8.9]*2+[11.0]*2+[10.1]*2+[17.6]*2+[28.6]*4)/100,
    # Distant cancer死亡率
    dD=np.array(
        [0]*8+[42.9]*6+[41.0]*2+[46.7]*2+[52.7]*2+[54.6]*2+[70.3]*4)/100,

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
)


class AgeGenderHPVModel2(AgeGenderModel):

    """
    与1相比的区别：
        1. 将男性的患病者（D）去掉
        2. 所有的转换使用单个参数来描述，不再是比例和速率的组合
        3. 不再使用sparse matrix来计算，这样更加容易维护
        4. 加入了对于rho（性接触矩阵）的参数
    """

    def __init__(self, cal_cumulate=False, **kwargs):

        base_kwargs = {}
        for name in base_dp.keys():
            if name in kwargs:
                base_kwargs[name] = kwargs[name]
        super().__init__(**base_kwargs)

        for key, value in defaults_parameters.items():
            if key in kwargs:
                value = kwargs[key]
            setattr(self, key, value)

        self.cal_cumulate = cal_cumulate

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

    def df_dt(self, _, X):
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

        # 计算一下出生人口
        born = np.dot(Ntf, self.fertilities)
        born_f, born_m = born * self.lambda_f, born * self.lambda_m

        dSf = self.phi*Rf-(alpha_f+self.psi*self.tau+self.dcq_f)*Sf
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

        dVf = self.psi*self.tau*Sf-self.dcq_f*Vf
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
            cS2V_f = self.psi*self.tau*Sf
            cres = np.concatenate([
                cR2S_f, cS2I_f, cI2P_f, cP2LC_f, cLC2RC_f, cRC2DC_f,
                cLC2d_f, cRC2d_f, cDC2d_f, cS2V_f
            ])
            res = np.concatenate([res, cres])

        return res

    def _reshpae_out(self, y):
        if self.cal_cumulate:
            y, ycum = y[:, :self.ndim], y[:, self.ndim:]
            y = y.reshape(-1, self.nrooms, self.nages)
            ycum = ycum.reshape(-1, 10, self.nages)
            return y, ycum
        return y.reshape(-1, self.nrooms, self.nages)

    def plot(self, t, y, plot_process=True, plot_dist=True):
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
            fgs["process"] = sns.relplot(
                data=df, x="t", y="y", hue="age",
                col="room", col_wrap=3, aspect=1, kind="line",
                facet_kws={"sharey": False}
            )
        if plot_dist:
            fgs["dist"] = sns.relplot(
                data=df, x="age", y="y", hue="t",
                col="room", col_wrap=3, aspect=1.5, kind="line",
                facet_kws={"sharey": False}
            )
            fgs["dist"].set_xticklabels(rotation=45)
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

    def plot_cumulative(self, t, ycum, plot_process=True, plot_dist=True):
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
            fgs["process"] = sns.relplot(
                data=df, x="t", y="y", hue="age",
                col="room", col_wrap=3, aspect=1, kind="line",
                facet_kws={"sharey": False}
            )
        if plot_dist:
            fgs["dist"] = sns.relplot(
                data=df, x="age", y="y", hue="t",
                col="room", col_wrap=3, aspect=1.5, kind="line",
                facet_kws={"sharey": False}
            )
            fgs["dist"].set_xticklabels(rotation=45)
        for fg in fgs.values():
            fg.set_titles("{col_name}")
        return fgs

