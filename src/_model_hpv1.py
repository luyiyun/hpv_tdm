import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_period_dtype
import seaborn as sns
from scipy import sparse as ssp

from ._sexual import compute_rho
from .utils import diagonal2coo
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
    # 自然抗体丢失速率 (未知)
    phi=1/5,  # 假设是3-5年
    # 疫苗覆盖率 (未知，待设置)
    psi=np.array([0] * 26),
    # 疫苗有效率 (2/4: 0.691, 9: 0.921)
    tau=0.691,
    # 持续感染者转换比 (0.1 - 0.2)
    beta_P=0.15,
    # 持续感染者->Local cancer转换比
    beta_LC=0.13,
    # Local cancer->Region cancer转换比
    beta_RC=0.10,
    # Region cancer->Distant cancer转换比
    beta_DC=0.30,
    # Local cancer死亡率
    dL=np.array(
        [0]*8+[0.7]*6+[0.6]*2+[0.8]*2+[1.9]*2+[4.2]*2+[11.6]*4)/100,
    # Region cancer死亡率
    dR=np.array(
        [0]*8+[13.4]*6+[8.9]*2+[11.0]*2+[10.1]*2+[17.6]*2+[28.6]*4)/100,
    # Distant cancer死亡率
    dD=np.array(
        [0]*8+[42.9]*6+[41.0]*2+[46.7]*2+[52.7]*2+[54.6]*2+[70.3]*4)/100,
    # 持续感染者->男性疾病转换比 (未知)
    beta_D=0.1,
    # 男性疾病死亡率 (未知)
    beta_dm=0.142,  # 选择了一个阴茎癌的死亡率
    # 普通感染者->持续感染者转换速率
    theta_I=1/2,
    # 持续感染者->Local cancer转换速率 (未知) 15-20年
    theta_P=1/15,
    # Local cancer->Region cancer转换速率 (未知)
    theta_LC=1/3,
    # Region cancer->Distant cancer转换速率 (未知)
    theta_RC=1/2,
    # Local cancer死亡速率 (未知)
    eta_L=1/15,
    # Region cancer死亡速率 (未知)
    eta_R=1/7,
    # Distant cancer死亡速率 (未知)
    eta_D=6.5/12,
    # 男性疾病死亡速率 (未知)
    theta_D=1/3,
    # 普通感染者恢复速率 (0.5 - 2)
    gamma_I=1.,
    # 持续感染者恢复速率 (2 - 4)
    gamma_P=1/3,
    # Local cancer恢复速率 (未知)
    gamma_LC=1.,
    # Region cancer恢复速率 (未知)
    gamma_RC=1/3.,
    # Distant cancer恢复速率 (未知)
    gamma_DC=1/5.,
    # 男性疾病恢复速率 (未知)
    gamma_D=1/3.,
)


class AgeGenderHPVModel1(AgeGenderModel):
    def __init__(self, **kwargs):

        base_kwargs = {}
        for name in base_dp.keys():
            if name in kwargs:
                base_kwargs[name] = kwargs[name]
        super().__init__(**base_kwargs)

        for key, value in defaults_parameters.items():
            if key in kwargs:
                value = kwargs[key]
            setattr(self, key, value)

        # 一些数量设置
        self.nrooms = 13
        self.nrooms_f = 8
        self.nrooms_m = 5
        self.ndim = self.nrooms * self.nages

        CP_f = self.c_f[:-1] #  * self.P_f[:-1] / self.P_f[1:]
        CP_m = self.c_m[:-1] #  * self.P_m[:-1] / self.P_m[1:]

        dcq_f = self.dc_f + self.q
        dcq_m = self.dc_m + self.q
        self.rho = compute_rho(self.agebins, 10, 0.05, 100, (13, 60))

        # 依次是：Sf, If, Pf, LCf, RCf, DCf, Rf, Vf, Sm, Im, Pm, Dm, Rm
        self._ii, self._jj, self._data = diagonal2coo([
            # Sf
            ((0, self.nages*6), self.nages, self.phi, None),
            ((0, 0), self.nages, -(self.psi*self.tau+dcq_f), CP_f),
            # If
            ((self.nages, self.nages), self.nages,
             -(self.beta_P*self.theta_I+(1-self.beta_P)*self.gamma_I+dcq_f),
             CP_f),
            # Pf
            ((self.nages*2, self.nages), self.nages,
             self.beta_P*self.theta_I, None),
            ((self.nages*2, self.nages*2), self.nages,
             -(self.beta_LC*self.theta_P+(1-self.beta_LC)*self.gamma_P+dcq_f),
             CP_f),
            # LC
            ((self.nages*3, self.nages*2), self.nages,
             self.beta_LC*self.theta_P, None),
            ((self.nages*3, self.nages*3), self.nages,
             -(self.beta_RC*self.theta_LC+self.dL*self.eta_L+
               (1-self.dL-self.beta_RC)*self.gamma_LC+dcq_f), CP_f),
            # RC
            ((self.nages*4, self.nages*3), self.nages,
             self.beta_RC*self.theta_LC, None),
            ((self.nages*4, self.nages*4), self.nages,
             -(self.beta_DC*self.theta_RC+self.dR*self.eta_R+
               (1-self.dR-self.beta_DC)*self.gamma_RC+dcq_f), CP_f),
            # DC
            ((self.nages*5, self.nages*4), self.nages,
             self.beta_DC*self.theta_RC, None),
            ((self.nages*5, self.nages*5), self.nages,
             -(self.dD*self.eta_D+(1-self.dD)*self.gamma_DC+dcq_f), CP_f),
            # Rf
            ((self.nages*6, self.nages), self.nages,
             (1-self.beta_P)*self.gamma_I, None),
            ((self.nages*6, self.nages*2), self.nages,
             (1-self.beta_LC)*self.gamma_P, None),
            ((self.nages*6, self.nages*3), self.nages,
             (1-self.beta_RC-self.dL)*self.gamma_LC, None),
            ((self.nages*6, self.nages*4), self.nages,
             (1-self.beta_DC-self.dR)*self.gamma_RC, None),
            ((self.nages*6, self.nages*5), self.nages,
             (1-self.dD)*self.gamma_DC, None),
            ((self.nages*6, self.nages*6), self.nages,
             -(self.phi+dcq_f), CP_f),
            # Vf
            ((self.nages*7, 0), self.nages, self.tau*self.psi, None),
            ((self.nages*7, self.nages*7), self.nages, -dcq_f, CP_f),
            # Sm
            ((self.nages*8, self.nages*12), self.nages, self.phi, None),
            ((self.nages*8, self.nages*8), self.nages, -dcq_m, CP_m),
            # Im
            ((self.nages*9, self.nages*9), self.nages,
             -(self.beta_P*self.theta_I+(1-self.beta_P)*self.gamma_I+dcq_m),
             CP_m),
            # Pm
            ((self.nages*10, self.nages*9), self.nages,
             self.beta_P*self.theta_I, None),
            ((self.nages*10, self.nages*10), self.nages,
             -(self.beta_D*self.theta_P+(1-self.beta_D)*self.gamma_P+dcq_m),
             CP_m),
            # Dm
            ((self.nages*11, self.nages*10), self.nages,
             self.beta_D*self.theta_P, None),
            ((self.nages*11, self.nages*11), self.nages,
             -(self.beta_dm*self.theta_D+(1-self.beta_dm)*self.gamma_D+dcq_m),
             CP_m),
            # Rm
            ((self.nages*12, self.nages*9), self.nages,
             (1-self.beta_P)*self.gamma_I, None),
            ((self.nages*12, self.nages*10), self.nages,
             (1-self.beta_D)*self.gamma_P, None),
            ((self.nages*12, self.nages*11), self.nages,
             (1-self.beta_dm)*self.gamma_D, None),
            ((self.nages*12, self.nages*12), self.nages,
             -(self.phi+dcq_m), CP_m),
        ])

    def df_dt(self, _, X):
        nages = self.nages
        Xre = X.reshape(self.nrooms, self.nages)
        Ntf = Xre[:self.nrooms_f, :].sum(axis=0)
        Ntm = Xre[self.nrooms_f:, :].sum(axis=0)
        iPf = Xre[1:3].sum(axis=0) / Ntf
        iPm = Xre[9:11].sum(axis=0) / Ntm
        alpha_f = self.epsilon_f * self.omega_f * np.dot(self.rho, iPm)
        alpha_m = self.epsilon_m * self.omega_m * np.dot(self.rho, iPf)
        ii, jj, data = diagonal2coo([
            ((0, 0), nages, -alpha_f, None),
            ((nages, 0), nages, alpha_f, None),
            ((nages*8, nages*8), nages, -alpha_m, None),
            ((nages*9, nages*8), nages, alpha_m, None)
        ])
        ii_new, jj_new, data_new = (
            np.concatenate([self._ii, ii]),
            np.concatenate([self._jj, jj]),
            np.concatenate([self._data, data])
        )
        # coo在构建的时候，重复index的元素会直接加上
        matrix = ssp.coo_matrix((data_new, (ii_new, jj_new)),
                                shape=(self.ndim, self.ndim)).tocsr()
        res = matrix.dot(X)
        # NOTE: 之前的做法是错的。。
        # 加上出生人口
        born = np.dot(Ntf, self.fertilities)
        born_f, born_m = born * self.lambda_f, born * self.lambda_m
        res[0] += born_f
        res[self.nages*self.nrooms_f] += born_m
        return res

    def _reshpae_out(self, y):
        new_y = np.stack([
            y[:, i:(i+self.nages)] for i in range(0, y.shape[1], self.nages)
        ], axis=1)
        return new_y

    def plot(self, t, y, plot_process=True, plot_dist=True):
        rooms = np.array([
            "Sf", "If", "Pf", "LC", "RC", "DC", "Rf",
            "Vf", "Sm", "Im", "Pm", "Dm", "Rm"
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
                col="room", col_wrap=4, aspect=1, kind="line",
                facet_kws={"sharey": False}
            )
        if plot_dist:
            fgs["dist"] = sns.relplot(
                data=df, x="age", y="y", hue="t",
                col="room", col_wrap=3, aspect=2, kind="line",
                facet_kws={"sharey": False}
            )
            fgs["dist"].set_xticklabels(rotation=45)
        for fg in fgs.values():
            fg.set_titles("{col_name}")
        return fgs

