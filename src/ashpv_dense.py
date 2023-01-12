import numpy as np
from scipy.integrate import solve_ivp
from scipy import sparse as ssp

from ._demographic import compute_c, find_q, compute_P
from ._sexual import compute_rho


def AddDiagnol(mat, lf, n, major, minor=None):
    x0, x1 = lf
    i0, i1 = list(range(x0, x0+n)), list(range(x1, x1+n))
    mat[i0, i1] = major
    if minor is not None:
        i0, i1 = list(range(x0+1, x0+n)), list(range(x0, x0+n-1))
        mat[i0, i1] = minor


class DenseAgeStructedHPVModel:
    def __init__(
        self,
        agebins,  # 年龄分组，第一个和最后一个必须分别是0和np.inf
        fertilities,  # 每个年龄组的生育力
        deathes,  # 每个年龄组的死亡率
        epsilon_f,  # 女性接触HPV性伴侣被传染的概率
        epsilon_m,  # 男性接触HPV性伴侣被传染的概率
        omega_f,  # 女性每年性伴侣数量
        omega_m,  # 男性每年性伴侣数量
        phi,  # 自然抗体丢失速率
        psi,  # 疫苗覆盖率
        tau,  # 疫苗有效率
        beta_P,  # 持续感染者转换比
        beta_LC,  # Local cancer转换比
        beta_RC,  # Region cancer转换比
        beta_DC,  # Distant cancer转换比
        beta_D,  # 男性疾病转换比
        beta_dm,  # 男性疾病死亡率
        dL,  # Local cancer死亡率
        dR,  # Region cancer死亡率
        dD,  # Distant cancer死亡率
        theta_I,  # 普通感染者->持续感染者转换速率
        theta_P,  # 持续感染者->Local cancer转换速率
        theta_LC,  # Local cancer->Region cancer转换速率
        theta_RC,  # Region cancer->Distant cancer转换速率
        # theta_P,  # 持续感染者->男性疾病转换速率
        eta_L,  # Local cancer死亡速率
        eta_R,  # Region cancer死亡速率
        eta_D,  # Distant cancer死亡速率
        theta_D,  # 男性疾病死亡速率
        gamma_I,  # 普通感染者恢复速率
        gamma_P,  # 持续感染者恢复速率
        gamma_LC,  # Local cancer恢复速率
        gamma_RC,  # Region cancer恢复速率
        gamma_DC,  # Distant cancer恢复速率
        gamma_D,   # 男性疾病恢复速率
        total0,  # 初始总人口数量
        q_is_zero=False,  # 是否直接设置q为0，或者利用生育力和死亡率计算q
        rtol=1e-8,
        atol=1e-8
    ):
        nages = len(agebins) - 1
        nrooms, nrooms_female = 13, 8
        n = nages * nrooms
        self.ndim = n
        self.rtol, self.atol = rtol, atol

        assert np.isclose(agebins[0], 0.)
        assert np.isclose(agebins[-1], np.inf)
        assert len(fertilities) == nages
        assert len(deathes) == nages

        self.nrooms = nrooms

        self.agebins = agebins
        self.fertilities = fertilities
        self.deathes = deathes
        self.epsilon_f = epsilon_f
        self.epsilon_m = epsilon_m
        self.omega_f = omega_f
        self.omega_m = omega_m
        self.phi = phi
        self.psi = psi
        self.tau = tau
        # 转移概率
        self.beta_P = beta_P
        self.beta_LC = beta_LC
        self.beta_RC = beta_RC
        self.beta_DC = beta_DC
        self.beta_D = beta_D
        self.beta_dm = beta_dm
        self.dL = dL
        self.dR = dR
        self.dD = dD
        # 转换率
        self.theta_I = theta_I
        self.theta_P = theta_P
        self.theta_LC = theta_LC
        self.theta_RC = theta_RC
        self.theta_D = theta_D
        self.eta_L = eta_L
        self.eta_R = eta_R
        self.eta_D = eta_D
        # 恢复率
        self.gamma_I = gamma_I
        self.gamma_P = gamma_P
        self.gamma_LC = gamma_LC
        self.gamma_RC = gamma_RC
        self.gamma_DC = gamma_DC
        self.gamma_D = gamma_D
        self.total0 = total0
        self.q_is_zero = q_is_zero,

        self.agedelta = self.agebins[1:] - self.agebins[:-1]
        # TODO: q=0时需要对出生率进行调整
        self.q = 0. if q_is_zero else find_q(self.fertilities,
                                             self.deathes,
                                             self.agedelta)[0]
        self.c = compute_c(self.deathes, self.q, self.agedelta)
        self.P = compute_P(total0, self.deathes, self.q, self.c)
        CP = self.c[:-1] * self.P[:-1] / self.P[1:]
        born = 0.5 * self.P.sum() / self.P[0]
        dcq = deathes + self.c + self.q
        self.rho = compute_rho(agebins, 10, 0.05, 100, (13, 60))
        self.nages = nages

        # TODO: 男女使用相同的死亡率？？？
        # 依次是：Sf, If, Pf, LCf, RCf, DCf, Rf, Vf, Sm, Im, Pm, Dm, Rm
        self._matrix = np.zeros((n, n), dtype=float)
        # 出生人数
        self._matrix[0, 0] += born
        self._matrix[nages * nrooms_female, 0] += born
        # Sf
        AddDiagnol(self._matrix, (0, nages*6), nages, phi)
        AddDiagnol(self._matrix, (0, 0), nages, -(psi*tau+dcq), CP)  # 缺了一项
        # If
        # 缺了一项
        AddDiagnol(self._matrix, (nages, nages), nages,
                   -(beta_P*theta_I+(1-beta_P)*gamma_I+dcq), CP)
        # Pf
        AddDiagnol(self._matrix, (nages*2, nages), nages, beta_P*theta_I)
        # temp = beta_LC * theta_P1 + beta_RC * theta_P2 + beta_DC * theta_P3
        # temp += (1 - beta_LC - beta_RC - beta_DC) * gamma_P
        AddDiagnol(self._matrix, (nages*2, nages*2), nages,
                   -(beta_LC*theta_P+(1-beta_LC)*gamma_P+dcq), CP)
        # LCf
        AddDiagnol(self._matrix, (nages*3, nages*2), nages, beta_LC*theta_P)
        AddDiagnol(
            self._matrix, (nages*3, nages*3), nages,
            -(beta_RC*theta_LC+dL*eta_L+(1-dL-beta_RC)*gamma_LC+dcq), CP
        )
        # RCf
        AddDiagnol(self._matrix, (nages*4, nages*3), nages, beta_RC*theta_LC)
        AddDiagnol(
            self._matrix, (nages*4, nages*4), nages,
            -(beta_DC*theta_RC+dR*eta_R+(1-dR-beta_DC)*gamma_RC+dcq), CP
        )
        # DCf
        AddDiagnol(self._matrix, (nages*5, nages*4), nages, beta_DC*theta_RC)
        AddDiagnol(
            self._matrix, (nages*5, nages*5), nages,
            -(dD*eta_D+(1-dD)*gamma_DC+dcq), CP
        )
        # Rf
        AddDiagnol(self._matrix, (nages*6, nages), nages, (1-beta_P)*gamma_I)
        AddDiagnol(self._matrix, (nages*6, nages*2), nages,
                   (1-beta_LC)*gamma_P)
        AddDiagnol(self._matrix, (nages*6, nages*3), nages,
                   (1-beta_RC-dL)*gamma_LC)
        AddDiagnol(self._matrix, (nages*6, nages*4), nages,
                   (1-beta_DC-dR)*gamma_RC)
        AddDiagnol(self._matrix, (nages*6, nages*5), nages,
                   (1-dD)*gamma_DC)
        AddDiagnol(self._matrix, (nages*6, nages*6), nages, -(phi+dcq), CP)
        # Vf
        AddDiagnol(self._matrix, (nages*7, 0), nages, tau*psi)
        AddDiagnol(self._matrix, (nages*7, nages*7), nages, -dcq, CP)
        # Sm
        AddDiagnol(self._matrix, (nages*8, nages*12), nages, phi)
        AddDiagnol(self._matrix, (nages*8, nages*8), nages, -dcq, CP)  # 缺了一项
        # Im
        # 缺了一项
        AddDiagnol(self._matrix, (nages*9, nages*9), nages,
                   -(beta_P*theta_I+(1-beta_P)*gamma_I+dcq), CP)
        # Pm
        AddDiagnol(self._matrix, (nages*10, nages*9), nages, beta_P*theta_I)
        AddDiagnol(self._matrix, (nages*10, nages*10), nages,
                   -(beta_D*theta_P+(1-beta_D)*gamma_P+dcq), CP)
        # Dm
        AddDiagnol(self._matrix, (nages*11, nages*10), nages, beta_D*theta_P)
        AddDiagnol(self._matrix, (nages*11, nages*11), nages,
                   -(beta_dm*theta_D+(1-beta_dm)*gamma_D+dcq), CP)
        # Rm
        AddDiagnol(self._matrix, (nages*12, nages*9), nages,
                   (1-beta_P)*gamma_I)
        AddDiagnol(self._matrix, (nages*12, nages*10), nages,
                   (1-beta_D)*gamma_P)
        AddDiagnol(self._matrix, (nages*12, nages*11), nages,
                   (1-beta_dm)*gamma_D)
        AddDiagnol(self._matrix, (nages*12, nages*12), nages, -(phi+dcq), CP)

    def df_dt(self, t, X):
        nages = self.nages
        alpha_f = self.epsilon_f * self.omega_f * \
            np.matmul(self.rho, X[nages:(nages*2)] + X[(nages*2):(nages*3)])
        alpha_m = self.epsilon_m * self.omega_m * \
            np.matmul(self.rho,
                      X[(nages*9):(nages*10)] + X[(nages*10):(nages*11)])

        matrix = self._matrix.copy()
        AddDiagnol(matrix, (0, 0), nages, -alpha_f)
        AddDiagnol(matrix, (nages, 0), nages, -alpha_f)
        AddDiagnol(matrix, (nages*8, nages*8), nages, -alpha_m)
        AddDiagnol(matrix, (nages*9, nages*8), nages, -alpha_m)
        matrix = ssp.csr_matrix(matrix)
        return matrix.dot(X)

    def predict(self, t_span, t_eval=None, init=None):
        if init is None:
            init = np.zeros(self.ndim)
            Pprop = self.P / self.total0
            init[0:self.nages] = 0.45 * Pprop
            init[self.nages:(self.nages*2)] = 0.05*Pprop
            init[(self.nages*8):(self.nages*9)] = 0.45*Pprop
            init[(self.nages*9):(self.nages*10)] = 0.05*Pprop
        res = solve_ivp(
            self.df_dt, t_span=t_span, y0=init,
            t_eval=t_eval, rtol=self.rtol, atol=self.atol
        )
        t = res.t
        y = []
        for i in range(0, self.ndim, self.nages):
            y.append(res.y[i:(i+self.nages), :].T)
        yp = np.stack(y, axis=1)  # nt x nrooms x nages
        Nt = self.P * np.exp(self.q*t)[:, None]
        yn = yp * Nt[:, None]
        return t, yp, yn
