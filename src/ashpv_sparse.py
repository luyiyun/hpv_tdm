import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy import sparse as ssp
from tqdm import tqdm

from ._demographic import compute_c, compute_P, find_q_newton
from ._sexual import compute_rho


def construct_coo_component(args):
    ii, jj, data = [], [], []
    for (x0, x1), n, major, minor in args:
        i0, i1 = np.arange(x0, x0+n), np.arange(x1, x1+n)
        ii.append(i0)
        jj.append(i1)
        if isinstance(major, (float, int)):
            major = np.full((n,), major)
        data.append(major)
        if minor is not None:
            i0, i1 = np.arange(x0+1, x0+n), np.arange(x0, x0+n-1)
            ii.append(i0)
            jj.append(i1)
            if isinstance(minor, (float, int)):
                minor = np.full((n,), minor)
            data.append(minor)
    ii = np.concatenate(ii)
    jj = np.concatenate(jj)
    data = np.concatenate(data)
    return ii, jj, data


class SparseAgeStructedHPVModel:
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
        rtol=1e-5,
        atol=1e-5
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
        # 因为我们输入的是女性生育力，而我们面向的是全人群（包括男性）
        self.fertilities = fertilities / 2
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
        # TODO: 我们是不是需要将cancer和disease所导致的死亡也加入到deathes中
        # TODO: 为什么生育率减半后，q变成负值了，也就是人在减少？？
        #   或者，创建一个因疾病死亡的仓室，来储存这些死亡的人数
        self.q = 0. if q_is_zero else find_q_newton(self.fertilities,
                                                    self.deathes,
                                                    self.agedelta)[0]
        self.c = compute_c(self.deathes, self.q, self.agedelta)
        self.P = compute_P(total0, self.deathes, self.q, self.c)
        CP = self.c[:-1] * self.P[:-1] / self.P[1:]
        born = 0.5 * (self.P * self.fertilities).sum() / self.P[0]
        dcq = deathes + self.c + self.q
        self.rho = compute_rho(agebins, 10, 0.05, 100, (13, 60))
        self.nages = nages

        # TODO: 男女使用相同的死亡率？？？
        # 依次是：Sf, If, Pf, LCf, RCf, DCf, Rf, Vf, Sm, Im, Pm, Dm, Rm
        # self._matrix = np.zeros((n, n), dtype=float)
        # 出生人数
        self._ii, self._jj, self._data = construct_coo_component([
            # Sf
            ((0, nages*6), nages, phi, None),
            ((0, 0), nages, -(psi*tau+dcq), CP),
            # If
            ((nages, nages), nages,
             -(beta_P*theta_I+(1-beta_P)*gamma_I+dcq), CP),
            # Pf
            ((nages*2, nages), nages, beta_P*theta_I, None),
            ((nages*2, nages*2), nages,
             -(beta_LC*theta_P+(1-beta_LC)*gamma_P+dcq), CP),
            # LC
            ((nages*3, nages*2), nages, beta_LC*theta_P, None),
            ((nages*3, nages*3), nages,
             -(beta_RC*theta_LC+dL*eta_L+(1-dL-beta_RC)*gamma_LC+dcq), CP),
            # RC
            ((nages*4, nages*3), nages, beta_RC*theta_LC, None),
            ((nages*4, nages*4), nages,
             -(beta_DC*theta_RC+dR*eta_R+(1-dR-beta_DC)*gamma_RC+dcq), CP),
            # DC
            ((nages*5, nages*4), nages, beta_DC*theta_RC, None),
            ((nages*5, nages*5), nages,
             -(dD*eta_D+(1-dD)*gamma_DC+dcq), CP),
            # Rf
            ((nages*6, nages), nages, (1-beta_P)*gamma_I, None),
            ((nages*6, nages*2), nages, (1-beta_LC)*gamma_P, None),
            ((nages*6, nages*3), nages, (1-beta_RC-dL)*gamma_LC, None),
            ((nages*6, nages*4), nages, (1-beta_DC-dR)*gamma_RC, None),
            ((nages*6, nages*5), nages, (1-dD)*gamma_DC, None),
            ((nages*6, nages*6), nages, -(phi+dcq), CP),
            # Vf
            ((nages*7, 0), nages, tau*psi, None),
            ((nages*7, nages*7), nages, -dcq, CP),
            # Sm
            ((nages*8, nages*12), nages, phi, None),
            ((nages*8, nages*8), nages, -dcq, CP),
            # Im
            ((nages*9, nages*9), nages,
             -(beta_P*theta_I+(1-beta_P)*gamma_I+dcq), CP),
            # Pm
            ((nages*10, nages*9), nages, beta_P*theta_I, None),
            ((nages*10, nages*10), nages,
             -(beta_D*theta_P+(1-beta_D)*gamma_P+dcq), CP),
            # Dm
            ((nages*11, nages*10), nages, beta_D*theta_P, None),
            ((nages*11, nages*11), nages,
             -(beta_dm*theta_D+(1-beta_dm)*gamma_D+dcq), CP),
            # Rm
            ((nages*12, nages*9), nages, (1-beta_P)*gamma_I, None),
            ((nages*12, nages*10), nages, (1-beta_D)*gamma_P, None),
            ((nages*12, nages*11), nages, (1-beta_dm)*gamma_D, None),
            ((nages*12, nages*12), nages, -(phi+dcq), CP),
        ])
        self._ii = np.r_[self._ii, 0, nages * nrooms_female]
        self._jj = np.r_[self._jj, 0, 0]
        self._data = np.r_[self._data, born, born]

    def df_dt(self, t, X):
        nages = self.nages
        alpha_f = self.epsilon_f * self.omega_f * \
            np.matmul(self.rho, X[nages:(nages*2)] + X[(nages*2):(nages*3)])
        alpha_m = self.epsilon_m * self.omega_m * \
            np.matmul(self.rho,
                      X[(nages*9):(nages*10)] + X[(nages*10):(nages*11)])
        ii, jj, data = construct_coo_component([
            ((0, 0), nages, -alpha_f, None),
            ((nages, 0), nages, -alpha_f, None),
            ((nages*8, nages*8), nages, -alpha_m, None),
            ((nages*9, nages*8), nages, -alpha_m, None)
        ])
        ii_new, jj_new, data_new = (
            np.concatenate([self._ii, ii]),
            np.concatenate([self._jj, jj]),
            np.concatenate([self._data, data])
        )
        matrix = ssp.coo_matrix((data_new, (ii_new, jj_new)),
                                shape=(self.ndim, self.ndim)).tocsr()
        return matrix.dot(X)

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

    def predict(self, t_span, t_eval=None, init=None, backend="solve_ivp"):
        assert backend in ["solve_ivp", "odeint"]
        if init is None:
            init = np.zeros(self.ndim)
            Pprop = self.P / self.total0
            init[0:self.nages] = 0.45 * Pprop
            init[self.nages:(self.nages*2)] = 0.05*Pprop
            init[(self.nages*8):(self.nages*9)] = 0.45*Pprop
            init[(self.nages*9):(self.nages*10)] = 0.05*Pprop

        if backend == "solve_ivp":
            with tqdm(total=1000, unit="‰") as pbar:
                res = solve_ivp(
                    self.df_dt_pbar, t_span=t_span, y0=init,
                    t_eval=t_eval, rtol=self.rtol, atol=self.atol,
                    args=[pbar, [t_span[0], (t_span[1]-t_span[0]) / 1000]]
                )
            t = res.t
            y = res.y.T

        if backend == "odeint":
            if t_eval is None:
                t_eval = np.linspace(*t_span, num=1000)
            y, infodict = odeint(
                self.df_dt, init, t=t_eval,
                tfirst=True, rtol=self.rtol, atol=self.atol, full_output=True,
                printmessg=True
            )
            t = t_eval

        y_ = []
        for i in range(0, self.ndim, self.nages):
            y_.append(y[:, i:(i+self.nages)])
        yp = np.stack(y_, axis=1)  # nt x nrooms x nages
        Nt = self.P * np.exp(self.q*t)[:, None]
        yn = yp * Nt[:, None]
        return t, yp, yn
