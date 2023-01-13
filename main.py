import os
import os.path as osp
import logging

import numpy as np

# from src.ashpv_dense import DenseAgeStructedHPVModel
from src.ashpv_sparse import SparseAgeStructedHPVModel
from src import utils as U


logging.basicConfig(level=logging.INFO,
                    format="[%(levelname)s][%(asctime)s]%(message)s")


args = dict(
    # 年龄分组，第一个和最后一个必须分别是0和np.inf
    agebins=np.array([
        0, 1, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 25, 27, 30,
        35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, np.inf
    ]),
    # 每个年龄组的生育率
    fertilities=np.array(
        [0] * 8 +
        [4.61, 18.62, 29.73, 74.39, 112.60, 106.95, 68.57, 26.08,
            5.23, 0.44] +
        [0] * 8
    ) / 1000,
    # 每个年龄组的死亡率（总）
    # deathes=np.array([
    #     7.64, 0.52, 0.08, 0.58, 0.60, 0.14, 0.22, 0.35, 0.33, 0.44, 0.35,
    #     0.47, 0.72, 0.42, 0.57, 1.05, 1.20, 1.74, 3.26, 4.49, 8.91, 13.44,
    #     23.21, 35.70, 73.98, 135.09
    # ]) / 1000,
    # 每个年龄组的死亡率（女）（女性死亡人数/女性人数）
    deathes_female=np.array([
        6.68, 0.21, 0., 0.48, 0.41, 0.13, 0., 0.61, 0., 0.12, 0.67, 0.36,
        0.50, 0.27, 0.26, 0.57, 0.75, 1.09, 1.79, 2.64, 5.97, 9.51, 18.25,
        27.70, 61.45, 118.39
    ]) / 1000,
    # 每个年龄组的死亡率（男）
    deathes_male=np.array([
        8.49, 0.31, 0.15, 0.66, 0.76, 0.14, 0.41, 0.14, 0.61, 0.71, 0., 0.57,
        0.92, 0.55, 0.87, 1.51, 1.63, 2.38, 4.71, 6.31, 11.82, 17.52, 28.45,
        44.76, 90.01, 160.64
    ]) / 1000,
    # 出生婴儿女婴占比
    lambda_f=0.4681,
    # 出生婴儿男婴占比
    lambda_m=0.5319,
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
    gamma_P=3,
    # Local cancer恢复速率 (未知)
    gamma_LC=1.,
    # Region cancer恢复速率 (未知)
    gamma_RC=1/3.,
    # Distant cancer恢复速率 (未知)
    gamma_DC=1/5.,
    # 男性疾病恢复速率 (未知)
    gamma_D=1/3.,
    # 女性初始总人口数量
    total0_f=500000,
    # 男性初始总人口数量
    total0_m=500000,
    # 是否直接设置q为0，或者利用生育力和死亡率计算q
    q_is_zero=True,
)

result_root = "./results"
os.makedirs(result_root, exist_ok=True)

logging.info("[main] Model init ... ")
model = SparseAgeStructedHPVModel(**args)

logging.info("[main] plot initial population distribution ... ")
fg = U.plot_initial_population(model.agebins, model.P_f, model.P_m)
fg.savefig(osp.join(result_root, "init_population.png"))

logging.info("[main] start prediction ...")
res_t, res_yp, res_yn = model.predict(
    t_span=(0, 20), backend="solve_ivp"
)
logging.info("[main] saving results ...")
np.save(osp.join(result_root, "res_t.npy"), res_t)
np.save(osp.join(result_root, "res_yp.npy"), res_yp)
np.save(osp.join(result_root, "res_yn.npy"), res_yn)

logging.info("[main] plot results ...")
fg = U.plot_detailed_process(model.agebins, res_t, res_yn)
fg.savefig(osp.join(result_root, "detailed_result.png"))

