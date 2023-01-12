import numpy as np
import scipy.stats as sst
import optuna


def find_proper_gamma_parameter(agebins, omega, n_trials=500):
    omega = np.array(omega)
    omega = omega / np.sum(omega)

    def objective_find_gamma_parameter(trial: optuna.Trial):
        a = trial.suggest_float("a", 1, 10, log=False)
        # b = trial.suggest_float("b", 1, 10, log=False)
        loc = trial.suggest_float("loc", 10, 14, log=False)
        scale = trial.suggest_float("scale", 30, 50, log=False)
        p = sst.gamma.cdf(agebins, a, loc, scale)
        # p = sst.skewnorm.cdf(agebins, a, loc, scale)
        # p = sst.beta.cdf(agebins, a=a, b=b, loc=loc, scale=scale)
        p = p[1:] - p[:-1]
        return np.sum((p - omega)**2)

    study = optuna.create_study()
    study.optimize(objective_find_gamma_parameter, n_trials=n_trials)

    return study.best_params


def compute_rho(
    agebins, sex_window=10, decline_rate=0.05, n=100, bounds=(13, 60)
):
    mat = np.eye(n)
    for i in range(1, sex_window+1):
        row = np.arange(0, n-i)
        col = np.arange(i, n)
        tmp = 1 - decline_rate*i
        mat[row, col] = tmp
        mat[col, row] = tmp
    l, u = bounds
    mat[:, :l] = 0
    mat[:, u:] = 0
    mat[:l, :] = 0
    mat[u:, :] = 0

    agebins = np.r_[agebins[:-1].astype(int), n]
    num_ages = len(agebins) - 1
    mat2 = np.zeros((num_ages, num_ages), dtype=float)
    for r, (rowi, rowj) in enumerate(zip(agebins[:-1], agebins[1:])):
        for c, (coli, colj) in enumerate(zip(agebins[:-1], agebins[1:])):
            mat2[r, c] = mat[rowi:rowj, coli:colj].sum()

    mat2 = mat2 / mat2.sum(axis=1, keepdims=True)
    mat2 = np.nan_to_num(mat2, 0.)

    return mat2
