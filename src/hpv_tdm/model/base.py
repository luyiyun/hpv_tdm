from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from inspect import isfunction
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.integrate import odeint, solve_ivp
from tqdm import tqdm

from ..result import SimulationResult
from ..utils import agebins_to_labels
from ._demographic import (
    compute_c,
    compute_fq,
    compute_population_by_age,
    find_q_newton,
)
from ._sexual import compute_rho

if TYPE_CHECKING:
    from ..config import AggregateModelConfig, SubtypeGroupedModelConfig


class ODESystemModel(ABC):
    def _df_dt_with_progress(
        self,
        t: float,
        x: np.ndarray,
        pbar: tqdm,
        state: list[float],
    ) -> np.ndarray:
        last_t, delta_t = state
        steps = int((t - last_t) / delta_t)
        pbar.update(steps)
        state[0] = last_t + delta_t * steps
        return self.df_dt(t, x)

    def predict(
        self,
        init: np.ndarray,
        t_span: tuple[float, float],
        t_eval: np.ndarray | None = None,
        backend: str = "solve_ivp",
        verbose: bool = False,
    ) -> dict[str, np.ndarray]:
        if backend not in {"solve_ivp", "odeint"}:
            raise ValueError(f"unsupported backend: {backend}")
        if backend == "solve_ivp":
            if verbose:
                with tqdm(total=1000, unit="‰") as pbar:
                    result = solve_ivp(
                        self._df_dt_with_progress,
                        t_span=t_span,
                        y0=init,
                        t_eval=t_eval,
                        rtol=self.rtol,
                        atol=self.atol,
                        args=[pbar, [t_span[0], (t_span[1] - t_span[0]) / 1000]],
                    )
            else:
                result = solve_ivp(
                    self.df_dt,
                    t_span=t_span,
                    y0=init,
                    t_eval=t_eval,
                    rtol=self.rtol,
                    atol=self.atol,
                )
            return self._neat_predict_results(result.t, result.y.T)
        if t_eval is None:
            t_eval = np.linspace(*t_span, num=1000)
        y, _ = odeint(
            self.df_dt,
            init,
            t=t_eval,
            tfirst=True,
            rtol=self.rtol,
            atol=self.atol,
            full_output=True,
            printmessg=False,
        )
        return self._neat_predict_results(t_eval, y)

    @abstractmethod
    def df_dt(self, t: float, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _neat_predict_results(
        self,
        t: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError


class BaseHPVTransmissionModel(ODESystemModel):
    model_name = "base"

    def __init__(
        self, config: AggregateModelConfig | SubtypeGroupedModelConfig
    ) -> None:
        self.set_config(config)

    def set_config(
        self,
        config: AggregateModelConfig | SubtypeGroupedModelConfig,
    ) -> None:
        self.config = config
        demography = config.demography
        population = config.population
        transmission = config.transmission

        self.agebins = np.asarray(demography.agebins, dtype=float)
        self.fertilities = np.asarray(demography.fertilities, dtype=float)
        self.deathes_female = np.asarray(demography.deathes_female, dtype=float)
        self.deathes_male = np.asarray(demography.deathes_male, dtype=float)
        self.lambda_f = demography.lambda_f
        self.lambda_m = demography.lambda_m
        self.total_female = population.total_female
        self.total_male = population.total_male
        self.q_is_zero = demography.q_is_zero
        self.rtol = demography.rtol
        self.atol = demography.atol
        self.verbose = transmission.verbose

        self.epsilon_f = transmission.epsilon_f
        self.epsilon_m = transmission.epsilon_m
        self.omega_f = np.asarray(transmission.omega_f, dtype=float)
        self.omega_m = np.asarray(transmission.omega_m, dtype=float)
        self.partner_window = transmission.partner_window
        self.partner_decline = transmission.partner_decline
        self.partner_interval = transmission.partner_interval
        self.phi = transmission.phi
        self.beta_I = transmission.beta_I
        self.beta_P = transmission.beta_P
        self.beta_LC = transmission.beta_LC
        self.beta_RC = transmission.beta_RC
        self.dL = np.asarray(transmission.dL, dtype=float)
        self.dR = np.asarray(transmission.dR, dtype=float)
        self.dD = np.asarray(transmission.dD, dtype=float)
        self.gamma_I = transmission.gamma_I
        self.gamma_P = transmission.gamma_P
        self.gamma_LC = transmission.gamma_LC
        self.gamma_RC = transmission.gamma_RC
        self.gamma_DC = transmission.gamma_DC
        self.cal_cumulate = transmission.cal_cumulate
        self.vacc_prefer = transmission.vacc_prefer

        self.nages = len(self.agebins) - 1
        self.agebin_names = agebins_to_labels(self.agebins)
        self.agedelta = self.agebins[1:] - self.agebins[:-1]
        if self.q_is_zero:
            self.q = 0.0
            factor = (
                compute_fq(
                    self.deathes_female,
                    self.fertilities,
                    0.0,
                    self.agedelta,
                    lam=self.lambda_f,
                )[0]
                + 1
            )
            if self.verbose:
                logging.info("[init] fertility adjustment factor %.6f", 1 / factor)
            self.fertilities = self.fertilities / factor
        else:
            self.q = find_q_newton(
                self.lambda_f,
                self.fertilities,
                self.deathes_female,
                self.agedelta,
                verbose=self.verbose,
            )[0]
        self.c_f = compute_c(self.deathes_female, self.q, self.agedelta)
        self.c_m = compute_c(self.deathes_male, self.q, self.agedelta)
        self.P_f = compute_population_by_age(
            self.total_female,
            self.deathes_female,
            self.q,
            self.c_f,
        )
        self.P_m = compute_population_by_age(
            self.total_male,
            self.deathes_male,
            self.q,
            self.c_m,
        )
        self.dc_f = self.deathes_female + self.c_f
        self.dc_m = self.deathes_male + self.c_m
        self.dcq_f = self.dc_f + self.q
        self.dcq_m = self.dc_m + self.q
        self.c_f_ = self.c_f[:-1]
        self.c_m_ = self.c_m[:-1]
        self.rho = compute_rho(
            self.agebins,
            self.partner_window,
            self.partner_decline,
            100,
            self.partner_interval,
        )

    def to_config(self) -> AggregateModelConfig | SubtypeGroupedModelConfig:
        return self.config

    def simulate(
        self,
        init_state: np.ndarray | str | None = None,
        *,
        t_span: tuple[float, float] | None = None,
        n_eval: int | None = None,
        backend: str | None = None,
        verbose: bool | None = None,
    ) -> SimulationResult:
        if init_state is None:
            if self.config.simulation.init_state_path is None:
                init = self.default_initial_state()
            else:
                init = np.load(self.config.simulation.init_state_path)
        elif isinstance(init_state, str):
            init = np.load(init_state)
        else:
            init = np.asarray(init_state, dtype=float)
        if self.cal_cumulate:
            cumulative_size = self.nages * len(self.cumulative_state_spec)
            if init.size == self.ndim:
                # 允许直接传入“主状态”向量，此时自动补零累计量，便于重启模拟。
                init = np.concatenate([init, np.zeros(cumulative_size, dtype=float)])
            elif init.size != self.ndim + cumulative_size:
                raise ValueError(
                    "init_state size must match either the main state vector "
                    "or the full state-plus-cumulative vector"
                )
        elif init.size != self.ndim:
            raise ValueError("init_state size must match the main state vector")
        resolved_t_span = t_span or tuple(self.config.simulation.t_span)
        resolved_n_eval = self.config.simulation.n_eval if n_eval is None else n_eval
        resolved_backend = backend or self.config.transmission.backend
        resolved_verbose = (
            self.config.transmission.verbose if verbose is None else verbose
        )
        t_eval = None
        if resolved_n_eval is not None:
            t_eval = np.linspace(
                resolved_t_span[0], resolved_t_span[1], resolved_n_eval
            )
        raw = self.predict(
            init=init,
            t_span=resolved_t_span,
            t_eval=t_eval,
            backend=resolved_backend,
            verbose=resolved_verbose,
        )
        return SimulationResult(
            time=raw["t"],
            state=raw["y"],
            cumulative=raw["ycum"],
            model_kind=self.config.model_kind,
            state_spec=self.state_spec,
            cumulative_state_spec=self.cumulative_state_spec,
            config_snapshot=self.config.model_dump(mode="json"),
        )

    def total_population(self, y: np.ndarray) -> np.ndarray:
        return self.total_female_population(y) + self.total_male_population(y)

    @abstractmethod
    def default_initial_state(self) -> np.ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def state_spec(self) -> list[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def cumulative_state_spec(self) -> list[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def nrooms(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def ndim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def total_female_population(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def total_male_population(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def incidence_matrix(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def mortality_matrix(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def cumulative_cecx(self, ycum: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def cumulative_vaccinated(self, ycum: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def cumulative_cecx_deaths(self, ycum: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def group_incidence_matrix(self, y: np.ndarray) -> dict[str, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def group_mortality_matrix(self, y: np.ndarray) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def vaccination_cost_per_age(
        self,
        dose_cost: float,
        doses_under_15: int,
        doses_over_15: int,
    ) -> np.ndarray:
        lower_bounds = self.agebins[:-1]
        doses = np.where(lower_bounds < 15, doses_under_15, doses_over_15)
        return doses.astype(float) * dose_cost

    def _resolve_callable_or_array(self, value: Any, t: float) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if np.isscalar(value):
            return np.full(self.nages, float(value))
        if isfunction(value):
            return np.asarray(value(t), dtype=float)
        raise TypeError(f"unsupported callable/array value: {type(value)}")
