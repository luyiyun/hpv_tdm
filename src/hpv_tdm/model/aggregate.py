from __future__ import annotations

import numpy as np

from ..config import AggregateModelConfig
from .base import BaseHPVTransmissionModel


class AgeSexAggregateHPVModel(BaseHPVTransmissionModel):
    model_name = "aggregate"

    def __init__(self, config: AggregateModelConfig) -> None:
        super().__init__(config)

    def set_config(self, config: AggregateModelConfig) -> None:
        super().set_config(config)
        product = config.vaccine_catalog.get_product(config.resolved_product_id())
        self.psi = np.asarray(config.resolved_coverage_by_age(), dtype=float)
        self.tau = product.aggregate_efficacy
        self._female_initial_state = np.asarray(
            config.transmission.female_initial_state,
            dtype=float,
        )
        self._male_initial_state = np.asarray(
            config.transmission.male_initial_state,
            dtype=float,
        )
        self.nrooms_f = 8
        self.nrooms_m = 4
        self._nrooms = 12
        self._ndim = self._nrooms * self.nages

    @property
    def state_spec(self) -> list[str]:
        return [
            "Sf",
            "If",
            "Pf",
            "LC",
            "RC",
            "DC",
            "Rf",
            "Vf",
            "Sm",
            "Im",
            "Pm",
            "Rm",
        ]

    @property
    def cumulative_state_spec(self) -> list[str]:
        return [
            "Recovery",
            "Infected",
            "Persisted",
            "LocalizedCancer",
            "RegionalCancer",
            "DistantCancer",
            "DeathedLC",
            "DeathedRC",
            "DeathedDC",
            "Vaccinated",
        ]

    @property
    def nrooms(self) -> int:
        return self._nrooms

    @property
    def ndim(self) -> int:
        return self._ndim

    def default_initial_state(self) -> np.ndarray:
        female = self._female_initial_state / self._female_initial_state.sum()
        male = self._male_initial_state / self._male_initial_state.sum()
        init_f = female[:, None] * self.P_f
        init_m = male[:, None] * self.P_m
        state = np.concatenate([init_f, init_m], axis=0).reshape(-1)
        if self.cal_cumulate:
            state = np.concatenate([state, np.zeros(self.nages * 10)])
        return state

    def df_dt(self, t: float, x: np.ndarray) -> np.ndarray:
        state = x[: self.ndim] if self.cal_cumulate else x
        reshaped = state.reshape(self.nrooms, self.nages)
        Ntf = reshaped[: self.nrooms_f].sum(axis=0)
        Ntm = reshaped[self.nrooms_f :].sum(axis=0)
        Sf, If, Pf, LC, RC, DC, Rf, Vf, Sm, Im, Pm, Rm = reshaped

        infectious_f = np.divide(If + Pf, Ntf, out=np.zeros_like(Ntf), where=Ntf > 0)
        infectious_m = np.divide(Im + Pm, Ntm, out=np.zeros_like(Ntm), where=Ntm > 0)
        alpha_f = self.epsilon_f * self.omega_f * np.dot(self.rho, infectious_m)
        alpha_m = self.epsilon_m * self.omega_m * np.dot(self.rho, infectious_f)

        psi = self._resolve_callable_or_array(self.psi, t)
        born = np.dot(Ntf, self.fertilities)
        born_f, born_m = born * self.lambda_f, born * self.lambda_m

        # 这里保留原模型的“优先给未感染易感者接种”的可选设定。
        if self.vacc_prefer:
            alpha_f = alpha_f * (1 - psi)

        dSf = self.phi * Rf - (alpha_f + psi * self.tau + self.dcq_f) * Sf
        dSf[0] += born_f
        dSf[1:] += Sf[:-1] * self.c_f_

        dIf = alpha_f * Sf - (self.beta_I + self.gamma_I + self.dcq_f) * If
        dIf[1:] += If[:-1] * self.c_f_

        dPf = self.beta_I * If - (self.beta_P + self.gamma_P + self.dcq_f) * Pf
        dPf[1:] += Pf[:-1] * self.c_f_

        dLC = (
            self.beta_P * Pf
            - (self.beta_LC + self.gamma_LC + self.dL + self.dcq_f) * LC
        )
        dLC[1:] += LC[:-1] * self.c_f_

        dRC = (
            self.beta_LC * LC
            - (self.beta_RC + self.gamma_RC + self.dR + self.dcq_f) * RC
        )
        dRC[1:] += RC[:-1] * self.c_f_

        dDC = self.beta_RC * RC - (self.gamma_DC + self.dD + self.dcq_f) * DC
        dDC[1:] += DC[:-1] * self.c_f_

        dRf = (
            self.gamma_I * If
            + self.gamma_P * Pf
            + self.gamma_LC * LC
            + self.gamma_RC * RC
            + self.gamma_DC * DC
            - (self.phi + self.dcq_f) * Rf
        )
        dRf[1:] += Rf[:-1] * self.c_f_

        dVf = psi * self.tau * Sf - self.dcq_f * Vf
        dVf[1:] += Vf[:-1] * self.c_f_

        dSm = self.phi * Rm - (alpha_m + self.dcq_m) * Sm
        dSm[0] += born_m
        dSm[1:] += Sm[:-1] * self.c_m_

        dIm = alpha_m * Sm - (self.beta_I + self.gamma_I + self.dcq_m) * Im
        dIm[1:] += Im[:-1] * self.c_m_

        dPm = self.beta_I * Im - (self.beta_P + self.gamma_P + self.dcq_m) * Pm
        dPm[1:] += Pm[:-1] * self.c_m_

        dRm = self.gamma_I * Im + self.gamma_P * Pm - (self.phi + self.dcq_m) * Rm
        dRm[1:] += Rm[:-1] * self.c_m_

        derivatives = np.concatenate(
            [dSf, dIf, dPf, dLC, dRC, dDC, dRf, dVf, dSm, dIm, dPm, dRm]
        )
        if self.cal_cumulate:
            cumulative = np.concatenate(
                [
                    self.phi * Rf,
                    alpha_f * Sf,
                    self.beta_I * If,
                    self.beta_P * Pf,
                    self.beta_LC * LC,
                    self.beta_RC * RC,
                    self.dL * LC,
                    self.dR * RC,
                    self.dD * DC,
                    psi * self.tau * Sf,
                ]
            )
            return np.concatenate([derivatives, cumulative])
        return derivatives

    def _neat_predict_results(
        self,
        t: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {"t": t}
        if self.cal_cumulate:
            y_main, ycum = y[:, : self.ndim], y[:, self.ndim :]
            result["y"] = y_main.reshape(-1, self.nrooms, self.nages)
            result["ycum"] = ycum.reshape(
                -1,
                len(self.cumulative_state_spec),
                self.nages,
            )
            return result
        result["y"] = y.reshape(-1, self.nrooms, self.nages)
        result["ycum"] = np.zeros((len(t), len(self.cumulative_state_spec), self.nages))
        return result

    def total_female_population(self, y: np.ndarray) -> np.ndarray:
        return y[:, : self.nrooms_f].sum(axis=1)

    def total_male_population(self, y: np.ndarray) -> np.ndarray:
        return y[:, self.nrooms_f :].sum(axis=1)

    def incidence_matrix(self, y: np.ndarray) -> np.ndarray:
        return self.beta_P * y[:, 2]

    def mortality_matrix(self, y: np.ndarray) -> np.ndarray:
        return self.dL * y[:, 3] + self.dR * y[:, 4] + self.dD * y[:, 5]

    def cumulative_cecx(self, ycum: np.ndarray) -> np.ndarray:
        return ycum[:, 3]

    def cumulative_vaccinated(self, ycum: np.ndarray) -> np.ndarray:
        return ycum[:, -1]

    def cumulative_cecx_deaths(self, ycum: np.ndarray) -> np.ndarray:
        return ycum[:, 6:9].sum(axis=1)

    def group_incidence_matrix(self, y: np.ndarray) -> dict[str, np.ndarray]:
        return {"aggregate": self.incidence_matrix(y)}

    def group_mortality_matrix(self, y: np.ndarray) -> dict[str, np.ndarray]:
        return {"aggregate": self.mortality_matrix(y)}
