from __future__ import annotations

import numpy as np

from ..config import SubtypeGroupedModelConfig
from .base import BaseHPVTransmissionModel


class AgeSexSubtypeGroupedHPVModel(BaseHPVTransmissionModel):
    model_name = "subtype_grouped"

    def __init__(self, config: SubtypeGroupedModelConfig) -> None:
        super().__init__(config)

    def set_config(self, config: SubtypeGroupedModelConfig) -> None:
        super().set_config(config)
        self.psi = np.asarray(config.resolved_coverage_by_age(), dtype=float)
        self.group_names = list(config.subtype_groups)
        self.initial_group_weights = np.asarray(
            [config.subtype_groups[name].initial_weight for name in self.group_names],
            dtype=float,
        )
        self.initial_group_weights = (
            self.initial_group_weights / self.initial_group_weights.sum()
        )
        self.persistence_multipliers = np.asarray(
            [
                config.subtype_groups[name].persistence_multiplier
                for name in self.group_names
            ],
            dtype=float,
        )
        self.cancer_progression_multipliers = np.asarray(
            [
                config.subtype_groups[name].cancer_progression_multiplier
                for name in self.group_names
            ],
            dtype=float,
        )
        product = config.vaccine_catalog.get_product(config.resolved_product_id())
        self.vaccine_group_protection = {
            name: float(product.group_protection.get(name, 0.0))
            for name in self.group_names
        }
        self.protection_vector = np.asarray(
            [self.vaccine_group_protection[name] for name in self.group_names],
            dtype=float,
        )
        self._female_initial_state = np.asarray(
            config.transmission.female_initial_state,
            dtype=float,
        )
        self._male_initial_state = np.asarray(
            config.transmission.male_initial_state,
            dtype=float,
        )
        self.ngroups = len(self.group_names)
        self._nrooms = 3 + self.ngroups * 9
        self._ndim = self._nrooms * self.nages
        self._build_indices()

    def _build_indices(self) -> None:
        offset = 0
        self._state_index: dict[str, int] = {}
        self._state_index["Sf"] = offset
        offset += 1
        self._state_index["Vf"] = offset
        offset += 1
        for prefix in ("If", "Pf", "LC", "RC", "DC", "Rf"):
            for group_name in self.group_names:
                self._state_index[f"{prefix}__{group_name}"] = offset
                offset += 1
        self._state_index["Sm"] = offset
        offset += 1
        for prefix in ("Im", "Pm", "Rm"):
            for group_name in self.group_names:
                self._state_index[f"{prefix}__{group_name}"] = offset
                offset += 1

        cum_offset = 0
        self._cum_index: dict[str, int] = {}
        self._cum_index["Vaccinated"] = cum_offset
        cum_offset += 1
        for prefix in (
            "Recovery",
            "Infected",
            "Persisted",
            "LocalizedCancer",
            "RegionalCancer",
            "DistantCancer",
            "DeathedLC",
            "DeathedRC",
            "DeathedDC",
        ):
            for group_name in self.group_names:
                self._cum_index[f"{prefix}__{group_name}"] = cum_offset
                cum_offset += 1

    @property
    def nrooms(self) -> int:
        return self._nrooms

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def state_spec(self) -> list[str]:
        names = ["Sf", "Vf"]
        for prefix in ("If", "Pf", "LC", "RC", "DC", "Rf"):
            names.extend(f"{prefix}__{group_name}" for group_name in self.group_names)
        names.append("Sm")
        for prefix in ("Im", "Pm", "Rm"):
            names.extend(f"{prefix}__{group_name}" for group_name in self.group_names)
        return names

    @property
    def cumulative_state_spec(self) -> list[str]:
        names = ["Vaccinated"]
        for prefix in (
            "Recovery",
            "Infected",
            "Persisted",
            "LocalizedCancer",
            "RegionalCancer",
            "DistantCancer",
            "DeathedLC",
            "DeathedRC",
            "DeathedDC",
        ):
            names.extend(f"{prefix}__{group_name}" for group_name in self.group_names)
        return names

    def default_initial_state(self) -> np.ndarray:
        female_base = self._female_initial_state / self._female_initial_state.sum()
        male_base = self._male_initial_state / self._male_initial_state.sum()
        init = np.zeros((self.nrooms, self.nages), dtype=float)
        init[self._state_index["Sf"]] = self.P_f * female_base[0]
        # subtype 模型这里显式采用 8 维女性初始状态：
        # [Sf, If, Pf, LC, RC, DC, Rf, Vf]
        init[self._state_index["Vf"]] = self.P_f * female_base[7]
        init[self._state_index["Sm"]] = self.P_m * male_base[0]
        for index, group_name in enumerate(self.group_names):
            initial_weight = self.initial_group_weights[index]
            init[self._state_index[f"If__{group_name}"]] = (
                self.P_f * female_base[1] * initial_weight
            )
            init[self._state_index[f"Pf__{group_name}"]] = (
                self.P_f * female_base[2] * initial_weight
            )
            init[self._state_index[f"LC__{group_name}"]] = (
                self.P_f * female_base[3] * initial_weight
            )
            init[self._state_index[f"RC__{group_name}"]] = (
                self.P_f * female_base[4] * initial_weight
            )
            init[self._state_index[f"DC__{group_name}"]] = (
                self.P_f * female_base[5] * initial_weight
            )
            init[self._state_index[f"Rf__{group_name}"]] = (
                self.P_f * female_base[6] * initial_weight
            )
            init[self._state_index[f"Im__{group_name}"]] = (
                self.P_m * male_base[1] * initial_weight
            )
            init[self._state_index[f"Pm__{group_name}"]] = (
                self.P_m * male_base[2] * initial_weight
            )
            init[self._state_index[f"Rm__{group_name}"]] = (
                self.P_m * male_base[3] * initial_weight
            )
        flat = init.reshape(-1)
        if self.cal_cumulate:
            flat = np.concatenate(
                [flat, np.zeros(self.nages * len(self.cumulative_state_spec))]
            )
        return flat

    def _group_block(self, reshaped: np.ndarray, prefix: str) -> np.ndarray:
        return np.stack(
            [
                reshaped[self._state_index[f"{prefix}__{name}"]]
                for name in self.group_names
            ],
            axis=0,
        )

    def df_dt(self, t: float, x: np.ndarray) -> np.ndarray:
        state = x[: self.ndim] if self.cal_cumulate else x
        reshaped = state.reshape(self.nrooms, self.nages)
        Sf = reshaped[self._state_index["Sf"]]
        Vf = reshaped[self._state_index["Vf"]]
        Sm = reshaped[self._state_index["Sm"]]
        If = self._group_block(reshaped, "If")
        Pf = self._group_block(reshaped, "Pf")
        LC = self._group_block(reshaped, "LC")
        RC = self._group_block(reshaped, "RC")
        DC = self._group_block(reshaped, "DC")
        Rf = self._group_block(reshaped, "Rf")
        Im = self._group_block(reshaped, "Im")
        Pm = self._group_block(reshaped, "Pm")
        Rm = self._group_block(reshaped, "Rm")

        Ntf = (
            Sf
            + Vf
            + If.sum(axis=0)
            + Pf.sum(axis=0)
            + LC.sum(axis=0)
            + RC.sum(axis=0)
            + DC.sum(axis=0)
            + Rf.sum(axis=0)
        )
        Ntm = Sm + Im.sum(axis=0) + Pm.sum(axis=0) + Rm.sum(axis=0)
        infectious_f = np.divide(
            If + Pf,
            Ntf[None, :],
            out=np.zeros_like(If),
            where=Ntf[None, :] > 0,
        )
        infectious_m = np.divide(
            Im + Pm,
            Ntm[None, :],
            out=np.zeros_like(Im),
            where=Ntm[None, :] > 0,
        )
        alpha_f = (
            self.epsilon_f * self.omega_f[None, :] * np.dot(infectious_m, self.rho.T)
        )
        alpha_m = (
            self.epsilon_m * self.omega_m[None, :] * np.dot(infectious_f, self.rho.T)
        )
        psi = self._resolve_callable_or_array(self.psi, t)
        alpha_f_from_s = alpha_f * (1 - psi[None, :]) if self.vacc_prefer else alpha_f
        alpha_vaccinated = alpha_f * (1 - self.protection_vector[:, None])

        born = np.dot(Ntf, self.fertilities)
        born_f, born_m = born * self.lambda_f, born * self.lambda_m

        # 这里仍然采用“共享易感池 + 分亚型感染链”的第一版简化建模。
        dSf = (
            self.phi * Rf.sum(axis=0)
            - (alpha_f_from_s.sum(axis=0) + psi + self.dcq_f) * Sf
        )
        dSf[0] += born_f
        dSf[1:] += Sf[:-1] * self.c_f_

        dVf = psi * Sf - (alpha_vaccinated.sum(axis=0) + self.dcq_f) * Vf
        dVf[1:] += Vf[:-1] * self.c_f_

        dSm = self.phi * Rm.sum(axis=0) - (alpha_m.sum(axis=0) + self.dcq_m) * Sm
        dSm[0] += born_m
        dSm[1:] += Sm[:-1] * self.c_m_

        dIf = (
            alpha_f_from_s * Sf[None, :]
            + alpha_vaccinated * Vf[None, :]
            - (
                self.beta_I * self.persistence_multipliers[:, None]
                + self.gamma_I
                + self.dcq_f[None, :]
            )
            * If
        )
        dPf = (
            self.beta_I * self.persistence_multipliers[:, None] * If
            - (
                self.beta_P * self.cancer_progression_multipliers[:, None]
                + self.gamma_P
                + self.dcq_f[None, :]
            )
            * Pf
        )
        dLC = (
            self.beta_P * self.cancer_progression_multipliers[:, None] * Pf
            - (
                self.beta_LC * self.cancer_progression_multipliers[:, None]
                + self.gamma_LC
                + self.dL[None, :]
                + self.dcq_f[None, :]
            )
            * LC
        )
        dRC = (
            self.beta_LC * self.cancer_progression_multipliers[:, None] * LC
            - (
                self.beta_RC * self.cancer_progression_multipliers[:, None]
                + self.gamma_RC
                + self.dR[None, :]
                + self.dcq_f[None, :]
            )
            * RC
        )
        dDC = (
            self.beta_RC * self.cancer_progression_multipliers[:, None] * RC
            - (self.gamma_DC + self.dD[None, :] + self.dcq_f[None, :]) * DC
        )
        dRf = (
            self.gamma_I * If
            + self.gamma_P * Pf
            + self.gamma_LC * LC
            + self.gamma_RC * RC
            + self.gamma_DC * DC
            - (self.phi + self.dcq_f[None, :]) * Rf
        )
        dIm = (
            alpha_m * Sm[None, :]
            - (
                self.beta_I * self.persistence_multipliers[:, None]
                + self.gamma_I
                + self.dcq_m[None, :]
            )
            * Im
        )
        dPm = (
            self.beta_I * self.persistence_multipliers[:, None] * Im
            - (self.gamma_P + self.dcq_m[None, :]) * Pm
        )
        dRm = (
            self.gamma_I * Im
            + self.gamma_P * Pm
            - (self.phi + self.dcq_m[None, :]) * Rm
        )

        dIf[:, 1:] += If[:, :-1] * self.c_f_[None, :]
        dPf[:, 1:] += Pf[:, :-1] * self.c_f_[None, :]
        dLC[:, 1:] += LC[:, :-1] * self.c_f_[None, :]
        dRC[:, 1:] += RC[:, :-1] * self.c_f_[None, :]
        dDC[:, 1:] += DC[:, :-1] * self.c_f_[None, :]
        dRf[:, 1:] += Rf[:, :-1] * self.c_f_[None, :]
        dIm[:, 1:] += Im[:, :-1] * self.c_m_[None, :]
        dPm[:, 1:] += Pm[:, :-1] * self.c_m_[None, :]
        dRm[:, 1:] += Rm[:, :-1] * self.c_m_[None, :]

        derivatives = [dSf, dVf]
        for block in (dIf, dPf, dLC, dRC, dDC, dRf):
            derivatives.extend(list(block))
        derivatives.append(dSm)
        for block in (dIm, dPm, dRm):
            derivatives.extend(list(block))
        flat_derivative = np.concatenate(derivatives)

        if self.cal_cumulate:
            cumulative = [psi * Sf]
            blocks = (
                self.phi * Rf,
                alpha_f_from_s * Sf[None, :] + alpha_vaccinated * Vf[None, :],
                self.beta_I * self.persistence_multipliers[:, None] * If,
                self.beta_P * self.cancer_progression_multipliers[:, None] * Pf,
                self.beta_LC * self.cancer_progression_multipliers[:, None] * LC,
                self.beta_RC * self.cancer_progression_multipliers[:, None] * RC,
                self.dL[None, :] * LC,
                self.dR[None, :] * RC,
                self.dD[None, :] * DC,
            )
            for block in blocks:
                cumulative.extend(list(block))
            return np.concatenate([flat_derivative, np.concatenate(cumulative)])
        return flat_derivative

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
        female_indices = [self._state_index["Sf"], self._state_index["Vf"]]
        female_indices.extend(
            self._state_index[f"{prefix}__{name}"]
            for prefix in ("If", "Pf", "LC", "RC", "DC", "Rf")
            for name in self.group_names
        )
        return y[:, female_indices].sum(axis=1)

    def total_male_population(self, y: np.ndarray) -> np.ndarray:
        male_indices = [self._state_index["Sm"]]
        male_indices.extend(
            self._state_index[f"{prefix}__{name}"]
            for prefix in ("Im", "Pm", "Rm")
            for name in self.group_names
        )
        return y[:, male_indices].sum(axis=1)

    def group_incidence_matrix(self, y: np.ndarray) -> dict[str, np.ndarray]:
        return {
            group_name: self.beta_P
            * self.cancer_progression_multipliers[index]
            * y[:, self._state_index[f"Pf__{group_name}"]]
            for index, group_name in enumerate(self.group_names)
        }

    def incidence_matrix(self, y: np.ndarray) -> np.ndarray:
        return sum(self.group_incidence_matrix(y).values())

    def group_mortality_matrix(self, y: np.ndarray) -> dict[str, np.ndarray]:
        return {
            group_name: self.dL * y[:, self._state_index[f"LC__{group_name}"]]
            + self.dR * y[:, self._state_index[f"RC__{group_name}"]]
            + self.dD * y[:, self._state_index[f"DC__{group_name}"]]
            for group_name in self.group_names
        }

    def mortality_matrix(self, y: np.ndarray) -> np.ndarray:
        return sum(self.group_mortality_matrix(y).values())

    def cumulative_cecx(self, ycum: np.ndarray) -> np.ndarray:
        return sum(
            ycum[:, self._cum_index[f"LocalizedCancer__{group_name}"]]
            for group_name in self.group_names
        )

    def cumulative_vaccinated(self, ycum: np.ndarray) -> np.ndarray:
        return ycum[:, self._cum_index["Vaccinated"]]

    def cumulative_cecx_deaths(self, ycum: np.ndarray) -> np.ndarray:
        return sum(
            ycum[:, self._cum_index[f"{prefix}__{group_name}"]]
            for prefix in ("DeathedLC", "DeathedRC", "DeathedDC")
            for group_name in self.group_names
        )
