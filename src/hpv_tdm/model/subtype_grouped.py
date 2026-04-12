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
        # 共享真实人口分母 + 共享 persistent-risk pool + 各亚型组边际风险通道。
        # 这样既能减弱唯一易感池带来的假性替代，又能避免癌症事件直接按各组简单求和。
        self._nrooms = 3 + self.ngroups * 12
        self._ndim = self._nrooms * self.nages
        self._build_indices()

    def _build_indices(self) -> None:
        offset = 0
        self._state_index: dict[str, int] = {}
        self._state_index["Nf"] = offset
        offset += 1
        self._state_index["Nm"] = offset
        offset += 1
        self._state_index["Pany"] = offset
        offset += 1
        for prefix in ("Sf", "Vf", "If", "Pf", "LC", "RC", "DC", "Rf"):
            for group_name in self.group_names:
                self._state_index[f"{prefix}__{group_name}"] = offset
                offset += 1
        for prefix in ("Sm", "Im", "Pm", "Rm"):
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
        names = ["Nf", "Nm", "Pany"]
        for prefix in ("Sf", "Vf", "If", "Pf", "LC", "RC", "DC", "Rf"):
            names.extend(f"{prefix}__{group_name}" for group_name in self.group_names)
        for prefix in ("Sm", "Im", "Pm", "Rm"):
            names.extend(f"{prefix}__{group_name}" for group_name in self.group_names)
        return names

    def _union_count(self, block: np.ndarray, total: np.ndarray) -> np.ndarray:
        fraction = np.divide(
            block,
            total[None, :],
            out=np.zeros_like(block),
            where=total[None, :] > 0,
        )
        fraction = np.clip(fraction, 0.0, 1.0)
        return total * (1.0 - np.prod(1.0 - fraction, axis=0))

    def _cancer_flow_by_group(
        self,
        Pf: np.ndarray,
        Pany: np.ndarray,
    ) -> np.ndarray:
        raw_flow = self.beta_P * self.cancer_progression_multipliers[:, None] * Pf
        pf_sum = Pf.sum(axis=0)
        overlap_scale = np.divide(
            Pany,
            pf_sum,
            out=np.zeros_like(Pany),
            where=pf_sum > 0,
        )
        overlap_scale = np.clip(overlap_scale, 0.0, 1.0)
        return raw_flow * overlap_scale[None, :]

    def _cancer_flow_by_group_timeseries(self, y: np.ndarray) -> np.ndarray:
        Pf = np.stack(
            [
                y[:, self._state_index[f"Pf__{group_name}"]]
                for group_name in self.group_names
            ],
            axis=1,
        )
        raw_flow = self.beta_P * self.cancer_progression_multipliers[None, :, None] * Pf
        pf_sum = Pf.sum(axis=1)
        Pany = y[:, self._state_index["Pany"]]
        overlap_scale = np.divide(
            Pany,
            pf_sum,
            out=np.zeros_like(Pany),
            where=pf_sum > 0,
        )
        overlap_scale = np.clip(overlap_scale, 0.0, 1.0)
        return raw_flow * overlap_scale[:, None, :]

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
        init[self._state_index["Nf"]] = self.P_f
        init[self._state_index["Nm"]] = self.P_m
        # 这里的 group-specific S/V 状态表示“对该亚型组的边际风险状态”，
        # 而不是把个体硬分成只能感染一个型别的人群。
        for index, group_name in enumerate(self.group_names):
            initial_weight = self.initial_group_weights[index]
            if_f = self.P_f * female_base[1] * initial_weight
            pf_f = self.P_f * female_base[2] * initial_weight
            lc_f = self.P_f * female_base[3] * initial_weight
            rc_f = self.P_f * female_base[4] * initial_weight
            dc_f = self.P_f * female_base[5] * initial_weight
            rf_f = self.P_f * female_base[6] * initial_weight
            vf_f = self.P_f * female_base[7]
            sf_f = self.P_f - (vf_f + if_f + pf_f + lc_f + rc_f + dc_f + rf_f)

            im_m = self.P_m * male_base[1] * initial_weight
            pm_m = self.P_m * male_base[2] * initial_weight
            rm_m = self.P_m * male_base[3] * initial_weight
            sm_m = self.P_m - (im_m + pm_m + rm_m)

            init[self._state_index[f"Sf__{group_name}"]] = sf_f
            init[self._state_index[f"Vf__{group_name}"]] = vf_f
            init[self._state_index[f"If__{group_name}"]] = (
                if_f
            )
            init[self._state_index[f"Pf__{group_name}"]] = (
                pf_f
            )
            init[self._state_index[f"LC__{group_name}"]] = (
                lc_f
            )
            init[self._state_index[f"RC__{group_name}"]] = (
                rc_f
            )
            init[self._state_index[f"DC__{group_name}"]] = (
                dc_f
            )
            init[self._state_index[f"Rf__{group_name}"]] = (
                rf_f
            )
            init[self._state_index[f"Sm__{group_name}"]] = (
                sm_m
            )
            init[self._state_index[f"Im__{group_name}"]] = (
                im_m
            )
            init[self._state_index[f"Pm__{group_name}"]] = (
                pm_m
            )
            init[self._state_index[f"Rm__{group_name}"]] = (
                rm_m
            )
        init[self._state_index["Pany"]] = self._union_count(
            np.stack(
                [
                    init[self._state_index[f"Pf__{group_name}"]]
                    for group_name in self.group_names
                ],
                axis=0,
            ),
            self.P_f,
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
        Ntf = reshaped[self._state_index["Nf"]]
        Ntm = reshaped[self._state_index["Nm"]]
        Pany = reshaped[self._state_index["Pany"]]
        Sf = self._group_block(reshaped, "Sf")
        Vf = self._group_block(reshaped, "Vf")
        If = self._group_block(reshaped, "If")
        Pf = self._group_block(reshaped, "Pf")
        LC = self._group_block(reshaped, "LC")
        RC = self._group_block(reshaped, "RC")
        DC = self._group_block(reshaped, "DC")
        Rf = self._group_block(reshaped, "Rf")
        Sm = self._group_block(reshaped, "Sm")
        Im = self._group_block(reshaped, "Im")
        Pm = self._group_block(reshaped, "Pm")
        Rm = self._group_block(reshaped, "Rm")

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
        total_cecx_death = (
            self.dL[None, :] * LC + self.dR[None, :] * RC + self.dD[None, :] * DC
        ).sum(axis=0)
        raw_persistent_inflow = self.beta_I * self.persistence_multipliers[:, None] * If
        if_any = self._union_count(If, Ntf)
        if_sum = If.sum(axis=0)
        persistent_any_inflow = np.divide(
            if_any,
            if_sum,
            out=np.zeros_like(if_any),
            where=if_sum > 0,
        ) * raw_persistent_inflow.sum(axis=0)
        persistent_any_inflow = np.clip(persistent_any_inflow, 0.0, None)
        cancer_flow_by_group = self._cancer_flow_by_group(Pf, Pany)
        cancer_flow_total = cancer_flow_by_group.sum(axis=0)

        dNf = -(self.dcq_f * Ntf) - total_cecx_death
        dNf[0] += born_f
        dNf[1:] += Ntf[:-1] * self.c_f_

        dNm = -(self.dcq_m * Ntm)
        dNm[0] += born_m
        dNm[1:] += Ntm[:-1] * self.c_m_

        dPany = (
            persistent_any_inflow
            - cancer_flow_total
            - (self.gamma_P + self.dcq_f) * Pany
        )
        dPany[1:] += Pany[:-1] * self.c_f_

        # 这里改为：
        # 共享真实人口分母 + 共享 persistent-risk pool + 分亚型组边际风险通道。
        # 各组不再机械性地争抢唯一 Sf/Vf，同时癌症事件也不再按各组直接简单求和。
        dSf = (
            self.phi * Rf
            - (alpha_f_from_s + psi[None, :] + self.dcq_f[None, :]) * Sf
        )
        dSf[:, 0] += born_f
        dSf[:, 1:] += Sf[:, :-1] * self.c_f_[None, :]

        dVf = psi[None, :] * Sf - (alpha_vaccinated + self.dcq_f[None, :]) * Vf
        dVf[:, 1:] += Vf[:, :-1] * self.c_f_[None, :]

        dSm = self.phi * Rm - (alpha_m + self.dcq_m[None, :]) * Sm
        dSm[:, 0] += born_m
        dSm[:, 1:] += Sm[:, :-1] * self.c_m_[None, :]

        dIf = (
            alpha_f_from_s * Sf
            + alpha_vaccinated * Vf
            - (
                self.beta_I * self.persistence_multipliers[:, None]
                + self.gamma_I
                + self.dcq_f[None, :]
            )
            * If
        )
        dPf = (
            raw_persistent_inflow
            - (
                self.gamma_P
                + self.dcq_f[None, :]
            )
            * Pf
            - cancer_flow_by_group
        )
        dLC = (
            cancer_flow_by_group
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
            alpha_m * Sm
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

        derivatives = [dNf, dNm, dPany]
        for block in (dSf, dVf, dIf, dPf, dLC, dRC, dDC, dRf):
            derivatives.extend(list(block))
        for block in (dSm, dIm, dPm, dRm):
            derivatives.extend(list(block))
        flat_derivative = np.concatenate(derivatives)

        if self.cal_cumulate:
            # 累计接种人数使用各亚型组边际接种流入的平均值，作为实际接种人数的近似。
            cumulative = [np.mean(psi[None, :] * Sf, axis=0)]
            blocks = (
                self.phi * Rf,
                alpha_f_from_s * Sf + alpha_vaccinated * Vf,
                raw_persistent_inflow,
                cancer_flow_by_group,
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
        return y[:, self._state_index["Nf"]]

    def total_male_population(self, y: np.ndarray) -> np.ndarray:
        return y[:, self._state_index["Nm"]]

    def group_incidence_matrix(self, y: np.ndarray) -> dict[str, np.ndarray]:
        flow = self._cancer_flow_by_group_timeseries(y)
        return {
            group_name: flow[:, index]
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
