from __future__ import annotations

import numpy as np

from ..config import EvaluationConfig
from ..model._life_table import life_table
from ..result import EvaluationResult, SimulationResult


class Evaluator:
    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config

    def _discount_cumulative(self, value: np.ndarray, time: np.ndarray) -> np.ndarray:
        if value.shape[0] <= 1 or self.config.discount_rate <= 0:
            return value
        discount = 1 / np.power((1 + self.config.discount_rate), time[:-1])
        increments = value[1:] - value[:-1]
        if increments.ndim == 1:
            discounted = increments * discount
            discounted = np.r_[value[0], discounted]
        elif increments.ndim == 2:
            discounted = increments * discount[:, None]
            discounted = np.concatenate([value[[0]], discounted], axis=0)
        else:
            raise ValueError("unsupported cumulative value dimensionality")
        return np.cumsum(discounted, axis=0)

    def _evaluate_absolute(self, sim_result: SimulationResult) -> EvaluationResult:
        model = sim_result.get_model()
        female_population = model.total_female_population(sim_result.state)
        female_total = female_population.sum(axis=1)
        incidence_matrix = model.incidence_matrix(sim_result.state)
        mortality_matrix = model.mortality_matrix(sim_result.state)
        incidence = np.divide(
            incidence_matrix.sum(axis=1),
            female_total,
            out=np.zeros(sim_result.time.shape[0], dtype=float),
            where=female_total > 0,
        )
        mortality = np.divide(
            mortality_matrix.sum(axis=1),
            female_total,
            out=np.zeros(sim_result.time.shape[0], dtype=float),
            where=female_total > 0,
        )

        incidence_by_group = {}
        for name, matrix in model.group_incidence_matrix(sim_result.state).items():
            incidence_by_group[name] = np.divide(
                matrix.sum(axis=1),
                female_total,
                out=np.zeros(sim_result.time.shape[0], dtype=float),
                where=female_total > 0,
            )

        mortality_by_group = {}
        for name, matrix in model.group_mortality_matrix(sim_result.state).items():
            mortality_by_group[name] = np.divide(
                matrix.sum(axis=1),
                female_total,
                out=np.zeros(sim_result.time.shape[0], dtype=float),
                where=female_total > 0,
            )

        cumulative_cecx = model.cumulative_cecx(sim_result.cumulative)
        cumulative_vaccinated = model.cumulative_vaccinated(sim_result.cumulative)
        cumulative_cecx_deaths = model.cumulative_cecx_deaths(sim_result.cumulative)
        cumulative_cecx = self._discount_cumulative(cumulative_cecx, sim_result.time)
        cumulative_vaccinated = self._discount_cumulative(
            cumulative_vaccinated,
            sim_result.time,
        )
        cumulative_cecx_deaths = self._discount_cumulative(
            cumulative_cecx_deaths,
            sim_result.time,
        )

        coverage = np.asarray(model.config.resolved_coverage_by_age(), dtype=float)
        if model.config.vaccination.product_id is None and np.allclose(coverage, 0):
            vaccine_product_id = None
            cost_vector = np.zeros(model.nages, dtype=float)
        else:
            vaccine_product_id = model.config.resolved_product_id()
            product = model.config.vaccine_catalog.get_product(vaccine_product_id)
            cost_vector = model.vaccination_cost_per_age(
                dose_cost=product.dose_cost,
                doses_under_15=product.doses_under_15,
                doses_over_15=product.doses_over_15,
            )

        cost_vacc = np.dot(cumulative_vaccinated, cost_vector)
        cost_cecx = cumulative_cecx.sum(axis=1) * self.config.cost_per_cecx
        daly_fatal = cumulative_cecx_deaths.sum(axis=1) * self.config.daly_fatal
        daly_nofatal = (
            cumulative_cecx.sum(axis=1) - cumulative_cecx_deaths.sum(axis=1)
        ) * self.config.daly_nofatal

        table = life_table(
            model.deathes_female,
            model.agebins,
            method=self.config.life_table_method,
        )
        lifeloss = np.sum(cumulative_cecx_deaths * table["E"].values, axis=-1)
        return EvaluationResult(
            time=sim_result.time,
            incidence=incidence,
            mortality=mortality,
            incidence_by_group=incidence_by_group,
            mortality_by_group=mortality_by_group,
            cumulative_cecx=cumulative_cecx,
            cumulative_vaccinated=cumulative_vaccinated,
            cumulative_cecx_deaths=cumulative_cecx_deaths,
            cost_vacc=cost_vacc,
            cost_cecx=cost_cecx,
            daly_fatal=daly_fatal,
            daly_nofatal=daly_nofatal,
            lifeloss=lifeloss,
            config_snapshot=self.config.model_dump(mode="json"),
            vaccine_product_id=vaccine_product_id,
        )

    def evaluate(
        self,
        sim_result: SimulationResult,
        reference: SimulationResult | None = None,
    ) -> EvaluationResult:
        result = self._evaluate_absolute(sim_result)
        if reference is None:
            return result

        reference_result = self._evaluate_absolute(reference)
        total_cost = result.total_cost
        reference_cost = reference_result.total_cost
        total_daly = result.total_daly
        reference_daly = reference_result.total_daly
        cost_diff = total_cost - reference_cost
        daly_diff = reference_daly - total_daly
        icur = np.divide(
            cost_diff,
            daly_diff,
            out=np.full_like(cost_diff, fill_value=np.inf),
            where=daly_diff != 0,
        )
        avoid_cecx = reference_result.cumulative_cecx - result.cumulative_cecx
        avoid_cecx_deaths = (
            reference_result.cumulative_cecx_deaths - result.cumulative_cecx_deaths
        )
        avoid_daly = reference_result.total_daly - result.total_daly
        return EvaluationResult(
            time=result.time,
            incidence=result.incidence,
            mortality=result.mortality,
            incidence_by_group=result.incidence_by_group,
            mortality_by_group=result.mortality_by_group,
            cumulative_cecx=result.cumulative_cecx,
            cumulative_vaccinated=result.cumulative_vaccinated,
            cumulative_cecx_deaths=result.cumulative_cecx_deaths,
            cost_vacc=result.cost_vacc,
            cost_cecx=result.cost_cecx,
            daly_fatal=result.daly_fatal,
            daly_nofatal=result.daly_nofatal,
            lifeloss=result.lifeloss,
            config_snapshot=result.config_snapshot,
            vaccine_product_id=result.vaccine_product_id,
            icur=icur,
            avoid_cecx=avoid_cecx,
            avoid_cecx_deaths=avoid_cecx_deaths,
            avoid_daly=avoid_daly,
        )
