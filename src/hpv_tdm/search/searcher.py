from __future__ import annotations

from itertools import combinations, combinations_with_replacement
from pathlib import Path
from typing import Any

import numpy as np
import optuna

from ..config import SearchConfig
from ..evaluator import Evaluator
from ..model import BaseHPVTransmissionModel
from ..result import SearchResult, SimulationResult


class Searcher:
    def __init__(self, config: SearchConfig) -> None:
        self.config = config

    def _constraint_values(self, trial: optuna.trial.FrozenTrial) -> tuple[float]:
        incidence = float(trial.user_attrs.get("incidence", np.inf))
        return (incidence - self.config.incidence_threshold,)

    def _feasible_completed_trials(
        self,
        study: optuna.Study,
    ) -> list[optuna.trial.FrozenTrial]:
        return [
            trial
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
            and self._constraint_values(trial)[0] <= 0.0
        ]

    def _normalize_metric(self, value: float, scale: float) -> float:
        ratio = float(value / scale)
        if self.config.weighted_sum.transform == "ratio":
            return ratio
        return float(np.sign(ratio) * np.log1p(abs(ratio)))

    def _scalarized_objective(self, *, icur: float, incidence: float) -> float:
        incidence_scale = self.config.weighted_sum.incidence_scale
        if incidence_scale is None:
            incidence_scale = self.config.incidence_threshold
        if incidence_scale <= 0:
            raise ValueError("incidence scale must be positive for weighted_sum mode")
        normalized_icur = self._normalize_metric(
            icur,
            self.config.weighted_sum.icur_scale,
        )
        normalized_incidence = self._normalize_metric(
            incidence,
            incidence_scale,
        )
        return float(
            self.config.weighted_sum.icur_weight * normalized_icur
            + self.config.weighted_sum.incidence_weight * normalized_incidence
        )

    def _parse_age_selection(self, value: Any) -> list[int]:
        if isinstance(value, str):
            return [int(item) for item in value.split(",")]
        return [int(item) for item in value]

    def _parse_age_span(self, value: Any) -> tuple[int, int]:
        if isinstance(value, str):
            age0, age1 = value.split(":")
            return int(age0), int(age1)
        age0, age1 = value
        return int(age0), int(age1)

    def _candidate_product_ids(self, model: BaseHPVTransmissionModel) -> list[str]:
        if self.config.product_ids is not None:
            return self.config.product_ids
        return list(model.to_config().vaccine_catalog.products)

    def _build_candidate_from_params(
        self,
        base_config,
        params: dict[str, Any],
    ):
        coverage_by_age = [0.0] * base_config.nages
        strategy = self.config.strategy
        product_id = params["target_product_id"]
        if strategy == "one":
            coverage_by_age[int(params["target_age"])] = float(params["coverage"])
        elif strategy == "multi":
            ages = self._parse_age_selection(params["target_age"])
            for index, age in enumerate(ages):
                coverage_by_age[int(age)] = float(params[f"coverage{index}"])
        elif strategy == "conti":
            age0, age1 = self._parse_age_span(params["target_age_span"])
            for age in range(int(age0), int(age1) + 1):
                coverage_by_age[age] = float(params[f"coverage{age}"])
        elif strategy == "conti_one_cover":
            age0, age1 = self._parse_age_span(params["target_age_span"])
            for age in range(int(age0), int(age1) + 1):
                coverage_by_age[age] = float(params["coverage"])
        else:
            raise ValueError(f"unsupported search strategy: {strategy}")
        return base_config.with_vaccination(
            product_id=product_id,
            coverage_by_age=coverage_by_age,
        )

    def _suggest_candidate(self, trial: optuna.Trial, model: BaseHPVTransmissionModel):
        base_config = model.to_config()
        params: dict[str, Any] = {
            "target_product_id": trial.suggest_categorical(
                "target_product_id",
                self._candidate_product_ids(model),
            )
        }
        if self.config.strategy == "one":
            params["target_age"] = trial.suggest_categorical(
                "target_age",
                tuple(range(*self.config.age_index_span)),
            )
            params["coverage"] = trial.suggest_float("coverage", low=0.0, high=1.0)
        elif self.config.strategy == "multi":
            suggested_ages = list(
                ",".join(map(str, ages))
                for ages in combinations(
                    range(*self.config.age_index_span),
                    self.config.n_vacc_ages,
                )
            )
            params["target_age"] = trial.suggest_categorical(
                "target_age", suggested_ages
            )
            for index in range(self.config.n_vacc_ages):
                params[f"coverage{index}"] = trial.suggest_float(
                    f"coverage{index}",
                    low=0.0,
                    high=1.0,
                )
        elif self.config.strategy == "conti":
            suggested_spans = list(
                f"{age0}:{age1}"
                for age0, age1 in combinations_with_replacement(
                    range(*self.config.age_index_span),
                    2,
                )
            )
            age0, age1 = self._parse_age_span(
                trial.suggest_categorical("target_age_span", suggested_spans)
            )
            params["target_age_span"] = f"{age0}:{age1}"
            for age in range(age0, age1 + 1):
                params[f"coverage{age}"] = trial.suggest_float(
                    f"coverage{age}",
                    low=0.0,
                    high=1.0,
                )
        elif self.config.strategy == "conti_one_cover":
            suggested_spans = list(
                f"{age0}:{age1}"
                for age0, age1 in combinations_with_replacement(
                    range(*self.config.age_index_span),
                    2,
                )
            )
            params["target_age_span"] = trial.suggest_categorical(
                "target_age_span",
                suggested_spans,
            )
            params["coverage"] = trial.suggest_float("coverage", low=0.0, high=1.0)
        else:
            raise ValueError(f"unsupported search strategy: {self.config.strategy}")
        return self._build_candidate_from_params(base_config, params)

    def search(
        self,
        model: BaseHPVTransmissionModel,
        evaluator: Evaluator,
        *,
        output_dir: str | Path | None = None,
    ) -> SearchResult:
        base_config = model.to_config()
        if self.config.age_index_span[1] > base_config.nages:
            raise ValueError("search age_index_span is out of bounds for the model")

        reference_config = base_config.with_vaccination(
            product_id=None,
            coverage_by_age=[0.0] * base_config.nages,
        )
        model.set_config(reference_config)
        reference_simulation = model.simulate()
        reference_evaluation = evaluator.evaluate(reference_simulation)

        sampler = optuna.samplers.TPESampler(
            seed=self.config.seed,
            constraints_func=(
                self._constraint_values
                if self.config.objective_mode == "constrained"
                else None
            ),
        )
        storage_path: Path | None = None
        directions = (
            ["minimize", "minimize"]
            if self.config.objective_mode == "multi_objective"
            else ["minimize"]
        )
        if output_dir is None:
            study = optuna.create_study(
                study_name=self.config.study_name,
                sampler=sampler,
                directions=directions,
            )
        else:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            storage_path = output_path / self.config.storage_filename
            study = optuna.create_study(
                study_name=self.config.study_name,
                storage=f"sqlite:///{storage_path}",
                sampler=sampler,
                directions=directions,
                load_if_exists=True,
            )

        def objective(trial: optuna.Trial) -> tuple[float, float] | float:
            candidate_config = self._suggest_candidate(trial, model)
            model.set_config(candidate_config)
            simulation = model.simulate()
            evaluation_result = evaluator.evaluate(simulation, reference_simulation)
            icur = float(evaluation_result.icur[-1])
            incidence = float(evaluation_result.incidence[-1])
            trial.set_user_attr("icur", icur)
            trial.set_user_attr("incidence", incidence)
            if self.config.objective_mode == "multi_objective":
                return icur, incidence
            if self.config.objective_mode == "constrained":
                return icur
            scalarized = self._scalarized_objective(icur=icur, incidence=incidence)
            trial.set_user_attr("scalarized_objective", scalarized)
            return scalarized

        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            n_jobs=self.config.n_jobs,
        )

        best_trial = None
        if self.config.objective_mode == "multi_objective":
            best_value = np.inf
            for trial in sorted(study.best_trials, key=lambda item: item.values):
                if (
                    trial.values[1] <= self.config.incidence_threshold
                    and trial.values[0] < best_value
                ):
                    best_trial = trial
                    best_value = trial.values[0]
        elif self.config.objective_mode == "constrained":
            feasible_trials = self._feasible_completed_trials(study)
            if feasible_trials:
                best_trial = min(feasible_trials, key=lambda item: item.values[0])
        else:
            feasible_trials = self._feasible_completed_trials(study)
            if feasible_trials:
                best_trial = min(feasible_trials, key=lambda item: item.values[0])
            else:
                best_trial = study.best_trial

        best_simulation: SimulationResult | None = None
        best_evaluation = None
        best_model_config = None
        if best_trial is not None:
            best_model_config = self._build_candidate_from_params(
                base_config, best_trial.params
            )
            model.set_config(best_model_config)
            best_simulation = model.simulate()
            best_evaluation = evaluator.evaluate(best_simulation, reference_simulation)

        model.set_config(base_config)
        return SearchResult(
            config=self.config,
            study=study,
            study_storage_path=storage_path,
            reference_evaluation=reference_evaluation,
            best_model_config=best_model_config,
            best_simulation=best_simulation,
            best_evaluation=best_evaluation,
            best_trial=best_trial,
        )
