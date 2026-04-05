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

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        storage_path = output_dir / self.config.storage_filename
        sampler = optuna.samplers.TPESampler(seed=self.config.seed)
        study = optuna.create_study(
            study_name=self.config.study_name,
            storage=f"sqlite:///{storage_path}",
            sampler=sampler,
            directions=["minimize", "minimize"],
            load_if_exists=True,
        )

        def objective(trial: optuna.Trial) -> tuple[float, float]:
            candidate_config = self._suggest_candidate(trial, model)
            model.set_config(candidate_config)
            simulation = model.simulate()
            evaluation_result = evaluator.evaluate(simulation, reference_simulation)
            return float(evaluation_result.icur[-1]), float(
                evaluation_result.incidence[-1]
            )

        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            n_jobs=self.config.n_jobs,
        )

        best_trial = None
        best_value = np.inf
        for trial in sorted(study.best_trials, key=lambda item: item.values):
            if (
                trial.values[1] <= self.config.incidence_threshold
                and trial.values[0] < best_value
            ):
                best_trial = trial
                best_value = trial.values[0]

        best_simulation: SimulationResult | None = None
        best_evaluation = None
        if best_trial is not None:
            model.set_config(
                self._build_candidate_from_params(base_config, best_trial.params)
            )
            best_simulation = model.simulate()
            best_evaluation = evaluator.evaluate(best_simulation, reference_simulation)

        model.set_config(base_config)
        return SearchResult(
            config=self.config,
            study=study,
            reference_evaluation=reference_evaluation,
            best_simulation=best_simulation,
            best_evaluation=best_evaluation,
            best_trial=best_trial,
        )
