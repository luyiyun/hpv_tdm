from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import optuna
import pandas as pd

from ._io import ensure_parent, read_json_attr, write_json_attr
from .evaluation import EvaluationResult
from .simulation import SimulationResult


class SearchResult:
    def __init__(
        self,
        *,
        config,
        study: optuna.Study,
        study_storage_path: Path | None,
        reference_evaluation: EvaluationResult,
        best_model_config,
        best_simulation: SimulationResult | None,
        best_evaluation: EvaluationResult | None,
        best_trial: optuna.trial.FrozenTrial | None,
    ) -> None:
        self.config = config
        self.study = study
        self.study_storage_path = study_storage_path
        self.reference_evaluation = reference_evaluation
        self.best_model_config = best_model_config
        self.best_simulation = best_simulation
        self.best_evaluation = best_evaluation
        self.best_trial = best_trial

    def summary_table(self) -> pd.DataFrame:
        payload: dict[str, Any] = {
            "study_name": self.config.study_name,
            "n_trials": len(self.study.trials),
            "incidence_threshold": self.config.incidence_threshold,
            "has_best_trial": self.best_trial is not None,
        }
        if self.best_trial is not None:
            payload["best_trial_number"] = self.best_trial.number
            payload["best_icur"] = self.best_trial.values[0]
            payload["best_incidence"] = self.best_trial.values[1]
            for key, value in self.best_trial.params.items():
                payload[f"param_{key}"] = value
        return pd.DataFrame([payload])

    def plot_history(self, *, save_path: str | Path | None = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        completed_trials = [
            trial
            for trial in self.study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        axes[0].plot(
            [trial.number for trial in completed_trials],
            [trial.values[0] for trial in completed_trials],
            marker="o",
            linestyle="-",
        )
        axes[0].set_title("ICUR History")
        axes[0].set_xlabel("Trial")
        axes[0].set_ylabel("ICUR")
        axes[1].plot(
            [trial.number for trial in completed_trials],
            [trial.values[1] for trial in completed_trials],
            marker="o",
            linestyle="-",
        )
        axes[1].set_title("Incidence History")
        axes[1].set_xlabel("Trial")
        axes[1].set_ylabel("Incidence")
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=200)
        return fig

    def plot_pareto(self, *, save_path: str | Path | None = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6, 5))
        completed_trials = [
            trial
            for trial in self.study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        ax.scatter(
            [trial.values[1] for trial in completed_trials],
            [trial.values[0] for trial in completed_trials],
            alpha=0.5,
            label="all_trials",
        )
        ax.scatter(
            [trial.values[1] for trial in self.study.best_trials],
            [trial.values[0] for trial in self.study.best_trials],
            color="red",
            label="pareto_front",
        )
        ax.set_xlabel("Incidence")
        ax.set_ylabel("ICUR")
        ax.set_title("Pareto Front")
        ax.legend()
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=200)
        return fig

    def save(self, directory: str | Path) -> None:
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)

        self.config.to_json_file(target_dir / "search_config.json")
        with (target_dir / "best_trial.json").open("w", encoding="utf-8") as handle:
            json.dump(
                None
                if self.best_trial is None
                else {
                    "number": self.best_trial.number,
                    "values": self.best_trial.values,
                    "params": self.best_trial.params,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.write("\n")

        search_result_path = ensure_parent(target_dir / "search_result.h5")
        with h5py.File(search_result_path, "w") as handle:
            meta_group = handle.create_group("search")
            write_json_attr(
                meta_group, "config_snapshot", self.config.model_dump(mode="json")
            )
            write_json_attr(
                meta_group,
                "best_trial",
                None
                if self.best_trial is None
                else {
                    "number": self.best_trial.number,
                    "values": self.best_trial.values,
                    "params": self.best_trial.params,
                },
            )
            self.reference_evaluation._write_to_group(
                handle.create_group("reference_evaluation")
            )

        if self.study_storage_path is not None and self.study_storage_path.exists():
            storage_target = target_dir / self.config.storage_filename
            if self.study_storage_path.resolve() != storage_target.resolve():
                shutil.copy2(self.study_storage_path, storage_target)

        if self.best_model_config is not None:
            self.best_model_config.to_json_file(target_dir / "best_model_config.json")
        if self.best_simulation is not None:
            self.best_simulation.to_hdf(target_dir / "best_simulation.h5")
        if self.best_evaluation is not None:
            self.best_evaluation.to_hdf(target_dir / "best_evaluation.h5")

    @classmethod
    def from_dir(cls, directory: str | Path) -> "SearchResult":
        from ..config import (
            AggregateModelConfig,
            SearchConfig,
            SubtypeGroupedModelConfig,
        )

        root = Path(directory)
        config = SearchConfig.from_json_file(root / "search_config.json")
        study = optuna.load_study(
            study_name=config.study_name,
            storage=f"sqlite:///{root / config.storage_filename}",
        )
        with h5py.File(root / "search_result.h5", "r") as handle:
            reference_evaluation = EvaluationResult._from_group(
                handle["reference_evaluation"]
            )
            best_trial_payload = read_json_attr(handle["search"], "best_trial")
        best_trial = None
        if best_trial_payload is not None:
            best_trial = next(
                trial
                for trial in study.trials
                if trial.number == best_trial_payload["number"]
            )
        best_simulation = None
        best_evaluation = None
        best_model_config = None
        if (root / "best_model_config.json").exists():
            with (root / "best_model_config.json").open(
                "r", encoding="utf-8"
            ) as handle:
                payload = json.load(handle)
            model_kind = payload.get("model_kind", "aggregate")
            if model_kind == "aggregate":
                best_model_config = AggregateModelConfig.from_json_dict(payload)
            elif model_kind == "subtype_grouped":
                best_model_config = SubtypeGroupedModelConfig.from_json_dict(payload)
            else:
                raise ValueError(f"unsupported model kind: {model_kind}")
        if (root / "best_simulation.h5").exists():
            best_simulation = SimulationResult.from_hdf(root / "best_simulation.h5")
        if (root / "best_evaluation.h5").exists():
            best_evaluation = EvaluationResult.from_hdf(root / "best_evaluation.h5")
        return cls(
            config=config,
            study=study,
            study_storage_path=root / config.storage_filename,
            reference_evaluation=reference_evaluation,
            best_model_config=best_model_config,
            best_simulation=best_simulation,
            best_evaluation=best_evaluation,
            best_trial=best_trial,
        )
