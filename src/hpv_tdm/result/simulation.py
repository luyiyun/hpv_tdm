from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._io import ensure_parent, read_json_attr, write_json_attr

if TYPE_CHECKING:
    from ..model.base import BaseHPVTransmissionModel


class SimulationResult:
    def __init__(
        self,
        *,
        time: np.ndarray,
        state: np.ndarray,
        cumulative: np.ndarray,
        model_kind: str,
        state_spec: list[str],
        cumulative_state_spec: list[str],
        config_snapshot: dict[str, Any],
        model: BaseHPVTransmissionModel | None = None,
    ) -> None:
        self.time = np.asarray(time, dtype=float)
        self.state = np.asarray(state, dtype=float)
        self.cumulative = np.asarray(cumulative, dtype=float)
        self.model_kind = model_kind
        self.state_spec = list(state_spec)
        self.cumulative_state_spec = list(cumulative_state_spec)
        self.config_snapshot = config_snapshot
        self._model = model

    def get_model(self) -> BaseHPVTransmissionModel:
        if self._model is None:
            from ..config import AggregateModelConfig, SubtypeGroupedModelConfig
            from ..model import (
                AgeSexAggregateHPVModel,
                AgeSexSubtypeGroupedHPVModel,
            )

            if self.model_kind == "aggregate":
                config = AggregateModelConfig.model_validate(self.config_snapshot)
                self._model = AgeSexAggregateHPVModel(config)
            elif self.model_kind == "subtype_grouped":
                config = SubtypeGroupedModelConfig.model_validate(self.config_snapshot)
                self._model = AgeSexSubtypeGroupedHPVModel(config)
            else:
                raise ValueError(f"unsupported model kind: {self.model_kind}")
        return self._model

    @property
    def age_labels(self) -> np.ndarray:
        return self.get_model().agebin_names

    def summary_table(self) -> pd.DataFrame:
        model = self.get_model()
        female_population = model.total_female_population(self.state)
        incidence = np.divide(
            model.incidence_matrix(self.state).sum(axis=1),
            female_population.sum(axis=1),
            out=np.zeros(self.time.shape[0], dtype=float),
            where=female_population.sum(axis=1) > 0,
        )
        mortality = np.divide(
            model.mortality_matrix(self.state).sum(axis=1),
            female_population.sum(axis=1),
            out=np.zeros(self.time.shape[0], dtype=float),
            where=female_population.sum(axis=1) > 0,
        )
        return pd.DataFrame(
            [
                {
                    "model_kind": self.model_kind,
                    "t_final": float(self.time[-1]),
                    "female_population_final": float(female_population.sum(axis=1)[-1]),
                    "incidence_final": float(incidence[-1]),
                    "mortality_final": float(mortality[-1]),
                    "cumulative_cecx_final": float(
                        model.cumulative_cecx(self.cumulative).sum(axis=1)[-1]
                    ),
                    "cumulative_vaccinated_final": float(
                        model.cumulative_vaccinated(self.cumulative).sum(axis=1)[-1]
                    ),
                    "cumulative_cecx_deaths_final": float(
                        model.cumulative_cecx_deaths(self.cumulative).sum(axis=1)[-1]
                    ),
                }
            ]
        )

    def plot_incidence(
        self,
        *,
        by_group: bool = False,
        log: bool = False,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        model = self.get_model()
        female_population = model.total_female_population(self.state)
        fig, ax = plt.subplots(figsize=(8, 4))
        if by_group:
            for name, matrix in model.group_incidence_matrix(self.state).items():
                values = np.divide(
                    matrix.sum(axis=1),
                    female_population.sum(axis=1),
                    out=np.zeros(self.time.shape[0], dtype=float),
                    where=female_population.sum(axis=1) > 0,
                )
                ax.plot(self.time, values, label=name)
            ax.legend()
        else:
            values = np.divide(
                model.incidence_matrix(self.state).sum(axis=1),
                female_population.sum(axis=1),
                out=np.zeros(self.time.shape[0], dtype=float),
                where=female_population.sum(axis=1) > 0,
            )
            ax.plot(self.time, values)
        ax.set_xlabel("Time")
        ax.set_ylabel("Incidence")
        ax.set_title("HPV Incidence")
        if log:
            ax.set_yscale("log")
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=200)
        return fig

    def plot_mortality(
        self,
        *,
        by_group: bool = False,
        log: bool = False,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        model = self.get_model()
        female_population = model.total_female_population(self.state)
        fig, ax = plt.subplots(figsize=(8, 4))
        if by_group:
            for name, matrix in model.group_mortality_matrix(self.state).items():
                values = np.divide(
                    matrix.sum(axis=1),
                    female_population.sum(axis=1),
                    out=np.zeros(self.time.shape[0], dtype=float),
                    where=female_population.sum(axis=1) > 0,
                )
                ax.plot(self.time, values, label=name)
            ax.legend()
        else:
            values = np.divide(
                model.mortality_matrix(self.state).sum(axis=1),
                female_population.sum(axis=1),
                out=np.zeros(self.time.shape[0], dtype=float),
                where=female_population.sum(axis=1) > 0,
            )
            ax.plot(self.time, values)
        ax.set_xlabel("Time")
        ax.set_ylabel("Mortality")
        ax.set_title("Cervical Cancer Mortality")
        if log:
            ax.set_yscale("log")
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=200)
        return fig

    def _write_to_group(self, group: h5py.Group) -> None:
        group.create_dataset("time", data=self.time)
        group.create_dataset("state", data=self.state)
        group.create_dataset("cumulative", data=self.cumulative)
        group.attrs["model_kind"] = self.model_kind
        write_json_attr(group, "state_spec", self.state_spec)
        write_json_attr(group, "cumulative_state_spec", self.cumulative_state_spec)
        write_json_attr(group, "config_snapshot", self.config_snapshot)

    @classmethod
    def _from_group(cls, group: h5py.Group) -> "SimulationResult":
        return cls(
            time=group["time"][:],
            state=group["state"][:],
            cumulative=group["cumulative"][:],
            model_kind=str(group.attrs["model_kind"]),
            state_spec=read_json_attr(group, "state_spec"),
            cumulative_state_spec=read_json_attr(group, "cumulative_state_spec"),
            config_snapshot=read_json_attr(group, "config_snapshot"),
        )

    def to_hdf(self, path: str | Path) -> None:
        target = ensure_parent(path)
        with h5py.File(target, "w") as handle:
            self._write_to_group(handle.create_group("simulation"))

    @classmethod
    def from_hdf(cls, path: str | Path) -> "SimulationResult":
        with h5py.File(path, "r") as handle:
            return cls._from_group(handle["simulation"])
