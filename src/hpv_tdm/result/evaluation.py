from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._io import ensure_parent, read_json_attr, write_json_attr


class EvaluationResult:
    def __init__(
        self,
        *,
        time: np.ndarray,
        incidence: np.ndarray,
        mortality: np.ndarray,
        incidence_by_group: dict[str, np.ndarray],
        mortality_by_group: dict[str, np.ndarray],
        cumulative_cecx: np.ndarray,
        cumulative_vaccinated: np.ndarray,
        cumulative_cecx_deaths: np.ndarray,
        cost_vacc: np.ndarray,
        cost_cecx: np.ndarray,
        daly_fatal: np.ndarray,
        daly_nofatal: np.ndarray,
        lifeloss: np.ndarray,
        config_snapshot: dict[str, Any],
        vaccine_product_id: str | None,
        icur: np.ndarray | None = None,
        avoid_cecx: np.ndarray | None = None,
        avoid_cecx_deaths: np.ndarray | None = None,
        avoid_daly: np.ndarray | None = None,
    ) -> None:
        self.time = np.asarray(time, dtype=float)
        self.incidence = np.asarray(incidence, dtype=float)
        self.mortality = np.asarray(mortality, dtype=float)
        self.incidence_by_group = {
            key: np.asarray(value, dtype=float)
            for key, value in incidence_by_group.items()
        }
        self.mortality_by_group = {
            key: np.asarray(value, dtype=float)
            for key, value in mortality_by_group.items()
        }
        self.cumulative_cecx = np.asarray(cumulative_cecx, dtype=float)
        self.cumulative_vaccinated = np.asarray(cumulative_vaccinated, dtype=float)
        self.cumulative_cecx_deaths = np.asarray(cumulative_cecx_deaths, dtype=float)
        self.cost_vacc = np.asarray(cost_vacc, dtype=float)
        self.cost_cecx = np.asarray(cost_cecx, dtype=float)
        self.daly_fatal = np.asarray(daly_fatal, dtype=float)
        self.daly_nofatal = np.asarray(daly_nofatal, dtype=float)
        self.lifeloss = np.asarray(lifeloss, dtype=float)
        self.config_snapshot = config_snapshot
        self.vaccine_product_id = vaccine_product_id
        self.icur = None if icur is None else np.asarray(icur, dtype=float)
        self.avoid_cecx = (
            None if avoid_cecx is None else np.asarray(avoid_cecx, dtype=float)
        )
        self.avoid_cecx_deaths = (
            None
            if avoid_cecx_deaths is None
            else np.asarray(avoid_cecx_deaths, dtype=float)
        )
        self.avoid_daly = (
            None if avoid_daly is None else np.asarray(avoid_daly, dtype=float)
        )

    @property
    def total_cost(self) -> np.ndarray:
        return self.cost_vacc + self.cost_cecx

    @property
    def total_daly(self) -> np.ndarray:
        return self.daly_fatal + self.daly_nofatal + self.lifeloss

    def summary_table(self) -> pd.DataFrame:
        payload = {
            "t_final": float(self.time[-1]),
            "vaccine_product_id": self.vaccine_product_id,
            "incidence_final": float(self.incidence[-1]),
            "mortality_final": float(self.mortality[-1]),
            "cumulative_cecx_final": float(self.cumulative_cecx.sum(axis=1)[-1]),
            "cumulative_vaccinated_final": float(
                self.cumulative_vaccinated.sum(axis=1)[-1]
            ),
            "cumulative_cecx_deaths_final": float(
                self.cumulative_cecx_deaths.sum(axis=1)[-1]
            ),
            "cost_vacc_final": float(self.cost_vacc[-1]),
            "cost_cecx_final": float(self.cost_cecx[-1]),
            "total_cost_final": float(self.total_cost[-1]),
            "daly_fatal_final": float(self.daly_fatal[-1]),
            "daly_nofatal_final": float(self.daly_nofatal[-1]),
            "lifeloss_final": float(self.lifeloss[-1]),
            "total_daly_final": float(self.total_daly[-1]),
        }
        if self.icur is not None:
            payload["icur_final"] = float(self.icur[-1])
        if self.avoid_cecx is not None:
            payload["avoid_cecx_final"] = float(self.avoid_cecx.sum(axis=1)[-1])
        if self.avoid_cecx_deaths is not None:
            payload["avoid_cecx_deaths_final"] = float(
                self.avoid_cecx_deaths.sum(axis=1)[-1]
            )
        if self.avoid_daly is not None:
            payload["avoid_daly_final"] = float(self.avoid_daly[-1])
        return pd.DataFrame([payload])

    def plot_incidence(
        self,
        *,
        by_group: bool = False,
        log: bool = False,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 4))
        if by_group:
            for name, values in self.incidence_by_group.items():
                ax.plot(self.time, values, label=name)
            ax.legend()
        else:
            ax.plot(self.time, self.incidence)
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
        fig, ax = plt.subplots(figsize=(8, 4))
        if by_group:
            for name, values in self.mortality_by_group.items():
                ax.plot(self.time, values, label=name)
            ax.legend()
        else:
            ax.plot(self.time, self.mortality)
        ax.set_xlabel("Time")
        ax.set_ylabel("Mortality")
        ax.set_title("Cervical Cancer Mortality")
        if log:
            ax.set_yscale("log")
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=200)
        return fig

    def plot_cost(self, *, save_path: str | Path | None = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(self.time, self.cost_vacc, label="vaccination")
        ax.plot(self.time, self.cost_cecx, label="cervical_cancer")
        ax.plot(self.time, self.total_cost, label="total")
        ax.set_xlabel("Time")
        ax.set_ylabel("Cost")
        ax.set_title("Economic Cost")
        ax.legend()
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=200)
        return fig

    def plot_daly(self, *, save_path: str | Path | None = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(self.time, self.daly_fatal, label="fatal")
        ax.plot(self.time, self.daly_nofatal, label="nonfatal")
        ax.plot(self.time, self.lifeloss, label="lifeloss")
        ax.plot(self.time, self.total_daly, label="total")
        ax.set_xlabel("Time")
        ax.set_ylabel("DALY")
        ax.set_title("DALY")
        ax.legend()
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=200)
        return fig

    def plot_icur(
        self,
        *,
        save_path: str | Path | None = None,
        t_span: tuple[float, float] | None = None,
    ) -> plt.Figure:
        if self.icur is None:
            raise ValueError(
                "ICUR is only available when a reference simulation is provided"
            )
        fig, ax = plt.subplots(figsize=(8, 4))
        if t_span is None:
            mask = np.ones(self.time.shape[0], dtype=bool)
        else:
            mask = (self.time >= t_span[0]) & (self.time <= t_span[1])
        ax.plot(self.time[mask], self.icur[mask])
        ax.set_xlabel("Time")
        ax.set_ylabel("ICUR")
        ax.set_title("ICUR")
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=200)
        return fig

    def _write_metric_dict(
        self, group: h5py.Group, payload: dict[str, np.ndarray]
    ) -> None:
        for key, value in payload.items():
            group.create_dataset(key, data=value)

    def _write_to_group(self, group: h5py.Group) -> None:
        group.create_dataset("time", data=self.time)
        group.create_dataset("incidence", data=self.incidence)
        group.create_dataset("mortality", data=self.mortality)
        group.create_dataset("cumulative_cecx", data=self.cumulative_cecx)
        group.create_dataset("cumulative_vaccinated", data=self.cumulative_vaccinated)
        group.create_dataset(
            "cumulative_cecx_deaths",
            data=self.cumulative_cecx_deaths,
        )
        group.create_dataset("cost_vacc", data=self.cost_vacc)
        group.create_dataset("cost_cecx", data=self.cost_cecx)
        group.create_dataset("daly_fatal", data=self.daly_fatal)
        group.create_dataset("daly_nofatal", data=self.daly_nofatal)
        group.create_dataset("lifeloss", data=self.lifeloss)
        if self.icur is not None:
            group.create_dataset("icur", data=self.icur)
        if self.avoid_cecx is not None:
            group.create_dataset("avoid_cecx", data=self.avoid_cecx)
        if self.avoid_cecx_deaths is not None:
            group.create_dataset("avoid_cecx_deaths", data=self.avoid_cecx_deaths)
        if self.avoid_daly is not None:
            group.create_dataset("avoid_daly", data=self.avoid_daly)
        incidence_group = group.create_group("incidence_by_group")
        self._write_metric_dict(incidence_group, self.incidence_by_group)
        mortality_group = group.create_group("mortality_by_group")
        self._write_metric_dict(mortality_group, self.mortality_by_group)
        write_json_attr(group, "config_snapshot", self.config_snapshot)
        group.attrs["vaccine_product_id"] = (
            "" if self.vaccine_product_id is None else self.vaccine_product_id
        )

    @classmethod
    def _from_group(cls, group: h5py.Group) -> "EvaluationResult":
        return cls(
            time=group["time"][:],
            incidence=group["incidence"][:],
            mortality=group["mortality"][:],
            incidence_by_group={
                key: dataset[:] for key, dataset in group["incidence_by_group"].items()
            },
            mortality_by_group={
                key: dataset[:] for key, dataset in group["mortality_by_group"].items()
            },
            cumulative_cecx=group["cumulative_cecx"][:],
            cumulative_vaccinated=group["cumulative_vaccinated"][:],
            cumulative_cecx_deaths=group["cumulative_cecx_deaths"][:],
            cost_vacc=group["cost_vacc"][:],
            cost_cecx=group["cost_cecx"][:],
            daly_fatal=group["daly_fatal"][:],
            daly_nofatal=group["daly_nofatal"][:],
            lifeloss=group["lifeloss"][:],
            config_snapshot=read_json_attr(group, "config_snapshot"),
            vaccine_product_id=group.attrs["vaccine_product_id"] or None,
            icur=group["icur"][:] if "icur" in group else None,
            avoid_cecx=group["avoid_cecx"][:] if "avoid_cecx" in group else None,
            avoid_cecx_deaths=(
                group["avoid_cecx_deaths"][:] if "avoid_cecx_deaths" in group else None
            ),
            avoid_daly=group["avoid_daly"][:] if "avoid_daly" in group else None,
        )

    def to_hdf(self, path: str | Path) -> None:
        target = ensure_parent(path)
        with h5py.File(target, "w") as handle:
            self._write_to_group(handle.create_group("evaluation"))

    @classmethod
    def from_hdf(cls, path: str | Path) -> "EvaluationResult":
        with h5py.File(path, "r") as handle:
            return cls._from_group(handle["evaluation"])
