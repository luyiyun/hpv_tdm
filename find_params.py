from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Literal

import numpy as np
import optuna
from pydantic import Field, model_validator

from hpv_tdm import AgeSexSubtypeGroupedHPVModel, SubtypeGroupedModelConfig
from hpv_tdm.config import ConfigBase


class ObjectiveWeightsConfig(ConfigBase):
    incidence: float = Field(
        default=1.0,
        ge=0.0,
        description="总发病率误差在目标函数中的权重。",
    )
    infection_share: float = Field(
        default=10.0,
        ge=0.0,
        description="感染亚型占比误差在目标函数中的权重。",
    )
    cancer_share: float = Field(
        default=10.0,
        ge=0.0,
        description="癌症亚型占比误差在目标函数中的权重。",
    )
    incidence_trend: float = Field(
        default=10.0,
        ge=0.0,
        description="末期宫颈癌发病率趋势约束在目标函数中的权重。",
    )


class SearchBoundsConfig(ConfigBase):
    initial_infectious_ratio: tuple[float, float] = Field(
        default=(1e-4, 0.5),
        description="初始感染比例搜索范围。",
    )
    initial_weight_raw: tuple[float, float] = Field(
        default=(0.05, 10.0),
        description="初始亚型权重原始值搜索范围，后续会归一化为和为 1。",
    )
    persistence_multiplier: tuple[float, float] = Field(
        default=(0.05, 10.0),
        description="持续感染倍率搜索范围。",
    )
    cancer_progression_multiplier: tuple[float, float] = Field(
        default=(0.05, 10.0),
        description="癌症进展倍率搜索范围。",
    )


class FindParamsConfig(ConfigBase):
    optimizer: Literal["tpe", "cmaes"] = Field(
        default="cmaes",
        description="参数校准使用的优化算法，可选 tpe 或 cmaes。",
    )
    study_name: str = Field(
        default="hpv_tdm_find_params",
        description="Optuna study 名称。",
    )
    storage_filename: str = Field(
        default="study.db",
        description="Optuna SQLite 存储文件名。",
    )
    seed: int = Field(
        default=1234,
        description="Optuna 搜索随机种子。",
    )
    n_trials: int = Field(
        default=200,
        ge=1,
        description="Optuna 搜索 trial 数。",
    )
    n_jobs: int = Field(
        default=1,
        ge=1,
        description="Optuna 并行 worker 数。",
    )
    time_horizon: float = Field(
        default=200.0,
        gt=0.0,
        description="每次参数搜索使用的长时程模拟年限。",
    )
    n_eval: int = Field(
        default=201,
        ge=2,
        description="每次参数搜索使用的时间采样点数。",
    )
    tail_years: float = Field(
        default=20.0,
        gt=0.0,
        description="用来计算稳态指标的尾部平均时间窗口，单位为年。",
    )
    trend_window_years: float = Field(
        default=10.0,
        gt=0.0,
        description="用来约束末期宫颈癌发病率趋势的时间窗口，单位为年。",
    )
    disable_vaccination_during_calibration: bool = Field(
        default=True,
        description="参数校准时是否关闭接种策略，以便先拟合自然史稳态。",
    )
    target_incidence_per_100k: float = Field(
        default=16.56,
        description="目标宫颈癌年发病率，单位为每 10 万女性。",
    )
    target_infection_share_by_group: dict[str, float] = Field(
        default_factory=lambda: {
            "hr_16_18": 0.147,
            "hr_31_33_45_52_58": 0.468,
            "hr_other": 0.385,
        },
        description="目标感染亚型占比，键为亚型组名，值为目标比例。",
    )
    target_cancer_share_by_group: dict[str, float] = Field(
        default_factory=lambda: {
            "hr_16_18": 0.680,
            "hr_31_33_45_52_58": 0.214,
            "hr_other": 0.105,
        },
        description="目标宫颈癌亚型占比，键为亚型组名，值为目标比例。",
    )
    min_incidence_slope_per_100k_per_year: float = Field(
        default=0.0,
        description=(
            "末期宫颈癌发病率趋势下限，单位为每 10 万女性每年。"
            "默认要求末期趋势至少持平。"
        ),
    )
    max_incidence_slope_per_100k_per_year: float = Field(
        default=10.0,
        description=(
            "末期宫颈癌发病率趋势上限，单位为每 10 万女性每年。"
            "用于避免校准得到不合理的快速上升趋势。"
        ),
    )
    objective_weights: ObjectiveWeightsConfig = Field(
        default_factory=ObjectiveWeightsConfig,
        description="不同拟合目标在总损失中的权重。",
    )
    bounds: SearchBoundsConfig = Field(
        default_factory=SearchBoundsConfig,
        description="搜索参数范围。",
    )

    @model_validator(mode="after")
    def validate_targets(self) -> "FindParamsConfig":
        if self.tail_years >= self.time_horizon:
            raise ValueError("tail_years must be smaller than time_horizon")
        if self.trend_window_years >= self.time_horizon:
            raise ValueError("trend_window_years must be smaller than time_horizon")
        if sum(self.target_infection_share_by_group.values()) <= 0:
            raise ValueError(
                "target_infection_share_by_group must sum to a positive value"
            )
        if sum(self.target_cancer_share_by_group.values()) <= 0:
            raise ValueError(
                "target_cancer_share_by_group must sum to a positive value"
            )
        if (
            self.min_incidence_slope_per_100k_per_year
            > self.max_incidence_slope_per_100k_per_year
        ):
            raise ValueError(
                "min_incidence_slope_per_100k_per_year must be "
                "smaller than or equal to "
                "max_incidence_slope_per_100k_per_year"
            )
        return self


ObjectiveWeightsConfig.model_rebuild()
SearchBoundsConfig.model_rebuild()
FindParamsConfig.model_rebuild()


def _normalize_mapping(payload: dict[str, float]) -> dict[str, float]:
    total = sum(payload.values())
    if total <= 0:
        raise ValueError("mapping values must sum to a positive value")
    return {key: float(value / total) for key, value in payload.items()}


def _load_model_config(path: str | Path) -> SubtypeGroupedModelConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    config = SubtypeGroupedModelConfig.from_json_dict(payload)
    if config.model_kind != "subtype_grouped":
        raise ValueError(
            "find_params.py currently only supports subtype_grouped models"
        )
    return config


def _build_sampler(params_config: FindParamsConfig) -> optuna.samplers.BaseSampler:
    if params_config.optimizer == "tpe":
        return optuna.samplers.TPESampler(seed=params_config.seed)
    if importlib.util.find_spec("cmaes") is None:
        raise ModuleNotFoundError(
            "optimizer='cmaes' requires the 'cmaes' package to be installed"
        )
    return optuna.samplers.CmaEsSampler(seed=params_config.seed)


def _candidate_config(
    base_config: SubtypeGroupedModelConfig,
    params_config: FindParamsConfig,
    trial: optuna.Trial,
) -> SubtypeGroupedModelConfig:
    payload = base_config.model_dump(mode="python")
    # 校准参数时必须从当前 trial 定义的初始状态出发，不能复用外部配置里已有的稳态文件。
    payload["simulation"]["init_state_path"] = None
    group_names = list(base_config.subtype_groups)

    raw_initial_weights = {
        group_name: trial.suggest_float(
            f"initial_weight_raw__{group_name}",
            low=params_config.bounds.initial_weight_raw[0],
            high=params_config.bounds.initial_weight_raw[1],
            log=True,
        )
        for group_name in group_names
    }
    normalized_initial_weights = _normalize_mapping(raw_initial_weights)
    for group_name in group_names:
        payload["subtype_groups"][group_name]["initial_weight"] = (
            normalized_initial_weights[group_name]
        )
        payload["subtype_groups"][group_name]["persistence_multiplier"] = (
            trial.suggest_float(
                f"persistence_multiplier__{group_name}",
                low=params_config.bounds.persistence_multiplier[0],
                high=params_config.bounds.persistence_multiplier[1],
                log=True,
            )
        )
        payload["subtype_groups"][group_name]["cancer_progression_multiplier"] = (
            trial.suggest_float(
                f"cancer_progression_multiplier__{group_name}",
                low=params_config.bounds.cancer_progression_multiplier[0],
                high=params_config.bounds.cancer_progression_multiplier[1],
                log=True,
            )
        )

    initial_infectious_ratio = trial.suggest_float(
        "initial_infectious_ratio",
        low=params_config.bounds.initial_infectious_ratio[0],
        high=params_config.bounds.initial_infectious_ratio[1],
        log=True,
    )
    payload["transmission"]["female_initial_state"] = [
        1.0 - initial_infectious_ratio,
        initial_infectious_ratio,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    payload["transmission"]["male_initial_state"] = [
        1.0 - initial_infectious_ratio,
        initial_infectious_ratio,
        0.0,
        0.0,
    ]

    if params_config.disable_vaccination_during_calibration:
        payload["vaccination"] = {
            "product_id": None,
            "coverage_by_age": [0.0] * base_config.nages,
        }
    return SubtypeGroupedModelConfig.model_validate(payload)


def _tail_mask(time: np.ndarray, tail_years: float) -> np.ndarray:
    tail_start = float(time[-1] - tail_years)
    mask = time >= tail_start
    if not np.any(mask):
        raise ValueError("tail mask is empty; check time_horizon and tail_years")
    return mask


def _female_infection_share_by_group(
    model: AgeSexSubtypeGroupedHPVModel,
    state: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    grouped_burden: dict[str, float] = {}
    for group_name in model.group_names:
        if_index = model._state_index[f"If__{group_name}"]
        pf_index = model._state_index[f"Pf__{group_name}"]
        burden = state[mask, if_index].sum(axis=1) + state[mask, pf_index].sum(axis=1)
        grouped_burden[group_name] = float(np.mean(burden))
    return _normalize_mapping(grouped_burden)


def _cancer_share_by_group(
    model: AgeSexSubtypeGroupedHPVModel,
    state: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    female_population = model.total_female_population(state).sum(axis=1)
    grouped_incidence = {}
    for group_name, matrix in model.group_incidence_matrix(state).items():
        values = np.divide(
            matrix.sum(axis=1),
            female_population,
            out=np.zeros(state.shape[0], dtype=float),
            where=female_population > 0,
        )
        grouped_incidence[group_name] = float(np.mean(values[mask]))
    return _normalize_mapping(grouped_incidence)


def _total_incidence(
    model: AgeSexSubtypeGroupedHPVModel,
    state: np.ndarray,
    mask: np.ndarray,
) -> float:
    values = _cervical_cancer_incidence_series(model, state)
    return float(np.mean(values[mask]))


def _cervical_cancer_incidence_series(
    model: AgeSexSubtypeGroupedHPVModel,
    state: np.ndarray,
) -> np.ndarray:
    female_population = model.total_female_population(state).sum(axis=1)
    return np.divide(
        model.incidence_matrix(state).sum(axis=1),
        female_population,
        out=np.zeros(state.shape[0], dtype=float),
        where=female_population > 0,
    )


def _incidence_trend_per_100k_per_year(
    time: np.ndarray,
    incidence_series: np.ndarray,
    *,
    window_years: float,
) -> float:
    trend_start = float(time[-1] - window_years)
    mask = time >= trend_start
    if np.count_nonzero(mask) < 2:
        raise ValueError("trend window must include at least two evaluation points")
    slope, _ = np.polyfit(time[mask], incidence_series[mask] * 100_000.0, deg=1)
    return float(slope)


def _mean_square_error(
    observed: dict[str, float],
    target: dict[str, float],
) -> float:
    keys = list(target)
    return float(np.mean([(observed[key] - target[key]) ** 2 for key in keys]))


def _summary_payload(
    *,
    params_config: FindParamsConfig,
    metrics: dict[str, object],
    best_params: dict[str, float | str | int],
    initial_state_path: Path,
    ready_model_config_path: Path,
    calibrated_model_config_path: Path,
) -> dict[str, object]:
    return {
        "target_incidence_per_100k": params_config.target_incidence_per_100k,
        "matched_incidence_per_100k": metrics["incidence_per_100k"],
        "target_infection_share_by_group": (
            params_config.target_infection_share_by_group
        ),
        "matched_infection_share_by_group": metrics["infection_share_by_group"],
        "target_cancer_share_by_group": params_config.target_cancer_share_by_group,
        "matched_cancer_share_by_group": metrics["cancer_share_by_group"],
        "trend_window_years": params_config.trend_window_years,
        "min_incidence_slope_per_100k_per_year": (
            params_config.min_incidence_slope_per_100k_per_year
        ),
        "max_incidence_slope_per_100k_per_year": (
            params_config.max_incidence_slope_per_100k_per_year
        ),
        "matched_incidence_slope_per_100k_per_year": metrics["incidence_trend"],
        "objective": metrics["objective"],
        "initial_infectious_ratio": metrics["initial_infectious_ratio"],
        "best_params": best_params,
        "initial_state_path": str(initial_state_path.resolve()),
        "ready_model_config_path": str(ready_model_config_path.resolve()),
        "calibrated_model_config_path": str(calibrated_model_config_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate subtype-group parameters by matching long-run incidence, "
            "infection-type shares, and cancer-type shares."
        )
    )
    parser.add_argument(
        "--model-config",
        required=True,
        help="Path to subtype-grouped model config JSON.",
    )
    parser.add_argument(
        "--params-config",
        required=True,
        help="Path to parameter calibration config JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/find_params",
        help="Directory used to save calibration outputs.",
    )
    args = parser.parse_args()

    base_config = _load_model_config(args.model_config)
    params_config = FindParamsConfig.from_json_file(args.params_config)
    group_names = list(base_config.subtype_groups)
    if set(params_config.target_infection_share_by_group) != set(group_names):
        raise ValueError(
            "target_infection_share_by_group keys must match subtype group names"
        )
    if set(params_config.target_cancer_share_by_group) != set(group_names):
        raise ValueError(
            "target_cancer_share_by_group keys must match subtype group names"
        )

    params_config = FindParamsConfig.model_validate(
        {
            **params_config.model_dump(mode="python"),
            "target_infection_share_by_group": _normalize_mapping(
                dict(params_config.target_infection_share_by_group)
            ),
            "target_cancer_share_by_group": _normalize_mapping(
                dict(params_config.target_cancer_share_by_group)
            ),
        }
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    params_config.to_json_file(output_dir / "find_params_config.json")

    sampler = _build_sampler(params_config)
    storage_path = output_dir / params_config.storage_filename
    study = optuna.create_study(
        study_name=params_config.study_name,
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
        sampler=sampler,
        direction="minimize",
    )

    def objective(trial: optuna.Trial) -> float:
        candidate_config = _candidate_config(base_config, params_config, trial)
        model = AgeSexSubtypeGroupedHPVModel(candidate_config)
        result = model.simulate(
            t_span=(0.0, params_config.time_horizon),
            n_eval=params_config.n_eval,
            verbose=False,
        )
        mask = _tail_mask(result.time, params_config.tail_years)
        infection_share = _female_infection_share_by_group(model, result.state, mask)
        cancer_share = _cancer_share_by_group(model, result.state, mask)
        incidence = _total_incidence(model, result.state, mask)
        incidence_per_100k = incidence * 100_000.0
        incidence_trend = _incidence_trend_per_100k_per_year(
            result.time,
            _cervical_cancer_incidence_series(model, result.state),
            window_years=params_config.trend_window_years,
        )

        incidence_error = (
            (incidence_per_100k - params_config.target_incidence_per_100k)
            / params_config.target_incidence_per_100k
        ) ** 2
        infection_error = _mean_square_error(
            infection_share,
            params_config.target_infection_share_by_group,
        )
        cancer_error = _mean_square_error(
            cancer_share,
            params_config.target_cancer_share_by_group,
        )
        trend_error = 0.0
        if incidence_trend < params_config.min_incidence_slope_per_100k_per_year:
            # 若末期趋势持续下降，则给出强惩罚，
            # 避免把会自动消退到极低水平的参数选为最优。
            trend_error = (
                1.0
                + (
                    params_config.min_incidence_slope_per_100k_per_year
                    - incidence_trend
                )
                ** 2
            )
        elif incidence_trend > params_config.max_incidence_slope_per_100k_per_year:
            # 若末期趋势上升过快，同样给出强惩罚，
            # 保证校准后的自然史更接近持平或缓慢变化。
            trend_error = (
                1.0
                + (
                    incidence_trend
                    - params_config.max_incidence_slope_per_100k_per_year
                )
                ** 2
            )
        objective_value = (
            params_config.objective_weights.incidence * incidence_error
            + params_config.objective_weights.infection_share * infection_error
            + params_config.objective_weights.cancer_share * cancer_error
            + params_config.objective_weights.incidence_trend * trend_error
        )
        trial.set_user_attr("incidence_per_100k", incidence_per_100k)
        trial.set_user_attr("infection_share_by_group", infection_share)
        trial.set_user_attr("cancer_share_by_group", cancer_share)
        trial.set_user_attr(
            "incidence_slope_per_100k_per_year",
            incidence_trend,
        )
        return float(objective_value)

    study.optimize(
        objective,
        n_trials=params_config.n_trials,
        n_jobs=params_config.n_jobs,
    )

    best_trial = study.best_trial
    best_config = _candidate_config(base_config, params_config, best_trial)
    best_model = AgeSexSubtypeGroupedHPVModel(best_config)
    best_result = best_model.simulate(
        t_span=(0.0, params_config.time_horizon),
        n_eval=params_config.n_eval,
        verbose=False,
    )
    mask = _tail_mask(best_result.time, params_config.tail_years)
    infection_share = _female_infection_share_by_group(
        best_model, best_result.state, mask
    )
    cancer_share = _cancer_share_by_group(best_model, best_result.state, mask)
    incidence = _total_incidence(best_model, best_result.state, mask)
    incidence_per_100k = incidence * 100_000.0
    incidence_trend = _incidence_trend_per_100k_per_year(
        best_result.time,
        _cervical_cancer_incidence_series(best_model, best_result.state),
        window_years=params_config.trend_window_years,
    )

    initial_state_path = output_dir / "initial_state.npy"
    np.save(initial_state_path, best_result.state[-1].reshape(-1))

    calibrated_model_config_path = output_dir / "calibrated_model_config.json"
    best_config.to_json_file(calibrated_model_config_path)

    ready_payload = best_config.model_dump(mode="python")
    ready_payload["simulation"]["init_state_path"] = str(initial_state_path.resolve())
    ready_model_config = SubtypeGroupedModelConfig.model_validate(ready_payload)
    ready_model_config_path = output_dir / "model_config_with_init_state.json"
    ready_model_config.to_json_file(ready_model_config_path)

    best_result.to_hdf(output_dir / "calibration_simulation_result.h5")
    with (output_dir / "best_trial.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "number": best_trial.number,
                "value": best_trial.value,
                "params": best_trial.params,
                "user_attrs": best_trial.user_attrs,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")

    metrics = {
        "objective": float(best_trial.value),
        "incidence_per_100k": float(incidence_per_100k),
        "infection_share_by_group": infection_share,
        "cancer_share_by_group": cancer_share,
        "incidence_trend": float(incidence_trend),
        "initial_infectious_ratio": float(
            best_trial.params["initial_infectious_ratio"]
        ),
    }
    summary = _summary_payload(
        params_config=params_config,
        metrics=metrics,
        best_params=best_trial.params,
        initial_state_path=initial_state_path,
        ready_model_config_path=ready_model_config_path,
        calibrated_model_config_path=calibrated_model_config_path,
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("Parameter calibration summary")
    print(f"  objective: {best_trial.value:.8g}")
    print(
        "  incidence_per_100k: "
        f"{incidence_per_100k:.4f} "
        f"(target {params_config.target_incidence_per_100k:.4f})"
    )
    print(f"  infection_share_by_group: {infection_share}")
    print(f"  cancer_share_by_group: {cancer_share}")
    print(
        "  incidence_slope_per_100k_per_year: "
        f"{incidence_trend:.4f} "
        "("
        f"target range "
        f"{params_config.min_incidence_slope_per_100k_per_year:.4f}"
        " to "
        f"{params_config.max_incidence_slope_per_100k_per_year:.4f}"
        ")"
    )
    print(f"  calibrated_model_config: {calibrated_model_config_path.resolve()}")
    print(f"  ready_model_config: {ready_model_config_path.resolve()}")


if __name__ == "__main__":
    main()
