from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from optuna.trial import FixedTrial

from hpv_tdm import (
    AgeSexAggregateHPVModel,
    AgeSexSubtypeGroupedHPVModel,
    AggregateModelConfig,
    EvaluationConfig,
    Evaluator,
    SubtypeGroupedModelConfig,
)
from hpv_tdm.result import EvaluationResult, SimulationResult


def _small_aggregate_config(tmp_path) -> AggregateModelConfig:
    return AggregateModelConfig(
        population={"total_female": 50_000, "total_male": 50_000},
        simulation={
            "t_span": (0.0, 20.0),
            "n_eval": 11,
            "save_last_state": False,
            "generate_plots": False,
        },
        vaccination={"coverage_by_age": [0.0] * 26},
    )


def _small_subtype_config(tmp_path) -> SubtypeGroupedModelConfig:
    return SubtypeGroupedModelConfig(
        population={"total_female": 50_000, "total_male": 50_000},
        simulation={
            "t_span": (0.0, 20.0),
            "n_eval": 11,
            "save_last_state": False,
            "generate_plots": False,
        },
        vaccination={"coverage_by_age": [0.0] * 26},
    )


def test_aggregate_model_simulation_and_result_roundtrip(tmp_path) -> None:
    model = AgeSexAggregateHPVModel(_small_aggregate_config(tmp_path))
    result = model.simulate()

    assert isinstance(result, SimulationResult)
    assert result.state.shape[0] == model.config.simulation.n_eval

    path = tmp_path / "simulation_result.h5"
    result.to_hdf(path)
    loaded = SimulationResult.from_hdf(path)
    np.testing.assert_allclose(result.time, loaded.time)
    np.testing.assert_allclose(result.state, loaded.state)
    np.testing.assert_allclose(result.cumulative, loaded.cumulative)


def test_subtype_group_metrics_sum_to_total(tmp_path) -> None:
    config = _small_subtype_config(tmp_path).with_vaccination(
        product_id="nonavalent",
        coverage_by_age=[0.0] * 7 + [0.8] + [0.0] * 18,
    )
    model = AgeSexSubtypeGroupedHPVModel(config)
    result = model.simulate()
    total = model.incidence_matrix(result.state)
    grouped = model.group_incidence_matrix(result.state)
    reconstructed = sum(grouped.values())
    np.testing.assert_allclose(total, reconstructed)


def test_subtype_model_uses_split_group_parameters(tmp_path) -> None:
    base = _small_subtype_config(tmp_path).model_dump(mode="python")
    config = SubtypeGroupedModelConfig(
        **(
            base
            | {
                "subtype_groups": {
                    "hr_16_18": {
                        "label": "HPV 16/18",
                        "initial_weight": 0.5,
                        "persistence_multiplier": 1.4,
                        "cancer_progression_multiplier": 2.0,
                    },
                    "hr_31_33_45_52_58": {
                        "label": "HPV 31/33/45/52/58",
                        "initial_weight": 0.3,
                        "persistence_multiplier": 1.0,
                        "cancer_progression_multiplier": 0.8,
                    },
                    "hr_other": {
                        "label": "Other high-risk HPV",
                        "initial_weight": 0.2,
                        "persistence_multiplier": 0.7,
                        "cancer_progression_multiplier": 0.4,
                    },
                }
            }
        )
    )
    model = AgeSexSubtypeGroupedHPVModel(config)

    np.testing.assert_allclose(model.initial_group_weights, [0.5, 0.3, 0.2])
    np.testing.assert_allclose(model.persistence_multipliers, [1.4, 1.0, 0.7])
    np.testing.assert_allclose(model.cancer_progression_multipliers, [2.0, 0.8, 0.4])


def test_evaluator_outputs_absolute_and_incremental_results(tmp_path) -> None:
    config = _small_subtype_config(tmp_path)
    model = AgeSexSubtypeGroupedHPVModel(config)
    reference = model.simulate()

    vaccinated_config = config.with_vaccination(
        product_id="nonavalent",
        coverage_by_age=[0.0] * 7 + [0.9] + [0.0] * 18,
    )
    model.set_config(vaccinated_config)
    target = model.simulate()

    evaluator = Evaluator(EvaluationConfig(discount_rate=0.0))
    absolute = evaluator.evaluate(target)
    incremental = evaluator.evaluate(target, reference)

    assert isinstance(absolute, EvaluationResult)
    assert incremental.icur is not None
    assert incremental.incidence[-1] <= evaluator.evaluate(reference).incidence[-1]

    path = tmp_path / "evaluation_result.h5"
    incremental.to_hdf(path)
    loaded = EvaluationResult.from_hdf(path)
    np.testing.assert_allclose(incremental.total_cost, loaded.total_cost)
    np.testing.assert_allclose(incremental.total_daly, loaded.total_daly)


def test_nonavalent_outperforms_bivalent_in_subtype_model(tmp_path) -> None:
    config = _small_subtype_config(tmp_path)
    evaluator = Evaluator(EvaluationConfig(discount_rate=0.0))

    bivalent_model = AgeSexSubtypeGroupedHPVModel(
        config.with_vaccination(
            product_id="bivalent",
            coverage_by_age=[0.0] * 7 + [0.9] + [0.0] * 18,
        )
    )
    nonavalent_model = AgeSexSubtypeGroupedHPVModel(
        config.with_vaccination(
            product_id="nonavalent",
            coverage_by_age=[0.0] * 7 + [0.9] + [0.0] * 18,
        )
    )
    reference = AgeSexSubtypeGroupedHPVModel(config).simulate()

    bivalent_result = evaluator.evaluate(bivalent_model.simulate(), reference)
    nonavalent_result = evaluator.evaluate(nonavalent_model.simulate(), reference)

    assert nonavalent_result.incidence[-1] <= bivalent_result.incidence[-1]


def test_simulation_accepts_full_state_path_as_initial_state(tmp_path) -> None:
    config = _small_subtype_config(tmp_path)
    model = AgeSexSubtypeGroupedHPVModel(config)
    first_result = model.simulate()

    init_state_path = tmp_path / "stable_init.npy"
    np.save(init_state_path, first_result.state[-1].reshape(-1))

    config_with_init = SubtypeGroupedModelConfig(
        **(
            config.model_dump(mode="python")
            | {"simulation": config.simulation.model_dump(mode="python")}
        )
    )
    config_with_init = SubtypeGroupedModelConfig(
        **(
            config_with_init.model_dump(mode="python")
            | {
                "simulation": {
                    **config_with_init.simulation.model_dump(mode="python"),
                    "init_state_path": str(init_state_path),
                }
            }
        )
    )
    restarted_model = AgeSexSubtypeGroupedHPVModel(config_with_init)
    restarted = restarted_model.simulate()

    np.testing.assert_allclose(
        restarted.state[0].reshape(-1),
        first_result.state[-1].reshape(-1),
        rtol=1e-8,
        atol=1e-8,
    )


def test_find_initial_bracket_returns_none_when_target_unreachable() -> None:
    module_path = Path(__file__).resolve().parents[1] / "find_intital.py"
    spec = importlib.util.spec_from_file_location("find_intital", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load find_intital.py for testing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    coarse_results = [
        (0.0, 1.0e-8),
        (0.1, 2.0e-8),
        (0.2, 1.5e-8),
    ]
    assert module._find_bracket(coarse_results, 4.0e-5) is None


def test_find_params_ignores_existing_init_state_path_during_calibration(
    tmp_path,
) -> None:
    module_path = Path(__file__).resolve().parents[1] / "find_params.py"
    spec = importlib.util.spec_from_file_location("find_params", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load find_params.py for testing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    existing_init_path = tmp_path / "existing_init.npy"
    np.save(existing_init_path, np.zeros(10, dtype=float))

    base_config = SubtypeGroupedModelConfig(
        population={"total_female": 50_000, "total_male": 50_000},
        simulation={
            "t_span": (0.0, 20.0),
            "n_eval": 11,
            "save_last_state": False,
            "generate_plots": False,
            "init_state_path": str(existing_init_path),
        },
        vaccination={"coverage_by_age": [0.0] * 26},
    )
    params_config = module.FindParamsConfig()
    trial = FixedTrial(
        {
            "initial_weight_raw__hr_16_18": 0.147,
            "initial_weight_raw__hr_31_33_45_52_58": 0.468,
            "initial_weight_raw__hr_other": 0.385,
            "persistence_multiplier__hr_16_18": 1.0,
            "cancer_progression_multiplier__hr_16_18": 1.0,
            "persistence_multiplier__hr_31_33_45_52_58": 1.0,
            "cancer_progression_multiplier__hr_31_33_45_52_58": 1.0,
            "persistence_multiplier__hr_other": 1.0,
            "cancer_progression_multiplier__hr_other": 1.0,
            "initial_infectious_ratio": 0.01,
        }
    )

    candidate = module._candidate_config(base_config, params_config, trial)

    assert candidate.simulation.init_state_path is None


def test_find_params_trend_helper_reports_positive_slope() -> None:
    module_path = Path(__file__).resolve().parents[1] / "find_params.py"
    spec = importlib.util.spec_from_file_location("find_params", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load find_params.py for testing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    time = np.array([0.0, 5.0, 10.0, 15.0], dtype=float)
    incidence = np.array([1.0e-4, 1.2e-4, 1.4e-4, 1.6e-4], dtype=float)

    slope = module._incidence_trend_per_100k_per_year(
        time,
        incidence,
        window_years=10.0,
    )

    assert slope > 0.0


def test_find_params_sampler_selection_and_missing_cmaes_error() -> None:
    module_path = Path(__file__).resolve().parents[1] / "find_params.py"
    spec = importlib.util.spec_from_file_location("find_params", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load find_params.py for testing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    tpe_config = module.FindParamsConfig(optimizer="tpe")
    sampler = module._build_sampler(tpe_config)
    assert sampler.__class__.__name__ == "TPESampler"

    original_find_spec = module.importlib.util.find_spec
    module.importlib.util.find_spec = lambda name: (
        None if name == "cmaes" else original_find_spec(name)
    )
    try:
        with pytest.raises(ModuleNotFoundError, match="requires the 'cmaes' package"):
            module._build_sampler(module.FindParamsConfig(optimizer="cmaes"))
    finally:
        module.importlib.util.find_spec = original_find_spec


def test_find_params_rejects_invalid_trend_interval() -> None:
    module_path = Path(__file__).resolve().parents[1] / "find_params.py"
    spec = importlib.util.spec_from_file_location("find_params", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load find_params.py for testing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with pytest.raises(ValueError, match="must be smaller than or equal to"):
        module.FindParamsConfig(
            min_incidence_slope_per_100k_per_year=1.0,
            max_incidence_slope_per_100k_per_year=0.0,
        )


def test_simulate_time_horizon_override_updates_config() -> None:
    module_path = Path(__file__).resolve().parents[1] / "simulate.py"
    spec = importlib.util.spec_from_file_location("simulate_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load simulate.py for testing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config = AggregateModelConfig(
        simulation={"t_span": (0.0, 100.0)},
        vaccination={"coverage_by_age": [0.0] * 26},
    )

    overridden = module._override_time_horizon(config, 60.0)

    assert overridden.simulation.t_span == (0.0, 60.0)
