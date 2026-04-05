from __future__ import annotations

import numpy as np

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
            "output_dir": str(tmp_path / "aggregate"),
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
            "output_dir": str(tmp_path / "subtype"),
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
                        "infection_weight": 0.2,
                        "persistence_multiplier": 1.4,
                        "cancer_progression_multiplier": 2.0,
                    },
                    "hr_31_33_45_52_58": {
                        "label": "HPV 31/33/45/52/58",
                        "initial_weight": 0.3,
                        "infection_weight": 0.5,
                        "persistence_multiplier": 1.0,
                        "cancer_progression_multiplier": 0.8,
                    },
                    "hr_other": {
                        "label": "Other high-risk HPV",
                        "initial_weight": 0.2,
                        "infection_weight": 0.3,
                        "persistence_multiplier": 0.7,
                        "cancer_progression_multiplier": 0.4,
                    },
                }
            }
        )
    )
    model = AgeSexSubtypeGroupedHPVModel(config)

    np.testing.assert_allclose(model.initial_group_weights, [0.5, 0.3, 0.2])
    np.testing.assert_allclose(model.infection_group_weights, [0.2, 0.5, 0.3])
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
