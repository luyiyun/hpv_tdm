from __future__ import annotations

import pytest
from optuna.trial import TrialState

from hpv_tdm import (
    AgeSexAggregateHPVModel,
    AgeSexSubtypeGroupedHPVModel,
    AggregateModelConfig,
    EvaluationConfig,
    Evaluator,
    SearchConfig,
    Searcher,
    SubtypeGroupedModelConfig,
)
from hpv_tdm.result import SearchResult


@pytest.mark.parametrize("strategy", ["one", "multi", "conti", "conti_one_cover"])
@pytest.mark.parametrize(
    "objective_mode",
    ["multi_objective", "weighted_sum", "constrained"],
)
def test_search_smoke_and_save_roundtrip(
    tmp_path,
    strategy: str,
    objective_mode: str,
) -> None:
    model_config = AggregateModelConfig(
        population={"total_female": 20_000, "total_male": 20_000},
        simulation={
            "t_span": (0.0, 10.0),
            "n_eval": 6,
            "save_last_state": False,
            "generate_plots": False,
        },
        vaccination={"coverage_by_age": [0.0] * 26},
    )
    search_config = SearchConfig(
        n_trials=1,
        incidence_threshold=1.0,
        strategy=strategy,
        objective_mode=objective_mode,
        product_ids=["bivalent"],
    )

    model = AgeSexAggregateHPVModel(model_config)
    evaluator = Evaluator(EvaluationConfig(discount_rate=0.0))
    result = Searcher(search_config).search(
        model,
        evaluator,
        output_dir=tmp_path / f"{objective_mode}-{strategy}",
    )

    assert isinstance(result, SearchResult)
    assert result.study.directions is not None
    result.save(tmp_path / f"{objective_mode}-{strategy}")
    reloaded = SearchResult.from_dir(tmp_path / f"{objective_mode}-{strategy}")

    assert (tmp_path / f"{objective_mode}-{strategy}" / "study.db").exists()
    assert (tmp_path / f"{objective_mode}-{strategy}" / "search_result.h5").exists()
    assert (
        tmp_path / f"{objective_mode}-{strategy}" / "best_model_config.json"
    ).exists()
    assert reloaded.study.study_name == search_config.study_name


def test_constrained_mode_does_not_use_scalarized_objective(tmp_path) -> None:
    model_config = AggregateModelConfig(
        population={"total_female": 20_000, "total_male": 20_000},
        simulation={
            "t_span": (0.0, 10.0),
            "n_eval": 6,
            "save_last_state": False,
            "generate_plots": False,
        },
        vaccination={"coverage_by_age": [0.0] * 26},
    )
    search_config = SearchConfig(
        n_trials=1,
        incidence_threshold=1.0,
        strategy="one",
        objective_mode="constrained",
        product_ids=["bivalent"],
    )

    model = AgeSexAggregateHPVModel(model_config)
    evaluator = Evaluator(EvaluationConfig(discount_rate=0.0))
    searcher = Searcher(search_config)

    def _raise_if_called(*args, **kwargs):
        raise AssertionError("constrained mode should optimize raw ICUR directly")

    searcher._scalarized_objective = _raise_if_called  # type: ignore[method-assign]
    result = searcher.search(
        model,
        evaluator,
        output_dir=tmp_path / "constrained-no-scalarized",
    )

    assert isinstance(result, SearchResult)


def test_parallel_subtype_search_uses_independent_models(tmp_path) -> None:
    model_config = SubtypeGroupedModelConfig(
        population={"total_female": 20_000, "total_male": 20_000},
        simulation={
            "t_span": (0.0, 6.0),
            "n_eval": 5,
            "save_last_state": False,
            "generate_plots": False,
            "init_state_path": None,
        },
        vaccination={"coverage_by_age": [0.0] * 26},
    )
    search_config = SearchConfig(
        n_trials=2,
        n_jobs=2,
        incidence_threshold=1.0,
        strategy="conti_one_cover",
        objective_mode="multi_objective",
        product_ids=["bivalent", "nonavalent"],
    )

    model = AgeSexSubtypeGroupedHPVModel(model_config)
    evaluator = Evaluator(EvaluationConfig(discount_rate=0.0))
    result = Searcher(search_config).search(
        model,
        evaluator,
        output_dir=tmp_path / "parallel-subtype-search",
    )

    assert isinstance(result, SearchResult)
    completed_trials = [
        trial
        for trial in result.study.trials
        if trial.state == TrialState.COMPLETE
    ]
    assert len(completed_trials) == 2


def test_search_does_not_mutate_input_model_config(tmp_path) -> None:
    model_config = AggregateModelConfig(
        population={"total_female": 20_000, "total_male": 20_000},
        simulation={
            "t_span": (0.0, 10.0),
            "n_eval": 6,
            "save_last_state": False,
            "generate_plots": False,
        },
        vaccination={"coverage_by_age": [0.0] * 26},
    )
    search_config = SearchConfig(
        n_trials=1,
        incidence_threshold=1.0,
        strategy="one",
        objective_mode="multi_objective",
        product_ids=["bivalent"],
    )

    model = AgeSexAggregateHPVModel(model_config)
    before = model.to_config().model_dump(mode="json")
    evaluator = Evaluator(EvaluationConfig(discount_rate=0.0))
    result = Searcher(search_config).search(
        model,
        evaluator,
        output_dir=tmp_path / "model-not-mutated",
    )
    after = model.to_config().model_dump(mode="json")

    assert isinstance(result, SearchResult)
    assert after == before
