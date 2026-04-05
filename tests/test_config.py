from __future__ import annotations

import json

import pytest

from hpv_tdm import (
    AggregateModelConfig,
    EvaluationConfig,
    SearchConfig,
    SubtypeGroupedModelConfig,
)


def test_core_configs_support_defaults_and_partial_override() -> None:
    aggregate = AggregateModelConfig(
        simulation={"n_eval": 11},
        vaccination={"coverage_by_age": [0.0] * 26},
    )
    subtype = SubtypeGroupedModelConfig(
        vaccination={"coverage_by_age": [0.0] * 26},
    )
    evaluation = EvaluationConfig(discount_rate=0.0)
    search = SearchConfig(n_trials=12, strategy="conti_one_cover")

    assert aggregate.simulation.n_eval == 11
    assert aggregate.simulation.t_span == (0.0, 100.0)
    assert len(aggregate.transmission.female_initial_state) == 8
    assert subtype.model_kind == "subtype_grouped"
    assert len(subtype.transmission.female_initial_state) == 8
    assert evaluation.life_table_method == "prime"
    assert search.n_trials == 12


def test_model_config_json_merge_keeps_defaults(tmp_path) -> None:
    payload = {
        "model_kind": "subtype_grouped",
        "simulation": {"n_eval": 9},
        "vaccination": {"coverage_by_age": [0.0] * 26},
    }
    config_path = tmp_path / "model.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    config = SubtypeGroupedModelConfig.from_json_file(config_path)
    assert config.model_kind == "subtype_grouped"
    assert config.simulation.n_eval == 9
    assert config.simulation.t_span == (0.0, 100.0)
    assert set(config.subtype_groups) == {
        "hr_16_18",
        "hr_31_33_45_52_58",
        "hr_other",
    }
    assert config.subtype_groups["hr_16_18"].initial_weight == pytest.approx(0.147)
    assert config.subtype_groups["hr_16_18"].infection_weight == pytest.approx(0.147)
    assert config.subtype_groups["hr_16_18"].persistence_multiplier == pytest.approx(
        2.62
    )
    assert config.subtype_groups[
        "hr_31_33_45_52_58"
    ].cancer_progression_multiplier == pytest.approx(0.49)


def test_subtype_vaccine_group_protection_defaults_to_structural_zero_one() -> None:
    config = SubtypeGroupedModelConfig(vaccination={"coverage_by_age": [0.0] * 26})
    assert config.vaccine_catalog.products["bivalent"].group_protection == {
        "hr_16_18": 1.0
    }
    assert config.vaccine_catalog.products["quadrivalent"].group_protection == {
        "hr_16_18": 1.0
    }
    assert config.vaccine_catalog.products["nonavalent"].group_protection == {
        "hr_16_18": 1.0,
        "hr_31_33_45_52_58": 1.0,
    }


def test_search_config_roundtrip(tmp_path) -> None:
    config = SearchConfig(strategy="contiOneCover", n_trials=3, n_jobs=2)
    path = tmp_path / "search.json"
    config.to_json_file(path)
    loaded = SearchConfig.from_json_file(path)
    assert loaded.strategy == "conti_one_cover"
    assert loaded.n_trials == 3
    assert loaded.n_jobs == 2


def test_subtype_female_initial_state_requires_explicit_eight_components() -> None:
    with pytest.raises(ValueError, match="female_initial_state length must be 8"):
        SubtypeGroupedModelConfig(
            transmission={"female_initial_state": [0.85, 0.15, 0, 0, 0, 0, 0]},
            vaccination={"coverage_by_age": [0.0] * 26},
        )
